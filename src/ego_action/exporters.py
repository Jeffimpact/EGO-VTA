"""Serialize only complete, contiguous pseudo-action windows."""

from __future__ import annotations

import json

import h5py
import numpy as np
from mcap_ros2.writer import Writer

from .common import NS, SIZE

ACTION = ["dx_m", "dy_m", "dz_m", "rx_rad", "ry_rad", "rz_rad", "next_aperture"]
STATE = ["x_m", "y_m", "z_m", "qw", "qx", "qy", "qz", "aperture"]
HEADER = (
    "\n" + "=" * 80 + "\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n"
)
TIME = "\n" + "=" * 80 + "\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n"
POSE = (
    "std_msgs/Header header\ngeometry_msgs/Pose pose\n"
    + HEADER
    + TIME
    + "\n"
    + "=" * 80
    + "\nMSG: geometry_msgs/Pose\ngeometry_msgs/Point position\ngeometry_msgs/Quaternion orientation\n"
    + "\n"
    + "=" * 80
    + "\nMSG: geometry_msgs/Point\nfloat64 x\nfloat64 y\nfloat64 z\n"
    + "\n"
    + "=" * 80
    + "\nMSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w\n"
)
IMAGE = (
    "std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\n"
    "uint32 step\nuint8[] data\n" + HEADER + TIME
)
ARRAY = (
    "std_msgs/MultiArrayLayout layout\nfloat32[] data\n"
    + "\n"
    + "=" * 80
    + "\nMSG: std_msgs/MultiArrayLayout\nstd_msgs/MultiArrayDimension[] dim\nuint32 data_offset\n"
    + "\n"
    + "=" * 80
    + "\nMSG: std_msgs/MultiArrayDimension\nstring label\nuint32 size\nuint32 stride\n"
)


def put(group, name, array):
    group.create_dataset(name, data=array, compression="gzip", compression_opts=2, shuffle=True)


def write_hdf5(path, spec, stage, clips, labels, tokenizer, signals, frame_tasks, metadata):
    goal_tokens = np.array(tokenizer.encode(spec["task"]).ids, dtype=np.int64)
    with h5py.File(path, "w") as out:
        out.attrs["metadata_json"] = json.dumps(metadata)
        for part, (idx, chunk) in enumerate(clips):
            ep = out.create_group(f"data/episode_{part:04d}")
            ep.attrs["source_episode_id"] = spec["episode_id"]
            ep.attrs["action_convention"] = (
                "body-frame translation (m), rotation vector (rad), next aperture"
            )
            put(ep, "source/grid_index", idx)
            put(ep, "source/timestamp_ns", signals["timestamps"][idx])
            put(ep, "source/grid_timestamp_ns", signals["grid"][idx])
            horizon = chunk.shape[1]
            sequence = np.arange(idx[0], idx[-1] + horizon + 1)
            put(ep, "sequence/grid_index", sequence)
            put(ep, "sequence/timestamp_ns", signals["timestamps"][sequence])
            put(
                ep,
                "target/future_observation_index",
                np.arange(len(idx))[:, None] + np.arange(1, horizon + 1)[None, :],
            )
            for name in spec["cameras"]:
                put(ep, f"source/{name}_frame", signals["camera_idx"][name][idx])
                put(ep, f"source/{name}_timestamp_ns", signals["camera_times"][name][idx])
                put(ep, f"source/{name}_valid", signals["camera_idx"][name][idx] >= 0)
                images = ep.create_dataset(
                    f"sequence/image_{name}",
                    shape=(len(sequence), 3, SIZE, SIZE),
                    dtype="u1",
                    chunks=(1, 3, SIZE, SIZE),
                    compression="gzip",
                    compression_opts=2,
                )
                for j, grid_i in enumerate(sequence):
                    images[j] = stage[name][grid_i].transpose(2, 0, 1)
                put(ep, f"sequence/{name}_valid", signals["camera_idx"][name][sequence] >= 0)
                # Same-file virtual view exposes VLA anchors without duplicating compressed pixels.
                view = h5py.VirtualLayout(shape=(len(idx), 3, SIZE, SIZE), dtype="u1")
                view[:] = h5py.VirtualSource(".", images.name, shape=images.shape)[: len(idx)]
                ep.create_virtual_dataset(f"observation/image_{name}", view)
            put(ep, "observation/wrist_pose", signals["pose"][idx].astype("f4"))
            put(ep, "observation/grasp_aperture", signals["aperture"][idx, None].astype("f4"))
            for name, data in signals["sensors"].items():
                key = {"imu": "imu_telemetry", "fmg": "fmg_vector"}.get(name, name)
                put(ep, f"observation/{key}", data[idx].astype("f4"))
                put(ep, f"source/{name}_valid", np.isfinite(data[idx]).all(axis=1))
                put(ep, f"source/{name}_age_ns", signals["ages"][name][idx])
            put(ep, "action/delta_ee_pose", chunk[:, 0, :6])
            put(ep, "action/grasp_aperture", chunk[:, 0, 6:])
            put(ep, "action/chunk", chunk)
            put(ep, "annotation/task_tokens", goal_tokens)
            put(ep, "annotation/segment_index", frame_tasks[idx])
            ep.attrs["segments_json"] = json.dumps([labels[i] for i in np.unique(frame_tasks[idx])])


def write_mcap(path, h5path, spec, labels):
    with h5py.File(h5path, "r") as h5, open(path, "wb") as file:
        writer = Writer(file)
        pose_schema = writer.register_msgdef("geometry_msgs/msg/PoseStamped", POSE)
        img_schema = writer.register_msgdef("sensor_msgs/msg/Image", IMAGE)
        array_schema = writer.register_msgdef("std_msgs/msg/Float32MultiArray", ARRAY)
        text_schema = writer.register_msgdef("std_msgs/msg/String", "string data\n")
        writer.add_metadata(
            "episode",
            {
                "source_episode_id": spec["episode_id"],
                "reference": spec["reference"],
                "action_fields": ",".join(ACTION),
            },
        )
        for ep in h5["data"].values():
            for i, stamp in enumerate(ep["source/timestamp_ns"]):
                stamp = int(stamp)

                def send(topic, schema, msg, stamp=stamp, sequence=i):
                    writer.write_message(
                        "/ego/" + topic,
                        schema,
                        msg,
                        log_time=stamp,
                        publish_time=stamp,
                        sequence=sequence,
                    )

                def header(frame, time):
                    return {"stamp": {"sec": time // NS, "nanosec": time % NS}, "frame_id": frame}

                for name in spec["cameras"]:
                    if not ep[f"source/{name}_valid"][i]:
                        continue
                    img = ep[f"observation/image_{name}"][i].transpose(1, 2, 0)
                    send(
                        name + "/image",
                        img_schema,
                        {
                            "header": header(
                                name + "_optical", int(ep[f"source/{name}_timestamp_ns"][i])
                            ),
                            "height": SIZE,
                            "width": SIZE,
                            "encoding": "rgb8",
                            "is_bigendian": 0,
                            "step": SIZE * 3,
                            "data": img.tobytes(),
                        },
                    )
                x, y, z, qw, qx, qy, qz = map(float, ep["observation/wrist_pose"][i])
                send(
                    "wrist",
                    pose_schema,
                    {
                        "header": header(spec["reference"], stamp),
                        "pose": {
                            "position": {"x": x, "y": y, "z": z},
                            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
                        },
                    },
                )
                values = {"action": ep["action/chunk"][i, 0]}
                chunk = ep["action/chunk"][i]
                dims = [
                    {"label": "future_step", "size": len(chunk), "stride": chunk.size},
                    {"label": "component", "size": 7, "stride": 7},
                ]
                send(
                    "action_chunk",
                    array_schema,
                    {"layout": {"dim": dims, "data_offset": 0}, "data": chunk.ravel().tolist()},
                )
                for key in ("fmg_vector", "imu_telemetry", "imu_orientation", "gaze", "rig"):
                    if key in ep["observation"]:
                        values[key] = ep["observation"][key][i]
                for key, val in values.items():
                    if np.isfinite(val).all():
                        send(
                            key,
                            array_schema,
                            {"layout": {"dim": [], "data_offset": 0}, "data": val.tolist()},
                        )
                send(
                    "instruction",
                    text_schema,
                    {"data": labels[int(ep["annotation/segment_index"][i])]["instruction"]},
                )
        writer.finish()


def write_lerobot(root, h5path, spec, labels):
    try:
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets import LeRobotDataset
    except ImportError as exc:
        raise ValueError("LeRobot export requires Python 3.12+ and the 'lerobot' extra") from exc
    features = {
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION},
        "action_chunk": {"dtype": "float32", "shape": (spec.get("horizon", 1), 7), "names": None},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE},
        "source_grid_index": {"dtype": "int64", "shape": (1,), "names": ["source_grid_index"]},
        "capture_timestamp_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["capture_timestamp_ns"],
        },
    }
    for cam in spec["cameras"]:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (SIZE, SIZE, 3),
            "names": ["height", "width", "channels"],
        }
        features[f"camera_valid_{cam}"] = {"dtype": "int64", "shape": (1,), "names": ["valid"]}
    with h5py.File(h5path, "r") as h5:
        first = next(iter(h5["data"].values()))["observation"]
        sensor_keys = [
            k
            for k in ("fmg_vector", "imu_telemetry", "imu_orientation", "gaze", "rig")
            if k in first
        ]
        for key in sensor_keys:
            features["observation." + key] = {
                "dtype": "float32",
                "shape": (first[key].shape[1],),
                "names": None,
            }
            features[key + "_valid"] = {"dtype": "int64", "shape": (1,), "names": ["valid"]}
        ds = LeRobotDataset.create(
            repo_id="local/ego_action",
            root=root,
            fps=spec.get("fps", 20),
            features=features,
            robot_type="human_hand_pseudo_action",
            use_videos=True,
            rgb_encoder=RGBEncoderConfig(vcodec="h264"),
            encoder_threads=1,
        )
        try:
            for ep in h5["data"].values():
                for i, grid_i in enumerate(ep["source/grid_index"]):
                    frame = {
                        "action": ep["action/chunk"][i, 0],
                        "action_chunk": ep["action/chunk"][i],
                        "observation.state": np.r_[
                            ep["observation/wrist_pose"][i], ep["observation/grasp_aperture"][i]
                        ].astype("f4"),
                        "source_grid_index": np.array([grid_i], dtype="int64"),
                        "capture_timestamp_ns": np.array(
                            [ep["source/timestamp_ns"][i]], dtype="int64"
                        ),
                        "task": labels[int(ep["annotation/segment_index"][i])]["instruction"],
                    }
                    for cam in spec["cameras"]:
                        frame[f"observation.images.{cam}"] = ep[f"observation/image_{cam}"][
                            i
                        ].transpose(1, 2, 0)
                        frame[f"camera_valid_{cam}"] = np.array(
                            [ep[f"source/{cam}_valid"][i]], dtype="int64"
                        )
                    for key in sensor_keys:
                        vals = ep["observation"][key][i]
                        valid = np.isfinite(vals).all()
                        frame["observation." + key] = vals if valid else np.zeros_like(vals)
                        frame[key + "_valid"] = np.array([valid], dtype="int64")
                    ds.add_frame(frame)
                ds.save_episode(parallel_encoding=False)
        finally:
            ds.finalize()
