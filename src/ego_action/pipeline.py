"""Run one episode transactionally. Outputs contain a complete receipt or do not exist."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from pathlib import Path

import av
import cv2
import h5py
import numpy as np

from . import __version__
from .common import (
    FMG,
    GAZE,
    IMU,
    NS,
    POINTS,
    QUAT,
    SIZE,
    XYZ,
    media_paths,
    read_csv,
    read_json,
    require_range,
    runs,
    safe_name,
    sha256,
    write_json,
)
from .exporters import write_hdf5, write_lerobot, write_mcap
from .ingest import align, frames, nearest_indices, stage_camera
from .labels import make_tokenizer, segment
from .pose import (
    HandTracker,
    action_chunks,
    smooth_poses,
    transform_to_reference,
    wrist_pose,
)
from .quality import assess, derivatives

SENSOR_CHANNELS = {"imu": IMU, "fmg": FMG, "gaze": GAZE, "rig": ("pitch_deg", "lux")}
FORMATS = {"hdf5", "mcap", "lerobot"}


def validate(spec):
    safe_name(spec["episode_id"])
    if not isinstance(spec.get("task"), str) or not spec["task"].strip():
        raise ValueError("Supply a nonempty task instruction")
    if spec.get("reference") not in {"static_camera", "tracked_camera", "moving_untracked"}:
        raise ValueError("reference must be static_camera, tracked_camera, or moving_untracked")
    if spec["reference"] == "tracked_camera" and not spec.get("camera_pose"):
        raise ValueError("tracked_camera requires a timestamped T_reference_camera CSV")
    if (
        spec.get("fps", 20) not in {10, 20}
        or not isinstance(spec.get("horizon", 1), int)
        or not 1 <= spec.get("horizon", 1) <= 256
    ):
        raise ValueError("fps must be 10 or 20 and horizon an integer from 1 to 256")
    limits = np.asarray(spec["aperture_range_m"], float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or not 0 <= limits[0] < limits[1]:
        raise ValueError(
            "aperture_range_m must give increasing closed/open fingertip distances in metres"
        )
    if (
        "palm_width_m" in spec
        and (not np.isfinite(spec["palm_width_m"]) or spec["palm_width_m"] <= 0)
        or spec.get("hand", "Right") not in {"Right", "Left"}
    ):
        raise ValueError("Supply a positive palm measurement and hand Left or Right")
    if "primary" not in spec["cameras"]:
        raise ValueError("The cameras map must contain primary")
    for name, camera in spec["cameras"].items():
        safe_name(name)
        if (
            not isinstance(camera.get("start_time_ns"), int)
            or not 0 < camera.get("tolerance_ms", 25) <= 100
        ):
            raise ValueError(
                "Each camera requires an integer shared-clock start_time_ns and a positive tolerance"
            )
    if spec.get("sync_mode", "backward") not in {"backward", "interpolate"}:
        raise ValueError("sync_mode must be backward or interpolate")
    if set(spec.get("sensors", {})) - SENSOR_CHANNELS.keys():
        raise ValueError("Unknown sensor stream")
    if spec.get("landmarks") and spec.get("hand_model"):
        raise ValueError("Choose one landmark source per run")


def sensor_signals(spec, paths, targets):
    inputs_ok = np.ones(len(targets), dtype=bool)
    signals, ages = {}, {}
    for name, cfg in spec.get("sensors", {}).items():
        quantities = SENSOR_CHANNELS[name]
        path = paths["sensor/" + name]
        tim, arr = read_csv(path, quantities)
        if name in {"fmg", "gaze"}:
            require_range(arr[np.isfinite(arr)], 0, 1, name)
        tim = tim + int(cfg.get("offset_ns", 0))
        mode = spec.get("sync_mode", "backward")
        vals, age = align(tim, arr, targets, mode=mode, kind="cubic" if name == "fmg" else "linear")
        if name in {"fmg", "gaze"}:
            vals = np.clip(vals, 0, 1)
        signals[name], ages[name] = vals, age
        if cfg.get("required", True):
            inputs_ok &= age >= 0
        if name == "imu" and cfg.get("orientation", False):
            qt, quats = read_csv(path, QUAT)
            qs, qa = align(
                qt + int(cfg.get("offset_ns", 0)), quats, targets, mode=mode, kind="quaternion"
            )
            signals["imu_orientation"], ages["imu_orientation"] = qs, qa
            if cfg.get("required", True):
                inputs_ok &= qa >= 0
    return signals, ages, inputs_ok


def collect_geometry(spec, paths, primary):
    t = primary["timestamps"]
    points, pixels = primary["landmarks"], primary["pixels"]
    have_video = primary["indices"] >= 0
    if "landmarks" in paths:
        lt, raw = read_csv(paths["landmarks"], POINTS)
        pick = nearest_indices(lt, t, 15_000_000)
        ok = (pick >= 0) & have_video
        points[ok] = raw[pick[ok]].reshape(-1, 21, 3)
        cal = spec["cameras"]["primary"].get("calibration")
        if cal:
            # Project camera-space points into the RECTIFIED camera, as displayed in the review.
            k, size = np.asarray(cal["K"], float), np.asarray(cal["image_size"])
            project = points @ k.T
            pixels = (
                np.divide(
                    project[:, :, :2],
                    project[:, :, 2:],
                    out=np.full_like(pixels, np.nan),
                    where=project[:, :, 2:] > 0,
                )
                / size
            )
    visible = (
        np.isfinite(pixels[:, 0]).all(axis=1)
        & (pixels[:, 0] >= 0).all(axis=1)
        & (pixels[:, 0] <= 1).all(axis=1)
    )
    if "landmarks" in paths and not spec["cameras"]["primary"].get("calibration"):
        visible = np.isfinite(points[:, 0]).all(axis=1) & (points[:, 0, 2] > 0)
    visible &= have_video
    points[~have_video | (points[:, 0, 2] <= 0)] = np.nan
    if spec["reference"] == "tracked_camera":
        ct, trans = read_csv(paths["camera_pose"], XYZ + QUAT)
        xyz, _ = align(ct, trans[:, :3], t)
        quat, _ = align(ct, trans[:, 3:], t, kind="quaternion")
        points = transform_to_reference(points, np.column_stack([xyz, quat]))
    pose, aperture, metric = wrist_pose(points, spec["aperture_range_m"])
    return pixels, visible, pose, aperture, metric


def write_review(
    dest, stage, targets, primary, pixels, pose, aperture, speed, masks, frame_tasks, fps
):
    out = dest / "review.mp4"
    with av.open(str(out), "w") as video:
        enc = video.add_stream("libx264", rate=fps)
        enc.width = enc.height = SIZE
        enc.pix_fmt = "yuv420p"
        enc.options = {"crf": "18", "preset": "veryfast"}
        for i in range(len(targets)):
            image = stage["primary"][i]
            for j in (0, 4, 5, 8, 9, 17):
                point = pixels[i, j]
                if np.isfinite(point).all() and (point >= 0).all() and (point <= 1).all():
                    cv2.circle(
                        image, tuple(np.rint(point * (SIZE - 1)).astype(int)), 2, (45, 250, 145), -1
                    )
            for packet in enc.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                video.mux(packet)
        for packet in enc.encode():
            video.mux(packet)

    def scalar(v):
        return None if not np.isfinite(v) else float(v)

    with (dest / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "grid_index",
                "grid_timestamp_ns",
                "exposure_ns",
                "source_frame",
                "x_m",
                "y_m",
                "z_m",
                "aperture",
                "speed_m_s",
                "segment",
                *masks,
            ]
        )
        for i in range(len(targets)):
            w.writerow(
                [
                    i,
                    int(targets[i]),
                    int(primary["timestamps"][i]),
                    int(primary["indices"][i]),
                    *[scalar(v) for v in pose[i, :3]],
                    scalar(aperture[i]),
                    scalar(speed[i]),
                    frame_tasks[i],
                    *[int(v[i]) for v in masks.values()],
                ]
            )


def process(spec, paths, formats, work, dest):
    fps, horizon = spec.get("fps", 20), spec.get("horizon", 1)
    cameras = spec["cameras"]
    pts = np.fromiter(
        (t for t, _ in frames(paths["camera/primary"], cameras["primary"]["start_time_ns"])),
        dtype=np.int64,
    )
    if len(pts) < 2:
        raise ValueError("Primary video must contain at least two frames")
    targets = np.arange(pts[0], pts[-1] + 1, NS // fps, dtype=np.int64)
    if len(targets) <= horizon:
        raise ValueError("The recording is shorter than the requested action horizon")
    with h5py.File(work / "normalized-frames.h5", "w") as stage:
        tracker = HandTracker(paths["hand_model"], spec) if "hand_model" in paths else None
        try:
            main = stage_camera(
                paths["camera/primary"],
                cameras["primary"],
                targets,
                stage,
                "primary",
                track=tracker,
                source_times=pts,
            )
            cap = {"primary": main}
            exposures = np.where(main["indices"] >= 0, main["timestamps"], targets)
            for name, camera in cameras.items():
                if name != "primary":
                    cfg = {"tolerance_ms": 15, **camera}
                    cap[name] = stage_camera(paths["camera/" + name], cfg, exposures, stage, name)
        finally:
            if tracker is not None:
                tracker.close()
        primary = cap["primary"]
        t = primary["timestamps"]
        sensors, ages, inputs_ok = sensor_signals(spec, paths, t)
        for name, cam in cameras.items():
            if cam.get("required", True) or name == "primary":
                inputs_ok &= cap[name]["indices"] >= 0
        pixels, visible, raw_pose, aperture, metric = collect_geometry(spec, paths, primary)
        speed, acceler = derivatives(raw_pose[:, :3], t, metric & visible & inputs_ok)
        labels = segment(
            spec,
            paths,
            stage["primary"],
            speed,
            aperture,
            visible & metric & inputs_ok,
            fps,
            visible,
        )
        frame_tasks = np.zeros(len(t), int)
        for j, record in enumerate(labels):
            record.setdefault("target_object", None)
            record.setdefault("contact_established", None)
            frame_tasks[record["start_frame"] : record["end_frame"] + 1] = j
        qc, masks = assess(
            spec,
            visible,
            metric,
            inputs_ok,
            primary["brightness"],
            speed,
            acceler,
            aperture,
            sensors,
            labels,
            horizon,
        )
        qc["sensor_alignment"] = {}
        for name, age in ages.items():
            measured = age[age >= 0]
            qc["sensor_alignment"][name] = {
                "matched_frames": len(measured),
                "missing_frames": int((age < 0).sum()),
                "max_preceding_age_ms": round(float(measured.max()) / 1e6, 4)
                if len(measured)
                else None,
            }
        poses = smooth_poses(
            raw_pose, masks["usable"], t, causal=spec.get("sync_mode", "backward") == "backward"
        )
        clips = (
            [
                action_chunks(poses, aperture, a, b, horizon)
                for a, b in runs(masks["usable"])
                if b - a > horizon
            ]
            if qc["tier"] < 3
            else []
        )
        metadata = {
            "episode_id": spec["episode_id"],
            "source_operator_id": spec.get("source_operator_id"),
            "hardware_rig_type": spec.get("hardware_rig_type"),
            "synthetic": bool(spec.get("synthetic", False)),
            "fps": fps,
            "total_frames": len(t),
            "training_rows": sum(len(x) for x, _ in clips),
            "action_horizon": horizon,
            "reference": spec["reference"],
            "pose_source": "external_camera_landmarks"
            if "landmarks" in paths
            else "mediapipe_calibrated_pnp_estimate"
            if "hand_model" in paths
            else "none",
            "sync_mode": spec.get("sync_mode", "backward"),
            "max_sensor_age_ms": 15,
            "task_metadata": {**spec.get("task_metadata", {}), "high_level_goal": spec["task"]},
            "sub_task_segments": labels,
            "confidence_metrics": qc,
            "formats": sorted(formats) if clips else [],
            "version": __version__,
        }
        write_json(dest / "episode.json", metadata)
        write_review(
            dest, stage, targets, primary, pixels, poses, aperture, speed, masks, frame_tasks, fps
        )
        if clips:
            tier_dir = (
                dest / ("synthetic" if metadata["synthetic"] else "recorded") / f"tier-{qc['tier']}"
            )
            tier_dir.mkdir(parents=True)
            tok = make_tokenizer(paths.get("tokenizer"))
            tok.save(str(tier_dir / "tokenizer.json"))
            h5path = (tier_dir if "hdf5" in formats else work) / "episodes.h5"
            signals = {
                "timestamps": t,
                "grid": targets,
                "pose": poses,
                "aperture": aperture,
                "sensors": sensors,
                "ages": ages,
                "camera_times": {k: c["timestamps"] for k, c in cap.items()},
                "camera_idx": {k: c["indices"] for k, c in cap.items()},
            }
            write_hdf5(h5path, spec, stage, clips, labels, tok, signals, frame_tasks, metadata)
            if "mcap" in formats:
                write_mcap(tier_dir / "episode.mcap", h5path, spec, labels)
            if "lerobot" in formats:
                write_lerobot(tier_dir / "lerobot", h5path, spec, labels)
        else:
            (dest / "quarantine").mkdir()
            write_json(dest / "quarantine" / "diagnostics.json", qc)
        return metadata


def run(manifest, destination, formats=("hdf5", "mcap")):
    manifest, dest = Path(manifest).resolve(), Path(destination).resolve()
    spec = read_json(manifest)
    validate(spec)
    formats = set(formats)
    if not formats or formats - FORMATS:
        raise ValueError("Choose hdf5, mcap, or lerobot export")
    paths = media_paths(spec, manifest.parent)

    def source_hashes():
        return {k: sha256(path) for k, path in sorted({"manifest": manifest, **paths}.items())}

    hashes = source_hashes()
    recipe = {"inputs": hashes, "version": __version__, "formats": sorted(formats)}
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()

    def existing():
        receipt = dest / "receipt.json"
        if receipt.is_file():
            record = read_json(receipt)
            if record.get("fingerprint") == key:
                expected = record.get("output_sha256", {})
                available = {
                    p.relative_to(dest).as_posix(): p for p in dest.rglob("*") if p.is_file()
                }
                if expected and all(
                    k in available and sha256(available[k]) == h for k, h in expected.items()
                ):
                    return read_json(dest / "episode.json")
                raise ValueError(
                    f"Existing result failed output verification; use a new directory: {dest}"
                )
        raise FileExistsError(
            f"{dest} already contains a different run; select a new output directory"
        )

    if dest.exists():
        return existing()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".ego-run-") as tmp:
        work = Path(tmp)
        result = work / "result"
        result.mkdir()
        metadata = process(spec, paths, formats, work, result)
        if hashes != source_hashes():
            raise RuntimeError("An input changed during the run; retry after capture is complete")
        outputs = {
            p.relative_to(result).as_posix(): sha256(p) for p in result.rglob("*") if p.is_file()
        }
        write_json(result / "receipt.json", dict(recipe, fingerprint=key, output_sha256=outputs))
        try:
            result.rename(dest)
        except OSError:
            if dest.exists():
                return existing()
            raise
    return metadata
