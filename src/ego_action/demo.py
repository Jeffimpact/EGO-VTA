"""A reproducible moving-camera recording with independent world-space ground truth."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .common import FMG, IMU, NS, POINTS, QUAT, XYZ, read_json, write_json

EPOCH = 1_700_000_000 * NS
PALM = 0.070
LIMITS = [0.012, 0.095]
W, H = 384, 288
K = np.array([[340.0, 0, W / 2], [0, 340.0, H / 2], [0, 0, 1.0]])
VELOCITY = np.array([0.052, -0.012, 0.0])


def truth(t):
    wrist = np.array([-0.12, 0.005, 0.55]) + t * VELOCITY
    wrist_r = Rotation.from_euler("z", 0.25 + 0.25 * t)
    cam = np.array([0.045 * np.sin(t * 2.8), 0.01 * np.cos(t * 2.1), 0.0])
    cam_r = Rotation.from_euler("z", 0.22 * np.sin(2.1 * t))
    grip = np.clip(1 - (t - 0.6) / 0.65, 0, 1) + np.clip((t - 2.65) / 0.65, 0, 1)
    return wrist, wrist_r, cam, cam_r, float(grip)


def sample(t):
    wrist, wr, cam, cr, g = truth(t)
    hand = np.zeros((21, 3))
    for first, x, length in (
        (1, 0.05, 0.05),
        (5, 0.035, 0.088),
        (9, 0.0, 0.102),
        (13, -0.018, 0.094),
        (17, -0.035, 0.083),
    ):
        for k in range(4):
            hand[first + k] = [x, 0.05 + length * k / 3, 0.006 * np.sin(k * 0.7)]
    hand[9] = [0, 0.065, 0]
    hand[8] = [0.025, 0.13, 0.002]
    hand[4] = hand[8] + [LIMITS[0] + g * (LIMITS[1] - LIMITS[0]), 0, 0]
    pts = cr.inv().apply(wr.apply(hand) + wrist - cam)
    return pts, cam, cr.as_quat()[[3, 0, 1, 2]]


def render(points, faulted=False):
    image = np.full((H, W, 3), [63, 77, 81], dtype=np.uint8)
    for x in range(0, W, 24):
        cv2.line(image, (x, 0), (x, H), (69, 85, 88), 1)
    for y in range(0, H, 24):
        cv2.line(image, (0, y), (W, y), (69, 85, 88), 1)
    cv2.rectangle(image, (183, 164), (215, 191), (208, 137, 71), -1)
    if not faulted:
        pr = points @ K.T
        uv = np.round(pr[:, :2] / pr[:, 2:]).astype(np.int32)
        for j in (1, 5, 9, 13, 17):
            cv2.polylines(
                image, [uv[[0, j, j + 1, j + 2, j + 3]]], False, (232, 219, 195), 3, cv2.LINE_AA
            )
        for p in uv:
            cv2.circle(image, tuple(p), 3, (245, 220, 175), -1)
    cv2.putText(
        image,
        "SYNTHETIC CALIBRATION EPISODE",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (185, 205, 207),
        1,
    )
    return image


def write_csv(path, fields, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ns", *fields])
        writer.writerows(rows)


def write_demo(path, fault):
    landmarks, camera, ground = [], [], []
    with av.open(str(path / "primary.mp4"), "w") as out:
        encoder = out.add_stream("libx264", rate=30)
        encoder.width, encoder.height, encoder.pix_fmt = W, H, "yuv420p"
        for i in range(120):
            t, stamp = i / 30, EPOCH + round(i * NS / 30)
            points, cam, quat = sample(t)
            missing = fault == "occlusion" and 1.0 <= t < 1.5
            image = render(points, missing)
            for packet in encoder.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                out.mux(packet)
            landmarks.append([stamp, *(np.full(63, np.nan) if missing else points.ravel())])
            camera.append([stamp, *cam, *quat])
            wrist, wrist_r, _, _, g = truth(t)
            ground.append([stamp, *wrist, *wrist_r.as_quat()[[3, 0, 1, 2]], g])
        for packet in encoder.encode():
            out.mux(packet)
    write_csv(path / "landmarks.csv", POINTS, landmarks)
    write_csv(path / "camera_pose.csv", XYZ + QUAT, camera)
    write_csv(path / "ground_truth.csv", XYZ + QUAT + ("aperture",), ground)
    imu, force, rig = [], [], []
    for i in range(401):
        t, stamp = i / 100, EPOCH + i * 10_000_000
        g = truth(t)[-1]
        imu.append([stamp, 0.0, 0.0, 9.81, 0.0, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0])
        force.append([stamp, *([0.0 if fault == "force" else (0.08 + (1 - g) * 0.65)] * 6)])
        rig.append([stamp, 0.0, 420.0])
    write_csv(path / "imu.csv", IMU + QUAT, imu)
    write_csv(path / "fmg.csv", FMG, force)
    write_csv(path / "rig.csv", ("pitch_deg", "lux"), rig)
    annotations = [
        {
            "start_frame": 0,
            "end_frame": 11,
            "instruction": "Reach toward the block",
            "target_object": "block",
        },
        {
            "start_frame": 12,
            "end_frame": 26,
            "instruction": "Grasp the block",
            "target_object": "block",
        },
        {
            "start_frame": 27,
            "end_frame": 52,
            "instruction": "Carry the block across the mat",
            "target_object": "block",
        },
        {
            "start_frame": 53,
            "end_frame": 79,
            "instruction": "Release the block",
            "target_object": "block",
        },
    ]
    write_json(path / "annotations.json", annotations)
    spec = {
        "episode_id": "fixture_" + fault,
        "synthetic": True,
        "fixture_version": 1,
        "fault": fault,
        "task": "Move the block across the mat",
        "source_operator_id": "fixture",
        "hardware_rig_type": "synthetic_moving_camera",
        "reference": "tracked_camera",
        "fps": 20,
        "horizon": 4,
        "hand": "Right",
        "aperture_range_m": LIMITS,
        "palm_width_m": PALM,
        "cameras": {
            "primary": {
                "file": "primary.mp4",
                "start_time_ns": EPOCH,
                "calibration": {
                    "model": "pinhole",
                    "image_size": [W, H],
                    "K": K.tolist(),
                    "D": [0] * 5,
                },
            }
        },
        "camera_pose": "camera_pose.csv",
        "landmarks": "landmarks.csv",
        "landmark_quality": "validated_metric",
        "sensors": {
            "imu": {"file": "imu.csv", "orientation": True},
            "fmg": {"file": "fmg.csv"},
            "rig": {"file": "rig.csv"},
        },
        "annotations": "annotations.json",
        "sync_mode": "backward",
    }
    write_json(path / "episode.json", spec)


def create_demo(path, fault="none"):
    path = Path(path)

    def existing():
        saved = path / "episode.json"
        if (
            saved.is_file()
            and read_json(saved).get("fixture_version") == 1
            and read_json(saved).get("fault") == fault
        ):
            return saved
        raise FileExistsError(f"Choose a new demo folder; {path} already exists")

    if path.exists():
        return existing()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        result = Path(tmp) / "capture"
        result.mkdir()
        write_demo(result, fault)
        try:
            result.rename(path)
        except OSError:
            if path.exists():
                return existing()
            raise
    return path / "episode.json"
