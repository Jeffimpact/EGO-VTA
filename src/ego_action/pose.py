"""Camera-space hand reconstruction, wrist frames and body-frame action chunks."""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from .common import runs, unit


class HandObservation(NamedTuple):
    uv: np.ndarray
    camera_m: np.ndarray | None
    local_m: np.ndarray
    handedness_score: float


def recover_camera_points(model_points, image_points, k, palm_width, max_error_px):
    """Estimate scale with an operator measurement then solve camera extrinsics.

    Inputs are 21 hand-centred 3-D landmarks and 21 rectified pixel locations.
    The reprojection gate rejects visibly inconsistent reconstructions.
    """
    span = np.linalg.norm(model_points[5] - model_points[17])
    if not np.isfinite(model_points).all() or span < 1e-5 or not np.isfinite(image_points).all():
        return None
    points = np.ascontiguousarray(model_points * (palm_width / span), dtype=np.float64)
    px = np.ascontiguousarray(image_points, dtype=np.float64)
    stable = np.array([0, 1, 2, 5, 9, 13, 17])
    try:
        ok, rvec, tvec = cv2.solvePnP(points[stable], px[stable], k, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(points[stable], px[stable], k, None, rvec, tvec)
    except cv2.error:
        return None
    camera = points @ cv2.Rodrigues(rvec)[0].T + tvec.ravel()
    projected = cv2.projectPoints(points, rvec, tvec, k, None)[0].reshape(-1, 2)
    err = np.sqrt(np.mean(np.sum(np.square(projected - px), axis=1)))
    if (camera[:, 2] <= 0).any() or not np.isfinite(err) or err > max_error_px:
        return None
    return camera


class HandTracker:
    def __init__(self, model, spec):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ValueError(
                "Video tracking requires the 'tracking' extra and a hand_landmarker.task file"
            ) from exc
        self.mp, self.spec, self.origin = mp, spec, None
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def __call__(self, image, stamp, k):
        if self.origin is None:
            self.origin = stamp
        result = self.detector.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(image)),
            int((stamp - self.origin) // 1_000_000),
        )
        wanted = self.spec.get("hand", "Right")
        if self.spec.get("swap_handedness", False):
            wanted = {"Right": "Left", "Left": "Right"}[wanted]
        candidates = [
            i
            for i, cats in enumerate(result.handedness)
            if cats and cats[0].category_name == wanted and cats[0].score >= 0.65
        ]
        if len(candidates) != 1:
            return None
        i = candidates[0]
        uv = np.array([[p.x, p.y] for p in result.hand_landmarks[i]])
        world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[i]])
        palm = self.spec.get("palm_width_m")
        camera = None
        if k is not None and palm is not None:
            camera = recover_camera_points(
                world,
                uv * [image.shape[1], image.shape[0]],
                k,
                palm,
                self.spec.get("max_reprojection_px", 8.0),
            )
        return HandObservation(uv, camera, world, float(result.handedness[i][0].score))

    def close(self):
        self.detector.close()


def transform_to_reference(points, transform):
    """T_reference_camera poses are [x,y,z,qw,qx,qy,qz]."""
    result = np.full_like(points, np.nan)
    ok = np.isfinite(transform).all(axis=1) & (np.linalg.norm(transform[:, 3:], axis=1) > 1e-8)
    rot = Rotation.from_quat(transform[ok, 3:][:, [1, 2, 3, 0]]).as_matrix() if ok.any() else None
    if rot is not None:
        result[ok] = np.einsum("nij,nkj->nki", rot, points[ok]) + transform[ok, None, :3]
    return result


def wrist_pose(points, limits):
    # X across the palm (pinky to index), Y toward middle MCP, Z completes the triad.
    x = unit(points[:, 5] - points[:, 17])
    z = unit(np.cross(x, points[:, 9] - points[:, 0]))
    y = np.cross(z, x)
    mats = np.stack([x, y, z], axis=-1)
    ok = np.isfinite(points[:, [0, 4, 5, 8, 9, 17]]).all(axis=(1, 2)) & np.isfinite(mats).all(
        axis=(1, 2)
    )
    pose = np.full((len(points), 7), np.nan)
    if ok.any():
        q = Rotation.from_matrix(mats[ok]).as_quat()
        pose[ok] = np.column_stack([points[ok, 0], q[:, [3, 0, 1, 2]]])
    aperture = np.clip(
        (np.linalg.norm(points[:, 4] - points[:, 8], axis=1) - limits[0]) / (limits[1] - limits[0]),
        0,
        1,
    )
    return pose, aperture, ok


def smooth_poses(poses, good, timestamps, causal=True):
    """Local polynomial smoothing, separately on every contiguous valid span."""
    out = poses.copy()
    for a, b in runs(good):
        q = poses[a:b, 3:][:, [1, 2, 3, 0]].copy()
        for i in range(1, len(q)):
            if np.dot(q[i - 1], q[i]) < 0:
                q[i] *= -1
        w = min(5, len(q) if len(q) % 2 else len(q) - 1)
        if w >= 5:
            # A 30-to-20 Hz decimation alternates short and long exposure intervals.
            # Fit against actual exposure time so constant velocity stays constant.
            times = (timestamps[a:b] - timestamps[a]) / 1e9
            signal = np.column_stack([poses[a:b, :3], q])
            if not causal and np.allclose(
                np.diff(times), np.median(np.diff(times)), rtol=1e-5, atol=1e-8
            ):
                fitted = savgol_filter(signal, w, 2, axis=0, mode="interp")
            else:
                fitted = signal.copy()
                for j in range(w - 1 if causal else 0, len(times)):
                    t = times[j]
                    lo = j - w + 1 if causal else min(max(0, j - w // 2), len(times) - w)
                    dt = times[lo : lo + w] - t
                    design = np.column_stack([np.ones(w), dt, dt**2])
                    fitted[j] = np.linalg.lstsq(design, signal[lo : lo + w], rcond=None)[0][0]
            out[a:b, :3] = fitted[:, :3]
            q = unit(fitted[:, 3:])
        out[a:b, 3:] = q[:, [3, 0, 1, 2]]
    return out


def action_chunks(poses, aperture, start, end, horizon):
    """All H future poses relative to the same anchor; no padded terminal actions."""
    anchor = np.arange(start, end - horizon)
    base = Rotation.from_quat(poses[anchor, 3:][:, [1, 2, 3, 0]])
    chunks = np.empty((len(anchor), horizon, 7), np.float32)
    for step in range(1, horizon + 1):
        future = anchor + step
        rot = Rotation.from_quat(poses[future, 3:][:, [1, 2, 3, 0]])
        chunks[:, step - 1, :3] = base.inv().apply(poses[future, :3] - poses[anchor, :3])
        chunks[:, step - 1, 3:6] = (base.inv() * rot).as_rotvec()
        chunks[:, step - 1, 6] = aperture[future]
    return anchor, chunks
