"""Presentation-time video sampling and bounded alignment of independent clocks."""

from __future__ import annotations

import av
import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from .common import NS, SIZE, unit


def nearest_indices(times, targets, tolerance_ns):
    """Closest sample; ties choose the earlier one. Return -1 when out of range."""
    j = np.searchsorted(times, targets)
    left, right = np.clip(j - 1, 0, len(times) - 1), np.clip(j, 0, len(times) - 1)
    i = np.where(abs(times[right] - targets) < abs(times[left] - targets), right, left)
    return np.where(abs(times[i] - targets) <= tolerance_ns, i, -1)


def align(times, values, targets, *, tolerance_ns=15_000_000, mode="backward", kind="linear"):
    """Offline interpolation needs observations on both sides within tolerance."""
    if mode not in {"backward", "interpolate"}:
        raise ValueError("sync_mode must be backward or interpolate")
    finite = np.isfinite(values).all(axis=1)
    if kind == "quaternion":
        finite &= np.linalg.norm(values, axis=1) > 1e-8
    times, values = times[finite], values[finite]
    out = np.full((len(targets), values.shape[1]), np.nan)
    age = np.full(len(targets), -1, dtype=np.int64)
    if not len(times):
        return out, age
    if kind == "quaternion":
        values = unit(values)
    prev = np.searchsorted(times, targets, side="right") - 1
    age_candidate = targets - times[np.maximum(prev, 0)]
    ok = (prev >= 0) & (age_candidate <= tolerance_ns)
    if mode == "backward":
        out[ok] = values[prev[ok]]
        age[ok] = age_candidate[ok]
        return out, age
    exact = ok & (age_candidate == 0)
    out[exact], age[exact] = values[prev[exact]], 0
    for row in np.flatnonzero(ok & ~exact & (prev + 1 < len(times))):
        i, target = prev[row], targets[row]
        if times[i + 1] - target > tolerance_ns:
            continue
        a = (target - times[i]) / (times[i + 1] - times[i])
        if kind == "quaternion":  # scipy uses xyzw; the capture contract uses wxyz.
            pair = Rotation.from_quat(values[i : i + 2][:, [1, 2, 3, 0]])
            out[row] = Slerp([0.0, 1.0], pair)([a]).as_quat()[0, [3, 0, 1, 2]]
        elif kind == "cubic":
            # Only nearby, contiguous support points participate in the cubic fit.
            lo, hi = max(0, i - 1), min(len(times), i + 3)
            if hi - lo == 4 and np.max(np.diff(times[lo:hi])) <= 2 * tolerance_ns:
                x = (times[lo:hi] - times[i]) / NS
                out[row] = CubicSpline(x, values[lo:hi])((target - times[i]) / NS)
            else:
                out[row] = (1 - a) * values[i] + a * values[i + 1]
        else:
            out[row] = (1 - a) * values[i] + a * values[i + 1]
        age[row] = age_candidate[row]
    return out, age


def frames(path, start_ns):
    """The first decoded PTS is placed at start_ns on the capture's shared clock."""
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        origin, previous = None, None
        for frame in video.decode(stream):
            if frame.pts is None:
                raise ValueError(f"{path}: missing video PTS")
            stamp = round(frame.pts * frame.time_base * NS)
            if origin is None:
                origin = stamp
            if previous is not None and stamp <= previous:
                raise ValueError(f"{path}: non-monotonic presentation timestamps")
            previous = stamp
            yield int(start_ns) + stamp - origin, frame


def calibration(camera, shape):
    cal = camera.get("calibration")
    if not cal:
        return None
    h, w = shape
    if cal.get("image_size") != [w, h]:
        raise ValueError(f"Calibration image_size must match decoded width and height {[w, h]}")
    k = np.asarray(cal["K"], float)
    d = np.asarray(cal.get("D", [0] * 5), float)
    model = cal.get("model", "pinhole")
    if (
        k.shape != (3, 3)
        or not np.isfinite(k).all()
        or min(k[0, 0], k[1, 1]) <= 0
        or not np.allclose(k[2], [0, 0, 1])
        or not np.isfinite(d).all()
        or model not in {"pinhole", "fisheye"}
    ):
        raise ValueError("Invalid camera intrinsics or lens model")
    if model == "fisheye":
        if d.size != 4:
            raise ValueError("Fisheye calibration requires four distortion coefficients")
        maps = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, (w, h), cv2.CV_32FC1)
    else:
        if d.size not in {4, 5, 8, 12, 14}:
            raise ValueError("Invalid pinhole distortion coefficient count")
        maps = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), cv2.CV_32FC1)
    return k, maps


def stage_camera(path, camera, targets, h5, name, track=None, source_times=None):
    """Two streaming passes keep full-resolution RGB out of resident memory."""
    start = camera["start_time_ns"]
    times = (
        source_times
        if source_times is not None
        else np.fromiter((t for t, _ in frames(path, start)), dtype=np.int64)
    )
    if len(times) < 2:
        raise ValueError(f"{path}: at least two video frames required")
    idx = nearest_indices(times, targets, round(camera.get("tolerance_ms", 25) * 1e6))
    # A decoded frame cannot silently become a second independent observation.
    prev = -1
    for j, selected in enumerate(idx):
        if selected >= 0 and selected == prev:
            idx[j] = -1
        elif selected >= 0:
            prev = selected
    mapping = {int(i): n for n, i in enumerate(idx) if i >= 0}
    last_requested = max(mapping, default=-1)
    rgb = h5.create_dataset(
        name,
        (len(targets), SIZE, SIZE, 3),
        dtype="u1",
        chunks=(1, SIZE, SIZE, 3),
        compression="lzf",
    )
    brightness = np.full((len(targets), 2), np.nan)
    landmarks = np.full((len(targets), 21, 3), np.nan)
    pixels = np.full((len(targets), 21, 2), np.nan)
    local_points = np.full((len(targets), 21, 3), np.nan, dtype=np.float32)
    handedness_score = np.full(len(targets), np.nan, dtype=np.float32)
    inferred = np.zeros(len(targets), bool)
    rect, source_shape = None, None
    for i, (stamp, frame) in enumerate(frames(path, start)):
        if i > last_requested:
            break
        if i not in mapping:
            continue
        n = mapping[i]
        image = frame.to_ndarray(format="rgb24")
        rotation = int(getattr(frame, "rotation", 0)) % 360
        if rotation:
            if rotation % 90:
                raise ValueError("Supply an upright video for a non-quarter-turn display matrix")
            image = np.ascontiguousarray(np.rot90(image, rotation // 90))
        if source_shape is None:
            source_shape = image.shape[:2]
            rect = calibration(camera, source_shape)
        if image.shape[:2] != source_shape:
            raise ValueError(f"{path}: decoded resolution changes within the recording")
        if rect is not None:
            image = cv2.remap(image, *rect[1], cv2.INTER_LINEAR)
        if track is not None:
            result = track(image, stamp, None if rect is None else rect[0])
            if result is not None:
                pixels[n] = result.uv
                local_points[n] = result.local_m
                handedness_score[n] = result.handedness_score
                if result.camera_m is not None:
                    landmarks[n], inferred[n] = result.camera_m, True
        small = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        rgb[n] = small
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        brightness[n] = np.percentile(gray, 95), np.mean((gray <= 2) | (gray >= 253))
    return {
        "indices": idx,
        "timestamps": np.where(idx >= 0, times[np.maximum(idx, 0)], -1),
        "brightness": brightness,
        "landmarks": landmarks,
        "pixels": pixels,
        "inferred": inferred,
        "local_points": local_points,
        "handedness_score": handedness_score,
        "source_shape": source_shape,
    }
