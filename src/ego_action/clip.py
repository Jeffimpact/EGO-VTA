"""Turn one uninstrumented MP4 into a reviewed human-hand feature tensor on CPU."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import av
import cv2
import h5py
import numpy as np

from .common import NS, read_csv, read_json, runs, sha256, write_json
from .ingest import align, frames, stage_camera
from .pose import HandTracker

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_SHA = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
FPS = 20
EDGES = [(0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17)]
EDGES += [(base + i, base + i + 1) for base in (1, 5, 9, 13, 17) for i in range(3)]


def model_path(provided):
    if provided:
        path = Path(provided)
        if not path.is_file():
            raise ValueError(f"Hand model is missing at {path}")
        return path
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ego-vta"
    cache.mkdir(exist_ok=True, parents=True)
    path = cache / "hand_landmarker-v1.task"
    if path.is_file() and sha256(path) == MODEL_SHA:
        return path
    with tempfile.TemporaryDirectory(dir=cache) as tmp:
        incoming = Path(tmp) / "download"
        with urllib.request.urlopen(MODEL_URL, timeout=60) as res, incoming.open("wb") as file:
            shutil.copyfileobj(res, file)
        if sha256(incoming) != MODEL_SHA:
            raise ValueError("Downloaded model failed its SHA-256 check")
        incoming.replace(path)
    return path


def hand_features(track, pinch_range, calibrated):
    uv = track["pixels"]
    local = track["local_points"]
    width = np.linalg.norm(local[:, 5] - local[:, 17], axis=1)
    gap = np.linalg.norm(local[:, 4] - local[:, 8], axis=1)
    ratio = np.divide(gap, width, out=np.full_like(width, np.nan), where=width > 1e-5)
    low, high = pinch_range
    if not (0 <= low < high and np.isfinite(high)):
        raise ValueError("Pinch calibration ratios must be finite and strictly increasing")
    grip = np.clip((ratio - low) / (high - low), 0, 1)
    visible = np.isfinite(uv[:, [0, 4, 8, 5, 17]]).all(axis=(1, 2))
    visible &= ((uv[:, [0, 4, 8]] >= 0) & (uv[:, [0, 4, 8]] <= 1)).all(axis=(1, 2))
    uv_wrist = uv[:, 0].copy()
    if calibrated:
        spatial = track["landmarks"][:, 0]
        fields = ["wrist_x_m", "wrist_y_m", "wrist_z_m", "aperture"]
    else:
        h, w = track["source_shape"]
        span = np.linalg.norm((uv[:, 5] - uv[:, 17]) * [w / h, 1], axis=1)
        spatial = np.column_stack([uv_wrist, span])
        fields = ["wrist_u", "wrist_v", "projected_palm_span_in_image_heights", "aperture"]
    state = np.column_stack([spatial, grip]).astype("f4")
    valid = visible & np.isfinite(state).all(axis=1) & (track["indices"] >= 0)
    state[~valid] = np.nan
    actions = np.full_like(state, np.nan)
    transitions = np.r_[valid[:-1] & valid[1:], False]
    i = np.flatnonzero(transitions)
    actions[i, :3] = state[i + 1, :3] - state[i, :3]
    actions[i, 3] = state[i + 1, 3]
    return state, actions, fields, ratio, valid, transitions


def qc_report(track, valid, grid, pitch_path=None):
    reasons = []
    missing = runs(~valid)
    for a, b in missing:
        if b - a > 5:
            reasons.append({"code": "long_tracking_gap", "start_frame": a, "end_frame": b - 1})
    if np.mean(~valid) > 0.05:
        reasons.append({"code": "low_hand_coverage", "missing_fraction": float(np.mean(~valid))})
    pitch_status = "unmeasured"
    if pitch_path:
        pt, values = read_csv(pitch_path, ("pitch_deg",))
        pitch, age = align(pt, values, track["timestamps"])
        pitch_status = "provided_calibrated_pitch_log"
        if (age < 0).any():
            reasons.append({"code": "pitch_log_time_gap", "frames": int((age < 0).sum())})
        for a, b in runs(abs(pitch[:, 0]) > 35):
            reasons.append(
                {"code": "calibrated_pitch_exceeds_35_deg", "start_frame": a, "end_frame": b - 1}
            )
    return {
        "status": "rejected" if reasons else "accepted_for_feature_review",
        "policy_hz": FPS,
        "grid_rows": len(grid),
        "valid_hand_rows": int(valid.sum()),
        "coverage": float(valid.mean()),
        "longest_invalid_run": max((b - a for a, b in missing), default=0),
        "valid_spans_start_stop": runs(valid),
        "pitch_evidence": pitch_status,
        "warnings": reasons,
    }


def draw_hand(img, coords, valid, color=(81, 225, 166)):
    if not valid or not np.isfinite(coords).all():
        return
    pts = np.rint(coords).astype(np.int32)
    for a, b in EDGES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(
            img, tuple(p), 4 if i in (0, 4, 8) else 2, (250, 177, 91) if i in (4, 8) else color, -1
        )


def draw_text(img, text, xy, scale=0.47, color=(215, 225, 223)):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def diagnostics(dest, frames_rgb, track, state, valid, hand, make_gif):
    n = len(state)
    bg = np.full((348, 960, 3), (24, 36, 38), np.uint8)
    draw_text(bg, hand + " hand | source view", (12, 27), 0.58)
    draw_text(bg, "Hand-local 3D shape", (404, 27), 0.58)
    draw_text(bg, "Normalized pinch aperture", (661, 27), 0.58)
    draw_text(bg, "open  1", (661, 65), 0.40)
    draw_text(bg, "closed  0", (655, 270), 0.40)
    x0, x1, y0, y1 = 715, 946, 66, 263
    cv2.rectangle(bg, (x0, y0), (x1, y1), (84, 105, 104), 1)
    xx = np.linspace(x0, x1, n).round().astype(int)
    yy = y1 - np.nan_to_num(state[:, 3]) * (y1 - y0)
    for a, b in runs(valid):
        pts = np.column_stack([xx[a:b], yy[a:b]]).astype("int32")
        if len(pts) > 1:
            cv2.polylines(bg, [pts], False, (103, 211, 174), 2, cv2.LINE_AA)
    h, w = track["source_shape"]
    scale = min(380 / w, 245 / h)
    dims = (max(1, round(w * scale)), max(1, round(h * scale)))
    left = 9 + (380 - dims[0]) // 2
    top = 42 + (245 - dims[1]) // 2
    projection = np.array([[0.86, -0.16], [0.0, 0.94], [0.5, 0.28]])
    corners = np.array([[bool(j & (1 << k)) for k in range(3)] for j in range(8)])
    box_edges = [(a, a ^ (1 << j)) for a in range(8) for j in range(3) if not a & (1 << j)]
    gif_frames = []
    with av.open(str(dest / "diagnostic.mp4"), "w") as out:
        stream = out.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = 960, 348, "yuv420p"
        stream.options = {"crf": "19", "preset": "veryfast"}
        for i in range(n):
            img = bg.copy()
            rgb = cv2.resize(frames_rgb[i], dims)
            img[top : top + dims[1], left : left + dims[0]] = rgb
            draw_hand(img, np.array([left, top]) + track["pixels"][i] * dims, valid[i])
            if valid[i]:
                points = track["local_points"][i]
                # Use a fixed scale in all frames. The wrist and the bounding box are hand-centred.
                screen = lambda pts: np.rint((pts @ projection) * 1220 + [525, 161]).astype("int32")
                box = screen(np.where(corners, points.max(axis=0), points.min(axis=0)))
                for a, b in box_edges:
                    cv2.line(img, tuple(box[a]), tuple(box[b]), (79, 108, 109), 1, cv2.LINE_AA)
                draw_hand(img, screen(points), True)
            cv2.line(img, (xx[i], y0), (xx[i], y1), (228, 149, 86), 1)
            if valid[i]:
                cv2.circle(img, (xx[i], round(yy[i])), 4, (252, 171, 89), -1)
            draw_text(img, f"{i / FPS:.2f} s   |   frame {i + 1}/{n}", (12, 318), 0.53)
            draw_text(img, "local axes follow the hand model", (403, 316), 0.39)
            draw_text(
                img, f"aperture {state[i, 3]:.2f}" if valid[i] else "tracking gap", (698, 316), 0.49
            )
            for pkt in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(pkt)
            if make_gif and i < 10 * FPS and i % 2 == 0:
                from PIL import Image

                gif_frames.append(
                    Image.fromarray(cv2.resize(img, (640, 232), interpolation=cv2.INTER_AREA))
                )
        for pkt in stream.encode():
            out.mux(pkt)
    if gif_frames:
        gif_frames[0].save(
            dest / "preview.gif", save_all=True, append_images=gif_frames[1:], duration=100, loop=0
        )


def synthetic_streams(h5, exposures):
    first, last = max(0, int(exposures.min())), int(exposures.max())
    times = np.arange(first, last + 10_000_001, 10_000_000, dtype=np.int64)
    phase = (times - first)[:, None] / NS * np.pi * 2 + np.arange(6) * 0.15
    for name, values in {"imu": np.sin(phase), "fmg": (1 + np.sin(phase)) / 2}.items():
        group = h5.create_group("synthetic/" + name)
        values = values.astype("f4")
        group.attrs["provenance"] = "synthetic_sync_test_only"
        group.create_dataset("raw_timestamp_ns", data=times)
        group.create_dataset("raw_values", data=values)
        aligned, age = align(times, values, exposures)
        group.create_dataset("aligned", data=aligned.astype("f4"))
        group.create_dataset("age_ns", data=age)


def process_clip(
    source,
    output,
    *,
    model=None,
    hand="Right",
    pinch_range=(0.05, 1.5),
    calibration=None,
    pitch_csv=None,
    mock_sensors=False,
    gif=False,
    seconds=None,
    start_second=0.0,
):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    mk = model_path(model)
    cfg = read_json(calibration) if calibration else None
    if cfg and not (0 < float(cfg["palm_width_m"]) < 0.3):
        raise ValueError("Provide measured palm_width_m in the calibration file")
    camera = {"start_time_ns": 0}
    spec = {"hand": hand}
    if cfg:
        camera["calibration"], spec["palm_width_m"] = cfg, cfg["palm_width_m"]
    if not np.isfinite(start_second) or start_second < 0:
        raise ValueError("start-second must be nonnegative")
    start_ns = round(start_second * NS)
    timestamps = (t for t, _ in frames(source, 0))
    if seconds is not None:
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("seconds must be positive")
        limit = start_ns + round(seconds * NS)
        timestamps = itertools.takewhile(lambda t: t < limit, timestamps)
    pts = np.fromiter(timestamps, dtype=np.int64)
    if len(pts) < 2:
        raise ValueError("At least two decoded frames are required")
    grid = np.arange(start_ns, int(pts[-1]) + 1, NS // FPS, dtype=np.int64)
    if len(grid) < 2:
        raise ValueError("The selected interval must contain at least two policy frames")
    recipe = {
        "version": 1,
        "input_sha256": sha256(source),
        "model_sha256": sha256(mk),
        "hand": hand,
        "pinch_range": list(pinch_range),
        "calibration": cfg,
        "pitch_log_sha256": sha256(pitch_csv) if pitch_csv else None,
        "mock_sensors": mock_sensors,
        "gif": gif,
        "seconds": seconds,
        "start_second": start_second,
    }

    def reuse():
        p = output / "qc.json"
        if p.exists():
            saved = read_json(p)
            if (
                saved.get("recipe") == recipe
                and saved.get("files")
                and all(
                    Path(k).name == k and (output / k).is_file() and sha256(output / k) == v
                    for k, v in saved.get("files", {}).items()
                )
            ):
                return saved
        raise FileExistsError(f"Use a new output directory or the original inputs: {output}")

    # Also allow a pre-created empty output directory.
    if output.exists() and any(output.iterdir()):
        return reuse()
    output.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".clip-") as tmp:
        tmp = Path(tmp)
        dest = tmp / "result"
        dest.mkdir()
        with h5py.File(tmp / "stage.h5", "w") as stage:
            tracker = HandTracker(mk, spec)
            try:
                track = stage_camera(
                    source, camera, grid, stage, "images", track=tracker, source_times=pts
                )
            finally:
                tracker.close()
            state, actions, fields, ratio, valid, transitions = hand_features(
                track, pinch_range, bool(cfg)
            )
            report = qc_report(track, valid, grid, pitch_csv)
            report.update(
                decoded_frames=len(pts),
                selected_video_frames=int((track["indices"] >= 0).sum()),
                valid_transitions=int(transitions.sum()),
                state_fields=fields,
                units="camera_metres" if cfg else "image_coordinates_and_apparent_size",
                mock_sensors_used_for_qc=False,
                recipe=recipe,
            )
            diagnostics(dest, stage["images"], track, state, valid, hand, gif)
            if report["status"] == "accepted_for_feature_review":
                with h5py.File(dest / "episode.h5", "w") as h5:
                    stage.copy("images", h5)
                    h5.attrs.update(
                        schema="ego_action.clip.v1",
                        fps=FPS,
                        hand=hand,
                        episode_role="human_hand_observation_proxies",
                        source_file=source.name,
                        frame="per_exposure_camera" if cfg else "normalized_image",
                        pinch_ratio_range=np.asarray(pinch_range),
                        metric_camera_fit=bool(cfg),
                    )
                    h5["images"].attrs.update(
                        layout="THWC", color="RGB", normalization="uint8 0..255"
                    )
                    for name, values in {
                        "timestamp_ns": grid,
                        "source_timestamp_ns": track["timestamps"],
                        "source_frame_index": track["indices"],
                        "state": state,
                        "actions": actions,
                        "valid/landmarks": valid,
                        "valid/action": transitions,
                        "hand/landmarks_uv": track["pixels"].astype("f4"),
                        "hand/landmarks_hand_centered_m": track["local_points"],
                        "hand/pinch_ratio": ratio,
                        "hand/handedness_score": track["handedness_score"],
                    }.items():
                        h5.create_dataset(name, data=values, compression="gzip")
                    if cfg:
                        h5.create_dataset(
                            "hand/wrist_camera_m", data=track["landmarks"][:, 0].astype("f4")
                        )
                    h5["state"].attrs["fields"] = json.dumps(fields)
                    h5["actions"].attrs["fields"] = json.dumps(
                        ["delta_" + k for k in fields[:3]] + ["next_aperture"]
                    )
                    if mock_sensors:
                        synthetic_streams(h5, track["timestamps"])
        report["files"] = {p.name: sha256(p) for p in dest.iterdir() if p.is_file()}
        write_json(dest / "qc.json", report)
        try:
            dest.rename(output)
        except OSError:
            if output.exists():
                return reuse()
            raise
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        help="Local MediaPipe bundle. Default downloads the pinned Google model once",
    )
    parser.add_argument(
        "--start-second",
        type=float,
        default=0.0,
        help="Start of the analysis window in the source video",
    )
    parser.add_argument("--hand", choices=["Right", "Left"], default="Right")
    parser.add_argument("--seconds", type=float, help="Analyze N seconds from the selected start")
    parser.add_argument(
        "--pinch-range",
        type=float,
        nargs=2,
        default=[0.05, 1.5],
        metavar=("CLOSED", "OPEN"),
        help="Closed and open thumb–index distances in palm widths",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="JSON with K, D, image_size, model and measured palm_width_m",
    )
    parser.add_argument(
        "--pitch-csv",
        type=Path,
        help="Optional calibrated timestamp_ns,pitch_deg on the video-relative clock",
    )
    parser.add_argument("--mock-sensors", action="store_true")
    parser.add_argument(
        "--gif", action="store_true", help="Also write a synchronized preview of up to 10 seconds"
    )
    args = parser.parse_args(argv)
    try:
        r = process_clip(
            args.input,
            args.output_dir,
            model=args.model,
            hand=args.hand,
            pinch_range=args.pinch_range,
            calibration=args.calibration,
            pitch_csv=args.pitch_csv,
            mock_sensors=args.mock_sensors,
            gif=args.gif,
            seconds=args.seconds,
            start_second=args.start_second,
        )
        print(
            json.dumps(
                {
                    k: r[k]
                    for k in ["status", "grid_rows", "valid_hand_rows", "coverage", "warnings"]
                },
                indent=2,
            )
        )
        return 0 if r["status"] == "accepted_for_feature_review" else 3
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"ego-vta-clip: {exc}", file=sys.stderr)
        return 2
