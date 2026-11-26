"""Run the CPU tracker on two public videos and verify the resulting files."""

import argparse
import json
from pathlib import Path

import av
import h5py
import numpy as np
from check_pinch import compare
from ego_action.clip import process_clip
from fetch_egodex import fetch


def run(raw, output, model=None):
    tasks = ["peel_place_sticker", "pour"]
    if any(not (raw / f"sample_{t}.{ext}").is_file() for t in tasks for ext in ("mp4", "hdf5")):
        fetch(raw, tasks)
    positive = output / "sticker"
    negative = output / "pour"
    sticker = process_clip(
        raw / "sample_peel_place_sticker.mp4",
        positive,
        model=model,
        seconds=15,
        mock_sensors=True,
        gif=True,
    )
    if sticker["status"] != "accepted_for_feature_review":
        raise RuntimeError(f"Unexpected sticker rejection: {sticker['warnings']}")
    with h5py.File(positive / "episode.h5") as artifact:
        if artifact["images"].shape != (300, 224, 224, 3) or artifact["actions"].shape != (300, 4):
            raise ValueError("Unexpected image or action shape for the 15-second clip")
        np.testing.assert_array_equal(artifact["timestamp_ns"][:], np.arange(300) * 50_000_000)
        if np.isfinite(artifact["actions"][:][~artifact["valid/action"][:]]).any():
            raise ValueError("An invalid transition contains a finite pseudo-action")
        if artifact["synthetic/fmg"].attrs["provenance"] != "synthetic_sync_test_only":
            raise ValueError("Synthetic telemetry provenance is missing")
    with av.open(str(positive / "diagnostic.mp4")) as review:
        frame_count = sum(1 for _ in review.decode(video=0))
    if frame_count != 300:
        raise ValueError(f"Expected 300 synchronized review frames, got {frame_count}")
    pour = process_clip(raw / "sample_pour.mp4", negative, model=model)
    if pour["status"] != "rejected" or (negative / "episode.h5").exists():
        raise RuntimeError("Expected the whole pour episode to be quarantined")
    if not (negative / "qc.json").is_file() or not (negative / "diagnostic.mp4").is_file():
        raise RuntimeError("The rejected episode is missing its review materials")
    return {
        "sticker": {k: sticker[k] for k in ("grid_rows", "valid_hand_rows", "valid_transitions")},
        "pour": {k: pour[k] for k in ("grid_rows", "valid_hand_rows", "warnings")},
        "aperture_reference": compare(
            positive / "episode.h5", raw / "sample_peel_place_sticker.hdf5"
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/exports/cloud"))
    parser.add_argument("--model", type=Path, help="Optional pre-downloaded hand_landmarker.task")
    args = parser.parse_args()
    print(json.dumps(run(args.raw_dir, args.output_dir, args.model), indent=2))
