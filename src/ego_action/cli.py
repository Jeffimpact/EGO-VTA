from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .demo import create_demo
from .pipeline import run


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ego-vta", description="Build and inspect a calibrated manipulation episode."
    )
    subs = parser.add_subparsers(dest="command", required=True)
    demo = subs.add_parser("demo", help="Generate a moving-camera episode and run it end to end")
    demo.add_argument("folder", type=Path)
    demo.add_argument("--fault", choices=["none", "occlusion", "force"], default="none")
    build = subs.add_parser("run", help="Process a capture manifest")
    build.add_argument("manifest", type=Path)
    build.add_argument("output", type=Path)
    for sub in (demo, build):
        sub.add_argument(
            "--formats", default="hdf5,mcap", help="Comma-separated subset of hdf5,mcap,lerobot"
        )
    init = subs.add_parser("init-video", help="Create an input manifest for your own recording")
    init.add_argument("video", type=Path)
    init.add_argument("manifest", type=Path)
    init.add_argument("--task", required=True)
    init.add_argument("--model", type=Path, help="MediaPipe hand_landmarker.task")
    init.add_argument("--hand", choices=["Left", "Right"], default="Right")
    ui = subs.add_parser("inspect", help="Open the synchronized video and trajectory inspector")
    ui.add_argument("result", type=Path)
    ui.add_argument(
        "--html", type=Path, help="Save a portable, offline inspector instead of starting Streamlit"
    )
    stage = subs.add_parser("stage", help="Stage a capture under a local folder or S3 prefix")
    stage.add_argument("manifest", type=Path)
    stage.add_argument("prefix")
    fetch = subs.add_parser("fetch", help="Fetch and verify a staged capture")
    fetch.add_argument("uri")
    fetch.add_argument("folder", type=Path)
    subs.add_parser("monitor", help="Read live pitch/lux JSON lines from stdin and emit rig alerts")
    args = parser.parse_args(argv)
    try:
        if args.command == "monitor":
            from .monitor import monitor

            monitor(sys.stdin, sys.stdout)
            return 0
        if args.command == "stage":
            from .staging import stage_capture

            print(stage_capture(args.manifest, args.prefix))
            return 0
        if args.command == "fetch":
            from .staging import fetch_capture

            print(fetch_capture(args.uri, args.folder))
            return 0
        if args.command == "init-video":
            if not args.video.is_file() or (args.model and not args.model.is_file()):
                raise ValueError("Supply an existing video and model file")
            path = args.manifest.resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            spec = {
                "episode_id": "recording_001",
                "task": args.task,
                "reference": "moving_untracked",
                "fps": 20,
                "horizon": 1,
                "hand": args.hand,
                "aperture_range_m": [0.012, 0.095],
                "cameras": {
                    "primary": {
                        "file": os.path.relpath(args.video.resolve(), path.parent),
                        "start_time_ns": 0,
                    }
                },
            }
            if args.model:
                spec["hand_model"] = os.path.relpath(args.model.resolve(), path.parent)
            with path.open("x", encoding="utf-8") as f:
                json.dump(spec, f, indent=2)
            print(path)
            return 0
        if args.command == "inspect":
            from .review import inspect_episode

            return inspect_episode(args.result, args.html)
        if args.command == "demo":
            manifest, out = create_demo(args.folder / "capture", args.fault), args.folder / "result"
        else:
            manifest, out = args.manifest, args.output
        result = run(manifest, out, [s.strip() for s in args.formats.split(",")])
        q = result["confidence_metrics"]
        print(
            json.dumps(
                {
                    "result": str(out.resolve()),
                    "synthetic": result["synthetic"],
                    "frames": result["total_frames"],
                    "training_rows": result["training_rows"],
                    "tier": q["tier"],
                    "score": q["score"],
                    "reasons": q["reasons"],
                    "formats": result["formats"],
                },
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as e:
        print(f"ego-vta: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
