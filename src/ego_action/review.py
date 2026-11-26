"""A portable local inspector with video and 3-D wrist time locked to one cursor."""

from __future__ import annotations

import base64
import csv
import html
import json
import subprocess
import sys
from pathlib import Path

from .common import read_json


def render_review(result):
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise ValueError("Install the dashboard extra to create an interactive inspector") from exc
    result = Path(result)
    meta = read_json(result / "episode.json")
    with (result / "frames.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))

    def number(row, key):
        return float(row[key]) if row[key] else None

    data = {
        key: [number(row, key) for row in rows]
        for key in ("x_m", "y_m", "z_m", "aperture", "speed_m_s")
    }
    data["segment"] = [int(row["segment"]) for row in rows]
    data["usable"] = [bool(int(row["usable"])) for row in rows]
    data["source"] = [int(row["source_frame"]) for row in rows]
    data["problems"] = [
        [
            label
            for key, label in (
                ("spike", "motion spike"),
                ("lighting_bad", "lighting"),
                ("pitch_bad", "rig tilt"),
                ("idle", "idle"),
                ("force_mismatch", "grasp / force mismatch"),
                ("language_bad", "caption needs review"),
            )
            if int(row.get(key, 0))
        ]
        + ([] if int(row["inputs_ok"]) else ["stream gap"])
        + ([] if int(row["visible"]) else ["hand missing"])
        for row in rows
    ]
    data["instructions"] = [s["instruction"] for s in meta["sub_task_segments"]]
    data["fps"] = meta["fps"]
    q = meta["confidence_metrics"]
    replacements = {
        "TITLE": html.escape(meta["episode_id"]),
        "TASK": html.escape(meta["task_metadata"]["high_level_goal"]),
        "SOURCE": "Synthetic test recording" if meta["synthetic"] else "Recorded episode",
        "REF": html.escape(meta["reference"].replace("_", " ")),
        "TIER": str(q["tier"]),
        "SCORE": f"{q['score']:g}",
        "FRAMES": str(meta["total_frames"]),
        "ROWS": str(meta["training_rows"]),
        "REASONS": html.escape(
            ", ".join(x.replace("_", " ") for x in q["reasons"]) or "Passed episode quality gates"
        ),
        "VIDEO": base64.b64encode((result / "review.mp4").read_bytes()).decode(),
        "PLOTLY": get_plotlyjs(),
        "PAYLOAD": json.dumps(data, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e"),
    }
    from string import Template

    return Template(
        (Path(__file__).with_name("inspector.html")).read_text(encoding="utf-8")
    ).substitute(replacements)


def inspect_episode(result, output=None):
    result = Path(result).resolve()
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as f:
            f.write(render_review(result))
        print(output.resolve())
        return 0
    app = Path(__file__).with_name("streamlit_app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--", str(result)])
