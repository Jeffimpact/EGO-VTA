"""Emit rig status changes on a live line-oriented telemetry feed."""

from __future__ import annotations

import json
import math


def monitor(lines, output):
    previous = set()
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        stamp = record["timestamp_ns"]
        if not isinstance(stamp, int):
            raise TypeError("timestamp_ns must be an integer")
        if not {"pitch_deg", "lux"} & record.keys():
            raise ValueError("Provide calibrated pitch_deg or measured lux")
        active = set()
        for name, code in [("pitch_deg", "realign_camera"), ("lux", "increase_lighting")]:
            if name not in record:
                if code in previous:
                    active.add(code)
                continue
            value = float(record[name])
            if not math.isfinite(value) or (name == "lux" and value < 0):
                raise ValueError(f"Invalid {name}")
            if (name == "pitch_deg" and abs(value) > 35) or (name == "lux" and value < 15):
                active.add(code)
        for code in sorted(active ^ previous):
            print(
                json.dumps(
                    {
                        "timestamp_ns": stamp,
                        "event": code,
                        "status": "active" if code in active else "cleared",
                    }
                ),
                file=output,
                flush=True,
            )
        previous = active
