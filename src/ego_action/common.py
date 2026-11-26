"""File contracts shared by ingestion, quality checks, and export."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

NS = 1_000_000_000
SIZE = 224
IMU = ("ax", "ay", "az", "gx", "gy", "gz")
QUAT = ("qw", "qx", "qy", "qz")
FMG = tuple(f"fmg{i}" for i in range(6))
XYZ = ("x", "y", "z")
POINTS = tuple(f"p{i}_{axis}" for i in range(21) for axis in XYZ)
GAZE = ("u", "v")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", value):
        raise ValueError(f"Expected a simple identifier, got {value!r}")
    return value


def read_csv(path, fields):
    """Use integer nanoseconds even when timestamps exceed float64 precision."""
    times, values = [], []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = {"timestamp_ns", *fields} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing CSV columns {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            try:
                times.append(int(row["timestamp_ns"]))
                values.append(
                    [float(row[k]) if row[k] not in ("", None) else np.nan for k in fields]
                )
            except (TypeError, ValueError) as e:
                raise ValueError(f"{path}:{line}: invalid timestamp or measurement") from e
    t = np.asarray(times, dtype=np.int64)
    if not len(t) or np.any(t[1:] <= t[:-1]):
        raise ValueError(f"{path}: timestamps must be nonempty, strictly increasing and unique")
    return t, np.array(values, dtype=np.float64)


def runs(mask):
    """Half-open spans of True. A missing input always breaks a trajectory."""
    changes = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))


def unit(values, axis=-1):
    arr = np.asarray(values, dtype=float)
    length = np.linalg.norm(arr, axis=axis, keepdims=True)
    return np.divide(arr, length, out=np.full_like(arr, np.nan), where=length > 1e-8)


def require_range(values, low, high, name):
    if not np.isfinite(values).all() or np.any(
        (np.asarray(values) < low) | (np.asarray(values) > high)
    ):
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")


def media_paths(spec, root):
    paths = {f"camera/{k}": root / v["file"] for k, v in spec["cameras"].items()}
    paths.update({f"sensor/{k}": root / v["file"] for k, v in spec.get("sensors", {}).items()})
    for key in ("landmarks", "camera_pose", "annotations", "tokenizer", "hand_model"):
        if spec.get(key):
            paths[key] = root / spec[key]
    return paths
