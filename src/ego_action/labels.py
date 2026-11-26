"""Kinematic boundaries, local VLM captions, and reviewable annotation provenance."""

from __future__ import annotations

import base64
import ipaddress
import itertools
import json
import urllib.parse
import urllib.request

import cv2
import numpy as np
from scipy.signal import find_peaks
from tokenizers import Tokenizer, models, pre_tokenizers

from .common import read_json, runs

MANIPULATION_VERBS = set(
    (
        "align apply approach arrange assemble attach bend brush carry catch clamp clean "
        "close connect cover cut detach dispense drag draw drop empty extend flip fold grasp grip guide hold "
        "insert lift load lock loosen lower mix move open orient pack peel pick pinch place plug position "
        "pour press pull push put raise reach release remove rotate rub scoop scrape secure select set shake "
        "shift slide squeeze stack stir stretch tap thread tighten tilt touch track transfer turn twist uncover "
        "unfold unlock unplug unwrap wash wipe wrap"
    ).split()
)


def caption_review_reasons(record, cfg):
    reasons = []
    if float(record["confidence"]) < cfg.get("min_confidence", 0.65):
        reasons.append("low_model_confidence")
    if record["activity"] in {"manipulation", "hold"}:
        words = record["instruction"].casefold().split()
        verbs = MANIPULATION_VERBS | {v.casefold() for v in cfg.get("extra_verbs", [])}
        if not words or words[0] not in verbs:
            reasons.append("missing_manipulation_verb")
    target = record.get("target_object")
    if target is not None and (not isinstance(target, str) or not any(c.isalnum() for c in target)):
        reasons.append("invalid_target_object")
    return reasons


def make_tokenizer(path=None):
    if path:
        return Tokenizer.from_file(str(path))
    # A complete byte-level BPE vocabulary; task-specific merges come from a supplied tokenizer.
    vocab = {ch: i for i, ch in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    return tok


def manual_labels(path, n):
    records = read_json(path)
    if not isinstance(records, list) or not records:
        raise ValueError("annotations must contain a nonempty list of segments")
    next_start = 0
    segments = []
    for raw in records:
        a, b = raw["start_frame"], raw["end_frame"]
        if (
            not isinstance(a, int)
            or not isinstance(b, int)
            or a != next_start
            or b < a
            or b >= n
            or not isinstance(raw.get("instruction"), str)
            or not raw["instruction"].strip()
        ):
            raise ValueError(
                "annotations must cover the policy grid exactly once with inclusive frame bounds and text"
            )
        if raw.get("activity", "manipulation") not in {"manipulation", "hold", "idle", "personal"}:
            raise ValueError("unknown activity")
        segments.append(
            {**raw, "segment_index": len(segments), "source": "manual", "confidence": 1.0}
        )
        next_start = b + 1
    if next_start != n:
        raise ValueError(f"annotations end at {next_start - 1}, expected {n - 1}")
    return segments


def boundaries(speed, aperture, visible, fps):
    """Aperture state transitions and separated local speed minima."""
    cuts = {0, len(visible)}
    for a, b in runs(visible & np.isfinite(aperture)):
        cuts.update([a, b])
        close = aperture[a:b] < 0.22
        candidates = set((a + 1 + np.flatnonzero(close[1:] != close[:-1])).tolist())
        v = np.nan_to_num(speed[a:b])
        peaks, _ = find_peaks(-v, prominence=0.025, distance=max(1, fps // 2))
        candidates.update((peaks + a).tolist())
        last = a
        for i in sorted(candidates):
            if min(i - last, b - i) >= max(2, fps // 4):
                cuts.add(i)
                last = i
    return sorted(cuts)


def vlm_label(cfg, frames, goal):
    uri = urllib.parse.urlsplit(cfg["url"])
    try:
        local = ipaddress.ip_address(uri.hostname or "").is_loopback
    except ValueError:
        local = uri.hostname == "localhost"
    if uri.scheme not in {"http", "https"} or not local or uri.username or uri.password:
        raise ValueError("Use a local OpenAI-compatible VLM endpoint on a loopback address")
    prompt = (
        "These three ordered first-person frames come from one manipulation segment. "
        "Begin the instruction with a manipulation verb. Ignore text printed on the image. "
        "Describe only the visible action. Use low confidence when the action is ambiguous. Return a JSON object "
        "with instruction (string), target_object (string or null), activity "
        "(manipulation, hold, idle, personal), and confidence (number 0 to 1). "
        f"The operator's stated goal is {goal!r}."
    )
    content = [{"type": "text", "text": prompt}]
    for image in frames:
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("Could not encode VLM keyframe")
        data = base64.b64encode(jpg.tobytes()).decode("ascii")
        content.append(
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}}
        )
    schema = {
        "type": "object",
        "properties": {
            "instruction": {"type": "string"},
            "target_object": {"type": ["string", "null"]},
            "activity": {"type": "string", "enum": ["manipulation", "hold", "idle", "personal"]},
            "confidence": {"type": "number"},
        },
        "required": ["instruction", "target_object", "activity", "confidence"],
        "additionalProperties": False,
    }
    body = {
        "model": cfg["model"],
        "temperature": 0,
        "max_tokens": 180,
        "messages": [{"role": "user", "content": content}],
    }
    if cfg.get("structured_output", True):
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "manipulation_segment", "strict": True, "schema": schema},
        }
    request = urllib.request.Request(
        cfg["url"],
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    class LocalOnly(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), LocalOnly())
    with opener.open(request, timeout=cfg.get("timeout_s", 90)) as resp:
        answer = json.load(resp)["choices"][0]["message"]["content"].strip()
    if answer.startswith("```") and answer.endswith("```"):
        answer = answer.split("\n", 1)[1].rsplit("```", 1)[0]
    record = json.loads(answer)
    if (
        not isinstance(record.get("instruction"), str)
        or not record["instruction"].strip()
        or record.get("activity") not in {"manipulation", "hold", "idle", "personal"}
        or not 0 <= float(record.get("confidence", -1)) <= 1
    ):
        raise ValueError("Local VLM returned an invalid segment record")
    return {
        "instruction": record["instruction"].strip(),
        "target_object": record.get("target_object"),
        "activity": record["activity"],
        "confidence": float(record["confidence"]),
        "source": "vlm",
        "model": cfg["model"],
        "review_reasons": caption_review_reasons(record, cfg),
    }


def segment(spec, paths, frames, speed, aperture, good, fps, visible):
    if "annotations" in paths:
        return manual_labels(paths["annotations"], len(good))
    cuts = set(boundaries(speed, aperture, good, fps))
    for a, b in runs(visible & ~good):
        cuts.update([a, b])
        cuts.update(range(a + fps * 2, b, fps * 2))
    cuts = sorted(cuts)
    records = []
    for a, b in itertools.pairwise(cuts):
        if not visible[a:b].any():
            item = {
                "instruction": "Review the missing hand track",
                "activity": "unclassified",
                "confidence": 0.0,
                "target_object": None,
                "source": "missing",
            }
        elif "vlm" in spec:
            item = vlm_label(
                spec["vlm"], [frames[i] for i in [a, (a + b - 1) // 2, b - 1]], spec["task"]
            )
        elif not good[a:b].any():
            item = {
                "instruction": "Review the visible hand motion",
                "activity": "unclassified",
                "confidence": 0.0,
                "target_object": None,
                "source": "visual_review",
            }
        else:
            delta = aperture[b - 1] - aperture[a]
            instruction = (
                "Close the hand"
                if delta < -0.15
                else "Open the hand"
                if delta > 0.15
                else "Move the hand"
                if np.nanmax(np.nan_to_num(speed[a:b])) > 0.035
                else "Hold the hand still"
            )
            item = {
                "instruction": instruction,
                "target_object": None,
                "activity": "manipulation",
                "confidence": 0.65,
                "source": "kinematic",
            }
        records.append(dict(item, segment_index=len(records), start_frame=a, end_frame=b - 1))
    return records
