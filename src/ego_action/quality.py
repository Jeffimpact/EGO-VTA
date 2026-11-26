"""Numerical gates applied before smoothing or serializing training labels."""

from __future__ import annotations

import numpy as np

from .common import NS, runs


def derivatives(position, timestamps, valid):
    velocity = np.full_like(position, np.nan)
    acceleration = np.full(len(valid), np.nan)
    for a, b in runs(valid):
        if b - a < 2:
            continue
        dt = np.diff(timestamps[a:b]) / NS
        vel = np.diff(position[a:b], axis=0) / dt[:, None]
        velocity[a + 1 : b] = vel
        if len(vel) > 1:
            acceleration[a + 2 : b] = np.linalg.norm(np.diff(vel, axis=0), axis=1) / (
                (dt[1:] + dt[:-1]) / 2
            )
    return np.linalg.norm(velocity, axis=1), acceleration


def assess(
    spec,
    visible,
    metric_valid,
    inputs_ok,
    brightness,
    speed,
    accel,
    aperture,
    sensors,
    labels,
    horizon,
):
    fps = spec.get("fps", 20)
    n = len(visible)
    reasons = []
    metrics = {"fov": 100.0 * float(np.mean(visible))}
    spike = (speed > 1.8) | (accel > 15.0)
    metrics["smoothness"] = (
        100.0 * (1 - np.mean(spike[metric_valid])) if metric_valid.any() else 0.0
    )
    poor_light = (brightness[:, 0] < 18) | (brightness[:, 1] > 0.98)
    pitch = np.zeros(n, bool)
    rig = sensors.get("rig")
    rig_checked = False
    if rig is not None:
        drift, lux = rig.T
        pitch = abs(drift) > 35.0
        poor_light |= lux < 15.0
        rig_checked = bool(np.isfinite(rig).all())
    if pitch.any():
        reasons.append("rig_pitch_exceeds_35_deg")
    if poor_light.mean() > 0.05:
        reasons.append("underexposed_or_clipped")
    fov_runs = [[a, b - 1] for a, b in runs(~visible) if (b - a) / fps > 0.25]
    if (~visible).mean() > 0.05:
        reasons.append("hand_missing_over_5_percent")

    idle = np.zeros(n, bool)
    language_bad = np.zeros(n, bool)
    hold = np.zeros(n, bool)
    label_score, verified_labels = 0.0, True
    for record in labels:
        a, b = record["start_frame"], record["end_frame"] + 1
        if record.get("review_reasons"):
            language_bad[a:b] = True
        else:
            label_score += (b - a) * record["confidence"]
        verified_labels &= record["source"] == "manual"
        if record.get("activity") in {"idle", "personal"}:
            idle[a:b] = True
        if record.get("activity") == "hold":
            hold[a:b] = True
    for a, b in runs((speed < 0.008) & metric_valid & ~hold):
        if b - a > 3 * fps:
            idle[a:b] = True
    metrics["language"] = 100.0 * label_score / n
    fmg = sensors.get("fmg")
    mismatch = np.zeros(n, bool)
    fm_checked = False
    force_correlation = None
    if fmg is not None:
        fvalid = np.isfinite(fmg).all(axis=1)
        closed = fvalid & (aperture <= 0.10) & metric_valid
        fm_checked = bool(closed.any())
        mismatch = closed & np.all(fmg <= 0.02, axis=1)
        paired = fvalid & metric_valid & np.isfinite(aperture)
        if paired.sum() >= 3:
            closure, pressure = 1 - aperture[paired], fmg[paired].mean(axis=1)
            if np.ptp(closure) > 0.1 and np.ptp(pressure) > 0.02:
                force_correlation = float(np.corrcoef(closure, pressure)[0, 1])
    metrics["fmg"] = 100.0 * (1 - mismatch.sum() / max(1, closed.sum())) if fm_checked else 0.0
    if force_correlation is not None:
        metrics["fmg"] *= float(np.clip(force_correlation, 0, 1))
    if mismatch.any():
        reasons.append("closed_hand_without_fmg_response")
    if (~inputs_ok).mean() > 0.05:
        reasons.append("required_stream_or_clock_loss_over_5_percent")
    if not metric_valid.any():
        reasons.append("metric_wrist_unavailable")
    if spec["reference"] == "moving_untracked":
        reasons.append("moving_camera_without_reference_pose")
    if language_bad.mean() > 0.05:
        reasons.append("vlm_instruction_requires_review")
    valid = (
        metric_valid
        & inputs_ok
        & visible
        & ~spike
        & ~poor_light
        & ~idle
        & ~pitch
        & ~mismatch
        & ~language_bad
    )
    if not any(b - a > horizon for a, b in runs(valid)):
        reasons.append("no_contiguous_action_window")
    score = sum(
        w * metrics[k]
        for k, w in {"fov": 0.35, "smoothness": 0.25, "language": 0.20, "fmg": 0.20}.items()
    )
    if score < 75:
        reasons.append("weighted_score_below_75")
    high_evidence = bool(
        visible.all()
        and inputs_ok.all()
        and metric_valid.all()
        and not spike.any()
        and fm_checked
        and force_correlation is not None
        and force_correlation >= 0.5
        and rig_checked
        and verified_labels
        and spec.get("landmarks")
        and spec.get("landmark_quality") == "validated_metric"
    )
    tier = 3 if reasons else (1 if score >= 90 and high_evidence else 2)
    masks = {
        "visible": visible,
        "metric_valid": metric_valid,
        "inputs_ok": inputs_ok,
        "spike": spike,
        "lighting_bad": poor_light,
        "pitch_bad": pitch,
        "idle": idle,
        "force_mismatch": mismatch,
        "language_bad": language_bad,
        "usable": valid,
    }
    report = {
        "tier": tier,
        "score": round(float(score), 4),
        "components": {k: round(float(v), 4) for k, v in metrics.items()},
        "reasons": reasons,
        "long_fov_losses_inclusive": fov_runs,
        "frame_counts": {k: int(v.sum()) for k, v in masks.items()},
        "fmg_correlation_assessed": force_correlation is not None,
        "closure_fmg_pearson": force_correlation,
        "pitch_and_lux_assessed_all_frames": rig_checked,
        "language_human_verified": bool(verified_labels),
        "tier1_evidence_complete": high_evidence,
        "lighting_proxy": "8-bit luma 95th percentile <18 or >98% pixels clipped; physical lux only from rig sensor",
    }
    return report, masks
