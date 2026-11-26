"""Compare the extracted pinch ratio with independent ARKit joint translations."""

import argparse
import json

import h5py
import numpy as np
from scipy.stats import spearmanr


def compare(clip_path, reference_path, min_confidence=0.5):
    with h5py.File(clip_path) as h5, h5py.File(reference_path) as ref:
        side = h5.attrs["hand"].lower()
        names = [
            side + s
            for s in ("ThumbTip", "IndexFingerTip", "IndexFingerKnuckle", "LittleFingerKnuckle")
        ]
        valid = h5["valid/landmarks"][:] & (h5["source_frame_index"][:] >= 0)
        idx = h5["source_frame_index"][:][valid]
        estimate = h5["hand/pinch_ratio"][:][valid]
        pts = [ref["transforms/" + joint][:][idx, :3, 3] for joint in names]
        conf = np.stack([ref["confidences/" + joint][:][idx] for joint in names]).min(0)
        palm = np.linalg.norm(pts[2] - pts[3], axis=1)
        usable = (conf >= min_confidence) & np.isfinite(palm) & (palm > 1e-5)
        ratio = np.linalg.norm(pts[0] - pts[1], axis=1) / np.maximum(palm, 1e-8)
        usable &= np.isfinite(estimate) & np.isfinite(ratio)
        x, y = estimate[usable], ratio[usable]
        corr = (
            float(spearmanr(x, y).statistic)
            if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0
            else None
        )
        return {
            "paired_frames": len(x),
            "min_joint_confidence": min_confidence,
            "spearman_rho": corr,
            "mean_absolute_error_in_palm_widths": float(np.mean(abs(x - y))) if len(x) else None,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip_h5")
    parser.add_argument("original_egodex_hdf5")
    args = parser.parse_args()
    print(json.dumps(compare(args.clip_h5, args.original_egodex_hdf5), indent=2))
