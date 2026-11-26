import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import h5py
import numpy as np
from ego_action.clip import hand_features, process_clip, qc_report, synthetic_streams
from ego_action.pose import HandObservation


def observed_hand():
    local = np.zeros((21, 3), dtype=np.float32)
    local[5, 0], local[17, 0] = 0.04, -0.04
    local[4, 0], local[8, 0] = 0.01, -0.03
    uv = 0.5 + local[:, :2]
    return HandObservation(uv, None, local, 0.95)


class ClipMilestone(unittest.TestCase):
    def test_proxy_units_and_missing_transitions(self):
        obs = observed_hand()
        uv = np.repeat(obs.uv[None], 5, axis=0)
        uv[1, :, 0] += 0.1
        uv[2, 8, 0] = 1.3
        track = {
            "pixels": uv,
            "local_points": np.repeat(obs.local_m[None], 5, axis=0),
            "source_shape": (100, 200),
            "indices": np.arange(5),
        }
        state, action, fields, ratio, valid, transition = hand_features(track, (0, 1), False)
        self.assertEqual(fields[2], "projected_palm_span_in_image_heights")
        np.testing.assert_allclose(ratio, 0.5)
        np.testing.assert_allclose(state[0], [0.5, 0.5, 0.16, 0.5], atol=1e-6)
        np.testing.assert_allclose(action[0], [0.1, 0, 0, 0.5], atol=1e-6)
        np.testing.assert_equal(valid, [True, True, False, True, True])
        np.testing.assert_equal(transition, [True, False, False, True, False])
        self.assertTrue(np.isnan(action[~transition]).all())

    def test_pitch_and_synthetic_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = np.arange(20, dtype="int64") * 50_000_000
            track, valid = {"timestamps": clock}, np.ones(20, dtype=bool)
            clean = qc_report(track, valid, clock)
            self.assertEqual(clean["pitch_evidence"], "unmeasured")
            log = Path(tmp) / "pitch.csv"
            log.write_text(
                "timestamp_ns,pitch_deg\n"
                + "".join(f"{t},{40 if j == 4 else 0}\n" for j, t in enumerate(clock))
            )
            tilted = qc_report(track, valid, clock, log)
            self.assertEqual(
                tilted["warnings"],
                [
                    {
                        "code": "calibrated_pitch_exceeds_35_deg",
                        "start_frame": 4,
                        "end_frame": 4,
                    }
                ],
            )
            with h5py.File(Path(tmp) / "sim.h5", "w") as h5:
                synthetic_streams(h5, clock)
                for stream in ("imu", "fmg"):
                    g = h5["synthetic/" + stream]
                    self.assertEqual(g.attrs["provenance"], "synthetic_sync_test_only")
                    self.assertEqual(g["aligned"].shape, (20, 6))
                    np.testing.assert_equal(g["age_ns"][:], 0)

    def test_clip_export_replay_and_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "desk.mp4"
            with av.open(str(video), "w") as container:
                stream = container.add_stream("libx264", rate=30)
                stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
                for i in range(30):
                    rgb = np.full((64, 96, 3), i * 4, np.uint8)
                    for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            model = root / "fake-model"
            model.write_bytes(b"test detector")
            out = root / "accepted"
            with patch("ego_action.clip.HandTracker") as detector:
                detector.return_value.return_value = observed_hand()
                result = process_clip(video, out, model=model, start_second=0.25, seconds=0.5)
                self.assertEqual(result["valid_hand_rows"], 10)
                self.assertEqual(result["valid_transitions"], 9)
                with h5py.File(out / "episode.h5") as h5:
                    self.assertEqual(h5["images"].shape, (10, 224, 224, 3))
                    self.assertEqual(h5["actions"].shape, (10, 4))
                    np.testing.assert_array_equal(
                        h5["timestamp_ns"][:], 250_000_000 + np.arange(10) * 50_000_000
                    )
                    self.assertTrue(np.isnan(h5["actions"][-1]).all())
                    self.assertNotIn("synthetic", h5)
                with av.open(str(out / "diagnostic.mp4")) as diagnostic:
                    self.assertEqual(sum(1 for _ in diagnostic.decode(video=0)), 10)
                again = process_clip(video, out, model=model, start_second=0.25, seconds=0.5)
                self.assertEqual(again["files"], result["files"])
                detector.return_value.return_value = None
                lost = root / "rejected"
                negative = process_clip(video, lost, model=model)
                self.assertEqual(negative["status"], "rejected")
                self.assertEqual(negative["valid_hand_rows"], 0)
                self.assertFalse((lost / "episode.h5").exists())
                self.assertTrue((lost / "diagnostic.mp4").is_file())
                self.assertEqual(
                    json.loads((lost / "qc.json").read_text())["longest_invalid_run"], 20
                )
