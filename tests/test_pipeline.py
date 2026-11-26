import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from ego_action.common import QUAT, XYZ, read_csv, read_json, write_json
from ego_action.demo import EPOCH, PALM, K, create_demo, sample
from ego_action.ingest import align, nearest_indices
from ego_action.pipeline import run
from ego_action.pose import action_chunks, recover_camera_points
from ego_action.quality import assess
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation


class NumericContracts(unittest.TestCase):
    def test_backward_clock_no_future_or_stale_measurement(self):
        actual = EPOCH + np.array([0, 10, 40], dtype=np.int64) * 1_000_000
        targets = EPOCH + np.array([-1, 9, 25, 26, 39, 40], dtype=np.int64) * 1_000_000
        out, age = align(actual, np.arange(3.0)[:, None], targets)
        np.testing.assert_equal(out[:, 0], [np.nan, 0, 1, np.nan, np.nan, 2])
        self.assertEqual(age[2], 15_000_000)
        self.assertEqual(age[3], -1)

    def test_offline_slerp_and_strict_future_bound(self):
        q = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, -1]], float)
        t = np.array([0, 20, 80]) * 1_000_000
        vals, _ = align(
            t, q, np.array([10, 30, 80]) * 1_000_000, mode="interpolate", kind="quaternion"
        )
        np.testing.assert_allclose(abs(vals[0]), [2**-0.5, 0, 0, 2**-0.5])
        self.assertTrue(np.isnan(vals[1]).all())
        self.assertAlmostEqual(np.linalg.norm(vals[2]), 1)

    def test_cubic_interpolation_is_bounded(self):
        t = np.arange(4) * 10_000_000
        vals, _ = align(
            t,
            (np.arange(4.0) ** 3)[:, None],
            np.array([15_000_000, 60_000_000]),
            mode="interpolate",
            kind="cubic",
        )
        self.assertAlmostEqual(vals[0, 0], 1.5**3)
        self.assertTrue(np.isnan(vals[1, 0]))

    def test_body_frame_delta_and_end_of_episode(self):
        angles = [90, 100, 110, 120]
        q = Rotation.from_rotvec(np.deg2rad(angles)[:, None] * [[0, 0, 1]]).as_quat()[
            :, [3, 0, 1, 2]
        ]
        poses = np.column_stack([np.zeros(4), np.arange(4), np.zeros(4), q])
        idx, chunk = action_chunks(poses, np.array([1.0, 0.8, 0.5, 0.0]), 0, 4, 2)
        self.assertEqual(idx.tolist(), [0, 1])
        np.testing.assert_allclose(chunk[0, :, :3], [[1, 0, 0], [2, 0, 0]], atol=1e-6)
        np.testing.assert_allclose(chunk[0, :, 5], np.deg2rad([10, 20]), atol=1e-6)
        np.testing.assert_allclose(chunk[0, :, 6], [0.8, 0.5])

    def test_metric_pnp_recovers_camera_translation(self):
        actual, _, _ = sample(0.8)
        centred = actual - actual.mean(axis=0)
        uvw = actual @ K.T
        uv = uvw[:, :2] / uvw[:, 2:]
        got = recover_camera_points(centred, uv, K, PALM, 0.1)
        np.testing.assert_allclose(got, actual, atol=1e-5)
        noisy = uv.copy()
        noisy[8] += [80, -80]
        self.assertIsNone(recover_camera_points(centred, noisy, K, PALM, 1.0))

    def test_nearest_retains_integer_precision(self):
        idx = nearest_indices(np.array([EPOCH, EPOCH + 4]), np.array([EPOCH + 2, EPOCH + 3]), 2)
        self.assertEqual(idx.tolist(), [0, 1])

    def test_idle_rejection_pitch_lux_and_spike(self):
        n = 90
        spec = {"reference": "static_camera", "fps": 20}
        good = np.ones(n, bool)
        lum = np.tile([115.0, 0.03], (n, 1))
        speed, acc = np.ones(n) * 0.02, np.zeros(n)
        speed[:62] = 0
        speed[69], acc[70] = 2.0, 16.0
        rig = np.tile([0.0, 10.0], (n, 1))
        rig[-1, 0] = 36.0
        labels = [
            {
                "start_frame": 0,
                "end_frame": n - 1,
                "instruction": "Move the object",
                "source": "manual",
                "confidence": 1.0,
            }
        ]
        report, masks = assess(
            spec, good, good, good, lum, speed, acc, np.ones(n), {"rig": rig}, labels, 1
        )
        self.assertEqual(report["tier"], 3)
        self.assertIn("rig_pitch_exceeds_35_deg", report["reasons"])
        self.assertIn("underexposed_or_clipped", report["reasons"])
        self.assertEqual(int(masks["spike"].sum()), 2)
        self.assertEqual(int(masks["idle"].sum()), 62)


class EpisodeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.manifest = create_demo(cls.root / "capture")
        cls.out = cls.root / "result"
        cls.meta = run(cls.manifest, cls.out)
        cls.h5 = next(cls.out.rglob("episodes.h5"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_hdf5_against_independent_world_trajectory(self):
        t, ground = read_csv(self.manifest.parent / "ground_truth.csv", XYZ + QUAT + ("aperture",))
        with (self.out / "frames.csv").open() as f:
            stamps = np.array([int(row["exposure_ns"]) for row in csv.DictReader(f)])
        oracle = ground[np.searchsorted(t, stamps)]
        with h5py.File(self.h5) as f:
            ep = f["data/episode_0000"]
            self.assertEqual(ep["observation/image_primary"].shape, (76, 3, 224, 224))
            self.assertEqual(ep["action/chunk"].shape, (76, 4, 7))
            self.assertEqual(ep["sequence/image_primary"].shape, (80, 3, 224, 224))
            np.testing.assert_equal(ep["target/future_observation_index"][-1], [76, 77, 78, 79])
            np.testing.assert_equal(
                ep["observation/image_primary"][12], ep["sequence/image_primary"][12]
            )
            self.assertEqual(ep["annotation/task_tokens"].dtype, np.dtype("int64"))
            base = Rotation.from_quat(oracle[:76, 3:7][:, [1, 2, 3, 0]])
            for step in range(1, 5):
                future = oracle[step : step + 76]
                chunk = ep["action/chunk"][:, step - 1]
                predicted_xyz = oracle[:76, :3] + base.apply(chunk[:, :3])
                np.testing.assert_allclose(predicted_xyz, future[:, :3], atol=1e-6)
                predicted_r = base * Rotation.from_rotvec(chunk[:, 3:6])
                expected_r = Rotation.from_quat(future[:, 3:7][:, [1, 2, 3, 0]])
                self.assertLess(np.max((predicted_r.inv() * expected_r).magnitude()), 1e-5)
                np.testing.assert_allclose(chunk[:, 6], future[:, 7], atol=1e-6)
            np.testing.assert_equal(ep["source/timestamp_ns"][:], stamps[:76])
        self.assertTrue(self.meta["synthetic"])
        self.assertEqual(self.meta["confidence_metrics"]["tier"], 1)

    def test_mcap_cdr_and_hdf5_agree(self):
        p = next(self.out.rglob("*.mcap"))
        with p.open("rb") as file, h5py.File(self.h5) as f:
            ep = f["data/episode_0000"]
            msgs = list(
                make_reader(file, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                    topics=["/ego/action_chunk"]
                )
            )
            self.assertEqual(len(msgs), 76)
            _, ch, m, data = msgs[37]
            self.assertEqual(ch.message_encoding, "cdr")
            self.assertEqual(m.log_time, ep["source/timestamp_ns"][37])
            np.testing.assert_allclose(np.array(data.data).reshape(4, 7), ep["action/chunk"][37])
        with p.open("rb") as file:
            s, _, _, data = next(
                make_reader(file, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                    topics=["/ego/primary/image"]
                )
            )
            self.assertEqual(
                (s.name, data.encoding, data.width, data.height),
                ("sensor_msgs/msg/Image", "rgb8", 224, 224),
            )

    def test_quarantine_overrides_score_and_keeps_capture(self):
        for fault, reason in [
            ("occlusion", "hand_missing_over_5_percent"),
            ("force", "closed_hand_without_fmg_response"),
        ]:
            manifest = create_demo(self.root / fault, fault)
            dest = self.root / (fault + "-result")
            data = run(manifest, dest)
            self.assertGreaterEqual(data["confidence_metrics"]["score"], 75)
            self.assertIn(reason, data["confidence_metrics"]["reasons"])
            self.assertEqual(data["training_rows"], 0)
            self.assertFalse(list(dest.rglob("*.h5")))
            self.assertTrue((manifest.parent / "primary.mp4").is_file())

    def test_untracked_camera_and_sensorless_review(self):
        spec = read_json(self.manifest)
        spec["reference"] = "moving_untracked"
        for key in ["sensors", "landmarks", "camera_pose", "annotations"]:
            spec.pop(key)
        p = self.manifest.parent / "video_only.json"
        write_json(p, spec)
        meta = run(p, self.root / "video-only")
        self.assertEqual(meta["training_rows"], 0)
        self.assertIn("moving_camera_without_reference_pose", meta["confidence_metrics"]["reasons"])
        self.assertIn("metric_wrist_unavailable", meta["confidence_metrics"]["reasons"])

    def test_short_loss_splits_future_action_windows(self):
        spec = read_json(self.manifest)
        src = self.manifest.parent
        with (src / "landmarks.csv").open() as f:
            rows = list(csv.reader(f))
        # Frame 45 is on both the source and target grids (1.50 seconds).
        rows[46][1:] = [""] * 63
        with (src / "one_gap.csv").open("w", newline="") as f:
            csv.writer(f).writerows(rows)
        spec["landmarks"] = "one_gap.csv"
        p = src / "one_gap.json"
        write_json(p, spec)
        dest = self.root / "one-gap"
        data = run(p, dest, ["hdf5"])
        self.assertEqual(data["confidence_metrics"]["tier"], 2)
        with h5py.File(next(dest.rglob("*.h5"))) as f:
            self.assertEqual(len(f["data"]), 2)
            for ep in f["data"].values():
                anchors = ep["source/grid_index"][:]
                self.assertTrue(np.all((anchors + 4 < 30) | (anchors > 30)))
                self.assertTrue(np.all(np.diff(anchors) == 1))

    def test_idempotence_conflicting_recipe_and_failed_retry(self):
        self.assertEqual(run(self.manifest, self.out), self.meta)
        with self.assertRaises(FileExistsError):
            run(self.manifest, self.out, ["hdf5"])
        spec = read_json(self.manifest)
        spec["annotations"] = "bad_labels.json"
        write_json(self.manifest.parent / spec["annotations"], [])
        p = self.manifest.parent / "bad_input.json"
        write_json(p, spec)
        out = self.root / "failed-run"
        with self.assertRaises(ValueError):
            run(p, out)
        self.assertFalse(out.exists())
        write_json(
            self.manifest.parent / spec["annotations"],
            read_json(self.manifest.parent / "annotations.json"),
        )
        self.assertEqual(run(p, out)["training_rows"], 76)

    @unittest.skipUnless(
        importlib.util.find_spec("lerobot"), "install the LeRobot extra to test its native reader"
    )
    def test_native_lerobot_reader_decodes_camera_and_actions(self):
        from lerobot.datasets import LeRobotDataset

        dest = self.root / "lerobot-check"
        run(self.manifest, dest, ["hdf5", "lerobot"])
        dataset = LeRobotDataset(
            "local/ego_action", root=next(dest.rglob("lerobot")), video_backend="pyav"
        )
        self.assertEqual(len(dataset), 76)
        with h5py.File(self.h5) as f:
            ep = f["data/episode_0000"]
            for i in [0, 37, 75]:
                row = dataset[i]
                self.assertEqual(tuple(row["observation.images.primary"].shape), (3, 224, 224))
                np.testing.assert_allclose(row["action"], ep["action/chunk"][i, 0])
                np.testing.assert_allclose(row["action_chunk"], ep["action/chunk"][i])
                self.assertEqual(
                    row["capture_timestamp_ns"].item(), int(ep["source/timestamp_ns"][i])
                )


if __name__ == "__main__":
    unittest.main()
