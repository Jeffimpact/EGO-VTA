import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from ego_action.common import read_json, write_json
from ego_action.demo import create_demo
from ego_action.labels import caption_review_reasons, vlm_label
from ego_action.monitor import monitor
from ego_action.pipeline import run
from ego_action.staging import fetch_capture, stage_capture


class ExternalInterfaces(unittest.TestCase):
    def test_rig_alert_edges_and_partial_channels(self):
        records = [
            {"timestamp_ns": 1, "pitch_deg": 36},
            {"timestamp_ns": 2, "lux": 10},
            {"timestamp_ns": 3, "lux": 20},
            {"timestamp_ns": 4, "pitch_deg": 25},
        ]
        out = io.StringIO()
        monitor([json.dumps(row) for row in records], out)
        events = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(
            [(e["event"], e["status"]) for e in events],
            [
                ("realign_camera", "active"),
                ("increase_lighting", "active"),
                ("increase_lighting", "cleared"),
                ("realign_camera", "cleared"),
            ],
        )

    def test_staged_capture_round_trip_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            manifest = create_demo(base / "source")
            prefix = (base / "objects").as_uri()
            uri = stage_capture(manifest, prefix)
            self.assertEqual(stage_capture(manifest, prefix), uri)
            fetched = fetch_capture(uri, base / "fetched")
            self.assertEqual(fetch_capture(uri, base / "fetched"), fetched)
            metadata = run(fetched, base / "result", ["hdf5"])
            self.assertEqual(metadata["training_rows"], 76)
            stage = next((base / "objects").rglob("episode.json"))
            bad = stage.parent / read_json(stage)["landmarks"]
            bad.write_bytes(b"broken input")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                fetch_capture(uri, base / "second")
            self.assertFalse((base / "second").exists())

    def test_automatic_boundaries_and_local_vlm_request(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                query = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(query)
                label = {
                    "instruction": "Move the block",
                    "target_object": "block",
                    "activity": "manipulation",
                    "confidence": 0.91,
                }
                data = json.dumps(
                    {"choices": [{"message": {"content": json.dumps(label)}}]}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                original = create_demo(root / "capture")
                cfg = read_json(original)
                cfg.pop("annotations")
                cfg["vlm"] = {
                    "model": "contract-test",
                    "url": f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                }
                manifest = original.parent / "auto.json"
                write_json(manifest, cfg)
                meta = run(manifest, root / "result", ["hdf5"])
                labels = meta["sub_task_segments"]
                self.assertGreater(len(labels), 1)
                self.assertEqual(len(requests), len(labels))
                self.assertEqual(meta["confidence_metrics"]["tier"], 2)
                self.assertTrue(all(s["source"] == "vlm" for s in labels))
                for q in requests:
                    parts = q["messages"][0]["content"]
                    self.assertEqual(len([p for p in parts if p["type"] == "image_url"]), 3)
                    self.assertEqual(q["response_format"]["type"], "json_schema")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_overlay_text_and_low_confidence_require_review(self):
        bad = {
            "instruction": "Synthetic Calibration Episode",
            "activity": "manipulation",
            "target_object": "{",
            "confidence": 0.2,
        }
        self.assertEqual(
            set(caption_review_reasons(bad, {})),
            {"low_model_confidence", "missing_manipulation_verb", "invalid_target_object"},
        )
        task = {
            "instruction": "Crimp the wire",
            "activity": "manipulation",
            "target_object": "wire",
            "confidence": 0.9,
        }
        self.assertFalse(caption_review_reasons(task, {"extra_verbs": ["crimp"]}))

    def test_vlm_requires_local_address_before_sending_frames(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            vlm_label(
                {"url": "https://example.com", "model": "test"}, [np.zeros((8, 8, 3), "u1")], "Move"
            )
