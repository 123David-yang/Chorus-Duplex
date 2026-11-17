import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from scripts.production.moss_scale.upload_delivery import _matches_remote, validate_delivery


class DeliveryUploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "completed").mkdir()
        (self.root / "audio").mkdir()
        self.ident = "case-001"
        annotation = {
            "dialogue_id": self.ident,
            "participants": [],
            "turns": [{"turn_id": 0, "speaker": "assistant", "kind": "speech",
                       "text": "Hello", "addressee": "user_01", "gap_before_ms": 0}],
        }
        names = (f"{self.ident}.flac", f"{self.ident}_babble.flac",
                 f"{self.ident}.timeline.json")
        for name in names[:2]:
            sf.write(self.root / "audio" / name, np.full((480, 2), 0.01), 48000)
        timeline = [{"turn_id": 0, "speaker": "assistant", "kind": "speech",
                     "text": "Hello", "addressee": "user_01", "start_sec": 0.0, "end_sec": 0.01}]
        (self.root / "audio" / names[2]).write_text(json.dumps(timeline), encoding="utf8")
        hashes = {}
        for name in names:
            with (self.root / "audio" / name).open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        receipt = {"annotation": annotation, "frames": 480, "sample_rate": 48000, "sha256": hashes}
        (self.root / "completed" / f"{self.ident}.json").write_text(json.dumps(receipt), encoding="utf8")
        self.audit = patch("scripts.text.quality.audit", return_value={"passed": True})
        self.audit.start()

    def tearDown(self):
        self.audit.stop()
        self.tmp.cleanup()

    def test_valid_delivery_and_tampered_audio(self):
        annotation, files = validate_delivery(self.root, self.ident)
        self.assertEqual(annotation["dialogue_id"], self.ident)
        self.assertEqual(len(files), 3)
        frozen = files[f"audio/{self.ident}.flac"]
        self.assertIsInstance(frozen, bytes)
        with (self.root / "audio" / f"{self.ident}.flac").open("ab") as stream:
            stream.write(b"modified")
        self.assertEqual(files[f"audio/{self.ident}.flac"], frozen)
        with self.assertRaisesRegex(ValueError, "modified"):
            validate_delivery(self.root, self.ident)

    def test_rejects_sensitive_metadata_and_signed_urls(self):
        from scripts.production.moss_scale.upload_delivery import _assert_public_safe
        with self.assertRaisesRegex(ValueError, "Sensitive field"):
            _assert_public_safe({"profile": {"api_key": "example"}})
        with self.assertRaisesRegex(ValueError, "private components"):
            _assert_public_safe({"source_url": "HTTPS://example.invalid/file?signature=example"})
        with self.assertRaisesRegex(ValueError, "Sensitive field"):
            _assert_public_safe({"profile": {"token": "example"}})
        with self.assertRaisesRegex(ValueError, "Sensitive field"):
            _assert_public_safe({"profile": {"openai_api_key": "example"}})
        _assert_public_safe({"source_url": "https://example.invalid/file"})

    def test_rejects_sensitive_timeline(self):
        name = f"{self.ident}.timeline.json"
        path = self.root / "audio" / name
        timeline = json.loads(path.read_text(encoding="utf8"))
        timeline[0]["api_key"] = "example"
        path.write_text(json.dumps(timeline), encoding="utf8")
        receipt_path = self.root / "completed" / f"{self.ident}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf8"))
        receipt["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt_path.write_text(json.dumps(receipt), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "Sensitive field"):
            validate_delivery(self.root, self.ident)

    def test_rejects_mismatched_annotation_turn_id(self):
        path = self.root / "completed" / f"{self.ident}.json"
        receipt = json.loads(path.read_text(encoding="utf8"))
        receipt["annotation"]["turns"][0]["turn_id"] = 999
        path.write_text(json.dumps(receipt), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_delivery(self.root, self.ident)

    def test_rejects_mismatched_timeline_addressee(self):
        name = f"{self.ident}.timeline.json"
        path = self.root / "audio" / name
        timeline = json.loads(path.read_text(encoding="utf8"))
        timeline[0]["addressee"] = "user_02"
        path.write_text(json.dumps(timeline), encoding="utf8")
        receipt_path = self.root / "completed" / f"{self.ident}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf8"))
        receipt["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt_path.write_text(json.dumps(receipt), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_delivery(self.root, self.ident)

    def test_rejects_path_traversal(self):
        with self.assertRaisesRegex(ValueError, "Invalid dialogue ID"):
            validate_delivery(self.root, "../case-001")

    def test_python_api_redacts_network_exception(self):
        from scripts.production.moss_scale.upload_delivery import publish
        with patch("scripts.production.moss_scale.upload_delivery._publish",
                   side_effect=RuntimeError("https://example.invalid/?signature=private")):
            with self.assertRaises(RuntimeError) as cm:
                publish(self.root, "example/dataset", [self.ident], execute=True)
        self.assertNotIn("signature", str(cm.exception))
        self.assertNotIn("example.invalid", str(cm.exception))

    def test_remote_git_and_lfs_hashes(self):
        content = b"audio bytes"
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        self.assertTrue(_matches_remote(SimpleNamespace(lfs=None, blob_id=blob), content))
        self.assertTrue(_matches_remote(SimpleNamespace(lfs={"sha256": hashlib.sha256(content).hexdigest()}), content))
        self.assertFalse(_matches_remote(SimpleNamespace(lfs={"sha256": "wrong"}), content))


if __name__ == "__main__":
    unittest.main()
