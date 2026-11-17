import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import os
from scripts.production.moss_scale.model_lock import bind_models


class ProductionModelLockTests(unittest.TestCase):
    def test_hard_linked_lock_is_rejected_without_rewriting_target(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            outside = root/'outside.json'
            outside.write_text('{}')
            os.link(outside, root/'model-lock.json')
            identity = {"preset": {"algorithm": "qwen3-snapshot-sha256-v1", "sha256": "a"*64, "tts_model_type": "custom_voice"}}
            with self.assertRaisesRegex(ValueError, "Linked"):
                bind_models(root, identity)
            self.assertEqual(outside.read_text(), '{}')

    def test_unfinished_staging_and_duplicates_prevent_first_binding(self):
        for name in ('staging', 'duplicates'):
            with self.subTest(name=name), TemporaryDirectory() as folder:
                root = Path(folder)
                (root/name).mkdir()
                (root/name/'old.bin').write_bytes(b'old')
                identity = {"preset": {"algorithm": "qwen3-snapshot-sha256-v1", "sha256": "a"*64, "tts_model_type": "custom_voice"}}
                with self.assertRaisesRegex(ValueError, "no proven"):
                    bind_models(root, identity)
                self.assertFalse((root/'model-lock.json').exists())

    def test_first_run_resume_and_model_change(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            identity = {"preset": {"algorithm": "qwen3-snapshot-sha256-v1", "sha256": "a"*64,
                                   "tts_model_type": "custom_voice", "path": "private-sentinel"}}
            first = bind_models(root, identity)
            self.assertEqual(first, bind_models(root, identity))
            self.assertNotIn("private-sentinel", (root/"model-lock.json").read_text())
            original = (root/"model-lock.json").read_bytes()
            identity["preset"]["sha256"] = "b"*64
            with self.assertRaisesRegex(ValueError, "changed"):
                bind_models(root, identity)
            self.assertEqual((root/"model-lock.json").read_bytes(), original)

    def test_old_audio_does_not_get_backfilled(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"audio").mkdir()
            (root/"audio"/"old.flac").write_bytes(b"old")
            identity = {"preset": {"algorithm": "qwen3-snapshot-sha256-v1", "sha256": "a"*64,
                                   "tts_model_type": "custom_voice"}}
            with self.assertRaisesRegex(ValueError, "no proven"):
                bind_models(root, identity)
            self.assertFalse((root/"model-lock.json").exists())
