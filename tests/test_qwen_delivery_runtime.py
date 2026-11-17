import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from scripts.production.moss_scale import delivery
from scripts.production.moss_scale.upload_delivery import _assert_public_safe
from scripts.sound.synthesis.backends import QwenBackend, RegistryBackend


class QwenDeliveryRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("staging", "audio", "completed"):
            (self.root / name).mkdir()
        self.dialogue = {
            "dialogue_id": "test-000001",
            "profile": {"tts_backend": "qwen3", "language": "en"},
            "turns": [{"gap_before_ms": 0, "kind": "speech"}],
        }

    def backend(self, device="cuda:1", capability=(8, 0)):
        cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                                     current_device=lambda: 1,
                                     get_device_capability=lambda index: capability)
        torch = types.SimpleNamespace(float32="torch.float32", bfloat16="torch.bfloat16", cuda=cuda)
        with patch.dict(sys.modules, {"torch": torch, "qwen_tts": types.SimpleNamespace(Qwen3TTSModel=object())}), \
             patch.dict(os.environ, {"TTS_DEVICE": device}, clear=True):
            backend = QwenBackend(self.root)
        backend.cache_dir = self.root / "private-path-sentinel"
        backend.private_key = "secret-key-sentinel"
        return backend

    def deliver(self, synth):
        def render(_dialogue, _synth, clean, timeline):
            sf.write(clean, np.full((480, 2), .01, dtype=np.float32), delivery.SAMPLE_RATE)
            timeline.write_text("[]", encoding="utf-8")
            return 480, 0

        def stitch(_room, path, *_args, **_kwargs):
            path.write_bytes(b"temporary background")

        def mix(clean, _bg, *, out_mixed, **_kwargs):
            shutil.copyfile(clean, out_mixed)

        with patch.object(delivery, "background_rooms", return_value=[Path("rm01/babb")]), \
             patch.object(delivery, "render", side_effect=render), \
             patch.object(delivery, "stitch_clips", side_effect=stitch), \
             patch.object(delivery, "mix", side_effect=mix):
            return delivery.deliver(self.dialogue, synth, self.root)

    def test_real_qwen_backend_runtime_is_written_to_receipt_and_index(self):
        backend = self.backend()
        original = copy.deepcopy(self.dialogue)
        receipt = self.deliver(types.SimpleNamespace(backend=backend))
        expected = {"engine": "qwen3", "device": "cuda:1", "dtype": "bfloat16"}
        self.assertEqual(receipt["annotation"]["profile"]["tts_runtime"], expected)
        saved = json.loads((self.root / "completed" / "test-000001.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["annotation"]["profile"]["tts_runtime"], expected)
        _assert_public_safe(saved["annotation"])
        delivery.DeliveryIndex(self.root, target=1)
        annotation = json.loads((self.root / "annotations.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(annotation["profile"]["tts_runtime"], expected)
        self.assertEqual(self.dialogue, original)
        for private in (str(self.root), "private-path-sentinel", "secret-key-sentinel"):
            self.assertNotIn(private, json.dumps(saved))

    def test_registry_wrapper_uses_underlying_real_qwen(self):
        wrapper = object.__new__(RegistryBackend)
        wrapper.backend = self.backend(device="cpu")
        receipt = self.deliver(types.SimpleNamespace(backend=wrapper))
        self.assertEqual(receipt["annotation"]["profile"]["tts_runtime"],
                         {"engine": "qwen3", "device": "cpu", "dtype": "float32"})

    def test_lookalike_non_qwen_and_unknown_are_not_attributed(self):
        fake = types.SimpleNamespace(device="cuda:1", dtype="torch.bfloat16", engine="qwen3")
        for backend in (fake, None):
            with self.subTest(backend=type(backend).__name__):
                self.assertIsNone(delivery._qwen_runtime(types.SimpleNamespace(backend=backend)))
        wrapper = object.__new__(RegistryBackend)
        wrapper.backend = fake
        self.assertIsNone(delivery._qwen_runtime(types.SimpleNamespace(backend=wrapper)))
        incomplete_wrapper = object.__new__(RegistryBackend)
        self.assertIsNone(delivery._qwen_runtime(types.SimpleNamespace(backend=incomplete_wrapper)))
        self.dialogue["profile"]["tts_runtime"] = {"engine": "qwen3", "device": "fake-secret-sentinel"}
        receipt = self.deliver(types.SimpleNamespace(backend=fake))
        self.assertNotIn("tts_runtime", receipt["annotation"]["profile"])
        self.assertNotIn("fake-secret-sentinel", json.dumps(receipt))
        self.assertEqual(self.dialogue["profile"]["tts_runtime"]["device"], "fake-secret-sentinel")

    def test_qwen_runtime_overwrites_stale_script_value(self):
        self.dialogue["profile"]["tts_runtime"] = {"engine": "wrong", "device": "stale"}
        receipt = self.deliver(types.SimpleNamespace(backend=self.backend()))
        self.assertEqual(receipt["annotation"]["profile"]["tts_runtime"],
                         {"engine": "qwen3", "device": "cuda:1", "dtype": "bfloat16"})

    def test_untrusted_or_unknown_device_and_dtype_are_not_published(self):
        for field, value in (("device", "cuda:1/private-path-sentinel"),
                             ("device", "cuda:01"), ("device", "cuda"),
                             ("dtype", "torch.float16"), ("dtype", "secret-key-sentinel")):
            with self.subTest(field=field, value=value):
                backend = self.backend()
                setattr(backend, field, value)
                self.assertIsNone(delivery._qwen_runtime(types.SimpleNamespace(backend=backend)))
        backend = self.backend()
        backend.device = "cuda:1/private-path-sentinel"
        receipt = self.deliver(types.SimpleNamespace(backend=backend))
        self.assertNotIn("tts_runtime", receipt["annotation"]["profile"])
        self.assertNotIn("private-path-sentinel", json.dumps(receipt))

    def test_inconsistent_cpu_dtype_and_failing_dtype_are_ignored(self):
        backend = self.backend(device="cpu")
        backend.dtype = "torch.bfloat16"
        self.assertIsNone(delivery._qwen_runtime(types.SimpleNamespace(backend=backend)))
        class UnreadableDtype:
            def __str__(self):
                raise ValueError("private-key-sentinel")
        backend.dtype = UnreadableDtype()
        self.assertIsNone(delivery._qwen_runtime(types.SimpleNamespace(backend=backend)))

    def test_legacy_pending_receipt_is_recovered_without_runtime_backfill(self):
        ident = self.dialogue["dialogue_id"]
        stage = self.root / "staging" / ident
        stage.mkdir()
        files = (f"{ident}.flac", f"{ident}_babble.flac", f"{ident}.timeline.json")
        sha = {}
        for name in files:
            content = ("legacy " + name).encode("utf-8")
            (stage / name).write_bytes(content)
            sha[name] = hashlib.sha256(content).hexdigest()
        old = {"annotation": copy.deepcopy(self.dialogue), "sha256": sha,
               "frames": 480, "sample_rate": delivery.SAMPLE_RATE, "credited_seconds": .01}
        (stage / "delivery.pending.json").write_text(json.dumps(old), encoding="utf-8")
        with patch.object(delivery, "render", side_effect=AssertionError("must not resynthesize")):
            recovered = delivery.deliver(self.dialogue, types.SimpleNamespace(backend=self.backend()), self.root)
        self.assertEqual(recovered, old)
        self.assertNotIn("tts_runtime", recovered["annotation"]["profile"])
        self.assertEqual(json.loads((self.root / "completed" / f"{ident}.json").read_text(encoding="utf-8")), old)


if __name__ == "__main__":
    unittest.main()
