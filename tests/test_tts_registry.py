import hashlib
import io
import json
import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from scripts.sound.synthesis import backends
from scripts.text import voices


class VoiceRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "voice.wav"
        self.reference.write_bytes(b"sample reference audio")
        self.digest = hashlib.sha256(self.reference.read_bytes()).hexdigest()
        self.registry = self.root / "voices.json"
        self.registry.write_text(json.dumps({
            "schema_version": 1,
            "backends": {"moss_onnx": {
                "assistant": "assistant",
                "voices": [
                    {"id": "assistant", "kind": "preset", "speaker": "Junhao"},
                    {"id": "speaker_one", "kind": "preset", "speaker": "Xiaoyu"},
                    {"id": "speaker_two", "kind": "reference", "reference_audio": "voice.wav",
                     "sha256": self.digest, "license": "CC BY 4.0", "source_url": "https://example.org"},
                ],
            }},
        }), encoding="utf-8")

    def test_registration_resolves_both_kinds_and_keeps_assistant_fixed(self):
        with patch.dict(os.environ, {"VOICE_REGISTRY": str(self.registry), "TTS_BACKEND": "moss_onnx"}, clear=True):
            one = voices.voices_for_dialogue("first", 1)
            two = voices.voices_for_dialogue("second", 1)
            self.assertEqual(one["assistant"], "assistant")
            self.assertEqual(two["assistant"], "assistant")
            self.assertEqual(voices.preset_for_voice("assistant"), "Junhao")
            self.assertIsNone(voices.reference_for_voice("speaker_one"))
            self.assertEqual(Path(voices.reference_for_voice("speaker_two")), self.reference)
            self.assertEqual(voices.reference_voices()["speaker_two"]["sha256"], self.digest)
            with self.assertRaisesRegex(ValueError, "not registered"):
                voices.reference_for_voice("unknown")

    def test_reference_tampering_after_loading_is_rejected(self):
        with patch.dict(os.environ, {"VOICE_REGISTRY": str(self.registry), "TTS_BACKEND": "moss_onnx"}, clear=True):
            voices.reference_for_voice("speaker_two")
            self.reference.write_bytes(b"changed reference")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                voices.reference_for_voice("speaker_two")

    def test_unselected_backend_does_not_require_its_reference(self):
        data = json.loads(self.registry.read_text(encoding="utf-8"))
        data["backends"]["qwen3"] = {
            "assistant": "ryan",
            "voices": [
                {"id": "ryan", "kind": "preset", "speaker": "Ryan"},
                {"id": "remote_voice", "kind": "reference", "reference_audio": "not-downloaded.wav"},
            ],
        }
        self.registry.write_text(json.dumps(data), encoding="utf-8")
        with patch.dict(os.environ, {"VOICE_REGISTRY": str(self.registry), "TTS_BACKEND": "moss_onnx"}, clear=True):
            self.assertEqual(voices.voices_for_dialogue("test", 1)["assistant"], "assistant")
        with patch.dict(os.environ, {"VOICE_REGISTRY": str(self.registry), "TTS_BACKEND": "qwen3"}, clear=True):
            with self.assertRaises(FileNotFoundError):
                voices.voice_registry()

    def test_signed_source_url_is_removed_from_published_metadata(self):
        data = json.loads(self.registry.read_text(encoding="utf-8"))
        ref = data["backends"]["moss_onnx"]["voices"][2]
        ref["source_url"] = "https://user:password@example.org/sample.wav?token=secret#fragment"
        self.registry.write_text(json.dumps(data), encoding="utf-8")
        with patch.dict(os.environ, {"VOICE_REGISTRY": str(self.registry), "TTS_BACKEND": "moss_onnx"}, clear=True):
            self.assertEqual(voices.reference_voices()["speaker_two"]["source_url"],
                             "https://example.org/sample.wav")

    def test_legacy_manifest_and_default_moss_are_preserved(self):
        manifest = self.root / "legacy.json"
        manifest.write_text(json.dumps({"schema_version": 1, "voices": [{
            "id": "legacy", "reference_audio": "voice.wav", "sha256": self.digest,
            "license": "CC BY 4.0", "source_url": "https://example.org/file?token=secret",
        }]}), encoding="utf-8")
        with patch.dict(os.environ, {"VOICE_MANIFEST": str(manifest), "TTS_BACKEND": "moss_onnx"}, clear=True):
            self.assertEqual(voices.voices_for_dialogue("test", 1)["assistant"], "Junhao")
            self.assertEqual(Path(voices.reference_for_voice("legacy")), self.reference)
            self.assertEqual(voices.reference_voices()["legacy"]["source_url"], "https://example.org/file")
            self.assertEqual(voices.preset_for_voice("Zhiming"), "Zhiming")

    def test_registry_does_not_allow_cloned_assistant(self):
        data = json.loads(self.registry.read_text(encoding="utf-8"))
        data["backends"]["moss_onnx"]["assistant"] = "speaker_two"
        self.registry.write_text(json.dumps(data), encoding="utf-8")
        with patch.dict(os.environ, {"VOICE_REGISTRY": str(self.registry), "TTS_BACKEND": "moss_onnx"}, clear=True):
            with self.assertRaisesRegex(ValueError, "assistant must use a preset"):
                voices.voice_registry()

    def test_adapter_resolves_preset_alias(self):
        class Backend:
            def synthesize(self, text, voice, reference_audio, max_seconds):
                return voice, reference_audio
        with patch.dict(os.environ, {"VOICE_REGISTRY": str(self.registry), "TTS_BACKEND": "moss_onnx"}, clear=True):
            adapter = backends.RegistryBackend(Backend())
            self.assertEqual(adapter.synthesize("hello", "assistant", None, 1), ("Junhao", None))

    def test_adapter_rejects_voice_mapping_changes_mid_run(self):
        class Backend:
            def synthesize(self, text, voice, reference_audio, max_seconds):
                return voice
        with patch.dict(os.environ, {"VOICE_REGISTRY": str(self.registry), "TTS_BACKEND": "moss_onnx"}, clear=True):
            adapter = backends.RegistryBackend(Backend())
            data = json.loads(self.registry.read_text(encoding="utf-8"))
            data["backends"]["moss_onnx"]["voices"][0]["speaker"] = "Zhiming"
            self.registry.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed during synthesis"):
                adapter.synthesize("hello", "assistant", None, 1)


class BackendTests(unittest.TestCase):
    def test_production_entry_respects_backend_and_device(self):
        from scripts.production.moss_scale import runner
        seen = []
        def inspect_configuration():
            seen.append((os.environ["TTS_BACKEND"], os.environ["TTS_DEVICE"], os.environ["MOSS_TEXT_LANGUAGE"]))
            return []
        argv = ["scale_moss", "--check", "--tts-backend", "qwen3", "--tts-device", "cuda:0", "--language", "en"]
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", argv), \
             patch.object(runner, "load_env"), patch.object(runner, "check_inputs", side_effect=inspect_configuration):
            runner.main()
        self.assertEqual(seen, [("qwen3", "cuda:0", "en")])

    def test_completed_receipts_cannot_mix_backends_and_legacy_means_moss(self):
        from scripts.production.moss_scale import runner
        with TemporaryDirectory() as folder:
            root = Path(folder)
            completed = root / "completed"
            completed.mkdir()
            (completed / "old.json").write_text(json.dumps({
                "annotation": {"dialogue_id": "old", "profile": {}, "turns": []},
                "credited_seconds": 1,
            }), encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True), patch.object(runner, "load_env"), \
                 patch.object(sys, "argv", ["scale_moss", "--root", folder, "--hours", "0", "--tts-backend", "moss_onnx"]):
                runner.main()
            with patch.dict(os.environ, {}, clear=True), patch.object(runner, "load_env"), \
                 patch.object(sys, "stderr", new_callable=io.StringIO) as stderr, \
                 patch.object(sys, "argv", ["scale_moss", "--root", folder, "--hours", "0", "--tts-backend", "qwen3"]):
                with self.assertRaises(SystemExit):
                    runner.main()
                self.assertIn("separate --root for each TTS backend", stderr.getvalue())

    def test_qwen_english_presets_and_reference_use_separate_models(self):
        calls = []
        class Model:
            def __init__(self, name):
                self.name = name
                config = types.SimpleNamespace(output_sample_rate=24000, decode_upsample_rate=1920)
                self.model = types.SimpleNamespace(speech_tokenizer=types.SimpleNamespace(config=config))
            def create_voice_clone_prompt(self, **kwargs):
                calls.append(("prompt", kwargs))
                return "prompt"
            def generate_voice_clone(self, **kwargs):
                calls.append(("clone", kwargs))
                return [np.array([0.1])], 24000
            def generate_custom_voice(self, **kwargs):
                calls.append(("preset", kwargs))
                return [np.array([0.2])], 24000
        class Factory:
            @staticmethod
            def from_pretrained(name, **kwargs):
                calls.append(("load", name))
                return Model(name)
        torch = types.SimpleNamespace(float32="float32", float16="float16")
        qwen = types.SimpleNamespace(Qwen3TTSModel=Factory)
        with TemporaryDirectory() as folder, patch.dict(sys.modules, {"torch": torch, "qwen_tts": qwen}):
            reference = Path(folder) / "voice.wav"
            reference.write_bytes(b"audio")
            with patch.dict(os.environ, {"TTS_DEVICE": "cpu", "MOSS_TEXT_LANGUAGE": "en"}, clear=True):
                backend = backends.QwenBackend(Path(folder))
                backend.synthesize("hello", "Ryan", None, 4)
                backend.synthesize("hello", "guest", str(reference), 4)
                backend.synthesize("again", "guest", str(reference), 4)
        self.assertEqual([entry[0] for entry in calls].count("load"), 2)
        self.assertEqual([entry[0] for entry in calls].count("prompt"), 1)
        self.assertEqual(next(kwargs["language"] for name, kwargs in calls if name == "preset"), "English")
        self.assertEqual(next(kwargs["language"] for name, kwargs in calls if name == "clone"), "English")
        self.assertTrue(all(kwargs["max_new_tokens"] == 50 for name, kwargs in calls if name in ("preset", "clone")))

    def test_qwen_marks_near_budget_audio_as_truncated(self):
        class Model:
            def __init__(self):
                config = types.SimpleNamespace(output_sample_rate=24000, decode_upsample_rate=1920)
                self.model = types.SimpleNamespace(speech_tokenizer=types.SimpleNamespace(config=config))
            def generate_custom_voice(self, **kwargs):
                return [np.full(96000, 0.1, dtype=np.float32)], 24000
        class Factory:
            @staticmethod
            def from_pretrained(name, **kwargs):
                return Model()
        torch = types.SimpleNamespace(float32="float32", float16="float16")
        qwen = types.SimpleNamespace(Qwen3TTSModel=Factory)
        with patch.dict(sys.modules, {"torch": torch, "qwen_tts": qwen}), \
             patch.dict(os.environ, {"TTS_DEVICE": "cpu"}, clear=True):
            result = backends.QwenBackend(None).synthesize("hello", "Ryan", None, 4)
        self.assertTrue(result.truncated)
        self.assertEqual(result.provenance["max_new_tokens"], 50)

    def test_moss_forwards_cuda_and_rejects_silent_fallback(self):
        class Runtime:
            selected = None
            def __init__(self, **kwargs):
                self.execution_provider = self.selected or kwargs.get("execution_provider", "cpu")
                self.codec_meta = {"codec_config": {"sample_rate": 24000, "downsample_rate": 1920, "channels": 1}}
                self.manifest = {"generation_defaults": {"max_new_frames": 100}}
        runtime = types.SimpleNamespace(OnnxTtsRuntime=Runtime)
        ort = types.SimpleNamespace(__version__="test")
        with patch.dict(sys.modules, {"onnx_tts_runtime": runtime, "onnxruntime": ort}):
            with patch.dict(os.environ, {"TTS_DEVICE": "cuda"}, clear=True):
                instance = backends.MossOnnxBackend(Path("unused.wav"))
                self.assertEqual(instance.runtime.execution_provider, "cuda")
                Runtime.selected = "cpu"
                with self.assertRaisesRegex(RuntimeError, "did not select CUDA"):
                    backends.MossOnnxBackend(Path("unused.wav"))


if __name__ == "__main__":
    unittest.main()
