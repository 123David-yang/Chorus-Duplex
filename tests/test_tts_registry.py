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
import soundfile as sf

from scripts.sound.synthesis import backends
from scripts.text import voices


class VoiceRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "voice.wav"
        sf.write(self.reference, np.zeros(4800, dtype=np.float32), 24000)
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

    def test_production_cast_uses_distinct_registered_voices(self):
        from scripts.production.moss_scale.script import distinct_voice_cast
        with patch.dict(os.environ, {"VOICE_REGISTRY": str(self.registry), "TTS_BACKEND": "moss_onnx"}, clear=True):
            count, mapping, third = distinct_voice_cast("case", 6, True)
            self.assertEqual(count, 1)
            self.assertTrue(third)
            self.assertEqual(mapping["assistant"], "assistant")
            self.assertEqual(len(set(mapping.values())), len(mapping))
            self.assertEqual(set(mapping.values()), {"assistant", "speaker_one", "speaker_two"})

    def test_single_participant_voice_works_without_third_party(self):
        from scripts.production.moss_scale.script import distinct_voice_cast
        with patch.dict(os.environ, {"MOSS_VOICE_POOL": "Zhiming", "TTS_BACKEND": "moss_onnx"}, clear=True):
            count, mapping, third = distinct_voice_cast("single", 1, False)
            self.assertEqual((count, third), (1, False))
            self.assertEqual(mapping, {"assistant": "Junhao", "user_01": "Zhiming"})
            count, mapping, third = distinct_voice_cast("single", 1, True)
            self.assertEqual((count, third), (1, False))

    def test_reference_tampering_after_loading_is_rejected(self):
        with patch.dict(os.environ, {"VOICE_REGISTRY": str(self.registry), "TTS_BACKEND": "moss_onnx"}, clear=True):
            voices.reference_for_voice("speaker_two")
            self.reference.write_bytes(b"changed reference")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                voices.reference_for_voice("speaker_two")

    def test_invalid_reference_audio_is_rejected_before_synthesis(self):
        data = json.loads(self.registry.read_text(encoding="utf-8"))
        data["backends"]["moss_onnx"]["voices"][2].pop("sha256")
        self.registry.write_text(json.dumps(data), encoding="utf-8")
        self.reference.write_bytes(b"not a valid audio file")
        with patch.dict(os.environ, {"VOICE_REGISTRY": str(self.registry), "TTS_BACKEND": "moss_onnx"}, clear=True):
            with self.assertRaisesRegex(ValueError, "cannot be fully decoded"):
                voices.voice_registry()

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

    def test_legacy_reference_named_with_kokoro_prefix_is_not_lost_for_other_backends(self):
        manifest = self.root / "legacy.json"
        manifest.write_text(json.dumps({"schema_version": 1, "voices": [{
            "id": "kokoro_reference", "reference_audio": "voice.wav", "sha256": self.digest,
            "license": "CC BY 4.0", "source_url": "https://example.org/file", "gender": "女声",
        }]}), encoding="utf-8")
        for backend in ("moss_onnx", "qwen3"):
            with self.subTest(backend=backend), patch.dict(os.environ, {"VOICE_MANIFEST": str(manifest), "TTS_BACKEND": backend}, clear=True):
                self.assertEqual(Path(voices.reference_for_voice("kokoro_reference")), self.reference)
                self.assertEqual(voices.gender_of("kokoro_reference"), "女声")
        with patch.dict(os.environ, {"TTS_BACKEND": "kokoro"}, clear=True):
            self.assertIsNone(voices.reference_for_voice("kokoro_003"))
            self.assertEqual(voices.gender_of("kokoro_003"), "女声")

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
    def test_qwen_offline_model_id_is_resolved_without_network(self):
        calls = []
        class Factory:
            @staticmethod
            def from_pretrained(name, **kwargs):
                calls.append(name)
                return object()
        torch = types.SimpleNamespace(float32="float32")
        with TemporaryDirectory() as folder, \
             patch.dict(sys.modules, {"torch": torch, "qwen_tts": types.SimpleNamespace(Qwen3TTSModel=Factory)}), \
             patch.dict(os.environ, {"TTS_DEVICE": "cpu", "HF_HUB_OFFLINE": "1"}, clear=True), \
             patch("huggingface_hub.snapshot_download", return_value=folder) as download:
            backend = backends.QwenBackend(None)
            backend._model("preset")
            self.assertEqual(calls, [folder])
            self.assertTrue(download.call_args.kwargs["local_files_only"])
            backend._model("preset")
            self.assertEqual(download.call_count, 1)
            backend.models.clear()
            download.side_effect = FileNotFoundError("private-cache-path-sentinel")
            with self.assertRaises(RuntimeError) as failure:
                backend._model("preset")
            self.assertNotIn("private-cache-path-sentinel", str(failure.exception))
            self.assertEqual(len(calls), 1)

    def test_qwen_unindexed_cuda_uses_current_device_for_precision_and_loading(self):
        calls = []
        class Factory:
            @staticmethod
            def from_pretrained(name, **kwargs):
                calls.append(kwargs)
                return object()
        cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                                     current_device=lambda: 1,
                                     get_device_capability=lambda index: (8, 0) if index == 0 else (7, 5))
        torch = types.SimpleNamespace(float32="float32", bfloat16="bfloat16", cuda=cuda)
        with patch.dict(sys.modules, {"torch": torch, "qwen_tts": types.SimpleNamespace(Qwen3TTSModel=Factory)}), \
             patch.dict(os.environ, {"TTS_DEVICE": "cuda"}, clear=True):
            backend = backends.QwenBackend(None)
            backend._model("preset")
        self.assertEqual(backend.device, "cuda:1")
        self.assertEqual(calls[0]["device_map"], "cuda:1")
        self.assertEqual(calls[0]["dtype"], "float32")

    def test_qwen_precision_uses_bfloat16_on_ampere_and_float32_on_older_cuda(self):
        class Factory:
            @staticmethod
            def from_pretrained(name, **kwargs):
                return kwargs["dtype"]
        for major, expected in ((8, "bfloat16"), (7, "float32")):
            cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                         get_device_capability=lambda index: (major, 0))
            torch = types.SimpleNamespace(float32="float32", bfloat16="bfloat16", cuda=cuda)
            with self.subTest(major=major), patch.dict(sys.modules, {"torch": torch, "qwen_tts": types.SimpleNamespace(Qwen3TTSModel=Factory)}), \
                 patch.dict(os.environ, {"TTS_DEVICE": "cuda:0"}, clear=True):
                backend = backends.QwenBackend(None)
                self.assertEqual(backend._model("preset"), expected)

    def test_qwen_prepare_rejects_wrong_checkpoint_or_voice_before_drafting(self):
        backend = object.__new__(backends.QwenBackend)
        def model(kind, speakers=("Ryan", "Vivian"), languages=("English", "Chinese")):
            return types.SimpleNamespace(model=types.SimpleNamespace(tts_model_type=kind),
                                         get_supported_speakers=lambda: speakers,
                                         get_supported_languages=lambda: languages)
        registry = {"voices": {"assistant": {"kind": "preset", "speaker": "Ryan"},
                               "user": {"kind": "preset", "speaker": "Vivian"}}}
        with patch.dict(os.environ, {"MOSS_TEXT_LANGUAGE": "en"}, clear=True), \
             patch("scripts.text.voices.voice_registry", return_value=registry):
            with patch.object(backend, "_model", return_value=model("base")):
                with self.assertRaisesRegex(ValueError, "CustomVoice"):
                    backend.prepare()
            with patch.object(backend, "_model", return_value=model("custom_voice", speakers=("Ryan",))):
                with self.assertRaisesRegex(ValueError, "configured speakers"):
                    backend.prepare()
            with patch.object(backend, "_model", side_effect=[model("custom_voice"), model("custom_voice")]):
                with self.assertRaisesRegex(ValueError, "Base"):
                    backend.prepare(include_references=True)
            with patch.object(backend, "_model", side_effect=[model("custom_voice"), model("base")]):
                backend.prepare(include_references=True)

    def test_legacy_voice_manifest_is_automatically_selected_only_for_moss(self):
        from scripts.production.moss_scale import runner
        with TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "voices" / "AISHELL-3" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}", encoding="utf-8")
            for backend, expected in (("moss_onnx", str(manifest)), ("qwen3", None), ("kokoro", None)):
                seen = []
                with self.subTest(backend=backend), patch.dict(os.environ, {}, clear=True), \
                     patch.object(runner, "load_env"), \
                     patch("scripts.production.moss_scale.config.resources_root", return_value=root), \
                     patch.object(runner, "check_inputs", side_effect=lambda: seen.append(os.environ.get("VOICE_MANIFEST")) or []), \
                     patch.object(sys, "stdout", new_callable=io.StringIO), \
                     patch.object(sys, "argv", ["scale_moss", "--check", "--tts-backend", backend]):
                    runner.main()
                self.assertEqual(seen, [expected])

    def test_qwen_model_load_errors_do_not_echo_remote_secrets(self):
        class Factory:
            @staticmethod
            def from_pretrained(name, **kwargs):
                raise RuntimeError("signed-download-token=private-sentinel")
        torch = types.SimpleNamespace(float32="float32", float16="float16")
        qwen = types.SimpleNamespace(Qwen3TTSModel=Factory)
        with patch.dict(sys.modules, {"torch": torch, "qwen_tts": qwen}), \
             patch.dict(os.environ, {"TTS_DEVICE": "cpu"}, clear=True):
            backend = backends.QwenBackend(None)
            with self.assertRaises(RuntimeError) as exc:
                backend.prepare()
        self.assertIn("Qwen model initialization failed", str(exc.exception))
        self.assertNotIn("private-sentinel", str(exc.exception))

    def test_qwen_cache_defaults_to_resources_without_relocating_credentials(self):
        torch = types.SimpleNamespace(float32="float32", float16="float16")
        qwen = types.SimpleNamespace(Qwen3TTSModel=object())
        with TemporaryDirectory() as folder, patch.dict(sys.modules, {"torch": torch, "qwen_tts": qwen}), \
             patch.dict(os.environ, {"TTS_DEVICE": "cpu", "DUPLEX_RESOURCES_DIR": folder}, clear=True):
            backend = backends.QwenBackend(None)
            self.assertEqual(backend.cache_dir, Path(folder) / "hf_cache" / "hub")
            self.assertEqual(os.environ["HF_HUB_CACHE"], str(Path(folder) / "hf_cache" / "hub"))
            self.assertEqual(os.environ["HF_XET_CACHE"], str(Path(folder) / "hf_cache" / "xet"))
            self.assertNotIn("HF_HOME", os.environ)

    def test_continuous_run_stops_after_repeated_failures(self):
        from scripts.production.moss_scale import runner
        with TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True), \
             patch.object(runner, "load_env"), patch.object(runner, "check_inputs", return_value=[]), \
             patch.object(runner, "FastSynth"), patch.object(runner, "recover_deliveries"), \
             patch.object(runner, "DeliveryIndex") as index, \
             patch.object(runner, "generate", side_effect=ValueError("synthetic failure")) as generate, \
             patch.object(sys, "stdout", new_callable=io.StringIO), \
             patch.object(sys, "argv", ["scale_moss", "--root", folder, "--continuous"]):
            index.return_value.hours = 0
            with self.assertRaisesRegex(RuntimeError, "连续12个候选出错"):
                runner.main()
            rows = (Path(folder) / "events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 12)
            self.assertLessEqual(generate.call_count, 13)

    def test_qwen_preflight_rejects_unavailable_cuda_index(self):
        from scripts.production.moss_scale import runner
        cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
        torch = types.SimpleNamespace(cuda=cuda)
        with patch.dict(os.environ, {"TTS_BACKEND": "qwen3", "TTS_DEVICE": "cuda:999", "LLM_API_KEY": "fake"}, clear=True), \
             patch.dict(sys.modules, {"torch": torch}), \
             patch.object(runner.importlib.util, "find_spec", return_value=object()), \
             patch.object(runner, "resolve_vad_model") as vad, \
             patch.object(runner, "background_rooms"), \
             patch("scripts.text.voices.reference_voices", return_value={}), \
             patch("scripts.text.voices.voices_for_dialogue", return_value={}):
            vad.return_value.is_file.return_value = True
            self.assertIn("Requested Qwen CUDA device index is unavailable.", runner.check_inputs())

    def test_qwen_model_load_failure_happens_before_any_script_request(self):
        from scripts.production.moss_scale import runner
        with TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True), \
             patch.object(runner, "load_env"), patch.object(runner, "check_inputs", return_value=[]), \
             patch.object(runner, "recover_deliveries"), patch.object(runner, "DeliveryIndex") as index, \
             patch.object(runner, "FastSynth") as synth, patch.object(runner, "generate") as generate, \
             patch("scripts.text.voices.reference_voices", return_value={}), \
             patch.object(sys, "argv", ["scale_moss", "--root", folder, "--hours", ".01", "--tts-backend", "qwen3"]):
            index.return_value.hours = 0
            synth.return_value.backend.prepare.side_effect = RuntimeError("synthetic load failure")
            with self.assertRaisesRegex(RuntimeError, "synthetic load failure"):
                runner.main()
            generate.assert_not_called()

    def test_failed_backend_exception_is_not_written_to_event_logs(self):
        from scripts.production.moss_scale import runner
        with TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True), \
             patch.object(runner, "load_env"), patch.object(runner, "check_inputs", return_value=[]), \
             patch.object(runner, "FastSynth"), patch.object(runner, "recover_deliveries"), \
             patch.object(runner, "DeliveryIndex") as index, \
             patch.object(sys, "stdout", new_callable=io.StringIO) as stdout, \
             patch.object(sys, "argv", ["scale_moss", "--root", folder, "--continuous"]):
            index.return_value.hours = 0
            def fail(*args):
                (Path(folder) / "STOP").touch()
                raise RuntimeError("signature=private-sentinel")
            with patch.object(runner, "generate", side_effect=fail):
                runner.main()
            log = (Path(folder) / "events.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("private-sentinel", log)
            self.assertNotIn("private-sentinel", stdout.getvalue())
            self.assertEqual(json.loads(log)["code"], "other")

    def test_stop_during_draft_does_not_prefetch_another_paid_request(self):
        from scripts.production.moss_scale import runner
        with TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True), \
             patch.object(runner, "load_env"), patch.object(runner, "check_inputs", return_value=[]), \
             patch.object(runner, "FastSynth"), patch.object(runner, "recover_deliveries"), \
             patch.object(runner, "DeliveryIndex") as index, \
             patch.object(runner, "deliver", return_value={"credited_seconds": 1}) as deliver, \
             patch.object(sys, "argv", ["scale_moss", "--root", folder, "--continuous"]):
            index.return_value.hours = 0
            index.return_value.add.return_value = 1 / 3600
            def stop_during_draft(*args):
                (Path(folder) / "STOP").touch()
                return {"dialogue_id": "first"}
            with patch.object(runner, "generate", side_effect=stop_during_draft) as generate:
                runner.main()
            self.assertEqual(generate.call_count, 1)
            deliver.assert_not_called()

    def test_stop_file_prevents_finite_target_from_starting_synthesis(self):
        from scripts.production.moss_scale import runner
        with TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True), \
             patch.object(runner, "load_env"), patch.object(runner, "check_inputs", return_value=[]), \
             patch.object(runner, "FastSynth") as synth, \
             patch.object(sys, "argv", ["scale_moss", "--root", folder, "--hours", "1"]):
            (Path(folder) / "STOP").touch()
            runner.main()
            synth.assert_not_called()

    def test_tagged_run_resumes_without_mixing_different_namespaces(self):
        from scripts.production.moss_scale import runner
        with TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True), patch.object(runner, "load_env"):
            with patch.object(sys, "argv", ["scale_moss", "--root", folder, "--hours", "0", "--run-tag", "run-a"]):
                runner.main()
            with patch.object(sys, "argv", ["scale_moss", "--root", folder, "--hours", "0"]):
                runner.main()
            self.assertEqual(os.environ["CHORUS_RUN_TAG"], "run-a")
            with patch.object(sys, "stderr", new_callable=io.StringIO), \
                 patch.object(sys, "argv", ["scale_moss", "--root", folder, "--hours", "0", "--run-tag", "run-b"]):
                with self.assertRaises(SystemExit):
                    runner.main()

    def test_kokoro_model_root_honors_shared_resource_directory_and_override(self):
        from scripts.production.moss_scale.config import kokoro_model_root
        with TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.dict(os.environ, {"DUPLEX_RESOURCES_DIR": str(root)}, clear=True):
                self.assertEqual(kokoro_model_root(), root / "models" / "kokoro-multi-lang-v1_1")
                os.environ["KOKORO_MODEL_DIR"] = str(root / "custom")
                self.assertEqual(kokoro_model_root(), root / "custom")

    def test_preflight_reports_missing_backend_dependencies(self):
        from scripts.production.moss_scale.runner import check_inputs
        with TemporaryDirectory() as folder, patch("importlib.util.find_spec", return_value=None):
            with patch.dict(os.environ, {"TTS_BACKEND": "moss_onnx", "MOSS_MODEL_DIR": folder}, clear=True):
                self.assertIn("MOSS requires the onnxruntime Python package.", check_inputs())
            with patch.dict(os.environ, {"TTS_BACKEND": "kokoro", "KOKORO_MODEL_DIR": folder}, clear=True):
                errors = check_inputs()
                self.assertIn("Kokoro requires the sherpa_onnx Python package.", errors)
                self.assertTrue(any("Kokoro model assets are missing" in error for error in errors))

    def test_explicit_resources_choose_the_matching_background_directory(self):
        from scripts.config import resolve_background_root
        with TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.dict(os.environ, {"DUPLEX_RESOURCES_DIR": str(root)}, clear=True), \
                 patch.object(Path, "is_dir", return_value=True):
                self.assertEqual(resolve_background_root(), root / "background" / "voices_babble")
                os.environ["BG_SOURCE_DIR"] = str(root / "custom")
                self.assertEqual(resolve_background_root(), root / "custom")

    def test_moss_preflight_detects_missing_external_weights(self):
        from scripts.production.moss_scale.runner import check_moss_assets
        with TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = {"model_files": {"tts_meta": "tts.json", "codec_meta": "codec.json", "tokenizer_model": "tokenizer.model"}}
            tts_files = {"prefill": "moss_tts_prefill.onnx", "decode_step": "decode.onnx", "local_decoder": "local.onnx"}
            codec_files = {"encode": "encode.onnx", "decode_full": "decode_full.onnx", "decode_step": "decode_step.onnx"}
            (root / "browser_poc_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "tts.json").write_text(json.dumps({"files": tts_files}), encoding="utf-8")
            (root / "codec.json").write_text(json.dumps({"files": codec_files}), encoding="utf-8")
            for name in (*tts_files.values(), *codec_files.values(), "tokenizer.model"):
                (root / name).write_bytes(b"x")
            self.assertIn("moss_tts_global_shared.data", " ".join(check_moss_assets(root)))
            (root / "moss_tts_global_shared.data").write_bytes(b"x")
            self.assertEqual(check_moss_assets(root), [])

    def test_verified_asset_install_recovers_from_failed_publish(self):
        import hashlib
        from scripts.sound.tools.verified_download import install_verified
        data = b"test asset"
        digest = hashlib.sha256(data).hexdigest()
        with TemporaryDirectory() as folder:
            destination = Path(folder) / "model.bin"
            with patch("urllib.request.urlopen", side_effect=lambda *a, **k: io.BytesIO(data)):
                with patch("os.link", side_effect=OSError("interrupted")):
                    with self.assertRaises(OSError):
                        install_verified("https://example.test/file", destination, digest, 100)
                self.assertFalse(destination.exists())
                self.assertEqual(list(Path(folder).glob(".download-*.tmp")), [])
                self.assertEqual(install_verified("https://example.test/file", destination, digest, 100), destination)
                self.assertEqual(destination.read_bytes(), data)
                self.assertEqual(install_verified("https://example.test/file", destination, digest, 100), destination)
                with self.assertRaises(FileExistsError):
                    install_verified("https://example.test/file", destination, "0" * 64, 100)
                self.assertEqual(destination.read_bytes(), data)

    def test_moss_download_cache_follows_resources_and_respects_override(self):
        from scripts.sound.tools import download_moss
        calls = []
        sdk = types.SimpleNamespace(snapshot_download=lambda **kwargs: calls.append(kwargs))
        with TemporaryDirectory() as folder, patch.dict(sys.modules, {"huggingface_hub": sdk}):
            root = Path(folder)
            with patch.dict(os.environ, {"DUPLEX_RESOURCES_DIR": str(root)}, clear=True):
                download_moss.download(root / "models")
                self.assertTrue(Path(os.environ["HF_XET_CACHE"]).is_relative_to(root))
                self.assertEqual(len(calls), 2)
            with patch.dict(os.environ, {"DUPLEX_RESOURCES_DIR": str(root), "HF_XET_CACHE": "custom-cache"}, clear=True):
                download_moss.download(root / "models")
                self.assertEqual(os.environ["HF_XET_CACHE"], "custom-cache")

    def test_bundled_moss_reference_loader_resamples_stereo(self):
        import importlib
        import soundfile as sf
        runtime_path = Path(__file__).resolve().parents[1] / "scripts" / "sound" / "moss_runtime"
        with TemporaryDirectory() as folder, patch.object(sys, "path", [str(runtime_path), *sys.path]):
            audio_path = Path(folder) / "reference.wav"
            sf.write(audio_path, np.column_stack((np.full(800, .2), np.full(800, -.1))), 8000)
            runtime_cls = importlib.import_module("onnx_tts_runtime").OnnxTtsRuntime
            runtime = object.__new__(runtime_cls)
            runtime.codec_meta = {"codec_config": {"sample_rate": 16000, "channels": 1}}
            audio = runtime._load_reference_audio(audio_path)
        self.assertEqual(audio.shape, (1, 1, 1600))
        self.assertEqual(audio.dtype, np.float32)
        self.assertTrue(audio.flags.c_contiguous)
        self.assertTrue(np.isfinite(audio).all())
        self.assertAlmostEqual(float(audio.mean()), .05, places=2)

    def test_environment_loaded_after_import_selects_pinned_vad(self):
        from scripts import config
        from scripts.sound.synthesis.vad import SileroVAD
        with TemporaryDirectory() as folder:
            root = Path(folder)
            legacy = root / "legacy.onnx"
            legacy.touch()
            pinned = root / "resources" / "models" / "silero_vad_v512.onnx"
            pinned.parent.mkdir(parents=True)
            pinned.touch()
            env_file = root / "test.env"
            env_file.write_text(f"DUPLEX_RESOURCES_DIR={root / 'resources'}\n", encoding="utf-8")
            inputs = [types.SimpleNamespace(name=name) for name in ("input", "state", "sr")]
            outputs = [types.SimpleNamespace(name=name) for name in ("output", "stateN")]
            session = types.SimpleNamespace(get_inputs=lambda: inputs, get_outputs=lambda: outputs)
            with patch.dict(os.environ, {}, clear=True), patch.object(config, "_legacy_vad", legacy):
                config.load_env(env_file)
                self.assertEqual(config.resolve_vad_model(), pinned)
                with patch("onnxruntime.InferenceSession", return_value=session) as create_session:
                    SileroVAD()
                self.assertEqual(create_session.call_args.args[0], str(pinned))
                os.environ["VAD_MODEL"] = str(legacy)
                self.assertEqual(config.resolve_vad_model(), legacy)

    def test_production_entry_does_not_override_the_bundled_runtime(self):
        from scripts.production.moss_scale import runner
        seen = []
        with patch.dict(os.environ, {}, clear=True), patch.object(runner, "load_env"), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(sys, "argv", ["scale_moss", "--check"]), \
             patch.object(runner, "check_inputs", side_effect=lambda: seen.append(os.environ.get("MOSS_DIR")) or []):
            runner.main()
        self.assertEqual(seen, [None])

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
        load_options = []
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
                load_options.append(kwargs)
                return Model(name)
        torch = types.SimpleNamespace(float32="float32", float16="float16")
        qwen = types.SimpleNamespace(Qwen3TTSModel=Factory)
        with TemporaryDirectory() as folder, patch.dict(sys.modules, {"torch": torch, "qwen_tts": qwen}):
            reference = Path(folder) / "voice.wav"
            reference.write_bytes(b"audio")
            with patch.dict(os.environ, {"TTS_DEVICE": "cpu", "MOSS_TEXT_LANGUAGE": "en",
                                         "HF_HUB_CACHE": str(Path(folder)/"hub"), "HF_XET_CACHE": str(Path(folder)/"xet")}, clear=True):
                backend = backends.QwenBackend(Path(folder))
                backend.synthesize("hello", "Ryan", None, 4)
                backend.synthesize("hello", "guest", str(reference), 4)
                backend.synthesize("again", "guest", str(reference), 4)
        self.assertEqual([entry[0] for entry in calls].count("load"), 2)
        self.assertEqual([entry[0] for entry in calls].count("prompt"), 1)
        self.assertTrue(all(option["cache_dir"] == str(Path(folder)/"hub") for option in load_options))
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
            backend = backends.QwenBackend(None)
            result = backend.synthesize("hello", "Ryan", None, 4)
            with self.assertRaisesRegex(ValueError, "more than two codec frames"):
                backend.synthesize("hello", "Ryan", None, .16)
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
