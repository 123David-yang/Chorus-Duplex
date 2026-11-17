import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from scripts.sound.synthesis.backends import QwenBackend
from scripts.sound.synthesis.qwen_production import prepare_models
from tests.test_model_identity import _snapshot, _tensor_bytes


def backend(root):
    value = object.__new__(QwenBackend)
    value.cache_dir = root/'cache'
    value.models = {}
    value.factory = Mock()
    value.device = 'cpu'
    value.dtype = 'float32'
    return value


class QwenProductionBindingTests(unittest.TestCase):
    def test_runner_checks_identity_before_recovery_and_completed_target_return(self):
        from scripts.production.moss_scale import runner
        for failure in (False, True):
            with self.subTest(failure=failure), TemporaryDirectory() as folder, \
                 patch.dict(os.environ, {}, clear=True), patch.object(runner, 'load_env'), \
                 patch('scripts.sound.synthesis.backends.create_backend') as create, \
                 patch('scripts.text.voices.reference_voices', return_value={}), \
                 patch.object(runner, 'recover_deliveries') as recover, \
                 patch.object(runner, 'DeliveryIndex') as index, \
                 patch.object(runner, 'generate') as generate, \
                 patch.object(sys, 'argv', ['scale_moss', '--root', folder, '--hours', '0', '--tts-backend', 'qwen3']):
                events = []
                def bind(*args, **kwargs):
                    events.append('identity')
                    if failure:
                        raise RuntimeError('model changed')
                create.return_value.prepare_production.side_effect = bind
                recover.side_effect = lambda *args: events.append('recover')
                index.return_value.hours = 1
                if failure:
                    with self.assertRaisesRegex(RuntimeError, 'model changed'):
                        runner.main()
                    recover.assert_not_called()
                    index.assert_not_called()
                else:
                    runner.main()
                    self.assertEqual(events, ['identity', 'recover'])
                generate.assert_not_called()
                create.return_value.prepare.assert_not_called()

    def test_loaded_backend_cannot_be_rebound(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            value = backend(root)
            value.models['preset'] = object()
            with self.assertRaisesRegex(RuntimeError, "before loading"):
                prepare_models(value, root)
            self.assertFalse((root/'model-lock.json').exists())

    def test_binds_both_roles_and_loads_the_same_paths(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            preset, clone, output = root/'preset', root/'clone', root/'output'
            _snapshot(preset)
            _snapshot(clone)
            (preset/'config.json').write_text(json.dumps({'model_type': 'qwen3_tts', 'tts_model_type': 'custom_voice'}))
            output.mkdir()
            with patch.dict(os.environ, {'QWEN_PRESET_MODEL': str(preset), 'TTS_MODEL': str(clone)}, clear=True):
                first = backend(root)
                lock = prepare_models(first, output, True)
                first.factory.from_pretrained.assert_not_called()
                first._model('preset')
                self.assertEqual(first.factory.from_pretrained.call_args.args[0], str(preset))
                self.assertEqual(prepare_models(backend(root), output, True), lock)
                original = (output/'model-lock.json').read_bytes()
                (clone/'model.safetensors').write_bytes(_tensor_bytes(b'changed'))
                changed = backend(root)
                with self.assertRaises(RuntimeError):
                    prepare_models(changed, output, True)
                changed.factory.from_pretrained.assert_not_called()
                self.assertFalse(hasattr(changed, 'production_paths'))
                self.assertEqual((output/'model-lock.json').read_bytes(), original)

    def test_unknown_existing_audio_is_not_bound(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            preset, output = root/'preset', root/'output'
            _snapshot(preset)
            (preset/'config.json').write_text(json.dumps({'model_type': 'qwen3_tts', 'tts_model_type': 'custom_voice'}))
            (output/'audio').mkdir(parents=True)
            (output/'audio'/'old.flac').write_bytes(b'old')
            with patch.dict(os.environ, {'QWEN_PRESET_MODEL': str(preset)}, clear=True):
                value = backend(root)
                with self.assertRaises(RuntimeError):
                    prepare_models(value, output)
                value.factory.from_pretrained.assert_not_called()
                self.assertFalse((output/'model-lock.json').exists())
