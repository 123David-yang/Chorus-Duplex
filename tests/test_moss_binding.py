from pathlib import Path
from tempfile import TemporaryDirectory
import os
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from scripts.production.moss_scale import runner
from scripts.production.moss_scale.moss_binding import prepare_moss_models, verify_loaded_moss


class MossBindingTests(unittest.TestCase):
    def test_binding_pins_loader_path_and_loaded_path_must_match(self):
        with TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            identity = {'algorithm': 'moss-onnx-assets-sha256-v1', 'sha256': 'a'*64}
            root = Path(folder)
            with patch('scripts.production.moss_scale.moss_binding.moss_model_root', return_value=root), \
                 patch('scripts.production.moss_scale.moss_binding.moss_content_identity', return_value=identity):
                path, actual = prepare_moss_models(root)
                self.assertEqual(os.environ['MOSS_MODEL_DIR'], str(path))
                backend = SimpleNamespace(model_dir=path)
                verify_loaded_moss(backend, path, actual)
                self.assertEqual(backend.production_identity, identity)
                with patch('scripts.production.moss_scale.moss_binding.moss_content_identity', return_value={**identity, 'sha256': 'b'*64}):
                    with self.assertRaisesRegex(ValueError, 'changed during loading'):
                        verify_loaded_moss(backend, path, actual)
                with self.assertRaises(ValueError):
                    verify_loaded_moss(SimpleNamespace(model_dir=path/'elsewhere'), path, actual)

    def test_binding_happens_before_recovery_even_when_target_met(self):
        with TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True), \
             patch.object(runner, 'load_env'), patch.object(runner, 'FastSynth') as synth, \
             patch.object(runner, 'DeliveryIndex') as index, \
             patch.object(runner, 'recover_deliveries') as recover, \
             patch('scripts.production.moss_scale.moss_binding.prepare_moss_models') as bind, \
             patch.object(sys, 'argv', ['scale_moss', '--root', folder, '--hours', '0', '--bind-moss-model']):
            order = []
            bind.side_effect = lambda *args: order.append('bind') or (Path(folder), {})
            recover.side_effect = lambda *args: order.append('recover')
            index.return_value.hours = 0
            runner.main()
            self.assertEqual(order, ['bind', 'recover'])
            synth.assert_not_called()
            bind.side_effect = ValueError('Changed model')
            recover.reset_mock()
            with self.assertRaises(ValueError):
                runner.main()
            recover.assert_not_called()

    def test_persistent_lock_enforces_check_without_cli_flag(self):
        with TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True), \
             patch.object(runner, 'load_env'), patch.object(runner, 'recover_deliveries') as recover, \
             patch('scripts.production.moss_scale.moss_binding.prepare_moss_models', side_effect=ValueError('Changed model')) as bind, \
             patch.object(sys, 'argv', ['scale_moss', '--root', folder, '--hours', '0']):
            (Path(folder)/'model-lock.json').write_text('{}')
            with self.assertRaises(ValueError):
                runner.main()
            bind.assert_called_once()
            recover.assert_not_called()
