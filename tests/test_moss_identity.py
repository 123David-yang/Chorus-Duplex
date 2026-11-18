from pathlib import Path
from tempfile import TemporaryDirectory
import os
import json
import unittest
from unittest.mock import patch
from scripts.sound.synthesis.moss_identity import moss_content_identity
from scripts.production.moss_scale.model_lock import bind_models


class MossIdentityTests(unittest.TestCase):
    def create(self, root):
        for name in ('browser_poc_manifest.json', 'tts_browser_onnx_meta.json',
                     'codec_browser_onnx_meta.json', 'tokenizer.model', 'graph.onnx', 'weights.data'):
            (root/name).write_bytes(b'asset-'+name.encode())
        (root/'browser_poc_manifest.json').write_text(json.dumps({'model_files': {
            'tts_meta': 'tts_browser_onnx_meta.json', 'codec_meta': 'codec_browser_onnx_meta.json',
            'tokenizer_model': 'tokenizer.model'}}))
        for name in ('tts_browser_onnx_meta.json', 'codec_browser_onnx_meta.json'):
            (root/name).write_text(json.dumps({'files': {'graph': 'graph.onnx'},
                                             'external_data_files': {'graph.onnx': ['weights.data']}}))

    def test_stable_across_paths_but_changes_with_weights_and_metadata(self):
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            left, right = Path(a), Path(b)
            self.create(left)
            self.create(right)
            original = moss_content_identity(left)
            self.assertEqual(original, moss_content_identity(right))
            for name in ('weights.data', 'tokenizer.model'):
                with self.subTest(name=name):
                    path = right/name
                    data = path.read_bytes()
                    path.write_bytes(b'changed')
                    self.assertNotEqual(original['sha256'], moss_content_identity(right)['sha256'])
                    path.write_bytes(data)

    def test_outside_manifest_reference_fails_and_unusual_asset_suffix_is_hashed(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.create(root)
            before = moss_content_identity(root)
            (root/'custom.bin').write_bytes(b'new data')
            self.assertNotEqual(before['sha256'], moss_content_identity(root)['sha256'])
            manifest = root/'browser_poc_manifest.json'
            data = json.loads(manifest.read_text())
            data['model_files']['tokenizer_model'] = '../outside.model'
            manifest.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                moss_content_identity(root)

    def test_missing_empty_and_hardlinked_assets_fail_without_paths(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.create(root)
            asset = root/'weights.data'
            asset.write_bytes(b'')
            with self.assertRaisesRegex(ValueError, '^MOSS content fingerprint could not be verified$'):
                moss_content_identity(root)
            asset.unlink()
            os.link(root/'graph.onnx', asset)
            with self.assertRaises(ValueError):
                moss_content_identity(root)
            asset.unlink()
            (root/'tokenizer.model').unlink()
            with self.assertRaises(ValueError):
                moss_content_identity(root)

    def test_moss_lock_refuses_changed_assets_and_qwen_identity(self):
        with TemporaryDirectory() as folder, TemporaryDirectory() as output:
            root = Path(folder)
            self.create(root)
            identity = moss_content_identity(root)
            original = bind_models(output, {'model': identity}, backend='moss_onnx')
            self.assertEqual(original, bind_models(output, {'model': identity}, backend='moss_onnx'))
            (root/'weights.data').write_bytes(b'new weights')
            with self.assertRaisesRegex(ValueError, 'changed'):
                bind_models(output, {'model': moss_content_identity(root)}, backend='moss_onnx')
            with self.assertRaises(ValueError):
                bind_models(output, {'model': identity})

    def test_directory_scan_failure_is_not_silently_omitted(self):
        with TemporaryDirectory() as folder:
            self.create(Path(folder))
            def failed_walk(*args, **kwargs):
                kwargs['onerror'](PermissionError('private-path-marker'))
                return iter(())
            with patch('scripts.sound.synthesis.moss_identity.os.walk', side_effect=failed_walk):
                with self.assertRaisesRegex(ValueError, '^MOSS content fingerprint could not be verified$'):
                    moss_content_identity(folder)
