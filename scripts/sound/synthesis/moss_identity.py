"""Content continuity fingerprint for a static local MOSS ONNX asset tree.

This does not authenticate model origin or replace the runner's asset preflight.
All regular files in the asset tree are covered; local caches are excluded.
Known manifest/metadata file references must remain inside the covered tree.
This does not parse arbitrary ONNX dependency graphs. No paths are returned.
"""
from pathlib import Path
import hashlib
import json
import os
import stat


def moss_content_identity(root):
    root = Path(root).absolute()
    try:
        def checked(path, directory=False):
            info = path.lstat()
            if (stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400
                or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
                or not directory and info.st_nlink != 1):
                raise ValueError('Unsafe model path')
            return info

        for parent in (root, *root.parents):
            checked(parent, True)

        def inventory():
            found = []
            def fail(error):
                raise error
            for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail):
                checked(Path(directory), True)
                dirs[:] = sorted(name for name in dirs if name not in {'.cache', '.git', '__pycache__'})
                for name in dirs:
                    checked(Path(directory)/name, True)
                for name in sorted(files):
                    path = Path(directory)/name
                    checked(path)
                    found.append(path)
            return sorted(found, key=lambda path: path.relative_to(root).as_posix())

        files = inventory()
        names = {path.name for path in files}
        if not {'browser_poc_manifest.json', 'tts_browser_onnx_meta.json',
                'codec_browser_onnx_meta.json', 'tokenizer.model'} <= names or not any(path.suffix == '.onnx' for path in files):
            raise ValueError('Incomplete model layout')
        manifests = [root/name for name in ('browser_poc_manifest.json',
                     'MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json',
                     'MOSS-TTS-Nano-ONNX-CPU/browser_poc_manifest.json')]
        manifest = next(path for path in manifests if path in files)
        def reference(base, relative):
            if not isinstance(relative, str) or not relative:
                raise ValueError('Invalid model reference')
            path = base/relative
            if not path.exists():
                path = base/relative.replace('MOSS-TTS-Nano-ONNX-CPU', 'MOSS-TTS-Nano-100M-ONNX').replace(
                    'MOSS-Audio-Tokenizer-Nano-ONNX-CPU', 'MOSS-Audio-Tokenizer-Nano-ONNX')
            resolved = path.resolve()
            if not resolved.is_relative_to(root) or resolved not in files:
                raise ValueError('Model reference outside covered assets')
            return resolved
        metadata_digests = {}
        def read_metadata(path):
            raw = path.read_bytes()
            metadata_digests[path] = hashlib.sha256(raw).hexdigest()
            return json.loads(raw)
        model_files = read_metadata(manifest)['model_files']
        reference(manifest.parent, model_files.get('tokenizer_model', 'tokenizer.model'))
        for key in ('tts_meta', 'codec_meta'):
            meta = reference(manifest.parent, model_files[key])
            metadata = read_metadata(meta)
            for relative in metadata['files'].values():
                reference(meta.parent, relative)
            for relatives in metadata.get('external_data_files', {}).values():
                if not isinstance(relatives, list):
                    raise ValueError('Invalid external data inventory')
                for relative in relatives:
                    reference(meta.parent, relative)
        rows = []
        verified = []
        for path in files:
            before = checked(path)
            if before.st_size <= 0:
                raise ValueError('Empty model asset')
            with path.open('rb') as stream:
                opened = os.fstat(stream.fileno())
                if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
                    raise ValueError('Model asset changed')
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if path in metadata_digests and digest != metadata_digests[path]:
                    raise ValueError('Model metadata changed after reference validation')
                after = os.fstat(stream.fileno())
            current = checked(path)
            if any((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
                    before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) for info in (after, current)):
                raise ValueError('Model asset changed')
            rows.append([path.relative_to(root).as_posix(), before.st_size, digest])
            verified.append((path, before))
        if inventory() != files:
            raise ValueError('Model inventory changed')
        for path, before in verified:
            current = checked(path)
            if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
                raise ValueError('Model asset changed after hashing')
        digest = hashlib.sha256(json.dumps(rows, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()
        return {'algorithm': 'moss-onnx-assets-sha256-v1', 'sha256': digest,
                'file_count': len(rows), 'verification': 'content-continuity-only'}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, StopIteration):
        raise ValueError('MOSS content fingerprint could not be verified') from None
