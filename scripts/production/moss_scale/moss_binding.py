"""Opt-in MOSS root binding; caller holds ProductionLock and has not loaded TTS."""
import os
from pathlib import Path
from scripts.config import MOSS_DIR
from scripts.sound.synthesis.moss_identity import moss_content_identity
from .config import moss_model_root
from .model_lock import bind_models


def prepare_moss_models(root):
    path = moss_model_root(Path(os.environ.get('MOSS_DIR', str(MOSS_DIR)))).absolute()
    identity = moss_content_identity(path)
    bind_models(root, {'model': identity}, backend='moss_onnx')
    # Pin the constructor to the same verified directory, regardless of fallback discovery.
    path = path.resolve()
    os.environ['MOSS_MODEL_DIR'] = str(path)
    return path, identity


def verify_loaded_moss(backend, path, identity):
    if Path(backend.model_dir).resolve() != path or moss_content_identity(path) != identity:
        raise ValueError('MOSS model changed during loading; no production permitted')
    backend.production_identity = identity
