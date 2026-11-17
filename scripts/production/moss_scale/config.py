"""Filesystem defaults for the production runner."""
import os
from pathlib import Path

from scripts.config import ROOT


def resources_root() -> Path:
    default = Path("D:/DuplexVoice_resources") if os.name == "nt" else ROOT / "resources"
    return Path(os.environ.get("DUPLEX_RESOURCES_DIR", str(default))).expanduser()



def moss_model_root(runtime_dir: Path | None = None) -> Path:
    explicit = os.environ.get("MOSS_MODEL_DIR")
    if explicit:
        return Path(explicit).expanduser()
    portable = resources_root() / "models" / "moss-tts-nano-onnx"
    if portable.is_dir():
        return portable
    if runtime_dir is None:
        from scripts.config import MOSS_DIR
        runtime_dir = MOSS_DIR
    return Path(runtime_dir) / "models"


def background_root() -> Path:
    from scripts.config import resolve_background_root
    return resolve_background_root()


def kokoro_model_root() -> Path:
    return Path(os.environ.get("KOKORO_MODEL_DIR", str(resources_root() / "models" / "kokoro-multi-lang-v1_1"))).expanduser()


def default_output_root() -> Path:
    default = Path("D:/DuplexVoice_production/release_moss_scale_20261115") if os.name == "nt" else ROOT / "output" / "moss"
    return Path(os.environ.get("DUPLEX_PRODUCTION_ROOT", str(default))).expanduser()


# Retained for other modules that use these constants at import time.
RESOURCES = resources_root()
BABBLE = background_root()
DEFAULT_ROOT = default_output_root()
