"""Filesystem defaults for the production runner."""
import os
from pathlib import Path

from scripts.config import ROOT


def resources_root() -> Path:
    default = Path("D:/DuplexVoice_resources") if os.name == "nt" else ROOT / "resources"
    return Path(os.environ.get("DUPLEX_RESOURCES_DIR", str(default))).expanduser()


def background_root() -> Path:
    default = Path("D:/DuplexVoice_downloads/VOICES_babble/data") if os.name == "nt" else ROOT / "data" / "voices_babble"
    return Path(os.environ.get("BG_SOURCE_DIR", str(default))).expanduser()


def default_output_root() -> Path:
    default = Path("D:/DuplexVoice_production/release_moss_scale_20261115") if os.name == "nt" else ROOT / "output" / "moss"
    return Path(os.environ.get("DUPLEX_PRODUCTION_ROOT", str(default))).expanduser()


# Retained for other modules that use these constants at import time.
RESOURCES = resources_root()
BABBLE = background_root()
DEFAULT_ROOT = default_output_root()
