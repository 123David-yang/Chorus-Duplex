"""Portable defaults for the public MOSS pipeline."""
from pathlib import Path

from scripts.config import ROOT

RESOURCES = ROOT / "resources"
BABBLE = ROOT / "data" / "voices_babble"
DEFAULT_ROOT = ROOT / "output" / "moss"
