"""Compatibility entry point for the MOSS scale-up pipeline."""

from .moss_scale.config import BABBLE, DEFAULT_ROOT, RESOURCES
from .moss_scale.delivery import deliver, refresh
from .moss_scale.render import FastSynth, render
from .moss_scale.runner import main
from .moss_scale.script import generate, validate
from .moss_scale.storage import atomic_json

__all__ = [
    "BABBLE", "DEFAULT_ROOT", "RESOURCES", "FastSynth", "atomic_json",
    "deliver", "generate", "main", "refresh", "render", "validate",
]

if __name__ == "__main__":
    main()
