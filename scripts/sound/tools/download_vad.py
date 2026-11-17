"""Download the pinned Silero VAD model used by the production pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.config import load_env
from scripts.production.moss_scale.config import resources_root
from scripts.sound.tools.verified_download import install_verified

URL = ("https://raw.githubusercontent.com/snakers4/silero-vad/"
       "6478567951ae5c9979ad7b234185b5515f4be7a1/src/silero_vad/data/silero_vad.onnx")
SHA256 = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"
MAX_BYTES = 4 * 1024 * 1024


def install(destination: Path) -> Path:
    return install_verified(URL, destination, SHA256, MAX_BYTES)


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=resources_root() / "models" / "silero_vad_v512.onnx")
    args = parser.parse_args()
    print(f"Silero VAD ready: {install(args.output)}")


if __name__ == "__main__":
    main()
