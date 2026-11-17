"""Download one licensed VOiCES babble clip for a short setup smoke test."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.config import load_env
from scripts.production.moss_scale.config import resources_root
from scripts.sound.tools.verified_download import install_verified

URL = ("https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/"
       "distant-16k/distractors/rm1/babb/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo.wav")
SHA256 = "ea4bf547280c5f950c853682bfd52bf9ff3ab7c3b4c075e687dff59f9e459243"
MAX_BYTES = 1024 * 1024


def install(root: Path) -> Path:
    destination = root.expanduser() / "rm1" / "babb" / "voices_sample.wav"
    return install_verified(URL, destination, SHA256, MAX_BYTES)


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=resources_root() / "background" / "voices_babble")
    args = parser.parse_args()
    print(f"VOiCES sample ready: {install(args.output_root)}")
    print("One five-second clip is for smoke tests; use a varied, licensed corpus for production.")


if __name__ == "__main__":
    main()
