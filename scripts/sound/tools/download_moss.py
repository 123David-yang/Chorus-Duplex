"""Download the pinned public MOSS ONNX models used by Chorus-Duplex."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts.config import load_env
from scripts.production.moss_scale.config import resources_root

MODELS = {
    "MOSS-TTS-Nano-100M-ONNX": "f52645cb467506d8e18e746ddd59482685b74e58",
    "MOSS-Audio-Tokenizer-Nano-ONNX": "ceff0d0749bfb3fa2d61149794ec6feef0d1e1ae",
}


def download(root: Path) -> None:
    # Set before importing the SDK so its download caches follow the resource disk.
    os.environ.setdefault("HF_HOME", str(resources_root() / "hf_cache"))
    os.environ.setdefault("HF_XET_CACHE", str(Path(os.environ["HF_HOME"]) / "xet"))
    from huggingface_hub import snapshot_download

    root = root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    for name, revision in MODELS.items():
        snapshot_download(
            repo_id=f"OpenMOSS-Team/{name}", revision=revision, token=False,
            local_dir=str(root / name),
            allow_patterns=["*.onnx", "*.data", "*.json", "tokenizer.model"],
        )
        print(f"Ready: {name} ({revision[:12]})")


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    default = os.environ.get("MOSS_MODEL_DIR") or str(resources_root() / "models" / "moss-tts-nano-onnx")
    parser.add_argument("--output", type=Path, default=Path(default))
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__":
    main()
