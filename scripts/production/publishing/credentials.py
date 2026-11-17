"""Resolve Hugging Face credentials in process without hardcoded local paths."""
from __future__ import annotations

import os
from pathlib import Path


def read_token_file(path: str) -> str:
    try:
        token = Path(path).expanduser().read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeError):
        raise RuntimeError("Configured Hugging Face token file is unavailable or invalid") from None
    if not token:
        raise RuntimeError("Configured Hugging Face token file is empty")
    return token


def hf_token() -> str:
    from scripts.config import load_env
    load_env()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    explicit = os.environ.get("HF_TOKEN_PATH")
    if not token and explicit:
        token = read_token_file(explicit)
    if not token:
        from huggingface_hub import get_token
        token = get_token()
    if not token:
        raise RuntimeError("No authorized Hugging Face credential is available")
    return token
