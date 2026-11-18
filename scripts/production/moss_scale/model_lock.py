"""Bind new production to content fingerprints; caller holds ProductionLock."""
import json
import re
import os
import stat
import tempfile
from pathlib import Path


def _reject_link(path):
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
            or stat.S_ISREG(info.st_mode) and info.st_nlink != 1):
        raise ValueError("Linked model-lock path is not allowed")


def bind_models(root, identities, *, backend="qwen3"):
    root = Path(root).absolute()
    for directory in (root, *root.parents):
        _reject_link(directory)
    roles = ({"preset"}, {"preset", "clone"}) if backend == "qwen3" else ({"model"},)
    if backend not in {"qwen3", "moss_onnx"} or not isinstance(identities, dict) or set(identities) not in roles:
        raise ValueError("Invalid model roles")
    safe = {}
    for role, identity in identities.items():
        if not isinstance(identity, dict):
            raise ValueError("Invalid model identity")
        digest = identity.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid model identity")
        if backend == "qwen3":
            expected = "custom_voice" if role == "preset" else "base"
            if identity.get("algorithm") != "qwen3-snapshot-sha256-v1" or identity.get("tts_model_type") != expected:
                raise ValueError("Invalid model identity")
            safe[role] = {"sha256": digest, "tts_model_type": expected}
        else:
            if identity.get("algorithm") != "moss-onnx-assets-sha256-v1":
                raise ValueError("Invalid model identity")
            safe[role] = {"sha256": digest, "algorithm": identity["algorithm"]}
    payload = {"schema_version": 1, "backend": backend, "verification": "content-continuity-only", "models": safe}
    path = root / "model-lock.json"
    if path.exists() or path.is_symlink():
        _reject_link(path)
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            raise ValueError("Model lock is unreadable") from None
        if previous != payload:
            raise ValueError("Model content changed; use a new production root")
    else:
        for name in ("audio", "completed", "staging", "duplicates"):
            directory = root / name
            if directory.exists() or directory.is_symlink():
                _reject_link(directory)
                if any(directory.iterdir()):
                    raise ValueError("Existing audio has no proven model identity; use a new production root")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                             prefix=".model-lock-", suffix=".tmp", delete=False) as file:
                temporary = Path(file.name)
                json.dump(payload, file, ensure_ascii=True, sort_keys=True)
                file.flush()
                os.fsync(file.fileno())
            # Publishing without replacement also refuses a concurrently created lock.
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return payload
