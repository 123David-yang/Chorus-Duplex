"""Resolve and bind Qwen content before paid generation; no model loading here."""
import hashlib
import os
from pathlib import Path

from .model_identity import qwen_content_identity
from .snapshot_copy import materialize_snapshot
from scripts.production.moss_scale.model_lock import bind_models


def prepare_models(backend, root, include_references=False):
    if getattr(backend, "models", {}):
        raise RuntimeError("Bind Qwen production before loading models; create a fresh backend")
    from scripts.production.moss_scale.config import resources_root
    names = {"preset": os.environ.get("QWEN_PRESET_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")}
    if include_references:
        names["clone"] = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    paths, identities = {}, {}
    try:
        for role, name in names.items():
            source = Path(name).expanduser()
            if not source.is_dir():
                from huggingface_hub import snapshot_download
                offline = os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "ON", "YES", "TRUE"}
                source = Path(snapshot_download(name, cache_dir=str(backend.cache_dir), local_files_only=offline))
            if source.parent.name == "snapshots":
                key = hashlib.sha256((source.parent.parent.name + ":" + source.name).encode()).hexdigest()
                destination = resources_root() / "qwen_materialized" / key
                if not destination.exists():
                    materialize_snapshot(source, destination)
                source = destination
            identities[role] = qwen_content_identity(source, expected_model_type="base" if role == "clone" else "custom_voice")
            paths[role] = str(source)
        payload = bind_models(root, identities)
    except Exception:
        raise RuntimeError("Qwen production model preparation failed; verify snapshots and use a compatible production root") from None
    backend.production_paths = paths
    backend.production_identity = payload
    return payload
