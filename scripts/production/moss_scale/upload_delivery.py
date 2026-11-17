"""Append validated clean/babble deliveries to a Hugging Face dataset.

Credentials come from the Hugging Face SDK, never from command-line arguments.
The existing annotations file is preserved and updated with an optimistic
parent-commit guard. Repeating an already completed upload is safe.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _matches_remote(info, payload: Path | bytes) -> bool:
    size = payload.stat().st_size if isinstance(payload, Path) else len(payload)
    if isinstance(payload, Path):
        sha256 = _sha256(payload)
    else:
        sha256 = hashlib.sha256(payload).hexdigest()
    lfs = getattr(info, "lfs", None)
    if lfs:
        remote_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        return remote_sha == sha256
    digest = hashlib.sha1(f"blob {size}\0".encode())
    if isinstance(payload, Path):
        with payload.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    else:
        digest.update(payload)
    return getattr(info, "blob_id", None) == digest.hexdigest()


MAX_BATCH_BYTES = 256 * 1024 * 1024

_SENSITIVE_FIELD = re.compile(
    r"(?i)^(?:token|.*api[_-]?key|.*(?:password|secret|access[_-]?token|refresh[_-]?token|"
    r"hf[_-]?token|github[_-]?token|llm[_-]?api[_-]?key|credential|private[_-]?key|authorization))$"
)
_SECRET_TEXT = re.compile(
    r"(?i)(?:hf_[A-Za-z0-9]{20,}|(?:ghp_|gho_|ghs_|ghu_|ghr_|github_pat_)[A-Za-z0-9_]{20,}|"
    r"sk-[A-Za-z0-9_-]{16,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)
_URL = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)


def _assert_public_safe(value):
    """Reject likely credentials; checks reduce risk but are not an exhaustive secret scan."""
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and _SENSITIVE_FIELD.search(key):
                raise ValueError("Sensitive field in public annotation")
            _assert_public_safe(item)
    elif isinstance(value, list):
        for item in value:
            _assert_public_safe(item)
    elif isinstance(value, str):
        if _SECRET_TEXT.search(value):
            raise ValueError("Possible credential in public annotation")
        for match in _URL.finditer(value):
            parsed = urlsplit(match.group(0))
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Public annotation URL contains private components")




def validate_delivery(root: Path, ident: str, max_bytes: int = MAX_BATCH_BYTES) -> tuple[dict, dict[str, bytes]]:
    import soundfile as sf
    from scripts.text.quality import audit
    from scripts.text.utils.schema import Dialogue

    if not re.fullmatch(r"[A-Za-z0-9_-]+", ident):
        raise ValueError("Invalid dialogue ID")
    root = root.resolve()
    receipt_dir = (root / "completed").resolve()
    receipt_path = (receipt_dir / f"{ident}.json").resolve()
    if not receipt_dir.is_relative_to(root) or receipt_path.parent != receipt_dir:
        raise ValueError("Receipt outside delivery root")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    annotation = receipt["annotation"]
    if annotation["dialogue_id"] != ident:
        raise ValueError("Receipt dialogue ID mismatch")
    _assert_public_safe(annotation)
    result = audit(Dialogue.from_jsonl_line(json.dumps(annotation, ensure_ascii=False)))
    if not result["passed"]:
        raise ValueError(f"Dialogue rule audit failed for {ident}")
    names = (f"{ident}.flac", f"{ident}_babble.flac", f"{ident}.timeline.json")
    if set(receipt["sha256"]) != set(names):
        raise ValueError("Unexpected receipt file list")
    audio_root = (root / "audio").resolve()
    if not audio_root.is_relative_to(root):
        raise ValueError("Audio directory outside delivery root")
    paths = {}
    remaining = max_bytes
    for name in names:
        file = (audio_root / name).resolve()
        if file.parent != audio_root or not file.is_file():
            raise ValueError(f"Missing or modified delivery artifact: {name}")
        # Freeze exact bytes before validation and upload; later file changes cannot alter them.
        with file.open("rb") as stream:
            payload = stream.read(remaining + 1)
        if len(payload) > remaining:
            raise ValueError("Upload batch exceeds the memory limit; split the batch")
        if hashlib.sha256(payload).hexdigest() != receipt["sha256"][name]:
            raise ValueError(f"Missing or modified delivery artifact: {name}")
        paths[f"audio/{name}"] = payload
        remaining -= len(payload)
    frames, sample_rate = receipt["frames"], receipt["sample_rate"]
    if type(frames) is not int or type(sample_rate) is not int or frames <= 0 or sample_rate != 48000:
        raise ValueError("Invalid frame count or sample rate")
    for name in names[:2]:
        info = sf.info(io.BytesIO(paths[f"audio/{name}"]))
        if (info.frames, info.samplerate, info.channels) != (frames, sample_rate, 2):
            raise ValueError("Clean/noisy audio format differs from the receipt")
    timeline = json.loads(paths[f"audio/{names[2]}"])
    _assert_public_safe(timeline)
    turns = annotation["turns"]
    if len(timeline) != len(turns):
        raise ValueError("Timeline turn count differs from annotation")
    for index, (item, turn) in enumerate(zip(timeline, turns)):
        start, end = item["start_sec"], item["end_sec"]
        if not (isinstance(start, (int, float)) and isinstance(end, (int, float))
                and math.isfinite(start) and math.isfinite(end)
                and 0 <= start < end <= frames / sample_rate + .001):
            raise ValueError("Timeline outside audio")
        if (item["turn_id"] != index or turn.get("turn_id") != index
                or any(item[key] != turn[key] for key in ("speaker", "kind", "text", "addressee"))):
            raise ValueError("Timeline and annotation disagree")
    return annotation, paths


def _hf_token():
    """Reuse the project's credential order without printing credentials."""
    from huggingface_hub import get_token
    from scripts.config import load_env

    load_env()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    candidates = (Path.home() / "Desktop" / "pass-word" / "hf_token.txt",
                  Path.home() / "Desktop" / "hf_token.txt")
    for path in candidates:
        if not token and path.is_file():
            token = path.read_text(encoding="utf-8-sig").strip()
    token = token or get_token()
    if not token:
        raise RuntimeError("No authorized Hugging Face credential is available")
    return token


def _publish(root: Path, repo: str, ids: list[str], execute: bool = False) -> dict:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid HF dataset repo")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Supply unique explicit sample IDs")
    artifacts: dict[str, bytes] = {}
    annotations = []
    remaining = MAX_BATCH_BYTES
    for ident in ids:
        annotation, files = validate_delivery(root, ident, remaining)
        annotations.append(annotation)
        artifacts.update(files)
        remaining -= sum(map(len, files.values()))
    if not execute:
        return {"status": "validated_locally", "samples": len(ids), "files": len(artifacts),
                "repo": repo, "ids": ids}
    token = _hf_token()
    api = HfApi(token=token)
    head = api.repo_info(repo, repo_type="dataset").sha
    index_path = Path(hf_hub_download(repo, "annotations.jsonl", repo_type="dataset",
                                     revision=head, token=token))
    original = index_path.read_bytes()
    existing = {}
    for line in original.splitlines():
        row = json.loads(line)
        if row["dialogue_id"] in existing:
            raise ValueError("Remote index already contains duplicate IDs")
        existing[row["dialogue_id"]] = row
    additions = []
    for annotation in annotations:
        prior = existing.get(annotation["dialogue_id"])
        if prior is not None and prior != annotation:
            raise ValueError("Remote index contains a different annotation with this ID")
        if prior is None:
            additions.append(annotation)
    if additions:
        if original and not original.endswith(b"\n"):
            original += b"\n"
        original += b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                             for row in additions)
        artifacts["annotations.jsonl"] = original
    paths_info = api.get_paths_info(repo, list(artifacts), revision=head, repo_type="dataset")
    remote = {item.path: item for item in paths_info}
    ops = []
    for path, payload in artifacts.items():
        if path == "annotations.jsonl":
            ops.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=payload))
        elif path in remote:
            if not _matches_remote(remote[path], payload):
                raise ValueError(f"Refusing to overwrite existing artifact: {path}")
        else:
            ops.append(CommitOperationAdd(path_in_repo=path,
                       path_or_fileobj=payload))
    if ops:
        head = api.create_commit(repo, repo_type="dataset", parent_commit=head, operations=ops,
                                 commit_message=f"Add {len(ids)} validated duplex samples").oid
    verified = {item.path: item for item in
                api.get_paths_info(repo, list(artifacts), revision=head, repo_type="dataset")}
    for path, payload in artifacts.items():
        if path not in verified or not _matches_remote(verified[path], payload):
            raise RuntimeError(f"Remote verification failed: {path}")
    return {"status": "uploaded_and_verified", "samples": len(ids), "revision": head,
            "repo": repo, "ids": ids, "new_annotations": len(additions)}



def publish(root: Path, repo: str, ids: list[str], execute: bool = False) -> dict:
    """Publish via a sanitized boundary, including when called from Python."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        return _publish(root, repo, ids, execute)
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        suffix = f", HTTP {status}" if isinstance(status, int) else ""
        raise RuntimeError(f"Publication failed ({type(exc).__name__}{suffix})") from None
    finally:
        logging.disable(previous_disable)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--id", action="append", required=True, dest="ids")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    # Third-party warnings and exceptions may contain signed download/upload URLs.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        result = publish(args.root, args.repo, args.ids, args.execute)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
