"""Append validated clean/babble deliveries to a Hugging Face dataset.

Credentials come from the Hugging Face SDK, never from command-line arguments.
The existing annotations file is preserved and updated with an optimistic
parent-commit guard. Repeating an already completed upload is safe.
Errors retain a fixed publication stage, exception class and optional HTTP status
without exposing upstream exception messages. Known transport retry markers remain.
Hash-pinned streaming batches may reconcile once after an unclassified commit
RuntimeError; reconciliation re-reads the remote head before considering a write.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import json
import logging
import math
import os
import re
import sys
import urllib.error
import warnings
from contextvars import ContextVar
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
_PUBLICATION_STAGE = ContextVar("publication_stage", default="validation")

_SENSITIVE_FIELD = re.compile(
    r"(?i)^(?:token|.*api[_-]?key|.*(?:password|secret|access[_-]?token|refresh[_-]?token|"
    r"hf[_-]?token|github[_-]?token|llm[_-]?api[_-]?key|credential|private[_-]?key|authorization))$"
)
_SECRET_TEXT = re.compile(
    r"(?i)(?:hf_[A-Za-z0-9]{20,}|(?:ghp_|gho_|ghs_|ghu_|ghr_|github_pat_)[A-Za-z0-9_]{20,}|"
    r"sk-[A-Za-z0-9_-]{16,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"(?<![A-Z0-9_])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9_])|"
    r"(?<![A-Z0-9_])bearer\s+[A-Za-z0-9._~+/=-]{8,}|"
    r"(?<![A-Z0-9_])(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|secret|authorization|token)\s*[:=]\s*\S{8,})"
)
_URL = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_PRIVATE_PUBLIC_TEXT = re.compile(
    r"(?i)(?:(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![A-Z0-9_-])|"
    r"(?<![A-Z0-9])(?:\+?86[- ]?)?1[3-9]\d(?:[- ]?\d){8}(?![A-Z0-9])|"
    r"(?<![A-Z0-9])[A-Z]:[\\/]|\\\\[A-Z0-9_.-]+[\\/]|"
    r"(?<![A-Z0-9.:/])/(?:home|Users|root)/[^\s/]+)"
)


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
        if _PRIVATE_PUBLIC_TEXT.search(value):
            raise ValueError("Possible private contact or local path in public annotation")
        for match in _URL.finditer(value):
            parsed = urlsplit(match.group(0))
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Public annotation URL contains private components")




def validate_delivery(root: Path, ident: str, max_bytes: int = MAX_BATCH_BYTES,
                      expected_receipt_sha256: str | None = None) -> tuple[dict, dict[str, bytes]]:
    import soundfile as sf
    import numpy as np
    from scripts.text.quality import audit
    from scripts.text.utils.schema import Dialogue

    if not re.fullmatch(r"[A-Za-z0-9_-]+", ident):
        raise ValueError("Invalid dialogue ID")
    root = root.resolve()
    receipt_dir = (root / "completed").resolve()
    receipt_path = (receipt_dir / f"{ident}.json").resolve()
    if not receipt_dir.is_relative_to(root) or receipt_path.parent != receipt_dir:
        raise ValueError("Receipt outside delivery root")
    receipt_bytes = receipt_path.read_bytes()
    if expected_receipt_sha256 is not None and hashlib.sha256(receipt_bytes).hexdigest() != expected_receipt_sha256:
        raise ValueError("Receipt changed before upload validation")
    receipt = json.loads(receipt_bytes)
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
        try:
            with sf.SoundFile(io.BytesIO(paths[f"audio/{name}"])) as audio:
                if (audio.frames, audio.samplerate, audio.channels) != (frames, sample_rate, 2):
                    raise ValueError("Clean/noisy audio format differs from the receipt")
                decoded = 0
                nonzero = False
                while decoded < frames:
                    block = audio.read(min(65536, frames - decoded), dtype="float32", always_2d=True)
                    if not len(block):
                        break
                    if not np.isfinite(block).all():
                        raise ValueError("Clean/noisy audio contains nonfinite samples")
                    nonzero = nonzero or bool(np.any(block != 0))
                    decoded += len(block)
                if decoded != frames:
                    raise ValueError("Clean/noisy audio ends before its claimed length")
                if not nonzero:
                    raise ValueError("Clean/noisy audio is entirely silent")
        except RuntimeError:
            raise ValueError("Clean/noisy audio cannot be fully decoded") from None
    timeline = json.loads(paths[f"audio/{names[2]}"])
    _assert_public_safe(timeline)
    turns = annotation["turns"]
    credited = receipt.get("credited_seconds")
    if (isinstance(credited, bool) or not isinstance(credited, (int, float)) or not math.isfinite(credited)
            or not 0 < credited <= frames / sample_rate + .001 or not turns):
        raise ValueError("Invalid credited duration")
    pauses = [turn.get("pause_ms", 0) or 0 for turn in turns]
    first_gap = turns[0].get("gap_before_ms", 0)
    if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in pauses + [first_gap])):
        raise ValueError("Invalid pause or initial gap")
    expected = frames / sample_rate - sum(pauses) / 1000 - first_gap / 1000
    if abs(credited - expected) > .01:
        raise ValueError("Credited duration disagrees with audio and pauses")
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
    """Reuse the project's credential resolver without printing credentials."""
    from scripts.production.publishing.credentials import hf_token
    return hf_token()


def _publish(root: Path, repo: str, ids: list[str], execute: bool = False,
             expected_receipt_hashes: dict[str, str] | None = None) -> dict:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid HF dataset repo")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Supply unique explicit sample IDs")
    if expected_receipt_hashes is not None and set(expected_receipt_hashes) != set(ids):
        raise ValueError("Receipt hashes must match the requested IDs")
    artifacts: dict[str, bytes] = {}
    annotations = []
    remaining = MAX_BATCH_BYTES
    for ident in ids:
        expected = expected_receipt_hashes[ident] if expected_receipt_hashes is not None else None
        annotation, files = validate_delivery(root, ident, remaining, expected)
        annotations.append(annotation)
        artifacts.update(files)
        remaining -= sum(map(len, files.values()))
    if not execute:
        return {"status": "validated_locally", "samples": len(ids), "files": len(artifacts),
                "repo": repo, "ids": ids}
    _PUBLICATION_STAGE.set("credentials")
    token = _hf_token()
    api = HfApi(token=token)
    _PUBLICATION_STAGE.set("repository")
    head = api.repo_info(repo, repo_type="dataset").sha
    _PUBLICATION_STAGE.set("index_metadata")
    index_info = api.get_paths_info(repo, ["annotations.jsonl"], revision=head, repo_type="dataset")
    if index_info:
        _PUBLICATION_STAGE.set("index_download")
        index_path = Path(hf_hub_download(repo, "annotations.jsonl", repo_type="dataset",
                                         revision=head, token=token))
        original = index_path.read_bytes()
    else:
        original = b""  # An initialized dataset may not have an annotation index yet.
    _PUBLICATION_STAGE.set("index_merge")
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
    _PUBLICATION_STAGE.set("artifact_metadata")
    paths_info = api.get_paths_info(repo, list(artifacts), revision=head, repo_type="dataset")
    remote = {item.path: item for item in paths_info}
    _PUBLICATION_STAGE.set("operation_build")
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
        _PUBLICATION_STAGE.set("commit")
        head = api.create_commit(repo, repo_type="dataset", parent_commit=head, operations=ops,
                                 commit_message=f"Add {len(ids)} validated duplex samples").oid
    _PUBLICATION_STAGE.set("verification_metadata")
    verified = {item.path: item for item in
                api.get_paths_info(repo, list(artifacts), revision=head, repo_type="dataset")}
    _PUBLICATION_STAGE.set("verification_content")
    for path, payload in artifacts.items():
        if path not in verified or not _matches_remote(verified[path], payload):
            raise RuntimeError(f"Remote verification failed: {path}")
    return {"status": "uploaded_and_verified", "samples": len(ids), "revision": head,
            "repo": repo, "ids": ids, "new_annotations": len(additions)}



def publish(root: Path, repo: str, ids: list[str], execute: bool = False,
            expected_receipt_hashes: dict[str, str] | None = None) -> dict:
    """Publish via a sanitized boundary, including when called from Python."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    previous_disable = logging.root.manager.disable
    stage_token = _PUBLICATION_STAGE.set("validation")
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frozen_ids = list(ids)
            frozen_hashes = dict(expected_receipt_hashes) if expected_receipt_hashes is not None else None
            for attempt in range(2):
                _PUBLICATION_STAGE.set("validation")
                try:
                    return _publish(root, repo, frozen_ids, execute, frozen_hashes)
                except RuntimeError as error:
                    if (attempt == 0 and execute and frozen_hashes is not None
                            and type(error) is RuntimeError and _PUBLICATION_STAGE.get() == "commit"
                            and getattr(error, "response", None) is None):
                        # Re-enter the ordinary idempotent path: validate frozen receipt
                        # hashes, query current head/index/artifacts, then only add missing
                        # content. A committed-but-lost response causes no second commit.
                        continue
                    raise
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        suffix = f", stage={_PUBLICATION_STAGE.get()}"
        suffix += f", HTTP {status}" if isinstance(status, int) else ""
        transient = isinstance(exc, (ConnectionError, TimeoutError, http.client.IncompleteRead, urllib.error.URLError))
        try:
            import httpx
            transient = transient or isinstance(exc, httpx.TransportError)
        except ImportError:
            pass
        transient = transient or isinstance(status, int) and (status in (408, 409, 425, 429) or status >= 500)
        if transient:
            suffix += ", retryable"
        raise RuntimeError(f"Publication failed ({type(exc).__name__}{suffix})") from None
    finally:
        _PUBLICATION_STAGE.reset(stage_token)
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
