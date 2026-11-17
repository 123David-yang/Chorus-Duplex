"""Upload completed deliveries in batches while producers continue running.

Receipt hashes and remote revisions are persisted only after publication and
remote verification. A retry after an interrupted commit is safe because the
underlying publisher rejects changed annotations or artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

from scripts.production.state.lock import ProductionLock
from .storage import atomic_json
from .upload_delivery import MAX_BATCH_BYTES, publish


def _receipt_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pending_receipts(roots: list[Path], state: Path, repo: str) -> list[dict]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid HF dataset repo")
    pending = []
    clean_hashes = set()
    repo_key = hashlib.sha256(repo.encode("utf-8")).hexdigest()[:16]
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError("Production root does not exist")
        marker_root = state / "uploaded" / repo_key / hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
        for receipt in sorted((root / "completed").glob("*.json")):
            ident = receipt.stem
            if not re.fullmatch(r"[A-Za-z0-9_-]+", ident):
                raise ValueError("Invalid completed receipt ID")
            raw = receipt.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            row = json.loads(raw)
            if row["annotation"]["dialogue_id"] != ident:
                raise ValueError("Receipt ID mismatch")
            clean_digest = row.get("sha256", {}).get(f"{ident}.flac")
            if not isinstance(clean_digest, str) or not re.fullmatch(r"[a-f0-9]{64}", clean_digest):
                raise ValueError("Invalid clean audio digest")
            if clean_digest in clean_hashes:
                raise ValueError("Duplicate clean audio across production receipts")
            clean_hashes.add(clean_digest)
            seconds = row["credited_seconds"]
            frames, sample_rate = row.get("frames"), row.get("sample_rate")
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("Invalid credited duration")
            if (type(frames) is not int or type(sample_rate) is not int or frames <= 0 or sample_rate <= 0
                    or seconds > frames / sample_rate + .001):
                raise ValueError("Credited duration exceeds the audio duration")
            marker = marker_root / f"{ident}.json"
            if marker.exists():
                saved = json.loads(marker.read_text(encoding="utf-8"))
                if saved.get("receipt_sha256") != digest or saved.get("dialogue_id") != ident or saved.get("repo") != repo:
                    raise ValueError("An already uploaded receipt has changed")
                continue
            names = (f"{ident}.flac", f"{ident}_babble.flac", f"{ident}.timeline.json")
            size = sum((root / "audio" / name).stat().st_size for name in names)
            if size > MAX_BATCH_BYTES:
                raise ValueError("A delivery exceeds the upload memory limit")
            pending.append({"root": root, "id": ident, "receipt": receipt, "digest": digest,
                            "marker": marker, "size": size, "seconds": seconds})
    return pending


def upload_batch(items: list[dict], repo: str, execute: bool) -> dict:
    if not items or any(item["root"] != items[0]["root"] for item in items):
        raise ValueError("A batch must contain one or more receipts from the same root")
    result = publish(items[0]["root"], repo, [item["id"] for item in items], execute,
                     expected_receipt_hashes={item["id"]: item["digest"] for item in items})
    if execute:
        if result.get("status") != "uploaded_and_verified" or not result.get("revision"):
            raise RuntimeError("Publisher did not verify the remote commit")
        for item in items:
            if _receipt_hash(item["receipt"]) != item["digest"]:
                raise ValueError("Receipt changed during upload")
        for item in items:
            item["marker"].parent.mkdir(parents=True, exist_ok=True)
            atomic_json(item["marker"], {"dialogue_id": item["id"], "receipt_sha256": item["digest"],
                                        "repo": repo, "revision": result["revision"],
                                        "credited_seconds": item["seconds"]})
    return result


def next_batch(pending: list[dict], batch_size: int, last_root: Path | None = None) -> list[dict]:
    if not pending:
        return []
    roots = list(dict.fromkeys(item["root"] for item in pending))
    root = roots[(roots.index(last_root) + 1) % len(roots)] if last_root in roots else roots[0]
    selected = []
    size = 0
    for item in pending:
        if item["root"] != root:
            continue
        if len(selected) >= batch_size or size + item["size"] > MAX_BATCH_BYTES:
            break
        selected.append(item)
        size += item["size"]
    return selected


def _retryable(error: RuntimeError) -> bool:
    return ", retryable)" in str(error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True, help="production root; may be repeated")
    parser.add_argument("--state-dir", type=Path, required=True, help="durable local upload state")
    parser.add_argument("--repo", required=True, help="Hugging Face dataset owner/name")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--flush-seconds", type=float, default=90)
    parser.add_argument("--done-file", type=Path, help="producer creates this after all producers exit")
    parser.add_argument("--follow", action="store_true", help="wait for new receipts until --done-file exists")
    parser.add_argument("--execute", action="store_true", help="publish; omitted validates one local batch only")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 100 or not 1 <= args.poll_seconds <= 300 or not 1 <= args.flush_seconds <= 3600:
        parser.error("Invalid batch or polling limits")
    if args.follow and (not args.done_file or not args.execute):
        parser.error("--follow requires --done-file and --execute")
    if len({root.resolve() for root in args.root}) != len(args.root):
        parser.error("Production roots must be distinct")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with ProductionLock(args.state_dir):
        first_pending = None
        retries = 0
        last_root = None
        while True:
            atomic_json(args.state_dir / "progress.json", {"stage": "scanning", "timestamp": time.time()})
            pending = pending_receipts(args.root, args.state_dir, args.repo)
            done = bool(args.done_file and args.done_file.exists())
            if not pending:
                first_pending = None
                if not args.follow or done:
                    print(json.dumps({"status": "complete", "pending": 0}), flush=True)
                    return
            else:
                first_pending = first_pending or time.monotonic()
                batch = next_batch(pending, args.batch_size, last_root)
                ready = not args.follow or done or len(pending) >= args.batch_size or time.monotonic() - first_pending >= args.flush_seconds
                if ready:
                    try:
                        atomic_json(args.state_dir / "progress.json", {"stage": "uploading", "timestamp": time.time(),
                                                                      "samples": len(batch)})
                        result = upload_batch(batch, args.repo, args.execute)
                    except RuntimeError as exc:
                        retries += 1
                        if not _retryable(exc) or retries > 5:
                            raise
                        print(json.dumps({"status": "retrying", "attempt": retries}), flush=True)
                        atomic_json(args.state_dir / "progress.json", {"stage": "retrying", "timestamp": time.time(),
                                                                      "attempt": retries})
                        time.sleep(min(60, 2 ** retries))
                        continue
                    print(json.dumps({"status": result["status"], "samples": len(batch),
                                      "verified_seconds": round(sum(item["seconds"] for item in batch), 2)}), flush=True)
                    if not args.follow:
                        return
                    last_root = batch[0]["root"]
                    retries = 0
                    first_pending = None
                    continue
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
