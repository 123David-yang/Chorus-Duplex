"""Coordinate exact clean-audio hashes among the producers of one run."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path


def _owner(root: Path, ident: str) -> str:
    return hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest() + ":" + ident


def _payload(root: Path, ident: str, path: Path) -> tuple[str, str, str]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", ident):
        raise ValueError("Invalid delivery ID in deduplication index")
    row = json.loads(path.read_text(encoding="utf-8"))
    digest = row["sha256"].get(f"{ident}.flac")
    if row["annotation"]["dialogue_id"] != ident or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Invalid receipt in deduplication index")
    return digest, _owner(root, ident), ident


def _connect(path: Path):
    con = sqlite3.connect(path, timeout=30)
    con.execute("CREATE TABLE IF NOT EXISTS tracks (digest TEXT PRIMARY KEY, owner TEXT NOT NULL, ident TEXT NOT NULL)")
    return con


def rebuild(path: Path, roots: list[Path]) -> None:
    """Rebuild from durable receipts and pending claims while producers are stopped."""
    with closing(_connect(path)) as con, con:
        con.execute("DELETE FROM tracks")
        for root in roots:
            for receipt in sorted((root / "completed").glob("*.json")):
                digest, owner, ident = _payload(root, receipt.stem, receipt)
                try:
                    con.execute("INSERT INTO tracks VALUES (?, ?, ?)", (digest, owner, ident))
                except sqlite3.IntegrityError:
                    raise ValueError("Existing completed receipts contain duplicate clean audio; preserve and review them before resuming") from None
        for root in roots:
            for pending in sorted((root / "staging").glob("*/delivery.pending.json")):
                ident = pending.parent.name
                if (root / "duplicates" / f"{ident}.json").exists():
                    continue
                con.execute("INSERT OR IGNORE INTO tracks VALUES (?, ?, ?)", _payload(root, ident, pending))


def claim(path: Path, root: Path, ident: str, digest: str) -> str | None:
    """Atomically reserve a hash; return the earlier ID on an exact duplicate."""
    if not re.fullmatch(r"[a-f0-9]{64}", digest) or not re.fullmatch(r"[A-Za-z0-9_-]+", ident):
        raise ValueError("Invalid clean audio deduplication key")
    owner = _owner(root, ident)
    with closing(_connect(path)) as con, con:
        con.execute("INSERT OR IGNORE INTO tracks VALUES (?, ?, ?)", (digest, owner, ident))
        stored_owner, previous = con.execute("SELECT owner, ident FROM tracks WHERE digest = ?", (digest,)).fetchone()
    return None if stored_owner == owner else previous
