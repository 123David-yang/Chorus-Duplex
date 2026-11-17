"""Install a small verified asset without overwriting or leaving a partial target."""
from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path


def _matches(path: Path, expected_sha256: str) -> bool:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == expected_sha256


def install_verified(url: str, destination: Path, expected_sha256: str, max_bytes: int) -> Path:
    destination = destination.expanduser()
    if destination.exists():
        if _matches(destination, expected_sha256):
            return destination
        raise FileExistsError("Destination exists with different contents")
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Downloaded asset failed its size or SHA-256 check")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".download-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        # A hard link atomically creates the target and fails if it already exists.
        # The temp file is in the same directory, so no cross-filesystem move occurs.
        try:
            try:
                os.link(temporary, destination)
            except OSError as exc:
                # Windows rename is also non-overwriting and works on volumes
                # such as exFAT that do not support hard links.
                if os.name == "nt" and getattr(exc, "winerror", None) in (1, 50, 87):
                    os.rename(temporary, destination)
                else:
                    raise
        except FileExistsError:
            if not _matches(destination, expected_sha256):
                raise
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
