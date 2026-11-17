"""Materialize a standard HF cache snapshot without following arbitrary links."""
from pathlib import Path
import os
import re
import shutil


def materialize_snapshot(source, destination):
    source, destination = Path(source).absolute(), Path(destination).absolute()
    try:
        if (source.parent.name != "snapshots" or not re.fullmatch(r"[a-f0-9]{40}", source.name)
                or not source.parent.parent.name.startswith("models--")):
            raise ValueError("Not a standard model cache snapshot")
        for parent in (source, *source.parents):
            if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
                raise ValueError("Linked snapshot directory")
        blobs = source.parent.parent / "blobs"
        if not blobs.is_dir() or blobs.is_symlink() or getattr(blobs, "is_junction", lambda: False)():
            raise ValueError("Invalid cache blobs directory")
        for parent in destination.parents:
            if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
                raise ValueError("Linked destination directory")
        if destination == source or source in destination.parents:
            raise ValueError("Destination is inside source snapshot")
        files = []
        from .model_identity import _PRIVATE_NAMES, _PRIVATE_EXTENSIONS
        for folder, dirs, names in os.walk(source, followlinks=False):
            for name in dirs + names:
                path = Path(folder) / name
                if (name.casefold() in _PRIVATE_NAMES or name.casefold().startswith(".env.")
                        or name.casefold().endswith(_PRIVATE_EXTENSIONS)
                        or getattr(path, "is_junction", lambda: False)()):
                    raise ValueError("Unsafe snapshot entry")
                if path.is_symlink():
                    target = path.resolve(strict=True)
                    if name in dirs or target.parent != blobs or not target.is_file():
                        raise ValueError("Snapshot link escapes model blobs")
                elif not path.is_file() and name in names:
                    raise ValueError("Invalid snapshot file")
                if name in names and path.stat().st_nlink != 1:
                    raise ValueError("Snapshot contains a hard-linked file")
                if name in names:
                    files.append((path, path.relative_to(source)))
        destination.mkdir(parents=True, exist_ok=False)
        for path, relative in files:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        return destination
    except (OSError, ValueError, RuntimeError):
        raise ValueError("Model snapshot materialization failed; check cache layout and destination") from None
