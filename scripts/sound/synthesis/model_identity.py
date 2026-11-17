"""Verify the identity of an offline, materialized Qwen3-TTS snapshot.

The caller must supply the *complete* file-to-SHA-256 inventory obtained from
a trusted source.  This module never downloads, loads or executes model code.
Every regular file in the directory is covered; extra files, empty files,
links and the absence of the known Qwen3-TTS 12Hz layout are rejected.  In
particular, a Hugging Face cache snapshot containing symlinks must first be
materialized as a separate directory of regular files.

The inventory is a trust boundary: matching it does not authenticate an
inventory assembled from the untrusted directory itself, prove model quality,
or prove that all possible Qwen versions use this layout.  File metadata is
checked before and after reading, but this is not an adversarial filesystem
isolation mechanism.  No absolute path or filename is returned in the identity
or included in the exceptions raised by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
from collections.abc import Mapping


class ModelIdentityError(ValueError):
    """A path-free error with a stable machine-readable reason code."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(f"Model snapshot verification failed: {code}")


_ROOT_FILES = frozenset({
    "config.json", "generation_config.json", "merges.txt",
    "preprocessor_config.json", "tokenizer_config.json", "vocab.json",
})
_SPEECH_FILES = frozenset({
    "speech_tokenizer/config.json", "speech_tokenizer/configuration.json",
    "speech_tokenizer/preprocessor_config.json",
})
_MAX_METADATA_BYTES = 32 * 1024 * 1024
_READ_BYTES = 1024 * 1024
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_PRIVATE_NAMES = frozenset({
    ".git", ".ssh", ".gnupg", ".aws", ".codex", ".netrc", ".npmrc",
    ".pypirc", ".gitconfig", ".git-credentials", ".env", "token",
    "token.txt", "credentials", "credentials.json", "authorized_keys",
    "id_rsa", "id_ed25519",
})
_PRIVATE_EXTENSIONS = (".pem", ".key", ".p12", ".pfx")
_SUPPORTED_INDEXES = frozenset({"model.safetensors.index.json",
                                 "speech_tokenizer/model.safetensors.index.json"})


def _fail(code: str) -> None:
    raise ModelIdentityError(code)


def _relative_name(value: object) -> str:
    """Allow only portable, unambiguous paths below the snapshot root."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        _fail("invalid_inventory_path")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        _fail("invalid_inventory_path")
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    parts = value.split("/")
    if (posix.is_absolute() or windows.is_absolute() or windows.drive
            or any(part in ("", ".", "..") or part.endswith((".", " "))
                   or PureWindowsPath(part).is_reserved()
                   or any(char in part for char in '<>"|?*') for part in parts)
            or posix.as_posix() != value):
        _fail("invalid_inventory_path")
    return value


def _inventory(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        _fail("trusted_inventory_required")
    result: dict[str, str] = {}
    folded: set[str] = set()
    for name, digest in value.items():
        name = _relative_name(name)
        if name.casefold() in folded:
            _fail("ambiguous_inventory_path")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            _fail("invalid_inventory_digest")
        folded.add(name.casefold())
        result[name] = digest.lower()
    return result


def _signature(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_mode, value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns,
            getattr(value, "st_file_attributes", 0), value.st_nlink)


def _handle_signature(value: os.stat_result) -> tuple[int, ...]:
    # Windows fstat may report the write time as ctime even when lstat reports
    # the creation time; lstat-to-lstat comparisons still include ctime.
    signature = _signature(value)
    return signature[:5] + signature[6:]


def _reject_link(value: os.stat_result) -> None:
    # Windows junctions and other reparse points are not always is_symlink().
    if (stat.S_ISLNK(value.st_mode)
            or getattr(value, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
        _fail("snapshot_contains_link")


def _scan(root: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Collect files and directories without intentionally following links."""
    root_stat = root.lstat()
    _reject_link(root_stat)
    if not stat.S_ISDIR(root_stat.st_mode):
        _fail("snapshot_not_directory")
    found = {"": ("dir", _signature(root_stat))}
    folded: set[str] = set()

    def walk(folder: Path, prefix: str) -> None:
        with os.scandir(folder) as iterator:
            children = sorted(iterator, key=lambda child: child.name)
        for child in children:
            name = f"{prefix}/{child.name}" if prefix else child.name
            _relative_name(name)
            part = child.name.casefold()
            if (part in _PRIVATE_NAMES or part.startswith(".env.")
                    or part.endswith(_PRIVATE_EXTENSIONS)):
                # Catch common credential locations before opening any contents.
                # Arbitrary names cannot be detected: use a dedicated snapshot.
                _fail("snapshot_contains_private_path")
            if name.casefold() in folded:
                _fail("ambiguous_snapshot_path")
            folded.add(name.casefold())
            # On Windows DirEntry.stat can report zero inode/device/link count;
            # Path.lstat obtains them and can be compared with an open handle.
            info = Path(child.path).lstat()
            _reject_link(info)
            if stat.S_ISDIR(info.st_mode):
                found[name] = ("dir", _signature(info))
                walk(Path(child.path), name)
            elif stat.S_ISREG(info.st_mode):
                if info.st_size <= 0:
                    _fail("snapshot_contains_empty_file")
                # A hard link can expose a file whose other name is outside the
                # snapshot.  Require materialized, separately owned files.
                if info.st_nlink != 1:
                    _fail("snapshot_contains_hard_link")
                found[name] = ("file", _signature(info))
            else:
                _fail("snapshot_contains_nonregular_file")

    walk(root, "")
    return found


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("invalid_metadata")
        result[key] = value
    return result


def _json_object(data: bytes) -> dict:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_no_duplicate_keys,
                           parse_constant=lambda _: _fail("invalid_metadata"))
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        value = None
    if not isinstance(value, dict):
        _fail("invalid_metadata")
    return value


def _tensor_header(data: bytes, payload_size: int) -> set[str]:
    header = _json_object(data)
    spans: list[tuple[int, int]] = []
    names: set[str] = set()
    for name, item in header.items():
        if name == "__metadata__":
            if not isinstance(item, dict):
                _fail("invalid_safetensors")
            continue
        if not isinstance(name, str) or not name or not isinstance(item, dict):
            _fail("invalid_safetensors")
        offsets, shape, dtype = item.get("data_offsets"), item.get("shape"), item.get("dtype")
        if (not isinstance(dtype, str) or not dtype or not isinstance(shape, list)
                or any(type(dim) is not int or dim < 0 for dim in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)):
            _fail("invalid_safetensors")
        start, end = offsets
        if not 0 <= start <= end <= payload_size:
            _fail("invalid_safetensors")
        spans.append((start, end))
        names.add(name)
    if not names:
        _fail("invalid_safetensors")
    position = 0
    for start, end in sorted(spans):
        if start != position:
            _fail("invalid_safetensors")
        position = end
    if position != payload_size:
        _fail("invalid_safetensors")
    return names


def _hash_file(path: Path, expected: tuple[int, ...], capture_json: bool) -> tuple[str, bytes | None, set[str] | None]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        if _handle_signature(os.fstat(fd)) != expected[:5] + expected[6:]:
            _fail("snapshot_changed")
        digest = hashlib.sha256()
        data = bytearray() if capture_json else None
        tensor_names = None
        with os.fdopen(fd, "rb", closefd=False) as stream:
            if path.name.endswith(".safetensors"):
                prefix = stream.read(8)
                if len(prefix) != 8:
                    _fail("invalid_safetensors")
                header_size = int.from_bytes(prefix, "little")
                if header_size <= 0 or header_size > _MAX_METADATA_BYTES or 8 + header_size >= expected[3]:
                    _fail("invalid_safetensors")
                header = stream.read(header_size)
                if len(header) != header_size:
                    _fail("snapshot_changed")
                tensor_names = _tensor_header(header, expected[3] - 8 - header_size)
                digest.update(prefix)
                digest.update(header)
            if data is not None and expected[3] > _MAX_METADATA_BYTES:
                _fail("metadata_too_large")
            while block := stream.read(_READ_BYTES):
                digest.update(block)
                if data is not None:
                    data.extend(block)
        if _handle_signature(os.fstat(fd)) != expected[:5] + expected[6:]:
            _fail("snapshot_changed")
        current = path.lstat()
        _reject_link(current)
        if _signature(current) != expected:
            _fail("snapshot_changed")
        return digest.hexdigest(), bytes(data) if data is not None else None, tensor_names
    finally:
        os.close(fd)


def _validate_layout(files: set[str], metadata: dict[str, dict], tensors: dict[str, set[str]]) -> str:
    if not (_ROOT_FILES | _SPEECH_FILES) <= files:
        _fail("incomplete_qwen_layout")
    config = metadata["config.json"]
    if (config.get("model_type") != "qwen3_tts"
            or config.get("tts_model_type") not in ("base", "custom_voice")):
        _fail("unsupported_qwen_model")
    if metadata["speech_tokenizer/config.json"].get("model_type") != "qwen3_tts_tokenizer_12hz":
        _fail("unsupported_speech_tokenizer")
    for prefix in ("", "speech_tokenizer/"):
        standalone = prefix + "model.safetensors"
        index = prefix + "model.safetensors.index.json"
        if standalone not in files and index not in files:
            _fail("missing_model_weights")
        if index in files:
            mapping = metadata[index].get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                _fail("invalid_weight_index")
            expected_tensors: dict[str, set[str]] = {}
            for tensor, shard in mapping.items():
                if not isinstance(tensor, str) or not tensor:
                    _fail("invalid_weight_index")
                shard = _relative_name(shard)
                if not shard.endswith(".safetensors"):
                    _fail("invalid_weight_index")
                full = prefix + shard
                if full not in files or full not in tensors:
                    _fail("missing_model_shard")
                expected_tensors.setdefault(full, set()).add(tensor)
            if any(tensors[path] != names for path, names in expected_tensors.items()):
                _fail("invalid_weight_index")
    return config["tts_model_type"]


def _qwen_snapshot_identity(root: str | os.PathLike[str], *, expected_files: Mapping[str, str] | None = None,
                           expected_sha256: str | None = None,
                           expected_model_type: str | None = None, _content_only=False) -> dict[str, str | int]:
    """Verify a snapshot against a caller-trusted, complete relative-path inventory.

    ``expected_files`` maps POSIX-style relative file paths to SHA-256 hex.
    It must come from a trusted fixed revision or a separately verified local
    export; deriving it from this same untrusted directory proves no origin.
    Every regular file must match exactly, and known Qwen 12Hz layout and
    safetensors shard structure must pass. ``expected_sha256``, when supplied,
    additionally pins the canonical tree digest from a previous verified run.
    ``expected_model_type`` can require "base" or "custom_voice" so that a
    clone model cannot accidentally be bound to the preset role, or vice versa.

    This deliberately supports the known Qwen3-TTS 12Hz Base/CustomVoice layout
    only.  It can reject other valid layouts.  It does not load a model, validate
    tensor shapes against architecture, authenticate the inventory's origin,
    check inference quality, or prevent an adversary modifying the filesystem.
    """
    try:
        inventory = None if _content_only else _inventory(expected_files)
        if expected_model_type is not None and expected_model_type not in ("base", "custom_voice"):
            _fail("invalid_expected_model_type")
        if expected_sha256 is not None:
            if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
                _fail("invalid_expected_identity")
            expected_sha256 = expected_sha256.lower()
        raw = os.fspath(root)
        if not isinstance(raw, str) or not raw or "\x00" in raw or ".." in Path(raw).parts:
            _fail("invalid_snapshot_path")
        folder = Path(raw).absolute()
        # Refuse explicit links in the path to the root as well as inside it.
        for ancestor in (folder, *folder.parents):
            _reject_link(ancestor.lstat())
        before = _scan(folder)
        files = {name for name, (kind, _) in before.items() if kind == "file"}
        if inventory is not None and files != set(inventory):
            _fail("inventory_file_set_mismatch")
        if not (_ROOT_FILES | _SPEECH_FILES) <= files:
            _fail("incomplete_qwen_layout")
        for prefix in ("", "speech_tokenizer/"):
            if not ({prefix + "model.safetensors", prefix + "model.safetensors.index.json"} & files):
                _fail("missing_model_weights")
        if any(name.endswith(".safetensors.index.json") and name not in _SUPPORTED_INDEXES
               for name in files):
            _fail("unsupported_weight_index")
        metadata_names = (_ROOT_FILES | _SPEECH_FILES) - {"merges.txt"}
        metadata_names |= {name for name in files if name.endswith(".safetensors.index.json")}
        metadata: dict[str, dict] = {}
        tensors: dict[str, set[str]] = {}
        rows: list[tuple[str, int, str]] = []
        priority = {"config.json": 0, "speech_tokenizer/config.json": 1}
        for name in sorted(files, key=lambda item: (priority.get(item, 2), item)):
            expected = before[name][1]
            digest, data, tensor_names = _hash_file(folder.joinpath(*name.split("/")), expected,
                                                    name in metadata_names)
            if inventory is not None and digest != inventory[name]:
                _fail("inventory_digest_mismatch")
            if data is not None:
                metadata[name] = _json_object(data)
                if name == "config.json":
                    model_config = metadata[name]
                    if (model_config.get("model_type") != "qwen3_tts"
                            or model_config.get("tts_model_type") not in ("base", "custom_voice")):
                        _fail("unsupported_qwen_model")
                    if (expected_model_type is not None
                            and model_config.get("tts_model_type") != expected_model_type):
                        _fail("snapshot_model_type_mismatch")
                elif (name == "speech_tokenizer/config.json"
                      and metadata[name].get("model_type") != "qwen3_tts_tokenizer_12hz"):
                    _fail("unsupported_speech_tokenizer")
            if tensor_names is not None:
                tensors[name] = tensor_names
            rows.append((name, expected[3], digest))
        if _scan(folder) != before:
            _fail("snapshot_changed")
        model_type = _validate_layout(files, metadata, tensors)
        if expected_model_type is not None and model_type != expected_model_type:
            _fail("snapshot_model_type_mismatch")
        rows.sort()
        payload = json.dumps({"algorithm": "qwen3-snapshot-sha256-v1", "files": rows},
                             ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
        tree = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and tree != expected_sha256:
            _fail("snapshot_identity_mismatch")
        return {"algorithm": "qwen3-snapshot-sha256-v1", "sha256": tree,
                "verification": "content-only" if _content_only else "external-inventory",
                "tts_model_type": model_type, "file_count": len(rows),
                "total_bytes": sum(size for _, size, _ in rows)}
    except ModelIdentityError:
        raise
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
        pass
    # Raise outside the handler: even exception-context inspection cannot leak
    # an operating-system error containing the absolute path.
    raise ModelIdentityError("snapshot_unreadable")


def qwen_snapshot_identity(root, *, expected_files=None, expected_sha256=None, expected_model_type=None):
    """Verify materialized files against a caller-trusted complete inventory."""
    return _qwen_snapshot_identity(root, expected_files=expected_files,
                                   expected_sha256=expected_sha256, expected_model_type=expected_model_type)


def qwen_content_identity(root, *, expected_sha256=None, expected_model_type=None):
    """Fingerprint a materialized model for continuity, without authenticating its source.

    First-run hashes provide a baseline only. Links remain prohibited; HF cache
    snapshots must be materialized before using this function.
    """
    return _qwen_snapshot_identity(root, expected_sha256=expected_sha256,
                                   expected_model_type=expected_model_type, _content_only=True)
