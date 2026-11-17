"""主角团/第三方音色分配：按 dialogue_id 确定性抽样，中文音色池（6 个）。

- assistant 固定 Junhao（AI 同一角色的跨对话身份一致）
- 主角团用户：每条对话从余下 5 个中文音色中错开选（对话内一致、跨对话有变化 → 音色多样性）；
  **超过 5 人时音色池循环复用**（中文内置音色共 6 个、assistant 已占 1 个，2026-09-16 人数扩到 1~6
  后的已知限制；音色克隆扩容后解决）
- 第三方：从剩余音色里错开分配（全中文音色；此前误用的英文音色已弃）
提示词与音频层共用本模块 → 音色表与人设性别自动对齐。
"""
from __future__ import annotations

import random
import os
import json
import hashlib
from pathlib import Path
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit
import zlib

ASSISTANT_VOICE = "Junhao"  # 男声，固定
_USER_POOL = ["Zhiming", "Weiguo", "Xiaoyu", "Yuewen", "Lingyu"]  # 2 男 3 女
_GENDER = {"Junhao": "男声", "Zhiming": "男声", "Weiguo": "男声",
           "Xiaoyu": "女声", "Yuewen": "女声", "Lingyu": "女声"}
_QWEN_GENDER = {"Ryan": "男声", "Aiden": "男声", "Uncle_Fu": "男声",
                "Dylan": "男声", "Eric": "男声", "Vivian": "女声",
                "Serena": "女声", "Ono_Anna": "女声", "Sohee": "女声"}


@lru_cache(maxsize=256)
def _check_reference_audio(path: str, size: int, modified_ns: int) -> None:
    """Validate the complete reference once per observed file version."""
    import numpy as np
    import soundfile as sf
    try:
        with sf.SoundFile(path) as audio:
            if audio.samplerate <= 0 or audio.channels <= 0 or audio.frames <= 0:
                raise ValueError("Invalid reference audio format")
            expected = audio.frames
            frames = 0
            while frames < expected:
                block = audio.read(min(65536, expected - frames), dtype="float32", always_2d=True)
                if not len(block):
                    break
                if not np.all(np.isfinite(block)):
                    raise ValueError("Invalid reference samples")
                frames += len(block)
            if frames != expected:
                raise ValueError("Incomplete reference audio")
    except (OSError, RuntimeError, ValueError):
        raise ValueError("Reference audio cannot be fully decoded") from None


def _validate_reference(row: dict) -> None:
    path = Path(row["reference_audio"])
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = row.get("sha256")
    if expected:
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual.lower() != expected.lower():
            raise ValueError(f"Reference audio checksum mismatch for {row['id']}")
    stat = path.stat()
    _check_reference_audio(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _public_source_url(value: str | None) -> str | None:
    """Exclude userinfo and transient query credentials from published provenance."""
    if not value:
        return None
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        return None
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    if parsed.port is not None:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme.lower(), host, parsed.path, "", ""))


@lru_cache(maxsize=8)
def _load_voice_registry(path: str, modified_ns: int, backend: str) -> dict:
    registry_path = Path(path)
    raw = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    if raw.get("schema_version") != 1 or not isinstance(raw.get("backends"), dict):
        raise ValueError("Voice registry must have schema_version 1 and backends")
    config = raw["backends"].get(backend)
    if config is None:
        raise ValueError(f"Voice registry has no configuration for backend {backend}")
    entries = {}
    for item in config["voices"]:
        row = dict(item)
        voice_id = row.get("id")
        if not isinstance(voice_id, str) or not voice_id or voice_id in entries:
            raise ValueError(f"Invalid or duplicate voice ID in {backend}")
        kind = row.get("kind")
        if kind == "preset":
            row["speaker"] = row.get("speaker", voice_id)
            if not isinstance(row["speaker"], str) or not row["speaker"]:
                raise ValueError(f"Invalid preset speaker for {voice_id}")
        elif kind == "reference":
            reference = Path(row["reference_audio"]).expanduser()
            if not reference.is_absolute():
                reference = registry_path.parent / reference
            row["reference_audio"] = str(reference.resolve())
            _validate_reference(row)
            if not row.get("sha256"):
                with reference.open("rb") as stream:
                    row["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            row["source_url"] = _public_source_url(row.get("source_url"))
            row.setdefault("license", None)
        else:
            raise ValueError(f"Voice {voice_id} must be preset or reference")
        entries[voice_id] = row
    assistant = config.get("assistant")
    if assistant not in entries:
        raise ValueError(f"Backend {backend} needs a registered assistant voice")
    if entries[assistant]["kind"] != "preset":
        raise ValueError(f"Backend {backend} assistant must use a preset voice")
    assistant_speaker = entries[assistant]["speaker"].casefold()
    if any(key != assistant and row["kind"] == "preset"
           and row["speaker"].casefold() == assistant_speaker
           for key, row in entries.items()):
        raise ValueError(f"Backend {backend} assistant preset must be distinct from participant presets")
    if len(entries) < 2:
        raise ValueError(f"Backend {backend} needs at least one participant voice")
    if backend == "kokoro" and any(row["kind"] == "reference" for row in entries.values()):
        raise ValueError("Kokoro does not support reference voice cloning")
    return {"assistant": assistant, "voices": entries}


def voice_registry() -> dict | None:
    path = os.environ.get("VOICE_REGISTRY")
    if not path:
        return None
    file = Path(path).expanduser()
    backend = os.environ.get("TTS_BACKEND", "moss_onnx")
    return _load_voice_registry(str(file.resolve()), file.stat().st_mtime_ns, backend)


def preset_for_voice(voice: str) -> str:
    registry = voice_registry()
    if registry is None:
        return voice
    row = registry["voices"].get(voice)
    if row is None:
        raise ValueError(f"Voice {voice} is not registered for this backend")
    return row["speaker"] if row["kind"] == "preset" else voice


def voices_for_dialogue(dialogue_id: str, n_users: int) -> dict[str, str]:
    rng = random.Random(zlib.crc32(f"voices-{dialogue_id}".encode()))
    registry = voice_registry()
    if registry is not None:
        assistant = registry["assistant"]
        pool = [key for key in registry["voices"] if key != assistant]
        rng.shuffle(pool)
        mapping = {"assistant": assistant}
        for i in range(n_users):
            mapping[f"user_{i + 1:02d}"] = pool[i % len(pool)]
        remaining = [voice for voice in pool if voice not in mapping.values()] or pool
        for i in range(6):
            mapping[f"third_party_{i + 1:02d}"] = remaining[i % len(remaining)]
        return mapping
    backend = os.environ.get("TTS_BACKEND", "moss_onnx")
    if backend not in ("moss_onnx", "qwen3", "kokoro"):
        raise ValueError("Custom TTS backends require VOICE_REGISTRY")
    if os.environ.get("TTS_BACKEND")=="kokoro":
        pool=[f"kokoro_{i:03d}" for i in range(3,103) if i!=58]
        if os.environ.get("KOKORO_VOICE_POOL"):
            pool=[f"kokoro_{int(v):03d}" for v in os.environ["KOKORO_VOICE_POOL"].split(",")]
            if len(set(pool))!=len(pool) or any(not 3<=int(v.split("_")[1])<=102 or v=="kokoro_058" for v in pool):raise ValueError("Invalid calibrated Kokoro voice pool")
            if len(pool)<n_users:raise ValueError("Calibrated voice pool too small for users")
        rng.shuffle(pool)
        mapping={"assistant":"kokoro_058"}
        for i in range(n_users):mapping[f"user_{i+1:02d}"]=pool[i]
        remaining=pool[n_users:] or pool
        for i in range(6):mapping[f"third_party_{i+1:02d}"]=remaining[i%len(remaining)]
        return mapping
    extra = reference_voices()
    reference_only=os.environ.get("TTS_BACKEND")=="qwen3"
    if reference_only:
        if set(extra) & set(_QWEN_GENDER):
            raise ValueError("Qwen reference IDs cannot duplicate preset voice IDs")
        assistant = "Ryan"
        pool = [voice for voice in _QWEN_GENDER if voice != assistant] + list(extra)
    else:
        assistant=ASSISTANT_VOICE
        pool=_USER_POOL[:] + list(extra)
        if os.environ.get("MOSS_VOICE_POOL"):
            pool=[v.strip() for v in os.environ["MOSS_VOICE_POOL"].split(",") if v.strip()]
            if len(set(pool))!=len(pool) or any(v not in _USER_POOL and v not in extra for v in pool):raise ValueError("Invalid calibrated MOSS voice pool")
            if len(pool)<n_users:raise ValueError("MOSS voice pool needs enough distinct users")
    rng.shuffle(pool)
    out = {"assistant": assistant}
    for i in range(n_users):
        out[f"user_{i + 1:02d}"] = pool[i % len(pool)]
    rest = [v for v in pool if v not in out.values()]
    for k in range(1, 7):  # 预置 6 个第三方槽位（校验器限第三方发声 ≤3 轮，留裕量防 LLM 跳号）
        out[f"third_party_{k:02d}"] = rest[(k - 1) % len(rest)] if rest else pool[k % len(pool)]
    return out


def fallback_voice(speaker: str) -> str:
    """字典外 speaker（如 LLM 写出 third_party_09）的兜底音色：从用户池按名字确定性选。

    **绝不回落到 assistant 的 Junhao**——保证"assistant 音色跨对话唯一"不被破坏。
    """
    registry = voice_registry()
    if registry is not None or os.environ.get("TTS_BACKEND") in ("kokoro", "qwen3"):
        return voices_for_dialogue(speaker, 1)["user_01"]
    return _USER_POOL[zlib.crc32(f"fb-{speaker}".encode()) % len(_USER_POOL)]


def gender_of(voice: str) -> str:
    registry = voice_registry()
    if registry is not None:
        return registry["voices"].get(voice, {}).get("gender", "未知")
    if os.environ.get("TTS_BACKEND") == "kokoro" and voice.startswith("kokoro_"):
        sid=int(voice.split("_",1)[1])
        return "女声" if 3<=sid<=57 else "男声" if 58<=sid<=102 else "未知"
    return reference_voices().get(voice, {}).get("gender", _GENDER.get(voice, _QWEN_GENDER.get(voice, "未知")))


def voice_table_text(dialogue_id: str, n_users: int) -> str:
    """给提示词的音色表（只讲性别，人设必须对齐）。"""
    v = voices_for_dialogue(dialogue_id, n_users)
    users = "、".join(f"user_{i + 1:02d} = {gender_of(v[f'user_{i + 1:02d}'])}"
                      for i in range(n_users))
    return f"{users}；assistant = {gender_of(v['assistant'])}"


@lru_cache(maxsize=8)
def _load_reference_manifest(path: str, modified_ns: int) -> dict:
    manifest_path=Path(path)
    raw=json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if raw.get("schema_version")!=1: raise ValueError("Unsupported voice manifest version")
    out={}
    for row in raw["voices"]:
        if row["id"] in _GENDER or row["id"] in out: raise ValueError("Duplicate voice ID")
        reference=Path(row["reference_audio"])
        if not reference.is_absolute(): reference=manifest_path.parent/reference
        if not row.get("license") or not row.get("source_url"): raise ValueError("Voice provenance missing")
        out[row["id"]]={**row,"reference_audio":str(reference),
                        "source_url":_public_source_url(row["source_url"])}
        _validate_reference(out[row["id"]])
    return out

def reference_voices() -> dict:
    registry = voice_registry()
    if registry is not None:
        return {key: row for key, row in registry["voices"].items() if row["kind"] == "reference"}
    path=os.environ.get("VOICE_MANIFEST")
    return _load_reference_manifest(path,Path(path).stat().st_mtime_ns) if path else {}

def reference_for_voice(voice: str) -> str | None:
    registry = voice_registry()
    if registry is not None:
        row = registry["voices"].get(voice)
        if row is None:
            raise ValueError(f"Voice {voice} is not registered for this backend")
        if row["kind"] == "reference":
            _validate_reference(row)
            return row["reference_audio"]
        return None
    if os.environ.get("TTS_BACKEND") == "kokoro" and voice.startswith("kokoro_"):
        return None
    row=reference_voices().get(voice)
    if row:
        _validate_reference(row)
        return row["reference_audio"]
    # Preset IDs outside the Chinese default pool belong to the selected backend.
    # Let that backend validate its own presets; custom backends may use numeric/string IDs.
    return None


def voice_bindings_snapshot(voice_ids) -> dict:
    """Freeze the actual speaker or reference digest behind each script voice ID."""
    registry = voice_registry()
    references = reference_voices()
    bindings = {}
    for voice in sorted(set(voice_ids)):
        if registry is not None:
            row = registry["voices"].get(voice)
            if row is None:
                raise ValueError(f"Voice {voice} is not registered for this backend")
            if row["kind"] == "preset":
                bindings[voice] = {"kind": "preset", "speaker": row["speaker"]}
                continue
        else:
            row = references.get(voice)
            if row is None:
                bindings[voice] = {"kind": "preset", "speaker": voice}
                continue
        _validate_reference(row)
        with Path(row["reference_audio"]).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        bindings[voice] = {"kind": "reference", "sha256": digest}
    return bindings
