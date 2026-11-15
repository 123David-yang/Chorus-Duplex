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
from pathlib import Path
from functools import lru_cache
import zlib

ASSISTANT_VOICE = "Junhao"  # 男声，固定
_USER_POOL = ["Zhiming", "Weiguo", "Xiaoyu", "Yuewen", "Lingyu"]  # 2 男 3 女
_GENDER = {"Junhao": "男声", "Zhiming": "男声", "Weiguo": "男声",
           "Xiaoyu": "女声", "Yuewen": "女声", "Lingyu": "女声"}


def voices_for_dialogue(dialogue_id: str, n_users: int) -> dict[str, str]:
    rng = random.Random(zlib.crc32(f"voices-{dialogue_id}".encode()))
    if os.environ.get("TTS_BACKEND")=="kokoro":
        pool=[f"kokoro_{i:03d}" for i in range(3,103) if i!=58]
        if os.environ.get("KOKORO_VOICE_POOL"):
            pool=[f"kokoro_{int(v):03d}" for v in os.environ["KOKORO_VOICE_POOL"].split(",")]
            if len(set(pool))!=len(pool) or any(not 3<=int(v.split("_")[1])<=102 or v=="kokoro_058" for v in pool):raise ValueError("Invalid calibrated Kokoro voice pool")
            if len(pool)<n_users+1:raise ValueError("Calibrated voice pool too small for users plus third party")
        rng.shuffle(pool)
        mapping={"assistant":"kokoro_058"}
        for i in range(n_users):mapping[f"user_{i+1:02d}"]=pool[i]
        remaining=pool[n_users:]
        for i in range(6):mapping[f"third_party_{i+1:02d}"]=remaining[i%len(remaining)]
        return mapping
    extra = reference_voices()
    reference_only=os.environ.get("TTS_BACKEND")=="qwen3"
    if reference_only:
        if len(extra)<n_users+2: raise ValueError("Qwen needs at least users+2 reference voices")
        assistant=next(iter(extra))
        pool=[v for v in extra if v!=assistant]
    else:
        assistant=ASSISTANT_VOICE
        pool=_USER_POOL[:] + list(extra)
        if os.environ.get("MOSS_VOICE_POOL"):
            pool=[v.strip() for v in os.environ["MOSS_VOICE_POOL"].split(",") if v.strip()]
            if len(set(pool))!=len(pool) or any(v not in _USER_POOL and v not in extra for v in pool):raise ValueError("Invalid calibrated MOSS voice pool")
            if len(pool)<n_users+1:raise ValueError("MOSS voice pool needs distinct users and third party")
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
    return _USER_POOL[zlib.crc32(f"fb-{speaker}".encode()) % len(_USER_POOL)]


def gender_of(voice: str) -> str:
    if voice.startswith("kokoro_"):
        sid=int(voice.split("_",1)[1])
        return "女声" if 3<=sid<=57 else "男声" if 58<=sid<=102 else "未知"
    return reference_voices().get(voice, {}).get("gender", _GENDER.get(voice, "未知"))


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
        if not reference.is_file(): raise FileNotFoundError(reference)
        if not row.get("license") or not row.get("source_url"): raise ValueError("Voice provenance missing")
        out[row["id"]]={**row,"reference_audio":str(reference)}
    return out

def reference_voices() -> dict:
    path=os.environ.get("VOICE_MANIFEST")
    return _load_reference_manifest(path,Path(path).stat().st_mtime_ns) if path else {}

def reference_for_voice(voice: str) -> str | None:
    if voice.startswith("kokoro_"):return None
    row=reference_voices().get(voice)
    if row: return row["reference_audio"]
    # Preset IDs outside the Chinese default pool belong to the selected backend.
    # Let that backend validate its own presets; custom backends may use numeric/string IDs.
    return None
