"""集中配置（config.py）—— 全项目路径 / 外部依赖 / 数据化配置的唯一来源，消灭硬编码。

用法：
    from scripts.config import ROOT, AUDIO_DIR, MOSS_DIR, SCENARIOS, load_env
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def default_resources_root() -> Path:
    return Path("D:/DuplexVoice_resources") if os.name == "nt" and Path("D:/").is_dir() else ROOT / "resources"

# 数据链路目录（与 run.py 的 stage 一一对应）
OUTPUT_DIR = ROOT / "output"
ANNOTATIONS_DIR = ROOT / "data" / "annotations"   # ① text  产物
AUDIO_DIR = ROOT / "output" / "audio"             # ② audio 产物（逐条对话 wav）
BG_POOL_DIR = ROOT / "output" / "bg_pool"         # ③ bg    产物（旧：预拼接池；新流程改直接读素材即时拼接）
def resolve_background_root() -> Path:
    explicit = os.environ.get("BG_SOURCE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    explicit_resources = os.environ.get("DUPLEX_RESOURCES_DIR")
    if explicit_resources:
        return Path(explicit_resources).expanduser() / "background" / "voices_babble"
    local = (Path("D:/DuplexVoice_downloads/VOICES_babble/data") if os.name == "nt"
             else Path("/mnt/d/DuplexVoice_downloads/VOICES_babble/data"))
    for legacy in (local, ROOT / "data" / "voices_babble"):
        if legacy.is_dir():
            return legacy
    resources = Path(os.environ.get("DUPLEX_RESOURCES_DIR", str(default_resources_root())))
    return resources / "background" / "voices_babble"


BG_SOURCE_DIR = resolve_background_root()  # 外部背景素材根（rm1~4/babb/）
MIXED_DIR = ROOT / "output" / "mixed"             # ④ mix   产物（成品 + 背景轨）
LOGS_DIR = Path(os.environ.get("DUPLEX_LOGS_DIR", str(ROOT / "logs")))

# 外部依赖
_bundled_moss = ROOT / "scripts" / "sound" / "moss_runtime"
_local_moss = Path("D:/Project_Vibe/moss-tts-official")
_default_moss = (_bundled_moss if (_bundled_moss / "onnx_tts_runtime.py").is_file()
                 else _local_moss if os.name == "nt" and _local_moss.is_dir()
                 else ROOT / "external" / "moss-tts-nano")
MOSS_DIR = Path(os.environ.get("MOSS_DIR", str(_default_moss)))
_legacy_vad = ROOT / "scripts" / "sound" / "models" / "silero_vad.onnx"


def resolve_vad_model() -> Path:
    """Resolve after .env loading; prefer the pinned model, with legacy fallback."""
    explicit = os.environ.get("VAD_MODEL")
    if explicit:
        return Path(explicit).expanduser()
    resources = Path(os.environ.get("DUPLEX_RESOURCES_DIR", str(default_resources_root()))).expanduser()
    pinned = resources / "models" / "silero_vad_v512.onnx"
    return pinned if pinned.is_file() or not _legacy_vad.is_file() else _legacy_vad


VAD_MODEL = resolve_vad_model()  # compatibility for callers importing the constant

# 运行记录（所有 stage 共用；断点续传的状态依据 = 产物文件本身，此日志用于追溯）
RUN_LOG = LOGS_DIR / "run.jsonl"

# ---------------- 数据化配置（2026-09-16 用户要求：配置类内容进数据文件，不进脚本） ----------------
# 场景：data/scenarios.json（每条含 desc 描述 / default_users 默认人数 / profile 是否启用参与画像）
_SCEN = json.loads((ROOT / "data" / "scenarios.json").read_text(encoding="utf-8"))
SCENARIOS = {name: cfg["desc"] for name, cfg in _SCEN.items()}
DEFAULT_USERS = {name: cfg.get("default_users", 2) for name, cfg in _SCEN.items()}
PROFILE_SCENARIOS = {name for name, cfg in _SCEN.items() if cfg.get("profile")}
# 主题领域：data/topics.json（JSON 数组）
TOPIC_DOMAINS = json.loads((ROOT / "data" / "topics.json").read_text(encoding="utf-8"))


def load_env(path: Path | None = None) -> None:
    """.env → os.environ（不覆盖已存在的环境变量）。"""
    env_file = path or ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
