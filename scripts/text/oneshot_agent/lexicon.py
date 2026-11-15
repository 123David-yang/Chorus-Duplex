"""附和语库（lexicon.py）—— data/lexicons/backchannels.jsonl 的加载与抽样。

库设计（2026-09-13 调研后自建）：
- HF 无现成中文文本级附和语库；分类框架借鉴 Switchboard SwDA
  （Acknowledge-Backchannel / Backchannel-in-question-form）与对话分析惯例
- 字段：type（功能类型） + tone（语体：neutral/elder/youth/formal）
- 用途：① 注入一体直出 prompt 作示例（提升多样性）② 后续可做确定性抽样注入

类型：ack 认同 / receipt 接收 / surprise 惊讶 / empathy 共情 /
      amuse 笑意 / nudge 催促 / bcq 疑问式附和 / realize 恍然
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from scripts.config import ROOT

BANK_PATH = ROOT / "data" / "lexicons" / "backchannels.jsonl"
TYPES = ("ack", "receipt", "surprise", "empathy", "amuse", "nudge", "bcq", "realize")

_cache: list[dict] | None = None


def load_bank(path: str | Path | None = None) -> list[dict]:
    """加载全库（带缓存）。每行 {"text", "type", "tone"}。"""
    global _cache
    if _cache is None:
        p = Path(path) if path else BANK_PATH
        _cache = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return _cache


def sample_backchannels(k: int = 12, tone: str | None = None,
                        rng: random.Random | None = None) -> list[str]:
    """抽样 k 条附和语文本（可按语体过滤）；尽量覆盖不同功能类型。"""
    rng = rng or random.Random()
    pool = [e for e in load_bank() if tone is None or e["tone"] == tone]
    if not pool:
        return []
    by_type: dict[str, list[str]] = {}
    for e in pool:
        by_type.setdefault(e["type"], []).append(e["text"])
    texts: list[str] = []
    while len(texts) < k:
        for t in TYPES:
            if t in by_type and len(texts) < k:
                texts.append(rng.choice(by_type[t]))
    # 去重保序（同文本可能因类型循环被重复抽到）
    seen, out = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
