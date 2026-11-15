"""数据模型（v2）：text 侧产出，sound 侧消费的契约。

v2 升级点（双工时序）：
- Turn.turn_id：轮次 id（0 起，序列化时按顺序自动赋值）
- Turn.gap_before_ms：与上一轮的间隔（ms），负值 = 提前开口（重叠/打断/附和）
- Turn.kind：speech / backchannel / interruption / side_talk
  （side_talk = 非正式成员（第三方，speaker 为 third_party_NN）实际发声的轮，正常拼接，仅作标记）
- Turn.responds_to：回应/综合了哪几轮（综合回复标注）
- Turn.addressee：支持列表（一句话指向多人）

命名规范：参与者 id 为 user_01 ~ user_99（两位补零，最多 99 人）+ assistant。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Event:
    """对话中的交互事件（打断/附和/侧语等）。"""
    type: str                       # "interruption" | "user_backchannel" | "side_talk" | ...
    text: str                       # 事件话语内容
    turn_index: int                 # 事件落在第几轮（0 起）
    position: Optional[float] = None  # 0~1，在目标话语中的大致位置（音频对齐用）


@dataclass
class Participant:
    """一位对话参与者（用户侧；assistant 固定存在）。"""
    id: str                         # "user_01" / "user_02" / ...（两位补零）
    persona: str
    style: str
    speech_rate: float = 1.0        # 泊松发言强度 λ
    state: str = "active"           # active / silent / gone / returned


DEFAULT_GAP_MS = 250  # 话轮间默认停顿（人类对话中位数 ~200ms，Stivers et al. 2009）
# 总轮数硬上限：请求轮数 + 该松弛量，超出判不合格重试（2026-09-16 用户要求控制 turn 上限；
# 提示词给模型同一数字，校验器硬查——LLM 此前会超产到请求的 1.8 倍，长对话也更吃 TTS 内存）
TURN_CAP_SLACK = 5


@dataclass
class Turn:
    """一轮话语（id 置首字段；其后是内容 + 时序 + 指向）。"""
    turn_id: int = 0                # 轮次 id（0 起；to_jsonl_line 按顺序自动赋值）
    speaker: str = ""               # "user_01" / "user_02" / "assistant"
    text: str = ""                  # 实际说出的内容；被打断的轮 = 半截文本（不补全）
    addressee: str | list[str] = "assistant"  # assistant / user_01 / ... / third_party / none；或列表
    gap_before_ms: int = DEFAULT_GAP_MS  # 与上一轮结束的间隔；负 = 提前开口（重叠）
    kind: str = "speech"            # speech / backchannel / interruption / side_talk
    insert_ratio: float | None = None  # 仅 backchannel：插到上一轮的第百分之几处（0~1）
    responds_to: list[int] = field(default_factory=list)  # 回应/综合了哪几轮（0 起序号）
    events: list[Event] = field(default_factory=list)
    anchor_text: str | None = None   # BC: exact heard prefix of the latest main turn
    trigger: str | None = None      # 仅 assistant 轮：cue / volunteer / invited / coordinate
    parts: list[str] | None = None  # incomplete: spoken fragments
    pause_ms: int | None = None    # incomplete: silence between fragments


@dataclass
class Dialogue:
    """一条完整一对多双工对话样本。"""
    dialogue_id: str
    scenario: str = "participating_multi_qa"
    setting: str = ""
    topic: str = ""
    participants: list[Participant] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    profile: dict = field(default_factory=dict)  # 参与画像（力度/计划/实际分布，见 text/oneshot_agent/profile.py）

    @staticmethod
    def from_file(path: str | Path) -> "Dialogue":
        """读取单个对话文件（美化排版的单 JSON 对象）。"""
        lines = load_dialogue_lines(path)
        if not lines:
            raise ValueError(f"空文件或无法解析: {path}")
        return Dialogue.from_jsonl_line(lines[0])

    def to_jsonl_line(self) -> str:
        d = asdict(self)
        for i, t in enumerate(d["turns"]):
            t["turn_id"] = i  # 契约保证：turn_id 恒等于轮序（0 起）
            if t["parts"] is None:
                t.pop("parts")
            if t["pause_ms"] is None:
                t.pop("pause_ms")
        return json.dumps(d, ensure_ascii=False)

    @staticmethod
    def from_jsonl_line(line: str) -> "Dialogue":
        d = json.loads(line)
        parts = [Participant(**p) for p in d.get("participants", [])]
        turns = [Turn(**{**t, "turn_id": t.get("turn_id", i),
                         "events": [Event(**e) for e in t.get("events", [])]})
                 for i, t in enumerate(d["turns"])]
        return Dialogue(
            dialogue_id=d["dialogue_id"], scenario=d.get("scenario", ""),
            setting=d.get("setting", ""), topic=d.get("topic", ""),
            participants=parts, turns=turns, profile=d.get("profile", {}),
        )


def load_dialogue_lines(path: str | Path) -> list[str]:
    """读取对话文件 → 规范 JSONL 行列表。

    兼容三种写法：JSONL（一行一条）/ 美化排版的单 JSON 对象（演示手写用）/ JSON 数组。
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return [ln for ln in text.splitlines() if ln.strip()]
    if isinstance(obj, dict):
        return [json.dumps(obj, ensure_ascii=False)]
    if isinstance(obj, list):
        return [json.dumps(o, ensure_ascii=False) for o in obj]
    return []
