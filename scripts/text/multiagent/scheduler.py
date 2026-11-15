"""调度器（多 Agent 路径）：泊松抽样决定"谁发言"。

机制：每个在场参与者有一个泊松发言强度 λ（speech_rate），
调度器为每个人抽样"下一次发言时刻"（间隔 = Exp(λ) 指数分布），
每轮取"下次发言时刻"最早的参与者发言；发言后为其重新抽样。
参与状态：gone → 不参与抽样；returned → 恢复参与。
"""
from __future__ import annotations

import random
from typing import Optional

from ..utils.schema import Dialogue, Participant


class Scheduler:
    ASSISTANT_RATE = 1.0  # assistant 固定发言强度

    def __init__(self, dialogue: Dialogue, rng: random.Random):
        self.dialogue = dialogue
        self.rng = rng
        self.now = 0.0  # 当前时间（轮次推进的模拟时钟）
        self.next_t: dict[str, float] = {}
        for p in dialogue.participants:
            if p.state == "active":
                self.next_t[p.id] = self.rng.expovariate(max(p.speech_rate, 0.1))
        # assistant 也是对话参与者（固定强度）
        self.next_t["assistant"] = self.rng.expovariate(self.ASSISTANT_RATE)

    def _active_ids(self) -> list[str]:
        ids = [p.id for p in self.dialogue.participants if p.state != "gone"]
        return ids + ["assistant"]

    def next_speaker(self) -> Optional[str]:
        """返回下一个发言者 id（下次发言时刻最早者）。"""
        cand = [pid for pid in self._active_ids() if pid in self.next_t]
        if not cand:
            return None
        return min(cand, key=lambda pid: self.next_t[pid])

    def after_speech(self, speaker_id: str) -> None:
        """发言结束后：推进时钟并重新为该参与者抽样。"""
        self.now = max(self.now, self.next_t.get(speaker_id, self.now))
        if speaker_id == "assistant":
            self.next_t[speaker_id] = self.now + self.rng.expovariate(self.ASSISTANT_RATE)
            return
        for p in self.dialogue.participants:
            if p.id == speaker_id:
                if p.state != "gone":
                    self.next_t[p.id] = self.now + self.rng.expovariate(max(p.speech_rate, 0.1))
                break

    def apply_state(self, pid: str, state: str) -> None:
        """应用参与状态变更（离场/回归）。"""
        for p in self.dialogue.participants:
            if p.id == pid:
                p.state = state
                if state == "gone":
                    self.next_t.pop(pid, None)
                elif state in ("active", "returned") and pid not in self.next_t:
                    self.next_t[pid] = self.now + self.rng.expovariate(max(p.speech_rate, 0.1))
                break

    def pick_addressee(self, pid: str) -> str:
        """按 addressee_pref 概率抽样本轮指向（MVP 简化：均匀偏好）。"""
        others = [p.id for p in self.dialogue.participants if p.id != pid and p.state != "gone"]
        if pid == "assistant":
            # AI 发言只可能指向用户（或第三方），不指向自己
            options = others + ["third_party"]
            weights = [1.0] * len(others) + [0.3]
        else:
            options = ["assistant"] + others + ["third_party"]
            weights = [3.0] + [1.0] * len(others) + [0.3]  # 默认更常指向 AI
        return self.rng.choices(options, weights=weights, k=1)[0]
