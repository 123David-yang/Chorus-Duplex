"""剧本校验与修复（validate.py）—— LLM 直出结果的守门人。

对 build_dialogue 的输出做两件事（记录修复 / 判失败触发重试）：
- 自动修：id 规范化、kind 一致性、gap·insert_ratio clamp、事件白名单限频、非法 speaker/addressee 处理
- 判失败（返回 None，由上层重试）：第三方时序错乱、插曲落尾(<3轮)、有效轮 <4、
  超总轮数硬上限（请求+TURN_CAP_SLACK）、参与画像偏离 >1、介入节奏不达标
"""
from __future__ import annotations

import re
from typing import Any

from ..utils.schema import DEFAULT_GAP_MS, TURN_CAP_SLACK, Dialogue, Event, Participant, Turn
from .profile import TRIGGERS, apply_realized, rhythm_ok, rhythm_valid

KINDS = {"speech", "backchannel", "interruption", "side_talk"}
MAX_SIDE_TALK = 3                  # 第三方实际发声的轮数上限（超出降级为 speech）
GAP_CLAMP = (-5000, 2000)          # 合理区间外 clamp（LLM 偶发乱填）
EXTRA_ADDRESSEES = {"assistant", "third_party", "none"}
# 非语音声音事件（场景层，音频素材后续接入）：白名单 + 限频
EVENT_TYPES = {"phone_ring", "doorbell", "notification", "music", "knock", "alarm"}
MAX_EVENTS_PER_TURN = 2
MAX_EVENTS_PER_DIALOGUE = 3

_UID_A = re.compile(r"^user[_\-\s]?([a-z])$", re.I)   # user_a / user a / userA
# 第三方对话的时序检查：同时含"场景上下文"与"应答语"的用户轮，不能出现在第三方开口之前
_TP_CONTEXT = re.compile(r"(喂|门铃|敲门|手机响|接个电话|接一下|外卖|快递)")
_REPLY_MARK = re.compile(r"(谢谢|放门口|好嘞|麻烦你|收到|知道了|挂了|行行|好的，放|你先放)")
_UID_N = re.compile(r"^user[_\-\s]?(\d{1,3})$", re.I)  # user_1 / user1 / user_01
_UID_TP = re.compile(r"^third[_\-\s]?party(?:[_\-\s]?(\d{1,2}))?$", re.I)  # third_party / third_party_01
def _is_third_party(sid: str) -> bool:
    return bool(_UID_TP.match(str(sid or "").strip()))


def _normalize_tp(raw: str, idx: int = 0) -> str:
    """third party 任意写法 → third_party_NN（无编号时默认 01）。"""
    m = _UID_TP.match(str(raw or "").strip())
    n = int(m.group(1)) if (m and m.group(1)) else 1
    return f"third_party_{n:02d}"


def _normalize_uid(raw: str, idx: int) -> str:
    """任意写法 → user_NN（两位补零）；无法识别则按位次分配。"""
    s = str(raw or "").strip()
    m = _UID_A.match(s)
    if m:
        return f"user_{ord(m.group(1).lower()) - 96:02d}"
    m = _UID_N.match(s)
    if m and 1 <= int(m.group(1)) <= 99:
        return f"user_{int(m.group(1)):02d}"
    return f"user_{idx + 1:02d}"


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def build_dialogue(script: dict, dialogue_id: str, scenario: str,
                   profile: dict | None = None,
                   turns_requested: int | None = None) -> tuple[Dialogue | None, list[str]]:
    """校验 + 修复 LLM 输出 → Dialogue，返回 (dialogue, 修复记录)。

    profile: 参与画像（采样自 oneshot_agent/profile.py）；给出时校验 assistant 轮分布并回填实际值。
    turns_requested: 请求轮数；给出时按 TURN_CAP_SLACK 硬控总轮数上限（超出判失败重试）。
    """
    fixes: list[str] = []
    parts = script.get("participants") or []
    if not isinstance(parts, list) or not parts or any(not isinstance(p, dict) for p in parts):
        return None, ["participants 必须是含 id/persona/style 的对象数组"]

    d = Dialogue(dialogue_id=dialogue_id, scenario=scenario,
                 setting=str(script.get("setting", "")), topic=str(script.get("topic", "")))
    id_map: dict[str, str] = {}  # LLM 原写法 → user_NN
    for i, pp in enumerate(parts):
        raw_id = str(pp.get("id") or "")
        if raw_id.strip() == "assistant":
            fixes.append("participants 中的 assistant 元数据已移除，保留固定 AI 身份")
            continue
        norm = _normalize_uid(raw_id, len(d.participants))
        if norm in {p.id for p in d.participants}:
            return None, ["重复参与者 ID"]
        id_map[raw_id.strip()] = norm
        d.participants.append(Participant(
            id=norm, persona=str(pp.get("persona", "")), style=str(pp.get("style", "")),
        ))
    if not d.participants: return None, ["缺少正式用户"]
    valid_ids = {p.id for p in d.participants} | {"assistant"}
    allowed_addr = valid_ids | EXTRA_ADDRESSEES

    def resolve(ref) -> str:
        """把指向/说话人引用解析为规范 id（容忍 user_a / user1 等写法）。"""
        s = str(ref or "").strip()
        if s == "assistant": return s
        if s in id_map:
            return id_map[s]
        if _UID_A.match(s) or _UID_N.match(s):
            return _normalize_uid(s, 0)
        if _UID_TP.match(s):
            return _normalize_tp(s)
        return s

    turns_raw: list[dict[str, Any]] = script.get("turns") or []
    if not isinstance(turns_raw, list) or not turns_raw or any(not isinstance(t, dict) for t in turns_raw):
        return None, ["turns 必须是话轮对象数组"]

    raw_to_new = {}
    n_events = 0     # 非语音声音事件计数（整条限频）
    n_side_talk = 0  # 第三方实际发声轮计数（整条限频）
    for idx, t in enumerate(turns_raw):
        speaker = resolve(t.get("speaker", ""))
        text = str(t.get("text", "")).strip()
        if speaker not in valid_ids and not _is_third_party(speaker):
            fixes.append(f"[{idx}] 非法 speaker={speaker!r}，跳过该轮")
            continue
        if not text:
            fixes.append(f"[{idx}] 空文本，跳过该轮")
            continue

        kind = str(t.get("kind", "speech"))
        if kind not in KINDS:
            fixes.append(f"[{idx}] 非法 kind={kind!r} → speech")
            kind = "speech"

        # side_talk：第三方（非正式成员）实际发声的轮——正常拼接，仅标记；speaker 与 kind 必须一致
        if _is_third_party(speaker) and kind != "side_talk":
            fixes.append(f"[{idx}] 第三方 speaker 的 kind {kind!r} → side_talk")
            kind = "side_talk"
        elif kind == "side_talk" and not _is_third_party(speaker):
            fixes.append(f"[{idx}] kind=side_talk 但 speaker={speaker!r} 非第三方 → speech")
            kind = "speech"
        if kind == "side_talk":
            n_side_talk += 1
            if n_side_talk > MAX_SIDE_TALK:
                return None, [f"[{idx}] 第三方发声次数超限，重试"]

        try:
            gap = int(t.get("gap_before_ms", DEFAULT_GAP_MS))
        except (TypeError, ValueError):
            gap = DEFAULT_GAP_MS
        clamped = _clamp(gap, *GAP_CLAMP)
        if clamped != gap:
            fixes.append(f"[{idx}] gap {gap} → {clamped}（clamp）")
        gap = clamped

        addr = t.get("addressee", "assistant")
        if isinstance(addr, list):
            addr = ["third_party" if _is_third_party(a) else resolve(a) for a in addr]
            bad = [a for a in addr if a not in allowed_addr]
            if bad:
                fixes.append(f"[{idx}] 非法 addressee={bad!r}，已剔除")
            addr = [a for a in addr if a in allowed_addr] or ["assistant"]
            if len(addr) == 1:
                addr = addr[0]  # 单目标不写成列表
        else:
            addr = "third_party" if _is_third_party(str(addr)) else resolve(addr)
            if addr not in allowed_addr:
                fixes.append(f"[{idx}] 非法 addressee={addr!r} → assistant")
                addr = "assistant"

        rt = t.get("responds_to") or []
        if not isinstance(rt, list) or any(type(x) is not int or x < 0 or x >= idx for x in rt):
            return None, [f"[{idx}] responds_to 引用自身/未来或格式错误，拒绝而不伪造修正"]
        if any(x not in raw_to_new for x in rt):
            return None, [f"[{idx}] responds_to 引用已被跳过的无效轮，重试"]
        rt = list(dict.fromkeys(raw_to_new[x] for x in rt))

        # backchannel：位置用 insert_ratio（插到上一轮的百分比处）；不能是第 0 轮
        insert_ratio = None
        if kind == "backchannel":
            cleaned = text.rstrip("。，、！？；~～· ")
            if cleaned != text:
                fixes.append(f"[{idx}] 附和尾标点已清理: {text!r} → {cleaned!r}")
                text = cleaned
            if not d.turns:
                fixes.append(f"[{idx}] 首轮不能是 backchannel → speech")
                kind = "speech"
            else:
                try:
                    insert_ratio = float(t.get("insert_ratio", 0.6))
                except (TypeError, ValueError):
                    insert_ratio = 0.6
                clamped_r = min(max(insert_ratio, 0.2), 0.9)
                if clamped_r != insert_ratio:
                    fixes.append(f"[{idx}] insert_ratio {insert_ratio} → {clamped_r}（clamp）")
                insert_ratio = clamped_r

        # 非语音声音事件（场景层）：白名单 + position clamp + 限频
        evs: list[Event] = []
        raw_evs = t.get("events") or []
        if isinstance(raw_evs, list):
            for e in raw_evs:
                if not isinstance(e, dict):
                    continue
                etype = str(e.get("type", "")).strip()
                if etype not in EVENT_TYPES:
                    fixes.append(f"[{idx}] 非法声音事件 type={etype!r}，已丢弃")
                    continue
                if len(evs) >= MAX_EVENTS_PER_TURN or n_events >= MAX_EVENTS_PER_DIALOGUE:
                    fixes.append(f"[{idx}] 声音事件超限（每轮≤{MAX_EVENTS_PER_TURN}、整条≤"
                                 f"{MAX_EVENTS_PER_DIALOGUE}），已丢弃 {etype}")
                    continue
                try:
                    pos = float(e.get("position", 0.0))
                except (TypeError, ValueError):
                    pos = 0.0
                clamped_p = min(max(pos, 0.0), 1.0)
                if clamped_p != pos:
                    fixes.append(f"[{idx}] 事件 position {pos} → {clamped_p}（clamp）")
                evs.append(Event(type=etype, text=str(e.get("text", ""))[:40],
                                 turn_index=len(d.turns), position=clamped_p))
                n_events += 1

        # assistant 轮的触发类型（参与画像用）：cue / volunteer / invited / coordinate
        trigger = None
        if speaker == "assistant":
            raw_trig = str(t.get("trigger", "")).strip().lower()
            if raw_trig in TRIGGERS:
                trigger = raw_trig
            elif raw_trig:
                fixes.append(f"[{idx}] 非法 trigger={raw_trig!r}，已忽略")

        raw_to_new[idx] = len(d.turns)
        d.turns.append(Turn(speaker=speaker, text=text, addressee=addr,
                            gap_before_ms=gap, kind=kind, insert_ratio=insert_ratio,
                            responds_to=rt, events=evs, trigger=trigger,
                            anchor_text=t.get("anchor_text") if kind=="backchannel" else None))

    # 第三方对话的两类硬性问题 → 判失败重试（标注错误/时序错乱都会污染训练标签）
    for i in range(len(d.turns) - 1):
        cur, nxt = d.turns[i], d.turns[i + 1]
        # ① 第三方台词后紧接的用户回应被标成 assistant → 高置信度自动改标 third_party（保留修正记录）
        if (cur.speaker.startswith("third_party")
                and nxt.speaker != "assistant"
                and not nxt.speaker.startswith("third_party")
                and nxt.addressee == "assistant" and _REPLY_MARK.search(nxt.text)):
            fixes.append(f"[{i + 1}] 回应第三方却标 assistant → third_party（自动修正）")
            nxt.addressee = "third_party"
        # ② 主角在第三方开口前就"回应"（时序错乱）：前一轮同时含场景上下文与应答语
        if (cur.speaker != "assistant" and not cur.speaker.startswith("third_party")
                and nxt.speaker.startswith("third_party")
                and _TP_CONTEXT.search(cur.text) and _REPLY_MARK.search(cur.text)):
            return None, [f"[{i}] 主角在第三方开口前已作回应（时序错乱），重试"]

    # ③ 侧写插曲位置（2026-09-15 用户要求）：插曲后主线至少还有 3 轮——
    #    外卖收尾=对话被掐断，"回归主题/抗干扰"训练不到
    tp_idx = [i for i, t in enumerate(d.turns)
              if t.speaker.startswith("third_party") or t.addressee == "third_party"
              or (isinstance(t.addressee, list) and "third_party" in t.addressee)]
    after = len(d.turns) - 1 - max(tp_idx) if tp_idx else None
    if after is not None and after < 3:
        return None, [f"插曲落在结尾（其后仅 {after} 轮 < 3），重试"]

    if len(d.turns) < 4:
        return None, [f"有效轮数不足（{len(d.turns)} < 4）"]

    # 参与画像：回填实际分布（assistant 次数/间隔/触发），记录与计划的偏差
    if profile:
        try:apply_realized(d, profile)
        except ValueError:return None,["invalid_duplex_response_pair"]
        planned = profile["assistant_turns"]
        realized = profile["realized"]["assistant_turns"]
        intervals = profile["realized"]["intervals"]
        if abs(realized - planned) > 1:
            # 计划是硬约束：偏差 >1 判失败（由 run.py 重试同一条计划）
            return None, [f"assistant 实发言 {realized} 次，偏离计划 {planned} 次（重试）"]
        if not rhythm_valid(d,profile):
            return None, [f"介入节奏 {intervals}（{profile['intensity']}档）不达标（重试）"]

    # 总轮数硬上限（2026-09-16 用户要求控 turn）：请求轮数 + 松弛量，超出判失败重试
    req = turns_requested or (profile or {}).get("turns_requested")
    if req and len(d.turns) > req + TURN_CAP_SLACK:
        return None, [f"总轮数 {len(d.turns)} 超硬上限 {req + TURN_CAP_SLACK}（请求 {req}），重试"]
    from ..quality import audit
    report = audit(d)
    if not report["passed"]:
        return None, fixes + [str(row) for row in report["issues"]]
    return d, fixes

