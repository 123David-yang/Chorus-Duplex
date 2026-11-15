"""Shared duplex constraints; deterministic audits never invent missing dialogue content."""
from __future__ import annotations
import copy
import json
import math
import random
import re
import zlib
from .utils.schema import Dialogue, Turn

MODULES = ("barge_in", "ai_backchannel", "identity_ask")

def coverage_plan(dialogue_id: str, turns: int) -> list[str]:
    # Stable across retries, balanced in expectation; do not filter away harder modules.
    if turns < 8:
        return []
    seed = zlib.crc32(("duplex-v2:" + dialogue_id).encode())
    return [MODULES[seed % len(MODULES)]]

def rules_block(dialogue_id: str = "", turns: int = 12) -> str:
    required = coverage_plan(dialogue_id, turns)
    bc_policy = "AI 必须完成一次指向它的附和。" if "ai_backchannel" in required else "本条 AI 不做 backchannel，附和由其他用户承担；AI 留出轮数完成主模块。"
    return """
【双工质量契约 v3，优先于泛化示例】
- 台词的知识只能来自该轮之前已说出口的历史。人设/导演计划属于私有资料；
  assistant 不得从中得知姓名、职业、经历。用户也只能知道自己的私有背景。
  用户自我介绍或别人先自然喊出名字后，assistant 才能使用该称呼。首次引入姓名
  必须由用户完成，不能由 assistant 猜测；不要用内部编号作口语称呼。
- 台词绝不能念出 user_01、turn_id、responds_to、anchor_text 或“第5轮”这类内部定位标签；用自然语言承接内容。
- 对明确问 AI 的问题必须回应；没有下一次 AI 回答位置的用户轮，只能回应/补充，不得新增一个无人回答的 AI 问题。
- 用户打断 AI 提出新限制后，下一条 AI 主轮要立即承接，不要等多个人换话题后才响应。
- responds_to 是 0 起的原始 turns 数组索引，必须严格小于当前索引；不可引用自己或未来。
- backchannel 的宿主是最近一条非 backchannel 主轮，不是最近数组元素。
  附和者必须与宿主不同；addressee 必须是宿主，responds_to 只含宿主索引。
  附和不引入新信息、不提问、不接管话权，正文去标点后 1~6 字；使用 insert_ratio=0.4~0.75。
  自言自语的“嗯”是 speech，不是附和；完整论述也不是附和。
  对连续内容的附和只能表明在听，不得提前同意尚未说出的结论；避免宿主最后才出现否定/转折。
- interruption 只能是另一人抢话，gap_before_ms=-300~-600；宿主是上一主轮，
  两人必须不同。自己改口/接电话是 speech（非负间隔），不得标跨人打断。
  用户打断 AI 时，AI 原句必须是尚未说完的半句；用户明确修正/停止/另提需求，后续 AI 回应它。
- AI 的短附和与长回答分别标注；AI 附和的宿主 addressee 必须显式含 assistant。
  如果上一主轮指向另一个用户，就换一个用户附和，不得让 AI 附和。不要为了凑数量打扰私聊或第三方通话。
- 非参与型插曲之后至少 3 条主轮且至少 1 条 AI 发言，回到插曲前的具体话题。
- 导演可计划交互，但不得伪造离场：离场必须有说出口的依据。
""" + "\n本条必须自然覆盖的模块（计入总轮数和 AI 参与计划，不额外增加轮数）：" + json.dumps(required) + "\n" + bc_policy + """
barge_in=至少一次用户打断 AI 半句，AI 随后响应新请求；
ai_backchannel=至少一次 AI 对指向它的用户主轮作简短附和，双方 speaker 不同；
identity_ask=按顺序的三条主轮（中间允许其他用户接话，以遵守参与频率）：assistant 问某位用户怎么称呼 -> 被问用户自报名字 -> assistant 用该名字回应主话题。
  必须是 AI 问用户名字；用户询问 AI 叫什么、不算此模块。名字不能提前出现于 AI 台词。
未列出的模块可自然出现，不要机械堆叠；不要把姓名写进 assistant 的隐含背景。
"""

def strong_question(text):
    return bool(re.search(r"[?？]|能不能|能否|是不是|有没有|要不要|可不可以|应不应该",text))

def ai_request_candidates(d):
    return [i for i,t in enumerate(d.turns) if t.speaker.startswith("user_") and t.kind!="backchannel"
            and (t.addressee=="assistant" or isinstance(t.addressee,list) and "assistant" in t.addressee)
            and re.search(r"[?？]|请|帮我|帮我们|能不能|能否|是不是|有没有|要不要|麻烦|想问|推荐|怎么|如何",t.text)]


def audit(d: Dialogue, required: list[str] | None = None) -> dict:
    issues = []
    def flag(code, i, detail):
        issues.append({"code": code, "turn_id": i, "detail": detail})
    valid = {p.id for p in d.participants} | {"assistant"}
    host = None
    counts = {"barge_in": 0, "ai_backchannel": 0, "backchannel": 0, "interruption": 0}
    for i, t in enumerate(d.turns):
        if t.speaker not in valid and not t.speaker.startswith("third_party_"):
            flag("invalid_speaker", i, t.speaker)
        if t.kind not in {"speech","backchannel","interruption","side_talk"}:
            flag("invalid_kind", i, t.kind)
        if t.speaker.startswith("third_party_") != (t.kind=="side_talk"):
            flag("side_talk_speaker_mismatch", i, t.speaker)
        addr = t.addressee if isinstance(t.addressee, list) else [t.addressee]
        if not addr or any(a not in valid | {"third_party", "all", "none"} for a in addr):
            flag("invalid_addressee", i, str(addr))
        if t.speaker in addr:
            flag("self_address", i, "speaker cannot be their own listener")
        if any(type(r) is not int or not 0 <= r < i for r in t.responds_to):
            flag("noncausal_response", i, str(t.responds_to))
        if not t.text.strip(): flag("empty_text", i, "empty utterance")
        if re.search(r"(?:user_\d+|third_party_\d+|\bturn[_ -]?\d+\b|\bturn_id\b|\bresponds_to\b|\banchor_text\b|第\s*\d+\s*轮)",t.text,re.I):
            flag("spoken_internal_metadata",i,t.text)
        if t.kind == "backchannel":
            counts["backchannel"] += 1
            if host is None:
                flag("missing_host", i, "backchannel has no main turn")
            else:
                h = d.turns[host]
                anchor=t.anchor_text
                if anchor is not None:
                    if not isinstance(anchor,str) or len(anchor)<3 or not h.text.startswith(anchor) or len(h.text)-len(anchor)<4:
                        flag("invalid_semantic_anchor", i, "must be an exact host prefix with remaining speech")
                elif d.profile.get("require_semantic_anchors"):
                    flag("missing_semantic_anchor", i, "production backchannels require a heard-content prefix")
                if h.speaker == t.speaker: flag("self_backchannel", i, f"host={host}")
                if addr != [h.speaker] or t.responds_to != [host]:
                    flag("backchannel_anchor", i, f"must target host {host}: {h.speaker}")
                if t.speaker == "assistant" and h.addressee != "assistant" and not (
                    isinstance(h.addressee, list) and "assistant" in h.addressee):
                    flag("uninvited_ai_backchannel", i, "host was not addressing AI")
            letters = ''.join(c for c in t.text if c.isalnum())
            if any(word in t.text for word in ("的话","我觉得","你可以","但是","因为","所以","要不")):
                flag("contentful_backchannel", i, t.text)
            if not 1 <= len(letters) <= 6 or any(c in t.text for c in "?？"):
                flag("long_or_question_backchannel", i, t.text)
            if not isinstance(t.insert_ratio, (float, int)) or not math.isfinite(t.insert_ratio) or not .2 <= t.insert_ratio <= .9:
                flag("invalid_insert_ratio", i, str(t.insert_ratio))
            if t.speaker == "assistant": counts["ai_backchannel"] += 1
        else:
            if t.kind == "interruption" or t.gap_before_ms < 0:
                counts["interruption"] += 1
                if t.kind != "interruption": flag("unlabelled_overlap", i, "negative gap on speech")
                if host is None or d.turns[host].speaker == t.speaker:
                    flag("self_or_missing_interruption", i, f"host={host}")
                if t.gap_before_ms >= 0: flag("nonoverlapping_interruption", i, "requires negative gap")
                if host is not None and d.turns[host].speaker == "assistant" and t.speaker.startswith("user_"):
                    counts["barge_in"] += 1
                    if d.profile.get("require_semantic_anchors"):
                        response=next((j for j in range(i+1,len(d.turns)) if d.turns[j].speaker=="assistant" and d.turns[j].kind!="backchannel"),None)
                        if response is None or response>i+2 or i not in d.turns[response].responds_to:
                            flag("delayed_or_unlinked_barge_response",i,"AI must acknowledge the new request within the next two slots")
            host = i
        for e in t.events:
            if e.turn_index != i: flag("event_index", i, str(e.turn_index))
    side = [i for i,t in enumerate(d.turns) if t.speaker.startswith("third_party_") or
            t.addressee == "third_party" or isinstance(t.addressee,list) and "third_party" in t.addressee]
    if side:
        tail = [t for t in d.turns[max(side)+1:] if t.kind != "backchannel"]
        if len(tail)<3 or not any(t.speaker=="assistant" for t in tail):
            flag("missing_mainline_return", max(side), "need >=3 main turns including AI")
    required = required if required is not None else d.profile.get("required_modules", [])
    # Identity introductions and interrupted half-sentences need semantic evidence from the reviewer.
    for module in required:
        if module in counts and not counts[module]: flag("missing_module", -1, module)
    if "assistant_turns" in d.profile:
        positions=[i for i,t in enumerate(d.turns) if t.speaker=="assistant"]
        intervals=[b-a-1 for a,b in zip(positions,positions[1:])]
        from .oneshot_agent.profile import rhythm_valid
        if abs(len(positions)-d.profile["assistant_turns"])>1 or not rhythm_valid(d,d.profile):
            flag("participation_profile", -1, "AI count/rhythm differs from plan")
    return {"version":"duplex-quality-v3", "passed":not issues, "issues":issues,
            "coverage":counts, "required_modules":required}

def augmentation_candidates(d: Dialogue, probability: float, seed: int) -> list[int]:
    if not 0 <= probability <= 1: raise ValueError("probability must be in [0,1]")
    rng=random.Random(seed)
    candidates=[]
    for i,t in enumerate(d.turns[:-1]):
        window=d.turns[max(0,i-2):i+1]
        if len(window)<3 or any(x.kind in ("backchannel","interruption","side_talk") or x.addressee=="third_party" for x in window): continue
        if t.kind!="speech" or t.speaker.startswith("third_party") or t.events or len(t.text)<12: continue
        if any(x in t.text for x in ("？","?","但是","不过","并不","不是","等下")): continue
        if d.turns[i+1].kind in ("backchannel","interruption") or d.turns[i+1].gap_before_ms<0: continue
        if rng.random()<probability: candidates.append(i)
    return candidates

def insert_backchannel(d: Dialogue, anchor: int, speaker: str, text: str, ratio: float, anchor_text: str | None = None) -> Dialogue:
    """Pure operation: remap causal edges/events after inserting; never mutate originals."""
    if not 0<=anchor<len(d.turns) or d.turns[anchor].kind!="speech": raise ValueError("invalid host")
    h=d.turns[anchor]
    if speaker==h.speaker or speaker not in {p.id for p in d.participants}|{"assistant"}: raise ValueError("invalid listener")
    if not .4<=ratio<=.75: raise ValueError("invalid ratio")
    out=copy.deepcopy(d); at=anchor+1
    for t in out.turns:
        t.responds_to=[r+1 if r>=at else r for r in t.responds_to]
    out.turns.insert(at,Turn(speaker=speaker,text=text,kind="backchannel",addressee=h.speaker,
                           responds_to=[anchor],insert_ratio=ratio,gap_before_ms=0,anchor_text=anchor_text,
                           trigger="cue" if speaker=="assistant" else None))
    for i,t in enumerate(out.turns):
        t.turn_id=i
        for e in t.events: e.turn_index=i
    if "assistant_turns" in out.profile:
        from .oneshot_agent.profile import apply_realized
        apply_realized(out,out.profile)
    report=audit(out)
    if not report["passed"]: raise ValueError("augmentation violates deterministic contract")
    return out
