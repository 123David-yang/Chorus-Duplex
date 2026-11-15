"""参与画像（participation profile）——控制 assistant 在多轮对话中的介入分布。

动机（devlog 06）：多人对话 ≠ 1:1 严格交替，assistant 的介入频率本身是被控分布。
此前 side_talk 只覆盖一个角落（"整条 1~2 次、仅被点名"），需要按画像分层采样扩大覆盖。

第一批轴（2026-09-13）：
- 参与力度 intensity（low/mid/high）：由"介入间隔"（两次 assistant 发言之间的用户轮数）定义
- 触发类型 triggers：cue 被点名 / volunteer 主动补充 / invited 被邀请 / coordinate 协调状态

用法：
    prof = sample_profile(rng, turns, num_users)
    ... 把 profile_block(prof) 注入 prompt ...
    apply_realized(dialogue, prof)  # 校验后回填实际分布（数据集自带标签）
"""
from __future__ import annotations

import random
from typing import Any

INTENSITY_WEIGHTS = {"low": 0.30, "mid": 0.40, "high": 0.30}  # 分层配额（用户 2026-09-13 认可）
TRIGGERS = ("cue", "volunteer", "invited", "coordinate")
TRIGGER_LABELS = {
    "cue": "被明确点名/直接提问，立即回答",
    "volunteer": "没被点名，但顺着话题主动补充有用的信息/建议",
    "invited": "被其他用户邀请参与（如'让AI也说说''问问它'）",
    "coordinate": "协调/提醒其他参与者（如'xx还没说呢''要不要总结一下'）",
}
_TRIGGER_WEIGHTS = {   # 力度越低越依赖点名
    "low": [0.70, 0.15, 0.10, 0.05],
    "mid": [0.45, 0.30, 0.15, 0.10],
    "high": [0.30, 0.35, 0.20, 0.15],
}


def sample_profile(rng: random.Random, turns: int, num_users: int) -> dict[str, Any]:
    """采样一条对话的参与计划：力度 → 发言次数 + 触发类型序列。"""
    intensity = rng.choices(list(INTENSITY_WEIGHTS),
                            weights=list(INTENSITY_WEIGHTS.values()))[0]
    user_turns = max(3, round(turns * num_users / (num_users + 1)))
    # 提升 assistant 回复次数（2026-09-16 用户要求）：各档整体上调（low 2~3 / mid ≈1/2.5 / high ≈1/2.2）
    # 上限收敛（2026-09-17 实测）：计划 5~6 次时模型交付率明显偏低（重试率 ~30%）
    # → mid 封顶 4、high 封顶 5，保住"AI 回复次数提升"的同时降低无效重试
    k = {"low": rng.choice([2, 3]),
         "mid": min(4, max(3, round(user_turns / 2.5))),
         "high": min(5, max(4, round(user_turns / 2.2)))}[intensity]
    k = max(1, min(k, user_turns))
    triggers = rng.choices(TRIGGERS, weights=_TRIGGER_WEIGHTS[intensity], k=k)
    # 目标发言位置：按预计总轮数均匀铺开 + 小幅抖动（比只说"间隔"更可执行）
    total_est = user_turns + k
    step = total_est / (k + 1)
    positions = sorted({max(1, min(total_est - 1, round(j * step + rng.uniform(-1.2, 1.2))))
                        for j in range(1, k + 1)})
    return {"intensity": intensity, "assistant_turns": k,
            "triggers": triggers, "user_turns_est": user_turns,
            "turns_requested": turns, "target_turns": positions}


def rhythm_ok(intensity: str, intervals: list[int]) -> bool:
    """介入节奏是否落在该力度的宽容区间（不达标 → 判失败重试）。"""
    if not intervals:
        return True
    avg = sum(intervals) / len(intervals)
    if intensity == "low":
        return min(intervals) >= 3
    if intensity == "mid":
        return 1 <= avg <= 7
    return avg <= 5  # high：接近轮流（4→5：模型对高介入节奏常差一档，降低无效重试）


def rhythm_intervals(dialogue,prof):
    positions=[i for i,t in enumerate(dialogue.turns) if t.speaker=="assistant"]
    pairs=prof.get("duplex_response_pairs",[])
    if not isinstance(pairs,list):raise ValueError("Invalid duplex response pairs")
    allowed=set()
    for pair in pairs:
        if not isinstance(pair,list) or len(pair)!=2 or any(type(i)is not int for i in pair):raise ValueError("Invalid pair")
        a,b=pair
        if not (0<=a<b<len(dialogue.turns) and b==a+2 and dialogue.turns[a].speaker==dialogue.turns[b].speaker=="assistant"
                and dialogue.turns[a+1].kind=="interruption" and dialogue.turns[a+1].speaker.startswith("user_")
                and dialogue.turns[a+1].responds_to==[a] and a+1 in dialogue.turns[b].responds_to):raise ValueError("Pair lacks actual immediate interrupted-response evidence")
        allowed.add((a,b))
    identity_pairs=prof.get("identity_response_pairs",[])
    if not isinstance(identity_pairs,list):raise ValueError("Invalid identity response pairs")
    for pair in identity_pairs:
        if not isinstance(pair,list) or len(pair)!=2 or any(type(i)is not int for i in pair):raise ValueError("Invalid identity pair")
        a,b=pair
        if not 0<=a<b<len(dialogue.turns) or b!=a+2:raise ValueError("Invalid identity pair extent")
        question,answer,ack=dialogue.turns[a:b+1]
        if not (question.speaker==ack.speaker=="assistant" and question.kind==answer.kind==ack.kind=="speech"
                and answer.speaker.startswith("user_") and question.addressee==answer.speaker==ack.addressee
                and answer.addressee=="assistant" and answer.responds_to==[a] and a+1 in ack.responds_to):raise ValueError("Identity pair lacks actual same-user question/answer/ack evidence")
        allowed.add((a,b))
    return [b-a-1 for a,b in zip(positions,positions[1:]) if (a,b) not in allowed]


def rhythm_valid(dialogue,prof):
    try:
        filtered=rhythm_intervals(dialogue,prof)
        positions=[i for i,t in enumerate(dialogue.turns) if t.speaker=="assistant"]
        raw=[b-a-1 for a,b in zip(positions,positions[1:])]
        return rhythm_ok(prof["intensity"],filtered if prof["intensity"]=="low" else raw)
    except (ValueError,KeyError,TypeError):return False


_RHYTHM = {
    "low": "低频介入，两次之间要隔较多轮（≥3 轮其他说话人的发言）",
    "mid": "大约每 2~3 轮其他说话人的发言介入一次",
    "high": "大约每 1~2 轮其他说话人的发言就介入一次（参与度明显高于其他档）",
}


def profile_block(prof: dict[str, Any]) -> str:
    """把计划转成 prompt 里的硬性要求块。"""
    k = prof["assistant_turns"]
    positions = prof.get("target_turns", [])
    positions = "、".join(str(p) for p in positions) if positions else "均匀分布"
    labels = "、".join(f"第{i + 1}次 {TRIGGER_LABELS[t]}" for i, t in enumerate(prof["triggers"]))
    return f"""
本条对话的 assistant 参与计划（硬性要求，请严格遵守；**不按此计划会被判为不合格剧本**）：
- assistant 必须在对话中**真实出场 {k} 次**（每次都有 speaker="assistant" 的轮；不得只被提及而不出场）；
- **建议发言位置**：约第 {positions} 轮（轮号从 0 起，允许 ±1 轮）；
- **介入节奏**：{_RHYTHM[prof["intensity"]]}；
- 若交互图声明 duplex_response_pairs，被打断的 AI 半句与紧接用户新限制的 AI 回应是一个响应对，允许紧邻；原始次数和全部间隔仍如实记录，其他位置遵守参与节奏；
- 若交互图声明 identity_response_pairs，仅问称呼→同一用户自报→AI立即向该用户确认这一完整交互允许紧邻；其他间隔仍须满足原节奏。
- 每次发言的触发方式依次为：{labels}；
- 每次 assistant 发言，在该轮 JSON 里加 "trigger" 字段，值为英文类型
  （cue / volunteer / invited / coordinate，按上面顺序对应）；
- 主动发言（volunteer/coordinate）要自然：顺着当前话题补充，不打断正说到一半的话；
  计划里没有 assistant 发言的轮次，assistant 一律沉默（不产出该轮）。"""


def apply_realized(dialogue: Any, prof: dict[str, Any]) -> None:
    """把实际分布写进 dialogue.profile（供批次统计与数据集标签）。"""
    # 注意：turn_id 在序列化时才赋值，这里用枚举下标（校验阶段恒为 0）
    asst_idx = [i for i, t in enumerate(dialogue.turns) if t.speaker == "assistant"]
    # 介入间隔：两次 assistant 发言之间的"非 assistant"轮数
    intervals = [asst_idx[i] - asst_idx[i - 1] - 1 for i in range(1, len(asst_idx))]
    prof["realized"] = {
        "assistant_turns": len(asst_idx),
        "intervals": intervals,
        "rhythm_intervals_excluding_verified_response_pairs":rhythm_intervals(dialogue,prof),
        "first_at_turn": asst_idx[0] if asst_idx else None,
        "triggers": [t.trigger for t in dialogue.turns if t.speaker == "assistant"],
    }
    dialogue.profile = prof
