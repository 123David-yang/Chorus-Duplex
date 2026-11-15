"""一体直出（generate.py）—— 单次大提示词 → 完整剧本 JSON（主链路，放量主力）。

对照 multiagent/（导演→调度器→逐轮交互）：本策略 1 次 LLM 调用出整条剧本，
打断/附和是"设计"出来的（全局规划）；校验修复见 validate.py（不合格判失败交由上层重试）。
"""
from __future__ import annotations

import json
import random
import re
import zlib

from . import lexicon
from .profile import sample_profile
from ..utils.schema import Dialogue
from ..utils import llm
from scripts.config import PROFILE_SCENARIOS, SCENARIOS, LOGS_DIR

from . import prompts
from .validate import build_dialogue


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def generate_dialogue(dialogue_id: str, scenario: str, num_users: int, turns: int,
                      persona_pool: list[str] | None = None,
                      rng=None) -> Dialogue | None:
    """一体直出：单次调用生成完整剧本（失败返回 None，由上层重试）。

    rng: 传则用于采样参与画像（可复现）；None 时按 dialogue_id 自建种子。
    """
    profile = None
    if scenario in PROFILE_SCENARIOS:
        # 按 dialogue_id 固定种子：重试（attempt）不换计划，避免"难满足的画像被筛掉"的偏倚
        seed = zlib.crc32(f"profile-{dialogue_id}".encode())
        profile = sample_profile(random.Random(seed), turns, num_users)

    # V3 keeps one-call content generation, but moves label/coverage planning out of LLM free-form output.
    from .planned import generate_planned, turn_plan
    if turn_plan(dialogue_id, scenario, num_users, turns, profile) is not None:
        return generate_planned(dialogue_id, scenario, num_users, turns, profile, persona_pool)
    bc_examples = lexicon.sample_backchannels(k=12)
    prompt = prompts.oneshot_script_prompt(
        scenario, SCENARIOS.get(scenario, ""), num_users, turns, persona_pool, bc_examples,
        profile, dialogue_id=dialogue_id)
    out = llm.chat(prompt, "请直接输出完整剧本 JSON", temperature=0.9, max_tokens=4096)
    script = _parse_json(out)
    if not script:
        print(f"    ⚠ {dialogue_id}: 输出不是合法 JSON（{len(out)} 字符）")
        return None
    folder = LOGS_DIR / "oneshot_v2"
    folder.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", dialogue_id)
    with (folder / (safe_id + ".jsonl")).open("a", encoding="utf-8") as trace:
        trace.write(json.dumps({"stage":"generated", "script":script}, ensure_ascii=False) + "\n")
    d, fixes = build_dialogue(script, dialogue_id, scenario, profile, turns_requested=turns)
    # One bounded correction, guided by concrete violations; a failed correction remains rejected.
    from ..quality import audit, coverage_plan
    violations = audit(d, coverage_plan(dialogue_id, turns))["issues"] if d is not None else fixes
    if violations:
        feedback = json.dumps({"original":script,"violations":violations}, ensure_ascii=False)
        corrected = llm.chat(prompt, "修正下面原剧本的明确错误；保持人物/话题，只输出完整 JSON。必须完成本条模块，不可用其他模块替代。\n" + feedback,
                             temperature=0.3, max_tokens=4096)
        revised = _parse_json(corrected)
        if revised:
            with (folder / (safe_id + ".jsonl")).open("a", encoding="utf-8") as trace:
                trace.write(json.dumps({"stage":"bounded_revision","script":revised,"feedback":violations}, ensure_ascii=False) + "\n")
            d, fixes = build_dialogue(revised, dialogue_id, scenario, profile, turns_requested=turns)
    if fixes:
        print(f"    ⚠ {dialogue_id} 校验修复 {len(fixes)} 处: {fixes[:3]}")
    if d is not None and len(d.participants) != num_users:
        print(f"    invalid participant count for {dialogue_id}")
        return None
    if profile and d is not None:
        print(f"    📊 {dialogue_id} 参与画像 {profile['intensity']}：计划 assistant {profile['assistant_turns']} 次"
              f" → 实际 {profile['realized']['assistant_turns']} 次，间隔 {profile['realized']['intervals']}")
    return d
