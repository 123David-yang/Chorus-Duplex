"""多 Agent 路径的提示词模板（prompts.py）—— 导演 / 用户角色扮演 / 助手应答。

仅被 multiagent/generate.py 使用（2026-09-16 从 text/prompts.py 迁入：按"唯一使用方"归属）。
主链路的直出模板见 scripts/text/oneshot_agent/prompts.py。
"""
from __future__ import annotations

import json
import re


def director_prompt(scenario: str, num_users: int, persona_pool: list[str] | None = None) -> str:
    """导演 Agent：根据场景设计剧本（参与者 + 控制参数），输出 JSON。

    persona_pool: 从 PersonaHub 抽样的英文人物背景（导演直接选用，不现场发挥）。
    """
    if persona_pool:
        pool_text = "\n".join(f"- {p}" for p in persona_pool)
        return f"""你是双工多人语音对话的导演。请为以下场景设计一份对话剧本。

场景类型：{scenario}
参与者数量：{num_users} 位用户（id 从 user_01 开始，依次 user_02、user_03…）+ 1 个 AI 助手（assistant）

人物背景池（必须从下面直接挑选 {num_users} 位，不要自创人物）：
{pool_text}

要求：
- 把选中人物的英文背景转换为中文口语化人设（保留职业、性格、生活背景，可添加说话风格）。
- 设定与人物背景相符的对话场景（setting）和主题（topic）。

输出 JSON（只输出 JSON，不要任何其他内容）：
{{
  "setting": "对话发生的场景",
  "topic": "对话主题",
  "participants": [
    {{"id": "user_01", "persona": "中文人设", "style": "说话风格", "speech_rate": 0.5~1.5 之间的数字}}
  ]
}}"""
    return f"""你是双工多人语音对话的导演。请为以下场景设计一份对话剧本。

场景类型：{scenario}
参与者数量：{num_users} 位用户（id 从 user_01 开始，依次 user_02、user_03…）+ 1 个 AI 助手（assistant）

输出 JSON（只输出 JSON，不要任何其他内容）：
{{
  "setting": "对话发生的场景，如：家庭晚餐后的客厅",
  "topic": "对话主题，如：周末去哪里玩",
  "participants": [
    {{"id": "user_01", "persona": "一句话人设（身份+性格）", "style": "说话风格", "speech_rate": 0.5~1.5 之间的数字}}
  ]
}}"""


def p1_user_roleplay_multi(pid: str, persona: dict, participants_desc: str) -> str:
    return (f"你只扮演 {pid}，自己的私有人设：{json.dumps(persona, ensure_ascii=False)}。"
            "其他角色的背景不能假定已知。只根据已说出口的历史回应。"
            '只输出 JSON，含 text、addressee、kind、responds_to；不得替别人发言。')


def assistant_prompt_multi(participants_desc: str) -> str:
    return ('你是多人语音助手；参与者背景不是公开知识，只能使用历史台词建立的身份信息。'
            '对指向你的话回应；不要介入第三方通话。只输出当前轮 JSON，含 text、addressee、kind、responds_to。')


def parse_script(text: str) -> dict | None:
    """从导演输出中解析剧本 JSON（容忍 ```json 包裹和多余文本）。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
