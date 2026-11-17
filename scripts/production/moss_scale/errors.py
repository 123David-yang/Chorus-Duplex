"""Produce useful fixed diagnostics without recording arbitrary exception text."""
from __future__ import annotations

import json
import re


_HINTS = (
    ("台词编号与事件计划不一致", "turn_ids", "台词编号应与事件计划完全一致。"),
    ("缺少两段台词", "missing_parts", "指定轮次需要两个非空台词片段。"),
    ("未完成的第一段", "incomplete_start", "未完成的第一段须在尚未说完的位置停住。"),
    ("附和", "backchannel", "检查附和长度、对象和已说出口的锚点。"),
    ("同时请求", "simultaneous_requests", "检查两人请求及其共同回应。"),
    ("此前本人明确偏好冲突", "choice_preference_conflict", "同一角色的选择不得无故与此前明确偏好相反。"),
    ("犹豫接话须标出尚未说完的位置", "handoff_start", "犹豫接话的未完成语句须以省略号或破折号收尾。"),
    ("犹豫接话", "hesitation_handoff", "检查未完成话轮与助手的承接关系。"),
    ("角色编号", "spoken_metadata", "不要在台词中读出内部角色编号。"),
    ("generation_budget_exhausted", "tts_budget", "语音生成达到长度上限。"),
    ("LLM network request failed", "llm_network", "剧本服务网络请求失败。"),
    ("LLM response was incomplete", "llm_incomplete_response", "剧本服务返回了不完整响应。"),
    ("LLM request blocked a cross-origin redirect", "llm_redirect_blocked", "剧本服务重定向被安全限制拦截。"),
)


def safe_diagnostic(exc: Exception) -> dict:
    name = type(exc).__name__
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", name):
        name = "Error"
    if isinstance(exc, json.JSONDecodeError):
        return {"type": name, "code": "invalid_json", "hint": "输出须是有效 JSON。"}
    text = str(exc)
    if re.fullmatch(r"LLM API HTTP \d{3}", text):
        return {"type": name, "code": "llm_http", "hint": text}
    for needle, code, hint in _HINTS:
        if needle in text:
            return {"type": name, "code": code, "hint": hint}
    return {"type": name, "code": "other", "hint": "生成或交付失败；异常原文未写入日志。"}
