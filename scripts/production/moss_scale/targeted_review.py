"""Optional, bounded review of a locally flagged electrical-fault response.

This is a limited writing aid, not a safety classifier. It touches only new
plain assistant turns, never existing deliveries or event/timing metadata.
"""
from __future__ import annotations

import json
import re

from scripts.text.quality import electrical_fault_answer_needs_review, unsupported_assistant_action
from scripts.text.utils.llm import chat


_PRIVATE_TEXT = re.compile(
    r"(?i)(?:(?<![A-Z0-9_])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9_])|"
    r"(?<![A-Z0-9_])bearer\s+[A-Z0-9._~+/=-]{8,}|"
    r"(?<![A-Z0-9_])(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|secret|authorization)\s*[:=]\s*\S{4,}|"
    r"(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![A-Z0-9_-])|"
    r"(?<![A-Z0-9])(?:\+?86[- ]?)?1[3-9]\d(?:[- ]?\d){8}(?![A-Z0-9])|"
    r"(?<![A-Z0-9])(?:[A-Z]:[\\/]|/home/|/Users/|/root/|\\\\))"
)


def _has_private_text(value: str) -> bool:
    """Conservative filter; unknown credential formats and identities may be missed."""
    return bool(_PRIVATE_TEXT.search(value))


def review_once(draft: dict, slots: list[dict], language: str) -> tuple[dict, dict | None]:
    """Review at most one pair and make at most one extra model call."""
    turns = draft["turns"]
    skipped_index = None
    for index in range(1, len(turns)):
        question, answer = turns[index - 1], turns[index]
        address = question.get("addressee")
        addresses = address if isinstance(address, list) else [address]
        if (not question["speaker"].startswith("user_") or answer["speaker"] != "assistant"
                or not ({"assistant", "all"} & set(addresses)) or index - 1 not in answer.get("responds_to", [])
                or not electrical_fault_answer_needs_review(question["text"], answer["text"])):
            continue
        slot = slots[index]
        if (answer["kind"] != "speech" or slot.get("form") or slot.get("fixed_text")
                or answer.get("interaction") or (index + 1 < len(turns)
                    and turns[index + 1]["kind"] in {"backchannel", "interruption"})):
            skipped_index = index
            continue
        def assistant_visible(turn):
            if turn.get("kind") == "side_talk":
                return False
            target = turn.get("addressee")
            targets = target if isinstance(target, list) else [target]
            return turn["speaker"] == "assistant" or bool({"assistant", "all"} & set(targets))

        context = [{"speaker": turn["speaker"], "text": turn["text"]}
                   for turn in turns[max(0, index - 3):index + 1] if assistant_visible(turn)]
        if any(_has_private_text(turn["text"]) for turn in context):
            return {"status": "sensitive_context_skipped", "turn_id": index, "calls": 0}, None
        called = False
        try:
            from .upload_delivery import _assert_public_safe
            _assert_public_safe(context)
            system = (
                "Review one assistant response in a synthetic conversation. Identify whether an explicit electrical "
                "fault is still unaddressed, considering if the speaker already stopped using the appliance or is "
                "speaking hypothetically. If a change is needed, preserve the response's purpose and add concise, "
                "context-appropriate advice to stop use, disconnect power only if safe, and seek qualified help. "
                "Never suggest touching a live device or claim the assistant performed a real action. "
                "Use the same language as the original response. Return only a JSON object with decision "
                "equal to 'keep', 'rewrite', or 'uncertain', and text containing the complete replacement only for rewrite."
            )
            called = True
            raw = chat(system, json.dumps({"language": language, "context": context}, ensure_ascii=False),
                       temperature=.2, max_tokens=320, json_output=True)
            result = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
            decision = result.get("decision") if isinstance(result, dict) else None
            if decision in ("keep", "uncertain"):
                return {"status": decision, "turn_id": index, "calls": 1}, None
            replacement = result.get("text") if decision == "rewrite" else None
            if not isinstance(replacement, str):
                raise ValueError("Invalid review response")
            replacement = replacement.strip()
            if (not replacement or len(replacement) > (360 if language == "en" else 190)
                    or _has_private_text(replacement)
                    or electrical_fault_answer_needs_review(question["text"], replacement)
                    or unsupported_assistant_action(replacement)
                    or re.search(r"(?:不要|不必|不用|无需|无须|不需要|没必要)(?:停止使用|停用)|继续(?:使用|通电)|"
                                 r"\b(?:keep using|do not stop using|don't stop using|no need to stop using|"
                                 r"keep (?:it )?plugged in|leave (?:it )?plugged in)\b", replacement, re.I)):
                raise ValueError("Review response requires further inspection")
            _assert_public_safe(replacement)
        except Exception:
            return {"status": "unavailable_or_uncertain", "turn_id": index, "calls": int(called)}, None
        previous = answer["text"]
        answer["text"] = replacement
        return {"status": "rewritten", "turn_id": index, "calls": 1}, {"original": previous, "revised": replacement}
    if skipped_index is not None:
        return {"status": "skipped_structural", "turn_id": skipped_index, "calls": 0}, None
    return {"status": "not_triggered", "calls": 0}, None
