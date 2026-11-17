"""Regression tests for existing MOSS annotations and simultaneous requests."""
import json
import http.client
import io
import os
import random
import socket
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.production.moss_scale.render import render
from scripts.production.moss_scale.script import dialogue_id, generate, validate, incomplete_has_content
from scripts.production.moss_scale.slots import draft_from_slots, plan_slots, shorten_backchannel_prefix
from scripts.text.quality import (ai_request_candidates, audit, repeated_incomplete_start,
                                  numeric_safety_advice_needs_review, prior_preference_conflict,
                                  unsupported_assistant_action, possible_group_decision,
                                  device_capability_needs_review, possible_two_option_plan,
                                  microphone_answer_needs_review, electrical_fault_answer_needs_review)
from scripts.text.simultaneous_choices import (ALTERNATE_CHOICES, CHOICE_OPTIONS, CHOICE_CONFIRMATIONS, choice_option_sets,
                                               generation_choice_option_sets, choice_request_texts,
                                               generation_choice_request_texts)
from scripts.text.utils.schema import Dialogue
from scripts.text.utils.llm import chat, capture_calls


class FakeSynth:
    def line(self, content, voice, offset=0):
        return np.full(24000, 0.1, dtype=np.float32)


def draft(language, mode, seed=15):
    users = ["user_01", "user_02"]
    slots = plan_slots(users, ["simultaneous_requests"], False, False, False,
                       False, False, random.Random(seed), 12, language)
    context = {"场景": "planning", "领域": "travel", "关系": "friends"}

    def chat(system, request, **kwargs):
        turns = json.loads(request)["待写轮次"]
        answers = {}
        for key, turn in turns.items():
            interaction = turn.get("interaction")
            if interaction == "simultaneous_requests":
                if language == "en":
                    text = "Can we leave at eight?" if "overlap_with" not in turn else "Can we leave at nine?"
                else:
                    text = "我们八点出发可以吗？" if "overlap_with" not in turn else "我们九点出发可以吗？"
            elif interaction == "conflict_clarification":
                text = "I heard eight and nine. Which time should we choose?" if language == "en" else "我听到八点和九点，你们想选哪个时间？"
            elif "两个互斥" in turn.get("cue", ""):
                text = "We can leave at eight or nine. Which works?" if language == "en" else "我们可以八点或者九点出发，选哪个？"
            else:
                text = "Let's discuss our plan." if language == "en" else "我们来商量一下安排。"
            answers[key] = text
        return json.dumps(answers, ensure_ascii=False)

    with patch("scripts.production.moss_scale.slots.chat", side_effect=chat):
        result = draft_from_slots(slots, users, context, mode, [], language)
    validate(result, users, [], False, language=language, simultaneous_requests=True)
    result["dialogue_id"] = "simultaneous-test"
    result["profile"] = {"language": language, "voice_assignments": {
        "assistant": "assistant", "user_01": "voice1", "user_02": "voice2"
    }}
    for i, turn in enumerate(result["turns"]):
        turn.update(turn_id=i, gap_before_ms=-350 if turn.get("overlap_with") is not None else 200,
                    insert_ratio=None, events=[], trigger=None)
    return result


class DialogueQualityTests(unittest.TestCase):
    def test_english_sentence_boundary_is_not_sent_as_leading_tts_punctuation(self):
        calls = []
        class CaptureSynth(FakeSynth):
            def line(self, content, voice, offset=0):
                calls.append(content)
                return super().line(content, voice, offset)
        d = {"dialogue_id": "boundary", "profile": {"language": "en", "voice_assignments": {"assistant": "host", "user_01": "listener"}},
             "turns": [{"turn_id": 0, "speaker": "assistant", "kind": "speech", "addressee": "user_01", "text": "Check the label. Then compare the rating.", "gap_before_ms": 0},
                       {"turn_id": 1, "speaker": "user_01", "kind": "backchannel", "addressee": "assistant", "text": "Got it.", "anchor_text": "Check the label", "responds_to": [0], "gap_before_ms": 0}]}
        with tempfile.TemporaryDirectory() as folder:
            render(d, CaptureSynth(), Path(folder)/"clean.flac", Path(folder)/"timeline.json")
        self.assertEqual(calls[:2], ["Check the label", "Then compare the rating."])
        calls.clear()
        d["turns"][0]["text"] = "Check the label, .5 is the measured value."
        with tempfile.TemporaryDirectory() as folder:
            render(d, CaptureSynth(), Path(folder)/"clean.flac", Path(folder)/"timeline.json")
        self.assertEqual(calls[1], ".5 is the measured value.")

    def test_targeted_model_rewrite_remains_marked_for_review(self):
        data = draft("en", "direct")
        data["profile"]["targeted_safety_review"] = {"status": "rewritten", "turn_id": 1, "calls": 1}
        result = audit(Dialogue.from_jsonl_line(json.dumps(data)))
        self.assertIn("targeted_review_unverified", [warning["code"] for warning in result["warnings"]])
        self.assertTrue(result["passed"])

    def test_narrow_semantic_warnings_are_hints_not_rejections(self):
        self.assertTrue(device_capability_needs_review("A headset jack covers listening and speaking."))
        self.assertFalse(device_capability_needs_review("Check whether the headset has a microphone."))
        self.assertTrue(possible_two_option_plan("我们只有一个锅，没法同时煮饭和煮面。", "那得按顺序来。"))
        self.assertFalse(possible_two_option_plan("我们只有一个锅。", "先确认设备型号。"))
        self.assertTrue(microphone_answer_needs_review("How do the microphones compare?", "A wired connection can be stable."))
        self.assertFalse(microphone_answer_needs_review("How do the microphones compare?", "Check each microphone model first."))
        faulty = "我的电水壶接触不太好，烧水时得用手按着。"
        self.assertTrue(electrical_fault_answer_needs_review(faulty, "那你想买多大容量的？"))
        self.assertFalse(electrical_fault_answer_needs_review(faulty, "建议先停用，安全时断电并请专业人员检查。"))
        self.assertTrue(electrical_fault_answer_needs_review("The kettle won’t turn off.", "Don't unplug it."))
        self.assertFalse(electrical_fault_answer_needs_review("The kettle has no smoke.", "What size is it?"))

    def test_partial_agreement_does_not_automatically_prove_group_consensus(self):
        self.assertTrue(possible_group_decision("Okay, that's settled—let's move on."))
        self.assertFalse(possible_group_decision("Is that settled?"))
        data = draft("en", "direct")
        data["participants"].append({"id": "user_03", "persona": "friend", "style": "natural"})
        data["turns"][-1]["text"] = "Okay, that's settled—let's move on."
        result = audit(Dialogue.from_jsonl_line(json.dumps(data)))
        self.assertIn("possible_group_consensus", [warning["code"] for warning in result["warnings"]])
        self.assertTrue(result["passed"])  # A warning is not proof that the decision is invalid.

    def test_incomplete_needs_content_before_the_hesitation(self):
        for text in ("我想一下……", "嗯，那个……"):
            self.assertFalse(incomplete_has_content(text, "zh"))
        for text in ("Um...", "Let me think…"):
            self.assertFalse(incomplete_has_content(text, "en"))
        self.assertTrue(incomplete_has_content("出发的时间是……那个……", "zh"))
        self.assertTrue(incomplete_has_content("The departure time is… um…", "en"))

    def test_explicit_prior_preference_reversal_is_detected(self):
        turns = [{"speaker": "user_03", "text": "I actually prefer Saturday because I have plans Sunday afternoon."}]
        options = ["clean the house on Saturday", "clean the house on Sunday"]
        self.assertTrue(prior_preference_conflict(turns, "user_03", 1, options, 1, "en"))
        self.assertFalse(prior_preference_conflict(turns, "user_03", 1, options, 0, "en"))
        turns = [{"speaker": "user_01", "text": "我更倾向上午，下午还要处理材料。"}]
        self.assertTrue(prior_preference_conflict(turns, "user_01", 1, ["上午开会", "下午开会"], 1, "zh"))

    def test_future_reminder_claims_are_flagged_but_current_reminder_is_allowed(self):
        self.assertTrue(unsupported_assistant_action("到点我来提醒你们对照两栏。"))
        self.assertTrue(unsupported_assistant_action("I'll remind you tomorrow."))
        self.assertFalse(unsupported_assistant_action("我提醒一下，先核对时间。"))

    def test_quantified_food_safety_advice_is_flagged_for_review_not_rejection(self):
        self.assertTrue(numeric_safety_advice_needs_review("Steam the dumplings for 10–15 minutes."))
        self.assertTrue(numeric_safety_advice_needs_review("先蒸10到15分钟。"))
        self.assertTrue(numeric_safety_advice_needs_review("冰块超过两厘米就先敲小。"))
        self.assertTrue(numeric_safety_advice_needs_review("The standard return window is thirty days."))
        self.assertFalse(numeric_safety_advice_needs_review("We'll meet at 9 in the morning."))

    def test_unique_run_tags_preserve_legacy_ids_and_separate_new_runs(self):
        with patch.dict(os.environ, {}, clear=True):
            old_id = dialogue_id(17, "en", "moss_onnx")
            first = dialogue_id(17, "en", "moss_onnx", run_tag="gpu-a")
            second = dialogue_id(17, "en", "moss_onnx", run_tag="gpu-b")
            self.assertEqual(old_id, "moss-scale-en-20261115-000017")
            self.assertEqual(len({old_id, first, second}), 3)
            with self.assertRaises(ValueError):
                dialogue_id(17, "en", "moss_onnx", run_tag="../bad")

    def test_simultaneous_choice_templates_vary_and_remain_valid(self):
        for language in ("zh", "en"):
            seen = {"choice_request": set(), "choice_setup": set(), "choice_confirmation": set()}
            outcomes = set()
            for seed in range(40):
                dialogue = draft(language, "direct", seed=seed)
                for turn in dialogue["turns"]:
                    if turn.get("interaction") in seen:
                        seen[turn["interaction"]].add(turn["text"])
                    if turn.get("interaction") == "choice_acknowledgement":
                        outcomes.add(turn["choice_outcome"])
            for interaction, texts in seen.items():
                self.assertGreaterEqual(len(texts), 3, (language, interaction))
            self.assertEqual(outcomes, {"agreement", "open"})

    def test_each_supported_topic_has_two_distinct_option_pairs(self):
        for language in ("zh", "en"):
            self.assertEqual(set(CHOICE_OPTIONS[language]), set(ALTERNATE_CHOICES[language]))
            for topic, first in CHOICE_OPTIONS[language].items():
                second = ALTERNATE_CHOICES[language][topic]
                self.assertEqual(len(set(first)), 2, topic)
                self.assertEqual(len(set(second)), 2, topic)
                self.assertNotEqual(first, second, topic)
                self.assertEqual(len(choice_option_sets(language, topic)), 3, topic)
        self.assertNotIn(("看电影", "看纪录片"), generation_choice_option_sets("zh", "娱乐影视"))
        self.assertIn(("看电影", "看纪录片"), choice_option_sets("zh", "娱乐影视"))
        choices = ("在室内办活动", "在室外办活动")
        legacy = "我们在在室内办活动和在室外办活动之间犹豫，你有什么建议？"
        self.assertIn(legacy, choice_request_texts("zh", choices))
        self.assertNotIn(legacy, generation_choice_request_texts("zh", choices))

    def test_schema_round_trip_keeps_moss_optional_fields(self):
        original = draft("en", "direct")
        d = Dialogue.from_jsonl_line(json.dumps(original))
        restored = json.loads(d.to_jsonl_line())
        self.assertEqual([t.get("interaction") for t in original["turns"]],
                         [t.get("interaction") for t in restored["turns"]])
        self.assertEqual([t.get("overlap_with") for t in original["turns"]],
                         [t.get("overlap_with") for t in restored["turns"]])
        self.assertEqual([t.get("choice_outcome") for t in original["turns"]],
                         [t.get("choice_outcome") for t in restored["turns"]])

    def test_previous_choice_confirmation_format_remains_valid(self):
        for language in ("zh", "en"):
            dialogue = draft(language, "direct")
            turns = dialogue["turns"]
            ack_index = next(i for i, turn in enumerate(turns) if turn.get("interaction") == "choice_acknowledgement")
            resolution_index = ack_index - 1
            del turns[ack_index]
            for i, turn in enumerate(turns):
                turn["turn_id"] = i
                turn["responds_to"] = [ref - int(ref > ack_index) for ref in turn["responds_to"] if ref != ack_index]
                if turn.get("overlap_with") is not None and turn["overlap_with"] > ack_index:
                    turn["overlap_with"] -= 1
            confirmation = turns[ack_index]
            confirmation.pop("choice_outcome", None)
            confirmation["text"] = CHOICE_CONFIRMATIONS[language][0]
            confirmation["addressee"] = turns[resolution_index]["speaker"]
            confirmation["responds_to"] = [resolution_index]
            validate(dialogue, ["user_01", "user_02"], [], False, language=language, simultaneous_requests=True)
            self.assertTrue(audit(Dialogue.from_jsonl_line(json.dumps(dialogue)))["passed"])

    def test_direct_and_interactive_bilingual_overlap_and_clarification(self):
        for language in ("zh", "en"):
            for mode in ("direct", "interactive"):
                with self.subTest(language=language, mode=mode):
                    dialogue = draft(language, mode)
                    self.assertTrue(audit(Dialogue.from_jsonl_line(json.dumps(dialogue)))["passed"])
                    with tempfile.TemporaryDirectory() as temp:
                        root = Path(temp)
                        render(dialogue, FakeSynth(), root / "audio.flac", root / "timeline.json")
                        timeline = json.loads((root / "timeline.json").read_text(encoding="utf-8"))
                    second = next(row for row in timeline if "overlap_with" in row)
                    first = timeline[second["overlap_with"]]
                    response = timeline[second["turn_id"] + 1]
                    self.assertLess(second["start_sec"], first["end_sec"])
                    self.assertGreaterEqual(response["start_sec"], max(first["end_sec"], second["end_sec"]))
                    self.assertEqual(response["speaker"], "assistant")
                    self.assertEqual(dialogue["turns"][response["turn_id"]]["responds_to"],
                                     [first["turn_id"], second["turn_id"]])
                    self.assertTrue(audit(Dialogue.from_jsonl_line(json.dumps(dialogue)))["passed"])

    def test_simultaneous_requests_need_two_speakers_and_a_joint_reply(self):
        dialogue = draft("zh", "direct")
        turns = dialogue["turns"]
        second = next(t for t in turns if t.get("overlap_with") is not None)
        first = turns[second["overlap_with"]]
        second["speaker"] = first["speaker"]
        with self.assertRaises(ValueError):
            validate(dialogue, ["user_01", "user_02"], [], False, simultaneous_requests=True)

    def test_audit_rejects_contradictory_choice_or_wrong_confirmation_addressee(self):
        for language in ("zh", "en"):
            dialogue = draft(language, "direct")
            resolution = next(t for t in dialogue["turns"] if t.get("interaction") == "choice_resolution")
            selected = resolution["choice_index"]
            resolution["text"] = (("我选第二个。", "我选第一个。") if language == "zh"
                                  else ("I'll choose the second option.", "I'll choose the first option."))[selected]
            report = audit(Dialogue.from_jsonl_line(json.dumps(dialogue)))
            self.assertIn("invalid_simultaneous_requests", {issue["code"] for issue in report["issues"]})

            dialogue = draft(language, "direct")
            resolution = next(t for t in dialogue["turns"] if t.get("interaction") == "choice_resolution")
            confirmation = next(t for t in dialogue["turns"] if t.get("interaction") == "choice_confirmation")
            confirmation["addressee"] = next(user for user in ("user_01", "user_02") if user != resolution["speaker"])
            report = audit(Dialogue.from_jsonl_line(json.dumps(dialogue)))
            self.assertIn("invalid_simultaneous_requests", {issue["code"] for issue in report["issues"]})

    def test_quality_rejects_unpaired_or_nonoverlapping_requests(self):
        dialogue = draft("en", "direct")
        second = next(t for t in dialogue["turns"] if t.get("overlap_with") is not None)
        second["gap_before_ms"] = 100
        report = audit(Dialogue.from_jsonl_line(json.dumps(dialogue)))
        self.assertIn("invalid_simultaneous_requests", {issue["code"] for issue in report["issues"]})

    def test_vague_or_unrelated_setup_does_not_pass_as_two_options(self):
        dialogue = draft("en", "direct")
        request = next(t for t in dialogue["turns"] if t.get("interaction") == "choice_request")
        request["text"] = "Should we pick a restaurant, or should we not decide anything yet?"
        with self.assertRaises(ValueError):
            validate(dialogue, ["user_01", "user_02"], [], False, language="en", simultaneous_requests=True)
        report = audit(Dialogue.from_jsonl_line(json.dumps(dialogue)))
        self.assertIn("invalid_simultaneous_requests", {issue["code"] for issue in report["issues"]})

    def test_users_must_select_opposite_options(self):
        dialogue = draft("en", "direct")
        second = next(t for t in dialogue["turns"] if t.get("overlap_with") is not None)
        first = dialogue["turns"][second["overlap_with"]]
        second["choice_index"] = first["choice_index"]
        with self.assertRaises(ValueError):
            validate(dialogue, ["user_01", "user_02"], [], False, language="en", simultaneous_requests=True)
        report = audit(Dialogue.from_jsonl_line(json.dumps(dialogue)))
        self.assertIn("invalid_simultaneous_requests", {issue["code"] for issue in report["issues"]})

    def test_request_text_must_agree_with_choice_index(self):
        dialogue = draft("en", "direct")
        second = next(t for t in dialogue["turns"] if t.get("overlap_with") is not None)
        first = dialogue["turns"][second["overlap_with"]]
        first["text"] = second["text"]
        with self.assertRaises(ValueError):
            validate(dialogue, ["user_01", "user_02"], [], False, language="en", simultaneous_requests=True)
        report = audit(Dialogue.from_jsonl_line(json.dumps(dialogue)))
        self.assertIn("invalid_simultaneous_requests", {issue["code"] for issue in report["issues"]})

    def test_single_assistant_addressee_list_is_supported(self):
        dialogue = draft("en", "direct")
        second = next(t for t in dialogue["turns"] if t.get("overlap_with") is not None)
        dialogue["turns"][second["overlap_with"]]["addressee"] = ["assistant"]
        validate(dialogue, ["user_01", "user_02"], [], False, language="en", simultaneous_requests=True)
        self.assertTrue(audit(Dialogue.from_jsonl_line(json.dumps(dialogue)))["passed"])

    def test_rule_audit_accepts_setup_paraphrase_with_the_same_two_choices(self):
        dialogue = draft("en", "direct")
        setup = next(t for t in dialogue["turns"] if t.get("interaction") == "choice_setup")
        first, second = setup["choice_options"]
        setup["text"] = f"Should we {first} or {second}?"
        self.assertTrue(audit(Dialogue.from_jsonl_line(json.dumps(dialogue)))["passed"])

    def test_a_user_must_answer_before_assistant_continues(self):
        dialogue = draft("en", "direct")
        second = next(t for t in dialogue["turns"] if t.get("overlap_with") is not None)
        resolution = dialogue["turns"][second["turn_id"] + 2]
        resolution["speaker"] = "assistant"
        with self.assertRaises(ValueError):
            validate(dialogue, ["user_01", "user_02"], [], False, language="en", simultaneous_requests=True)
        report = audit(Dialogue.from_jsonl_line(json.dumps(dialogue)))
        self.assertIn("missing_conflict_clarification", {issue["code"] for issue in report["issues"]})

    def test_resolution_follows_responders_initial_preference(self):
        users = ["user_01", "user_02"]
        for seed in range(100):
            slots = plan_slots(users, ["simultaneous_requests"], False, False, False,
                               False, False, random.Random(seed), 12, "en")
            requests = [t for t in slots if t.get("interaction") == "simultaneous_requests"]
            resolution = next(t for t in slots if t.get("interaction") == "choice_resolution")
            original = next(t for t in requests if t["speaker"] == resolution["speaker"])
            self.assertEqual(resolution["choice_index"], original["choice_index"], f"seed {seed}")

    def test_planned_event_hosts_do_not_create_adjacent_assistant_turns(self):
        users = ["user_01", "user_02"]
        events = ["backchannel", "interruption", "incomplete", "side_talk", "simultaneous_requests"]
        for seed in range(100):
            slots = plan_slots(users, events, True, False, True,
                               False, True, random.Random(seed), 20, "en")
            for before, after in zip(slots, slots[1:]):
                self.assertFalse(before["speaker"] == after["speaker"] == "assistant", f"seed {seed}")

    def test_choice_setup_has_a_user_request_and_closes_previous_user_turn(self):
        users = ["user_01", "user_02"]
        events = ["backchannel", "interruption", "incomplete", "side_talk", "simultaneous_requests"]
        for seed in range(100):
            slots = plan_slots(users, events, True, False, True,
                               False, True, random.Random(seed), 20, "en")
            setup = next(i for i, turn in enumerate(slots) if turn.get("interaction") == "choice_setup")
            request = slots[setup-1]
            self.assertEqual(request["interaction"], "choice_request", f"seed {seed}")
            self.assertEqual(slots[setup]["responds_to"], [setup-1])
            self.assertTrue(all(option in request["fixed_text"] for option in request["choice_options"]))
            previous = slots[setup-2]
            self.assertTrue(previous["speaker"] == "assistant" or previous["kind"] == "backchannel",
                            f"seed {seed}")

    def test_render_rejects_binding_changes_between_turns(self):
        dialogue = draft("en", "direct")
        voices = dialogue["profile"]["voice_assignments"]
        bindings = {voice: {"kind": "preset", "speaker": voice} for voice in voices.values()}
        dialogue["profile"]["voice_bindings"] = bindings
        seen = 0

        def snapshot(ids):
            nonlocal seen
            seen += 1
            voice = next(iter(ids))
            value = bindings[voice] if seen == 1 else {"kind": "preset", "speaker": "changed"}
            return {voice: value}

        with tempfile.TemporaryDirectory() as temp, \
             patch("scripts.text.voices.voice_bindings_snapshot", side_effect=snapshot):
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "Voice binding changed"):
                render(dialogue, FakeSynth(), root / "audio.flac", root / "timeline.json")

    def test_explicit_real_world_execution_is_flagged_without_blocking_advice(self):
        self.assertTrue(unsupported_assistant_action("I'll book the tickets tonight."))
        self.assertTrue(unsupported_assistant_action("I’ll create the spreadsheet and share the link."))
        self.assertTrue(unsupported_assistant_action("我已经买好油漆了。"))
        self.assertFalse(unsupported_assistant_action("You could buy the tickets tonight."))
        self.assertFalse(unsupported_assistant_action("We can check the list together."))
        self.assertFalse(unsupported_assistant_action("I'll share a few suggestions."))

    def test_repeated_incomplete_start_is_only_an_audit_warning(self):
        self.assertTrue(repeated_incomplete_start([
            "我本来想先按收入比例存，那个，那个……",
            "我本来想先按收入比例存，后来决定固定存一笔。",
        ]))
        self.assertFalse(repeated_incomplete_start(["我们明天……", "九点出发。"]))

    def test_english_requests_without_question_marks_are_review_candidates(self):
        dialogue = draft("en", "direct")
        dialogue["turns"][0]["text"] = "Please explain the options."
        self.assertIn(0, ai_request_candidates(Dialogue.from_jsonl_line(json.dumps(dialogue))))

    def test_interactive_chunks_receive_the_same_private_preferences(self):
        users = ["user_01", "user_02"]
        slots = plan_slots(users, ["simultaneous_requests"], False, False, False,
                           False, False, random.Random(11), 12, "en", topic="travel plans")
        requests = []

        def chat(system, request, **kwargs):
            data = json.loads(request)
            requests.append(data)
            return json.dumps({key: "Let's talk." for key in data["待写轮次"]})

        with patch("scripts.production.moss_scale.slots.chat", side_effect=chat):
            draft_from_slots(slots, users, {"场景": "planning", "领域": "travel plans", "关系": "friends"},
                             "interactive", [], "en")
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["全局场景线索"], requests[1]["全局场景线索"])
        self.assertEqual(requests[0]["角色私有倾向"], requests[1]["角色私有倾向"])
        self.assertEqual(len(requests[0]["角色私有倾向"]), 2)
        self.assertEqual(len(set(requests[0]["角色私有倾向"].values())), 2)

    def test_long_acknowledgement_reuses_its_complete_short_prefix(self):
        self.assertEqual(shorten_backchannel_prefix("Got it, a test patch makes sense.", "en"), "Got it")
        self.assertEqual(shorten_backchannel_prefix("The test patch can go by the doorway.", "en"),
                         "The test patch can go by the doorway.")
        self.assertEqual(shorten_backchannel_prefix("我觉得可以", "zh"), "我觉得可以")

    def test_internal_role_followed_by_chinese_characters_is_rejected(self):
        dialogue = draft("zh", "direct")
        dialogue["turns"][0]["text"] = "可以让user_01建文档。"
        with self.assertRaises(ValueError):
            validate(dialogue, ["user_01", "user_02"], [], False, simultaneous_requests=True)

    def test_cached_dialogue_rejects_a_different_tts_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            (root / "scripts" / "qwen3-scale-20261115-000001.json").write_text(
                json.dumps({"profile": {"tts_backend": "moss_onnx"}, "turns": []}), encoding="utf-8"
            )
            with patch.dict(os.environ, {"TTS_BACKEND": "qwen3"}):
                with self.assertRaisesRegex(ValueError, "different TTS backend"):
                    generate(1, root)

    def test_dialogue_ids_distinguish_backends_and_preserve_moss_names(self):
        self.assertEqual(dialogue_id(17, "en", "moss_onnx"), "moss-scale-en-20261115-000017")
        self.assertEqual(dialogue_id(17, "zh", "qwen3"), "qwen3-scale-20261115-000017")
        self.assertEqual(dialogue_id(17, "zh", "kokoro"), "kokoro-scale-20261115-000017")
        custom = dialogue_id(17, "en", "package.module:factory")
        self.assertEqual(custom, dialogue_id(17, "en", "package.module:factory"))
        self.assertNotEqual(custom, dialogue_id(17, "en", "package:module.factory"))
        self.assertNotIn("/", custom)

    def test_cached_voice_bindings_cannot_silently_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            cached = draft("zh", "direct")
            cached["profile"]["tts_backend"] = "moss_onnx"
            cached["profile"]["voice_bindings"] = {"assistant": {"kind": "preset", "speaker": "original"}}
            (root / "scripts" / "moss-scale-20261115-000001.json").write_text(
                json.dumps(cached, ensure_ascii=False), encoding="utf-8"
            )
            changed = {"assistant": {"kind": "preset", "speaker": "different"}}
            with patch.dict(os.environ, {"TTS_BACKEND": "moss_onnx"}, clear=True), \
                 patch("scripts.production.moss_scale.script.voice_bindings_snapshot", return_value=changed):
                with self.assertRaisesRegex(ValueError, "Voice bindings changed"):
                    generate(1, root)

    def test_old_cache_with_registry_requires_new_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            cached = draft("zh", "direct")
            (root / "scripts" / "moss-scale-20261115-000001.json").write_text(
                json.dumps(cached, ensure_ascii=False), encoding="utf-8"
            )
            with patch.dict(os.environ, {"TTS_BACKEND": "moss_onnx", "VOICE_REGISTRY": "registry.json"}, clear=True):
                with self.assertRaisesRegex(ValueError, "no voice bindings"):
                    generate(1, root)
            with patch.dict(os.environ, {"TTS_BACKEND": "moss_onnx"}, clear=True):
                self.assertEqual(generate(1, root)["dialogue_id"], "simultaneous-test")

    def test_existing_moss_optional_fields_and_incomplete(self):
        obj = {
            "dialogue_id": "old", "participants": [{"id": "user_01", "persona": "", "style": ""}],
            "profile": {"language": "en"},
            "turns": [
                {"speaker": "user_01", "text": "One moment...", "kind": "incomplete",
                 "addressee": "assistant", "parts": ["One moment..."], "pause_ms": 2600,
                 "resolution": "assistant_handoff"},
                {"speaker": "assistant", "text": "Would you like a suggestion?", "kind": "speech",
                 "addressee": "user_01", "responds_to": [0]},
                {"speaker": "third_party_01", "text": "Excuse me.", "kind": "side_talk",
                 "addressee": "user_01", "side_talk_entry": "unsolicited", "gap_before_ms": -180},
                {"speaker": "user_01", "text": "Yes?", "kind": "speech",
                 "addressee": "third_party_01", "responds_to": [2]},
                {"speaker": "user_01", "text": "Let's get back to it.", "kind": "speech",
                 "addressee": "assistant"},
                {"speaker": "assistant", "text": "Sure.", "kind": "speech",
                 "addressee": "user_01", "responds_to": [4]},
                {"speaker": "user_01", "text": "Thanks.", "kind": "speech", "addressee": "assistant"},
            ],
        }
        restored = Dialogue.from_jsonl_line(json.dumps(obj))
        self.assertTrue(audit(restored)["passed"])
        roundtrip = json.loads(restored.to_jsonl_line())
        self.assertEqual(roundtrip["turns"][0]["resolution"], "assistant_handoff")
        self.assertEqual(roundtrip["turns"][2]["side_talk_entry"], "unsolicited")


class LLMNetworkTests(unittest.TestCase):
    def test_remote_llm_http_is_refused_but_loopback_and_https_are_allowed(self):
        from scripts.text.utils.llm import _config
        for endpoint in ("http://example.invalid/v1", "http://192.168.1.7/v1",
                         "https://name:pass@example.invalid/v1"):
            with self.subTest(endpoint=endpoint), patch.dict(os.environ, {"LLM_API_KEY": "fake", "LLM_BASE_URL": endpoint}):
                with self.assertRaises(ValueError):
                    _config()
        for endpoint in ("http://127.0.0.1:8080/v1", "http://localhost:8080/v1", "http://[::1]:8080/v1",
                         "https://example.invalid/v1"):
            with self.subTest(endpoint=endpoint), patch.dict(os.environ, {"LLM_API_KEY": "fake", "LLM_BASE_URL": endpoint}):
                self.assertEqual(_config()[0], endpoint)

    def test_complete_json_surviving_early_close_is_used_without_a_second_paid_call(self):
        payload = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "complete reply"}}]}).encode()
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def read(self): raise http.client.IncompleteRead(payload, 7)
        class Opener:
            calls = 0
            def open(self, *args, **kwargs):
                self.calls += 1
                return Response()
        opener = Opener()
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False), \
             patch("scripts.text.utils.llm.urllib.request.build_opener", return_value=opener), capture_calls() as calls:
            self.assertEqual(chat("system", "request"), "complete reply")
        self.assertEqual(opener.calls, 1)
        self.assertTrue(calls[0]["recovered_complete_json"])

    def test_truncated_json_is_not_retried_or_treated_as_complete(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def read(self): raise http.client.IncompleteRead(b'{"choices":[', 50)
        class Opener:
            calls = 0
            def open(self, *args, **kwargs):
                self.calls += 1
                return Response()
        opener = Opener()
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False), \
             patch("scripts.text.utils.llm.urllib.request.build_opener", return_value=opener):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                chat("system", "request")
        self.assertEqual(opener.calls, 1)

    def test_incomplete_paid_response_is_not_retried(self):
        class Opener:
            calls = 0

            def open(self, *args, **kwargs):
                self.calls += 1
                raise http.client.IncompleteRead(b"partial")

        opener = Opener()
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False), \
             patch("scripts.text.utils.llm.urllib.request.build_opener", return_value=opener):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                chat("system", "request")
        self.assertEqual(opener.calls, 1)

    def test_dns_failure_can_be_retried_without_sending_request(self):
        class Opener:
            calls = 0

            def open(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise urllib.error.URLError(socket.gaierror("dns failure"))
                return io.BytesIO(b'{"choices":[{"message":{"content":"ok"}}]}')

        opener = Opener()
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False), \
             patch("scripts.text.utils.llm.urllib.request.build_opener", return_value=opener), \
             patch("scripts.text.utils.llm.time.sleep"):
            self.assertEqual(chat("system", "request"), "ok")
        self.assertEqual(opener.calls, 2)


if __name__ == "__main__":
    unittest.main()
