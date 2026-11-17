import json
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.production.moss_scale.targeted_review import review_once


class TargetedSafetyReviewTests(unittest.TestCase):
    def draft(self):
        return {"turns": [
            {"speaker": "user_01", "kind": "speech", "addressee": "assistant", "text": "我的电水壶接触不太好，烧水时得用手按着。"},
            {"speaker": "assistant", "kind": "speech", "responds_to": [0], "text": "那你想买多大容量的？"},
        ]}

    def slots(self):
        return [{"kind": "speech", "form": None}, {"kind": "speech", "form": None}]

    def test_rewrites_one_plain_answer_and_retains_private_trace(self):
        draft = self.draft()
        revised = "建议先停用，安全时断电并请专业人员检查；之后再考虑容量。"
        response = json.dumps({"decision": "rewrite", "text": revised}, ensure_ascii=False)
        with patch("scripts.production.moss_scale.targeted_review.chat", return_value=response) as call:
            summary, trace = review_once(draft, self.slots(), "zh")
        self.assertEqual(summary, {"status": "rewritten", "turn_id": 1, "calls": 1})
        self.assertEqual(trace["original"], "那你想买多大容量的？")
        self.assertEqual(draft["turns"][1]["text"], revised)
        self.assertEqual(call.call_count, 1)

    def test_keeps_uncertain_and_rejects_contradictory_rewrite(self):
        for response in (json.dumps({"decision": "uncertain"}),
                         json.dumps({"decision": "rewrite", "text": "不用停用，继续通电就好。"}, ensure_ascii=False),
                         json.dumps({"decision": "rewrite", "text": "无需停用，这个现象很常见。"}, ensure_ascii=False),
                         json.dumps({"decision": "rewrite", "text": "No need to stop using it; keep it plugged in."}),
                         "not-json"):
            draft = self.draft()
            original = draft["turns"][1]["text"]
            with patch("scripts.production.moss_scale.targeted_review.chat", return_value=response):
                summary, trace = review_once(draft, self.slots(), "zh")
            self.assertIn(summary["status"], ("uncertain", "unavailable_or_uncertain"))
            self.assertIsNone(trace)
            self.assertEqual(draft["turns"][1]["text"], original)

    def test_skips_anchor_and_unflagged_input_without_model_calls(self):
        draft = self.draft()
        slots = self.slots()
        slots[1]["form"] = "anchor"
        with patch("scripts.production.moss_scale.targeted_review.chat") as call:
            result, trace = review_once(draft, slots, "zh")
            self.assertEqual(result["status"], "skipped_structural")
            self.assertIsNone(trace)
            draft["turns"][0]["text"] = "我们今天几点出门？"
            result, _ = review_once(draft, self.slots(), "zh")
            self.assertEqual(result["status"], "not_triggered")
            call.assert_not_called()

    def test_skips_answer_that_hosts_an_interruption(self):
        draft = self.draft()
        draft["turns"].append({"speaker": "user_01", "kind": "interruption", "responds_to": [1],
                               "addressee": "assistant", "text": "等一下，还有个问题。"})
        slots = self.slots() + [{"kind": "interruption", "form": None}]
        with patch("scripts.production.moss_scale.targeted_review.chat") as call:
            result, _ = review_once(draft, slots, "zh")
        self.assertEqual(result["status"], "skipped_structural")
        call.assert_not_called()

    def test_sensitive_context_is_not_sent(self):
        examples = ("sk-" + "x" * 25, "AKIA" + "X" * 16, "Bearer " + "x" * 25,
                    "api_key=" + "fake" * 8, "example" + "@example.invalid", "138" + "00000000",
                    "138-" + "0000-0000", "138 " + "0000 0000", "\\\\" + "server\\share\\file",
                    "C:" + "\\\u005c" + "Users\\person\\file")
        for example in examples:
            with self.subTest(kind=examples.index(example)):
                draft = self.draft()
                draft["turns"][0]["text"] += "附" + example + "请保管"
                with patch("scripts.production.moss_scale.targeted_review.chat") as call:
                    result, _ = review_once(draft, self.slots(), "zh")
                self.assertIn(result["status"], ("sensitive_context_skipped", "unavailable_or_uncertain"))
                self.assertEqual(result["calls"], 0)
                call.assert_not_called()

    def test_sensitive_model_output_is_not_written_into_the_draft(self):
        draft = self.draft()
        original = draft["turns"][1]["text"]
        secret_like = "AKIA" + "X" * 16
        response = json.dumps({"decision": "rewrite", "text": "建议停用并联系专业人员。" + secret_like}, ensure_ascii=False)
        with patch("scripts.production.moss_scale.targeted_review.chat", return_value=response):
            result, trace = review_once(draft, self.slots(), "zh")
        self.assertEqual(result["status"], "unavailable_or_uncertain")
        self.assertIsNone(trace)
        self.assertEqual(draft["turns"][1]["text"], original)

    def test_does_not_rewrite_a_private_exchange_or_unrelated_reply(self):
        draft = self.draft()
        draft["turns"][0]["addressee"] = "user_02"
        draft["turns"][1]["responds_to"] = []
        with patch("scripts.production.moss_scale.targeted_review.chat") as call:
            result, _ = review_once(draft, self.slots(), "zh")
        self.assertEqual(result["status"], "not_triggered")
        call.assert_not_called()

    def test_earlier_side_talk_is_not_sent_to_the_model(self):
        private = [
            {"speaker": "user_02", "kind": "speech", "addressee": "user_03", "text": "PRIVATE_EXCHANGE_SENTINEL"},
            {"speaker": "user_03", "kind": "speech", "addressee": "user_02", "text": "PRIVATE_EXCHANGE_REPLY"},
        ]
        target = self.draft()["turns"]
        target[1]["responds_to"] = [2]
        draft = {"turns": private + target}
        slots = [{"kind": "speech", "form": None} for _ in draft["turns"]]
        with patch("scripts.production.moss_scale.targeted_review.chat", return_value='{"decision":"keep"}') as call:
            result, _ = review_once(draft, slots, "zh")
        self.assertEqual(result["status"], "keep")
        payload = call.call_args.args[1]
        self.assertNotIn("PRIVATE_EXCHANGE_SENTINEL", payload)
        self.assertNotIn("PRIVATE_EXCHANGE_REPLY", payload)
        self.assertIn("电水壶", payload)

    def test_failed_revalidation_restores_the_original_without_redrafting(self):
        from scripts.production.moss_scale import script
        original = self.draft()
        original.update(setting="home", topic="shopping", participants=[])
        original["turns"][0]["responds_to"] = []
        original["turns"][1]["addressee"] = "user_01"
        def reviewer(draft, slots, language):
            previous = draft["turns"][1]["text"]
            draft["turns"][1]["text"] = "changed response"
            return {"status": "rewritten", "turn_id": 1, "calls": 1}, {"original": previous, "revised": "changed response"}
        def validate(draft, *args, **kwargs):
            if draft["turns"][1]["text"] == "changed response":
                raise ValueError("simulated structural rejection")
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {"CHORUS_TARGETED_SAFETY_REVIEW": "1", "TTS_BACKEND": "moss_onnx"}), \
             patch.object(script, "distinct_voice_cast", return_value=(2, {"assistant": "Junhao", "user_01": "Xiaoyu", "user_02": "Weiguo"}, False)), \
             patch.object(script, "draft_from_slots", return_value=copy.deepcopy(original)) as draft_call, \
             patch.object(script, "validate", side_effect=validate), \
             patch.object(script, "reference_voices", return_value={}), \
             patch.object(script, "voice_bindings_snapshot", return_value={}), \
             patch.object(script, "apply_rule_quality"), \
             patch("scripts.production.moss_scale.targeted_review.review_once", side_effect=reviewer):
            root = Path(folder)
            (root / "scripts").mkdir()
            (root / "staging").mkdir()
            result = script.generate(0, root, "direct", "zh")
            trace = json.loads(next((root / "staging").glob("*-safety-review-*.json")).read_text(encoding="utf-8"))
            self.assertNotIn("original", trace)
            self.assertNotIn("revised", trace)
        self.assertEqual(draft_call.call_count, 1)
        self.assertEqual(result["turns"][1]["text"], original["turns"][1]["text"])
        self.assertEqual(result["profile"]["targeted_safety_review"]["status"], "rewrite_rejected")


if __name__ == "__main__":
    unittest.main()
