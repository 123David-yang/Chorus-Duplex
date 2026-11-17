"""Offline coverage for context-sensitive production drafting instructions."""
import json
import random
import unittest
from unittest.mock import patch

from scripts.production.moss_scale.script import validate
from scripts.production.moss_scale.slots import draft_from_slots, plan_slots


class DialogueContextPromptTests(unittest.TestCase):
    def test_closing_reply_targets_the_actual_final_user(self):
        for count in (1, 2, 4):
            users = [f"user_{i+1:02d}" for i in range(count)]
            for seed in range(20):
                slots = plan_slots(users, ["side_talk"], False, False, False,
                                   False, False, random.Random(seed), 14, "en")
                self.assertEqual(slots[-1]["addressee"], slots[-2]["speaker"])
                self.assertEqual(slots[-1]["responds_to"], [len(slots)-2])

    def test_side_talk_resume_and_speaker_context_reach_both_drafting_modes(self):
        users = ["user_01", "user_02"]
        context = {"场景": "一起讨论", "领域": "饮食与生活", "关系": "朋友"}
        for language in ("zh", "en"):
            for mode in ("direct", "interactive"):
                # Exercise both entry styles and resume turns on either side of the
                # interactive drafting boundary.
                for seed, target in ((0, 12), (2, 14)):
                    with self.subTest(language=language, mode=mode, seed=seed, target=target):
                        slots = plan_slots(users, ["side_talk"], False, False, False,
                                           False, False, random.Random(seed), target, language)
                        aside = next(i for i, slot in enumerate(slots)
                                     if slot["kind"] == "side_talk")
                        resume = aside + 2
                        requests = []

                        def fake_chat(system, user, **kwargs):
                            request = json.loads(user)
                            requests.append((system, request))
                            # No network call: return one spoken line for each planned turn.
                            return json.dumps({key: f"line {key}" for key in request["待写轮次"]})

                        with patch("scripts.production.moss_scale.slots.chat", side_effect=fake_chat):
                            result = draft_from_slots(slots, users, context, mode, [], language)

                        self.assertEqual(len(requests), 1 if mode == "direct" else 2)
                        self.assertTrue(all("自己的新情况可以自然补充" in system for system, _ in requests))
                        if language == "en":
                            self.assertTrue(all("Attribute each stated restriction to its speaker" in system
                                                for system, _ in requests))
                        planned = {key: value for _, request in requests
                                   for key, value in request["待写轮次"].items()}
                        cue = planned[str(resume)]["cue"]
                        self.assertIn("不必宣布", cue)
                        self.assertIn("提到别人的限制时说清是谁", cue)
                        if mode == "interactive" and resume >= len(slots) // 2:
                            second_request = requests[1][1]
                            history = second_request["前半段已说过的台词"]
                            self.assertEqual(history["0"]["speaker"], slots[0]["speaker"])
                            self.assertEqual(history["0"]["kind"], slots[0]["kind"])
                            self.assertEqual(history["0"]["addressee"], slots[0]["addressee"])
                            self.assertEqual(history["0"]["text"], "line 0")
                            self.assertEqual(history[str(aside)]["speaker"], "third_party_01")
                            self.assertEqual(history[str(aside)]["addressee"], slots[aside]["addressee"])
                        # Drafting instructions must not become delivery fields or alter event links.
                        for slot, turn in zip(slots, result["turns"]):
                            self.assertNotIn("cue", turn)
                            for key in ("speaker", "kind", "addressee", "responds_to"):
                                self.assertEqual(turn[key], slot[key])
                        validate(result, users, ["side_talk"], False, language=language)


if __name__ == "__main__":
    unittest.main()
