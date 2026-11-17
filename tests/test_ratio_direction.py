"""Narrow, non-blocking checks for reversed Chinese two-item ratios."""
import unittest
from unittest.mock import patch

from scripts.text.quality import audit
from scripts.text.utils.schema import Dialogue, Participant, Turn


def dialogue(*lines):
    turns = []
    for index, text in enumerate(lines):
        assistant = index % 2 == 0
        turns.append(Turn(speaker="assistant" if assistant else "user_01", text=text,
                          addressee="user_01" if assistant else "assistant",
                          responds_to=[index - 1] if index else []))
    return Dialogue(dialogue_id="ratio-test", participants=[Participant("user_01", "", "")],
                    turns=turns, profile={"language": "zh"})


def ratio_warnings(report):
    return [warning for warning in report["warnings"] if warning["code"] == "possible_reversed_ratio"]


class RatioDirectionTests(unittest.TestCase):
    def test_realistic_reversed_question_is_only_a_warning(self):
        d = dialogue("可以先按白米和糙米二比一试试，具体看口感。", "我明白了。",
                     "刚才说到主食比例，糙米和白米二比一的话要煮多久？")
        before = d.to_jsonl_line()
        with patch("scripts.text.utils.llm.chat", side_effect=AssertionError("unexpected model call")) as chat:
            report = audit(d)
        chat.assert_not_called()
        self.assertTrue(report["passed"])
        self.assertEqual(report["issues"], [])
        self.assertEqual(d.to_jsonl_line(), before)
        warnings = ratio_warnings(report)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["turn_id"], 2)
        self.assertIn("turn 0", warnings[0]["detail"])
        self.assertIn("intentional change", warnings[0]["detail"])

    def test_equivalent_written_ratios_and_single_character_nouns(self):
        cases = [
            ("先按白米和糙米2:1。", "糙米与白米的配比为二比一吗？"),
            ("白米和糙米是2:1。", "糙米和白米的比例二比一吗？"),
            ("米和水一比二。", "水和米1:2对吗？"),
            ("油和水2比1。", "水和油4:2吗？"),
            ("苹果和梨两比一。", "梨和苹果2：1吗？"),
        ]
        for first, second in cases:
            with self.subTest(first=first, second=second):
                self.assertEqual(len(ratio_warnings(audit(dialogue(first, second)))), 1)

    def test_unchanged_mix_equal_ratios_or_different_items_do_not_warn(self):
        cases = [
            ("白米和糙米二比一。", "白米和糙米2:1对吗？"),
            ("白米和糙米二比一。", "糙米和白米一比二对吗？"),
            ("白米和糙米一比一。", "糙米和白米1:1对吗？"),
            ("盐和糖二比一。", "糖和奶二比一对吗？"),
            ("白米和糙米二比一。", "糙米和大米二比一对吗？"),
            ("白米和糙米十一比一。", "糙米和白米十一比一对吗？"),
            ("白米和糙米2.5:1。", "糙米和白米2.5:1对吗？"),
        ]
        for first, second in cases:
            with self.subTest(first=first, second=second):
                self.assertEqual(ratio_warnings(audit(dialogue(first, second))), [])

    def test_clear_change_or_correction_is_not_treated_as_a_mistake(self):
        first = "白米和糙米二比一。"
        for revised in ("后来我改成糙米和白米二比一。", "我换成糙米和白米二比一。",
                        "我刚才说错了，应该是糙米和白米二比一。"):
            with self.subTest(revised=revised):
                self.assertEqual(ratio_warnings(audit(dialogue(first, revised))), [])
        # An unmarked trial may be intentional too; the warning asks for context.
        warning = ratio_warnings(audit(dialogue(first, "我准备试试糙米和白米二比一。")))
        self.assertEqual(len(warning), 1)
        self.assertIn("intentional change", warning[0]["detail"])
        # A question about whether someone misspoke is not itself a correction.
        self.assertEqual(len(ratio_warnings(audit(dialogue(
            first, "我刚才说错了吗？糙米和白米二比一对吗？")))), 1)
        # A mistaken restatement followed by a correction in the same turn
        # should not produce a warning for the superseded wording.
        self.assertEqual(ratio_warnings(audit(dialogue(
            first, "糙米和白米二比一吗？不对，应该是白米和糙米二比一。"))), [])

    def test_later_mentions_compare_with_latest_explicitly_changed_ratio(self):
        d = dialogue("白米和糙米二比一。", "后来我改成糙米和白米二比一。",
                     "那白米和糙米二比一对吗？")
        warnings = ratio_warnings(audit(d))
        self.assertEqual([warning["turn_id"] for warning in warnings], [2])
        self.assertIn("turn 1", warnings[0]["detail"])


if __name__ == "__main__":
    unittest.main()
