"""Offline checks for electrical warnings in multi-user dialogue."""
import unittest
from unittest.mock import patch

from scripts.production.moss_scale.targeted_review import review_once
from scripts.text.quality import audit
from scripts.text.utils.schema import Dialogue, Participant, Turn


def two_interruptions(first, second, answer, refs=None):
    return Dialogue(
        dialogue_id="multi-safety-test",
        participants=[Participant("user_01", "", ""), Participant("user_02", "", "")],
        profile={"language": "en"},
        turns=[
            Turn(speaker="assistant", addressee="all", text="Let us finish the packing plan."),
            Turn(speaker="user_01", addressee="assistant", kind="interruption", gap_before_ms=-350,
                 responds_to=[0], text=first),
            Turn(speaker="user_02", addressee="assistant", kind="interruption", gap_before_ms=-350,
                 responds_to=[0], text=second),
            Turn(speaker="assistant", addressee="all", responds_to=[1, 2] if refs is None else refs,
                 text=answer),
        ],
    )


def electrical_warnings(dialogue):
    report = audit(dialogue)
    return report, [warning for warning in report["warnings"]
                    if warning["code"] == "electrical_fault_safety_review"]


class MultiInterruptSafetyTests(unittest.TestCase):
    def test_nonadjacent_interruption_is_reviewed_without_model_call_or_mutation(self):
        d = two_interruptions("The power cord has a frayed spot near the plug.",
                              "My bag is already full.",
                              "Let's set the cord aside for inspection and find another bag.")
        before = d.to_jsonl_line()
        with patch("scripts.text.utils.llm.chat", side_effect=AssertionError("unexpected model call")) as model:
            report, warnings = electrical_warnings(d)
        model.assert_not_called()
        self.assertTrue(report["passed"])
        self.assertEqual(d.to_jsonl_line(), before)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["turn_id"], 3)
        self.assertIn("turn(s): 1", warnings[0]["detail"])

    def test_all_referenced_user_turns_and_legacy_fault_are_included_once(self):
        d = two_interruptions("The power cord is frayed.", "The kettle is smoking.",
                              "Let's look at both problems.", refs=[1, 2, 1])
        report, warnings = electrical_warnings(d)
        self.assertTrue(report["passed"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("turn(s): 1, 2", warnings[0]["detail"])

    def test_known_safety_response_suppresses_the_review_hint(self):
        answers = ["Stop using the power cord and disconnect it if safe.",
                   "电源线先停用，安全时断电并请专业人员检查。",
                   "这根电源线不能再用了。"]
        for answer in answers:
            with self.subTest(answer=answer):
                d = two_interruptions("The power cord is frayed.", "My bag is full.", answer)
                self.assertEqual(electrical_warnings(d)[1], [])

    def test_affirmative_damage_and_negated_safety_are_flagged(self):
        questions = ["The power cord has a frayed spot near the plug.",
                     "The electrical wire insulation is torn.",
                     "The power cord is not damaged but has a frayed spot.",
                     "电源线外皮磨破了，铜丝露出来。",
                     "电线没破皮，但铜丝已经裸露。",
                     "电线没有冒烟，但插头在冒火花。"]
        for question in questions:
            with self.subTest(question=question):
                d = two_interruptions(question, "The bag is full.", "Let's set it aside.")
                self.assertEqual(len(electrical_warnings(d)[1]), 1)
        for answer in ("Do not unplug it; keep using it.", "不用断电，继续用。"):
            with self.subTest(answer=answer):
                d = two_interruptions("The power cord is frayed.", "The bag is full.", answer)
                self.assertEqual(len(electrical_warnings(d)[1]), 1)

    def test_negated_electrician_is_not_a_safety_recommendation(self):
        cases = [("电源线破损。", "不用找电工，继续使用。"),
                 ("The power cord is frayed.", "do not call an electrician; keep using it."),
                 ("The power cord is frayed.", "You do not need an electrician; keep using the cord."),
                 ("The power cord is frayed.", "An electrician is not required; keep using the cord.")]
        for question, answer in cases:
            with self.subTest(question=question, answer=answer):
                d = two_interruptions(question, "The bag is full.", answer)
                report, warnings = electrical_warnings(d)
                self.assertTrue(report["passed"])
                self.assertEqual(len(warnings), 1)

        # A separately stated instruction not to keep using it remains a
        # recognised safety response even if consulting an electrician is negated.
        safe_cases = [("电源线破损。", "不要继续用。"), ("电源线破损。", "不用找电工，但不要继续用。"),
                      ("The power cord is frayed.", "do not call an electrician; stop using it."),
                      ("电源线破损。", "请联系电工。")]
        for question, answer in safe_cases:
            with self.subTest(question=question, answer=answer):
                d = two_interruptions(question, "The bag is full.", answer)
                self.assertEqual(electrical_warnings(d)[1], [])

    def test_negated_faults_and_unrelated_damage_do_not_warn(self):
        questions = ["The power cord is not frayed.",
                     "The power cord has no fraying or exposed wires.",
                     "No wires are exposed on the electrical cord.",
                     "电源线没有破损或露铜。",
                     "电线没破皮，也没有铜丝裸露。",
                     "水壶没有冒烟，也没有漏电。",
                     "电源线完好，桌布破了。"]
        for question in questions:
            with self.subTest(question=question):
                d = two_interruptions(question, "The bag is full.", "Okay, let's continue.")
                self.assertEqual(electrical_warnings(d)[1], [])

    def test_only_explicitly_referenced_user_turns_are_checked(self):
        d = two_interruptions("The power cord is frayed.", "My bag is full.",
                              "Let's find another bag.", refs=[2])
        self.assertEqual(electrical_warnings(d)[1], [])
        # An explicit reference to a different earlier turn does not cause an
        # unrelated immediately preceding user turn to be inferred as a source.
        d.turns[3].responds_to = [0]
        d.turns[2].text = "The power cord is frayed."
        self.assertEqual(electrical_warnings(d)[1], [])

    def test_empty_reference_falls_back_only_to_an_addressed_neighbor(self):
        d = two_interruptions("The bag is full.", "The power cord is frayed.",
                              "Let's set it aside.", refs=[])
        self.assertEqual(len(electrical_warnings(d)[1]), 1)
        d.turns[2].addressee = "user_01"
        self.assertEqual(electrical_warnings(d)[1], [])

    def test_new_wire_hint_does_not_enable_the_optional_paid_reviewer(self):
        draft = {"turns": [
            {"speaker": "user_01", "kind": "speech", "addressee": "assistant",
             "text": "The power cord is frayed near the plug."},
            {"speaker": "assistant", "kind": "speech", "responds_to": [0],
             "text": "Let's set it aside for inspection."},
        ]}
        with patch("scripts.production.moss_scale.targeted_review.chat") as model:
            summary, _ = review_once(draft, [{}, {}], "en")
        self.assertEqual(summary["status"], "not_triggered")
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
