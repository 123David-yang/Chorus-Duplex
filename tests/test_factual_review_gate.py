from copy import deepcopy
import unittest
import json
from unittest.mock import patch
from scripts.text.judge.review import DIMENSIONS, decide, review_scale_draft
from scripts.text.utils.schema import Dialogue, Participant, Turn


class FactualReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.dialogue = Dialogue(dialogue_id='fact-test', participants=[Participant('user_01', '', '')],
                                turns=[Turn(speaker='user_01', addressee='assistant', text='我准备先整理一下。')])
        self.report = {'dimensions': {name: {'score': 5, 'evidence_turn_ids': [0], 'reason': 'No unsupported assertion in this turn.'} for name in DIMENSIONS},
                       'critical_issues': [], 'module_evidence': {}, 'ai_request_audit': []}

    def test_missing_fact_review_cannot_be_accepted(self):
        report = deepcopy(self.report)
        report['dimensions'].pop('factuality')
        self.assertEqual(decide(self.dialogue, report)['decision'], 'quarantine')

    def test_low_fact_score_or_explicit_critical_issue_rejects(self):
        report = deepcopy(self.report)
        report['dimensions']['factuality']['score'] = 3
        self.assertEqual(decide(self.dialogue, report)['decision'], 'reject')
        report = deepcopy(self.report)
        report['critical_issues'] = [{'code': 'unsafe_advice', 'turn_ids': [0], 'reason': 'fixture'}]
        self.assertEqual(decide(self.dialogue, report)['decision'], 'reject')

    def test_complete_evidenced_report_remains_accepted(self):
        self.assertEqual(decide(self.dialogue, self.report)['decision'], 'accept')


class ScaleFactualReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.draft = {'turns': [{'speaker': 'assistant', 'text': 'Let us check the conditions first.', 'kind': 'speech'}]}
        self.scores = {name: {'score': 5, 'turn_ids': [0], 'reason': 'Evidence from this turn.'}
                       for name in ('causality', 'addressing', 'duplex', 'naturalness', 'factuality')}

    def review(self):
        with patch('scripts.text.judge.review.llm.chat', return_value=json.dumps({'dimensions': self.scores})):
            return review_scale_draft(self.draft)

    def test_low_factuality_cannot_be_averaged_away(self):
        self.scores['factuality']['score'] = 3
        result = self.review()
        self.assertGreater(result['mean_score'], 4)
        self.assertEqual(result['decision'], 'reject')

    def test_missing_factuality_fails_closed(self):
        del self.scores['factuality']
        with self.assertRaisesRegex(ValueError, 'five dimensions'):
            self.review()

    def test_complete_review_accepts_without_additional_calls(self):
        self.assertEqual(self.review()['decision'], 'accept')

