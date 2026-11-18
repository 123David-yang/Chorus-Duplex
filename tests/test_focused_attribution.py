import json
import unittest
from unittest.mock import patch
from scripts.text.judge.attribution import review_attribution


class FocusedAttributionTests(unittest.TestCase):
    def setUp(self):
        self.draft = {'turns': [
            {'speaker': 'user_03', 'addressee': 'all', 'text': 'I need Thursday free for a lab report.'},
            {'speaker': 'user_04', 'addressee': 'assistant', 'text': 'You said you need Thursday free.'}]}
        self.claim = {'recall_turn': 1, 'source_turn': 0, 'original_owner': 'user_03',
                      'recalled_owner': 'assistant', 'consistent': False, 'reason': 'Different owner.'}

    def run_review(self, claims):
        with patch('scripts.text.judge.attribution.llm.chat', return_value=json.dumps({'claims': claims})):
            return review_attribution(self.draft)

    def test_wrong_owner_rejects_and_correct_owner_accepts(self):
        self.assertEqual(self.run_review([self.claim])['decision'], 'reject')
        self.draft['turns'][1]['addressee'] = 'user_03'
        self.claim.update(recalled_owner='user_03', consistent=True)
        self.assertEqual(self.run_review([self.claim])['decision'], 'accept')

    def test_uncertainty_is_not_acceptance(self):
        self.claim.update(source_turn=None, original_owner=None, consistent=None)
        self.assertEqual(self.run_review([self.claim])['decision'], 'quarantine')

    def test_missing_or_noncausal_evidence_fails(self):
        with self.assertRaises(ValueError):
            self.run_review([])
        self.claim['source_turn'] = 1
        with self.assertRaises(ValueError):
            self.run_review([self.claim])

    def test_no_candidate_does_not_call_or_claim_acceptance(self):
        self.draft['turns'][1]['text'] = 'Let us discuss the plan.'
        with patch('scripts.text.judge.attribution.llm.chat') as call:
            self.assertEqual(review_attribution(self.draft)['decision'], 'not_applicable')
        call.assert_not_called()

    def test_model_cannot_override_owners_or_forge_recipient(self):
        self.claim['consistent'] = True
        self.assertEqual(self.run_review([self.claim])['decision'], 'reject')
        self.claim['recalled_owner'] = 'user_03'
        with self.assertRaisesRegex(ValueError, 'recipient'):
            self.run_review([self.claim])

    def test_missing_fields_are_not_implicit_uncertainty(self):
        del self.claim['consistent']
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.run_review([self.claim])
