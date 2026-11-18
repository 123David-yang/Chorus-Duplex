import json
import random
import unittest
from unittest.mock import patch
from scripts.production.moss_scale.slots import plan_slots, draft_from_slots


class RoleplaySlotTests(unittest.TestCase):
    def test_backchannel_sees_only_anchor_and_later_turn_sees_full_host(self):
        slots = [
            {'speaker': 'assistant', 'addressee': 'all', 'kind': 'speech', 'responds_to': [], 'cue': '', 'form': 'anchor'},
            {'speaker': 'user_01', 'addressee': 'assistant', 'kind': 'backchannel', 'responds_to': [0], 'cue': '', 'form': None},
            {'speaker': 'assistant', 'addressee': 'user_01', 'kind': 'speech', 'responds_to': [1], 'cue': '', 'form': None}]
        seen = []
        def fake(system, user, **kwargs):
            request = json.loads(user)
            key = next(iter(request['待写轮次']))
            if key != '0':
                seen.append(request['前半段已说过的台词']['0']['text'])
            return json.dumps({key: ['The first point is clear', 'The remaining information is not yet audible.'] if key == '0' else 'Right.'})
        with patch('scripts.production.moss_scale.slots.chat', side_effect=fake):
            draft_from_slots(slots, ['user_01'], {'场景': 'planning', '领域': 'study', '关系': 'friends'}, 'roleplay', [], 'en')
        self.assertEqual(seen[0], 'The first point is clear')
        self.assertEqual(seen[1], ['The first point is clear', 'The remaining information is not yet audible.'])

    def test_invalid_first_output_stops_before_later_paid_calls(self):
        slots = [{'speaker': 'assistant', 'addressee': 'all', 'kind': 'speech', 'responds_to': [], 'cue': '', 'form': 'anchor'},
                 {'speaker': 'user_01', 'addressee': 'assistant', 'kind': 'backchannel', 'responds_to': [0], 'cue': '', 'form': None}]
        with patch('scripts.production.moss_scale.slots.chat', return_value='{"0": "wrong shape"}') as call:
            with self.assertRaises(ValueError):
                draft_from_slots(slots, ['user_01'], {}, 'roleplay', [], 'en')
        self.assertEqual(call.call_count, 1)

    def test_each_call_writes_only_its_role_and_receives_causal_history(self):
        users = ['user_01', 'user_02']
        slots = plan_slots(users, ['side_talk'], False, False, False, False, False, random.Random(2), 12, 'en')
        requests = []
        def fake(system, user, **kwargs):
            request = json.loads(user)
            requests.append(request)
            self.assertEqual(len(request['待写轮次']), 1)
            key, slot = next(iter(request['待写轮次'].items()))
            self.assertEqual(request['当前角色'], slot['speaker'])
            history = request.get('前半段已说过的台词', {})
            self.assertEqual(set(map(int, history)), set(range(int(key))))
            self.assertEqual(kwargs['max_tokens'], 700)
            return json.dumps({key: 'A natural line.'})
        with patch('scripts.production.moss_scale.slots.chat', side_effect=fake):
            result = draft_from_slots(slots, users, {'场景': 'planning', '领域': 'study', '关系': 'friends'}, 'roleplay', [], 'en')
        self.assertEqual(len(requests), len(slots))
        for before, after in zip(slots, result['turns']):
            for key in ('speaker', 'addressee', 'kind', 'responds_to'):
                self.assertEqual(before[key], after[key])

    def test_fixed_turn_uses_no_paid_call(self):
        slots = [{'speaker': 'user_01', 'addressee': 'assistant', 'kind': 'speech',
                  'responds_to': [], 'cue': '', 'form': None, 'fixed_text': 'Fixed words.'}]
        with patch('scripts.production.moss_scale.slots.chat') as call:
            result = draft_from_slots(slots, ['user_01'], {'场景': 'planning', '领域': 'study', '关系': 'friends'}, 'roleplay', [], 'en')
        self.assertEqual(result['turns'][0]['text'], 'Fixed words.')
        call.assert_not_called()
