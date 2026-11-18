from copy import deepcopy
import json
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import os
from scripts.production.moss_scale.anchor_alignment import align_backchannel_anchors


class AnchorAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.draft={'turns':[{'kind':'speech','text':'先剥皮，也可以直接切块省时间，按你们偏好决定就好。'},
            {'kind':'backchannel','text':'那就切块吧。','anchor_text':'先剥皮','responds_to':[0]},
            {'kind':'backchannel','text':'省时间要紧。','anchor_text':'先剥皮','responds_to':[0]}]}

    def review(self,anchor):
        with patch('scripts.production.moss_scale.anchor_alignment.chat',return_value=json.dumps({'anchors':[{'host_turn_id':0,'anchor_text':anchor}]})):
            return align_backchannel_anchors(self.draft)

    def test_group_moves_to_shared_supported_prefix_without_changing_speech(self):
        before=deepcopy(self.draft)
        result=self.review('先剥皮，也可以直接切块省时间')
        self.assertEqual(result['changed_hosts'],1)
        for index in (1,2):self.assertEqual(self.draft['turns'][index]['anchor_text'],'先剥皮，也可以直接切块省时间')
        for index in range(3):self.assertEqual(self.draft['turns'][index]['text'],before['turns'][index]['text'])

    def test_invalid_or_late_anchor_does_not_mutate(self):
        for anchor in (None,'改写过的句子',self.draft['turns'][0]['text']):
            before=deepcopy(self.draft)
            with self.assertRaises(ValueError):self.review(anchor)
            self.assertEqual(self.draft,before)

    def test_no_acknowledgements_uses_no_call(self):
        self.draft['turns']=self.draft['turns'][:1]
        with patch('scripts.production.moss_scale.anchor_alignment.chat') as call:
            self.assertEqual(align_backchannel_anchors(self.draft)['calls'],0)
        call.assert_not_called()

    def test_later_invalid_host_leaves_all_earlier_groups_unchanged(self):
        self.draft['turns'] += [{'kind':'speech','text':'另一个完整信息，还有足够长的后续内容。'},
            {'kind':'backchannel','text':'对。','anchor_text':'另一个完整信息','responds_to':[3]}]
        before=deepcopy(self.draft)
        response={'anchors':[{'host_turn_id':0,'anchor_text':'先剥皮，也可以直接切块省时间'},
                             {'host_turn_id':3,'anchor_text':None}]}
        with patch('scripts.production.moss_scale.anchor_alignment.chat',return_value=json.dumps(response)):
            with self.assertRaises(ValueError):align_backchannel_anchors(self.draft)
        self.assertEqual(self.draft,before)

    def test_alignment_failure_does_not_trigger_another_paid_draft(self):
        from scripts.production.moss_scale.script import generate
        with TemporaryDirectory() as folder, patch.dict(os.environ,{'TTS_BACKEND':'moss_onnx','CHORUS_ALIGN_BACKCHANNELS':'1'},clear=True):
            root=Path(folder)
            (root/'scripts').mkdir();(root/'staging').mkdir()
            with patch('scripts.production.moss_scale.script.draft_from_slots',return_value=deepcopy(self.draft)) as drafting, \
                 patch('scripts.production.moss_scale.script.validate'), \
                 patch('scripts.production.moss_scale.anchor_alignment.align_backchannel_anchors',side_effect=ValueError('No suitable anchor')) as align:
                with self.assertRaises(ValueError):generate(1,root,'direct','zh')
            self.assertEqual(drafting.call_count,1)
            self.assertEqual(align.call_count,1)
