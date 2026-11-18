import hashlib
import json
import sys
import tempfile
import unittest
import warnings
import io
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from scripts.production.moss_scale.upload_delivery import _matches_remote, validate_delivery


class DeliveryUploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "completed").mkdir()
        (self.root / "audio").mkdir()
        self.ident = "case-001"
        annotation = {
            "dialogue_id": self.ident,
            "participants": [],
            "turns": [{"turn_id": 0, "speaker": "assistant", "kind": "speech",
                       "text": "Hello", "addressee": "user_01", "gap_before_ms": 0}],
        }
        names = (f"{self.ident}.flac", f"{self.ident}_babble.flac",
                 f"{self.ident}.timeline.json")
        for name in names[:2]:
            sf.write(self.root / "audio" / name, np.full((480, 2), 0.01), 48000)
        timeline = [{"turn_id": 0, "speaker": "assistant", "kind": "speech",
                     "text": "Hello", "addressee": "user_01", "start_sec": 0.0, "end_sec": 0.01}]
        (self.root / "audio" / names[2]).write_text(json.dumps(timeline), encoding="utf8")
        hashes = {}
        for name in names:
            with (self.root / "audio" / name).open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        receipt = {"annotation": annotation, "frames": 480, "sample_rate": 48000, "sha256": hashes,
                   "credited_seconds": 0.01}
        (self.root / "completed" / f"{self.ident}.json").write_text(json.dumps(receipt), encoding="utf8")
        self.audit = patch("scripts.text.quality.audit", return_value={"passed": True})
        self.audit.start()

    def tearDown(self):
        self.audit.stop()
        self.tmp.cleanup()

    def test_valid_delivery_and_tampered_audio(self):
        annotation, files = validate_delivery(self.root, self.ident)
        self.assertEqual(annotation["dialogue_id"], self.ident)
        self.assertEqual(len(files), 3)
        frozen = files[f"audio/{self.ident}.flac"]
        self.assertIsInstance(frozen, bytes)
        with (self.root / "audio" / f"{self.ident}.flac").open("ab") as stream:
            stream.write(b"modified")
        self.assertEqual(files[f"audio/{self.ident}.flac"], frozen)
        with self.assertRaisesRegex(ValueError, "modified"):
            validate_delivery(self.root, self.ident)

    def test_silent_audio_is_rejected_even_with_matching_receipt(self):
        for suffix in ('', '_babble'):
            with self.subTest(suffix=suffix):
                name = f'{self.ident}{suffix}.flac'
                path = self.root / 'audio' / name
                original = path.read_bytes()
                receipt_path = self.root / 'completed' / f'{self.ident}.json'
                original_receipt = receipt_path.read_bytes()
                sf.write(path, np.zeros((480, 2)), 48000)
                receipt = json.loads(original_receipt)
                receipt['sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
                receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'entirely silent'):
                    validate_delivery(self.root, self.ident)
                path.write_bytes(original)
                receipt_path.write_bytes(original_receipt)

    def test_late_signal_after_silence_remains_valid(self):
        for suffix in ('', '_babble'):
            name = f'{self.ident}{suffix}.flac'
            path = self.root / 'audio' / name
            audio = np.zeros((70000, 2))
            audio[-20:] = .01
            sf.write(path, audio, 48000)
            receipt_path = self.root / 'completed' / f'{self.ident}.json'
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            receipt['frames'] = len(audio)
            receipt['credited_seconds'] = len(audio) / 48000
            receipt['sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
        validate_delivery(self.root, self.ident)

    def test_nonfinite_decoded_audio_is_rejected_with_matching_hash(self):
        path = self.root / 'audio' / f'{self.ident}.flac'
        audio = np.full((480, 2), .01)
        audio[-1, 0] = np.nan
        # A matching receipt must not make a decodable float container trustworthy.
        sf.write(path, audio, 48000, format='WAV', subtype='FLOAT')
        receipt_path = self.root / 'completed' / f'{self.ident}.json'
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        receipt['sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            validate_delivery(self.root, self.ident)

    def test_truncated_flac_with_matching_hash_and_header_is_rejected(self):
        name = f"{self.ident}.flac"
        path = self.root / "audio" / name
        sf.write(path, np.random.default_rng(7).normal(0, .05, (24000, 2)), 48000)
        raw = path.read_bytes()
        path.write_bytes(raw[:len(raw) // 2])
        self.assertEqual(sf.info(path).frames, 24000)  # A plausible header alone is not enough.
        receipt_path = self.root / "completed" / f"{self.ident}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["frames"] = 24000
        receipt["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "audio.*(?:decoded|length)"):
            validate_delivery(self.root, self.ident)

    def test_new_delivery_carries_background_attribution_into_public_annotation(self):
        from scripts.production.moss_scale.delivery import deliver
        (self.root / "staging").mkdir()
        annotation = json.loads((self.root / "completed" / f"{self.ident}.json").read_text(encoding="utf-8"))["annotation"]
        def fake_render(dialogue, synth, clean, timeline):
            sf.write(clean, np.full((480, 2), .01), 48000)
            timeline.write_text(json.dumps([{"turn_id": 0, "speaker": "assistant", "kind": "speech",
                                             "text": "Hello", "addressee": "user_01",
                                             "start_sec": 0.0, "end_sec": 0.01}]), encoding="utf-8")
            return 480, 0
        def fake_stitch(room, background, duration, seed_key):
            background.write_bytes(b"synthetic")
        def fake_mix(clean, background, snr_db, out_mixed, out_bg, seed_key):
            shutil.copyfile(clean, out_mixed)
        with patch("scripts.production.moss_scale.delivery.background_rooms", return_value=[Path("rm1/babb")]), \
             patch("scripts.production.moss_scale.delivery.render", side_effect=fake_render), \
             patch("scripts.production.moss_scale.delivery.stitch_clips", side_effect=fake_stitch), \
             patch("scripts.production.moss_scale.delivery.mix", side_effect=fake_mix):
            receipt = deliver(annotation, None, self.root)
        source = receipt["annotation"]["profile"]["background_provenance"]
        self.assertEqual(source["license"], "CC-BY-4.0")
        self.assertEqual(source["applies_to"], "babble_variant")
        public_annotation, _ = validate_delivery(self.root, self.ident)
        self.assertEqual(public_annotation["profile"]["background_provenance"], source)
        self.assertNotIn(str(self.root), json.dumps(source))

    def test_credential_resolution_uses_explicit_process_paths_without_logging_them(self):
        from scripts.production.publishing.credentials import hf_token, read_token_file
        token_file = self.root / "test-token"
        token_file.write_text("synthetic-token", encoding="utf-8")
        with patch("scripts.config.load_env"), patch.dict("os.environ", {"HF_TOKEN_PATH": str(token_file)}, clear=True):
            self.assertEqual(hf_token(), "synthetic-token")
        with patch("scripts.config.load_env"), patch.dict("os.environ", {"HF_TOKEN_PATH": str(token_file), "HF_TOKEN": "process-token"}, clear=True):
            self.assertEqual(hf_token(), "process-token")
        token_file.unlink()
        with patch("scripts.config.load_env"), patch.dict("os.environ", {"HF_TOKEN_PATH": str(token_file)}, clear=True):
            with self.assertRaises(RuntimeError) as exc:
                hf_token()
            self.assertNotIn(str(token_file), str(exc.exception))
        token_file.write_bytes(b"\xffsynthetic-sensitive-fragment")
        with self.assertRaises(RuntimeError) as exc:
            read_token_file(str(token_file))
        self.assertNotIn(str(token_file), str(exc.exception))
        self.assertNotIn("synthetic-sensitive-fragment", str(exc.exception))

    def test_duplicate_clean_audio_is_rejected_before_upload(self):
        from scripts.production.moss_scale.stream_upload import pending_receipts
        original = json.loads((self.root / "completed" / f"{self.ident}.json").read_text(encoding="utf-8"))
        duplicate = json.loads(json.dumps(original))
        duplicate_id = "case-002"
        duplicate["annotation"]["dialogue_id"] = duplicate_id
        duplicate["sha256"] = {name.replace(self.ident, duplicate_id): digest
                               for name, digest in original["sha256"].items()}
        (self.root / "completed" / f"{duplicate_id}.json").write_text(json.dumps(duplicate), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate clean audio"):
            pending_receipts([self.root], self.root / "upload-state", "example/dataset")

    def test_legacy_cli_defaults_to_official_hugging_face_endpoint(self):
        from scripts.cli import push_dataset
        with patch.object(push_dataset, "HF_HOME", self.root / "hf-cache"), patch.dict("os.environ", {}, clear=True):
            push_dataset._setup_hf_env()
            import os
            self.assertEqual(os.environ["HF_ENDPOINT"], "https://huggingface.co")

    def test_rejects_sensitive_metadata_and_signed_urls(self):
        from scripts.production.moss_scale.upload_delivery import _assert_public_safe
        with self.assertRaisesRegex(ValueError, "Sensitive field"):
            _assert_public_safe({"profile": {"api_key": "example"}})
        with self.assertRaisesRegex(ValueError, "private components"):
            _assert_public_safe({"source_url": "HTTPS://example.invalid/file?signature=example"})
        with self.assertRaisesRegex(ValueError, "Sensitive field"):
            _assert_public_safe({"profile": {"token": "example"}})
        with self.assertRaisesRegex(ValueError, "Sensitive field"):
            _assert_public_safe({"profile": {"openai_api_key": "example"}})
        for value in ("AKIA" + "X" * 16, "Bearer " + "x" * 25, "token=" + "x" * 25):
            with self.assertRaisesRegex(ValueError, "Possible credential"):
                _assert_public_safe({"text": "附" + value + "请保管"})
        for value in ("person" + "@example.invalid", "138" + "00000000", "138-" + "0000-0000",
                      "138 " + "0000 0000", "\\\\" + "server\\share\\file", "C:" + "\\Users\\person"):
            with self.assertRaisesRegex(ValueError, "private contact or local path"):
                _assert_public_safe({"text": "附" + value + "请保管"})
        _assert_public_safe({"source_url": "https://example.invalid/file"})
        _assert_public_safe({"sha256": "a" + "13800000000" + "b" * 52})

    def test_rejects_sensitive_timeline(self):
        name = f"{self.ident}.timeline.json"
        path = self.root / "audio" / name
        timeline = json.loads(path.read_text(encoding="utf8"))
        timeline[0]["api_key"] = "example"
        path.write_text(json.dumps(timeline), encoding="utf8")
        receipt_path = self.root / "completed" / f"{self.ident}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf8"))
        receipt["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt_path.write_text(json.dumps(receipt), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "Sensitive field"):
            validate_delivery(self.root, self.ident)

    def test_rejects_mismatched_annotation_turn_id(self):
        path = self.root / "completed" / f"{self.ident}.json"
        receipt = json.loads(path.read_text(encoding="utf8"))
        receipt["annotation"]["turns"][0]["turn_id"] = 999
        path.write_text(json.dumps(receipt), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_delivery(self.root, self.ident)

    def test_rejects_mismatched_timeline_addressee(self):
        name = f"{self.ident}.timeline.json"
        path = self.root / "audio" / name
        timeline = json.loads(path.read_text(encoding="utf8"))
        timeline[0]["addressee"] = "user_02"
        path.write_text(json.dumps(timeline), encoding="utf8")
        receipt_path = self.root / "completed" / f"{self.ident}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf8"))
        receipt["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt_path.write_text(json.dumps(receipt), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_delivery(self.root, self.ident)

    def test_rejects_path_traversal(self):
        with self.assertRaisesRegex(ValueError, "Invalid dialogue ID"):
            validate_delivery(self.root, "../case-001")

    def test_rejects_inflated_credited_duration(self):
        path = self.root / "completed" / f"{self.ident}.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["credited_seconds"] = 1000000
        path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "credited duration"):
            validate_delivery(self.root, self.ident)

    def test_python_api_redacts_network_exception(self):
        from scripts.production.moss_scale.upload_delivery import publish
        with patch("scripts.production.moss_scale.upload_delivery._publish",
                   side_effect=RuntimeError("https://example.invalid/?signature=private")):
            with self.assertRaises(RuntimeError) as cm:
                publish(self.root, "example/dataset", [self.ident], execute=True)
        self.assertNotIn("signature", str(cm.exception))
        self.assertNotIn("example.invalid", str(cm.exception))

    def test_commit_failure_reports_stage_without_remote_text_and_resets_context(self):
        from scripts.production.moss_scale.upload_delivery import publish
        class Api:
            def __init__(self, **kwargs):pass
            def repo_info(self,*args,**kwargs):return SimpleNamespace(sha='head')
            def get_paths_info(self,*args,**kwargs):return []
            def create_commit(self,*args,**kwargs):
                raise ConnectionError('https://private.invalid/?token=sensitive-marker')
        hub=SimpleNamespace(HfApi=Api,CommitOperationAdd=lambda **kwargs:SimpleNamespace(**kwargs),
                            hf_hub_download=lambda *args,**kwargs:None)
        with patch.dict(sys.modules,{'huggingface_hub':hub}), \
             patch('scripts.production.moss_scale.upload_delivery._hf_token',return_value='synthetic-token'):
            with self.assertRaises(RuntimeError) as failure:
                publish(self.root,'example/dataset',[self.ident],True)
            message=str(failure.exception)
            self.assertIn('stage=commit',message)
            self.assertIn('retryable',message)
            self.assertNotIn('sensitive-marker',message)
            self.assertNotIn('private.invalid',message)
            with self.assertRaises(RuntimeError) as next_failure:
                publish(self.root,'invalid',[self.ident],True)
            self.assertIn('stage=validation',str(next_failure.exception))
            self.assertNotIn('stage=commit',str(next_failure.exception))

    def test_publish_does_not_emit_upstream_warning_content(self):
        from scripts.production.moss_scale.upload_delivery import publish
        def failing(*args, **kwargs):
            warnings.warn("signature=private-warning")
            raise ConnectionError("signature=private-error")
        with patch("scripts.production.moss_scale.upload_delivery._publish", side_effect=failing), \
             patch.object(sys, "stderr", new_callable=io.StringIO) as stderr:
            with self.assertRaises(RuntimeError) as cm:
                publish(self.root, "example/dataset", [self.ident], execute=True)
        self.assertNotIn("private", stderr.getvalue())
        self.assertNotIn("private", str(cm.exception))
        self.assertIn("retryable", str(cm.exception))

    def test_unclassified_recovery_is_bounded_and_requires_pinned_commit(self):
        from scripts.production.moss_scale import upload_delivery as uploader
        for stage,pinned,http_status,expected_calls in (
            ('commit',True,None,2),('verification_content',True,None,1),
            ('commit',False,None,1),('commit',True,403,1)):
            with self.subTest(stage=stage,pinned=pinned,http_status=http_status):
                def failing(*args,**kwargs):
                    uploader._PUBLICATION_STAGE.set(stage)
                    error=RuntimeError('private-signature-marker')
                    if http_status is not None:error.response=SimpleNamespace(status_code=http_status)
                    raise error
                with patch.object(uploader,'_publish',side_effect=failing) as publish:
                    with self.assertRaises(RuntimeError) as failure:
                        uploader.publish(self.root,'example/dataset',[self.ident],True,
                            expected_receipt_hashes={self.ident:'a'*64} if pinned else None)
                self.assertEqual(publish.call_count,expected_calls)
                self.assertNotIn('private-signature-marker',str(failure.exception))
                self.assertNotIn('retryable',str(failure.exception))

    def test_stream_resume_after_remote_commit_response_is_lost(self):
        self._stream_resume_after_lost_response(ConnectionError)

    def test_unclassified_commit_failure_reconciles_without_duplicate_commit(self):
        self._stream_resume_after_lost_response(RuntimeError)

    def _stream_resume_after_lost_response(self, error_type):
        from scripts.production.moss_scale.stream_upload import main, pending_receipts
        from scripts.production.moss_scale.produce_and_upload import _verified_hours
        state = self.root / "upload-state"
        remote = {}  # A new initialized dataset need not have an index yet.
        revisions = []
        non_lfs = set()
        index = self.root / "remote-index.jsonl"

        class Operation:
            def __init__(self, path_in_repo, path_or_fileobj):
                self.path = path_in_repo
                self.payload = path_or_fileobj

        class FakeApi:
            def __init__(self, token):
                pass
            def repo_info(self, *args, **kwargs):
                return SimpleNamespace(sha=f"head-{len(revisions)}")
            def get_paths_info(self, repo, paths, **kwargs):
                items = []
                for path in paths:
                    if path not in remote:
                        continue
                    payload = remote[path]
                    blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
                    items.append(SimpleNamespace(path=path,
                        lfs=None if path in non_lfs else {"sha256": hashlib.sha256(payload).hexdigest()}, blob_id=blob))
                return items
            def create_commit(self, repo, *, parent_commit, operations, **kwargs):
                if parent_commit != f"head-{len(revisions)}":
                    raise ValueError("unexpected parent")
                for op in operations:
                    remote[op.path] = op.payload
                revisions.append(parent_commit)
                if len(revisions) == 1:
                    raise error_type("response lost after commit")
                return SimpleNamespace(oid=f"head-{len(revisions)}")

        def download(*args, **kwargs):
            index.write_bytes(remote["annotations.jsonl"])
            return str(index)

        hub = SimpleNamespace(CommitOperationAdd=Operation, HfApi=FakeApi, hf_hub_download=download)
        done = self.root / "producer-done"
        done.touch()
        argv = ["stream_upload", "--root", str(self.root), "--state-dir", str(state), "--repo", "example/dataset",
                "--follow", "--execute", "--done-file", str(done), "--poll-seconds", "1"]
        with patch.dict(sys.modules, {"huggingface_hub": hub}), \
             patch("scripts.production.moss_scale.upload_delivery._hf_token", return_value="test"), \
             patch("scripts.production.moss_scale.produce_and_upload._hf_token", return_value="test"), \
             patch("scripts.production.moss_scale.stream_upload.time.sleep"), patch.object(sys, "argv", argv):
            main()
            self.assertEqual(len(revisions), 1)
            self.assertEqual(pending_receipts([self.root], state, "example/dataset"), [])
            self.assertEqual(len(pending_receipts([self.root], state, "example/other")), 1)
            hours, revision = _verified_hours({"zh": self.root}, state, "example/dataset")
            self.assertAlmostEqual(hours["zh"], 0.01 / 3600)
            self.assertEqual(revision, "head-1")
            removed = f"audio/{self.ident}.flac"
            saved = remote.pop(removed)
            with self.assertRaisesRegex(RuntimeError, "Current remote verification failed"):
                _verified_hours({"zh": self.root}, state, "example/dataset")
            remote[removed] = saved
            timeline_name = f"audio/{self.ident}.timeline.json"
            timeline_path = self.root / timeline_name
            original_timeline = timeline_path.read_bytes()
            non_lfs.add(timeline_name)
            timeline_path.write_bytes(b"changed timeline")
            remote[timeline_name] = b"changed timeline"
            with self.assertRaisesRegex(RuntimeError, "Current remote verification failed"):
                _verified_hours({"zh": self.root}, state, "example/dataset")
            timeline_path.write_bytes(original_timeline)
            remote[timeline_name] = original_timeline
            receipt_path = self.root / "completed" / f"{self.ident}.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["credited_seconds"] = 0.009
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                pending_receipts([self.root], state, "example/dataset")

    def test_stream_batches_rotate_between_busy_roots(self):
        from scripts.production.moss_scale.stream_upload import next_batch
        a, b = Path("a"), Path("b")
        pending = [{"root": a, "id": "a1", "size": 1}, {"root": a, "id": "a2", "size": 1},
                   {"root": b, "id": "b1", "size": 1}]
        first = next_batch(pending, 1)
        second = next_batch(pending, 1, last_root=first[0]["root"])
        self.assertEqual((first[0]["id"], second[0]["id"]), ("a1", "b1"))

    def test_remote_git_and_lfs_hashes(self):
        content = b"audio bytes"
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        self.assertTrue(_matches_remote(SimpleNamespace(lfs=None, blob_id=blob), content))
        self.assertTrue(_matches_remote(SimpleNamespace(lfs={"sha256": hashlib.sha256(content).hexdigest()}), content))
        self.assertFalse(_matches_remote(SimpleNamespace(lfs={"sha256": "wrong"}), content))


if __name__ == "__main__":
    unittest.main()
