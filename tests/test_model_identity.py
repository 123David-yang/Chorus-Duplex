import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import traceback
import unittest
from unittest.mock import patch

from scripts.sound.synthesis import model_identity
from scripts.sound.synthesis.model_identity import ModelIdentityError, qwen_snapshot_identity


def _tensor_bytes(values: bytes = b"abcd", name: str = "weight") -> bytes:
    header = json.dumps({name: {"dtype": "U8", "shape": [len(values)],
                               "data_offsets": [0, len(values)]}},
                        separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + values


def _write(path: Path, value: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value.encode("utf-8") if isinstance(value, str) else value)


def _snapshot(root: Path) -> None:
    contents = {
        "config.json": json.dumps({"model_type": "qwen3_tts", "tts_model_type": "base"}),
        "generation_config.json": "{}",
        "preprocessor_config.json": "{}",
        "tokenizer_config.json": "{}",
        "merges.txt": "a b\n",
        "vocab.json": '{"a":0}',
        "model.safetensors": _tensor_bytes(),
        "speech_tokenizer/config.json": '{"model_type":"qwen3_tts_tokenizer_12hz"}',
        "speech_tokenizer/configuration.json": "{}",
        "speech_tokenizer/preprocessor_config.json": "{}",
        "speech_tokenizer/model.safetensors": _tensor_bytes(b"wxyz", "codec"),
    }
    for name, content in contents.items():
        _write(root / name, content)


def _inventory(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


class ModelIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.first = self.base / "first"
        _snapshot(self.first)
        self.inventory = _inventory(self.first)

    def assert_failure(self, code, root=None, **kwargs):
        with self.assertRaises(ModelIdentityError) as caught:
            qwen_snapshot_identity(root or self.first, expected_files=kwargs.pop("expected_files", self.inventory), **kwargs)
        self.assertEqual(caught.exception.code, code)
        self.assertNotIn(str(self.base), str(caught.exception))

    def test_equal_contents_at_different_absolute_paths_have_same_identity(self):
        second = self.base / "other" / "different-parent"
        shutil.copytree(self.first, second)
        first_id = qwen_snapshot_identity(self.first, expected_files=self.inventory)
        from scripts.sound.synthesis.model_identity import qwen_content_identity
        content = qwen_content_identity(self.first)
        self.assertEqual(content["sha256"], first_id["sha256"])
        self.assertEqual(content["verification"], "content-only")
        self.assertEqual(first_id["verification"], "external-inventory")
        second_id = qwen_snapshot_identity(second, expected_files=self.inventory)
        self.assertEqual(first_id, second_id)
        self.assertEqual(first_id["file_count"], len(self.inventory))
        self.assertEqual(len(first_id["sha256"]), 64)
        self.assertNotIn(str(self.base), json.dumps(first_id))
        self.assertEqual(qwen_snapshot_identity(second, expected_files=self.inventory,
                                                expected_sha256=first_id["sha256"]), first_id)

    def test_changed_weights_require_new_inventory_and_change_identity(self):
        first_id = qwen_snapshot_identity(self.first, expected_files=self.inventory)
        _write(self.first / "model.safetensors", _tensor_bytes(b"abce"))
        self.assert_failure("inventory_digest_mismatch")
        changed_inventory = _inventory(self.first)
        changed = qwen_snapshot_identity(self.first, expected_files=changed_inventory)
        self.assertNotEqual(first_id["sha256"], changed["sha256"])
        self.assert_failure("snapshot_identity_mismatch", expected_files=changed_inventory,
                            expected_sha256=first_id["sha256"])

    def test_model_roles_are_explicit_and_cannot_be_swapped(self):
        identity = qwen_snapshot_identity(self.first, expected_files=self.inventory, expected_model_type="base")
        self.assertEqual(identity["tts_model_type"], "base")
        with patch.object(model_identity, "_hash_file", wraps=model_identity._hash_file) as reader:
            self.assert_failure("snapshot_model_type_mismatch", expected_model_type="custom_voice")
        self.assertEqual(reader.call_count, 1)  # fail before reading multi-GB weights
        self.assert_failure("invalid_expected_model_type", expected_model_type="invented")
        _write(self.first / "config.json", '{"model_type":"qwen3_tts","tts_model_type":"custom_voice"}')
        custom = qwen_snapshot_identity(self.first, expected_files=_inventory(self.first),
                                        expected_model_type="custom_voice")
        self.assertEqual(custom["tts_model_type"], "custom_voice")
        self.assertNotEqual(identity["sha256"], custom["sha256"])

    def test_missing_trusted_inventory_and_extra_or_missing_files_fail_closed(self):
        self.assert_failure("trusted_inventory_required", expected_files={})
        with self.assertRaisesRegex(ModelIdentityError, "trusted_inventory_required"):
            qwen_snapshot_identity(self.first)
        _write(self.first / "unexpected.bin", b"extra")
        self.assert_failure("inventory_file_set_mismatch")
        (self.first / "unexpected.bin").unlink()
        (self.first / "vocab.json").unlink()
        self.assert_failure("inventory_file_set_mismatch")

    def test_inventory_paths_cannot_escape_or_use_windows_paths(self):
        for name in ("../outside", "/absolute/file", "a/../b", "a//b", "a/./b",
                     r"C:\outside", "C:relative", r"dir\file", "a/\x00b", "CON/config.json",
                     "a/name.", "a/name ", "a/file?.json"):
            with self.subTest(name=name):
                self.assert_failure("invalid_inventory_path", expected_files={name: "0" * 64})

    def test_empty_and_corrupted_safetensors_are_rejected(self):
        _write(self.first / "empty", b"")
        self.assert_failure("snapshot_contains_empty_file", expected_files={**self.inventory, "empty": "0" * 64})
        (self.first / "empty").unlink()
        _write(self.first / "model.safetensors", b"not-a-safetensors-file")
        self.assert_failure("invalid_safetensors", expected_files=_inventory(self.first))

    def test_indexed_shards_are_checked_and_cannot_escape(self):
        (self.first / "model.safetensors").unlink()
        shard = "model-00001-of-00001.safetensors"
        _write(self.first / shard, _tensor_bytes(b"abcd", "weight"))
        index = self.first / "model.safetensors.index.json"
        _write(index, json.dumps({"weight_map": {"weight": shard}}))
        self.assertEqual(qwen_snapshot_identity(self.first, expected_files=_inventory(self.first))["file_count"],
                         len(_inventory(self.first)))
        _write(index, json.dumps({"weight_map": {"weight": "../outside.safetensors"}}))
        self.assert_failure("invalid_inventory_path", expected_files=_inventory(self.first))
        _write(index, json.dumps({"weight_map": {"weight": "missing.safetensors"}}))
        self.assert_failure("missing_model_shard", expected_files=_inventory(self.first))
        _write(index, json.dumps({"weight_map": {"wrong_tensor": shard}}))
        self.assert_failure("invalid_weight_index", expected_files=_inventory(self.first))

    def test_invalid_or_incomplete_known_qwen_layout_is_rejected(self):
        (self.first / "speech_tokenizer" / "configuration.json").unlink()
        self.assert_failure("incomplete_qwen_layout", expected_files=_inventory(self.first))
        _write(self.first / "speech_tokenizer" / "configuration.json", "{}")
        _write(self.first / "config.json", '{"model_type":"something_else"}')
        self.assert_failure("unsupported_qwen_model", expected_files=_inventory(self.first))

    def test_detects_modification_during_hashing(self):
        original = model_identity._hash_file
        changed = False
        def mutate_after_read(path, signature, capture_json):
            nonlocal changed
            result = original(path, signature, capture_json)
            if not changed:
                changed = True
                _write(self.first / "merges.txt", "different content\n")
            return result
        with patch.object(model_identity, "_hash_file", side_effect=mutate_after_read):
            with self.assertRaises(ModelIdentityError) as caught:
                qwen_snapshot_identity(self.first, expected_files=self.inventory)
        self.assertIn(caught.exception.code, ("snapshot_changed", "inventory_digest_mismatch"))
        self.assertNotIn(str(self.base), str(caught.exception))

    def test_detects_file_added_after_last_read(self):
        original = model_identity._hash_file
        reads = 0
        def add_after_last_read(path, signature, capture_json):
            nonlocal reads
            result = original(path, signature, capture_json)
            reads += 1
            if reads == len(self.inventory):
                _write(self.first / "new.bin", b"added after read")
            return result
        with patch.object(model_identity, "_hash_file", side_effect=add_after_last_read):
            self.assert_failure("snapshot_changed")

    def test_simulated_unreadable_file_does_not_disclose_absolute_path(self):
        with patch.object(model_identity, "_hash_file", side_effect=PermissionError(str(self.first / "secret"))):
            with self.assertRaises(ModelIdentityError) as caught:
                qwen_snapshot_identity(self.first, expected_files=self.inventory)
        self.assertEqual(caught.exception.code, "snapshot_unreadable")
        rendered = "".join(traceback.format_exception(caught.exception))
        self.assertNotIn(str(self.base), rendered)
        self.assertIsNone(caught.exception.__context__)

    def test_unknown_weight_index_and_private_files_are_rejected_before_reading(self):
        _write(self.first / "other.safetensors.index.json", '{"weight_map":{}}')
        self.assert_failure("unsupported_weight_index", expected_files=_inventory(self.first))
        (self.first / "other.safetensors.index.json").unlink()
        _write(self.first / ".git" / "config", "private sentinel")
        with patch.object(model_identity, "_hash_file", side_effect=AssertionError("contents opened")):
            self.assert_failure("snapshot_contains_private_path", expected_files=self.inventory)

    def test_hard_link_to_outside_file_is_rejected(self):
        outside = self.base / "outside.bin"
        outside.write_bytes(b"outside")
        link = self.first / "linked.bin"
        try:
            os.link(outside, link)
        except (OSError, NotImplementedError):
            self.skipTest("The test filesystem does not support hard links")
        self.assert_failure("snapshot_contains_hard_link", expected_files=self.inventory)

    @unittest.skipUnless(os.name == "nt", "Windows junction test")
    def test_windows_junctions_are_rejected_inside_and_above_snapshot(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "sentinel").write_text("outside", encoding="utf-8")
        inside_link = self.first / "junction"
        command = ["cmd", "/c", "mklink", "/J", str(inside_link), str(outside)]
        made = subprocess.run(command, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if made.returncode:
            self.skipTest("The test filesystem does not support directory junctions")
        try:
            self.assert_failure("snapshot_contains_link")
        finally:
            os.rmdir(inside_link)
        alias = self.base / "alias"
        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(alias), str(self.first)],
                              capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if made.returncode:
            self.skipTest("The test filesystem does not support directory junctions")
        try:
            self.assert_failure("snapshot_contains_link", root=alias)
        finally:
            os.rmdir(alias)
        self.assertTrue((outside / "sentinel").is_file())

    def test_symlink_inside_snapshot_is_rejected_when_platform_allows_creation(self):
        link = self.first / "linked"
        try:
            os.symlink(self.base / "outside", link)
        except (OSError, NotImplementedError):
            self.skipTest("Creating symbolic links requires extra operating-system privileges")
        self.addCleanup(link.unlink, missing_ok=True)
        self.assert_failure("snapshot_contains_link")

    def test_windows_reparse_points_are_rejected(self):
        fake = type("FakeStat", (), {"st_mode": 0, "st_file_attributes": 0x400})()
        with self.assertRaisesRegex(ModelIdentityError, "snapshot_contains_link"):
            model_identity._reject_link(fake)


if __name__ == "__main__":
    unittest.main()
