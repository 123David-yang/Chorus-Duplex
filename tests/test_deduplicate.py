import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

from scripts.production.moss_scale.deduplicate import claim, rebuild
from scripts.production.moss_scale.delivery import recover_deliveries


class SharedDeduplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.db = base / "dedupe.sqlite3"
        self.roots = [base / "zh", base / "en"]
        for root in self.roots:
            for subdir in ("completed", "staging", "duplicates"):
                (root / subdir).mkdir(parents=True, exist_ok=True)

    def receipt(self, root, ident, pending=False):
        directory = root / "staging" / ident if pending else root / "completed"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ("delivery.pending.json" if pending else f"{ident}.json")
        path.write_text(json.dumps({"annotation": {"dialogue_id": ident},
                                    "sha256": {f"{ident}.flac": "a" * 64}}), encoding="utf-8")
        return path

    def test_concurrent_cross_language_claim_and_idempotent_resume(self):
        rebuild(self.db, self.roots)
        barrier = Barrier(2)
        def try_claim(item):
            root, ident = item
            barrier.wait()
            return ident, claim(self.db, root, ident, "b" * 64)
        with ThreadPoolExecutor(max_workers=2) as pool:
            result = list(pool.map(try_claim, zip(self.roots, ("zh-001", "en-001"))))
        winner = next(ident for ident, previous in result if previous is None)
        loser = next(ident for ident, previous in result if previous is not None)
        self.assertEqual(next(previous for ident, previous in result if ident == loser), winner)
        root = self.roots[0] if winner == "zh-001" else self.roots[1]
        self.assertIsNone(claim(self.db, root, winner, "b" * 64))

    def test_pending_recovery_skips_duplicate_and_preserves_both(self):
        first = self.receipt(self.roots[0], "zh-001", pending=True)
        second = self.receipt(self.roots[1], "en-001", pending=True)
        rebuild(self.db, self.roots)
        with patch("scripts.production.moss_scale.delivery.finish_delivery") as finish:
            recover_deliveries(self.roots[0], self.db)
            recover_deliveries(self.roots[1], self.db)
        finish.assert_called_once_with(self.roots[0], first)
        self.assertTrue(second.exists())
        marker = json.loads((self.roots[1] / "duplicates" / "en-001.json").read_text(encoding="utf-8"))
        self.assertEqual(marker["duplicate_of"], "zh-001")

    def test_existing_duplicate_receipts_are_preserved_and_block_resume(self):
        first = self.receipt(self.roots[0], "zh-001")
        second = self.receipt(self.roots[1], "en-001")
        with self.assertRaisesRegex(ValueError, "duplicate clean audio"):
            rebuild(self.db, self.roots)
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())


if __name__ == "__main__":
    unittest.main()
