import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.production.moss_scale.delivery import DeliveryIndex


class DeliveryIndexTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "completed").mkdir()

    def receipt(self, ident, seconds, kind):
        row = {
            "annotation": {
                "dialogue_id": ident,
                "turns": [{"kind": kind, "text": "sample"}],
            },
            "credited_seconds": seconds,
        }
        path = self.root / "completed" / f"{ident}.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        return row

    def annotations(self):
        return [
            json.loads(line)["dialogue_id"]
            for line in (self.root / "annotations.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    def test_existing_new_duplicate_and_out_of_order(self):
        old = self.receipt("d002", 10, "speech")
        index = DeliveryIndex(self.root, 1)
        self.assertAlmostEqual(index.hours, 10 / 3600)
        new = self.receipt("d003", 20, "backchannel")
        self.assertAlmostEqual(index.add(new), 30 / 3600)
        self.assertAlmostEqual(index.add(new), 30 / 3600)
        earlier = self.receipt("d001", 30, "interruption")
        self.assertAlmostEqual(index.add(earlier), 60 / 3600)
        self.assertEqual(self.annotations(), ["d001", "d002", "d003"])
        status = json.loads((self.root / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["dialogues"], 3)
        self.assertEqual(status["turn_kinds"], {"speech": 1, "backchannel": 1, "interruption": 1})
        self.assertAlmostEqual(index.add(old), 60 / 3600)

    def test_crash_rebuild_prevents_duplicate_or_missing_annotation(self):
        self.receipt("d001", 10, "speech")
        index = DeliveryIndex(self.root, 1)
        second = self.receipt("d002", 20, "backchannel")
        with patch.object(index, "_write_status", side_effect=OSError("simulated failure")):
            with self.assertRaises(OSError):
                index.add(second)
        third = self.receipt("d003", 30, "incomplete")
        self.assertAlmostEqual(index.add(third), 60 / 3600)
        self.assertEqual(self.annotations(), ["d001", "d002", "d003"])
        (self.root / "annotations.jsonl").write_text("partial", encoding="utf-8")
        restored = DeliveryIndex(self.root, 1)
        self.assertAlmostEqual(restored.hours, 60 / 3600)
        self.assertEqual(self.annotations(), ["d001", "d002", "d003"])


if __name__ == "__main__":
    unittest.main()
