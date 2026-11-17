import json
import unittest
from scripts.production.moss_scale.errors import DraftShapeError, safe_diagnostic


class DraftShapeDiagnosticTests(unittest.TestCase):
    def test_locates_bad_shape_without_echoing_text(self):
        for value, kind, size in (("private-sentinel", "str", None),
                                  (["private-sentinel"], "list", 1), (None, "NoneType", None)):
            report = safe_diagnostic(DraftShapeError(7, value))
            self.assertEqual((report["turn_id"], report["received_type"], report["received_items"]), (7, kind, size))
            self.assertIn("第7轮", report["hint"])
            self.assertNotIn("private-sentinel", json.dumps(report))
