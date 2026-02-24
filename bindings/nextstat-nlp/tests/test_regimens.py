"""Tests for regimen extraction with heuristic backend."""
from __future__ import annotations

import unittest

from nextstat_nlp.regimens import extract_regimens


class TestRegimenExtraction(unittest.TestCase):
    def test_heuristic_extracts_iv_regimen(self):
        texts = ["Patient 101: 500 mg IV infusion, QD dosing."]
        table = extract_regimens(texts, backend="heuristic")
        self.assertGreaterEqual(len(table.records), 1)
        rec = table.records[0]
        self.assertEqual(rec.dose, 500.0)
        self.assertEqual(rec.route, "IV")

    def test_heuristic_extracts_oral_regimen(self):
        texts = ["Patient 102: 250 mg oral twice daily."]
        table = extract_regimens(texts, backend="heuristic")
        self.assertGreaterEqual(len(table.records), 1)
        rec = table.records[0]
        self.assertEqual(rec.dose, 250.0)
        self.assertEqual(rec.route, "oral")

    def test_empty_text_no_dose(self):
        table = extract_regimens(["no dosing info"], backend="heuristic")
        self.assertEqual(len(table.records), 0)

    def test_backend_env(self):
        table = extract_regimens(["500 mg IV"], backend="heuristic")
        self.assertEqual(table.backend_env["backend"], "heuristic")

    def test_text_hash(self):
        texts = ["Patient 101: 500 mg IV"]
        table = extract_regimens(texts, backend="heuristic")
        for r in table.records:
            self.assertIsNotNone(r.text_hash)
            self.assertEqual(len(r.text_hash), 64)


if __name__ == "__main__":
    unittest.main()
