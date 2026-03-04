"""Tests for regimen extraction with heuristic backend."""
from __future__ import annotations

import unittest

from nextstat_nlp.regimens import extract_regimens, to_nextstat_regimens
from nextstat_nlp.schemas import RegimenRecord


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

    def test_to_nextstat_regimens_expand_frequency(self):
        rec = RegimenRecord(
            subject_id="S1",
            dose=10.0,
            route="oral",
            start_time=0.0,
            duration=2.0,  # course duration: 2 days
            frequency="QD",
            amount_units="mg",
        )
        out = to_nextstat_regimens([rec], expand_frequency=True)
        self.assertEqual(len(out), 1)
        events = out[0]["events"]
        self.assertEqual([e["time"] for e in events], [0.0, 1.0, 2.0])
        self.assertTrue(all(e["duration"] == 0.0 for e in events))

    def test_to_nextstat_regimens_infusion_duration(self):
        rec = RegimenRecord(
            subject_id="S2",
            dose=1000.0,
            route="IV",
            start_time=0.0,
            infusion_duration=2.0 / 24.0,
            frequency="once",
            amount_units="mg",
        )
        out = to_nextstat_regimens([rec])
        ev = out[0]["events"][0]
        self.assertAlmostEqual(ev["duration"], 2.0 / 24.0, places=6)


if __name__ == "__main__":
    unittest.main()
