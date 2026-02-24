"""Tests for survival extraction pipeline."""
from __future__ import annotations

import unittest

from nextstat_nlp import extract_survival_records
from nextstat_nlp._errors import ExtractionFailed


class TestSurvivalExtraction(unittest.TestCase):
    def test_heuristic_backend_extracts_required_fields(self) -> None:
        texts = [
            "Subject 001: progressed at day 84. Age 63.",
            "Subject 002: withdrew at week 12 (lost to follow-up). Age 58.",
        ]
        ds = extract_survival_records(texts, backend="heuristic")
        self.assertEqual(len(ds.records), 2)
        self.assertEqual(ds.records[0].time, 84.0)
        self.assertEqual(ds.records[0].event, 1)
        self.assertEqual(ds.records[1].time, 84.0)  # 12 weeks
        self.assertEqual(ds.records[1].event, 0)

    def test_design_matrix_shape(self) -> None:
        texts = [
            "Subject 001: progressed at day 10. Age 50.",
            "Subject 002: withdrew at week 1. Age 60.",
        ]
        ds = extract_survival_records(texts, backend="heuristic")
        time, event, X, feature_names = ds.to_design_matrix()
        self.assertEqual(time, [10.0, 7.0])
        self.assertEqual(event, [1, 0])
        # Heuristic backend now extracts covariates
        self.assertIn("age", feature_names)
        self.assertEqual(len(X), 2)

    def test_months_format(self) -> None:
        texts = ["Subject 001: died at 6 months."]
        ds = extract_survival_records(texts, backend="heuristic")
        self.assertAlmostEqual(ds.records[0].time, 182.625, places=2)
        self.assertEqual(ds.records[0].event, 1)

    def test_years_format(self) -> None:
        texts = ["Subject 001: alive at 2 years."]
        ds = extract_survival_records(texts, backend="heuristic")
        self.assertAlmostEqual(ds.records[0].time, 730.5, places=1)
        self.assertEqual(ds.records[0].event, 0)

    def test_number_before_unit(self) -> None:
        """Most common clinical format: '84 days', not 'day 84'."""
        texts = ["Subject 001: progressed at 84 days."]
        ds = extract_survival_records(texts, backend="heuristic")
        self.assertEqual(ds.records[0].time, 84.0)

    def test_covariates_extracted(self) -> None:
        texts = ["Subject 001: progressed at day 84. Age 63. Dose 20 mg."]
        ds = extract_survival_records(texts, backend="heuristic")
        r = ds.records[0]
        self.assertEqual(r.covariates.get("age"), 63.0)
        self.assertEqual(r.covariates.get("dose"), 20.0)

    def test_text_hash_present(self) -> None:
        texts = ["Subject 001: died at day 10."]
        ds = extract_survival_records(texts, backend="heuristic")
        self.assertIsNotNone(ds.records[0].text_hash)
        self.assertEqual(len(ds.records[0].text_hash), 64)

    def test_document_id_passthrough(self) -> None:
        texts = ["Subject 001: died at day 10."]
        ds = extract_survival_records(texts, backend="heuristic", document_id="doc-123")
        self.assertEqual(ds.records[0].document_id, "doc-123")

    def test_missing_fields_raises(self) -> None:
        with self.assertRaises(ExtractionFailed):
            extract_survival_records(["no survival info at all"], backend="heuristic")

    def test_auto_subject_id(self) -> None:
        texts = ["progressed at day 10."]  # No subject ID in text
        ds = extract_survival_records(texts, backend="heuristic")
        self.assertEqual(ds.records[0].subject_id, "0000")


if __name__ == "__main__":
    unittest.main()
