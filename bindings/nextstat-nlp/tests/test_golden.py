"""Golden-file reproducibility tests (heuristic backend — CI-safe)."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from nextstat_nlp._provenance import collect_provenance
from nextstat_nlp.backends import get_backend
from nextstat_nlp.priors import extract_prior_candidates
from nextstat_nlp.regimens import extract_regimens
from nextstat_nlp.survival import extract_survival_records

_GOLDEN_DIR = Path(__file__).parent / "golden"


def _read_golden(name: str) -> list[str]:
    path = _GOLDEN_DIR / name
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


class TestGoldenSurvival(unittest.TestCase):
    def test_oncology_extraction(self):
        texts = _read_golden("survival_oncology.txt")
        ds = extract_survival_records(texts, backend="heuristic")
        self.assertEqual(len(ds.records), 5)

        # Subject 001: progressed at day 84
        r0 = ds.records[0]
        self.assertEqual(r0.time, 84.0)
        self.assertEqual(r0.event, 1)
        self.assertIn("age", r0.covariates)
        self.assertEqual(r0.covariates["age"], 63.0)

        # Subject 002: lost to follow-up at week 12
        r1 = ds.records[1]
        self.assertEqual(r1.time, 84.0)
        self.assertEqual(r1.event, 0)

        # Subject 003: died at 6 months
        r2 = ds.records[2]
        self.assertAlmostEqual(r2.time, 182.625, places=2)
        self.assertEqual(r2.event, 1)

        # Subject 004: alive at 2 years
        r3 = ds.records[3]
        self.assertAlmostEqual(r3.time, 730.5, places=1)
        self.assertEqual(r3.event, 0)

        # Subject 005: relapsed at day 142
        r4 = ds.records[4]
        self.assertEqual(r4.time, 142.0)
        self.assertEqual(r4.event, 1)

    def test_design_matrix_has_covariates(self):
        texts = _read_golden("survival_oncology.txt")
        ds = extract_survival_records(texts, backend="heuristic")
        times, events, X, names = ds.to_design_matrix()
        self.assertEqual(len(times), 5)
        self.assertIn("age", names)


class TestGoldenPriors(unittest.TestCase):
    def test_pk_abstract_extraction(self):
        texts = _read_golden("priors_pk_abstract.txt")
        bundle = extract_prior_candidates(texts, backend="heuristic")
        # Should extract at least CL and V1 priors
        param_names = {c.param_name for c in bundle.candidates}
        self.assertTrue(
            param_names & {"CL", "V1"},
            f"Expected CL and V1 priors, got {param_names}",
        )
        for c in bundle.candidates:
            self.assertEqual(c.status, "candidate")


class TestGoldenRegimens(unittest.TestCase):
    def test_protocol_extraction(self):
        texts = _read_golden("regimen_protocol.txt")
        table = extract_regimens(texts, backend="heuristic")
        self.assertGreaterEqual(len(table.records), 2)
        doses = {r.dose for r in table.records}
        self.assertTrue(doses & {500.0, 250.0, 100.0})


class TestProvenance(unittest.TestCase):
    def test_provenance_bundle_serializable(self):
        be = get_backend("heuristic")
        texts = ["Test text for provenance."]
        prov = collect_provenance(be, texts)
        d = prov.to_dict()
        # Must be JSON-serializable
        s = json.dumps(d)
        self.assertIn("heuristic", s)
        self.assertEqual(len(prov.input_hashes), 1)
        self.assertEqual(len(prov.input_hashes[0]), 64)


if __name__ == "__main__":
    unittest.main()
