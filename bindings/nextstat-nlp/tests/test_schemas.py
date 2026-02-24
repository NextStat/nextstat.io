"""Tests for schema dataclasses — creation, immutability, serialisation."""
from __future__ import annotations

import unittest
from dataclasses import asdict

from nextstat_nlp.schemas import (
    ExtractedSpan,
    PriorBundle,
    PriorCandidate,
    RegimenRecord,
    RegimenTable,
    SurvivalDataset,
    SurvivalRecord,
)


class TestExtractedSpan(unittest.TestCase):
    def test_creation(self):
        s = ExtractedSpan(label="age", text="63", start=10, end=12, score=0.95)
        self.assertEqual(s.label, "age")
        self.assertEqual(s.score, 0.95)

    def test_frozen(self):
        s = ExtractedSpan(label="age", text="63", start=0, end=2)
        with self.assertRaises(AttributeError):
            s.label = "dose"  # type: ignore[misc]


class TestSurvivalRecord(unittest.TestCase):
    def test_creation(self):
        r = SurvivalRecord(subject_id="001", time=84.0, event=1, covariates={"age": 63.0})
        self.assertEqual(r.subject_id, "001")
        self.assertEqual(r.covariates["age"], 63.0)

    def test_to_dict(self):
        r = SurvivalRecord(subject_id="001", time=84.0, event=1)
        d = asdict(r)
        self.assertEqual(d["subject_id"], "001")
        self.assertEqual(d["time"], 84.0)


class TestSurvivalDataset(unittest.TestCase):
    def test_design_matrix(self):
        records = [
            SurvivalRecord(subject_id="A", time=10.0, event=1, covariates={"age": 50.0}),
            SurvivalRecord(subject_id="B", time=20.0, event=0, covariates={"age": 60.0}),
        ]
        ds = SurvivalDataset(records=records, backend_env={})
        times, events, X, names = ds.to_design_matrix()
        self.assertEqual(times, [10.0, 20.0])
        self.assertEqual(events, [1, 0])
        self.assertEqual(names, ["age"])
        self.assertEqual(X, [[50.0], [60.0]])


class TestPriorCandidate(unittest.TestCase):
    def test_creation(self):
        p = PriorCandidate(param_name="CL", dist="lognormal", location=3.5, scale=1.2, units="L/h")
        self.assertEqual(p.param_name, "CL")
        self.assertEqual(p.status, "candidate")

    def test_frozen(self):
        p = PriorCandidate(param_name="CL", dist="normal", location=1.0, scale=0.5)
        with self.assertRaises(AttributeError):
            p.status = "approved"  # type: ignore[misc]


class TestRegimenRecord(unittest.TestCase):
    def test_creation(self):
        r = RegimenRecord(subject_id="101", dose=500.0, route="IV", amount_units="mg", frequency="QD")
        self.assertEqual(r.dose, 500.0)
        self.assertEqual(r.route, "IV")

    def test_serialization(self):
        r = RegimenRecord(subject_id="101", dose=500.0, route="IV")
        d = asdict(r)
        self.assertEqual(d["dose"], 500.0)


class TestPriorBundle(unittest.TestCase):
    def test_empty(self):
        b = PriorBundle(candidates=[], backend_env={"backend": "heuristic"})
        self.assertEqual(len(b.candidates), 0)


class TestRegimenTable(unittest.TestCase):
    def test_empty(self):
        t = RegimenTable(records=[], backend_env={"backend": "heuristic"})
        self.assertEqual(len(t.records), 0)


if __name__ == "__main__":
    unittest.main()
