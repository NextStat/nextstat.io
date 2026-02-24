"""Tests for prior-candidate extraction with heuristic backend."""
from __future__ import annotations

import unittest

from nextstat_nlp.priors import extract_prior_candidates


class TestPriorExtraction(unittest.TestCase):
    def test_heuristic_extracts_pk_priors(self):
        texts = [
            "CL was estimated with a lognormal prior, mean 3.5 SD 1.2 L/h.",
            "V1 followed a normal distribution, mean 45.0 SD 12.0 L.",
        ]
        bundle = extract_prior_candidates(texts, backend="heuristic")
        self.assertGreaterEqual(len(bundle.candidates), 2)

        # Check first candidate
        cl_cands = [c for c in bundle.candidates if c.param_name == "CL"]
        self.assertTrue(len(cl_cands) >= 1, f"Expected CL candidate, got {bundle.candidates}")
        cl = cl_cands[0]
        self.assertEqual(cl.dist, "lognormal")
        self.assertEqual(cl.location, 3.5)
        self.assertEqual(cl.scale, 1.2)
        self.assertEqual(cl.status, "candidate")

    def test_heuristic_extracts_v1(self):
        texts = ["V1 followed a normal distribution, mean 45.0 SD 12.0 L."]
        bundle = extract_prior_candidates(texts, backend="heuristic")
        v1_cands = [c for c in bundle.candidates if c.param_name == "V1"]
        self.assertTrue(len(v1_cands) >= 1)
        v1 = v1_cands[0]
        self.assertEqual(v1.dist, "normal")
        self.assertEqual(v1.location, 45.0)
        self.assertEqual(v1.scale, 12.0)

    def test_empty_text(self):
        bundle = extract_prior_candidates(["no prior info here"], backend="heuristic")
        self.assertEqual(len(bundle.candidates), 0)

    def test_backend_env(self):
        bundle = extract_prior_candidates(["CL mean 1.0 SD 0.5"], backend="heuristic")
        self.assertEqual(bundle.backend_env["backend"], "heuristic")

    def test_text_hash_present(self):
        texts = ["CL lognormal mean 3.5 SD 1.2 L/h"]
        bundle = extract_prior_candidates(texts, backend="heuristic")
        for c in bundle.candidates:
            self.assertIsNotNone(c.text_hash)
            self.assertEqual(len(c.text_hash), 64)  # SHA-256 hex


if __name__ == "__main__":
    unittest.main()
