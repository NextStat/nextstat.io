"""Golden parity tests for LMM marginal likelihood (Phase 9 Pack B).

These tests load pre-generated fixtures and verify:
1. NLL at truth params matches the golden reference (bit-exact on same platform).
2. MLE fit converges and achieves NLL <= golden NLL at truth.
3. Fitted parameters are within tolerance of golden MLE params.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import nextstat

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _load(name: str) -> dict:
    path = FIXTURES_DIR / name
    if not path.exists():
        pytest.skip(f"fixture not found: {path}")
    return json.loads(path.read_text())


def _build_model(fixture: dict):
    cfg = fixture["model_config"]
    data = fixture["data"]
    kwargs = dict(
        x=data["x"],
        y=data["y"],
        include_intercept=cfg["include_intercept"],
        group_idx=data["group_idx"],
        n_groups=data["n_groups"],
    )
    if cfg.get("random_slope_feature_idx") is not None:
        kwargs["random_slope_feature_idx"] = cfg["random_slope_feature_idx"]
    return nextstat.LmmMarginalModel(**kwargs)


class TestLmmRandomInterceptGolden:
    @pytest.fixture(autouse=True)
    def _load_fixture(self):
        self.fixture = _load("lmm_random_intercept.json")
        self.model = _build_model(self.fixture)

    def test_nll_at_truth_matches_golden(self):
        truth = self.fixture["expected"]["truth_params_vector"]
        nll = float(self.model.nll(truth))
        golden = self.fixture["expected"]["nll_at_truth"]
        assert abs(nll - golden) < 1e-6, f"NLL at truth: {nll} vs golden {golden}"

    def test_mle_converges(self):
        mle = nextstat.MaximumLikelihoodEstimator()
        r = mle.fit(self.model)
        assert r.converged

    def test_mle_nll_beats_truth(self):
        mle = nextstat.MaximumLikelihoodEstimator()
        r = mle.fit(self.model)
        golden_nll_truth = self.fixture["expected"]["nll_at_truth"]
        assert float(r.nll) <= golden_nll_truth + 1e-3

    def test_mle_params_within_tolerance(self):
        mle = nextstat.MaximumLikelihoodEstimator()
        r = mle.fit(self.model)
        golden = self.fixture["expected"]["mle_params"]
        names = self.model.parameter_names()
        fitted = dict(zip(names, [float(v) for v in r.parameters]))

        for name in names:
            got = fitted[name]
            ref = golden[name]
            tol = 0.05 if "log_" in name else 0.1
            assert abs(got - ref) < tol, f"{name}: {got} vs golden {ref} (tol={tol})"


class TestLmmRandomInterceptSlopeGolden:
    @pytest.fixture(autouse=True)
    def _load_fixture(self):
        self.fixture = _load("lmm_random_intercept_slope.json")
        self.model = _build_model(self.fixture)

    def test_nll_at_truth_matches_golden(self):
        truth = self.fixture["expected"]["truth_params_vector"]
        nll = float(self.model.nll(truth))
        golden = self.fixture["expected"]["nll_at_truth"]
        assert abs(nll - golden) < 1e-6, f"NLL at truth: {nll} vs golden {golden}"

    def test_mle_converges(self):
        mle = nextstat.MaximumLikelihoodEstimator()
        r = mle.fit(self.model)
        assert r.converged

    def test_mle_nll_beats_truth(self):
        mle = nextstat.MaximumLikelihoodEstimator()
        r = mle.fit(self.model)
        golden_nll_truth = self.fixture["expected"]["nll_at_truth"]
        assert float(r.nll) <= golden_nll_truth + 1e-3

    def test_mle_params_within_tolerance(self):
        mle = nextstat.MaximumLikelihoodEstimator()
        r = mle.fit(self.model)
        golden = self.fixture["expected"]["mle_params"]
        names = self.model.parameter_names()
        fitted = dict(zip(names, [float(v) for v in r.parameters]))

        for name in names:
            got = fitted[name]
            ref = golden[name]
            tol = 0.05 if "log_" in name else 0.1
            assert abs(got - ref) < tol, f"{name}: {got} vs golden {ref} (tol={tol})"
