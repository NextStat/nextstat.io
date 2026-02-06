"""Contract tests for the explicit Posterior API (Phase 3/5 standards)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import nextstat


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def test_posterior_logpdf_matches_negative_nll_gaussian_mean():
    m = nextstat.GaussianMeanModel([1.0, 2.0, 3.0, 4.0], sigma=2.0)
    post = nextstat.Posterior(m)

    theta = [0.25]
    lp = float(post.logpdf(theta))
    nll = float(m.nll(theta))
    assert lp == pytest.approx(-nll, rel=0.0, abs=1e-12)

    g = list(map(float, post.grad(theta)))
    g_nll = list(map(float, m.grad_nll(theta)))
    assert g == pytest.approx([-x for x in g_nll], rel=0.0, abs=1e-12)


def test_posterior_transform_roundtrip_histfactory_smoke():
    ws = json.loads((FIXTURES_DIR / "simple_workspace.json").read_text())
    m = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))
    post = nextstat.Posterior(m)

    theta0 = m.suggested_init()
    z = post.to_unconstrained(theta0)
    theta1 = post.to_constrained(z)

    assert len(z) == m.n_params()
    assert len(theta1) == m.n_params()
    assert theta1 == pytest.approx(theta0, rel=0.0, abs=1e-10)

    # Basic finiteness checks.
    assert float(post.logpdf(theta0)) == pytest.approx(-float(m.nll(theta0)), rel=0.0, abs=1e-10)
    assert all(v == v and abs(v) < 1e100 for v in post.grad(theta0))

