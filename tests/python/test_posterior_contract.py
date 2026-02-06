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


def test_posterior_normal_prior_affects_logpdf_and_grad_by_name():
    m = nextstat.GaussianMeanModel([1.0, 2.0, 3.0, 4.0], sigma=2.0)
    post = nextstat.Posterior(m)

    theta = [0.25]
    base_lp = float(post.logpdf(theta))
    base_g = list(map(float, post.grad(theta)))

    post.set_prior_normal("mu", center=0.0, width=1.0)
    lp = float(post.logpdf(theta))
    g = list(map(float, post.grad(theta)))

    expected_lp = base_lp - 0.5 * (theta[0] ** 2)
    expected_g = [base_g[0] - theta[0]]

    assert lp == pytest.approx(expected_lp, rel=0.0, abs=1e-12)
    assert g == pytest.approx(expected_g, rel=0.0, abs=1e-12)

    ps = post.priors()
    assert ps["mu"]["type"] == "normal"
    assert float(ps["mu"]["center"]) == pytest.approx(0.0)
    assert float(ps["mu"]["width"]) == pytest.approx(1.0)

    post.clear_priors()
    assert float(post.logpdf(theta)) == pytest.approx(base_lp, rel=0.0, abs=1e-12)


def test_posterior_map_fit_and_sampling_accept_posterior_smoke():
    # MAP should be pulled strongly toward the prior center.
    m = nextstat.GaussianMeanModel([0.0, 1.0, 2.0], sigma=1.0)
    post = nextstat.Posterior(m)
    post.set_prior_normal("mu", center=10.0, width=0.1)

    r = nextstat.map_fit(post)
    assert r.success
    assert float(r.bestfit[0]) > 9.0

    # Smoke: NUTS accepts Posterior (i.e. includes priors).
    raw = nextstat.sample(post, n_chains=1, n_warmup=20, n_samples=20, seed=7, max_treedepth=4)
    assert isinstance(raw, dict)
    assert raw.get("param_names") == ["mu"]
