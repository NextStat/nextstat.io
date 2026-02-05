"""Contract tests for Python bindings (API shape + error behavior).

These tests are intentionally independent from pyhf parity tests.
They validate:
- error types/messages for invalid inputs
- basic method contracts (lengths, return types)
- `fit()` wrapper and `MaximumLikelihoodEstimator.fit()` agree on shape
"""

import json
from pathlib import Path

import pytest

import nextstat


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


def test_model_from_workspace_rejects_invalid_json():
    with pytest.raises(ValueError):
        nextstat.HistFactoryModel.from_workspace("not json")


def test_model_basic_contracts_simple_workspace():
    ws = load_fixture("simple_workspace.json")
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))

    n = model.n_params()
    assert isinstance(n, int)
    assert n > 0

    names = model.parameter_names()
    init = model.suggested_init()
    bounds = model.suggested_bounds()

    assert len(names) == n
    assert len(init) == n
    assert len(bounds) == n

    poi = model.poi_index()
    assert poi is None or (0 <= int(poi) < n)

    nll = model.nll(init)
    assert isinstance(nll, float)
    assert nll == nll  # not NaN


def test_with_observed_main_length_mismatch_raises():
    ws = load_fixture("simple_workspace.json")
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))

    with pytest.raises(ValueError):
        model.with_observed_main([1.0])


def test_fit_wrappers_agree_on_shape_and_fields():
    ws = load_fixture("simple_workspace.json")
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))

    mle = nextstat.MaximumLikelihoodEstimator()
    res_obj = mle.fit(model)
    res_fn = nextstat.fit(model)

    for res in (res_obj, res_fn):
        bestfit = res.bestfit
        unc = res.uncertainties
        assert isinstance(bestfit, list)
        assert isinstance(unc, list)
        assert len(bestfit) == model.n_params()
        assert len(unc) == model.n_params()
        assert isinstance(res.nll, float)
        assert isinstance(res.twice_nll, float)
        assert isinstance(res.success, bool)
        assert isinstance(res.n_evaluations, int)


def test_fit_accepts_overridden_main_data():
    ws = load_fixture("simple_workspace.json")
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))

    # Override main observed data (2 bins for simple fixture).
    res = nextstat.fit(model, data=[53.0, 65.0])
    assert len(res.bestfit) == model.n_params()
    assert res.nll == res.nll


def test_hypotest_contracts():
    ws = load_fixture("simple_workspace.json")
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))

    cls = nextstat.hypotest(1.0, model)
    assert isinstance(cls, float)
    assert 0.0 <= cls <= 1.0

    cls2, tails = nextstat.hypotest(1.0, model, return_tail_probs=True)
    assert isinstance(cls2, float)
    assert isinstance(tails, list)
    assert len(tails) == 2


def test_profile_scan_contracts():
    ws = load_fixture("simple_workspace.json")
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))

    out = nextstat.profile_scan(model, [0.0, 0.5, 1.0])
    assert isinstance(out, dict)
    assert "poi_index" in out
    assert "mu_hat" in out
    assert "nll_hat" in out
    assert "points" in out
    assert isinstance(out["points"], list)
    assert len(out["points"]) == 3
    for p in out["points"]:
        assert isinstance(p, dict)
        assert {"mu", "q_mu", "nll_mu", "converged", "n_iter"} <= set(p.keys())


def test_upper_limit_contract():
    ws = load_fixture("simple_workspace.json")
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))

    ul = nextstat.upper_limit(model, alpha=0.05, lo=0.0, hi=5.0)
    assert isinstance(ul, float)
    assert ul >= 0.0
