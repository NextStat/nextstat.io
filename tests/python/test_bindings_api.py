"""Contract tests for Python bindings (API shape + error behavior).

These tests are intentionally independent from pyhf parity tests.
They validate:
- error types/messages for invalid inputs
- basic method contracts (lengths, return types)
- `fit()` wrapper and `MaximumLikelihoodEstimator.fit()` agree on shape
"""

import json
import math
from pathlib import Path

import pytest

import nextstat


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())

def _is_finite(x: float) -> bool:
    return math.isfinite(float(x))


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


def test_model_rejects_wrong_parameter_length_without_crashing():
    ws = load_fixture("simple_workspace.json")
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))

    # Too short
    with pytest.raises(ValueError):
        model.nll([1.0])
    with pytest.raises(ValueError):
        model.grad_nll([1.0])
    with pytest.raises(ValueError):
        model.expected_data([1.0])

    # Too long
    n = model.n_params()
    with pytest.raises(ValueError):
        model.nll([1.0] * (n + 1))
    with pytest.raises(ValueError):
        model.grad_nll([1.0] * (n + 1))
    with pytest.raises(ValueError):
        model.expected_data([1.0] * (n + 1))


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
        assert isinstance(res.n_iter, int)
        assert isinstance(res.n_fev, int)
        assert isinstance(res.n_gev, int)


def test_fit_accepts_overridden_main_data():
    ws = load_fixture("simple_workspace.json")
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))

    # Override main observed data (2 bins for simple fixture).
    res = nextstat.fit(model, data=[53.0, 65.0])
    assert len(res.bestfit) == model.n_params()
    assert res.nll == res.nll


def test_non_histfactory_models_reject_data_kwarg():
    gm = nextstat.GaussianMeanModel([0.0, 1.0, 2.0], 1.0)
    with pytest.raises(ValueError, match="data= is only supported for HistFactoryModel"):
        nextstat.fit(gm, data=[1.0])


def test_gaussian_mean_model_grad_contract():
    gm = nextstat.GaussianMeanModel([0.0, 1.0, 2.0], 1.0)
    init = gm.suggested_init()
    g = gm.grad_nll(init)
    assert isinstance(g, list)
    assert len(g) == gm.n_params()
    assert all(_is_finite(v) for v in g)


def test_linear_regression_model_grad_contract_smoke():
    x = [[0.0], [1.0], [2.0]]
    y = [1.0, 3.0, 5.0]
    m = nextstat.LinearRegressionModel(x, y, include_intercept=True)
    init = m.suggested_init()
    g = m.grad_nll(init)
    assert isinstance(g, list)
    assert len(g) == m.n_params()
    assert all(_is_finite(v) for v in g)


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


def test_fit_batch_contract_list_of_models():
    ws = load_fixture("simple_workspace.json")
    m0 = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))
    m1 = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))

    mle = nextstat.MaximumLikelihoodEstimator()
    results = mle.fit_batch([m0, m1])
    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert isinstance(r.bestfit, list)
        assert isinstance(r.uncertainties, list)
        assert isinstance(r.nll, float)
        assert isinstance(r.success, bool)


def test_fit_batch_contract_model_and_datasets():
    ws = load_fixture("simple_workspace.json")
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))

    mle = nextstat.MaximumLikelihoodEstimator()
    results = mle.fit_batch(model, datasets=[[53.0, 65.0], [54.0, 66.0]])
    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert isinstance(r.bestfit, list)
        assert isinstance(r.uncertainties, list)
        assert isinstance(r.nll, float)
        assert isinstance(r.success, bool)


def test_ranking_contract():
    ws = load_fixture("simple_workspace.json")
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))

    entries = nextstat.ranking(model)
    assert isinstance(entries, list)
    assert entries, "ranking should be non-empty for simple fixture"
    e0 = entries[0]
    assert isinstance(e0, dict)
    assert {"name", "delta_mu_up", "delta_mu_down", "pull", "constraint"} <= set(e0.keys())
