from __future__ import annotations

import json
from pathlib import Path


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "regression"


def _load(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


def _assert_vec_close(a, b, *, rtol: float, atol: float = 0.0):
    assert len(a) == len(b)
    for i, (ai, bi) in enumerate(zip(a, b)):
        ai = float(ai)
        bi = float(bi)
        diff = abs(ai - bi)
        scale = max(abs(ai), abs(bi), 1.0)
        assert diff <= atol + rtol * scale, f"idx={i}: {ai} vs {bi} (diff={diff})"


def test_glm_linear_fit_matches_fixture_beta_and_se():
    import nextstat

    fx = _load("ols_small.json")
    assert fx["kind"] == "ols"
    r = nextstat.glm.linear.fit(fx["x"], fx["y"], include_intercept=True)
    _assert_vec_close(r.coef, fx["beta_hat"], rtol=1e-10, atol=1e-12)
    _assert_vec_close(r.standard_errors, fx["se_hat"], rtol=1e-10, atol=1e-12)

    # Predict shape smoke.
    yhat = r.predict(fx["x"][:3])
    assert len(yhat) == 3


def test_glm_logistic_fit_matches_fixture_beta_and_se():
    import nextstat

    fx = _load("logistic_small.json")
    assert fx["kind"] == "logistic"
    y = [1 if float(v) >= 0.5 else 0 for v in fx["y"]]
    r = nextstat.glm.logistic.fit(fx["x"], y, include_intercept=True)

    _assert_vec_close(r.coef, fx["beta_hat"], rtol=2e-3, atol=1e-6)
    _assert_vec_close(r.standard_errors, fx["se_hat"], rtol=2e-2, atol=1e-6)

    p = r.predict_proba(fx["x"][:5])
    assert len(p) == 5
    assert all(0.0 <= pi <= 1.0 for pi in p)


def test_glm_poisson_fit_matches_fixture_beta_and_se():
    import nextstat

    fx = _load("poisson_small.json")
    assert fx["kind"] == "poisson"
    y = [int(round(float(v))) for v in fx["y"]]
    r = nextstat.glm.poisson.fit(fx["x"], y, include_intercept=True)

    _assert_vec_close(r.coef, fx["beta_hat"], rtol=2e-3, atol=1e-6)
    _assert_vec_close(r.standard_errors, fx["se_hat"], rtol=2e-2, atol=1e-6)

    rate = r.predict_rate(fx["x"][:5])
    assert len(rate) == 5
    assert all(ri > 0.0 for ri in rate)


def test_glm_negbin_fit_matches_fixture_beta_and_se():
    import nextstat

    fx = _load("negbin_small.json")
    assert fx["kind"] == "negbin"
    y = [int(round(float(v))) for v in fx["y"]]
    r = nextstat.glm.negbin.fit(fx["x"], y, include_intercept=True)

    _assert_vec_close(r.coef, fx["beta_hat"], rtol=3e-3, atol=1e-6)
    _assert_vec_close(r.standard_errors, fx["se_hat"], rtol=5e-2, atol=1e-6)
    assert abs(float(r.log_alpha) - float(fx["log_alpha_hat"])) < 5e-3

    mu = r.predict_mean(fx["x"][:5])
    assert len(mu) == 5
    assert all(mi > 0.0 for mi in mu)
