from __future__ import annotations

import json
from pathlib import Path


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "regression"


def _load(name: str) -> dict:
    return json.loads((FIXTURES_DIR / f"{name}.json").read_text())


def _assert_close_list(got, want, *, atol: float):
    assert len(got) == len(want)
    for i, (a, b) in enumerate(zip(got, want)):
        da = abs(float(a) - float(b))
        assert da <= atol, f"idx={i}: got={a}, want={b}, |diff|={da}"


def test_glm_linear_fit_matches_ols_fixture():
    import nextstat

    fx = _load("ols_small")
    r = nextstat.glm.linear.fit(fx["x"], fx["y"], include_intercept=fx["include_intercept"])
    _assert_close_list(r.coef, fx["beta_hat"], atol=1e-8)

    yhat = r.predict(fx["x"])
    assert len(yhat) == fx["n"]


def test_glm_logistic_fit_matches_fixture_beta_hat():
    import nextstat

    fx = _load("logistic_small")
    r = nextstat.glm.logistic.fit(fx["x"], fx["y"], include_intercept=fx["include_intercept"])
    _assert_close_list(r.coef, fx["beta_hat"], atol=1e-4)

    p = r.predict_proba(fx["x"])
    assert len(p) == fx["n"]
    assert all(0.0 <= float(v) <= 1.0 for v in p)


def test_glm_poisson_fit_matches_fixture_beta_hat():
    import nextstat

    fx = _load("poisson_small")
    r = nextstat.glm.poisson.fit(fx["x"], fx["y"], include_intercept=fx["include_intercept"])
    _assert_close_list(r.coef, fx["beta_hat"], atol=1e-4)

    mu = r.predict_mean(fx["x"])
    assert len(mu) == fx["n"]
    assert all(float(v) > 0.0 for v in mu)

