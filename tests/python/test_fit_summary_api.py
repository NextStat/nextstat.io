from __future__ import annotations

import math

import nextstat


def test_wald_summary_shapes_and_bounds():
    s = nextstat.summary.wald_summary([0.0, 1.0], [1.0, 0.5], names=["a", "b"], alpha=0.05)
    assert s["names"] == ["a", "b"]
    assert len(s["coef"]) == 2
    assert len(s["std_err"]) == 2
    assert len(s["stat"]) == 2
    assert len(s["p_value"]) == 2
    assert len(s["ci_low"]) == 2
    assert len(s["ci_high"]) == 2
    assert all(0.0 <= float(p) <= 1.0 for p in s["p_value"])


def test_fit_summary_linear_from_formula_matches_manual_names():
    data = {
        "y": [1.0, 2.0, 3.0, 4.0],
        "x": [0.0, 1.0, 2.0, 3.0],
    }
    fit, names = nextstat.glm.linear.from_formula("y ~ 1 + x", data)
    s = nextstat.summary.fit_summary(fit, names=names)
    assert s["names"] == ["Intercept", "x"]
    assert len(s["coef"]) == 2
    assert all(math.isfinite(float(v)) for v in s["coef"])


def test_fit_summary_logistic_from_formula_smoke():
    data = {
        "y": [0, 0, 1, 1, 1],
        "x": [-2.0, -1.0, 0.0, 1.0, 2.0],
    }
    fit, names = nextstat.glm.logistic.from_formula("y ~ 1 + x", data, l2=1.0)
    s = nextstat.summary.fit_summary(fit, names=names)
    assert s["names"] == ["Intercept", "x"]
    assert all(0.0 <= float(p) <= 1.0 for p in s["p_value"])


def test_summary_to_str_smoke():
    s = nextstat.summary.wald_summary([0.0], [1.0], names=["x"])
    txt = nextstat.summary.summary_to_str(s)
    assert "name" in txt
    assert "coef" in txt
    assert "x" in txt

