from __future__ import annotations

import pytest

import nextstat


def _manual_design(formula: str, data):
    y_name, terms, _ = nextstat.formula.parse_formula(formula)
    cols = nextstat.formula.to_columnar(data, [y_name] + terms)
    y, x_full, names_full = nextstat.formula.design_matrices(formula, cols)
    if names_full and names_full[0] == "Intercept":
        x = [row[1:] for row in x_full]
        include_intercept = True
        colnames = ["Intercept"] + list(names_full[1:])
    else:
        x = x_full
        include_intercept = False
        colnames = list(names_full)
    return y, x, include_intercept, colnames


def test_linear_from_formula_matches_manual_fit_numeric():
    data = {
        "y": [1.0, 2.0, 3.0, 4.0],
        "x1": [10.0, 11.0, 12.0, 13.0],
        "x2": [0.0, 1.0, 0.0, 1.0],
    }
    formula = "y ~ 1 + x1 + x2"

    r1, names1 = nextstat.glm.linear.from_formula(formula, data)
    y, x, include_intercept, names2 = _manual_design(formula, data)
    r2 = nextstat.glm.linear.fit(x, y, include_intercept=include_intercept)

    assert names1 == names2
    assert r1.include_intercept == r2.include_intercept
    assert r1.coef == pytest.approx(r2.coef)
    assert r1.standard_errors == pytest.approx(r2.standard_errors)
    assert r1.sigma2_hat == pytest.approx(r2.sigma2_hat)


def test_linear_from_formula_intercept_only_closed_form():
    data = {"y": [1.0, 2.0, 3.0, 5.0]}
    r, names = nextstat.glm.linear.from_formula("y ~ 1", data)
    assert names == ["Intercept"]
    assert r.coef == pytest.approx([2.75])


def test_logistic_from_formula_matches_manual_fit():
    data = {"y": [0, 0, 0, 1, 1, 1], "x": [-2.0, -1.0, 0.0, 0.0, 1.0, 2.0]}
    formula = "y ~ 1 + x"

    r1, names1 = nextstat.glm.logistic.from_formula(formula, data)
    y, x, include_intercept, names2 = _manual_design(formula, data)
    r2 = nextstat.glm.logistic.fit(x, [int(v) for v in y], include_intercept=include_intercept)

    assert names1 == names2
    assert r1.include_intercept == r2.include_intercept
    assert r1.coef == pytest.approx(r2.coef)
    assert r1.standard_errors == pytest.approx(r2.standard_errors)
    assert r1.nll == pytest.approx(r2.nll)
    assert r1.converged == r2.converged


def test_logistic_from_formula_fallback_l2_applies_on_separation():
    data = {"y": [0, 0, 0, 0, 1, 1, 1, 1], "x": [-3.0, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 3.0]}
    formula = "y ~ 1 + x"

    r, _names = nextstat.glm.logistic.from_formula(
        formula,
        data,
        fallback_l2=10.0,
        fallback_on_separation=True,
    )
    assert r.converged
    assert "fallback_l2_applied" in r.warnings


def test_poisson_from_formula_matches_manual_fit_with_offset_column():
    data = {
        "y": [1, 2, 1, 3, 4],
        "x": [0.0, 1.0, 0.5, -0.5, 1.5],
        "offset": [0.0, 0.1, -0.1, 0.0, 0.2],
    }
    formula = "y ~ 1 + x"

    r1, names1 = nextstat.glm.poisson.from_formula(formula, data, offset="offset")
    y, x, include_intercept, names2 = _manual_design(formula, data)
    r2 = nextstat.glm.poisson.fit(x, [int(v) for v in y], include_intercept=include_intercept, offset=data["offset"])

    assert names1 == names2
    assert r1.include_intercept == r2.include_intercept
    assert r1.coef == pytest.approx(r2.coef)
    assert r1.standard_errors == pytest.approx(r2.standard_errors)
    assert r1.nll == pytest.approx(r2.nll)
    assert r1.converged == r2.converged


def test_negbin_from_formula_matches_manual_fit():
    data = {"y": [0, 1, 2, 1, 3, 4], "x": [0.0, 0.5, 1.0, 0.0, 1.5, 2.0]}
    formula = "y ~ 1 + x"

    r1, names1 = nextstat.glm.negbin.from_formula(formula, data)
    y, x, include_intercept, names2 = _manual_design(formula, data)
    r2 = nextstat.glm.negbin.fit(x, [int(v) for v in y], include_intercept=include_intercept)

    assert names1 == names2
    assert r1.include_intercept == r2.include_intercept
    assert r1.coef == pytest.approx(r2.coef)
    assert r1.standard_errors == pytest.approx(r2.standard_errors)
    assert r1.nll == pytest.approx(r2.nll)
    assert r1.converged == r2.converged


def test_from_formula_accepts_list_of_dict_rows():
    rows = [
        {"y": 1.0, "x1": 0.0},
        {"y": 2.0, "x1": 1.0},
        {"y": 3.0, "x1": 2.0},
    ]
    r, names = nextstat.glm.linear.from_formula("y ~ 1 + x1", rows)
    assert names == ["Intercept", "x1"]
    assert len(r.coef) == 2


def test_from_formula_accepts_pandas_dataframe_when_available():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})
    r, names = nextstat.glm.linear.from_formula("y ~ 1 + x", df)
    assert names == ["Intercept", "x"]
    assert len(r.coef) == 2


def test_hier_logistic_random_intercept_from_formula_builds_and_fits_smoke():
    data = {
        "y": [0, 1, 0, 1, 0, 1],
        "x": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "g": ["A", "A", "B", "B", "A", "B"],
    }
    m = nextstat.hier.logistic_random_intercept_from_formula("y ~ 1 + x", data, group="g")
    fit = nextstat.fit(m)
    assert isinstance(fit.bestfit, list)
    assert len(fit.bestfit) == m.n_params()


def test_hier_linear_random_intercept_from_formula_builds_and_fits_smoke():
    data = {
        "y": [0.2, 1.1, 2.0, 3.2],
        "x": [0.0, 1.0, 2.0, 3.0],
        "g": ["A", "A", "B", "B"],
    }
    m = nextstat.hier.linear_random_intercept_from_formula("y ~ 1 + x", data, group="g")
    fit = nextstat.fit(m)
    assert isinstance(fit.bestfit, list)
    assert len(fit.bestfit) == m.n_params()
