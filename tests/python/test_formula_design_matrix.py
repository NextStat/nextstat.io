from __future__ import annotations

import pytest

import nextstat


def test_parse_minimal_formula():
    y, terms, intercept = nextstat.formula.parse_formula("y ~ 1 + x1 + x2")
    assert y == "y"
    assert terms == ["x1", "x2"]
    assert intercept is True


def test_drop_intercept_minus_one():
    y, terms, intercept = nextstat.formula.parse_formula("y ~ x1 + x2 + -1")
    assert y == "y"
    assert terms == ["x1", "x2"]
    assert intercept is False


def test_drop_intercept_zero():
    y, terms, intercept = nextstat.formula.parse_formula("y ~ 0 + x1")
    assert y == "y"
    assert terms == ["x1"]
    assert intercept is False


def test_design_matrices_numeric():
    data = {"y": [1.0, 2.0, 3.0], "x1": [10.0, 11.0, 12.0], "x2": [0.0, 1.0, 0.0]}
    y, x, names = nextstat.formula.design_matrices("y ~ 1 + x1 + x2", data)
    assert y == [1.0, 2.0, 3.0]
    assert names == ["Intercept", "x1", "x2"]
    assert len(x) == 3
    assert x[0] == [1.0, 10.0, 0.0]
    assert x[1] == [1.0, 11.0, 1.0]


def test_design_matrices_categorical_one_hot_deterministic():
    data = {
        "y": [0.0, 1.0, 0.0, 1.0],
        "color": ["red", "blue", "red", "green"],
    }
    y, x, names = nextstat.formula.design_matrices("y ~ 1 + color", data, categorical=["color"])
    assert y == [0.0, 1.0, 0.0, 1.0]
    # levels sorted: blue, green, red; with intercept drop first => green, red
    assert names == ["Intercept", "color[T.green]", "color[T.red]"]
    assert x[0] == [1.0, 0.0, 1.0]  # red
    assert x[1] == [1.0, 0.0, 0.0]  # blue (dropped baseline)
    assert x[3] == [1.0, 1.0, 0.0]  # green


def test_unsupported_rhs_operator_rejected():
    with pytest.raises(nextstat.formula.FormulaError, match="unsupported operator"):
        nextstat.formula.parse_formula("y ~ x1:x2")

