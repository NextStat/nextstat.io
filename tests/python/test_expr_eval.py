import numpy as np
import pytest


def test_expr_eval_scalar_arith_and_vars():
    from nextstat.analysis.expr_eval import eval_expr

    env = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([10.0, 20.0, 30.0])}
    out = eval_expr("a + 2*b", env)
    assert out.dtype == np.float64
    assert np.allclose(out, np.array([21.0, 42.0, 63.0]))


def test_expr_eval_functions():
    from nextstat.analysis.expr_eval import eval_expr

    env = {"x": np.array([9.0, 16.0])}
    assert np.allclose(eval_expr("sqrt(x)", env), np.array([3.0, 4.0]))
    assert np.allclose(eval_expr("pow(x, 0.5)", env), np.array([3.0, 4.0]))
    assert np.allclose(eval_expr("max(x, 10)", env), np.array([10.0, 16.0]))

    # ROOT/TMath spellings and additional math funcs (match Rust engine subset).
    assert np.allclose(eval_expr("fabs(-x)", env), np.array([9.0, 16.0]))
    assert np.allclose(eval_expr("TMath::Abs(-x)", env), np.array([9.0, 16.0]))
    assert np.allclose(eval_expr("TMath::Power(2, 3)", env), np.array([8.0, 8.0]))
    assert np.allclose(eval_expr("log10(x)", env), np.array([np.log10(9.0), np.log10(16.0)]))
    assert np.allclose(eval_expr("TMath::Log10(x)", env), np.array([np.log10(9.0), np.log10(16.0)]))
    assert np.allclose(eval_expr("sin(0) + cos(0)", env), np.array([1.0, 1.0]))
    assert np.allclose(eval_expr("atan2(1, 0)", env), np.array([np.pi / 2, np.pi / 2]))


def test_expr_eval_boolean_and_comparisons():
    from nextstat.analysis.expr_eval import eval_expr

    env = {"njet": np.array([3.0, 4.0, 5.0]), "pt": np.array([10.0, 30.0, 20.0])}
    out = eval_expr("njet >= 4 && pt > 25", env)
    assert np.allclose(out, np.array([0.0, 1.0, 0.0]))

    # ROOT truthiness: any non-zero is true (including negatives and NaN).
    env2 = {"x": np.array([0.0, -1.0, np.nan])}
    out2 = eval_expr("x ? 10 : 20", env2)
    assert np.allclose(out2, np.array([20.0, 10.0, 10.0]))
    out3 = eval_expr("!x", env2)
    assert np.allclose(out3, np.array([1.0, 0.0, 0.0]))


def test_expr_eval_ternary_is_lazy_for_indexing():
    from nextstat.analysis.expr_eval import eval_expr

    env = {
        "x": np.array([1.0, -1.0]),
        # Per-event arrays (jagged); second event has no element 0.
        "jets_pt": [[10.0], []],
    }
    out = eval_expr("x > 0 ? jets_pt[0] : 0", env)
    assert np.allclose(out, np.array([10.0, 0.0]))


def test_expr_eval_index_out_of_bounds_has_span():
    from nextstat.analysis.expr_eval import eval_expr_strict

    env = {"x": np.array([1.0]), "jets_pt": [[]]}
    with pytest.raises(ValueError) as e:
        eval_expr_strict("jets_pt[0]", env)
    msg = str(e.value)
    assert "line 1, col 8" in msg
    assert "out of bounds" in msg


def test_expr_eval_array_valued_var_requires_indexing():
    from nextstat.analysis.expr_eval import eval_expr

    env = {"jets_pt": [[1.0], [2.0]]}
    with pytest.raises(ValueError) as e:
        eval_expr("jets_pt + 1", env)
    assert "requires indexing" in str(e.value)


def test_expr_eval_dynamic_indexing():
    from nextstat.analysis.expr_eval import eval_expr

    env = {
        "jets_pt": [[10.0, 11.0], [], [30.0]],
        "idx": np.array([1.0, 0.0, 0.0]),
    }
    out = eval_expr("jets_pt[idx]", env)
    assert np.allclose(out, np.array([11.0, 0.0, 30.0]))
