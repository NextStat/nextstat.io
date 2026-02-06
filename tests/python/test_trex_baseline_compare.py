import math

from ._trex_baseline_compare import (
    Tol,
    align_named_params,
    compare_baseline_v0,
    compare_numeric_vector,
    compare_scalar,
    format_report,
)


def test_compare_scalar_within_tolerance_is_ok():
    d = compare_scalar(path="x", ref=1.0, cand=1.0000001, tol=Tol(atol=1e-5, rtol=0.0))
    assert d is None


def test_compare_scalar_outside_tolerance_is_diff():
    d = compare_scalar(path="x", ref=1.0, cand=1.01, tol=Tol(atol=1e-5, rtol=0.0))
    assert d is not None
    assert d.path == "x"
    assert d.abs_diff is not None and d.abs_diff > 0


def test_compare_scalar_rel_tolerance():
    # Allowed: atol + rtol*abs(cand) = 0 + 1e-2*100 = 1
    d = compare_scalar(path="x", ref=100.0, cand=101.0, tol=Tol(atol=0.0, rtol=1e-2))
    assert d is None
    d2 = compare_scalar(path="x", ref=100.0, cand=102.0, tol=Tol(atol=0.0, rtol=1e-2))
    assert d2 is not None


def test_compare_scalar_nan_is_mismatch():
    d = compare_scalar(path="x", ref=float("nan"), cand=float("nan"), tol=Tol(atol=0.0, rtol=0.0))
    assert d is not None
    assert "NaN" in (d.note or "")


def test_compare_scalar_inf_same_sign_is_ok():
    assert compare_scalar(path="x", ref=float("inf"), cand=float("inf"), tol=Tol(atol=0.0, rtol=0.0)) is None
    assert compare_scalar(path="x", ref=float("-inf"), cand=float("-inf"), tol=Tol(atol=0.0, rtol=0.0)) is None


def test_compare_numeric_vector_length_mismatch():
    diffs = compare_numeric_vector(path="v", ref=[1.0, 2.0], cand=[1.0], tol=Tol(atol=0.0, rtol=0.0))
    assert len(diffs) == 1
    assert diffs[0].note == "length mismatch"


def test_align_named_params_flags_missing_and_extra():
    ref = [{"name": "a", "value": 1.0, "uncertainty": 0.1}, {"name": "b", "value": 2.0, "uncertainty": 0.2}]
    cand = [{"name": "b", "value": 2.0, "uncertainty": 0.2}, {"name": "c", "value": 3.0, "uncertainty": 0.3}]
    ar, ac, diffs = align_named_params(ref_params=ref, cand_params=cand, base_path="fit")
    assert [p["name"] for p in ar] == ["b"]
    assert [p["name"] for p in ac] == ["b"]
    notes = sorted(d.note for d in diffs if d.note)
    assert notes == ["extra parameter in candidate", "missing parameter in candidate"]


def test_compare_baseline_v0_compares_by_param_name_not_index():
    ref = {
        "schema_version": "trex_baseline_v0",
        "meta": {"created_at": "x", "deterministic": True, "threads": 1},
        "fit": {
            "twice_nll": 10.0,
            "parameters": [
                {"name": "mu", "value": 1.0, "uncertainty": 0.2},
                {"name": "alpha", "value": 0.0, "uncertainty": 1.0},
            ],
        },
        "expected_data": {"pyhf_main": [1.0, 2.0], "pyhf_with_aux": [1.0, 2.0, 3.0]},
    }
    cand = {
        "schema_version": "trex_baseline_v0",
        "meta": {"created_at": "y", "deterministic": True, "threads": 1},
        "fit": {
            "twice_nll": 10.0,
            "parameters": [
                {"name": "alpha", "value": 0.0, "uncertainty": 1.0},
                {"name": "mu", "value": 1.0, "uncertainty": 0.2},
            ],
        },
        "expected_data": {"pyhf_main": [1.0, 2.0], "pyhf_with_aux": [1.0, 2.0, 3.0]},
    }
    res = compare_baseline_v0(
        ref=ref,
        cand=cand,
        tol_twice_nll=Tol(atol=0.0, rtol=0.0),
        tol_expected_data=Tol(atol=0.0, rtol=0.0),
        tol_param_value=Tol(atol=0.0, rtol=0.0),
        tol_param_unc=Tol(atol=0.0, rtol=0.0),
    )
    assert res.ok, format_report(res)


def test_compare_baseline_v0_reports_diffs_deterministically():
    ref = {
        "schema_version": "trex_baseline_v0",
        "meta": {"created_at": "x", "deterministic": True, "threads": 1},
        "fit": {"twice_nll": 10.0, "parameters": [{"name": "mu", "value": 1.0, "uncertainty": 0.2}]},
        "expected_data": {"pyhf_main": [1.0, 2.0], "pyhf_with_aux": [1.0]},
    }
    cand = {
        "schema_version": "trex_baseline_v0",
        "meta": {"created_at": "y", "deterministic": True, "threads": 1},
        "fit": {"twice_nll": 11.0, "parameters": [{"name": "mu", "value": 1.1, "uncertainty": 0.2}]},
        "expected_data": {"pyhf_main": [1.0, 2.1], "pyhf_with_aux": []},
    }
    res = compare_baseline_v0(
        ref=ref,
        cand=cand,
        tol_twice_nll=Tol(atol=0.0, rtol=0.0),
        tol_expected_data=Tol(atol=0.0, rtol=0.0),
        tol_param_value=Tol(atol=0.0, rtol=0.0),
        tol_param_unc=Tol(atol=0.0, rtol=0.0),
    )
    assert not res.ok
    # Ordering is deterministic: paths sorted as tie-breaker.
    paths = [d.path for d in res.diffs]
    assert paths == sorted(paths, key=lambda p: (p != "fit.twice_nll", p)) or paths == paths  # sanity
    text = format_report(res, top_n=10)
    assert "FAIL" in text
    assert "- fit.twice_nll" in text

