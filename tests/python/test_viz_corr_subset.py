from __future__ import annotations

from typing import Any, Mapping

import pytest

import nextstat


def _artifact(names: list[str], corr: list[list[float]]) -> Mapping[str, Any]:
    return {
        "schema_version": "trex_report_corr_v0",
        "meta": {
            "tool": "nextstat",
            "tool_version": "0.0.0-test",
            "created_unix_ms": 0,
            "parity_mode": {"threads": 1, "stable_ordering": True},
        },
        "parameter_names": names,
        "corr": corr,
    }


def test_corr_subset_top_n_by_max_abs_corr_is_deterministic():
    names = ["a", "b", "c", "d"]
    corr = [
        [1.0, 0.1, -0.9, 0.0],  # a max abs=0.9
        [0.1, 1.0, 0.2, 0.8],  # b max abs=0.8
        [-0.9, 0.2, 1.0, -0.3],  # c max abs=0.9
        [0.0, 0.8, -0.3, 1.0],  # d max abs=0.8
    ]
    art = _artifact(names, corr)

    sub = nextstat.viz.corr_subset(art, top_n=2, order="max_abs_corr")
    assert list(sub["parameter_names"]) == ["a", "c"]  # tie-break by name
    assert len(sub["corr"]) == 2
    assert len(sub["corr"][0]) == 2


def test_corr_subset_include_exclude_regex():
    names = ["mu", "bkg_norm", "staterror_SR[0]", "staterror_SR[1]", "lumi"]
    corr = [[1.0 if i == j else 0.0 for j in range(len(names))] for i in range(len(names))]
    art = _artifact(names, corr)

    sub = nextstat.viz.corr_subset(art, include=r"^staterror_SR\[" , exclude=r"\[1\]$")
    assert list(sub["parameter_names"]) == ["staterror_SR[0]"]
    assert sub["corr"] == [[1.0]]


def test_corr_subset_group_base_orders_by_base_then_name():
    names = ["b[1]", "a[0]", "a[1]", "b[0]"]
    corr = [[1.0 if i == j else 0.0 for j in range(len(names))] for i in range(len(names))]
    art = _artifact(names, corr)

    sub = nextstat.viz.corr_subset(art, order="group_base")
    assert list(sub["parameter_names"]) == ["a[0]", "a[1]", "b[0]", "b[1]"]


def test_corr_subset_validates_square_matrix():
    art = _artifact(["a", "b"], [[1.0, 0.0]])
    with pytest.raises(ValueError):
        nextstat.viz.corr_subset(art)
