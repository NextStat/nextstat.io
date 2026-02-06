"""Model zoo: broader deterministic parity checks vs pyhf reference.

These tests intentionally generate multiple synthetic workspaces (beyond static
fixtures) and compare NextStat NLL to pyhf NLL at:
  1) suggested init
  2) a few random parameter points within suggested bounds

If any mismatch appears, it is almost always a modifier-combination semantics
issue, bounds/fixed handling, or constraint term mismatch.
"""

from __future__ import annotations

import json
import random
from typing import Any

import pytest

import nextstat

from _tolerances import EXPECTED_DATA_ATOL, TWICE_NLL_ATOL, TWICE_NLL_RTOL

from _pyhf_model_zoo import (
    make_workspace_histo_normsys_staterror,
    make_workspace_multichannel,
    make_workspace_shapefactor_control_region,
)

pyhf = pytest.importorskip("pyhf")


def _pyhf_model_and_data(workspace: dict[str, Any], measurement_name: str):
    ws = pyhf.Workspace(workspace)
    model = ws.model(
        measurement_name=measurement_name,
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
    )
    data = ws.data(model)
    return model, data


def _pyhf_twice_nll(model, data, params) -> float:
    return float(pyhf.infer.mle.twice_nll(params, data, model).item())


def _map_params_by_name(src_names, src_params, dst_names, dst_init):
    dst_index = {name: i for i, name in enumerate(dst_names)}
    out = list(dst_init)
    for name, value in zip(src_names, src_params):
        out[dst_index[name]] = float(value)
    return out


def _sample_params(rng: random.Random, init: list[float], bounds: list[tuple[float, float]]):
    out: list[float] = []
    for x0, (lo, hi) in zip(init, bounds):
        lo_f = float(lo)
        hi_f = float(hi)
        if not (lo_f < hi_f):
            out.append(float(x0))
            continue
        # Sample in a tight-ish region around init to avoid extreme tails.
        span = hi_f - lo_f
        center = min(max(float(x0), lo_f), hi_f)
        half = 0.25 * span
        a = max(lo_f, center - half)
        b = min(hi_f, center + half)
        if not (a < b):
            a, b = lo_f, hi_f
        out.append(rng.uniform(a, b))
    return out


def _assert_nll_parity(workspace: dict[str, Any], measurement_name: str, ns_timing, *, seed: int, n_random: int):
    with ns_timing.time("pyhf"):
        pyhf_model, pyhf_data = _pyhf_model_and_data(workspace, measurement_name)
        pyhf_init = list(map(float, pyhf_model.config.suggested_init()))
        pyhf_bounds = [(float(a), float(b)) for a, b in pyhf_model.config.suggested_bounds()]

    with ns_timing.time("nextstat"):
        ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
        ns_names = ns_model.parameter_names()
        ns_init = ns_model.suggested_init()

    assert set(ns_names) == set(pyhf_model.config.par_names)

    def expected_data_full_ns(pyhf_params: list[float]) -> list[float]:
        ns_params = _map_params_by_name(pyhf_model.config.par_names, pyhf_params, ns_names, ns_init)
        with ns_timing.time("nextstat"):
            return [float(x) for x in ns_model.expected_data(ns_params)]

    def expected_data_main_ns(pyhf_params: list[float]) -> list[float]:
        ns_params = _map_params_by_name(pyhf_model.config.par_names, pyhf_params, ns_names, ns_init)
        with ns_timing.time("nextstat"):
            return [float(x) for x in ns_model.expected_data(ns_params, include_auxdata=False)]

    def expected_data_full_pyhf(pyhf_params: list[float]) -> list[float]:
        with ns_timing.time("pyhf"):
            return [float(x) for x in pyhf_model.expected_data(pyhf_params)]

    def expected_data_main_pyhf(pyhf_params: list[float]) -> list[float]:
        with ns_timing.time("pyhf"):
            return [float(x) for x in pyhf_model.expected_data(pyhf_params, include_auxdata=False)]

    def twice_nll_ns(pyhf_params: list[float]) -> float:
        ns_params = _map_params_by_name(pyhf_model.config.par_names, pyhf_params, ns_names, ns_init)
        with ns_timing.time("nextstat"):
            return 2.0 * float(ns_model.nll(ns_params))

    def twice_nll_pyhf(pyhf_params: list[float]) -> float:
        with ns_timing.time("pyhf"):
            return _pyhf_twice_nll(pyhf_model, pyhf_data, pyhf_params)

    # suggested init
    assert twice_nll_ns(pyhf_init) == pytest.approx(
        twice_nll_pyhf(pyhf_init),
        rel=TWICE_NLL_RTOL,
        abs=TWICE_NLL_ATOL,
    )
    assert expected_data_full_ns(pyhf_init) == pytest.approx(
        expected_data_full_pyhf(pyhf_init),
        rel=0.0,
        abs=EXPECTED_DATA_ATOL,
    )
    assert expected_data_main_ns(pyhf_init) == pytest.approx(
        expected_data_main_pyhf(pyhf_init),
        rel=0.0,
        abs=EXPECTED_DATA_ATOL,
    )

    # random points
    rng = random.Random(seed)
    for _ in range(n_random):
        p = _sample_params(rng, pyhf_init, pyhf_bounds)
        assert twice_nll_ns(p) == pytest.approx(
            twice_nll_pyhf(p),
            rel=TWICE_NLL_RTOL,
            abs=TWICE_NLL_ATOL,
        )
        assert expected_data_full_ns(p) == pytest.approx(
            expected_data_full_pyhf(p),
            rel=0.0,
            abs=EXPECTED_DATA_ATOL,
        )
        assert expected_data_main_ns(p) == pytest.approx(
            expected_data_main_pyhf(p),
            rel=0.0,
            abs=EXPECTED_DATA_ATOL,
        )

    # POI variations (if present)
    poi_idx = pyhf_model.config.poi_index
    if poi_idx is not None:
        for mu in [0.0, 0.5, 2.0]:
            p = list(pyhf_init)
            p[poi_idx] = mu
            assert twice_nll_ns(p) == pytest.approx(
                twice_nll_pyhf(p),
                rel=TWICE_NLL_RTOL,
                abs=TWICE_NLL_ATOL,
            )
            assert expected_data_full_ns(p) == pytest.approx(
                expected_data_full_pyhf(p),
                rel=0.0,
                abs=EXPECTED_DATA_ATOL,
            )
            assert expected_data_main_ns(p) == pytest.approx(
                expected_data_main_pyhf(p),
                rel=0.0,
                abs=EXPECTED_DATA_ATOL,
            )


@pytest.mark.parametrize(
    ("workspace", "measurement", "seed", "n_random"),
    [
        (make_workspace_multichannel(3), "m", 0, 6),
        (make_workspace_histo_normsys_staterror(10), "m", 1, 6),
        (make_workspace_shapefactor_control_region(4), "m", 2, 6),
    ],
)
def test_model_zoo_nll_parity(workspace, measurement, seed, n_random, ns_timing):
    _assert_nll_parity(workspace, measurement, ns_timing, seed=seed, n_random=n_random)
