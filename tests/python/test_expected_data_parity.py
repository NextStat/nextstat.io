"""Parity check for expected_data vs pyhf.

This exercises the Python binding `HistFactoryModel.expected_data`.
"""

from __future__ import annotations

import json
import random
from array import array
from pathlib import Path

import pytest

import nextstat

from _tolerances import EXPECTED_DATA_ATOL


pyhf = pytest.importorskip("pyhf")

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
NLL_ATOL = 5e-9
NLL_RTOL = 1e-6


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


def pyhf_model(workspace: dict, measurement_name: str):
    ws = pyhf.Workspace(workspace)
    return ws.model(
        measurement_name=measurement_name,
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
    )

def pyhf_model_and_data(workspace: dict, measurement_name: str):
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


def pyhf_nll(model, data, params) -> float:
    return float(pyhf.infer.mle.twice_nll(params, data, model).item()) / 2.0


def map_params_by_name(src_names, src_params, dst_names, dst_init):
    dst_index = {name: i for i, name in enumerate(dst_names)}
    out = list(dst_init)
    for name, value in zip(src_names, src_params):
        out[dst_index[name]] = float(value)
    return out


def sample_params(rng: random.Random, init: list[float], bounds: list[tuple[float, float]]):
    out: list[float] = []
    for x0, (lo, hi) in zip(init, bounds):
        lo_f = float(lo)
        hi_f = float(hi)
        if not (lo_f < hi_f):
            out.append(float(x0))
            continue
        span = hi_f - lo_f
        center = min(max(float(x0), lo_f), hi_f)
        half = 0.25 * span
        a = max(lo_f, center - half)
        b = min(hi_f, center + half)
        if not (a < b):
            a, b = lo_f, hi_f
        out.append(rng.uniform(a, b))
    return out


@pytest.mark.parametrize(
    ("fixture", "measurement"),
    [
        ("simple_workspace.json", "GaussExample"),
        ("complex_workspace.json", "measurement"),
        ("histfactory/workspace.json", "NominalMeasurement"),
    ],
)
def test_expected_data_matches_pyhf_at_suggested_init(fixture: str, measurement: str, ns_timing):
    workspace = load_fixture(fixture)
    with ns_timing.time("pyhf"):
        model = pyhf_model(workspace, measurement)
        pyhf_init = list(map(float, model.config.suggested_init()))

    with ns_timing.time("nextstat"):
        ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
        ns_params = map_params_by_name(
            model.config.par_names,
            pyhf_init,
            ns_model.parameter_names(),
            ns_model.suggested_init(),
        )

    with ns_timing.time("pyhf"):
        exp_pyhf = [float(x) for x in model.expected_data(pyhf_init)]
    with ns_timing.time("nextstat"):
        exp_ns = [float(x) for x in ns_model.expected_data(ns_params)]
    assert exp_ns == pytest.approx(exp_pyhf, rel=0.0, abs=EXPECTED_DATA_ATOL)

    with ns_timing.time("pyhf"):
        exp_pyhf_main = [float(x) for x in model.expected_data(pyhf_init, include_auxdata=False)]
    with ns_timing.time("nextstat"):
        exp_ns_main = [float(x) for x in ns_model.expected_data(ns_params, include_auxdata=False)]
    assert exp_ns_main == pytest.approx(exp_pyhf_main, rel=0.0, abs=EXPECTED_DATA_ATOL)

    # Buffer protocol should be supported (perf-critical for large parameter vectors).
    ns_params_buf = array("d", ns_params)
    with ns_timing.time("nextstat"):
        exp_ns_buf = [float(x) for x in ns_model.expected_data(ns_params_buf)]
    assert exp_ns_buf == pytest.approx(exp_pyhf, rel=0.0, abs=EXPECTED_DATA_ATOL)
    with ns_timing.time("nextstat"):
        exp_ns_main_buf = [float(x) for x in ns_model.expected_data(ns_params_buf, include_auxdata=False)]
    assert exp_ns_main_buf == pytest.approx(exp_pyhf_main, rel=0.0, abs=EXPECTED_DATA_ATOL)

@pytest.mark.parametrize(
    ("fixture", "measurement"),
    [
        ("simple_workspace.json", "GaussExample"),
        ("complex_workspace.json", "measurement"),
        ("histfactory/workspace.json", "NominalMeasurement"),
    ],
)
def test_nll_accepts_buffer_and_matches_pyhf_at_suggested_init(fixture: str, measurement: str, ns_timing):
    workspace = load_fixture(fixture)
    with ns_timing.time("pyhf"):
        model, data = pyhf_model_and_data(workspace, measurement)
        pyhf_init = list(map(float, model.config.suggested_init()))

    with ns_timing.time("nextstat"):
        ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
        ns_params_list = map_params_by_name(
            model.config.par_names,
            pyhf_init,
            ns_model.parameter_names(),
            ns_model.suggested_init(),
        )
    ns_params_buf = array("d", ns_params_list)

    with ns_timing.time("pyhf"):
        nll_pyhf = pyhf_nll(model, data, pyhf_init)
    with ns_timing.time("nextstat"):
        nll_ns = float(ns_model.nll(ns_params_list))
        nll_ns_buf = float(ns_model.nll(ns_params_buf))

    scale = max(abs(float(nll_pyhf)), 1.0)
    allowed = max(NLL_ATOL, NLL_RTOL * scale)

    assert abs(nll_ns - nll_pyhf) <= allowed
    assert abs(nll_ns_buf - nll_pyhf) <= allowed
    assert abs(nll_ns_buf - nll_ns) <= allowed


def test_grad_nll_accepts_buffer_matches_list_simple_fixture():
    # Gradient parity vs pyhf isn't part of Phase 1 contract yet.
    # This test only locks in that the buffer-protocol fast path is supported and consistent.
    workspace = load_fixture("simple_workspace.json")
    ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
    params = ns_model.suggested_init()
    params_buf = array("d", params)

    g_list = list(map(float, ns_model.grad_nll(params)))
    g_buf = list(map(float, ns_model.grad_nll(params_buf)))
    assert g_buf == pytest.approx(g_list, rel=0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("fixture", "measurement", "seed"),
    [
        ("simple_workspace.json", "GaussExample", 0),
        ("complex_workspace.json", "measurement", 1),
    ],
)
def test_expected_data_matches_pyhf_at_random_points(fixture: str, measurement: str, seed: int, ns_timing):
    workspace = load_fixture(fixture)
    with ns_timing.time("pyhf"):
        model = pyhf_model(workspace, measurement)
        pyhf_init = list(map(float, model.config.suggested_init()))
        pyhf_bounds = [(float(a), float(b)) for a, b in model.config.suggested_bounds()]

    with ns_timing.time("nextstat"):
        ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
        ns_init = ns_model.suggested_init()
        ns_names = ns_model.parameter_names()

    rng = random.Random(seed)
    for _ in range(5):
        pyhf_params = sample_params(rng, pyhf_init, pyhf_bounds)
        ns_params = map_params_by_name(
            model.config.par_names,
            pyhf_params,
            ns_names,
            ns_init,
        )

        with ns_timing.time("pyhf"):
            exp_pyhf = [float(x) for x in model.expected_data(pyhf_params)]
        with ns_timing.time("nextstat"):
            exp_ns = [float(x) for x in ns_model.expected_data(ns_params)]
        assert exp_ns == pytest.approx(exp_pyhf, rel=0.0, abs=EXPECTED_DATA_ATOL)

        with ns_timing.time("pyhf"):
            exp_pyhf_main = [float(x) for x in model.expected_data(pyhf_params, include_auxdata=False)]
        with ns_timing.time("nextstat"):
            exp_ns_main = [float(x) for x in ns_model.expected_data(ns_params, include_auxdata=False)]
        assert exp_ns_main == pytest.approx(exp_pyhf_main, rel=0.0, abs=EXPECTED_DATA_ATOL)
