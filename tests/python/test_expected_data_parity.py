"""Parity check for expected_data vs pyhf.

This exercises the Python binding `HistFactoryModel.expected_data`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

import nextstat

from _tolerances import EXPECTED_DATA_ATOL


pyhf = pytest.importorskip("pyhf")

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


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
    ],
)
def test_expected_data_matches_pyhf_at_suggested_init(fixture: str, measurement: str):
    workspace = load_fixture(fixture)
    model = pyhf_model(workspace, measurement)
    pyhf_init = list(map(float, model.config.suggested_init()))

    ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
    ns_params = map_params_by_name(
        model.config.par_names,
        pyhf_init,
        ns_model.parameter_names(),
        ns_model.suggested_init(),
    )

    exp_pyhf = [float(x) for x in model.expected_data(pyhf_init)]
    exp_ns = [float(x) for x in ns_model.expected_data(ns_params)]
    assert exp_ns == pytest.approx(exp_pyhf, rel=0.0, abs=EXPECTED_DATA_ATOL)

    exp_pyhf_main = [float(x) for x in model.expected_data(pyhf_init, include_auxdata=False)]
    exp_ns_main = [float(x) for x in ns_model.expected_data(ns_params, include_auxdata=False)]
    assert exp_ns_main == pytest.approx(exp_pyhf_main, rel=0.0, abs=EXPECTED_DATA_ATOL)


@pytest.mark.parametrize(
    ("fixture", "measurement", "seed"),
    [
        ("simple_workspace.json", "GaussExample", 0),
        ("complex_workspace.json", "measurement", 1),
    ],
)
def test_expected_data_matches_pyhf_at_random_points(fixture: str, measurement: str, seed: int):
    workspace = load_fixture(fixture)
    model = pyhf_model(workspace, measurement)
    pyhf_init = list(map(float, model.config.suggested_init()))
    pyhf_bounds = [(float(a), float(b)) for a, b in model.config.suggested_bounds()]

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

        exp_pyhf = [float(x) for x in model.expected_data(pyhf_params)]
        exp_ns = [float(x) for x in ns_model.expected_data(ns_params)]
        assert exp_ns == pytest.approx(exp_pyhf, rel=0.0, abs=EXPECTED_DATA_ATOL)

        exp_pyhf_main = [float(x) for x in model.expected_data(pyhf_params, include_auxdata=False)]
        exp_ns_main = [float(x) for x in ns_model.expected_data(ns_params, include_auxdata=False)]
        assert exp_ns_main == pytest.approx(exp_pyhf_main, rel=0.0, abs=EXPECTED_DATA_ATOL)
