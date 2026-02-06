"""Parity check for expected_data vs pyhf.

This exercises the Python binding `HistFactoryModel.expected_data`.
"""

from __future__ import annotations

import json
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
