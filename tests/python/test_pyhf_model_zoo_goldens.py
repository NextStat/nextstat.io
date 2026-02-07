"""Golden-based regression harness for HistFactory parity.

Unlike `test_pyhf_model_zoo.py`, this does NOT import `pyhf` at runtime.
Goldens are generated once via:
  tests/python/generate_pyhf_model_zoo_goldens.py
"""

from __future__ import annotations

import json
from array import array
from pathlib import Path
from typing import Any

import pytest

import nextstat

from _tolerances import EXPECTED_DATA_ATOL, TWICE_NLL_ATOL, TWICE_NLL_RTOL


REPO = Path(__file__).resolve().parents[2]
GOLDENS = REPO / "tests" / "fixtures" / "pyhf_model_zoo_goldens.json"

_GOLD = json.loads(GOLDENS.read_text())
_CASE_IDXS = list(range(len(_GOLD["cases"])))


def _map_params_by_name(src_names, src_params, dst_names, dst_init):
    dst_index = {name: i for i, name in enumerate(dst_names)}
    out = list(dst_init)
    for name, value in zip(src_names, src_params):
        out[dst_index[name]] = float(value)
    return out


@pytest.mark.parametrize("case_idx", _CASE_IDXS)
def test_histfactory_goldens(case_idx: int, ns_timing):
    case: dict[str, Any] = _GOLD["cases"][case_idx]

    workspace = case["workspace"]
    measurement = case["measurement"]
    gold_names = case["par_names"]

    # Measurement isn't used by NextStat (workspace JSON includes it), but it is
    # part of the golden identity and helps debugging.
    assert isinstance(measurement, str) and measurement

    with ns_timing.time("nextstat_build"):
        ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
        ns_names = ns_model.parameter_names()
        ns_init = ns_model.suggested_init()

    assert set(ns_names) == set(gold_names)

    points = case["points"]
    assert points and points[0]["label"] == "suggested_init"

    for i, pt in enumerate(points):
        params_pyhf_order = pt["params"]
        ns_params_list = _map_params_by_name(gold_names, params_pyhf_order, ns_names, ns_init)

        with ns_timing.time("nextstat"):
            twice_nll_ns = 2.0 * float(ns_model.nll(ns_params_list))
            expected_full_ns = [float(x) for x in ns_model.expected_data(ns_params_list)]
            expected_main_ns = [float(x) for x in ns_model.expected_data(ns_params_list, include_auxdata=False)]

        assert twice_nll_ns == pytest.approx(float(pt["twice_nll"]), rel=TWICE_NLL_RTOL, abs=TWICE_NLL_ATOL)
        assert expected_full_ns == pytest.approx(pt["expected_full"], rel=0.0, abs=EXPECTED_DATA_ATOL)
        assert expected_main_ns == pytest.approx(pt["expected_main"], rel=0.0, abs=EXPECTED_DATA_ATOL)

        # Lock in the perf-critical buffer-protocol surface once per case.
        if i == 0:
            ns_params_buf = array("d", ns_params_list)
            with ns_timing.time("nextstat_buf"):
                twice_nll_ns_buf = 2.0 * float(ns_model.nll(ns_params_buf))
                expected_full_ns_buf = [float(x) for x in ns_model.expected_data(ns_params_buf)]
                expected_main_ns_buf = [float(x) for x in ns_model.expected_data(ns_params_buf, include_auxdata=False)]

            assert twice_nll_ns_buf == pytest.approx(float(pt["twice_nll"]), rel=TWICE_NLL_RTOL, abs=TWICE_NLL_ATOL)
            assert expected_full_ns_buf == pytest.approx(pt["expected_full"], rel=0.0, abs=EXPECTED_DATA_ATOL)
            assert expected_main_ns_buf == pytest.approx(pt["expected_main"], rel=0.0, abs=EXPECTED_DATA_ATOL)
