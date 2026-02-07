"""Opt-in parity checks on real HEPData pyhf workspaces.

These tests are skipped unless the user downloads and materializes the external
workspaces via:

  python3 tests/hepdata/fetch_workspaces.py

The resulting JSONs are written under `tests/hepdata/workspaces/` and are not
committed to git.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import nextstat

from _tolerances import TWICE_NLL_ATOL, TWICE_NLL_RTOL


pyhf = pytest.importorskip("pyhf")


HEPDATA_WORKSPACES_DIR = Path(__file__).resolve().parents[1] / "hepdata" / "workspaces"


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
        if name not in dst_index:
            raise AssertionError(f"Destination missing parameter '{name}'")
        out[dst_index[name]] = float(value)
    return out


def _shift_params(params, bounds, shift: float = 0.123) -> list[float]:
    out: list[float] = []
    for x, (lo, hi) in zip(params, bounds):
        lo_f = float("-inf") if lo is None else float(lo)
        hi_f = float("inf") if hi is None else float(hi)
        y = float(x) + shift
        if y < lo_f:
            y = lo_f
        if y > hi_f:
            y = hi_f
        out.append(y)
    return out


def _workspace_cases() -> list[tuple[str, dict[str, Any]]]:
    if not HEPDATA_WORKSPACES_DIR.exists():
        pytest.skip(
            "HEPData workspaces not downloaded. Run: python3 tests/hepdata/fetch_workspaces.py",
            allow_module_level=True,
        )

    cases: list[tuple[str, dict[str, Any]]] = []
    for p in sorted(HEPDATA_WORKSPACES_DIR.glob("**/*.json")):
        try:
            w = json.loads(p.read_text())
        except Exception as e:  # pragma: no cover
            raise AssertionError(f"Failed to parse JSON workspace: {p}") from e
        cases.append((str(p.relative_to(HEPDATA_WORKSPACES_DIR)), w))
    if not cases:
        pytest.skip("HEPData workspaces dir exists but is empty.", allow_module_level=True)
    return cases


@pytest.mark.parametrize(("relpath", "workspace"), _workspace_cases())
def test_hepdata_workspaces_twice_nll_parity(relpath: str, workspace: dict[str, Any], ns_timing):
    measurement_name = workspace["measurements"][0]["name"]

    with ns_timing.time(f"pyhf:model+data:{relpath}"):
        try:
            pyhf_model, pyhf_data = _pyhf_model_and_data(workspace, measurement_name)
        except (
            pyhf.exceptions.InvalidModel,
            pyhf.exceptions.InvalidMeasurement,
            pyhf.exceptions.InvalidSpecification,
            pyhf.exceptions.InvalidWorkspaceOperation,
        ) as e:
            pytest.skip(f"pyhf cannot build model for {relpath}: {e}")
        pyhf_init = list(pyhf_model.config.suggested_init())
        pyhf_bounds = list(pyhf_model.config.suggested_bounds())

    with ns_timing.time(f"nextstat:model:{relpath}"):
        ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
        ns_names = ns_model.parameter_names()
        ns_init = ns_model.suggested_init()

    assert set(ns_names) == set(pyhf_model.config.par_names)

    # 1) Nominal suggested init
    with ns_timing.time(f"pyhf:twice_nll:init:{relpath}"):
        pyhf_val = _pyhf_twice_nll(pyhf_model, pyhf_data, pyhf_init)
    ns_params = _map_params_by_name(
        pyhf_model.config.par_names,
        pyhf_init,
        ns_names,
        ns_init,
    )
    with ns_timing.time(f"nextstat:nll:init:{relpath}"):
        ns_val = 2.0 * float(ns_model.nll(ns_params))
    assert ns_val == pytest.approx(pyhf_val, rel=TWICE_NLL_RTOL, abs=TWICE_NLL_ATOL)

    # 2) Shifted vector (non-trivial nuisance values)
    pyhf_shift = _shift_params(pyhf_init, pyhf_bounds, shift=0.123)
    with ns_timing.time(f"pyhf:twice_nll:shift:{relpath}"):
        pyhf_val_shift = _pyhf_twice_nll(pyhf_model, pyhf_data, pyhf_shift)
    ns_params_shift = _map_params_by_name(
        pyhf_model.config.par_names,
        pyhf_shift,
        ns_names,
        ns_init,
    )
    with ns_timing.time(f"nextstat:nll:shift:{relpath}"):
        ns_val_shift = 2.0 * float(ns_model.nll(ns_params_shift))
    assert ns_val_shift == pytest.approx(pyhf_val_shift, rel=TWICE_NLL_RTOL, abs=TWICE_NLL_ATOL)


@pytest.mark.slow
def test_hepdata_workspaces_mle_smoke(ns_timing):
    import os

    if os.environ.get("NS_RUN_SLOW") != "1":
        pytest.skip("Set NS_RUN_SLOW=1 to run slow HEPData MLE smoke tests.")

    for relpath, workspace in _workspace_cases():
        measurement_name = workspace["measurements"][0]["name"]
        with ns_timing.time(f"pyhf:model+data:{relpath}"):
            pyhf_model, pyhf_data = _pyhf_model_and_data(workspace, measurement_name)

        with ns_timing.time(f"pyhf:fit:{relpath}"):
            pyhf_bestfit = pyhf.infer.mle.fit(pyhf_data, pyhf_model)
            pyhf_nll = _pyhf_twice_nll(pyhf_model, pyhf_data, pyhf_bestfit) / 2.0

        with ns_timing.time(f"nextstat:fit:{relpath}"):
            ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
            mle = nextstat.MaximumLikelihoodEstimator()
            ns_res = mle.fit(ns_model)

        assert ns_res.converged, f"NextStat fit did not converge for {relpath}"
        assert ns_res.nll == pytest.approx(pyhf_nll, abs=1.0), f"NLL mismatch too large for {relpath}"

        # Guard against the old 1e6 placeholder uncertainty regression.
        for unc in ns_res.uncertainties:
            assert unc != pytest.approx(1e6, abs=1e-6), f"suspicious uncertainty=1e6 for {relpath}"
