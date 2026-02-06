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

from _tolerances import TWICE_NLL_ATOL, TWICE_NLL_RTOL


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


def _assert_nll_parity(workspace: dict[str, Any], measurement_name: str, *, seed: int, n_random: int):
    pyhf_model, pyhf_data = _pyhf_model_and_data(workspace, measurement_name)
    pyhf_init = list(map(float, pyhf_model.config.suggested_init()))
    pyhf_bounds = [(float(a), float(b)) for a, b in pyhf_model.config.suggested_bounds()]

    ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
    ns_names = ns_model.parameter_names()
    ns_init = ns_model.suggested_init()

    assert set(ns_names) == set(pyhf_model.config.par_names)

    def twice_nll_ns(pyhf_params: list[float]) -> float:
        ns_params = _map_params_by_name(pyhf_model.config.par_names, pyhf_params, ns_names, ns_init)
        return 2.0 * float(ns_model.nll(ns_params))

    def twice_nll_pyhf(pyhf_params: list[float]) -> float:
        return _pyhf_twice_nll(pyhf_model, pyhf_data, pyhf_params)

    # suggested init
    assert twice_nll_ns(pyhf_init) == pytest.approx(
        twice_nll_pyhf(pyhf_init),
        rel=TWICE_NLL_RTOL,
        abs=TWICE_NLL_ATOL,
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


def _make_workspace_multichannel(n_bins: int) -> dict[str, Any]:
    # 3 channels: SR, CR1, CR2. Signal only in SR. Backgrounds with shapesys.
    def ch(name: str, sig: float, bkg: float, unc: float):
        signal = {
            "name": "signal",
            "data": [sig] * n_bins,
            "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
        }
        background = {
            "name": "background",
            "data": [bkg + 0.1 * i for i in range(n_bins)],
            "modifiers": [{"name": f"shapesys_{name}", "type": "shapesys", "data": [unc] * n_bins}],
        }
        return {"name": name, "samples": [signal, background]}

    channels = [
        ch("SR", sig=5.0, bkg=100.0, unc=10.0),
        ch("CR1", sig=0.0, bkg=500.0, unc=30.0),
        ch("CR2", sig=0.0, bkg=800.0, unc=40.0),
    ]
    observations = []
    for c in channels:
        # Observed near nominal
        total = [sum(s["data"][i] for s in c["samples"]) for i in range(n_bins)]
        observations.append({"name": c["name"], "data": [float(x) for x in total]})

    return {
        "channels": channels,
        "observations": observations,
        "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": []}}],
        "version": "1.0.0",
    }


def _make_workspace_histo_normsys_staterror(n_bins: int) -> dict[str, Any]:
    # Single channel, two samples with:
    # - global lumi (constrained)
    # - background normsys + histosys
    # - staterror per bin
    signal = {
        "name": "signal",
        "data": [10.0] * n_bins,
        "modifiers": [{"name": "mu", "type": "normfactor", "data": None}, {"name": "lumi", "type": "lumi", "data": None}],
    }
    nominal = [200.0 + 0.25 * i for i in range(n_bins)]
    hi = [x * (1.08 + 0.01 * ((i % 5) - 2)) for i, x in enumerate(nominal)]
    lo = [x * (0.92 - 0.005 * ((i % 7) - 3)) for i, x in enumerate(nominal)]
    stat = [max(1.0, 0.15 * (x**0.5)) for x in nominal]
    background = {
        "name": "background",
        "data": nominal,
        "modifiers": [
            {"name": "lumi", "type": "lumi", "data": None},
            {"name": "bkg_norm", "type": "normsys", "data": {"hi": 1.05, "lo": 0.95}},
            {"name": "bkg_shape", "type": "histosys", "data": {"hi_data": hi, "lo_data": lo}},
            {"name": "staterror_c", "type": "staterror", "data": stat},
        ],
    }
    obs = [float(s + b) for s, b in zip(signal["data"], background["data"])]
    return {
        "channels": [{"name": "c", "samples": [signal, background]}],
        "observations": [{"name": "c", "data": obs}],
        "measurements": [
            {
                "name": "m",
                "config": {
                    "poi": "mu",
                    "parameters": [
                        {"name": "lumi", "inits": [1.0], "bounds": [[0.9, 1.1]], "auxdata": [1.0], "sigmas": [0.02]},
                    ],
                },
            }
        ],
        "version": "1.0.0",
    }


def _make_workspace_shapefactor_control_region(n_bins: int) -> dict[str, Any]:
    # Two channels: SR (signal+background) and CR (background-only with shapefactor).
    sr_signal = {
        "name": "signal",
        "data": [6.0] * n_bins,
        "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
    }
    sr_bkg = {
        "name": "background",
        "data": [80.0] * n_bins,
        "modifiers": [{"name": "sr_shape", "type": "histosys", "data": {"hi_data": [85.0] * n_bins, "lo_data": [75.0] * n_bins}}],
    }
    cr_bkg = {
        "name": "background",
        "data": [500.0 + i for i in range(n_bins)],
        "modifiers": [{"name": "sf_cr", "type": "shapefactor", "data": None}],
    }
    channels = [
        {"name": "SR", "samples": [sr_signal, sr_bkg]},
        {"name": "CR", "samples": [cr_bkg]},
    ]
    obs_sr = [float(a + b) for a, b in zip(sr_signal["data"], sr_bkg["data"])]
    obs_cr = [float(x) for x in cr_bkg["data"]]
    return {
        "channels": channels,
        "observations": [{"name": "SR", "data": obs_sr}, {"name": "CR", "data": obs_cr}],
        "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": []}}],
        "version": "1.0.0",
    }


@pytest.mark.parametrize(
    ("workspace", "measurement", "seed", "n_random"),
    [
        (_make_workspace_multichannel(3), "m", 0, 6),
        (_make_workspace_histo_normsys_staterror(10), "m", 1, 6),
        (_make_workspace_shapefactor_control_region(4), "m", 2, 6),
    ],
)
def test_model_zoo_nll_parity(workspace, measurement, seed, n_random):
    _assert_nll_parity(workspace, measurement, seed=seed, n_random=n_random)
