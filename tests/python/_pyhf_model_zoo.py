"""Synthetic pyhf workspaces used across the Apex2 parity harness.

This module is intentionally dependency-free: it only builds workspace dicts.
Both:
  - pytest model-zoo tests (tests/python/test_pyhf_model_zoo.py)
  - Apex2 report runners (tests/apex2_pyhf_validation_report.py)
import these generators to avoid drift.
"""

from __future__ import annotations

from typing import Any


def make_synthetic_shapesys_workspace(n_bins: int) -> dict[str, Any]:
    signal = {
        "name": "signal",
        "data": [5.0] * n_bins,
        "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
    }
    bkg = {
        "name": "background",
        "data": [50.0] * n_bins,
        "modifiers": [{"name": "uncorr_bkguncrt", "type": "shapesys", "data": [5.0] * n_bins}],
    }
    return {
        "channels": [{"name": "c", "samples": [signal, bkg]}],
        "observations": [{"name": "c", "data": [53.0] * n_bins}],
        "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": []}}],
        "version": "1.0.0",
    }


def make_workspace_multichannel(n_bins: int) -> dict[str, Any]:
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


def make_workspace_histo_normsys_staterror(n_bins: int) -> dict[str, Any]:
    # Single channel, two samples with:
    # - global lumi (constrained)
    # - background normsys + histosys
    # - staterror per bin
    signal = {
        "name": "signal",
        "data": [10.0] * n_bins,
        "modifiers": [
            {"name": "mu", "type": "normfactor", "data": None},
            {"name": "lumi", "type": "lumi", "data": None},
        ],
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


def make_workspace_shapefactor_control_region(n_bins: int) -> dict[str, Any]:
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

