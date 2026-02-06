"""Toy pull/bias regression tests on synthetic "model zoo" workspaces (opt-in).

These are intentionally slow and gated behind NS_RUN_SLOW=1.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import nextstat

from _tolerances import (
    COVERAGE_1SIGMA_DELTA_MAX,
    PULL_MEAN_DELTA_MAX,
    PULL_STD_DELTA_MAX,
)
from _pyhf_model_zoo import make_workspace_multichannel


pytestmark = pytest.mark.slow
pyhf = pytest.importorskip("pyhf")


N_TOYS = int(os.environ.get("NS_TOYS", "30"))
SEED = int(os.environ.get("NS_SEED", "0"))


def _pyhf_model_and_data(workspace: dict, measurement_name: str):
    ws = pyhf.Workspace(workspace)
    model = ws.model(
        measurement_name=measurement_name,
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
    )
    data = np.asarray(ws.data(model), dtype=float)
    return model, data


def _pyhf_nll(model, data: np.ndarray, params: np.ndarray) -> float:
    return float(pyhf.infer.mle.twice_nll(params, data, model).item()) / 2.0


def _numerical_uncertainties(model, data: np.ndarray, bestfit: np.ndarray) -> np.ndarray:
    """Diagonal uncertainties via numerical Hessian of NLL (pyhf reference)."""
    n = len(bestfit)
    h_step = 1e-4
    damping = 1e-9

    f0 = _pyhf_nll(model, data, bestfit)
    hess = np.zeros((n, n), dtype=float)

    for i in range(n):
        hi = h_step * max(abs(bestfit[i]), 1.0)
        xp = bestfit.copy()
        xm = bestfit.copy()
        xp[i] += hi
        xm[i] -= hi
        fp = _pyhf_nll(model, data, xp)
        fm = _pyhf_nll(model, data, xm)
        hess[i, i] = (fp - 2.0 * f0 + fm) / (hi * hi)

        for j in range(i + 1, n):
            hj = h_step * max(abs(bestfit[j]), 1.0)
            xpp = bestfit.copy()
            xpm = bestfit.copy()
            xmp = bestfit.copy()
            xmm = bestfit.copy()
            xpp[i] += hi
            xpp[j] += hj
            xpm[i] += hi
            xpm[j] -= hj
            xmp[i] -= hi
            xmp[j] += hj
            xmm[i] -= hi
            xmm[j] -= hj
            fij = (
                _pyhf_nll(model, data, xpp)
                - _pyhf_nll(model, data, xpm)
                - _pyhf_nll(model, data, xmp)
                + _pyhf_nll(model, data, xmm)
            ) / (4.0 * hi * hj)
            hess[i, j] = fij
            hess[j, i] = fij

    hess = hess + np.eye(n) * damping
    cov = np.linalg.inv(hess)
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


def test_pull_mu_regression_vs_pyhf_model_zoo_multichannel():
    if os.environ.get("NS_RUN_SLOW") != "1":
        pytest.skip("Set NS_RUN_SLOW=1 to run slow toy regression tests.")

    workspace = make_workspace_multichannel(3)
    model, data_nominal = _pyhf_model_and_data(workspace, measurement_name="m")

    # Choose "truth" parameters: POI=1, nuisance at suggested init
    pars_true = np.asarray(model.config.suggested_init(), dtype=float)
    poi_idx = int(model.config.poi_index)
    pars_true[poi_idx] = 1.0

    expected = np.asarray(model.expected_data(pars_true), dtype=float)
    n_main = int(model.config.nmaindata)

    pulls_pyhf = []
    pulls_ns = []
    cover_pyhf = []
    cover_ns = []

    ns_model = nextstat.from_pyhf(json.dumps(workspace))
    ns_poi_idx = ns_model.poi_index()
    assert ns_poi_idx is not None
    ns_poi_idx = int(ns_poi_idx)

    rng = np.random.default_rng(SEED)
    for _ in range(N_TOYS):
        toy = data_nominal.copy()
        toy[:n_main] = rng.poisson(expected[:n_main])

        try:
            bestfit_pyhf = np.asarray(pyhf.infer.mle.fit(toy, model), dtype=float)
            unc_pyhf = _numerical_uncertainties(model, toy, bestfit_pyhf)
        except Exception:
            continue

        mu_hat_pyhf = float(bestfit_pyhf[poi_idx])
        mu_sig_pyhf = float(unc_pyhf[poi_idx])
        if not (np.isfinite(mu_sig_pyhf) and mu_sig_pyhf > 0.0):
            continue

        try:
            res_ns = nextstat.fit(ns_model, data=toy[:n_main].tolist())
        except Exception:
            continue

        mu_hat_ns = float(res_ns.bestfit[ns_poi_idx])
        mu_sig_ns = float(res_ns.uncertainties[ns_poi_idx])
        if not (np.isfinite(mu_sig_ns) and mu_sig_ns > 0.0):
            continue

        pulls_pyhf.append((mu_hat_pyhf - 1.0) / mu_sig_pyhf)
        cover_pyhf.append(abs(mu_hat_pyhf - 1.0) <= mu_sig_pyhf)
        pulls_ns.append((mu_hat_ns - 1.0) / mu_sig_ns)
        cover_ns.append(abs(mu_hat_ns - 1.0) <= mu_sig_ns)

    n_used = min(len(pulls_pyhf), len(pulls_ns))
    min_used = max(10, int(0.5 * N_TOYS))
    if n_used < min_used:
        pytest.skip(f"insufficient valid toys: used={n_used} < min={min_used}")

    pulls_pyhf = np.asarray(pulls_pyhf[:n_used], dtype=float)
    pulls_ns = np.asarray(pulls_ns[:n_used], dtype=float)
    cover_pyhf = np.asarray(cover_pyhf[:n_used], dtype=bool)
    cover_ns = np.asarray(cover_ns[:n_used], dtype=bool)

    d_mean = float(pulls_ns.mean() - pulls_pyhf.mean())
    d_std = float(pulls_ns.std(ddof=1) - pulls_pyhf.std(ddof=1))
    d_cov = float(cover_ns.mean() - cover_pyhf.mean())

    assert abs(d_mean) <= PULL_MEAN_DELTA_MAX
    assert abs(d_std) <= PULL_STD_DELTA_MAX
    assert abs(d_cov) <= COVERAGE_1SIGMA_DELTA_MAX

