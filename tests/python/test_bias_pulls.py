"""Toy pull/bias smoke tests (regression vs pyhf).

These tests are intentionally slow and opt-in.
Run with:
  NS_RUN_SLOW=1 NS_TOYS=200 NS_SEED=0 pytest -v -m slow tests/python/test_bias_pulls.py

To include additional fixtures:
  NS_RUN_SLOW=1 NS_TOYS=200 NS_SEED=0 NS_FIXTURES=all pytest -v -m slow tests/python/test_bias_pulls.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.slow

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
N_TOYS = int(os.environ.get("NS_TOYS", "200"))
SEED = int(os.environ.get("NS_SEED", "0"))
FIXTURES = os.environ.get("NS_FIXTURES", "simple")

CASES = {
    "simple": ("simple_workspace.json", "GaussExample"),
    "complex": ("complex_workspace.json", "measurement"),
}


def selected_cases():
    if FIXTURES.strip().lower() == "all":
        keys = ["simple", "complex"]
    else:
        keys = [k.strip().lower() for k in FIXTURES.split(",") if k.strip()]

    out = []
    for k in keys:
        if k not in CASES:
            raise ValueError(f"Unknown fixture key '{k}'. Expected one of: {sorted(CASES)} or 'all'.")
        out.append((k, *CASES[k]))
    return out


def load_workspace(fixture: str) -> dict:
    return json.loads((FIXTURES_DIR / fixture).read_text())


def pyhf_model_and_data(workspace: dict, measurement_name: str):
    import pyhf

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


def pyhf_nll(model, data: np.ndarray, params: np.ndarray) -> float:
    import pyhf

    return float(pyhf.infer.mle.twice_nll(params, data, model).item()) / 2.0


def numerical_uncertainties(model, data: np.ndarray, bestfit: np.ndarray) -> np.ndarray:
    n = len(bestfit)
    h_step = 1e-4
    damping = 1e-9

    f0 = pyhf_nll(model, data, bestfit)
    hess = np.zeros((n, n), dtype=float)

    for i in range(n):
        hi = h_step * max(abs(bestfit[i]), 1.0)
        xp = bestfit.copy()
        xm = bestfit.copy()
        xp[i] += hi
        xm[i] -= hi
        fp = pyhf_nll(model, data, xp)
        fm = pyhf_nll(model, data, xm)
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
            fij = (pyhf_nll(model, data, xpp) - pyhf_nll(model, data, xpm) - pyhf_nll(model, data, xmp) + pyhf_nll(model, data, xmm)) / (
                4.0 * hi * hj
            )
            hess[i, j] = fij
            hess[j, i] = fij

    hess = hess + np.eye(n) * damping
    cov = np.linalg.inv(hess)
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


def test_pull_mu_regression_vs_pyhf():
    if os.environ.get("NS_RUN_SLOW") != "1":
        pytest.skip("Set NS_RUN_SLOW=1 to run slow toy regression tests.")

    import pyhf
    import nextstat

    for case_idx, (key, fixture, measurement) in enumerate(selected_cases()):
        rng = np.random.default_rng(SEED + case_idx)
        workspace = load_workspace(fixture)

        model, data_nominal = pyhf_model_and_data(workspace, measurement_name=measurement)

        # Choose "truth" parameters: POI=1, nuisance at suggested init
        pars_true = np.asarray(model.config.suggested_init(), dtype=float)
        poi_idx = int(model.config.poi_index)
        pars_true[poi_idx] = 1.0

        expected = np.asarray(model.expected_data(pars_true), dtype=float)
        n_main = int(model.config.nmaindata)

        pulls_pyhf = np.empty(N_TOYS, dtype=float)
        pulls_ns = np.empty(N_TOYS, dtype=float)
        cover_pyhf = np.empty(N_TOYS, dtype=bool)
        cover_ns = np.empty(N_TOYS, dtype=bool)

        ns_model = nextstat.from_pyhf(json.dumps(workspace))
        ns_poi_idx = ns_model.poi_index()
        assert ns_poi_idx is not None
        ns_poi_idx = int(ns_poi_idx)

        for i in range(N_TOYS):
            toy = data_nominal.copy()
            toy[:n_main] = rng.poisson(expected[:n_main])

            bestfit_pyhf = np.asarray(pyhf.infer.mle.fit(toy, model), dtype=float)
            unc_pyhf = numerical_uncertainties(model, toy, bestfit_pyhf)
            mu_hat_pyhf = float(bestfit_pyhf[poi_idx])
            mu_sig_pyhf = float(unc_pyhf[poi_idx])
            assert np.isfinite(mu_sig_pyhf) and mu_sig_pyhf > 0.0
            pulls_pyhf[i] = (mu_hat_pyhf - 1.0) / mu_sig_pyhf
            cover_pyhf[i] = abs(mu_hat_pyhf - 1.0) <= mu_sig_pyhf

            res_ns = nextstat.fit(ns_model, data=toy[:n_main].tolist())
            mu_hat_ns = float(res_ns.bestfit[ns_poi_idx])
            mu_sig_ns = float(res_ns.uncertainties[ns_poi_idx])
            assert np.isfinite(mu_sig_ns) and mu_sig_ns > 0.0
            pulls_ns[i] = (mu_hat_ns - 1.0) / mu_sig_ns
            cover_ns[i] = abs(mu_hat_ns - 1.0) <= mu_sig_ns

        # Compare summary statistics (regression vs pyhf, not absolute unbiasedness)
        d_mean = float(pulls_ns.mean() - pulls_pyhf.mean())
        d_std = float(pulls_ns.std(ddof=1) - pulls_pyhf.std(ddof=1))
        d_cov = float(cover_ns.mean() - cover_pyhf.mean())

        assert abs(d_mean) <= 0.05, f"{key}: |Delta mean(pull_mu)|={abs(d_mean):.4f} too large"
        assert abs(d_std) <= 0.05, f"{key}: |Delta std(pull_mu)|={abs(d_std):.4f} too large"
        assert abs(d_cov) <= 0.03, f"{key}: |Delta coverage_1sigma(mu)|={abs(d_cov):.4f} too large"
