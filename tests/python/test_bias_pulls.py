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

from _tolerances import (
    COVERAGE_1SIGMA_DELTA_MAX,
    PULL_MEAN_DELTA_MAX,
    PULL_STD_DELTA_MAX,
)

pytestmark = pytest.mark.slow

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
N_TOYS = int(os.environ.get("NS_TOYS", "40"))
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
    # Hessian inversion can fail numerically on some toys; callers should handle exceptions.
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

        pulls_pyhf = []
        pulls_ns = []
        cover_pyhf = []
        cover_ns = []

        ns_model = nextstat.from_pyhf(json.dumps(workspace))
        ns_poi_idx = ns_model.poi_index()
        assert ns_poi_idx is not None
        ns_poi_idx = int(ns_poi_idx)

        for _ in range(N_TOYS):
            toy = data_nominal.copy()
            toy[:n_main] = rng.poisson(expected[:n_main])

            try:
                bestfit_pyhf = np.asarray(pyhf.infer.mle.fit(toy, model), dtype=float)
                unc_pyhf = numerical_uncertainties(model, toy, bestfit_pyhf)
            except Exception:
                # Skip rare numerical failures (singular Hessian, non-convergence).
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
            pytest.skip(f"{key}: insufficient valid toys: used={n_used} < min={min_used}")

        pulls_pyhf = np.asarray(pulls_pyhf[:n_used], dtype=float)
        pulls_ns = np.asarray(pulls_ns[:n_used], dtype=float)
        cover_pyhf = np.asarray(cover_pyhf[:n_used], dtype=bool)
        cover_ns = np.asarray(cover_ns[:n_used], dtype=bool)

        # Compare summary statistics (regression vs pyhf, not absolute unbiasedness)
        d_mean = float(pulls_ns.mean() - pulls_pyhf.mean())
        d_std = float(pulls_ns.std(ddof=1) - pulls_pyhf.std(ddof=1))
        d_cov = float(cover_ns.mean() - cover_pyhf.mean())

        assert abs(d_mean) <= PULL_MEAN_DELTA_MAX, (
            f"{key}: |Delta mean(pull_mu)|={abs(d_mean):.4f} too large"
        )
        assert abs(d_std) <= PULL_STD_DELTA_MAX, f"{key}: |Delta std(pull_mu)|={abs(d_std):.4f} too large"
        assert abs(d_cov) <= COVERAGE_1SIGMA_DELTA_MAX, (
            f"{key}: |Delta coverage_1sigma(mu)|={abs(d_cov):.4f} too large"
        )

        # Write JSON artifact for CI archival (opt-in via NS_ARTIFACTS_DIR)
        artifacts_dir = os.environ.get("NS_ARTIFACTS_DIR")
        if artifacts_dir:
            out_dir = Path(artifacts_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            artifact = {
                "fixture": key,
                "n_toys_requested": N_TOYS,
                "n_toys_used": int(n_used),
                "seed": SEED,
                "pyhf": {
                    "pull_mean": float(pulls_pyhf.mean()),
                    "pull_std": float(pulls_pyhf.std(ddof=1)),
                    "coverage_1sigma": float(cover_pyhf.mean()),
                },
                "nextstat": {
                    "pull_mean": float(pulls_ns.mean()),
                    "pull_std": float(pulls_ns.std(ddof=1)),
                    "coverage_1sigma": float(cover_ns.mean()),
                },
                "delta": {
                    "mean": d_mean,
                    "std": d_std,
                    "coverage_1sigma": d_cov,
                },
            }
            (out_dir / f"bias_pulls_{key}.json").write_text(
                json.dumps(artifact, indent=2)
            )
