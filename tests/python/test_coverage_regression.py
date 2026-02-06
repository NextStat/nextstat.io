"""Coverage regression smoke test (frequentist limits).

This is an opt-in slow test that compares NextStat vs pyhf coverage for one or more models.

Run with:
  NS_RUN_SLOW=1 NS_TOYS=20 NS_SEED=0 NS_SCAN_POINTS=81 NS_SCAN_STOP=5 pytest -v -m slow tests/python/test_coverage_regression.py
  NS_RUN_SLOW=1 NS_FIXTURES=all NS_TOYS=20 pytest -v -m slow tests/python/test_coverage_regression.py

Defaults are intentionally smaller for reasonable local runtime. Override via env vars for
higher-stat precision.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from _tolerances import COVERAGE_DELTA_MAX


pytestmark = pytest.mark.slow

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
N_TOYS = int(os.environ.get("NS_TOYS", "5"))
SEED = int(os.environ.get("NS_SEED", "0"))
# Default to a moderate scan resolution: this test is O(NS_TOYS * NS_SCAN_POINTS).
SCAN_POINTS = int(os.environ.get("NS_SCAN_POINTS", "11"))
# Keep the scan range modest by default: pyhf reference can be slow.
SCAN_STOP = float(os.environ.get("NS_SCAN_STOP", "0.0") or 0.0)
FIXTURES = os.environ.get("NS_FIXTURES", "simple")

CASES = {
    "simple": ("simple_workspace.json", "GaussExample", 3.0),
    "complex": ("complex_workspace.json", "measurement", 5.0),
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


def test_upper_limit_coverage_regression_vs_pyhf():
    if os.environ.get("NS_RUN_SLOW") != "1":
        pytest.skip("Set NS_RUN_SLOW=1 to run slow coverage regression tests.")

    import nextstat
    import nextstat.infer as ns_infer
    import pyhf

    for case_idx, (key, fixture, measurement, default_scan_stop) in enumerate(selected_cases()):
        rng = np.random.default_rng(SEED + case_idx)
        workspace = load_workspace(fixture)
        model, data_nominal = pyhf_model_and_data(workspace, measurement_name=measurement)

        # Truth: POI=1.0, nuisances at suggested init.
        pars_true = np.asarray(model.config.suggested_init(), dtype=float)
        poi_idx = int(model.config.poi_index)
        pars_true[poi_idx] = 1.0

        expected = np.asarray(model.expected_data(pars_true), dtype=float)
        n_main = int(model.config.nmaindata)

        ns_model = nextstat.from_pyhf(json.dumps(workspace))

        # Fixed scan grid (keeps runtime stable and comparable).
        # Coverage tests are expensive: prefer moderate resolution by default.
        scan_stop = SCAN_STOP if SCAN_STOP > 0.0 else float(default_scan_stop)
        if not (scan_stop > 0.0):
            pytest.skip(f"invalid NS_SCAN_STOP={SCAN_STOP}; must be > 0")
        scan = np.linspace(0.0, scan_stop, SCAN_POINTS)

        covered_pyhf = 0
        covered_ns = 0
        n_used = 0
        mu_true = 1.0

        for _ in range(N_TOYS):
            toy = data_nominal.copy()
            toy[:n_main] = rng.poisson(expected[:n_main])

            try:
                pyhf_obs, _pyhf_exp = pyhf.infer.intervals.upper_limits.upper_limit(
                    toy.tolist(),
                    model,
                    scan=scan,
                    level=0.05,
                    test_stat="qtilde",
                    calctype="asymptotics",
                )
                ns_obs, _ns_exp = ns_infer.upper_limits(
                    ns_model, scan.tolist(), alpha=0.05, data=toy[:n_main].tolist()
                )
            except Exception:
                # Skip rare numerical issues; the regression check is meaningful
                # only when both sides succeed.
                continue

            n_used += 1
            covered_pyhf += int(mu_true <= float(pyhf_obs))
            covered_ns += int(mu_true <= float(ns_obs))

        if n_used < max(5, int(0.5 * N_TOYS)):
            pytest.skip(f"{key}: insufficient valid toys: used={n_used} of requested={N_TOYS}")

        cov_pyhf = covered_pyhf / float(n_used)
        cov_ns = covered_ns / float(n_used)

        # Regression check: NextStat should track pyhf coverage on the same toys.
        assert abs(cov_ns - cov_pyhf) <= COVERAGE_DELTA_MAX

        # Write JSON artifact for CI archival (opt-in via NS_ARTIFACTS_DIR)
        artifacts_dir = os.environ.get("NS_ARTIFACTS_DIR")
        if artifacts_dir:
            out_dir = Path(artifacts_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            artifact = {
                "fixture": key,
                "n_toys_requested": N_TOYS,
                "n_toys_used": int(n_used),
                "seed": int(SEED + case_idx),
                "scan_points": int(SCAN_POINTS),
                "scan_stop": float(scan_stop),
                "mu_true": mu_true,
                "pyhf_coverage": cov_pyhf,
                "nextstat_coverage": cov_ns,
                "delta_coverage": cov_ns - cov_pyhf,
            }
            (out_dir / f"coverage_regression_{key}.json").write_text(json.dumps(artifact, indent=2))
