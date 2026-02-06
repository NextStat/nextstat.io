"""Coverage regression smoke test (frequentist limits).

This is an opt-in slow test that compares NextStat vs pyhf coverage for a simple model.

Run with:
  NS_RUN_SLOW=1 NS_TOYS=20 NS_SEED=0 NS_SCAN_POINTS=81 pytest -v -m slow tests/python/test_coverage_regression.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import nextstat
import nextstat.infer as ns_infer

from _tolerances import COVERAGE_DELTA_MAX


pytestmark = pytest.mark.slow

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE = "simple_workspace.json"
N_TOYS = int(os.environ.get("NS_TOYS", "20"))
SEED = int(os.environ.get("NS_SEED", "0"))
SCAN_POINTS = int(os.environ.get("NS_SCAN_POINTS", "81"))


def load_workspace() -> dict:
    return json.loads((FIXTURES_DIR / FIXTURE).read_text())


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

    import pyhf

    rng = np.random.default_rng(SEED)
    workspace = load_workspace()
    model, data_nominal = pyhf_model_and_data(workspace, measurement_name="GaussExample")

    # Truth: POI=1.0, nuisances at suggested init.
    pars_true = np.asarray(model.config.suggested_init(), dtype=float)
    poi_idx = int(model.config.poi_index)
    pars_true[poi_idx] = 1.0

    expected = np.asarray(model.expected_data(pars_true), dtype=float)
    n_main = int(model.config.nmaindata)

    ns_model = nextstat.from_pyhf(json.dumps(workspace))

    # Fixed scan grid (keeps runtime stable and comparable).
    # Coverage tests are expensive: prefer moderate resolution by default.
    scan = np.linspace(0.0, 5.0, SCAN_POINTS)

    covered_pyhf = 0
    covered_ns = 0
    mu_true = 1.0

    for _ in range(N_TOYS):
        toy = data_nominal.copy()
        toy[:n_main] = rng.poisson(expected[:n_main])

        pyhf_obs, _pyhf_exp = pyhf.infer.intervals.upper_limits.upper_limit(
            toy.tolist(),
            model,
            scan=scan,
            level=0.05,
            test_stat="qtilde",
            calctype="asymptotics",
        )
        ns_obs, _ns_exp = ns_infer.upper_limits(ns_model, scan.tolist(), alpha=0.05, data=toy[:n_main].tolist())

        covered_pyhf += int(mu_true <= float(pyhf_obs))
        covered_ns += int(mu_true <= float(ns_obs))

    cov_pyhf = covered_pyhf / float(N_TOYS)
    cov_ns = covered_ns / float(N_TOYS)

    # Regression check: NextStat should track pyhf coverage on the same toys.
    assert abs(cov_ns - cov_pyhf) <= COVERAGE_DELTA_MAX

    # Write JSON artifact for CI archival (opt-in via NS_ARTIFACTS_DIR)
    artifacts_dir = os.environ.get("NS_ARTIFACTS_DIR")
    if artifacts_dir:
        out_dir = Path(artifacts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "fixture": "simple",
            "n_toys": N_TOYS,
            "mu_true": mu_true,
            "pyhf_coverage": cov_pyhf,
            "nextstat_coverage": cov_ns,
            "delta_coverage": cov_ns - cov_pyhf,
        }
        (out_dir / "coverage_regression_simple.json").write_text(
            json.dumps(artifact, indent=2)
        )
