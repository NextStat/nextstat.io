"""Parity check: `nextstat.unbinned.UnbinnedAnalysis` vs `nextstat unbinned-fit` CLI."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import nextstat


REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_cli() -> str:
    env = os.environ.get("NS_CLI_BIN")
    if env:
        return env
    for profile in ("release", "debug"):
        candidate = REPO_ROOT / "target" / profile / "nextstat"
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("nextstat")
    if found:
        return found
    pytest.skip("nextstat CLI not found (set NS_CLI_BIN or build ns-cli)")
    return ""


def _write_parquet(path: Path, *, values: np.ndarray) -> None:
    if importlib.util.find_spec("pyarrow") is None:
        pytest.skip("pyarrow is required for unbinned parity test")
    import pyarrow as pa
    import pyarrow.parquet as pq

    bounds = (60.0, 120.0)
    table = pa.table({"mass": pa.array(values.astype(np.float64), type=pa.float64())})
    table = table.replace_schema_metadata(
        {
            b"nextstat.schema_version": b"nextstat_unbinned_events_v1",
            b"nextstat.observables": json.dumps(
                [{"name": "mass", "bounds": [bounds[0], bounds[1]]}],
                separators=(",", ":"),
            ).encode(),
        }
    )
    pq.write_table(table, str(path))


def _build_spec(data_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                {"name": "mu", "init": 1.0, "bounds": [0.0, 5.0]},
                {"name": "mu_sig", "init": 91.0, "bounds": [85.0, 95.0]},
                {"name": "sigma_sig", "init": 2.5, "bounds": [0.5, 10.0]},
                {"name": "lambda_bkg", "init": -0.03, "bounds": [-0.1, -0.001]},
            ],
        },
        "channels": [
            {
                "name": "SR",
                "data": {"file": str(data_path)},
                "observables": [{"name": "mass", "bounds": [60.0, 120.0]}],
                "processes": [
                    {
                        "name": "signal",
                        "pdf": {"type": "gaussian", "observable": "mass", "params": ["mu_sig", "sigma_sig"]},
                        "yield": {"type": "scaled", "base_yield": 150.0, "scale": "mu"},
                    },
                    {
                        "name": "background",
                        "pdf": {"type": "exponential", "observable": "mass", "params": ["lambda_bkg"]},
                        "yield": {"type": "fixed", "value": 450.0},
                    },
                ],
            }
        ],
    }


def test_unbinned_analysis_fit_matches_cli(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    sig = rng.normal(91.2, 2.5, size=150)
    sig = sig[(sig >= 60.0) & (sig <= 120.0)]
    while sig.size < 150:
        extra = rng.normal(91.2, 2.5, size=64)
        extra = extra[(extra >= 60.0) & (extra <= 120.0)]
        sig = np.concatenate([sig, extra])
    sig = sig[:150]
    # Truncated exp(lambda=-0.03) on [60,120]
    u = rng.uniform(0.0, 1.0, size=450)
    lo, hi, lam = 60.0, 120.0, -0.03
    mass_b = np.log(np.exp(lam * lo) + u * (np.exp(lam * hi) - np.exp(lam * lo))) / lam
    obs = np.concatenate([sig, mass_b])
    rng.shuffle(obs)

    data_path = tmp_path / "obs.parquet"
    spec_path = tmp_path / "spec.json"
    _write_parquet(data_path, values=obs)
    spec_path.write_text(json.dumps(_build_spec(data_path), indent=2), encoding="utf-8")

    analysis = nextstat.unbinned.from_config(spec_path)
    py_fit = analysis.fit()

    cli = _find_cli()
    proc = subprocess.run(
        [cli, "unbinned-fit", "--config", str(spec_path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"CLI failed:\n{proc.stderr}\n{proc.stdout}")
    cli_fit = json.loads(proc.stdout)

    assert py_fit.converged == bool(cli_fit["converged"])
    assert py_fit.bestfit == pytest.approx(cli_fit["bestfit"], rel=1e-10, abs=1e-10)
    assert py_fit.nll == pytest.approx(float(cli_fit["nll"]), rel=1e-10, abs=1e-10)
