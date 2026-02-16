"""Parity check: `nextstat.fit_toys` (unbinned) vs `nextstat unbinned-fit-toys` CLI."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import nextstat
from _tolerances import GPU_FIT_NLL_ATOL, GPU_PARAM_ATOL, METAL_FIT_NLL_ATOL, METAL_PARAM_ATOL


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
                        "pdf": {
                            "type": "gaussian",
                            "observable": "mass",
                            "params": ["mu_sig", "sigma_sig"],
                        },
                        "yield": {"type": "scaled", "base_yield": 120.0, "scale": "mu"},
                    },
                    {
                        "name": "background",
                        "pdf": {
                            "type": "exponential",
                            "observable": "mass",
                            "params": ["lambda_bkg"],
                        },
                        "yield": {"type": "fixed", "value": 360.0},
                    },
                ],
            }
        ],
    }


def _emit_report(kind: str, payload: dict[str, Any]) -> None:
    out_dir = os.environ.get("NS_UNBINNED_TOY_PARITY_REPORT_DIR")
    if not out_dir:
        return
    p = Path(out_dir).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    out_path = p / f"{kind}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _generate_observed_mass(seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sig = rng.normal(91.0, 2.5, size=120)
    sig = sig[(sig >= 60.0) & (sig <= 120.0)]
    while sig.size < 120:
        extra = rng.normal(91.0, 2.5, size=64)
        extra = extra[(extra >= 60.0) & (extra <= 120.0)]
        sig = np.concatenate([sig, extra])
    sig = sig[:120]

    u = rng.uniform(0.0, 1.0, size=360)
    lo, hi, lam = 60.0, 120.0, -0.03
    bkg = np.log(np.exp(lam * lo) + u * (np.exp(lam * hi) - np.exp(lam * lo))) / lam
    obs = np.concatenate([sig, bkg])
    rng.shuffle(obs)
    return obs


def _prepare_spec(tmp_path: Path, *, seed: int = 17) -> Path:
    data_path = tmp_path / "obs.parquet"
    spec_path = tmp_path / "spec.json"
    _write_parquet(data_path, values=_generate_observed_mass(seed))
    spec_path.write_text(json.dumps(_build_spec(data_path), indent=2), encoding="utf-8")
    return spec_path


def _run_unbinned_fit_toys_cli(
    spec_path: Path,
    *,
    n_toys: int,
    seed: int,
    gpu: str | None = None,
) -> dict[str, Any]:
    cli = _find_cli()
    cmd = [
        cli,
        "unbinned-fit-toys",
        "--config",
        str(spec_path),
        "--n-toys",
        str(n_toys),
        "--seed",
        str(seed),
        "--threads",
        "1",
    ]
    if gpu is not None:
        cmd.extend(["--gpu", gpu])

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    if proc.returncode != 0:
        msg = f"{proc.stderr}\n{proc.stdout}".lower()
        if gpu is not None and (
            "not available at runtime" in msg
            or "support not compiled in" in msg
            or "no metal device found" in msg
            or "no device/driver" in msg
            or "requires building with" in msg
        ):
            pytest.skip(f"CLI backend '{gpu}' unavailable in this environment")
        raise RuntimeError(f"CLI failed:\n{proc.stderr}\n{proc.stdout}")
    return json.loads(proc.stdout)


def test_unbinned_fit_toys_matches_cli(tmp_path: Path) -> None:
    spec_path = _prepare_spec(tmp_path, seed=17)

    model = nextstat.UnbinnedModel.from_config(str(spec_path))
    params = model.suggested_init()
    poi_idx = model.poi_index()
    assert poi_idx is not None

    nextstat.set_threads(1)
    n_toys = 8
    seed = 123
    py_results = nextstat.fit_toys(model, params, n_toys=n_toys, seed=seed)
    assert len(py_results) == n_toys

    cli_out = _run_unbinned_fit_toys_cli(spec_path, n_toys=n_toys, seed=seed)

    assert cli_out["poi_index"] == poi_idx
    assert cli_out["results"]["n_error"] == 0
    assert cli_out["results"]["n_toys"] == n_toys
    assert cli_out["results"]["n_converged"] == sum(1 for r in py_results if r.converged)

    cli_converged = cli_out["results"]["converged"]
    cli_poi_hat = cli_out["results"]["poi_hat"]
    cli_nll = cli_out["results"]["nll"]
    assert len(cli_converged) == n_toys
    assert len(cli_poi_hat) == n_toys
    assert len(cli_nll) == n_toys

    conv_mismatch = 0
    max_abs_delta_poi = 0.0
    max_abs_delta_nll = 0.0
    compared_poi = 0
    compared_nll = 0
    for i, (py_fit, cli_ok, cli_mu, cli_val_nll) in enumerate(
        zip(py_results, cli_converged, cli_poi_hat, cli_nll, strict=True)
    ):
        assert cli_ok is not None, f"toy {i}: unexpected CLI None convergence flag"
        if py_fit.converged != bool(cli_ok):
            conv_mismatch += 1
        assert py_fit.converged == bool(cli_ok)

        if cli_mu is not None:
            delta_poi = abs(py_fit.parameters[poi_idx] - float(cli_mu))
            if math.isfinite(delta_poi):
                max_abs_delta_poi = max(max_abs_delta_poi, delta_poi)
                compared_poi += 1
            assert py_fit.parameters[poi_idx] == pytest.approx(
                float(cli_mu), rel=1e-6, abs=1e-6
            )
        if cli_val_nll is not None:
            delta_nll = abs(py_fit.nll - float(cli_val_nll))
            if math.isfinite(delta_nll):
                max_abs_delta_nll = max(max_abs_delta_nll, delta_nll)
                compared_nll += 1
            assert py_fit.nll == pytest.approx(float(cli_val_nll), rel=1e-6, abs=1e-6)

    _emit_report(
        "api_cli_cpu",
        {
            "schema_version": "nextstat.unbinned_toy_parity_report.v1",
            "kind": "api_cli",
            "backend": "cpu",
            "ok": True,
            "n_toys": n_toys,
            "seed": seed,
            "metrics": {
                "n_converged_python": sum(1 for r in py_results if r.converged),
                "n_converged_cli": sum(1 for x in cli_converged if bool(x)),
                "convergence_mismatch_count": conv_mismatch,
                "max_abs_delta_poi": max_abs_delta_poi,
                "max_abs_delta_nll": max_abs_delta_nll,
                "compared_poi": compared_poi,
                "compared_nll": compared_nll,
            },
            "thresholds": {
                "poi_abs_tol": 1e-6,
                "nll_abs_tol": 1e-6,
            },
        },
    )


@pytest.mark.parametrize(
    ("gpu_backend", "availability_attr", "nll_atol", "param_atol"),
    [
        ("cuda", "has_cuda", GPU_FIT_NLL_ATOL, GPU_PARAM_ATOL),
        ("metal", "has_metal", METAL_FIT_NLL_ATOL, METAL_PARAM_ATOL),
    ],
)
def test_unbinned_fit_toys_cli_gpu_parity(
    tmp_path: Path,
    gpu_backend: str,
    availability_attr: str,
    nll_atol: float,
    param_atol: float,
) -> None:
    has_backend = getattr(nextstat, availability_attr, None)
    if has_backend is None or not callable(has_backend):
        pytest.skip(f"nextstat.{availability_attr} is not available")
    if not bool(has_backend()):
        pytest.skip(f"{gpu_backend} backend is not available")

    spec_path = _prepare_spec(tmp_path, seed=19)
    n_toys = 8
    seed = 333

    cpu_out = _run_unbinned_fit_toys_cli(spec_path, n_toys=n_toys, seed=seed)
    gpu_out = _run_unbinned_fit_toys_cli(
        spec_path,
        n_toys=n_toys,
        seed=seed,
        gpu=gpu_backend,
    )

    assert gpu_out["results"]["n_toys"] == cpu_out["results"]["n_toys"] == n_toys
    assert gpu_out["poi_index"] == cpu_out["poi_index"]
    assert gpu_out["results"]["n_error"] == 0
    assert cpu_out["results"]["n_error"] == 0

    cpu_conv = cpu_out["results"]["converged"]
    gpu_conv = gpu_out["results"]["converged"]
    cpu_poi = cpu_out["results"]["poi_hat"]
    gpu_poi = gpu_out["results"]["poi_hat"]
    cpu_nll = cpu_out["results"]["nll"]
    gpu_nll = gpu_out["results"]["nll"]
    assert len(cpu_conv) == len(gpu_conv) == n_toys

    conv_mismatch = 0
    n_compared = 0
    max_abs_delta_poi = 0.0
    max_abs_delta_nll = 0.0
    for i, (c_ok, g_ok, c_mu, g_mu, c_nll, g_nll) in enumerate(
        zip(cpu_conv, gpu_conv, cpu_poi, gpu_poi, cpu_nll, gpu_nll, strict=True)
    ):
        assert c_ok is not None and g_ok is not None, f"toy {i}: unexpected None convergence flag"
        if bool(c_ok) != bool(g_ok):
            conv_mismatch += 1
            continue
        if not bool(c_ok):
            continue
        assert c_mu is not None and g_mu is not None, f"toy {i}: missing POI values"
        assert c_nll is not None and g_nll is not None, f"toy {i}: missing NLL values"

        delta_poi = abs(float(g_mu) - float(c_mu))
        if math.isfinite(delta_poi):
            max_abs_delta_poi = max(max_abs_delta_poi, delta_poi)
        delta_nll = abs(float(g_nll) - float(c_nll))
        if math.isfinite(delta_nll):
            max_abs_delta_nll = max(max_abs_delta_nll, delta_nll)

        assert float(g_mu) == pytest.approx(float(c_mu), abs=param_atol)
        assert float(g_nll) == pytest.approx(float(c_nll), abs=nll_atol)
        n_compared += 1

    assert conv_mismatch <= 1, f"too many convergence mismatches: {conv_mismatch}/{n_toys}"
    assert n_compared >= max(1, n_toys // 2), "too few converged toy pairs for parity check"

    _emit_report(
        f"cli_gpu_{gpu_backend}",
        {
            "schema_version": "nextstat.unbinned_toy_parity_report.v1",
            "kind": "cli_gpu",
            "backend": gpu_backend,
            "ok": True,
            "n_toys": n_toys,
            "seed": seed,
            "metrics": {
                "compared_pairs": n_compared,
                "convergence_mismatch_count": conv_mismatch,
                "max_abs_delta_poi": max_abs_delta_poi,
                "max_abs_delta_nll": max_abs_delta_nll,
            },
            "thresholds": {
                "poi_abs_tol": param_atol,
                "nll_abs_tol": nll_atol,
                "max_convergence_mismatch": 1,
            },
        },
    )
