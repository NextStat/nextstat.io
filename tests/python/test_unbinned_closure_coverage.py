"""Unbinned closure + coverage tests for canonical 1D PDFs.

Closure: generate a large dataset from known truth parameters, fit with NextStat
CLI (``nextstat unbinned-fit``), verify that each fitted parameter recovers the
truth within tolerance.

Coverage (slow, opt-in): generate many toy datasets, fit each, check that the
fraction of toys where the POI truth falls within the 1-sigma interval is
consistent with 68.3%.

Run closure tests (fast, default):
  pytest -v tests/python/test_unbinned_closure_coverage.py -k closure

Run coverage tests (slow, opt-in):
  NS_RUN_SLOW=1 NS_TOYS=200 pytest -v -m slow tests/python/test_unbinned_closure_coverage.py -k coverage

Requires: numpy, pyarrow.
The ``nextstat`` CLI binary must be on PATH or set via ``NS_CLI_BIN``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from _tolerances import (
    UNBINNED_CLOSURE_PARAM_ATOL,
    UNBINNED_CLOSURE_PARAM_RTOL,
    UNBINNED_COVERAGE_1SIGMA_HI,
    UNBINNED_COVERAGE_1SIGMA_LO,
)

# ---------------------------------------------------------------------------
# Helpers: synthetic data generation (bounded 1D PDFs via rejection sampling)
# ---------------------------------------------------------------------------


def _sample_truncated_gaussian(
    rng: np.random.Generator, mu: float, sigma: float, lo: float, hi: float, n: int
) -> np.ndarray:
    """Sample from Gaussian truncated to [lo, hi]."""
    samples = []
    while len(samples) < n:
        batch = rng.normal(mu, sigma, size=max(n * 2, 1024))
        batch = batch[(batch >= lo) & (batch <= hi)]
        samples.extend(batch.tolist())
    return np.array(samples[:n])


def _sample_truncated_exponential(
    rng: np.random.Generator, lam: float, lo: float, hi: float, n: int
) -> np.ndarray:
    """Sample from Exponential(lambda) truncated to [lo, hi].

    PDF ∝ exp(lambda * x) on [lo, hi].  lambda < 0 for a falling distribution.
    Uses inverse-CDF: F(x) = (exp(lam*x) - exp(lam*lo)) / (exp(lam*hi) - exp(lam*lo)).
    """
    if abs(lam) < 1e-14:
        return rng.uniform(lo, hi, size=n)
    u = rng.uniform(0.0, 1.0, size=n)
    ea = np.exp(lam * lo)
    eb = np.exp(lam * hi)
    return np.log(ea + u * (eb - ea)) / lam


def _sample_crystal_ball(
    rng: np.random.Generator,
    mu: float,
    sigma: float,
    alpha: float,
    n_tail: float,
    lo: float,
    hi: float,
    n: int,
) -> np.ndarray:
    """Sample from Crystal Ball PDF on [lo, hi] via rejection sampling."""
    from scipy.stats import crystalball  # type: ignore

    rv = crystalball(beta=abs(alpha), m=n_tail, loc=mu, scale=sigma)
    samples = []
    while len(samples) < n:
        batch = rv.rvs(size=max(n * 3, 2048), random_state=rng)
        batch = batch[(batch >= lo) & (batch <= hi)]
        samples.extend(batch.tolist())
    return np.array(samples[:n])


def _chebyshev_pdf_unnormed(x: np.ndarray, coeffs: List[float], lo: float, hi: float) -> np.ndarray:
    """Evaluate unnormalized Chebyshev PDF: 1 + c1*T1(t) + c2*T2(t) + ... where t = 2*(x-lo)/(hi-lo) - 1."""
    t = 2.0 * (x - lo) / (hi - lo) - 1.0
    val = np.ones_like(x)
    if len(coeffs) >= 1:
        val = val + coeffs[0] * t
    if len(coeffs) >= 2:
        val = val + coeffs[1] * (2.0 * t * t - 1.0)
    if len(coeffs) >= 3:
        val = val + coeffs[2] * (4.0 * t**3 - 3.0 * t)
    return np.maximum(val, 0.0)


def _sample_chebyshev(
    rng: np.random.Generator, coeffs: List[float], lo: float, hi: float, n: int
) -> np.ndarray:
    """Sample from Chebyshev PDF on [lo, hi] via rejection sampling."""
    x_grid = np.linspace(lo, hi, 10000)
    pdf_grid = _chebyshev_pdf_unnormed(x_grid, coeffs, lo, hi)
    pdf_max = float(pdf_grid.max()) * 1.05

    samples = []
    while len(samples) < n:
        batch_x = rng.uniform(lo, hi, size=max(n * 3, 4096))
        batch_u = rng.uniform(0.0, pdf_max, size=len(batch_x))
        pdf_vals = _chebyshev_pdf_unnormed(batch_x, coeffs, lo, hi)
        accepted = batch_x[batch_u <= pdf_vals]
        samples.extend(accepted.tolist())
    return np.array(samples[:n])


# ---------------------------------------------------------------------------
# Helpers: Parquet + spec file generation
# ---------------------------------------------------------------------------


def _write_observed_parquet(
    data: np.ndarray,
    obs_name: str,
    obs_bounds: Tuple[float, float],
    path: Path,
) -> None:
    """Write a single-column Float64 Parquet file with NextStat metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({obs_name: pa.array(data, type=pa.float64())})
    meta_kv = {
        b"nextstat.schema_version": b"nextstat_unbinned_events_v1",
        b"nextstat.observables": json.dumps(
            [{"name": obs_name, "bounds": list(obs_bounds)}], separators=(",", ":")
        ).encode(),
    }
    existing = table.schema.metadata or {}
    table = table.replace_schema_metadata({**existing, **meta_kv})
    pq.write_table(table, str(path))


def _write_spec_json(
    spec: Dict[str, Any],
    path: Path,
) -> None:
    """Write an unbinned_spec_v0 JSON file."""
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

# Each case: (name, obs_name, obs_bounds, parameters, processes, truth_params,
#             data_generator, n_events_closure, poi_name)

CaseType = Dict[str, Any]


def _gauss_exp_case() -> CaseType:
    """Gaussian signal + Exponential background (the canonical HEP case)."""
    obs_name = "mass"
    obs_bounds = (60.0, 120.0)
    truth = {
        "mu": 1.0,
        "mu_sig": 91.2,
        "sigma_sig": 2.5,
        "lambda_bkg": -0.03,
    }
    n_sig = 1000
    n_bkg = 3000

    def generate(rng: np.random.Generator) -> np.ndarray:
        sig = _sample_truncated_gaussian(
            rng, truth["mu_sig"], truth["sigma_sig"], *obs_bounds, int(truth["mu"] * n_sig)
        )
        bkg = _sample_truncated_exponential(rng, truth["lambda_bkg"], *obs_bounds, n_bkg)
        return np.concatenate([sig, bkg])

    parameters = [
        {"name": "mu", "init": 1.0, "bounds": [0.0, 5.0]},
        {"name": "mu_sig", "init": 91.0, "bounds": [85.0, 95.0]},
        {"name": "sigma_sig", "init": 2.5, "bounds": [0.5, 10.0]},
        {"name": "lambda_bkg", "init": -0.03, "bounds": [-0.1, -0.001]},
    ]
    processes = [
        {
            "name": "signal",
            "pdf": {"type": "gaussian", "observable": obs_name, "params": ["mu_sig", "sigma_sig"]},
            "yield": {"type": "scaled", "base_yield": float(n_sig), "scale": "mu"},
        },
        {
            "name": "background",
            "pdf": {"type": "exponential", "observable": obs_name, "params": ["lambda_bkg"]},
            "yield": {"type": "fixed", "value": float(n_bkg)},
        },
    ]
    return {
        "name": "gauss_exp",
        "obs_name": obs_name,
        "obs_bounds": obs_bounds,
        "parameters": parameters,
        "processes": processes,
        "truth": truth,
        "generate": generate,
        "n_events_closure": int(truth["mu"] * n_sig) + n_bkg,
        "poi_name": "mu",
        "poi_truth": truth["mu"],
    }


def _crystal_ball_exp_case() -> CaseType:
    """Crystal Ball signal + Exponential background."""
    obs_name = "mass"
    obs_bounds = (60.0, 120.0)
    truth = {
        "mu": 1.0,
        "mu_cb": 91.2,
        "sigma_cb": 2.5,
        "alpha_cb": 1.5,
        "n_cb": 5.0,
        "lambda_bkg": -0.025,
    }
    n_sig = 300
    n_bkg = 1500

    def generate(rng: np.random.Generator) -> np.ndarray:
        sig = _sample_crystal_ball(
            rng,
            truth["mu_cb"],
            truth["sigma_cb"],
            truth["alpha_cb"],
            truth["n_cb"],
            *obs_bounds,
            int(truth["mu"] * n_sig),
        )
        bkg = _sample_truncated_exponential(rng, truth["lambda_bkg"], *obs_bounds, n_bkg)
        return np.concatenate([sig, bkg])

    parameters = [
        {"name": "mu", "init": 1.0, "bounds": [0.0, 5.0]},
        {"name": "mu_cb", "init": 91.0, "bounds": [85.0, 95.0]},
        {"name": "sigma_cb", "init": 3.0, "bounds": [0.5, 10.0]},
        {"name": "alpha_cb", "init": 1.5, "bounds": [0.1, 5.0], "fixed": True},
        {"name": "n_cb", "init": 5.0, "bounds": [1.01, 50.0], "fixed": True},
        {"name": "lambda_bkg", "init": -0.02, "bounds": [-0.1, -0.001]},
    ]
    processes = [
        {
            "name": "signal",
            "pdf": {
                "type": "crystal_ball",
                "observable": obs_name,
                "params": ["mu_cb", "sigma_cb", "alpha_cb", "n_cb"],
            },
            "yield": {"type": "scaled", "base_yield": float(n_sig), "scale": "mu"},
        },
        {
            "name": "background",
            "pdf": {"type": "exponential", "observable": obs_name, "params": ["lambda_bkg"]},
            "yield": {"type": "fixed", "value": float(n_bkg)},
        },
    ]
    return {
        "name": "crystal_ball_exp",
        "obs_name": obs_name,
        "obs_bounds": obs_bounds,
        "parameters": parameters,
        "processes": processes,
        "truth": truth,
        "generate": generate,
        "n_events_closure": int(truth["mu"] * n_sig) + n_bkg,
        "poi_name": "mu",
        "poi_truth": truth["mu"],
    }


def _chebyshev_only_case() -> CaseType:
    """Pure Chebyshev background (order-1 polynomial shape fit).

    Uses a single Chebyshev coefficient to avoid positivity violations
    during optimization (higher-order polynomials can go negative).
    """
    obs_name = "x"
    obs_bounds = (0.0, 10.0)
    n_events = 5000
    truth = {
        "c1": -0.15,
    }

    def generate(rng: np.random.Generator) -> np.ndarray:
        return _sample_chebyshev(rng, [truth["c1"]], *obs_bounds, n_events)

    parameters = [
        {"name": "c1", "init": 0.0, "bounds": [-0.9, 0.9]},
    ]
    processes = [
        {
            "name": "background",
            "pdf": {"type": "chebyshev", "observable": obs_name, "params": ["c1"]},
            "yield": {"type": "fixed", "value": float(n_events)},
        },
    ]
    return {
        "name": "chebyshev_only",
        "obs_name": obs_name,
        "obs_bounds": obs_bounds,
        "parameters": parameters,
        "processes": processes,
        "truth": truth,
        "generate": generate,
        "n_events_closure": n_events,
        "poi_name": None,
        "poi_truth": None,
    }


ALL_CASES = [_gauss_exp_case, _crystal_ball_exp_case, _chebyshev_only_case]
CASE_IDS = ["gauss_exp", "crystal_ball_exp", "chebyshev_only"]


# ---------------------------------------------------------------------------
# Helpers: CLI binary resolution
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_cli_bin() -> str:
    """Locate the ``nextstat`` CLI binary."""
    env = os.environ.get("NS_CLI_BIN")
    if env:
        return env
    # Prefer release build in the repo.
    for profile in ("release", "debug"):
        candidate = _REPO_ROOT / "target" / profile / "nextstat"
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("nextstat")
    if found:
        return found
    pytest.skip("nextstat CLI binary not found (build with cargo build --release -p ns-cli)")
    return ""  # unreachable


# ---------------------------------------------------------------------------
# Helpers: build spec + fit
# ---------------------------------------------------------------------------


def _build_spec(
    case: CaseType,
    data_path: Path,
) -> Dict[str, Any]:
    """Build an unbinned_spec_v0 dict for a case."""
    spec: Dict[str, Any] = {
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                {k: v for k, v in p.items() if k != "fixed"}
                for p in case["parameters"]
            ],
        },
        "channels": [
            {
                "name": "SR",
                "data": {"file": str(data_path)},
                "observables": [
                    {
                        "name": case["obs_name"],
                        "bounds": list(case["obs_bounds"]),
                    }
                ],
                "processes": case["processes"],
            }
        ],
    }
    if case["poi_name"]:
        spec["model"]["poi"] = case["poi_name"]
    return spec


def _fit_unbinned(spec_path: Path) -> Dict[str, Any]:
    """Fit an unbinned model by invoking the ``nextstat unbinned-fit`` CLI."""
    cli = _find_cli_bin()
    proc = subprocess.run(
        [cli, "unbinned-fit", "--config", str(spec_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"nextstat unbinned-fit failed (rc={proc.returncode}):\n{proc.stderr}\n{proc.stdout}"
        )
    result = json.loads(proc.stdout)
    return {
        "parameters": result["bestfit"],
        "uncertainties": result["uncertainties"],
        "nll": result["nll"],
        "converged": result["converged"],
        "parameter_names": result["parameter_names"],
    }


# ---------------------------------------------------------------------------
# Closure tests (always run)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_fn", ALL_CASES, ids=CASE_IDS)
def test_unbinned_closure(case_fn, tmp_path: Path) -> None:
    """Closure: generate a large dataset from truth, fit, check parameter recovery."""
    case = case_fn()
    rng = np.random.default_rng(seed=42)
    data = case["generate"](rng)

    data_path = tmp_path / "observed.parquet"
    _write_observed_parquet(data, case["obs_name"], case["obs_bounds"], data_path)

    spec = _build_spec(case, data_path)
    spec_path = tmp_path / "spec.json"
    _write_spec_json(spec, spec_path)

    result = _fit_unbinned(spec_path)
    assert result["converged"], f"Fit did not converge for {case['name']}"

    param_names = result["parameter_names"]
    param_values = result["parameters"]
    truth = case["truth"]

    for i, name in enumerate(param_names):
        if name not in truth:
            continue
        # Skip parameters that are fixed (not free in the fit).
        param_spec = next((p for p in case["parameters"] if p["name"] == name), None)
        if param_spec and param_spec.get("fixed"):
            continue

        fitted = param_values[i]
        expected = truth[name]
        abs_diff = abs(fitted - expected)

        # Check both absolute and relative tolerance.
        abs_ok = abs_diff <= UNBINNED_CLOSURE_PARAM_ATOL
        rel_ok = abs_diff <= UNBINNED_CLOSURE_PARAM_RTOL * abs(expected) if expected != 0 else abs_ok

        assert abs_ok or rel_ok, (
            f"Closure failed for '{name}' in {case['name']}: "
            f"fitted={fitted:.6f}, truth={expected:.6f}, "
            f"|diff|={abs_diff:.6f}, atol={UNBINNED_CLOSURE_PARAM_ATOL}, "
            f"rtol={UNBINNED_CLOSURE_PARAM_RTOL}"
        )

    # Write JSON artifact.
    artifact = {
        "case": case["name"],
        "seed": 42,
        "n_events": len(data),
        "converged": result["converged"],
        "nll": result["nll"],
        "parameters": {
            name: {"fitted": param_values[i], "truth": truth.get(name)}
            for i, name in enumerate(param_names)
        },
    }
    (tmp_path / f"closure_{case['name']}.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )

    artifacts_dir = os.environ.get("NS_ARTIFACTS_DIR")
    if artifacts_dir:
        out_dir = Path(artifacts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"unbinned_closure_{case['name']}.json").write_text(
            json.dumps(artifact, indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Coverage tests (slow, opt-in)
# ---------------------------------------------------------------------------

N_COVERAGE_TOYS = int(os.environ.get("NS_TOYS", "100"))
COVERAGE_SEED = int(os.environ.get("NS_SEED", "12345"))

# Only cases with a POI are meaningful for coverage.
COVERAGE_CASES = [fn for fn in ALL_CASES if fn()["poi_name"] is not None]
COVERAGE_IDS = [fn()["name"] for fn in COVERAGE_CASES]


@pytest.mark.slow
@pytest.mark.parametrize("case_fn", COVERAGE_CASES, ids=COVERAGE_IDS)
def test_unbinned_coverage(case_fn, tmp_path: Path) -> None:
    """Coverage: run many toys, check that 1-sigma interval covers truth ~68%."""
    if os.environ.get("NS_RUN_SLOW") != "1":
        pytest.skip("Set NS_RUN_SLOW=1 to run slow unbinned coverage tests.")

    case = case_fn()
    poi_name = case["poi_name"]
    poi_truth = case["poi_truth"]

    covered = 0
    n_valid = 0

    for toy_idx in range(N_COVERAGE_TOYS):
        rng = np.random.default_rng(seed=COVERAGE_SEED + toy_idx)
        data = case["generate"](rng)

        data_path = tmp_path / f"toy_{toy_idx}.parquet"
        _write_observed_parquet(data, case["obs_name"], case["obs_bounds"], data_path)

        spec = _build_spec(case, data_path)
        spec_path = tmp_path / f"spec_{toy_idx}.json"
        _write_spec_json(spec, spec_path)

        try:
            result = _fit_unbinned(spec_path)
        except Exception:
            continue

        if not result["converged"]:
            continue

        param_names = result["parameter_names"]
        poi_idx_result = None
        for i, name in enumerate(param_names):
            if name == poi_name:
                poi_idx_result = i
                break

        if poi_idx_result is None:
            continue

        fitted_poi = result["parameters"][poi_idx_result]
        sigma_poi = result["uncertainties"][poi_idx_result]

        if sigma_poi <= 0 or not np.isfinite(sigma_poi):
            continue

        n_valid += 1
        if abs(fitted_poi - poi_truth) <= sigma_poi:
            covered += 1

    if n_valid < max(10, int(0.3 * N_COVERAGE_TOYS)):
        pytest.skip(
            f"{case['name']}: insufficient valid toys: {n_valid}/{N_COVERAGE_TOYS}"
        )

    coverage = covered / float(n_valid)

    assert UNBINNED_COVERAGE_1SIGMA_LO <= coverage <= UNBINNED_COVERAGE_1SIGMA_HI, (
        f"Coverage out of range for {case['name']}: "
        f"{coverage:.3f} not in [{UNBINNED_COVERAGE_1SIGMA_LO}, {UNBINNED_COVERAGE_1SIGMA_HI}] "
        f"(covered={covered}/{n_valid})"
    )

    # Write JSON artifact.
    artifact = {
        "case": case["name"],
        "n_toys_requested": N_COVERAGE_TOYS,
        "n_toys_valid": n_valid,
        "seed": COVERAGE_SEED,
        "poi_name": poi_name,
        "poi_truth": poi_truth,
        "coverage_1sigma": coverage,
        "covered": covered,
    }

    artifacts_dir = os.environ.get("NS_ARTIFACTS_DIR")
    if artifacts_dir:
        out_dir = Path(artifacts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"unbinned_coverage_{case['name']}.json").write_text(
            json.dumps(artifact, indent=2), encoding="utf-8"
        )
