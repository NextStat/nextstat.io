#!/usr/bin/env python3
"""Shared NextStat-only helpers for Bayesian sampler benchmarks."""

from __future__ import annotations

import math
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any


DEFAULT_N_CHAINS = 4
DEFAULT_N_WARMUP = 1000
DEFAULT_N_SAMPLES = 1000
DEFAULT_SEEDS = [42, 123, 777]
DATASET_SEED = 12345


@dataclass
class BenchResult:
    model: str
    engine: str
    seed: int
    wall_secs: float
    ess_bulk: dict[str, float] = field(default_factory=dict)
    ess_tail: dict[str, float] = field(default_factory=dict)
    r_hat: dict[str, float] = field(default_factory=dict)
    divergence_rate: float = 0.0
    ebfmi: list[float] = field(default_factory=list)
    metric_type: str = "diagonal"
    n_leapfrog_sampling_total: int | None = None
    n_leapfrog_warmup_total: int | None = None
    n_leapfrog_total: int | None = None
    mean_tree_depth: float | None = None
    mean_accept_prob: float | None = None


def gen_eight_schools_data() -> dict[str, Any]:
    """Rubin (1981) Eight Schools dataset."""
    return {
        "J": 8,
        "y": [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        "sigma": [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
    }


def gen_glm_logistic_data(
    n: int = 1000,
    p: int = 10,
    seed: int = DATASET_SEED,
) -> dict[str, Any]:
    rng = random.Random(int(seed))
    beta_true = [rng.gauss(0.0, 1.0) for _ in range(int(p))]
    alpha_true = 0.5
    x: list[list[float]] = []
    y: list[int] = []
    for _ in range(int(n)):
        row = [rng.gauss(0.0, 1.0) for _ in range(int(p))]
        logit = alpha_true + sum(beta * value for beta, value in zip(beta_true, row))
        prob = 1.0 / (1.0 + math.exp(-logit))
        x.append(row)
        y.append(1 if rng.random() < prob else 0)
    return {
        "N": n,
        "P": p,
        "X": x,
        "y": y,
    }


def _poisson_sample(rng: random.Random, lam: float) -> int:
    if lam <= 0.0:
        return 0
    if lam > 30.0:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    threshold = math.exp(-lam)
    draws = 0
    prod = 1.0
    while prod > threshold:
        draws += 1
        prod *= rng.random()
    return draws - 1


def gen_glm_poisson_data(
    n: int = 1000,
    p: int = 10,
    seed: int = DATASET_SEED,
) -> dict[str, Any]:
    rng = random.Random(int(seed))
    beta_true = [rng.gauss(0.0, 0.6) for _ in range(int(p))]
    alpha_true = 0.15
    x: list[list[float]] = []
    y: list[int] = []
    for _ in range(int(n)):
        row = [rng.gauss(0.0, 1.0) for _ in range(int(p))]
        eta = alpha_true + sum(beta * value for beta, value in zip(beta_true, row))
        lam = math.exp(eta)
        x.append(row)
        y.append(_poisson_sample(rng, lam))
    return {
        "N": n,
        "P": p,
        "X": x,
        "y": y,
    }


def gen_glm_negbin_data(
    n: int = 1000,
    p: int = 10,
    seed: int = DATASET_SEED,
) -> dict[str, Any]:
    rng = random.Random(int(seed))
    beta_true = [rng.gauss(0.0, 0.5) for _ in range(int(p))]
    alpha_true = -0.1
    dispersion = 0.7
    x: list[list[float]] = []
    y: list[int] = []
    for _ in range(int(n)):
        row = [rng.gauss(0.0, 1.0) for _ in range(int(p))]
        eta = alpha_true + sum(beta * value for beta, value in zip(beta_true, row))
        mu = math.exp(eta)
        lam = rng.gammavariate(1.0 / dispersion, dispersion * mu)
        x.append(row)
        y.append(_poisson_sample(rng, lam))
    return {
        "N": n,
        "P": p,
        "X": x,
        "y": y,
        "dispersion": dispersion,
    }


def _ns_available() -> bool:
    try:
        import nextstat

        return hasattr(nextstat, "sample")
    except ImportError:
        return False


def _assert_nextstat_harness_contract() -> None:
    """Fail fast when running against stale/mismatched nextstat bindings."""
    try:
        import nextstat  # type: ignore
    except ImportError as exc:
        raise RuntimeError("nextstat import failed; build/install local ns-py first") from exc

    if not hasattr(nextstat, "sample"):
        raise RuntimeError("nextstat.sample not found; API mismatch")
    if not hasattr(nextstat, "Posterior"):
        raise RuntimeError("nextstat.Posterior not found; API mismatch")


def _safe_git_rev() -> str | None:
    override = os.environ.get("NEXTSTAT_BENCH_GIT_COMMIT")
    if override:
        return override
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _detect_accelerator_runtime() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "cuda_runtime_available": None,
        "metal_runtime_available": None,
        "nvidia_smi_present": False,
        "nvidia_gpus": [],
    }

    try:
        import nextstat  # type: ignore

        has_cuda = getattr(nextstat, "has_cuda", None)
        if callable(has_cuda):
            meta["cuda_runtime_available"] = bool(has_cuda())
        has_metal = getattr(nextstat, "has_metal", None)
        if callable(has_metal):
            meta["metal_runtime_available"] = bool(has_metal())
    except Exception:
        pass

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return meta

    rows = [line.strip() for line in output.splitlines() if line.strip()]
    meta["nvidia_smi_present"] = True
    meta["nvidia_gpus"] = rows
    return meta


def collect_environment(
    metric: str,
    glm_n: int,
    glm_p: int,
    seeds: list[int],
    *,
    models: list[str] | None = None,
    methods: list[str] | None = None,
    n_chains: int = DEFAULT_N_CHAINS,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_samples: int = DEFAULT_N_SAMPLES,
    target_accept: float | None = None,
    max_treedepth: int | None = None,
) -> dict[str, Any]:
    benchmark_config: dict[str, Any] = {
        "models": models or ["std_normal_10d", "eight_schools", "glm_logistic"],
        "seeds": seeds,
        "n_chains": n_chains,
        "n_warmup": n_warmup,
        "n_samples": n_samples,
        "dataset_seed": DATASET_SEED,
        "metric": metric,
        "glm_n": glm_n,
        "glm_p": glm_p,
    }
    if methods:
        benchmark_config["methods"] = methods
    if target_accept is not None:
        benchmark_config["target_accept"] = target_accept
    if max_treedepth is not None:
        benchmark_config["max_treedepth"] = max_treedepth

    meta: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.replace("\n", " "),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "hostname": socket.gethostname(),
        "git_commit": _safe_git_rev(),
        "benchmark_config": benchmark_config,
        "target_density_parity": {
            "std_normal_10d": "Exact match across methods",
            "eight_schools": "Same non-centered Eight Schools target density across methods",
            "glm_logistic": "Same logistic regression posterior with alpha~N(0,5), beta_j~N(0,2.5) across methods",
            "funnel_ncp_10d": "Same non-centered Neal funnel target density across methods",
        },
        "thread_env": {
            key: os.environ.get(key)
            for key in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "RAYON_NUM_THREADS",
            ]
        },
        "accelerator_runtime": _detect_accelerator_runtime(),
        "execution_lane": {
            "host_policy": os.environ.get("NEXTSTAT_BENCH_HOST_POLICY"),
            "submit_host": os.environ.get("NEXTSTAT_BENCH_SUBMIT_HOST"),
            "build_host": os.environ.get("NEXTSTAT_BENCH_BUILD_HOST"),
            "execute_host": os.environ.get("NEXTSTAT_BENCH_EXECUTE_HOST", socket.gethostname()),
            "scheduler": os.environ.get("NEXTSTAT_BENCH_SCHEDULER"),
        },
    }

    try:
        import nextstat  # type: ignore

        meta["nextstat_version"] = getattr(nextstat, "__version__", None)
        meta["nextstat_module"] = getattr(nextstat, "__file__", None)
    except Exception:
        meta["nextstat_version"] = None
        meta["nextstat_module"] = None

    return meta


def _flatten_numeric_values(value: Any) -> list[float]:
    out: list[float] = []
    if isinstance(value, list):
        for item in value:
            out.extend(_flatten_numeric_values(item))
        return out
    try:
        numeric = float(value)
    except Exception:
        return out
    if math.isfinite(numeric):
        out.append(numeric)
    return out


def _sample_stats_summary(sample_stats: dict[str, Any]) -> dict[str, float | int | None]:
    n_leapfrog_values = _flatten_numeric_values(sample_stats.get("n_leapfrog"))
    n_leapfrog_warmup_values = _flatten_numeric_values(sample_stats.get("n_leapfrog_warmup_total"))
    tree_depth_values = _flatten_numeric_values(sample_stats.get("tree_depth"))
    accept_prob_values = _flatten_numeric_values(sample_stats.get("accept_prob"))
    n_leapfrog_sampling_total = int(sum(n_leapfrog_values)) if n_leapfrog_values else None
    n_leapfrog_warmup_total = (
        int(sum(n_leapfrog_warmup_values)) if n_leapfrog_warmup_values else None
    )
    if n_leapfrog_sampling_total is None and n_leapfrog_warmup_total is None:
        n_leapfrog_total = None
    else:
        n_leapfrog_total = int((n_leapfrog_sampling_total or 0) + (n_leapfrog_warmup_total or 0))
    return {
        "n_leapfrog_sampling_total": n_leapfrog_sampling_total,
        "n_leapfrog_warmup_total": n_leapfrog_warmup_total,
        "n_leapfrog_total": n_leapfrog_total,
        "mean_tree_depth": (
            float(sum(tree_depth_values) / len(tree_depth_values)) if tree_depth_values else None
        ),
        "mean_accept_prob": (
            float(sum(accept_prob_values) / len(accept_prob_values))
            if accept_prob_values
            else None
        ),
    }


def _ns_result_to_bench(
    model_name: str,
    engine: str,
    seed: int,
    wall_secs: float,
    result: dict[str, Any],
    metric: str,
) -> BenchResult:
    diag = result.get("diagnostics", {})
    sample_stats = result.get("sample_stats", {})
    stats_summary = _sample_stats_summary(sample_stats)

    return BenchResult(
        model=model_name,
        engine=engine,
        seed=seed,
        wall_secs=wall_secs,
        ess_bulk=diag.get("ess_bulk", {}),
        ess_tail=diag.get("ess_tail", {}),
        r_hat=diag.get("r_hat", {}),
        divergence_rate=diag.get("divergence_rate", 0.0),
        ebfmi=diag.get("ebfmi", []),
        metric_type=sample_stats.get("metric_type", metric),
        n_leapfrog_sampling_total=(
            int(stats_summary["n_leapfrog_sampling_total"])
            if stats_summary["n_leapfrog_sampling_total"] is not None
            else None
        ),
        n_leapfrog_warmup_total=(
            int(stats_summary["n_leapfrog_warmup_total"])
            if stats_summary["n_leapfrog_warmup_total"] is not None
            else None
        ),
        n_leapfrog_total=(
            int(stats_summary["n_leapfrog_total"])
            if stats_summary["n_leapfrog_total"] is not None
            else None
        ),
        mean_tree_depth=(
            float(stats_summary["mean_tree_depth"])
            if stats_summary["mean_tree_depth"] is not None
            else None
        ),
        mean_accept_prob=(
            float(stats_summary["mean_accept_prob"])
            if stats_summary["mean_accept_prob"] is not None
            else None
        ),
    )


def _run_nextstat_model(
    *,
    model_name: str,
    model: Any,
    seed: int,
    method: str,
    metric: str,
    n_chains: int,
    n_warmup: int,
    n_samples: int,
    target_accept: float,
    max_treedepth: int | None,
    sample_kwargs: dict[str, Any] | None = None,
) -> BenchResult:
    import nextstat

    kwargs: dict[str, Any] = {
        "method": method,
        "n_chains": n_chains,
        "n_warmup": n_warmup,
        "n_samples": n_samples,
        "seed": seed,
        "metric": metric,
        "target_accept": target_accept,
    }
    if max_treedepth is not None:
        kwargs["max_treedepth"] = max_treedepth
    if sample_kwargs:
        kwargs.update(sample_kwargs)

    t0 = time.perf_counter()
    result = nextstat.sample(model, **kwargs)
    wall_secs = time.perf_counter() - t0
    return _ns_result_to_bench(
        model_name,
        f"nextstat_{method}",
        seed,
        wall_secs,
        result,
        metric,
    )


def run_nextstat_std_normal(
    seed: int,
    *,
    dim: int = 10,
    method: str = "nuts",
    metric: str = "diagonal",
    n_chains: int = DEFAULT_N_CHAINS,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_samples: int = DEFAULT_N_SAMPLES,
    target_accept: float = 0.8,
    max_treedepth: int | None = 10,
    sample_kwargs: dict[str, Any] | None = None,
) -> BenchResult:
    import nextstat

    model = nextstat.StdNormalModel(dim=dim)
    return _run_nextstat_model(
        model_name=f"std_normal_{dim}d",
        model=model,
        seed=seed,
        method=method,
        metric=metric,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        target_accept=target_accept,
        max_treedepth=max_treedepth,
        sample_kwargs=sample_kwargs,
    )


def run_nextstat_eight_schools(
    seed: int,
    *,
    method: str = "nuts",
    metric: str = "diagonal",
    n_chains: int = DEFAULT_N_CHAINS,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_samples: int = DEFAULT_N_SAMPLES,
    target_accept: float = 0.8,
    max_treedepth: int | None = 10,
    sample_kwargs: dict[str, Any] | None = None,
) -> BenchResult:
    import nextstat

    data = gen_eight_schools_data()
    model = nextstat.EightSchoolsModel(y=data["y"], sigma=data["sigma"])
    return _run_nextstat_model(
        model_name="eight_schools",
        model=model,
        seed=seed,
        method=method,
        metric=metric,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        target_accept=target_accept,
        max_treedepth=max_treedepth,
        sample_kwargs=sample_kwargs,
    )


def run_nextstat_glm_logistic(
    seed: int,
    *,
    n: int = 1000,
    p: int = 10,
    method: str = "nuts",
    metric: str = "diagonal",
    n_chains: int = DEFAULT_N_CHAINS,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_samples: int = DEFAULT_N_SAMPLES,
    target_accept: float = 0.8,
    max_treedepth: int | None = 10,
    sample_kwargs: dict[str, Any] | None = None,
) -> BenchResult:
    import nextstat

    data = gen_glm_logistic_data(n=n, p=p)
    model = nextstat.LogisticRegressionModel(x=data["X"], y=data["y"])
    posterior = nextstat.Posterior(model)
    posterior.set_prior_normal("intercept", 0.0, 5.0)
    for j in range(p):
        posterior.set_prior_normal(f"beta{j + 1}", 0.0, 2.5)

    return _run_nextstat_model(
        model_name="glm_logistic",
        model=posterior,
        seed=seed,
        method=method,
        metric=metric,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        target_accept=target_accept,
        max_treedepth=max_treedepth,
        sample_kwargs=sample_kwargs,
    )


def run_nextstat_funnel_ncp(
    seed: int,
    *,
    dim: int = 10,
    method: str = "nuts",
    metric: str = "diagonal",
    n_chains: int = DEFAULT_N_CHAINS,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_samples: int = DEFAULT_N_SAMPLES,
    target_accept: float = 0.8,
    max_treedepth: int | None = 10,
    sample_kwargs: dict[str, Any] | None = None,
) -> BenchResult:
    import nextstat

    model = nextstat.FunnelNcpModel(dim=dim)
    return _run_nextstat_model(
        model_name=f"funnel_ncp_{dim}d",
        model=model,
        seed=seed,
        method=method,
        metric=metric,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        target_accept=target_accept,
        max_treedepth=max_treedepth,
        sample_kwargs=sample_kwargs,
    )


def run_nextstat_funnel(
    seed: int,
    *,
    dim: int = 10,
    method: str = "nuts",
    metric: str = "diagonal",
    n_chains: int = DEFAULT_N_CHAINS,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_samples: int = DEFAULT_N_SAMPLES,
    target_accept: float = 0.8,
    max_treedepth: int | None = 10,
    sample_kwargs: dict[str, Any] | None = None,
) -> BenchResult:
    import nextstat

    model = nextstat.FunnelModel(dim=dim)
    return _run_nextstat_model(
        model_name=f"funnel_{dim}d",
        model=model,
        seed=seed,
        method=method,
        metric=metric,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        target_accept=target_accept,
        max_treedepth=max_treedepth,
        sample_kwargs=sample_kwargs,
    )


def run_nextstat_glm_poisson(
    seed: int,
    *,
    n: int = 1000,
    p: int = 10,
    method: str = "nuts",
    metric: str = "diagonal",
    n_chains: int = DEFAULT_N_CHAINS,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_samples: int = DEFAULT_N_SAMPLES,
    target_accept: float = 0.8,
    max_treedepth: int | None = 10,
    sample_kwargs: dict[str, Any] | None = None,
) -> BenchResult:
    import nextstat

    data = gen_glm_poisson_data(n=n, p=p)
    model = nextstat.PoissonRegressionModel(x=data["X"], y=data["y"])
    posterior = nextstat.Posterior(model)
    posterior.set_prior_normal("intercept", 0.0, 5.0)
    for j in range(p):
        posterior.set_prior_normal(f"beta{j + 1}", 0.0, 2.5)

    return _run_nextstat_model(
        model_name="glm_poisson",
        model=posterior,
        seed=seed,
        method=method,
        metric=metric,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        target_accept=target_accept,
        max_treedepth=max_treedepth,
        sample_kwargs=sample_kwargs,
    )


def run_nextstat_glm_negbin(
    seed: int,
    *,
    n: int = 1000,
    p: int = 10,
    method: str = "nuts",
    metric: str = "diagonal",
    n_chains: int = DEFAULT_N_CHAINS,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_samples: int = DEFAULT_N_SAMPLES,
    target_accept: float = 0.8,
    max_treedepth: int | None = 10,
    sample_kwargs: dict[str, Any] | None = None,
) -> BenchResult:
    import nextstat

    data = gen_glm_negbin_data(n=n, p=p)
    model = nextstat.NegativeBinomialRegressionModel(x=data["X"], y=data["y"])
    posterior = nextstat.Posterior(model)
    posterior.set_prior_normal("intercept", 0.0, 5.0)
    for j in range(p):
        posterior.set_prior_normal(f"beta{j + 1}", 0.0, 2.5)
    posterior.set_prior_normal("log_alpha", 0.0, 1.0)

    return _run_nextstat_model(
        model_name="glm_negbin",
        model=posterior,
        seed=seed,
        method=method,
        metric=metric,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        target_accept=target_accept,
        max_treedepth=max_treedepth,
        sample_kwargs=sample_kwargs,
    )


def min_ess_per_sec(results: list[BenchResult], ess_key: str = "ess_bulk") -> float:
    per_seed: list[float] = []
    for result in results:
        ess_dict = getattr(result, ess_key, {})
        if not ess_dict or result.wall_secs <= 0:
            continue
        per_seed.append(min(ess_dict.values()) / result.wall_secs)
    return statistics.median(per_seed) if per_seed else 0.0


def min_ess_per_leapfrog(results: list[BenchResult], ess_key: str = "ess_bulk") -> float:
    per_seed: list[float] = []
    for result in results:
        ess_dict = getattr(result, ess_key, {})
        if (
            not ess_dict
            or not result.n_leapfrog_sampling_total
            or result.n_leapfrog_sampling_total <= 0
        ):
            continue
        per_seed.append(min(ess_dict.values()) / float(result.n_leapfrog_sampling_total))
    return statistics.median(per_seed) if per_seed else 0.0


def leapfrogs_per_sec(results: list[BenchResult]) -> float:
    per_seed: list[float] = []
    for result in results:
        if not result.n_leapfrog_total or result.n_leapfrog_total <= 0 or result.wall_secs <= 0:
            continue
        per_seed.append(float(result.n_leapfrog_total) / result.wall_secs)
    return statistics.median(per_seed) if per_seed else 0.0


def max_rhat(results: list[BenchResult]) -> float:
    rhs = [max(result.r_hat.values()) for result in results if result.r_hat]
    return statistics.median(rhs) if rhs else float("nan")


def median_wall(results: list[BenchResult]) -> float:
    vals = [result.wall_secs for result in results if result.wall_secs > 0]
    return statistics.median(vals) if vals else 0.0


def median_divergence_rate(results: list[BenchResult]) -> float | None:
    vals = [result.divergence_rate for result in results]
    return statistics.median(vals) if vals else None


def median_min_ebfmi(results: list[BenchResult]) -> float | None:
    per_seed = [min(result.ebfmi) for result in results if result.ebfmi]
    return statistics.median(per_seed) if per_seed else None


def median_mean_tree_depth(results: list[BenchResult]) -> float | None:
    vals = [result.mean_tree_depth for result in results if result.mean_tree_depth is not None]
    return statistics.median(vals) if vals else None


def median_mean_accept_prob(results: list[BenchResult]) -> float | None:
    vals = [result.mean_accept_prob for result in results if result.mean_accept_prob is not None]
    return statistics.median(vals) if vals else None
