#!/usr/bin/env python3
"""Deterministic realistic benchmark matrix for naive vs CUPED vs CURE on ads scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    REPO_ROOT / "bench_results" / "ads_variance_reduction_benchmark" / "ads_variance_reduction_benchmark.json"
)
DEFAULT_MARKDOWN_OUT = DEFAULT_OUT.with_suffix(".md")
DEFAULT_WORK_ROOT = DEFAULT_OUT.parent / "work"
DEFAULT_SCENARIO_MANIFEST = (
    REPO_ROOT / "tests" / "fixtures" / "variance_reduction_benchmark" / "scenario_matrix.json"
)
SCHEMA_VERSION = "nextstat.ads_variance_reduction_benchmark_result.v1"
SUITE = "ads_variance_reduction_matrix"
HOST_POLICY = "nextstat-bench"
METHOD_ORDER = ["naive", "cuped", "cure"]


@dataclass(frozen=True)
class CovariateSpec:
    name: str
    timing: str
    source_dataset: str | None


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    description: str
    metric_type: str
    n_per_arm: int
    seed: int
    primary_covariate: str
    sparsity_regime: str
    collinearity_regime: str
    covariates: tuple[CovariateSpec, ...]


@dataclass
class ScenarioData:
    spec: ScenarioSpec
    control_outcomes: list[float]
    variant_outcomes: list[float]
    control_covariates: list[list[float]]
    variant_covariates: list[list[float]]
    pooled_outcome_zero_fraction: float
    pooled_covariate_zero_fraction: float
    primary_covariate_index: int


class DeterministicRng:
    """Stable PRNG so scenario generation does not depend on Python's random implementation."""

    def __init__(self, seed: int) -> None:
        self._state = seed & ((1 << 64) - 1)
        self._normal_cache: float | None = None

    def _next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        value = self._state
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
        value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
        return value ^ (value >> 31)

    def uniform(self) -> float:
        return ((self._next_u64() >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def normal(self) -> float:
        if self._normal_cache is not None:
            out = self._normal_cache
            self._normal_cache = None
            return out
        u1 = max(self.uniform(), 1e-15)
        u2 = self.uniform()
        radius = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        z0 = radius * math.cos(theta)
        z1 = radius * math.sin(theta)
        self._normal_cache = z1
        return z0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_model() -> str:
    linux_cpuinfo = Path("/proc/cpuinfo")
    if linux_cpuinfo.exists():
        for line in linux_cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                _, _, value = line.partition(":")
                return value.strip()
    machine = platform.processor().strip() or platform.machine().strip()
    return machine or "unknown"


def _git_commit() -> str | None:
    env_value = os.environ.get("NEXTSTAT_BENCH_GIT_COMMIT", "").strip()
    if env_value:
        return env_value
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def _build_profile(binary: Path) -> str:
    parts = set(binary.parts)
    if "release" in parts:
        return "release"
    if "debug" in parts:
        return "debug"
    return "unknown"


def _timing_doc(samples: list[float]) -> dict[str, Any]:
    if not samples:
        raise RuntimeError("expected at least one timing sample")
    return {
        "min_s": round(min(samples), 6),
        "median_s": round(statistics.median(samples), 6),
        "max_s": round(max(samples), 6),
        "samples_s": [round(sample, 6) for sample in samples],
    }


def _run(cmd: list[str], *, cwd: Path) -> tuple[float, subprocess.CompletedProcess[str]]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return elapsed, proc


_NEXTSTAT_CACHE: Any | None = None


def _load_nextstat():
    global _NEXTSTAT_CACHE
    if _NEXTSTAT_CACHE is not None:
        return _NEXTSTAT_CACHE
    try:
        import nextstat  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Unable to import nextstat. Run `maturin develop -m bindings/ns-py/Cargo.toml` "
            "or set PYTHONPATH=bindings/ns-py/python before invoking the benchmark harness."
        ) from exc
    _NEXTSTAT_CACHE = nextstat
    return nextstat


def _configure_determinism(nextstat: Any, deterministic: bool) -> None:
    if not deterministic:
        return
    for key in (
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    if hasattr(nextstat, "set_threads"):
        nextstat.set_threads(1)
    if hasattr(nextstat, "set_eval_mode"):
        try:
            nextstat.set_eval_mode("parity")
        except Exception:
            pass


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values))


def _sample_variance(values: list[float], mean_value: float) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    return sum((value - mean_value) ** 2 for value in values) / float(n - 1)


def _zero_fraction(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if abs(value) <= 1e-15) / float(len(values))


def _flatten(matrix: list[list[float]]) -> list[float]:
    return [value for row in matrix for value in row]


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def _positive(x: float, floor: float = 0.0) -> float:
    return x if x > floor else floor


def _load_scenarios(path: Path) -> list[ScenarioSpec]:
    raw = _load_json(path)
    if raw.get("schema_version") not in (
        "nextstat.ads_variance_reduction_scenario_manifest.v1",
        "nextstat.ads_variance_reduction_matrix.v1",
    ):
        raise RuntimeError(f"unexpected matrix schema: {raw.get('schema_version')}")
    specs: list[ScenarioSpec] = []
    for item in raw.get("scenarios", []):
        if not isinstance(item, dict):
            raise RuntimeError("scenario manifest contains a non-object scenario")
        covariates = tuple(
            CovariateSpec(
                name=str(cov["name"]),
                timing=str(cov["timing"]),
                source_dataset=None if cov.get("source_dataset") is None else str(cov["source_dataset"]),
            )
            for cov in item.get("covariates", [])
        )
        specs.append(
            ScenarioSpec(
                scenario_id=str(item["scenario_id"]),
                description=str(item["description"]),
                metric_type=str(item["metric_type"]),
                n_per_arm=int(item["n_per_arm"]),
                seed=int(item["seed"]),
                primary_covariate=str(item["primary_covariate"]),
                sparsity_regime=str(item["sparsity_regime"]),
                collinearity_regime=str(item["collinearity_regime"]),
                covariates=covariates,
            )
        )
    return specs


def _make_scenario_data(spec: ScenarioSpec) -> ScenarioData:
    if spec.scenario_id == "revenue_dense_signal":
        return _generate_revenue_dense_signal(spec)
    if spec.scenario_id == "ratio_style_efficiency":
        return _generate_ratio_style_efficiency(spec)
    if spec.scenario_id == "sparse_new_user_conversion":
        return _generate_sparse_new_user_conversion(spec)
    if spec.scenario_id == "collinear_account_history":
        return _generate_collinear_account_history(spec)
    raise RuntimeError(f"unsupported scenario_id: {spec.scenario_id}")


def _finalize_scenario_data(
    spec: ScenarioSpec,
    control_outcomes: list[float],
    variant_outcomes: list[float],
    control_covariates: list[list[float]],
    variant_covariates: list[list[float]],
) -> ScenarioData:
    pooled_outcomes = control_outcomes + variant_outcomes
    pooled_covariates = _flatten(control_covariates) + _flatten(variant_covariates)
    names = [cov.name for cov in spec.covariates]
    primary_index = names.index(spec.primary_covariate)
    return ScenarioData(
        spec=spec,
        control_outcomes=control_outcomes,
        variant_outcomes=variant_outcomes,
        control_covariates=control_covariates,
        variant_covariates=variant_covariates,
        pooled_outcome_zero_fraction=_zero_fraction(pooled_outcomes),
        pooled_covariate_zero_fraction=_zero_fraction(pooled_covariates),
        primary_covariate_index=primary_index,
    )


def _generate_revenue_dense_signal(spec: ScenarioSpec) -> ScenarioData:
    rng = DeterministicRng(spec.seed)
    control_outcomes: list[float] = []
    variant_outcomes: list[float] = []
    control_covariates: list[list[float]] = []
    variant_covariates: list[list[float]] = []
    for arm_flag, outcomes, covariates in (
        (0, control_outcomes, control_covariates),
        (1, variant_outcomes, variant_covariates),
    ):
        for _ in range(spec.n_per_arm):
            account_quality = rng.normal()
            auction_pressure = rng.normal()
            pre_spend = _positive(140.0 + 32.0 * account_quality + 14.0 * auction_pressure + 8.0 * rng.normal())
            pre_clicks = _positive(500.0 + 70.0 * account_quality + 48.0 * auction_pressure + 20.0 * rng.normal())
            device_quality = _positive(1.0 + 0.14 * account_quality + 0.05 * rng.normal(), 0.05)
            outcome = (
                95.0
                + 0.42 * pre_spend
                + 0.055 * pre_clicks
                + 16.0 * device_quality
                + 9.0 * rng.normal()
                + 4.0 * arm_flag
            )
            outcomes.append(round(outcome, 8))
            covariates.append(
                [
                    round(pre_spend, 8),
                    round(pre_clicks, 8),
                    round(device_quality, 8),
                ]
            )
    return _finalize_scenario_data(spec, control_outcomes, variant_outcomes, control_covariates, variant_covariates)


def _generate_ratio_style_efficiency(spec: ScenarioSpec) -> ScenarioData:
    rng = DeterministicRng(spec.seed)
    control_outcomes: list[float] = []
    variant_outcomes: list[float] = []
    control_covariates: list[list[float]] = []
    variant_covariates: list[list[float]] = []
    for arm_flag, outcomes, covariates in (
        (0, control_outcomes, control_covariates),
        (1, variant_outcomes, variant_covariates),
    ):
        for _ in range(spec.n_per_arm):
            efficiency = rng.normal()
            cost_headwind = rng.normal()
            pre_ctr = _clamp(0.028 + 0.0045 * efficiency + 0.0012 * rng.normal(), 0.003, 0.12)
            pre_cpc = _clamp(1.24 - 0.08 * cost_headwind + 0.05 * rng.normal(), 0.2, 3.5)
            pre_conversion_rate = _clamp(
                0.011 + 0.0017 * efficiency - 0.0006 * cost_headwind + 0.0005 * rng.normal(),
                0.0005,
                0.08,
            )
            outcome = _clamp(
                0.92
                + 7.0 * pre_ctr
                - 0.33 * pre_cpc
                + 11.0 * pre_conversion_rate
                + 0.03 * rng.normal()
                + 0.022 * arm_flag,
                0.05,
                4.0,
            )
            outcomes.append(round(outcome, 8))
            covariates.append(
                [
                    round(pre_ctr, 8),
                    round(pre_cpc, 8),
                    round(pre_conversion_rate, 8),
                ]
            )
    return _finalize_scenario_data(spec, control_outcomes, variant_outcomes, control_covariates, variant_covariates)


def _generate_sparse_new_user_conversion(spec: ScenarioSpec) -> ScenarioData:
    rng = DeterministicRng(spec.seed)
    control_outcomes: list[float] = []
    variant_outcomes: list[float] = []
    control_covariates: list[list[float]] = []
    variant_covariates: list[list[float]] = []
    for arm_flag, outcomes, covariates in (
        (0, control_outcomes, control_covariates),
        (1, variant_outcomes, variant_covariates),
    ):
        for _ in range(spec.n_per_arm):
            quality = rng.normal()
            sessions = 0.0
            if rng.uniform() > 0.87:
                sessions = float(max(1, int(round(2.0 + 1.8 * max(0.0, quality + 1.0) + 0.6 * rng.normal()))))
            impressions = 0.0
            if rng.uniform() > 0.80:
                impressions = float(
                    max(1, int(round(25.0 + 18.0 * max(0.0, quality + 0.8) + 6.0 * rng.normal())))
                )
            installs = 0.0
            if rng.uniform() < _sigmoid(-1.6 + 0.9 * quality):
                installs = 1.0
            account_age_days = float(max(0, int(round(45.0 + 18.0 * rng.normal() + 10.0 * max(quality, -1.0)))))
            logit = (
                -4.95
                + 0.08 * sessions
                + 0.016 * impressions
                + 1.85 * installs
                + 0.024 * account_age_days
                + 0.18 * arm_flag
                + 0.32 * quality
            )
            outcome = 1.0 if rng.uniform() < _sigmoid(logit) else 0.0
            outcomes.append(outcome)
            covariates.append(
                [
                    round(sessions, 8),
                    round(impressions, 8),
                    round(installs, 8),
                    round(account_age_days, 8),
                ]
            )
    return _finalize_scenario_data(spec, control_outcomes, variant_outcomes, control_covariates, variant_covariates)


def _generate_collinear_account_history(spec: ScenarioSpec) -> ScenarioData:
    rng = DeterministicRng(spec.seed)
    control_outcomes: list[float] = []
    variant_outcomes: list[float] = []
    control_covariates: list[list[float]] = []
    variant_covariates: list[list[float]] = []
    for arm_flag, outcomes, covariates in (
        (0, control_outcomes, control_covariates),
        (1, variant_outcomes, variant_covariates),
    ):
        for _ in range(spec.n_per_arm):
            account = rng.normal()
            seasonality = rng.normal()
            pre_spend = _positive(180.0 + 36.0 * account + 12.0 * seasonality + 6.0 * rng.normal())
            pre_spend_x2 = 2.0 * pre_spend
            pre_impressions = _positive(14.0 * pre_spend + 25.0 * rng.normal())
            pre_clicks = _positive(0.082 * pre_impressions + 2.0 * rng.normal())
            pre_budget = _positive(0.55 * pre_spend + 4.0 * seasonality + 1.5 * rng.normal())
            outcome = (
                55.0
                + 0.11 * pre_spend
                + 0.0045 * pre_impressions
                + 0.62 * pre_clicks
                + 0.13 * pre_budget
                + 8.0 * rng.normal()
                + 3.5 * arm_flag
            )
            outcomes.append(round(outcome, 8))
            covariates.append(
                [
                    round(pre_spend, 8),
                    round(pre_spend_x2, 8),
                    round(pre_impressions, 8),
                    round(pre_clicks, 8),
                    round(pre_budget, 8),
                ]
            )
    return _finalize_scenario_data(spec, control_outcomes, variant_outcomes, control_covariates, variant_covariates)


def _naive_adjust(dataset: ScenarioData) -> dict[str, Any]:
    mean_control = _mean(dataset.control_outcomes)
    mean_variant = _mean(dataset.variant_outcomes)
    original_variance = (
        _sample_variance(dataset.control_outcomes, mean_control) / float(len(dataset.control_outcomes))
        + _sample_variance(dataset.variant_outcomes, mean_variant) / float(len(dataset.variant_outcomes))
    )
    return {
        "method": "naive",
        "mean_control": round(mean_control, 8),
        "mean_variant": round(mean_variant, 8),
        "effect": round(mean_variant - mean_control, 8),
        "original_variance": round(original_variance, 12),
        "adjusted_variance": round(original_variance, 12),
        "estimated_variance": round(original_variance, 12),
        "standard_error": round(math.sqrt(original_variance), 12),
        "solver": None,
        "regression_rank": None,
        "num_covariates": 0,
        "selected_covariates": [],
        "covariate_provenance": [],
        "provenance_validated": False,
        "pre_treatment_only": True,
        "r_squared": None,
        "variance_reduction_factor": None,
        "effective_sample_multiplier": None,
        "condition_number": None,
        "ridge_lambda": None,
        "ridge_used": False,
    }


def _cuped_adjust(dataset: ScenarioData) -> dict[str, Any]:
    nextstat = _load_nextstat()
    primary = dataset.primary_covariate_index
    covariate = dataset.spec.covariates[primary]
    result = nextstat.ads.cuped_adjust(
        dataset.control_outcomes,
        [row[primary] for row in dataset.control_covariates],
        dataset.variant_outcomes,
        [row[primary] for row in dataset.variant_covariates],
        covariate_name=covariate.name,
        covariate_provenance={
            "name": covariate.name,
            "timing": covariate.timing,
            "source_dataset": covariate.source_dataset,
        },
    )
    adjusted_variance = float(result["adjusted_variance"])
    return {
        "method": "cuped",
        "mean_control": round(float(result["mean_control"]), 8),
        "mean_variant": round(float(result["mean_variant"]), 8),
        "effect": round(float(result["effect"]), 8),
        "original_variance": round(float(result["original_variance"]), 12),
        "adjusted_variance": round(adjusted_variance, 12),
        "estimated_variance": round(adjusted_variance, 12),
        "standard_error": round(math.sqrt(max(adjusted_variance, 0.0)), 12),
        "solver": str(result["solver"]),
        "regression_rank": int(result["regression_rank"]),
        "num_covariates": int(result["num_covariates"]),
        "selected_covariates": [str(value) for value in result["selected_covariates"]],
        "covariate_provenance": [
            {
                "name": str(item["name"]),
                "timing": str(item["timing"]),
                "source_dataset": None
                if item.get("source_dataset") is None
                else str(item["source_dataset"]),
            }
            for item in result["covariate_provenance"]
        ],
        "provenance_validated": bool(result["provenance_validated"]),
        "pre_treatment_only": bool(result["pre_treatment_only"]),
        "r_squared": round(float(result["r_squared"]), 8),
        "variance_reduction_factor": round(float(result["variance_reduction_factor"]), 8),
        "effective_sample_multiplier": round(float(result["effective_sample_multiplier"]), 8),
        "condition_number": None
        if result["condition_number"] is None
        else round(float(result["condition_number"]), 8),
        "ridge_lambda": None if result["ridge_lambda"] is None else round(float(result["ridge_lambda"]), 12),
        "ridge_used": str(result["solver"]) == "ridge",
    }


def _cure_adjust(dataset: ScenarioData) -> dict[str, Any]:
    nextstat = _load_nextstat()
    result = nextstat.ads.cure_adjust(
        dataset.control_outcomes,
        dataset.control_covariates,
        dataset.variant_outcomes,
        dataset.variant_covariates,
        covariate_names=[cov.name for cov in dataset.spec.covariates],
        covariate_provenance=[
            {
                "name": cov.name,
                "timing": cov.timing,
                "source_dataset": cov.source_dataset,
            }
            for cov in dataset.spec.covariates
        ],
    )
    adjusted_variance = float(result["adjusted_variance"])
    return {
        "method": "cure",
        "mean_control": round(float(result["mean_control"]), 8),
        "mean_variant": round(float(result["mean_variant"]), 8),
        "effect": round(float(result["effect"]), 8),
        "original_variance": round(float(result["original_variance"]), 12),
        "adjusted_variance": round(adjusted_variance, 12),
        "estimated_variance": round(adjusted_variance, 12),
        "standard_error": round(math.sqrt(max(adjusted_variance, 0.0)), 12),
        "solver": str(result["solver"]),
        "regression_rank": int(result["regression_rank"]),
        "num_covariates": int(result["num_covariates"]),
        "selected_covariates": [str(value) for value in result["selected_covariates"]],
        "covariate_provenance": [
            {
                "name": str(item["name"]),
                "timing": str(item["timing"]),
                "source_dataset": None
                if item.get("source_dataset") is None
                else str(item["source_dataset"]),
            }
            for item in result["covariate_provenance"]
        ],
        "provenance_validated": bool(result["provenance_validated"]),
        "pre_treatment_only": bool(result["pre_treatment_only"]),
        "r_squared": round(float(result["r_squared"]), 8),
        "variance_reduction_factor": round(float(result["variance_reduction_factor"]), 8),
        "effective_sample_multiplier": round(float(result["effective_sample_multiplier"]), 8),
        "condition_number": None
        if result["condition_number"] is None
        else round(float(result["condition_number"]), 8),
        "ridge_lambda": None if result["ridge_lambda"] is None else round(float(result["ridge_lambda"]), 12),
        "ridge_used": str(result["solver"]) == "ridge",
    }


def _benchmark_method(dataset: ScenarioData, method: str) -> dict[str, Any]:
    if method == "naive":
        base = _naive_adjust(dataset)
    elif method == "cuped":
        base = _cuped_adjust(dataset)
    elif method == "cure":
        base = _cure_adjust(dataset)
    else:
        raise RuntimeError(f"unsupported benchmark method: {method}")

    return {
        "scenario_id": dataset.spec.scenario_id,
        "scenario_description": dataset.spec.description,
        "metric_type": dataset.spec.metric_type,
        "method": method,
        "n_per_arm": dataset.spec.n_per_arm,
        "available_covariates": len(dataset.spec.covariates),
        "primary_covariate": dataset.spec.primary_covariate,
        "scenario_sparsity_regime": dataset.spec.sparsity_regime,
        "scenario_collinearity_regime": dataset.spec.collinearity_regime,
        "pooled_outcome_zero_fraction": round(dataset.pooled_outcome_zero_fraction, 8),
        "pooled_covariate_zero_fraction": round(dataset.pooled_covariate_zero_fraction, 8),
        **base,
    }


def _case_id(spec: ScenarioSpec, method: str) -> str:
    return f"python_{spec.scenario_id}_{method}"


def _run_case(
    *,
    dataset: ScenarioData,
    method: str,
    repeats: int,
    warmups: int,
) -> dict[str, Any]:
    case_id = _case_id(dataset.spec, method)
    for _ in range(warmups):
        _benchmark_method(dataset, method)

    timings: list[float] = []
    details: dict[str, Any] | None = None
    for _ in range(repeats):
        started = time.perf_counter()
        details = _benchmark_method(dataset, method)
        timings.append(time.perf_counter() - started)

    if details is None:
        raise RuntimeError(f"case {case_id} did not emit benchmark details")

    return {
        "case_id": case_id,
        "surface": "python",
        "status": "ok",
        **_timing_doc(timings),
        "details": details,
    }


def _scenario_summary(spec: ScenarioSpec, results_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def _case(method: str) -> dict[str, Any]:
        return results_by_id[_case_id(spec, method)]

    naive = _case("naive")
    cuped = _case("cuped")
    cure = _case("cure")
    naive_median = float(naive["median_s"])
    cuped_median = float(cuped["median_s"])
    cure_median = float(cure["median_s"])
    naive_variance = float(naive["details"]["estimated_variance"])
    cuped_variance = float(cuped["details"]["estimated_variance"])
    cure_variance = float(cure["details"]["estimated_variance"])
    return {
        "scenario_id": spec.scenario_id,
        "metric_type": spec.metric_type,
        "n_per_arm": spec.n_per_arm,
        "available_covariates": len(spec.covariates),
        "primary_covariate": spec.primary_covariate,
        "sparsity_regime": spec.sparsity_regime,
        "collinearity_regime": spec.collinearity_regime,
        "pooled_outcome_zero_fraction": naive["details"]["pooled_outcome_zero_fraction"],
        "pooled_covariate_zero_fraction": naive["details"]["pooled_covariate_zero_fraction"],
        "naive_median_s": round(naive_median, 6),
        "cuped_median_s": round(cuped_median, 6),
        "cure_median_s": round(cure_median, 6),
        "cuped_runtime_ratio_vs_naive": round(cuped_median / naive_median, 6),
        "cure_runtime_ratio_vs_naive": round(cure_median / naive_median, 6),
        "naive_standard_error": naive["details"]["standard_error"],
        "cuped_standard_error": cuped["details"]["standard_error"],
        "cure_standard_error": cure["details"]["standard_error"],
        "cuped_variance_ratio_vs_naive": round(cuped_variance / naive_variance, 8),
        "cure_variance_ratio_vs_naive": round(cure_variance / naive_variance, 8),
        "cuped_effective_sample_multiplier": cuped["details"]["effective_sample_multiplier"],
        "cure_effective_sample_multiplier": cure["details"]["effective_sample_multiplier"],
        "cuped_r_squared": cuped["details"]["r_squared"],
        "cure_r_squared": cure["details"]["r_squared"],
        "cuped_solver": cuped["details"]["solver"],
        "cure_solver": cure["details"]["solver"],
        "cure_ridge_lambda": cure["details"]["ridge_lambda"],
        "best_method_by_standard_error": min(
            (
                ("naive", float(naive["details"]["standard_error"])),
                ("cuped", float(cuped["details"]["standard_error"])),
                ("cure", float(cure["details"]["standard_error"])),
            ),
            key=lambda item: item[1],
        )[0],
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Ads Variance Reduction Matrix Benchmark",
        "",
        f"- schema_version: `{report['schema_version']}`",
        f"- suite: `{report['suite']}`",
        f"- smoke: `{report['meta']['smoke']}`",
        f"- deterministic: `{report['meta']['deterministic']}`",
        f"- runs: `{report['protocol']['runs']}`",
        f"- warmups: `{report['protocol']['warmups']}`",
        f"- host_policy: `{report['meta']['host_policy']}`",
        f"- hostname: `{report['host']['hostname']}`",
        "",
        "## Scenario Summary",
        "",
        "| Scenario | n/arm | Naive (s) | CUPED (s) | CURE (s) | CURE/Naive | CUPED SE | CURE SE | CURE Solver |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in report["scenario_summaries"]:
        lines.append(
            f"| `{summary['scenario_id']}` | {summary['n_per_arm']} | "
            f"{summary['naive_median_s']:.6f} | {summary['cuped_median_s']:.6f} | "
            f"{summary['cure_median_s']:.6f} | {summary['cure_runtime_ratio_vs_naive']:.3f} | "
            f"{summary['cuped_standard_error']:.6f} | {summary['cure_standard_error']:.6f} | "
            f"`{summary['cure_solver']}` |"
        )
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Method | Median (s) | Min (s) | Max (s) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for case in report["results"]:
        lines.append(
            f"| `{case['case_id']}` | `{case['details']['method']}` | "
            f"{case['median_s']:.6f} | {case['min_s']:.6f} | {case['max_s']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- slowest_case_id: `{report['derived']['slowest_case_id']}`",
            f"- slowest_median_s: `{report['derived']['slowest_median_s']:.6f}`",
            f"- ridge_case_count: `{report['derived']['ridge_case_count']}`",
            f"- ridge_scenario_ids: `{', '.join(report['derived']['ridge_scenario_ids'])}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nextstat-bin", type=Path, required=True, help="Path to the nextstat CLI binary.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON output path.")
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=DEFAULT_MARKDOWN_OUT,
        help="Markdown summary output path.",
    )
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT, help="Scratch directory.")
    parser.add_argument("--runs", type=int, default=5, help="Measured repeats per case.")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup runs per case.")
    parser.add_argument("--smoke", action="store_true", help="Use a single measured repeat and no warmups.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Pin thread-related environment variables to 1 and request parity mode when available.",
    )
    parser.add_argument(
        "--scenario-manifest",
        type=Path,
        default=DEFAULT_SCENARIO_MANIFEST,
        help="Scenario manifest JSON path.",
    )
    parser.add_argument("--matrix", dest="scenario_manifest", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    nextstat_bin = args.nextstat_bin.resolve()
    if not nextstat_bin.exists():
        raise SystemExit(f"nextstat binary does not exist: {nextstat_bin}")

    runs = 1 if args.smoke else args.runs
    warmups = 0 if args.smoke else args.warmups
    if runs < 1:
        raise SystemExit("--runs must be >= 1")
    if warmups < 0:
        raise SystemExit("--warmups must be >= 0")

    args.work_root.mkdir(parents=True, exist_ok=True)

    _, version_proc = _run([str(nextstat_bin), "--version"], cwd=REPO_ROOT)
    version = version_proc.stdout.strip() or "unknown"

    nextstat = _load_nextstat()
    _configure_determinism(nextstat, args.deterministic)
    nextstat_version = getattr(nextstat, "__version__", "unknown")

    scenario_manifest_path = args.scenario_manifest.resolve()
    scenarios = _load_scenarios(scenario_manifest_path)
    datasets = [_make_scenario_data(spec) for spec in scenarios]

    results: list[dict[str, Any]] = []
    for dataset in datasets:
        for method in METHOD_ORDER:
            results.append(
                _run_case(
                    dataset=dataset,
                    method=method,
                    repeats=runs,
                    warmups=warmups,
                )
            )

    results_by_id = {str(item["case_id"]): item for item in results}
    scenario_summaries = [_scenario_summary(spec, results_by_id) for spec in scenarios]
    slowest = max(results, key=lambda case: float(case["median_s"]))
    ridge_case_ids = [
        str(case["case_id"])
        for case in results
        if isinstance(case.get("details"), dict) and bool(case["details"].get("ridge_used"))
    ]
    ridge_scenario_ids = sorted(
        {
            str(case["details"]["scenario_id"])
            for case in results
            if isinstance(case.get("details"), dict) and bool(case["details"].get("ridge_used"))
        }
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE,
        "meta": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runner": "scripts/benchmarks/bench_ads_variance_reduction_matrix.py",
            "host_policy": HOST_POLICY,
            "smoke": bool(args.smoke),
            "deterministic": bool(args.deterministic),
            "git_commit": _git_commit(),
            "scenario_manifest": str(scenario_manifest_path),
        },
        "protocol": {
            "runs": int(runs),
            "warmups": int(warmups),
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or platform.machine(),
            "cpu_model": _cpu_model(),
            "python_version": sys.version.split()[0],
        },
        "binary": {
            "path": str(nextstat_bin),
            "version": version,
            "sha256": _sha256_file(nextstat_bin),
            "build_profile": _build_profile(nextstat_bin),
        },
        "python": {
            "nextstat_version": str(nextstat_version),
        },
        "results": results,
        "scenario_summaries": scenario_summaries,
        "derived": {
            "all_cases_ok": all(case["status"] == "ok" for case in results),
            "case_count": len(results),
            "scenario_count": len(scenarios),
            "method_count": len(METHOD_ORDER),
            "python_case_count": len(results),
            "slowest_case_id": slowest["case_id"],
            "slowest_median_s": round(float(slowest["median_s"]), 6),
            "ridge_case_count": len(ridge_case_ids),
            "ridge_case_ids": ridge_case_ids,
            "ridge_scenario_count": len(ridge_scenario_ids),
            "ridge_scenario_ids": ridge_scenario_ids,
        },
    }

    _write_json(args.out, report)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(_render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
