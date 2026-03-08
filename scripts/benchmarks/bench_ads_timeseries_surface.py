#!/usr/bin/env python3
"""Deterministic benchmark harness for the ads + weekly timeseries stable surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "bench_results" / "ads_timeseries_surface" / "summary.json"
DEFAULT_MARKDOWN_OUT = DEFAULT_OUT.with_suffix(".md")
DEFAULT_WORK_ROOT = DEFAULT_OUT.parent / "work"
LOCAL_LEVEL_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "kalman_local_level_weekly.json"
LOCAL_LINEAR_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "kalman_local_linear_trend_weekly.json"
SCHEMA_VERSION = "nextstat.ads_timeseries_benchmark_result.v1"
SUITE = "ads_timeseries_surface"
HOST_POLICY = "nextstat-bench"
CASE_ORDER = [
    "python_beta_binomial_fit_from_counts",
    "python_delay_correction_fit_from_lag_buckets",
    "python_cuped_adjust",
    "python_cure_adjust",
    "python_response_curve_helpers",
    "python_kalman_local_level_weekly_filter",
    "python_kalman_local_linear_trend_weekly_filter",
    "cli_kalman_local_level_weekly_filter",
    "cli_kalman_local_linear_trend_weekly_filter",
]


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


def _load_nextstat():
    try:
        import nextstat  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Unable to import nextstat. Run `maturin develop -m bindings/ns-py/Cargo.toml` "
            "or set PYTHONPATH=bindings/ns-py/python before invoking the benchmark harness."
        ) from exc
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


CaseFn = Callable[[Path, Path, bool], dict[str, Any]]


def _python_beta_binomial_fit_from_counts(_: Path, __: Path, deterministic: bool) -> dict[str, Any]:
    nextstat = _load_nextstat()
    _configure_determinism(nextstat, deterministic)
    model = nextstat.ads.BetaBinomialModel.fit_from_counts(
        [12, 19, 10, 24, 31, 15, 18, 22],
        [900, 1200, 800, 1100, 1400, 950, 1000, 1250],
    )
    posterior = model.posterior(17, 110)
    return {
        "alpha": round(float(model.alpha), 6),
        "beta": round(float(model.beta), 6),
        "mean": round(float(model.mean()), 8),
        "overdispersion": round(float(model.overdispersion()), 8),
        "posterior_alpha": round(float(posterior.alpha), 6),
        "posterior_beta": round(float(posterior.beta), 6),
    }


def _python_delay_correction_fit_from_lag_buckets(_: Path, __: Path, deterministic: bool) -> dict[str, Any]:
    nextstat = _load_nextstat()
    _configure_determinism(nextstat, deterministic)
    model = nextstat.ads.DelayCorrectionModel.fit_from_lag_buckets(
        [(1.0, 296), (2.0, 193), (4.0, 252), (7.0, 173), (14.0, 80)]
    )
    corrected, uncertainty = model.correct(42.0, 3.0)
    return {
        "lambda": round(float(model.lambda_), 8),
        "lambda_se": round(float(model.lambda_se or 0.0), 8),
        "observed_fraction_3d": round(float(model.observed_fraction(3.0)), 8),
        "corrected_3d": round(float(corrected), 6),
        "uncertainty_3d": round(float(uncertainty), 6),
    }


def _python_cuped_adjust(_: Path, __: Path, deterministic: bool) -> dict[str, Any]:
    nextstat = _load_nextstat()
    _configure_determinism(nextstat, deterministic)
    result = nextstat.ads.cuped_adjust(
        [10.0, 12.0, 11.0, 13.0, 9.0, 14.0],
        [9.5, 11.0, 10.0, 12.0, 8.5, 13.0],
        [11.0, 13.0, 12.0, 14.0, 10.0, 15.0],
        [10.5, 12.0, 11.0, 13.0, 9.5, 14.0],
        covariate_name="pre_clicks",
        covariate_provenance={
            "name": "pre_clicks",
            "timing": "pre_treatment",
            "source_dataset": "ads_preperiod_clicks",
        },
    )
    return {
        "method": str(result["method"]),
        "solver": str(result["solver"]),
        "num_covariates": int(result["num_covariates"]),
        "r_squared": round(float(result["r_squared"]), 8),
        "variance_reduction_factor": round(float(result["variance_reduction_factor"]), 8),
        "effective_sample_multiplier": round(float(result["effective_sample_multiplier"]), 8),
        "effect": round(float(result["effect"]), 8),
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
        "condition_number": None
        if result["condition_number"] is None
        else round(float(result["condition_number"]), 8),
        "ridge_lambda": None if result["ridge_lambda"] is None else round(float(result["ridge_lambda"]), 12),
        "pre_treatment_only": bool(result["pre_treatment_only"]),
    }


def _python_cure_adjust(_: Path, __: Path, deterministic: bool) -> dict[str, Any]:
    nextstat = _load_nextstat()
    _configure_determinism(nextstat, deterministic)
    control_covariates = [
        [100.0, 200.0],
        [120.0, 240.0],
        [110.0, 220.0],
        [130.0, 260.0],
        [90.0, 180.0],
        [140.0, 280.0],
    ]
    variant_covariates = [
        [102.0, 204.0],
        [122.0, 244.0],
        [112.0, 224.0],
        [132.0, 264.0],
        [92.0, 184.0],
        [142.0, 284.0],
    ]
    result = nextstat.ads.cure_adjust(
        [10.0, 12.0, 11.0, 13.0, 9.0, 14.0],
        control_covariates,
        [11.0, 13.0, 12.0, 14.0, 10.0, 15.0],
        variant_covariates,
        covariate_names=["pre_clicks", "pre_impressions"],
        covariate_provenance=[
            {
                "name": "pre_clicks",
                "timing": "pre_treatment",
                "source_dataset": "ads_preperiod_clicks",
            },
            {
                "name": "pre_impressions",
                "timing": "pre_treatment",
                "source_dataset": "ads_preperiod_impressions",
            },
        ],
    )
    return {
        "method": str(result["method"]),
        "solver": str(result["solver"]),
        "num_covariates": int(result["num_covariates"]),
        "r_squared": round(float(result["r_squared"]), 8),
        "variance_reduction_factor": round(float(result["variance_reduction_factor"]), 8),
        "effective_sample_multiplier": round(float(result["effective_sample_multiplier"]), 8),
        "effect": round(float(result["effect"]), 8),
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
        "condition_number": None
        if result["condition_number"] is None
        else round(float(result["condition_number"]), 8),
        "ridge_lambda": None if result["ridge_lambda"] is None else round(float(result["ridge_lambda"]), 12),
        "pre_treatment_only": bool(result["pre_treatment_only"]),
    }


def _python_response_curve_helpers(_: Path, __: Path, deterministic: bool) -> dict[str, Any]:
    nextstat = _load_nextstat()
    _configure_determinism(nextstat, deterministic)
    low = float(nextstat.ads.hill(10.0, 50.0, 1.2))
    high = float(nextstat.ads.hill(100.0, 50.0, 1.2))
    transformed = [float(value) for value in nextstat.ads.adstock_geometric([100.0, 0.0, 0.0], 0.5)]
    return {
        "hill_low": round(low, 8),
        "hill_high": round(high, 8),
        "adstock_last": round(transformed[-1], 8),
        "adstock_len": len(transformed),
    }


def _python_kalman_local_level_weekly_filter(_: Path, __: Path, deterministic: bool) -> dict[str, Any]:
    nextstat = _load_nextstat()
    _configure_determinism(nextstat, deterministic)
    fixture = _load_json(LOCAL_LEVEL_FIXTURE)
    spec = fixture["local_level_weekly"]
    model = nextstat.timeseries.local_level_weekly_model(
        q_level=spec["q_level"],
        q_weekly=spec["q_weekly"],
        r=spec["r"],
        level0=spec["level0"],
        p0_level=spec["p0_level"],
        p0_weekly=spec["p0_weekly"],
    )
    result = nextstat.kalman_filter(model, fixture["ys"])
    return {
        "log_likelihood": round(float(result["log_likelihood"]), 6),
        "steps": len(result["filtered_means"]),
        "state_dim": len(result["filtered_means"][0]),
    }


def _python_kalman_local_linear_trend_weekly_filter(_: Path, __: Path, deterministic: bool) -> dict[str, Any]:
    nextstat = _load_nextstat()
    _configure_determinism(nextstat, deterministic)
    fixture = _load_json(LOCAL_LINEAR_FIXTURE)
    spec = fixture["local_linear_trend_weekly"]
    model = nextstat.timeseries.local_linear_trend_weekly_model(
        q_level=spec["q_level"],
        q_slope=spec["q_slope"],
        q_weekly=spec["q_weekly"],
        r=spec["r"],
        level0=spec["level0"],
        slope0=spec["slope0"],
        p0_level=spec["p0_level"],
        p0_slope=spec["p0_slope"],
        p0_weekly=spec["p0_weekly"],
    )
    result = nextstat.kalman_filter(model, fixture["ys"])
    return {
        "log_likelihood": round(float(result["log_likelihood"]), 6),
        "steps": len(result["filtered_means"]),
        "state_dim": len(result["filtered_means"][0]),
    }


def _cli_kalman_filter_case(case_root: Path, nextstat_bin: Path, fixture: Path) -> dict[str, Any]:
    output_path = case_root / "result.json"
    _, proc = _run(
        [
            str(nextstat_bin),
            "timeseries",
            "kalman-filter",
            "--input",
            str(fixture),
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
    )
    if proc.stdout:
        pass
    result = _load_json(output_path)
    return {
        "log_likelihood": round(float(result["log_likelihood"]), 6),
        "steps": len(result["filtered_means"]),
        "state_dim": len(result["filtered_means"][0]),
    }


def _cli_kalman_local_level_weekly_filter(case_root: Path, nextstat_bin: Path, _: bool) -> dict[str, Any]:
    return _cli_kalman_filter_case(case_root, nextstat_bin, LOCAL_LEVEL_FIXTURE)


def _cli_kalman_local_linear_trend_weekly_filter(case_root: Path, nextstat_bin: Path, _: bool) -> dict[str, Any]:
    return _cli_kalman_filter_case(case_root, nextstat_bin, LOCAL_LINEAR_FIXTURE)


CASES: list[dict[str, Any]] = [
    {
        "case_id": "python_beta_binomial_fit_from_counts",
        "surface": "python",
        "function": _python_beta_binomial_fit_from_counts,
    },
    {
        "case_id": "python_delay_correction_fit_from_lag_buckets",
        "surface": "python",
        "function": _python_delay_correction_fit_from_lag_buckets,
    },
    {
        "case_id": "python_cuped_adjust",
        "surface": "python",
        "function": _python_cuped_adjust,
    },
    {
        "case_id": "python_cure_adjust",
        "surface": "python",
        "function": _python_cure_adjust,
    },
    {
        "case_id": "python_response_curve_helpers",
        "surface": "python",
        "function": _python_response_curve_helpers,
    },
    {
        "case_id": "python_kalman_local_level_weekly_filter",
        "surface": "python",
        "function": _python_kalman_local_level_weekly_filter,
    },
    {
        "case_id": "python_kalman_local_linear_trend_weekly_filter",
        "surface": "python",
        "function": _python_kalman_local_linear_trend_weekly_filter,
    },
    {
        "case_id": "cli_kalman_local_level_weekly_filter",
        "surface": "cli",
        "function": _cli_kalman_local_level_weekly_filter,
    },
    {
        "case_id": "cli_kalman_local_linear_trend_weekly_filter",
        "surface": "cli",
        "function": _cli_kalman_local_linear_trend_weekly_filter,
    },
]


def _run_case(
    case: dict[str, Any],
    *,
    nextstat_bin: Path,
    work_root: Path,
    repeats: int,
    warmups: int,
    deterministic: bool,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    case_root = work_root / case_id
    case_root.mkdir(parents=True, exist_ok=True)
    fn = case["function"]

    for _ in range(warmups):
        fn(case_root, nextstat_bin, deterministic)

    timings: list[float] = []
    details: dict[str, Any] | None = None
    for _ in range(repeats):
        started = time.perf_counter()
        details = fn(case_root, nextstat_bin, deterministic)
        timings.append(time.perf_counter() - started)

    if details is None:
        raise RuntimeError(f"case {case_id} did not emit benchmark details")

    result = {
        "case_id": case_id,
        "surface": case["surface"],
        "status": "ok",
        **_timing_doc(timings),
        "details": details,
    }
    return result


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Ads + Time Series Stable Surface Benchmark",
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
        "| Case | Surface | Median (s) | Min (s) | Max (s) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for case in report["results"]:
        lines.append(
            f"| `{case['case_id']}` | `{case['surface']}` | "
            f"{case['median_s']:.6f} | {case['min_s']:.6f} | {case['max_s']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- slowest_case_id: `{report['derived']['slowest_case_id']}`",
            f"- slowest_median_s: `{report['derived']['slowest_median_s']:.6f}`",
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

    results = [
        _run_case(
            case,
            nextstat_bin=nextstat_bin,
            work_root=args.work_root,
            repeats=runs,
            warmups=warmups,
            deterministic=args.deterministic,
        )
        for case in CASES
    ]

    slowest = max(results, key=lambda case: float(case["median_s"]))
    report = {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE,
        "meta": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runner": "scripts/benchmarks/bench_ads_timeseries_surface.py",
            "host_policy": HOST_POLICY,
            "smoke": bool(args.smoke),
            "deterministic": bool(args.deterministic),
            "git_commit": _git_commit(),
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
        "derived": {
            "all_cases_ok": all(case["status"] == "ok" for case in results),
            "case_count": len(results),
            "python_case_count": sum(case["surface"] == "python" for case in results),
            "cli_case_count": sum(case["surface"] == "cli" for case in results),
            "slowest_case_id": slowest["case_id"],
            "slowest_median_s": round(float(slowest["median_s"]), 6),
        },
    }

    _write_json(args.out, report)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(_render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
