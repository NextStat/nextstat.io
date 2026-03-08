#!/usr/bin/env python3
"""Deterministic benchmark gate for Bayesian design report packaging."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_BENCH_SCRIPTS = REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "scripts"
if str(REPO_ROOT / "bindings" / "ns-py" / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "bindings" / "ns-py" / "python"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PUBLIC_BENCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PUBLIC_BENCH_SCRIPTS))

from bench_env import collect_environment  # type: ignore

import nextstat.audit as ns_audit
import nextstat.bayes_design as ns_bayes_design

from scripts.bayesian_design_report_bundle_performance_budget import (
    load_bayesian_design_report_bundle_performance_budget,
)


DEFAULT_OUT = REPO_ROOT / "bench_results" / "bayesian_design_report_bundle" / "summary.json"
DEFAULT_WORK_ROOT = DEFAULT_OUT.parent / "work"
REQUIRED_ARTIFACTS = (
    "meta.json",
    "manifest.json",
    "inputs/input.json",
    "outputs/design_report.md",
    "outputs/design_spec.json",
    "outputs/current_analysis.json",
    "outputs/operating_characteristics.json",
    "outputs/posterior_predictive.json",
    "outputs/prior_sensitivity.json",
    "outputs/provenance.json",
)


def _timing_doc(per_run_s: list[float]) -> dict[str, Any]:
    if not per_run_s:
        raise RuntimeError("expected at least one timing sample")
    return {
        "repeat": len(per_run_s),
        "policy": "min",
        "per_run_s": [round(value, 6) for value in per_run_s],
        "best_s": round(min(per_run_s), 6),
        "median_s": round(statistics.median(per_run_s), 6),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _allocate_session_root(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if not any(root.iterdir()):
        return root
    idx = 1
    while True:
        candidate = root / f"run_{idx:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        idx += 1


def _beta_small_spec() -> dict[str, Any]:
    return {
        "schema_version": "nextstat_beta_binomial_design_v0",
        "design_id": "bundle_beta_small",
        "control_prior": {"alpha": 1.0, "beta": 1.0},
        "treatment_prior": {"alpha": 1.0, "beta": 1.0},
        "looks": [
            {"id": "interim", "n_control": 20, "n_treatment": 20},
            {"id": "final", "n_control": 40, "n_treatment": 40},
        ],
        "decision_rules": {
            "success": {"posterior_probability_threshold": 0.95, "treatment_effect_margin": 0.0},
            "futility": {"posterior_probability_threshold": 0.20, "treatment_effect_margin": 0.0},
        },
        "analysis": {"credible_interval_level": 0.95},
        "simulation": {
            "n_replicates": 16,
            "seed": 123,
            "scenarios": [
                {"id": "null", "p_control": 0.40, "p_treatment": 0.40},
                {"id": "alt", "p_control": 0.40, "p_treatment": 0.60},
            ],
        },
    }


def _beta_large_spec() -> dict[str, Any]:
    looks = []
    for idx, total in enumerate((24, 48, 72, 96, 120, 144), start=1):
        looks.append({"id": f"look_{idx}", "n_control": total, "n_treatment": total})
    scenarios = []
    for idx, effect in enumerate((0.00, 0.03, 0.05, 0.08, 0.10, 0.13, 0.15, 0.18), start=1):
        scenarios.append(
            {
                "id": f"scenario_{idx:02d}",
                "p_control": 0.35 + (0.01 if idx % 2 == 0 else 0.0),
                "p_treatment": 0.35 + effect,
            }
        )
    return {
        "schema_version": "nextstat_beta_binomial_design_v0",
        "design_id": "bundle_beta_large",
        "control_prior": {"alpha": 1.0, "beta": 1.0},
        "treatment_prior": {"alpha": 1.0, "beta": 1.0},
        "looks": looks,
        "decision_rules": {
            "success": {"posterior_probability_threshold": 0.975, "treatment_effect_margin": 0.0},
            "futility": {"posterior_probability_threshold": 0.10, "treatment_effect_margin": 0.0},
        },
        "analysis": {"credible_interval_level": 0.95},
        "simulation": {
            "n_replicates": 12,
            "seed": 777,
            "scenarios": scenarios,
        },
    }


def _beta_small_campaign() -> dict[str, Any]:
    return {
        "schema_version": "nextstat_beta_binomial_prior_sensitivity_campaign_v0",
        "variants": [
            {
                "id": "skeptical",
                "control_prior": {"alpha": 1.0, "beta": 1.0},
                "treatment_prior": {"alpha": 1.0, "beta": 8.0},
            },
            {
                "id": "enthusiastic",
                "control_prior": {"alpha": 1.0, "beta": 1.0},
                "treatment_prior": {"alpha": 8.0, "beta": 1.0},
            },
        ],
    }


def _beta_large_campaign() -> dict[str, Any]:
    variants = []
    for idx, alpha in enumerate((1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0), start=1):
        variants.append(
            {
                "id": f"variant_{idx:02d}",
                "control_prior": {"alpha": 1.0, "beta": 1.0},
                "treatment_prior": {"alpha": alpha, "beta": max(0.75, 7.5 - alpha)},
            }
        )
    return {
        "schema_version": "nextstat_beta_binomial_prior_sensitivity_campaign_v0",
        "variants": variants,
    }


def _normal_small_spec() -> dict[str, Any]:
    return {
        "schema_version": "nextstat_normal_normal_design_v0",
        "design_id": "bundle_normal_small",
        "control_prior": {"mean": 0.0, "sd": 10.0},
        "treatment_prior": {"mean": 0.0, "sd": 10.0},
        "likelihood": {"known_sd_control": 1.0, "known_sd_treatment": 1.0},
        "looks": [
            {"id": "interim", "n_control": 10, "n_treatment": 10},
            {"id": "final", "n_control": 20, "n_treatment": 20},
        ],
        "decision_rules": {
            "success": {"posterior_probability_threshold": 0.975, "treatment_effect_margin": 0.0},
            "futility": {"posterior_probability_threshold": 0.10, "treatment_effect_margin": 0.0},
        },
        "analysis": {"credible_interval_level": 0.95},
        "simulation": {
            "n_replicates": 16,
            "seed": 456,
            "scenarios": [
                {"id": "null", "mean_control": 0.0, "mean_treatment": 0.0},
                {"id": "alt", "mean_control": 0.0, "mean_treatment": 0.75},
            ],
        },
    }


def _normal_large_spec() -> dict[str, Any]:
    looks = []
    for idx, total in enumerate((16, 32, 48, 64, 80, 96), start=1):
        looks.append({"id": f"look_{idx}", "n_control": total, "n_treatment": total})
    scenarios = []
    for idx, effect in enumerate((0.00, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05), start=1):
        scenarios.append(
            {
                "id": f"scenario_{idx:02d}",
                "mean_control": 0.0,
                "mean_treatment": effect,
            }
        )
    return {
        "schema_version": "nextstat_normal_normal_design_v0",
        "design_id": "bundle_normal_large",
        "control_prior": {"mean": 0.0, "sd": 10.0},
        "treatment_prior": {"mean": 0.0, "sd": 10.0},
        "likelihood": {"known_sd_control": 1.0, "known_sd_treatment": 1.0},
        "looks": looks,
        "decision_rules": {
            "success": {"posterior_probability_threshold": 0.99, "treatment_effect_margin": 0.0},
            "futility": {"posterior_probability_threshold": 0.05, "treatment_effect_margin": 0.0},
        },
        "analysis": {"credible_interval_level": 0.95},
        "simulation": {
            "n_replicates": 12,
            "seed": 888,
            "scenarios": scenarios,
        },
    }


def _normal_small_campaign() -> dict[str, Any]:
    return {
        "schema_version": "nextstat_normal_normal_prior_sensitivity_campaign_v0",
        "variants": [
            {
                "id": "skeptical",
                "control_prior": {"mean": 0.0, "sd": 10.0},
                "treatment_prior": {"mean": -1.0, "sd": 0.2},
            },
            {
                "id": "enthusiastic",
                "control_prior": {"mean": 0.0, "sd": 10.0},
                "treatment_prior": {"mean": 1.0, "sd": 0.2},
            },
        ],
    }


def _normal_large_campaign() -> dict[str, Any]:
    variants = []
    means = (-1.5, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 1.5)
    sds = (0.25, 0.3, 0.4, 0.6, 0.6, 0.4, 0.3, 0.25)
    for idx, (mean, sd) in enumerate(zip(means, sds, strict=True), start=1):
        variants.append(
            {
                "id": f"variant_{idx:02d}",
                "control_prior": {"mean": 0.0, "sd": 10.0},
                "treatment_prior": {"mean": mean, "sd": sd},
            }
        )
    return {
        "schema_version": "nextstat_normal_normal_prior_sensitivity_campaign_v0",
        "variants": variants,
    }


def _prepare_reports(work_root: Path) -> dict[str, dict[str, Any]]:
    reports_dir = work_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        "beta_small": {
            "design_family": "beta_binomial",
            "report_scale": "small",
            "builder": lambda: ns_bayes_design.build_beta_binomial_design_report(
                _beta_small_spec(),
                {"look_id": "interim", "control_successes": 8, "treatment_successes": 11},
                _beta_small_campaign(),
            ),
            "writer": ns_bayes_design.write_beta_binomial_design_report_bundle,
        },
        "beta_large": {
            "design_family": "beta_binomial",
            "report_scale": "large",
            "builder": lambda: ns_bayes_design.build_beta_binomial_design_report(
                _beta_large_spec(),
                {"look_id": "look_4", "control_successes": 33, "treatment_successes": 46},
                _beta_large_campaign(),
            ),
            "writer": ns_bayes_design.write_beta_binomial_design_report_bundle,
        },
        "normal_small": {
            "design_family": "normal_normal",
            "report_scale": "small",
            "builder": lambda: ns_bayes_design.build_normal_normal_design_report(
                _normal_small_spec(),
                {"look_id": "interim", "control_sample_mean": 0.1, "treatment_sample_mean": 0.8},
                _normal_small_campaign(),
            ),
            "writer": ns_bayes_design.write_normal_normal_design_report_bundle,
        },
        "normal_large": {
            "design_family": "normal_normal",
            "report_scale": "large",
            "builder": lambda: ns_bayes_design.build_normal_normal_design_report(
                _normal_large_spec(),
                {"look_id": "look_4", "control_sample_mean": 0.1, "treatment_sample_mean": 0.95},
                _normal_large_campaign(),
            ),
            "writer": ns_bayes_design.write_normal_normal_design_report_bundle,
        },
    }

    prepared: dict[str, dict[str, Any]] = {}
    for case_id, case in cases.items():
        report = case["builder"]()
        report_path = reports_dir / f"{case_id}_report.json"
        _write_json(report_path, report)
        prepared[case_id] = {
            "design_family": case["design_family"],
            "report_scale": case["report_scale"],
            "report": report,
            "report_path": report_path,
            "writer": case["writer"],
        }
    return prepared


def _measure_bundle(
    *,
    work_root: Path,
    case_id: str,
    report_path: Path,
    writer: Callable[[str | Path, dict[str, Any] | str | Path], dict[str, Any]],
    repeat: int,
) -> tuple[list[float], Path, dict[str, Any]]:
    run_root = work_root / "bundles" / case_id
    run_root.mkdir(parents=True, exist_ok=True)
    bundle_times: list[float] = []
    first_bundle_dir: Path | None = None
    first_summary: dict[str, Any] | None = None

    for idx in range(repeat):
        bundle_dir = run_root / f"bundle_{idx + 1:02d}"
        t0 = time.perf_counter()
        summary = writer(bundle_dir, report_path)
        dt = time.perf_counter() - t0
        bundle_times.append(dt)
        if first_bundle_dir is None:
            first_bundle_dir = bundle_dir
            first_summary = summary

    assert first_bundle_dir is not None
    assert first_summary is not None
    return bundle_times, first_bundle_dir, first_summary


def _measure_manifest_regen(bundle_dir: Path, repeat: int) -> list[float]:
    timings: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        ns_audit._write_manifest(bundle_dir)
        timings.append(time.perf_counter() - t0)
    return timings


def _bundle_sizes(bundle_dir: Path) -> dict[str, int]:
    total_bundle_bytes = 0
    for path in bundle_dir.rglob("*"):
        if path.is_file():
            total_bundle_bytes += path.stat().st_size
    return {
        "input_report_bytes": (bundle_dir / "inputs" / "input.json").stat().st_size,
        "total_bundle_bytes": total_bundle_bytes,
        "manifest_bytes": (bundle_dir / "manifest.json").stat().st_size,
        "markdown_bytes": (bundle_dir / "outputs" / "design_report.md").stat().st_size,
    }


def _required_artifacts_present(bundle_dir: Path) -> bool:
    return all((bundle_dir / rel).exists() for rel in REQUIRED_ARTIFACTS)


def _manifest_file_count(bundle_dir: Path) -> int:
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    return len(manifest.get("files", []))


def _build_case_doc(
    *,
    case_id: str,
    case_budget: dict[str, Any],
    prepared_case: dict[str, Any],
    work_root: Path,
    repeat: int,
    manifest_repeat: int,
) -> tuple[dict[str, Any], bool]:
    bundle_times, bundle_dir, summary = _measure_bundle(
        work_root=work_root,
        case_id=case_id,
        report_path=prepared_case["report_path"],
        writer=prepared_case["writer"],
        repeat=repeat,
    )
    manifest_times = _measure_manifest_regen(bundle_dir, manifest_repeat)
    meta = json.loads((bundle_dir / "meta.json").read_text(encoding="utf-8"))
    sizes = _bundle_sizes(bundle_dir)
    validation = {
        "summary_schema_version": summary["schema_version"],
        "summary_deterministic": bool(summary.get("deterministic")),
        "created_unix_ms_zero": meta.get("created_unix_ms") == 0,
        "required_artifacts_present": _required_artifacts_present(bundle_dir),
        "manifest_file_count": _manifest_file_count(bundle_dir),
    }
    budget_thresholds = {
        "max_bundle_duration_s": float(case_budget["max_bundle_duration_s"]),
        "max_manifest_regen_duration_s": float(case_budget["max_manifest_regen_duration_s"]),
        "max_bundle_bytes": int(case_budget["max_bundle_bytes"]),
        "max_manifest_bytes": int(case_budget["max_manifest_bytes"]),
    }
    bundle_timing = _timing_doc(bundle_times)
    manifest_timing = _timing_doc(manifest_times)
    budget_pass = {
        "bundle_duration": bundle_timing["best_s"] <= budget_thresholds["max_bundle_duration_s"],
        "manifest_regen_duration": manifest_timing["best_s"]
        <= budget_thresholds["max_manifest_regen_duration_s"],
        "bundle_bytes": sizes["total_bundle_bytes"] <= budget_thresholds["max_bundle_bytes"],
        "manifest_bytes": sizes["manifest_bytes"] <= budget_thresholds["max_manifest_bytes"],
    }
    ok = (
        validation["summary_schema_version"] == "nextstat_bayesian_design_report_bundle_v0"
        and validation["summary_deterministic"]
        and validation["created_unix_ms_zero"]
        and validation["required_artifacts_present"]
        and all(budget_pass.values())
    )
    case_doc = {
        "id": case_id,
        "design_family": prepared_case["design_family"],
        "report_scale": prepared_case["report_scale"],
        "input_mode": "path",
        "status": "ok" if ok else "fail",
        "timing": bundle_timing,
        "manifest_regen_timing": manifest_timing,
        "sizes": sizes,
        "budget_thresholds": budget_thresholds,
        "budget_pass": budget_pass,
        "validation": validation,
    }
    return case_doc, ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run the fast smoke configuration")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="record deterministic-mode intent in the emitted artifact",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    args = parser.parse_args(argv)

    budget = load_bayesian_design_report_bundle_performance_budget()
    mode_key = "smoke" if args.smoke else "release"
    mode_budget = budget["runner_modes"][mode_key]
    repeat = int(mode_budget["repeat"])
    manifest_repeat = int(mode_budget["manifest_repeat"])

    session_root = _allocate_session_root(args.work_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    prepared = _prepare_reports(session_root)
    cases = []
    exit_code = 0
    for case_id in ("beta_small", "beta_large", "normal_small", "normal_large"):
        try:
            case_doc, ok = _build_case_doc(
                case_id=case_id,
                case_budget=budget["cases"][case_id],
                prepared_case=prepared[case_id],
                work_root=session_root,
                repeat=repeat,
                manifest_repeat=manifest_repeat,
            )
            if not ok:
                exit_code = 1
        except Exception as exc:
            case_doc = {
                "id": case_id,
                "design_family": prepared[case_id]["design_family"],
                "report_scale": prepared[case_id]["report_scale"],
                "input_mode": "path",
                "status": "fail",
                "timing": {"repeat": 0, "policy": "min", "per_run_s": [], "best_s": 0.0, "median_s": 0.0},
                "manifest_regen_timing": {
                    "repeat": 0,
                    "policy": "min",
                    "per_run_s": [],
                    "best_s": 0.0,
                    "median_s": 0.0,
                },
                "sizes": {
                    "input_report_bytes": 0,
                    "total_bundle_bytes": 0,
                    "manifest_bytes": 0,
                    "markdown_bytes": 0,
                },
                "budget_thresholds": {
                    "max_bundle_duration_s": float(budget["cases"][case_id]["max_bundle_duration_s"]),
                    "max_manifest_regen_duration_s": float(
                        budget["cases"][case_id]["max_manifest_regen_duration_s"]
                    ),
                    "max_bundle_bytes": int(budget["cases"][case_id]["max_bundle_bytes"]),
                    "max_manifest_bytes": int(budget["cases"][case_id]["max_manifest_bytes"]),
                },
                "budget_pass": {
                    "bundle_duration": False,
                    "manifest_regen_duration": False,
                    "bundle_bytes": False,
                    "manifest_bytes": False,
                },
                "validation": {
                    "summary_schema_version": "",
                    "summary_deterministic": False,
                    "created_unix_ms_zero": False,
                    "required_artifacts_present": False,
                    "manifest_file_count": 0,
                },
                "error": f"{type(exc).__name__}: {exc}",
            }
            exit_code = 1
        cases.append(case_doc)

    report = {
        "schema_version": "nextstat.bayesian_design_report_bundle_benchmark_result.v1",
        "suite": "bayesian_design_report_bundle_packaging",
        "deterministic": bool(args.deterministic or args.smoke),
        "environment": collect_environment(),
        "budget": {
            "schema_version": budget["schema_version"],
            "manifest_path": "scripts/bayesian_design_report_bundle_performance_budget_v1.json",
            "runner_mode": mode_key,
            "repeat": repeat,
            "manifest_repeat": manifest_repeat,
        },
        "meta": {
            "host_policy": "nextstat-bench",
            "nextstat_command": [sys.executable, "scripts/benchmarks/bench_bayesian_design_report_bundle.py"],
            "smoke": bool(args.smoke),
            "out": str(args.out),
            "work_root": str(session_root),
        },
        "cases": cases,
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
