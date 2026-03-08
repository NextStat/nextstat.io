#!/usr/bin/env python3
"""Compare an ads + weekly time-series benchmark artifact against the accepted baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "nextstat.ads_timeseries_benchmark_compare_report.v1"
RESULT_SCHEMA_VERSION = "nextstat.ads_timeseries_benchmark_result.v1"
SUITE = "ads_timeseries_surface"
REQUIRED_HOST_POLICY = "nextstat-bench"
REQUIRED_HOSTNAME = "nextstat-bench"
REQUIRED_BUILD_PROFILE = "release"
DEFAULT_BASELINE = (
    REPO_ROOT / "benchmarks" / "artifacts" / "ads_timeseries_baselines" / "nextstat-bench" / "accepted.json"
)
DEFAULT_OUT = REPO_ROOT / "bench_results" / "ads_timeseries_surface" / "compare_report.json"
CASE_IDS = [
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

POLICY = {
    "schema_version": "nextstat.ads_timeseries_benchmark_compare_policy.v1",
    "required_host_policy": REQUIRED_HOST_POLICY,
    "required_hostname": REQUIRED_HOSTNAME,
    "required_build_profile": REQUIRED_BUILD_PROFILE,
    "require_non_smoke": True,
    "require_deterministic": True,
    "required_runs": 5,
    "required_warmups": 1,
    "case_ids": CASE_IDS,
    # Python helper timings are in the low-microsecond to low-submillisecond range
    # on the bench host, so a floor avoids treating scheduler noise as regressions.
    "python_metric": {
        "review_ratio": 2.0,
        "fail_ratio": 4.0,
        "min_baseline_value": 0.001,
    },
    "cli_metric": {
        "review_ratio": 1.30,
        "fail_ratio": 1.60,
        "min_baseline_value": 0.005,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"failed to read JSON from {path}: {exc}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_header(doc: dict[str, Any], path: Path) -> dict[str, Any]:
    meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
    protocol = doc.get("protocol") if isinstance(doc.get("protocol"), dict) else {}
    host = doc.get("host") if isinstance(doc.get("host"), dict) else {}
    binary = doc.get("binary") if isinstance(doc.get("binary"), dict) else {}
    results = doc.get("results") if isinstance(doc.get("results"), list) else []
    return {
        "path": str(path),
        "schema_version": doc.get("schema_version"),
        "suite": doc.get("suite"),
        "host_policy": meta.get("host_policy"),
        "hostname": host.get("hostname"),
        "build_profile": binary.get("build_profile"),
        "smoke": bool(meta.get("smoke")),
        "deterministic": bool(meta.get("deterministic")),
        "runs": protocol.get("runs"),
        "warmups": protocol.get("warmups"),
        "case_ids": [case.get("case_id") for case in results if isinstance(case, dict)],
    }


def _empty_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "schema_version": None,
        "suite": None,
        "host_policy": None,
        "hostname": None,
        "build_profile": None,
        "smoke": False,
        "deterministic": False,
        "runs": None,
        "warmups": None,
        "case_ids": [],
    }


def _empty_check(required: Any) -> dict[str, Any]:
    return {
        "required": required,
        "baseline": None,
        "current": None,
        "matches": False,
    }


def _field_doc(baseline: Any, current: Any) -> dict[str, Any]:
    return {
        "baseline": baseline,
        "current": current,
        "matches": baseline == current,
    }


def _empty_environment_checks() -> dict[str, Any]:
    return {
        "host_policy": _empty_check(REQUIRED_HOST_POLICY),
        "hostname": _empty_check(REQUIRED_HOSTNAME),
        "release_build": _empty_check(REQUIRED_BUILD_PROFILE),
        "non_smoke": _empty_check(True),
        "deterministic": _empty_check(True),
        "runs": _empty_check(int(POLICY["required_runs"])),
        "warmups": _empty_check(int(POLICY["required_warmups"])),
        "case_set": {
            "required": list(CASE_IDS),
            "baseline": [],
            "current": [],
            "matches": False,
        },
    }


def _check_header(header: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if header.get("schema_version") != RESULT_SCHEMA_VERSION:
        errors.append(f"unexpected_result_schema:{header.get('schema_version')}")
    if header.get("suite") != SUITE:
        errors.append(f"unexpected_suite:{header.get('suite')}")
    if header.get("host_policy") != REQUIRED_HOST_POLICY:
        errors.append(f"unexpected_host_policy:{header.get('host_policy')}")
    if header.get("hostname") != REQUIRED_HOSTNAME:
        errors.append(f"unexpected_hostname:{header.get('hostname')}")
    if header.get("build_profile") != REQUIRED_BUILD_PROFILE:
        errors.append(f"unexpected_build_profile:{header.get('build_profile')}")
    if bool(header.get("smoke")):
        errors.append("artifact_is_smoke")
    if not bool(header.get("deterministic")):
        errors.append("artifact_not_deterministic")
    if header.get("runs") != int(POLICY["required_runs"]):
        errors.append(f"unexpected_runs:{header.get('runs')}")
    if header.get("warmups") != int(POLICY["required_warmups"]):
        errors.append(f"unexpected_warmups:{header.get('warmups')}")
    return errors


def _find_case(payload: dict[str, Any], case_id: str) -> dict[str, Any] | None:
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    for item in results:
        if isinstance(item, dict) and item.get("case_id") == case_id:
            return item
    return None


def _as_finite_number(value: Any) -> float | None:
    if value is None or not isinstance(value, (int, float)):
        return None
    x = float(value)
    if not math.isfinite(x):
        return None
    return x


def _metric_doc(
    *,
    name: str,
    baseline_value: float | None,
    current_value: float | None,
    review_ratio: float,
    fail_ratio: float | None,
    min_baseline_value: float,
) -> dict[str, Any]:
    ratio = None
    reasons: list[str] = []
    if baseline_value is None and current_value is None:
        status = "not_applicable"
    elif baseline_value is None or current_value is None:
        status = "failed"
        reasons.append("metric_missing")
    elif baseline_value < min_baseline_value:
        status = "skipped_floor"
        if baseline_value > 0.0:
            ratio = current_value / baseline_value
    elif baseline_value <= 0.0:
        status = "skipped_floor"
    else:
        ratio = current_value / baseline_value
        if fail_ratio is not None and ratio > fail_ratio:
            status = "failed"
            reasons.append("ratio_exceeded_fail_threshold")
        elif ratio > review_ratio:
            status = "review"
            reasons.append("ratio_exceeded_review_threshold")
        else:
            status = "passed"

    return {
        "name": name,
        "status": status,
        "baseline_value": baseline_value,
        "current_value": current_value,
        "ratio": None if ratio is None else round(ratio, 6),
        "review_ratio": review_ratio,
        "fail_ratio": fail_ratio,
        "min_baseline_value": min_baseline_value,
        "reasons": reasons,
    }


def _compare_case(baseline_case: dict[str, Any], current_case: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    identity = {
        "surface": _field_doc(baseline_case.get("surface"), current_case.get("surface")),
        "status": _field_doc(baseline_case.get("status"), current_case.get("status")),
        "details": _field_doc(baseline_case.get("details"), current_case.get("details")),
    }
    for name, field in identity.items():
        if field["matches"]:
            continue
        if name == "details":
            warnings.append("details_changed")
        else:
            errors.append(f"{name}_changed")

    if baseline_case.get("status") != "ok":
        errors.append(f"baseline_case_status:{baseline_case.get('status')}")
    if current_case.get("status") != "ok":
        errors.append(f"current_case_status:{current_case.get('status')}")

    surface = str(current_case.get("surface") or baseline_case.get("surface") or "")
    metric_policy = POLICY["cli_metric"] if surface == "cli" else POLICY["python_metric"]
    median_metric = _metric_doc(
        name="median_s",
        baseline_value=_as_finite_number(baseline_case.get("median_s")),
        current_value=_as_finite_number(current_case.get("median_s")),
        review_ratio=float(metric_policy["review_ratio"]),
        fail_ratio=float(metric_policy["fail_ratio"]),
        min_baseline_value=float(metric_policy["min_baseline_value"]),
    )
    if median_metric["status"] == "review":
        warnings.append("median_ratio_requires_review")
    elif median_metric["status"] == "failed":
        errors.append("median_ratio_failed")

    if errors:
        status = "failed"
    elif median_metric["status"] == "review" or "details_changed" in warnings:
        status = "review"
    else:
        status = "passed"

    return {
        "id": str(current_case.get("case_id") or baseline_case.get("case_id")),
        "status": status,
        "surface": surface,
        "identity": identity,
        "metrics": [median_metric],
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Compare an ads + weekly time-series benchmark artifact against the accepted baseline.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help=f"Accepted baseline path (default: {DEFAULT_BASELINE})")
    parser.add_argument("--current", type=Path, required=True, help="Current benchmark artifact to compare")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Comparison report output path (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--fail-on-review",
        action="store_true",
        help="Return exit code 2 when the compare result requires review",
    )
    args = parser.parse_args(argv)

    top_level_errors: list[str] = []
    baseline_header = _empty_artifact(args.baseline)
    current_header = _empty_artifact(args.current)
    environment_checks = _empty_environment_checks()
    cases: list[dict[str, Any]] = []

    try:
        baseline_doc = _read_json(args.baseline)
    except FileNotFoundError:
        top_level_errors.append(f"baseline_missing:{args.baseline}")
        baseline_doc = {}
    try:
        current_doc = _read_json(args.current)
    except FileNotFoundError:
        top_level_errors.append(f"current_missing:{args.current}")
        current_doc = {}

    if baseline_doc:
        baseline_header = _artifact_header(baseline_doc, args.baseline)
        top_level_errors.extend(_check_header(baseline_header))
    if current_doc:
        current_header = _artifact_header(current_doc, args.current)
        top_level_errors.extend(_check_header(current_header))

    environment_checks["host_policy"] = _field_doc(
        baseline_header.get("host_policy"),
        current_header.get("host_policy"),
    )
    environment_checks["host_policy"]["required"] = REQUIRED_HOST_POLICY
    environment_checks["hostname"] = _field_doc(
        baseline_header.get("hostname"),
        current_header.get("hostname"),
    )
    environment_checks["hostname"]["required"] = REQUIRED_HOSTNAME
    environment_checks["release_build"] = _field_doc(
        baseline_header.get("build_profile"),
        current_header.get("build_profile"),
    )
    environment_checks["release_build"]["required"] = REQUIRED_BUILD_PROFILE
    environment_checks["non_smoke"] = _field_doc(
        not bool(baseline_header.get("smoke")),
        not bool(current_header.get("smoke")),
    )
    environment_checks["non_smoke"]["required"] = True
    environment_checks["deterministic"] = _field_doc(
        bool(baseline_header.get("deterministic")),
        bool(current_header.get("deterministic")),
    )
    environment_checks["deterministic"]["required"] = True
    environment_checks["runs"] = _field_doc(
        baseline_header.get("runs"),
        current_header.get("runs"),
    )
    environment_checks["runs"]["required"] = int(POLICY["required_runs"])
    environment_checks["warmups"] = _field_doc(
        baseline_header.get("warmups"),
        current_header.get("warmups"),
    )
    environment_checks["warmups"]["required"] = int(POLICY["required_warmups"])
    environment_checks["case_set"] = {
        "required": list(CASE_IDS),
        "baseline": list(baseline_header.get("case_ids", [])),
        "current": list(current_header.get("case_ids", [])),
        "matches": list(baseline_header.get("case_ids", [])) == list(current_header.get("case_ids", [])) == list(CASE_IDS),
    }

    if not environment_checks["case_set"]["matches"]:
        top_level_errors.append("case_set_mismatch")

    for case_id in CASE_IDS:
        baseline_case = _find_case(baseline_doc, case_id) if baseline_doc else None
        current_case = _find_case(current_doc, case_id) if current_doc else None
        if baseline_case is None or current_case is None:
            missing_errors = []
            if baseline_case is None:
                missing_errors.append("baseline_case_missing")
            if current_case is None:
                missing_errors.append("current_case_missing")
            cases.append(
                {
                    "id": case_id,
                    "status": "failed",
                    "surface": None if current_case is None else current_case.get("surface"),
                    "identity": {},
                    "metrics": [],
                    "errors": missing_errors,
                    "warnings": [],
                }
            )
            continue
        case_doc = _compare_case(baseline_case, current_case)
        case_doc["errors"] = [error for error in case_doc["errors"] if error is not None]
        cases.append(case_doc)

    failed_cases = sum(case["status"] == "failed" for case in cases)
    review_cases = sum(case["status"] == "review" for case in cases)
    skipped_floor_cases = sum(
        any(metric.get("status") == "skipped_floor" for metric in case.get("metrics", []))
        for case in cases
    )

    if top_level_errors or failed_cases > 0:
        status = "failed"
    elif review_cases > 0:
        status = "review"
    else:
        status = "passed"

    report = {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE,
        "status": status,
        "ok": status != "failed",
        "requires_review": status == "review",
        "baseline_path": str(args.baseline),
        "current_path": str(args.current),
        "policy": POLICY,
        "baseline": baseline_header,
        "current": current_header,
        "environment_checks": environment_checks,
        "cases": cases,
        "summary": {
            "failed_cases": failed_cases,
            "review_cases": review_cases,
            "skipped_floor_cases": skipped_floor_cases,
            "top_level_errors": top_level_errors,
        },
    }
    _write_json(args.out, report)

    if status == "failed" or (status == "review" and args.fail_on_review):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
