#!/usr/bin/env python3
"""Compare an M15 reporting benchmark artifact against the accepted baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "nextstat.m15_reporting_benchmark_compare_report.v1"
RESULT_SCHEMA_VERSION = "nextstat.m15_reporting_benchmark_result.v1"
SUITE = "m15_reporting"
REQUIRED_HOST_POLICY = "nextstat-bench"
REQUIRED_BUILD_PROFILE = "release"
REQUIRED_HOSTNAME = "nextstat-bench"
DEFAULT_BASELINE = (
    REPO_ROOT / "benchmarks" / "artifacts" / "m15_reporting_baselines" / "nextstat-bench" / "accepted.json"
)
DEFAULT_OUT = REPO_ROOT / "bench_results" / "m15_reporting" / "compare_report.json"
CASE_IDS = [
    "m15_assessment_table",
    "m15_map",
    "m15_mar",
    "m15_bundle",
    "validation_pack_base_json_only",
    "validation_pack_m15_json_only",
]
CASE_CATEGORIES = {
    "m15_assessment_table": "cli",
    "m15_map": "cli",
    "m15_mar": "cli",
    "m15_bundle": "cli",
    "validation_pack_base_json_only": "validation_pack",
    "validation_pack_m15_json_only": "validation_pack",
}

POLICY = {
    "schema_version": "nextstat.m15_reporting_benchmark_compare_policy.v1",
    "required_host_policy": REQUIRED_HOST_POLICY,
    "required_hostname": REQUIRED_HOSTNAME,
    "required_build_profile": REQUIRED_BUILD_PROFILE,
    "require_non_smoke": True,
    "required_runs": 5,
    "required_warmups": 1,
    "case_ids": CASE_IDS,
    "cli_metric": {
        "review_ratio": 1.50,
        "fail_ratio": 2.00,
        "min_baseline_value": 0.005,
    },
    "validation_pack_metric": {
        "review_ratio": 1.20,
        "fail_ratio": 1.40,
        "min_baseline_value": 0.05,
    },
    "derived_ratio_metric": {
        "review_ratio": 1.10,
        "fail_ratio": 1.20,
        "min_baseline_value": 0.50,
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
    host = doc.get("host") if isinstance(doc.get("host"), dict) else {}
    binary = doc.get("binary") if isinstance(doc.get("binary"), dict) else {}
    protocol = doc.get("protocol") if isinstance(doc.get("protocol"), dict) else {}
    results = doc.get("results") if isinstance(doc.get("results"), list) else []
    return {
        "path": str(path),
        "schema_version": doc.get("schema_version"),
        "suite": doc.get("suite"),
        "deterministic": bool(doc.get("deterministic")),
        "host_policy": meta.get("host_policy"),
        "hostname": host.get("hostname"),
        "build_profile": binary.get("build_profile"),
        "version": binary.get("version"),
        "smoke": bool(meta.get("smoke")),
        "runs": protocol.get("runs"),
        "warmups": protocol.get("warmups"),
        "case_ids": [case.get("case_id") for case in results if isinstance(case, dict)],
    }


def _empty_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "schema_version": None,
        "suite": None,
        "deterministic": False,
        "host_policy": None,
        "hostname": None,
        "build_profile": None,
        "version": None,
        "smoke": False,
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


def _empty_environment_checks() -> dict[str, Any]:
    return {
        "host_policy": _empty_check(REQUIRED_HOST_POLICY),
        "hostname": _empty_check(REQUIRED_HOSTNAME),
        "release_build": _empty_check(REQUIRED_BUILD_PROFILE),
        "deterministic": _empty_check(True),
        "non_smoke": _empty_check(True),
        "runs": _empty_check(int(POLICY["required_runs"])),
        "warmups": _empty_check(int(POLICY["required_warmups"])),
        "case_set": {
            "required": list(CASE_IDS),
            "baseline": [],
            "current": [],
            "matches": False,
        },
    }


def _field_doc(baseline: Any, current: Any) -> dict[str, Any]:
    return {
        "baseline": baseline,
        "current": current,
        "matches": baseline == current,
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
    if not bool(header.get("deterministic")):
        errors.append("artifact_not_deterministic")
    if POLICY["require_non_smoke"] and bool(header.get("smoke")):
        errors.append("artifact_is_smoke")
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


def _validated_schema_versions(case: dict[str, Any]) -> list[str]:
    validation = case.get("validation")
    if not isinstance(validation, dict):
        return []
    validated = validation.get("validated_artifacts")
    if not isinstance(validated, list):
        return []
    out: list[str] = []
    for item in validated:
        if isinstance(item, dict):
            out.append(str(item.get("expected_schema_version")))
    return out


def _validated_artifacts_ok(case: dict[str, Any]) -> bool:
    validation = case.get("validation")
    if not isinstance(validation, dict):
        return False
    validated = validation.get("validated_artifacts")
    if not isinstance(validated, list) or not validated:
        return False
    return all(isinstance(item, dict) and item.get("status") == "ok" for item in validated)


def _case_snapshot(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": case.get("status"),
        "median_s": case.get("median_s"),
        "min_s": case.get("min_s"),
        "max_s": case.get("max_s"),
        "runs": case.get("runs"),
        "warmups": case.get("warmups"),
        "validated_schema_versions": _validated_schema_versions(case),
    }


def _compare_case(baseline_case: dict[str, Any], current_case: dict[str, Any]) -> dict[str, Any]:
    case_id = str(baseline_case.get("case_id"))
    category = CASE_CATEGORIES[case_id]
    threshold_key = "cli_metric" if category == "cli" else "validation_pack_metric"
    thresholds = POLICY[threshold_key]

    errors: list[str] = []
    warnings: list[str] = []
    identity = {
        "case_status": _field_doc(baseline_case.get("status"), current_case.get("status")),
        "validated_schema_versions": _field_doc(
            _validated_schema_versions(baseline_case), _validated_schema_versions(current_case)
        ),
        "validated_artifact_count": _field_doc(
            len(_validated_schema_versions(baseline_case)), len(_validated_schema_versions(current_case))
        ),
    }

    if baseline_case.get("status") != "ok":
        errors.append(f"baseline_case_status:{baseline_case.get('status')}")
    if current_case.get("status") != "ok":
        errors.append(f"current_case_status:{current_case.get('status')}")
    if not _validated_artifacts_ok(baseline_case):
        errors.append("baseline_validated_artifacts_not_ok")
    if not _validated_artifacts_ok(current_case):
        errors.append("current_validated_artifacts_not_ok")
    if not identity["validated_schema_versions"]["matches"]:
        errors.append("validated_schema_versions_changed")

    metric = _metric_doc(
        name="median_s",
        baseline_value=_as_finite_number(baseline_case.get("median_s")),
        current_value=_as_finite_number(current_case.get("median_s")),
        review_ratio=float(thresholds["review_ratio"]),
        fail_ratio=float(thresholds["fail_ratio"]) if thresholds["fail_ratio"] is not None else None,
        min_baseline_value=float(thresholds["min_baseline_value"]),
    )

    if metric["status"] == "failed":
        errors.append(f"median_s:{','.join(metric['reasons']) or metric['status']}")
    elif metric["status"] == "review":
        warnings.append(f"median_s:{','.join(metric['reasons']) or metric['status']}")

    if errors:
        status = "failed"
    elif warnings:
        status = "review"
    else:
        status = "passed"

    return {
        "id": case_id,
        "category": category,
        "status": status,
        "ok": status != "failed",
        "requires_review": status == "review",
        "reasons": {
            "errors": errors,
            "warnings": warnings,
        },
        "identity": identity,
        "baseline": _case_snapshot(baseline_case),
        "current": _case_snapshot(current_case),
        "metrics": [metric],
    }


def _compare(baseline: dict[str, Any], current: dict[str, Any], baseline_path: Path, current_path: Path) -> dict[str, Any]:
    baseline_header = _artifact_header(baseline, baseline_path)
    current_header = _artifact_header(current, current_path)

    errors = _check_header(baseline_header) + _check_header(current_header)
    if baseline_header.get("host_policy") != current_header.get("host_policy"):
        errors.append("host_policy_mismatch")
    if baseline_header.get("hostname") != current_header.get("hostname"):
        errors.append("hostname_mismatch")

    required_case_ids = list(POLICY["case_ids"])
    case_set = {
        "required": required_case_ids,
        "baseline": list(baseline_header["case_ids"]),
        "current": list(current_header["case_ids"]),
        "matches": baseline_header["case_ids"] == required_case_ids and current_header["case_ids"] == required_case_ids,
    }
    if not case_set["matches"]:
        errors.append("case_set_mismatch")

    case_reports: list[dict[str, Any]] = []
    for case_id in required_case_ids:
        baseline_case = _find_case(baseline, case_id)
        current_case = _find_case(current, case_id)
        if baseline_case is None or current_case is None:
            case_reports.append(
                {
                    "id": case_id,
                    "category": CASE_CATEGORIES.get(case_id, "unknown"),
                    "status": "failed",
                    "ok": False,
                    "requires_review": False,
                    "reasons": {
                        "errors": ["missing_case"],
                        "warnings": [],
                    },
                    "identity": {},
                    "baseline": {},
                    "current": {},
                    "metrics": [],
                }
            )
            errors.append(f"missing_case:{case_id}")
            continue
        case_reports.append(_compare_case(baseline_case, current_case))

    derived_thresholds = POLICY["derived_ratio_metric"]
    derived_metric = _metric_doc(
        name="validation_pack_m15_over_base_median_ratio",
        baseline_value=_as_finite_number(
            baseline.get("derived", {}).get("validation_pack_m15_over_base_median_ratio")
            if isinstance(baseline.get("derived"), dict)
            else None
        ),
        current_value=_as_finite_number(
            current.get("derived", {}).get("validation_pack_m15_over_base_median_ratio")
            if isinstance(current.get("derived"), dict)
            else None
        ),
        review_ratio=float(derived_thresholds["review_ratio"]),
        fail_ratio=float(derived_thresholds["fail_ratio"]),
        min_baseline_value=float(derived_thresholds["min_baseline_value"]),
    )
    derived_errors: list[str] = []
    derived_warnings: list[str] = []
    if derived_metric["status"] == "failed":
        derived_errors.append(
            f"{derived_metric['name']}:{','.join(derived_metric['reasons']) or derived_metric['status']}"
        )
    elif derived_metric["status"] == "review":
        derived_warnings.append(
            f"{derived_metric['name']}:{','.join(derived_metric['reasons']) or derived_metric['status']}"
        )

    failed_cases = sum(1 for case in case_reports if case["status"] == "failed")
    review_cases = sum(1 for case in case_reports if case["status"] == "review")
    passed_cases = sum(1 for case in case_reports if case["status"] == "passed")
    metric_count = sum(len(case["metrics"]) for case in case_reports) + 1
    failed_metrics = sum(1 for case in case_reports for metric in case["metrics"] if metric["status"] == "failed")
    review_metrics = sum(1 for case in case_reports for metric in case["metrics"] if metric["status"] == "review")
    skipped_metrics = sum(
        1
        for case in case_reports
        for metric in case["metrics"]
        if metric["status"] in {"skipped_floor", "not_applicable"}
    )
    passed_metrics = sum(1 for case in case_reports for metric in case["metrics"] if metric["status"] == "passed")
    if derived_metric["status"] == "failed":
        failed_metrics += 1
    elif derived_metric["status"] == "review":
        review_metrics += 1
    elif derived_metric["status"] == "passed":
        passed_metrics += 1
    elif derived_metric["status"] in {"skipped_floor", "not_applicable"}:
        skipped_metrics += 1

    top_level_errors = list(errors) + derived_errors
    top_level_warnings = list(derived_warnings)

    if top_level_errors or failed_cases:
        status = "failed"
    elif review_cases or top_level_warnings:
        status = "review"
    else:
        status = "passed"

    return {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE,
        "status": status,
        "ok": status != "failed",
        "requires_review": status == "review",
        "baseline_path": str(baseline_path),
        "current_path": str(current_path),
        "policy": POLICY,
        "artifacts": {
            "baseline": baseline_header,
            "current": current_header,
        },
        "environment_checks": {
            "host_policy": {
                "required": REQUIRED_HOST_POLICY,
                "baseline": baseline_header.get("host_policy"),
                "current": current_header.get("host_policy"),
                "matches": baseline_header.get("host_policy") == current_header.get("host_policy") == REQUIRED_HOST_POLICY,
            },
            "hostname": {
                "required": REQUIRED_HOSTNAME,
                "baseline": baseline_header.get("hostname"),
                "current": current_header.get("hostname"),
                "matches": baseline_header.get("hostname") == current_header.get("hostname") == REQUIRED_HOSTNAME,
            },
            "release_build": {
                "required": REQUIRED_BUILD_PROFILE,
                "baseline": baseline_header.get("build_profile"),
                "current": current_header.get("build_profile"),
                "matches": baseline_header.get("build_profile")
                == current_header.get("build_profile")
                == REQUIRED_BUILD_PROFILE,
            },
            "deterministic": {
                "required": True,
                "baseline": baseline_header.get("deterministic"),
                "current": current_header.get("deterministic"),
                "matches": baseline_header.get("deterministic") is True and current_header.get("deterministic") is True,
            },
            "non_smoke": {
                "required": True,
                "baseline": not baseline_header.get("smoke"),
                "current": not current_header.get("smoke"),
                "matches": not baseline_header.get("smoke") and not current_header.get("smoke"),
            },
            "runs": {
                "required": int(POLICY["required_runs"]),
                "baseline": baseline_header.get("runs"),
                "current": current_header.get("runs"),
                "matches": baseline_header.get("runs") == current_header.get("runs") == int(POLICY["required_runs"]),
            },
            "warmups": {
                "required": int(POLICY["required_warmups"]),
                "baseline": baseline_header.get("warmups"),
                "current": current_header.get("warmups"),
                "matches": baseline_header.get("warmups") == current_header.get("warmups") == int(POLICY["required_warmups"]),
            },
            "case_set": case_set,
        },
        "summary": {
            "passed_cases": passed_cases,
            "review_cases": review_cases,
            "failed_cases": failed_cases,
            "metric_count": metric_count,
            "passed_metrics": passed_metrics,
            "review_metrics": review_metrics,
            "failed_metrics": failed_metrics,
            "skipped_metrics": skipped_metrics,
            "top_level_errors": top_level_errors,
            "top_level_warnings": top_level_warnings,
        },
        "derived": {
            "validation_pack_m15_over_base_median_ratio": derived_metric,
        },
        "cases": case_reports,
    }


def _missing_payload(*, baseline_path: Path, current_path: Path, error: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE,
        "status": "failed",
        "ok": False,
        "requires_review": False,
        "baseline_path": str(baseline_path),
        "current_path": str(current_path),
        "policy": POLICY,
        "artifacts": {
            "baseline": _empty_artifact(baseline_path),
            "current": _empty_artifact(current_path),
        },
        "environment_checks": _empty_environment_checks(),
        "summary": {
            "passed_cases": 0,
            "review_cases": 0,
            "failed_cases": 0,
            "metric_count": 0,
            "passed_metrics": 0,
            "review_metrics": 0,
            "failed_metrics": 0,
            "skipped_metrics": 0,
            "top_level_errors": [error],
            "top_level_warnings": [],
        },
        "derived": {
            "validation_pack_m15_over_base_median_ratio": _metric_doc(
                name="validation_pack_m15_over_base_median_ratio",
                baseline_value=None,
                current_value=None,
                review_ratio=float(POLICY["derived_ratio_metric"]["review_ratio"]),
                fail_ratio=float(POLICY["derived_ratio_metric"]["fail_ratio"]),
                min_baseline_value=float(POLICY["derived_ratio_metric"]["min_baseline_value"]),
            )
        },
        "cases": [],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compare M15 reporting benchmark artifact against the accepted release baseline."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help=f"Baseline JSON artifact path (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument("--current", type=Path, required=True, help="Current JSON artifact path")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output comparison report JSON path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--fail-on-review",
        dest="fail_on_review",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit non-zero when report status=review (default: false)",
    )
    args = parser.parse_args(argv)

    if not args.baseline.exists():
        payload = _missing_payload(
            baseline_path=args.baseline,
            current_path=args.current,
            error=f"baseline_missing:{args.baseline}",
        )
        _write_json(args.out, payload)
        return 2

    if not args.current.exists():
        payload = _missing_payload(
            baseline_path=args.baseline,
            current_path=args.current,
            error=f"current_missing:{args.current}",
        )
        _write_json(args.out, payload)
        return 2

    baseline = _read_json(args.baseline)
    current = _read_json(args.current)
    report = _compare(baseline, current, args.baseline, args.current)
    _write_json(args.out, report)

    if report["status"] == "failed":
        return 2
    if report["status"] == "review" and args.fail_on_review:
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
