#!/usr/bin/env python3
"""Compare a HEPData import benchmark artifact against the accepted baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "nextstat.hepdata_import_benchmark_compare_report.v1"
RESULT_SCHEMA_VERSION = "nextstat.hepdata_import_benchmark_result.v1"
SUITE = "hepdata_import"
REQUIRED_HOST_POLICY = "nextstat-bench"
DEFAULT_BASELINE = REPO_ROOT / "benchmarks" / "artifacts" / "hepdata_import_baselines" / "nextstat-bench" / "accepted.json"
DEFAULT_OUT = REPO_ROOT / "bench_results" / "hepdata_import_benchmark" / "compare_report.json"
CASE_IDS = [
    "curated_catalog",
    "direct_patch_catalog_cached",
    "curated_materialize_offline",
    "direct_materialize_network",
]
STAGE_METRICS = ["discovery_s", "download_s", "extract_s", "materialize_s"]

POLICY = {
    "schema_version": "nextstat.hepdata_import_benchmark_compare_policy.v1",
    "required_host_policy": REQUIRED_HOST_POLICY,
    "require_non_smoke": True,
    "case_ids": CASE_IDS,
    "import_total_s": {
        "review_ratio": 1.15,
        "fail_ratio": 1.35,
        "min_baseline_s": 0.01,
    },
    "fit_s": {
        "review_ratio": 1.10,
        "fail_ratio": 1.25,
        "min_baseline_s": 1.0,
    },
    "stage_metrics": {
        "review_ratio": 1.50,
        "min_baseline_s": 0.05,
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
    env = doc.get("environment") if isinstance(doc.get("environment"), dict) else {}
    cases = doc.get("cases") if isinstance(doc.get("cases"), list) else []
    return {
        "path": str(path),
        "schema_version": doc.get("schema_version"),
        "suite": doc.get("suite"),
        "deterministic": bool(doc.get("deterministic")),
        "host_policy": meta.get("host_policy"),
        "node": env.get("node"),
        "smoke": bool(meta.get("smoke")),
        "fit_enabled": bool(meta.get("fit_enabled")),
        "repeat": meta.get("repeat"),
        "fit_repeat": meta.get("fit_repeat"),
        "case_ids": [case.get("id") for case in cases if isinstance(case, dict)],
    }


def _empty_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "schema_version": None,
        "suite": None,
        "deterministic": False,
        "host_policy": None,
        "node": None,
        "smoke": False,
        "fit_enabled": False,
        "repeat": None,
        "fit_repeat": None,
        "case_ids": [],
    }


def _empty_environment_checks() -> dict[str, Any]:
    return {
        "host_policy": {
            "required": REQUIRED_HOST_POLICY,
            "baseline": None,
            "current": None,
            "matches": False,
        },
        "non_smoke": {
            "required": True,
            "baseline": False,
            "current": False,
            "matches": False,
        },
        "case_set": {
            "required": list(POLICY["case_ids"]),
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
    if not bool(header.get("deterministic")):
        errors.append("artifact_not_deterministic")
    if POLICY["require_non_smoke"] and bool(header.get("smoke")):
        errors.append("artifact_is_smoke")
    return errors


def _find_case(payload: dict[str, Any], case_id: str) -> dict[str, Any] | None:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return None
    for item in cases:
        if isinstance(item, dict) and item.get("id") == case_id:
            return item
    return None


def _as_finite_number(value: Any) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        return None
    x = float(value)
    if not math.isfinite(x):
        return None
    return x


def _case_stage(case: dict[str, Any], name: str) -> float | None:
    stages = case.get("stages")
    if not isinstance(stages, dict):
        return None
    return _as_finite_number(stages.get(name))


def _metric_doc(
    *,
    name: str,
    baseline_s: float | None,
    current_s: float | None,
    review_ratio: float,
    fail_ratio: float | None,
    min_baseline_s: float,
) -> dict[str, Any]:
    ratio = None
    reasons: list[str] = []
    if baseline_s is None and current_s is None:
        status = "not_applicable"
    elif baseline_s is None or current_s is None:
        status = "failed"
        reasons.append("metric_missing")
    elif baseline_s < min_baseline_s:
        status = "skipped_floor"
        if baseline_s > 0.0:
            ratio = current_s / baseline_s
    elif baseline_s <= 0.0:
        status = "skipped_floor"
    else:
        ratio = current_s / baseline_s
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
        "baseline_s": baseline_s,
        "current_s": current_s,
        "ratio": None if ratio is None else round(ratio, 6),
        "review_ratio": review_ratio,
        "fail_ratio": fail_ratio,
        "min_baseline_s": min_baseline_s,
        "reasons": reasons,
    }


def _field_doc(baseline: Any, current: Any) -> dict[str, Any]:
    return {
        "baseline": baseline,
        "current": current,
        "matches": baseline == current,
    }


def _normalize_dataset_identity(dataset: Any) -> Any:
    if not isinstance(dataset, dict):
        return dataset
    normalized = dict(dataset)
    doi = normalized.get("doi")
    if isinstance(doi, str) and doi:
        parsed = urlparse(doi)
        if parsed.scheme in {"http", "https"} and parsed.hostname in {"127.0.0.1", "localhost"} and parsed.path == "/download":
            normalized["doi"] = "https://doi.org/10.17182/hepdata.90607.v3/r3"
    return normalized


def _compare_case(baseline_case: dict[str, Any], current_case: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    identity = {
        "case_status": _field_doc(baseline_case.get("status"), current_case.get("status")),
        "summary_mode": _field_doc(baseline_case.get("summary_mode"), current_case.get("summary_mode")),
        "source_mode": _field_doc(baseline_case.get("source_mode"), current_case.get("source_mode")),
        "dataset": _field_doc(
            _normalize_dataset_identity(baseline_case.get("dataset")),
            _normalize_dataset_identity(current_case.get("dataset")),
        ),
        "validation": _field_doc(baseline_case.get("validation"), current_case.get("validation")),
        "artifacts": _field_doc(baseline_case.get("artifacts"), current_case.get("artifacts")),
        "fit_present": _field_doc("fit" in baseline_case, "fit" in current_case),
    }
    for name, field in identity.items():
        if not field["matches"]:
            errors.append(f"{name}_changed")

    if current_case.get("status") != "ok":
        errors.append(f"current_case_status:{current_case.get('status')}")
    if baseline_case.get("status") != "ok":
        errors.append(f"baseline_case_status:{baseline_case.get('status')}")

    import_metric = _metric_doc(
        name="import_total_s",
        baseline_s=_as_finite_number(baseline_case.get("timing", {}).get("best_s")),
        current_s=_as_finite_number(current_case.get("timing", {}).get("best_s")),
        review_ratio=float(POLICY["import_total_s"]["review_ratio"]),
        fail_ratio=float(POLICY["import_total_s"]["fail_ratio"]),
        min_baseline_s=float(POLICY["import_total_s"]["min_baseline_s"]),
    )
    metrics = [import_metric]

    fit_metric = _metric_doc(
        name="fit_s",
        baseline_s=_case_stage(baseline_case, "fit_s"),
        current_s=_case_stage(current_case, "fit_s"),
        review_ratio=float(POLICY["fit_s"]["review_ratio"]),
        fail_ratio=float(POLICY["fit_s"]["fail_ratio"]),
        min_baseline_s=float(POLICY["fit_s"]["min_baseline_s"]),
    )
    metrics.append(fit_metric)

    for stage_name in STAGE_METRICS:
        metrics.append(
            _metric_doc(
                name=stage_name,
                baseline_s=_case_stage(baseline_case, stage_name),
                current_s=_case_stage(current_case, stage_name),
                review_ratio=float(POLICY["stage_metrics"]["review_ratio"]),
                fail_ratio=None,
                min_baseline_s=float(POLICY["stage_metrics"]["min_baseline_s"]),
            )
        )

    for metric in metrics:
        if metric["status"] == "failed":
            errors.append(f"{metric['name']}:{','.join(metric['reasons']) or metric['status']}")
        elif metric["status"] == "review":
            warnings.append(f"{metric['name']}:{','.join(metric['reasons']) or metric['status']}")

    if errors:
        status = "failed"
    elif warnings:
        status = "review"
    else:
        status = "passed"

    return {
        "id": baseline_case.get("id"),
        "status": status,
        "ok": status != "failed",
        "requires_review": status == "review",
        "reasons": {
            "errors": errors,
            "warnings": warnings,
        },
        "identity": identity,
        "baseline": {
            "status": baseline_case.get("status"),
            "timing": baseline_case.get("timing"),
            "stages": baseline_case.get("stages"),
        },
        "current": {
            "status": current_case.get("status"),
            "timing": current_case.get("timing"),
            "stages": current_case.get("stages"),
        },
        "metrics": metrics,
    }


def _compare(baseline: dict[str, Any], current: dict[str, Any], baseline_path: Path, current_path: Path) -> dict[str, Any]:
    baseline_header = _artifact_header(baseline, baseline_path)
    current_header = _artifact_header(current, current_path)

    errors = _check_header(baseline_header) + _check_header(current_header)
    if baseline_header.get("host_policy") != current_header.get("host_policy"):
        errors.append("host_policy_mismatch")

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

    failed_cases = sum(1 for case in case_reports if case["status"] == "failed")
    review_cases = sum(1 for case in case_reports if case["status"] == "review")
    passed_cases = sum(1 for case in case_reports if case["status"] == "passed")
    metric_count = sum(len(case["metrics"]) for case in case_reports)
    failed_metrics = sum(
        1 for case in case_reports for metric in case["metrics"] if metric["status"] == "failed"
    )
    review_metrics = sum(
        1 for case in case_reports for metric in case["metrics"] if metric["status"] == "review"
    )
    skipped_metrics = sum(
        1
        for case in case_reports
        for metric in case["metrics"]
        if metric["status"] in {"skipped_floor", "not_applicable"}
    )
    passed_metrics = sum(
        1 for case in case_reports for metric in case["metrics"] if metric["status"] == "passed"
    )

    if errors or failed_cases:
        status = "failed"
    elif review_cases:
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
            "non_smoke": {
                "required": True,
                "baseline": not baseline_header.get("smoke"),
                "current": not current_header.get("smoke"),
                "matches": not baseline_header.get("smoke") and not current_header.get("smoke"),
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
            "top_level_errors": errors,
        },
        "cases": case_reports,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Compare HEPData import benchmark artifact against the accepted baseline.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help=f"Baseline JSON artifact path (default: {DEFAULT_BASELINE})")
    parser.add_argument("--current", type=Path, required=True, help="Current JSON artifact path")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Output comparison report JSON path (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--fail-on-review",
        dest="fail_on_review",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit non-zero when report status=review (default: false)",
    )
    args = parser.parse_args(argv)

    if not args.baseline.exists():
        payload = {
            "schema_version": SCHEMA_VERSION,
            "suite": SUITE,
            "status": "failed",
            "ok": False,
            "requires_review": False,
            "baseline_path": str(args.baseline),
            "current_path": str(args.current),
            "policy": POLICY,
            "artifacts": {
                "baseline": _empty_artifact(args.baseline),
                "current": _empty_artifact(args.current),
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
                "top_level_errors": [f"baseline_missing:{args.baseline}"],
            },
            "cases": [],
        }
        _write_json(args.out, payload)
        return 2

    if not args.current.exists():
        payload = {
            "schema_version": SCHEMA_VERSION,
            "suite": SUITE,
            "status": "failed",
            "ok": False,
            "requires_review": False,
            "baseline_path": str(args.baseline),
            "current_path": str(args.current),
            "policy": POLICY,
            "artifacts": {
                "baseline": _empty_artifact(args.baseline),
                "current": _empty_artifact(args.current),
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
                "top_level_errors": [f"current_missing:{args.current}"],
            },
            "cases": [],
        }
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
