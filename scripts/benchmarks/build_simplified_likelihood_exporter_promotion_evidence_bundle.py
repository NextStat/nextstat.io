#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from _simplified_likelihood_exporter_promotion_bundle import (
    ACCEPTED_COMMAND_NAMES,
    ACCEPTED_SCHEMA_NAMES,
    BUNDLE_SCHEMA_VERSION,
    EXPLICIT_BOUNDARIES,
    REQUIRED_BENCHMARK_HOST,
    REQUIRED_MAX_ABS_Q_MU_DIFF,
    REQUIRED_MAX_UPPER_LIMIT_RATIO_DEVIATION,
    REQUIRED_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP,
    REQUIRED_MIN_PUBLIC_EXPORT_MATRIX_CASE_COUNT,
    REQUIRED_MIN_TOTAL_EXPORT_MATRIX_CASE_COUNT,
    REPO_ROOT,
    STATIC_ARTIFACTS,
    bundle_path_for,
    load_json,
    now_utc,
    relative_or_absolute,
    sha256_path,
)


def _min_speedup_for_case_kind(benchmark: dict[str, Any], case_kind: str) -> float:
    export_matrix = (
        benchmark.get("export_matrix") if isinstance(benchmark.get("export_matrix"), dict) else {}
    )
    cases = export_matrix.get("cases") if isinstance(export_matrix.get("cases"), list) else []
    speeds = [
        float(case.get("bench", {}).get("speedup", {}).get("net_end_to_end_upper_limit", 0.0))
        for case in cases
        if isinstance(case, dict) and case.get("case_kind") == case_kind
    ]
    return min(speeds, default=0.0)


def _copy_with_inventory(
    *,
    source_path: Path,
    bundle_dir: Path,
    role: str,
    kind: str,
    benchmark: bool = False,
) -> dict[str, Any]:
    if not source_path.exists():
        raise FileNotFoundError(f"missing required exporter evidence artifact: {source_path}")
    bundle_path = bundle_path_for(source_path, benchmark=benchmark)
    target_path = bundle_dir / bundle_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return {
        "role": role,
        "kind": kind,
        "required": True,
        "source_path": relative_or_absolute(source_path),
        "bundle_path": str(bundle_path),
        "sha256": sha256_path(target_path),
        "bytes": target_path.stat().st_size,
    }


def _build_bundle(
    *,
    benchmark_artifact: Path,
    snapshot_report: Path,
    snapshot_index: Path,
    bundle_dir: Path,
    deterministic: bool,
) -> dict[str, Any]:
    benchmark = load_json(benchmark_artifact)
    snapshot = load_json(snapshot_report)
    current_index = load_json(snapshot_index)

    export_summary = benchmark.get("export_matrix", {}).get("summary", {})
    benchmark_summary = benchmark.get("summary", {})
    snapshot_source = snapshot.get("source_summary", {})

    benchmark_host = str(snapshot_source.get("benchmark_host") or benchmark.get("environment", {}).get("hostname") or "unknown")
    max_abs_q_mu_diff = float(export_summary.get("max_abs_q_mu_diff", 0.0))
    max_upper_limit_ratio_deviation = float(export_summary.get("max_upper_limit_ratio_deviation", 0.0))
    overall_min_net_end_to_end_upper_limit_speedup = float(
        export_summary.get("min_net_end_to_end_upper_limit_speedup", 0.0)
    )
    min_net_end_to_end_upper_limit_speedup = float(
        snapshot_source.get("export_matrix_synthetic_min_net_end_to_end_upper_limit_speedup")
        or _min_speedup_for_case_kind(benchmark, "synthetic")
    )
    public_min_net_end_to_end_upper_limit_speedup = float(
        snapshot_source.get(
            "export_matrix_public_reinterpretation_style_min_net_end_to_end_upper_limit_speedup"
        )
        or _min_speedup_for_case_kind(benchmark, "public_reinterpretation_style")
    )

    supports_committed_snapshot = (
        snapshot.get("status") == "persisted"
        and snapshot_source.get("benchmark_host") == REQUIRED_BENCHMARK_HOST
        and snapshot_source.get("export_matrix_status") == "ok"
        and int(snapshot_source.get("export_matrix_case_count", 0))
        >= REQUIRED_MIN_TOTAL_EXPORT_MATRIX_CASE_COUNT
        and int(snapshot_source.get("export_matrix_public_reinterpretation_style_case_count", 0))
        >= REQUIRED_MIN_PUBLIC_EXPORT_MATRIX_CASE_COUNT
    )
    research_grade_acceptance_supported = (
        benchmark_summary.get("status") == "ok"
        and bool(benchmark_summary.get("all_schema_valid", False))
        and bool(benchmark_summary.get("all_fidelity_gates_pass", False))
        and bool(benchmark_summary.get("all_performance_gates_pass", False))
        and bool(benchmark_summary.get("export_matrix_included", False))
        and benchmark_summary.get("export_matrix_status") == "ok"
        and export_summary.get("status") == "ok"
        and bool(export_summary.get("all_schema_valid", False))
        and bool(export_summary.get("all_fidelity_gates_pass", False))
        and bool(export_summary.get("all_performance_gates_pass", False))
        and supports_committed_snapshot
    )
    future_stable_review_ready = (
        research_grade_acceptance_supported
        and benchmark_host == REQUIRED_BENCHMARK_HOST
        and min_net_end_to_end_upper_limit_speedup
        >= REQUIRED_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP
        and max_abs_q_mu_diff <= REQUIRED_MAX_ABS_Q_MU_DIFF
        and max_upper_limit_ratio_deviation <= REQUIRED_MAX_UPPER_LIMIT_RATIO_DEVIATION
    )

    bundle_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        _copy_with_inventory(
            source_path=benchmark_artifact,
            bundle_dir=bundle_dir,
            role="benchmark_artifact",
            kind="benchmark_artifact",
            benchmark=True,
        ),
        _copy_with_inventory(
            source_path=snapshot_report,
            bundle_dir=bundle_dir,
            role="current_snapshot_report",
            kind="benchmark_snapshot_report",
        ),
        _copy_with_inventory(
            source_path=snapshot_index,
            bundle_dir=bundle_dir,
            role="current_snapshot_index",
            kind="benchmark_snapshot_index",
        ),
    ]
    for artifact_spec in STATIC_ARTIFACTS:
        artifacts.append(
            _copy_with_inventory(
                source_path=REPO_ROOT / artifact_spec["source_path"],
                bundle_dir=bundle_dir,
                role=artifact_spec["role"],
                kind=artifact_spec["kind"],
            )
        )

    benchmark_artifact_entry = artifacts[0]
    snapshot_report_entry = artifacts[1]
    snapshot_index_entry = artifacts[2]

    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "surface": "simplified_likelihood_exporter",
        "bundle_kind": "promotion_readiness_evidence",
        "support_class": "research-grade",
        "generated_at_utc": now_utc(deterministic),
        "source_snapshot": {
            "benchmark_artifact": {
                "source_path": benchmark_artifact_entry["source_path"],
                "bundle_path": benchmark_artifact_entry["bundle_path"],
                "schema_version": str(benchmark.get("schema_version", "")),
                "sha256": benchmark_artifact_entry["sha256"],
                "bytes": benchmark_artifact_entry["bytes"],
            },
            "snapshot_report": {
                "source_path": snapshot_report_entry["source_path"],
                "bundle_path": snapshot_report_entry["bundle_path"],
                "schema_version": str(snapshot.get("schema_version", "")),
                "sha256": snapshot_report_entry["sha256"],
                "bytes": snapshot_report_entry["bytes"],
            },
            "snapshot_index": {
                "source_path": snapshot_index_entry["source_path"],
                "bundle_path": snapshot_index_entry["bundle_path"],
                "suite": str(current_index.get("suite", "")),
                "sha256": snapshot_index_entry["sha256"],
                "bytes": snapshot_index_entry["bytes"],
            },
            "summary": {
                "status": "ok" if research_grade_acceptance_supported else "fail",
                "benchmark_host": benchmark_host,
                "persisted_snapshot": snapshot.get("status") == "persisted",
                "all_schema_valid": bool(export_summary.get("all_schema_valid", False)),
                "all_fidelity_gates_pass": bool(
                    export_summary.get("all_fidelity_gates_pass", False)
                ),
                "all_performance_gates_pass": bool(
                    export_summary.get("all_performance_gates_pass", False)
                ),
                "export_matrix_case_count": int(export_summary.get("case_count", 0)),
                "synthetic_case_count": int(export_summary.get("synthetic_case_count", 0)),
                "public_reinterpretation_style_case_count": int(
                    export_summary.get("public_reinterpretation_style_case_count", 0)
                ),
                "max_abs_q_mu_diff": max_abs_q_mu_diff,
                "max_upper_limit_ratio_deviation": max_upper_limit_ratio_deviation,
                "min_net_end_to_end_upper_limit_speedup": min_net_end_to_end_upper_limit_speedup,
                "overall_min_net_end_to_end_upper_limit_speedup": overall_min_net_end_to_end_upper_limit_speedup,
                "public_reinterpretation_style_min_net_end_to_end_upper_limit_speedup": public_min_net_end_to_end_upper_limit_speedup,
            },
        },
        "exporter_surface": {
            "accepted_command_names": list(ACCEPTED_COMMAND_NAMES),
            "accepted_schema_names": list(ACCEPTED_SCHEMA_NAMES),
            "explicit_boundaries": list(EXPLICIT_BOUNDARIES),
        },
        "promotion_claims": {
            "research_grade_acceptance_supported": research_grade_acceptance_supported,
            "future_stable_review_ready": future_stable_review_ready,
            "committed_snapshot_supported": supports_committed_snapshot,
        },
        "artifacts": artifacts,
        "summary": {
            "status": "ok" if research_grade_acceptance_supported else "fail",
            "artifact_count": len(artifacts),
            "required_artifact_count": len(artifacts),
            "benchmark_host": benchmark_host,
            "supports_future_stable_review": future_stable_review_ready,
            "supports_committed_snapshot": supports_committed_snapshot,
        },
    }
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-artifact", type=Path, required=True)
    parser.add_argument("--snapshot-report", type=Path, required=True)
    parser.add_argument("--snapshot-index", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    bundle = _build_bundle(
        benchmark_artifact=args.benchmark_artifact,
        snapshot_report=args.snapshot_report,
        snapshot_index=args.snapshot_index,
        bundle_dir=args.bundle_dir,
        deterministic=bool(args.deterministic),
    )
    out_path = args.bundle_dir / "promotion_evidence.json"
    out_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Simplified-likelihood exporter promotion evidence bundle:",
        f"status={bundle['summary']['status']}",
        f"artifacts={bundle['summary']['artifact_count']}",
        "future_stable_review_ready="
        f"{str(bundle['promotion_claims']['future_stable_review_ready']).lower()}",
        sep=" ",
    )
    print(f"Bundle written to {args.bundle_dir}")
    return 0 if bundle["summary"]["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
