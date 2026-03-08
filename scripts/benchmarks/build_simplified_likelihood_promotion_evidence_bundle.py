#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from _simplified_likelihood_promotion_bundle import (
    BUNDLE_SCHEMA_VERSION,
    PROMOTED_COMMANDS,
    REPO_ROOT,
    REQUIRED_BENCHMARK_HOST,
    REQUIRED_MIN_END_TO_END_UPPER_LIMIT_SPEEDUP,
    RESEARCH_GRADE_SURFACES,
    STATIC_ARTIFACTS,
    bundle_path_for,
    load_json,
    now_utc,
    relative_or_absolute,
    sha256_path,
)


def _copy_with_inventory(
    *,
    source_path: Path,
    bundle_dir: Path,
    role: str,
    kind: str,
    benchmark: bool = False,
) -> dict[str, Any]:
    if not source_path.exists():
        raise FileNotFoundError(f"missing required evidence artifact: {source_path}")
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


def _benchmark_host(report: dict[str, Any], benchmark_artifact: Path) -> str:
    env = report.get("environment", {})
    hostname = env.get("hostname")
    if isinstance(hostname, str) and hostname:
        return hostname
    source_path = relative_or_absolute(benchmark_artifact)
    for token in Path(source_path).parts:
        if token == REQUIRED_BENCHMARK_HOST:
            return token
    return "unknown"


def _build_bundle(
    *,
    benchmark_artifact: Path,
    bundle_dir: Path,
    deterministic: bool,
) -> dict[str, Any]:
    report = load_json(benchmark_artifact)
    public_matrix = report.get("public_fixture_matrix")
    if not isinstance(public_matrix, dict):
        raise ValueError(
            "benchmark artifact must include public_fixture_matrix; rerun Apex2 with --include-public-fixtures"
        )

    bundle_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        _copy_with_inventory(
            source_path=benchmark_artifact,
            bundle_dir=bundle_dir,
            role="benchmark_artifact",
            kind="benchmark_artifact",
            benchmark=True,
        )
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
    bench_summary = report["summary"]
    public_summary = public_matrix["summary"]
    benchmark_host = _benchmark_host(report, benchmark_artifact)
    min_speedup_end_to_end = float(bench_summary["bench"]["min_speedup_end_to_end_upper_limit"])
    supports_speedup_claim = (
        bench_summary["status"] == "ok"
        and benchmark_host == REQUIRED_BENCHMARK_HOST
        and min_speedup_end_to_end >= REQUIRED_MIN_END_TO_END_UPPER_LIMIT_SPEEDUP
    )

    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "surface": "simplified_likelihood",
        "bundle_kind": "promotion_evidence",
        "support_class": "stable",
        "generated_at_utc": now_utc(deterministic),
        "benchmark_evidence": {
            "source_path": benchmark_artifact_entry["source_path"],
            "bundle_path": benchmark_artifact_entry["bundle_path"],
            "schema_version": report["schema_version"],
            "sha256": benchmark_artifact_entry["sha256"],
            "bytes": benchmark_artifact_entry["bytes"],
            "summary": {
                "status": bench_summary["status"],
                "case_count": int(bench_summary["case_count"]),
                "all_schema_valid": bool(bench_summary["all_schema_valid"]),
                "all_fidelity_gates_pass": bool(bench_summary["all_fidelity_gates_pass"]),
                "all_performance_gates_pass": bool(bench_summary["all_performance_gates_pass"]),
                "min_speedup_fit": float(bench_summary["bench"]["min_speedup_fit"]),
                "min_speedup_upper_limit": float(bench_summary["bench"]["min_speedup_upper_limit"]),
                "min_speedup_end_to_end_upper_limit": min_speedup_end_to_end,
                "public_fixture_matrix_included": bool(
                    bench_summary.get("public_fixture_matrix_included", False)
                ),
                "public_fixture_matrix_status": str(
                    bench_summary.get("public_fixture_matrix_status", "fail")
                ),
                "public_fixture_matrix_fixture_count": int(
                    bench_summary.get("public_fixture_matrix_fixture_count", 0)
                ),
            },
            "public_fixture_matrix": {
                "status": public_summary["status"],
                "fixture_count": int(public_summary["fixture_count"]),
                "all_schema_valid": bool(public_summary["all_schema_valid"]),
                "all_runtime_gates_pass": bool(public_summary["all_runtime_gates_pass"]),
                "all_derived_fidelity_gates_pass": bool(
                    public_summary["all_derived_fidelity_gates_pass"]
                ),
                "jsonschema_available": bool(public_summary["jsonschema_available"]),
                "source_formats": list(public_summary["source_formats"]),
            },
        },
        "stable_subset": {
            "promoted_commands": list(PROMOTED_COMMANDS),
            "research_grade_surfaces": list(RESEARCH_GRADE_SURFACES),
        },
        "promotion_claims": {
            "stable_surface_supported": bench_summary["status"] == "ok",
            "speedup_target_supported": supports_speedup_claim,
            "public_fixture_matrix_supported": public_summary["status"] == "ok",
        },
        "artifacts": artifacts,
        "summary": {
            "status": "ok"
            if bench_summary["status"] == "ok" and public_summary["status"] == "ok"
            else "fail",
            "artifact_count": len(artifacts),
            "required_artifact_count": len(artifacts),
            "benchmark_host": benchmark_host,
            "supports_speedup_claim": supports_speedup_claim,
            "supports_public_fixture_matrix": public_summary["status"] == "ok",
        },
    }
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-artifact", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    bundle = _build_bundle(
        benchmark_artifact=args.benchmark_artifact,
        bundle_dir=args.bundle_dir,
        deterministic=bool(args.deterministic),
    )
    out_path = args.bundle_dir / "promotion_evidence.json"
    out_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Simplified-likelihood promotion evidence bundle:",
        f"status={bundle['summary']['status']}",
        f"artifacts={bundle['summary']['artifact_count']}",
        f"supports_speedup_claim={str(bundle['summary']['supports_speedup_claim']).lower()}",
        sep=" ",
    )
    print(f"Bundle written to {args.bundle_dir}")
    return 0 if bundle["summary"]["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
