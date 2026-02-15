#!/usr/bin/env python3
"""Validate benchmark JSON artifacts against local JSON schemas.

This is intended to be the single entry point for schema validation across all
benchmark suites and snapshot artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_MAP = {
    # Core HEP/pyhf parity suites
    "nextstat.benchmark_result.v1": "manifests/schema/benchmark_result_v1.schema.json",
    "nextstat.benchmark_suite_result.v1": "manifests/schema/benchmark_suite_result_v1.schema.json",
    "nextstat.pharma_benchmark_result.v1": "manifests/schema/pharma_benchmark_result_v1.schema.json",
    "nextstat.pharma_benchmark_suite_result.v1": "manifests/schema/pharma_benchmark_suite_result_v1.schema.json",
    # Bayesian (NUTS) suite
    "nextstat.bayesian_benchmark_result.v1": "manifests/schema/bayesian_benchmark_result_v1.schema.json",
    "nextstat.bayesian_benchmark_suite_result.v1": "manifests/schema/bayesian_benchmark_suite_result_v1.schema.json",
    "nextstat.bayesian_multiseed_summary.v1": "manifests/schema/bayesian_multiseed_summary_v1.schema.json",
    # ML suite
    "nextstat.ml_benchmark_result.v1": "manifests/schema/ml_benchmark_result_v1.schema.json",
    "nextstat.ml_benchmark_suite_result.v1": "manifests/schema/ml_benchmark_suite_result_v1.schema.json",
    # Econometrics suite
    "nextstat.econometrics_benchmark_result.v1": "manifests/schema/econometrics_benchmark_result_v1.schema.json",
    "nextstat.econometrics_benchmark_suite_result.v1": "manifests/schema/econometrics_benchmark_suite_result_v1.schema.json",
    # Additional suites
    "nextstat.glm_benchmark_result.v1": "manifests/schema/glm_benchmark_result_v1.schema.json",
    "nextstat.glm_benchmark_suite_result.v1": "manifests/schema/glm_benchmark_suite_result_v1.schema.json",
    "nextstat.survival_benchmark_result.v1": "manifests/schema/survival_benchmark_result_v1.schema.json",
    "nextstat.survival_benchmark_suite_result.v1": "manifests/schema/survival_benchmark_suite_result_v1.schema.json",
    "nextstat.timeseries_benchmark_result.v1": "manifests/schema/timeseries_benchmark_result_v1.schema.json",
    "nextstat.timeseries_benchmark_suite_result.v1": "manifests/schema/timeseries_benchmark_suite_result_v1.schema.json",
    "nextstat.evt_benchmark_result.v1": "manifests/schema/evt_benchmark_result_v1.schema.json",
    "nextstat.evt_benchmark_suite_result.v1": "manifests/schema/evt_benchmark_suite_result_v1.schema.json",
    "nextstat.insurance_benchmark_result.v1": "manifests/schema/insurance_benchmark_result_v1.schema.json",
    "nextstat.insurance_benchmark_suite_result.v1": "manifests/schema/insurance_benchmark_suite_result_v1.schema.json",
    "nextstat.meta_analysis_benchmark_result.v1": "manifests/schema/meta_analysis_benchmark_result_v1.schema.json",
    "nextstat.meta_analysis_benchmark_suite_result.v1": "manifests/schema/meta_analysis_benchmark_suite_result_v1.schema.json",
    "nextstat.mams_benchmark_result.v1": "manifests/schema/mams_benchmark_result_v1.schema.json",
    "nextstat.mams_benchmark_suite_result.v1": "manifests/schema/mams_benchmark_suite_result_v1.schema.json",
    "nextstat.montecarlo_safety_benchmark_result.v1": "manifests/schema/montecarlo_safety_benchmark_result_v1.schema.json",
    "nextstat.montecarlo_safety_benchmark_suite_result.v1": "manifests/schema/montecarlo_safety_benchmark_suite_result_v1.schema.json",
    # Snapshot artifacts
    "nextstat.baseline_manifest.v1": "manifests/schema/baseline_manifest_v1.schema.json",
    "nextstat.snapshot_index.v1": "manifests/schema/snapshot_index_v1.schema.json",
    "nextstat.replication_report.v1": "manifests/schema/replication_report_v1.schema.json",
    # Baselines
    "nextstat.hep_root_baseline_result.v1": "manifests/schema/hep_root_baseline_result_v1.schema.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _schema_path(repo_root: Path, schema_version: str) -> Path | None:
    rel = SCHEMA_MAP.get(schema_version)
    if not rel:
        return None
    return (repo_root / rel).resolve()


def _validate_instance(instance_path: Path, schema_path: Path) -> None:
    import jsonschema  # type: ignore

    schema = _load_json(schema_path)
    inst = _load_json(instance_path)
    jsonschema.validate(inst, schema)


def _iter_case_paths(obj: dict[str, Any]) -> list[str]:
    cases = obj.get("cases")
    if not isinstance(cases, list):
        return []
    out: list[str] = []
    for e in cases:
        if not isinstance(e, dict):
            continue
        p = e.get("path")
        if isinstance(p, str) and p.strip():
            out.append(p)
    return out


def validate_artifact(path: Path, repo_root: Path, strict: bool) -> int:
    try:
        obj = _load_json(path)
    except Exception as e:
        print(f"invalid json: {path}: {type(e).__name__}:{e}", file=sys.stderr)
        return 1

    if not isinstance(obj, dict):
        if strict:
            print(f"invalid artifact root (expected object): {path}", file=sys.stderr)
            return 1
        return 0

    sv = obj.get("schema_version")
    if not isinstance(sv, str) or not sv.strip():
        if strict:
            print(f"missing schema_version: {path}", file=sys.stderr)
            return 1
        return 0

    schema_path = _schema_path(repo_root, sv)
    if schema_path is None:
        if strict or sv.startswith("nextstat."):
            print(f"unknown schema_version={sv}: {path}", file=sys.stderr)
            return 1
        return 0

    try:
        _validate_instance(path, schema_path)
    except Exception as e:
        print(f"schema fail: {path}: {type(e).__name__}:{e}", file=sys.stderr)
        return 1

    # If this is a suite index, validate referenced case artifacts too.
    base = path.parent
    rc = 0
    for rel in _iter_case_paths(obj):
        case_path = (base / rel).resolve()
        rc |= validate_artifact(case_path, repo_root, strict=strict)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate NextStat benchmark artifacts (JSON schemas).")
    ap.add_argument("paths", nargs="+", help="Artifact JSON files and/or directories.")
    ap.add_argument("--strict", action="store_true", help="Fail on JSONs missing schema_version.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    targets: list[Path] = []
    for p in args.paths:
        pp = Path(p).resolve()
        if pp.is_dir():
            targets.extend(sorted(pp.rglob("*.json")))
        else:
            targets.append(pp)

    rc = 0
    for t in targets:
        rc |= validate_artifact(t, repo_root, strict=bool(args.strict))
    return 0 if rc == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
