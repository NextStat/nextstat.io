#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _simplified_likelihood_exporter_promotion_bundle import (
    BUNDLE_SCHEMA_VERSION,
    CHECK_SCHEMA_VERSION,
    REQUIRED_BENCHMARK_HOST,
    REQUIRED_MAX_ABS_Q_MU_DIFF,
    REQUIRED_MAX_UPPER_LIMIT_RATIO_DEVIATION,
    REQUIRED_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP,
    REQUIRED_ROLES,
    REPO_ROOT,
    load_json,
    now_utc,
    sha256_path,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _step(*, status: str, ok: bool, errors: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "status": status,
        "ok": ok,
        "errors": errors,
        **extra,
    }


def _bundle_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    summary = bundle.get("summary") if isinstance(bundle.get("summary"), dict) else {}
    return {
        "schema_version": bundle.get("schema_version"),
        "status": summary.get("status"),
        "support_class": bundle.get("support_class"),
        "benchmark_host": summary.get("benchmark_host"),
        "supports_future_stable_review": bool(
            summary.get("supports_future_stable_review", False)
        ),
        "supports_committed_snapshot": bool(summary.get("supports_committed_snapshot", False)),
        "artifact_count": int(summary.get("artifact_count", 0)),
        "required_artifact_count": int(summary.get("required_artifact_count", 0)),
    }


def _schema_check(bundle: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    schema_path = (
        REPO_ROOT
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_promotion_evidence_bundle_v0.schema.json"
    )
    try:
        import jsonschema  # type: ignore
    except Exception as exc:
        errors.append(f"missing_jsonschema:{exc}")
        return _step(status="failed", ok=False, errors=errors, schema_path=str(schema_path))

    try:
        schema = load_json(schema_path)
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(instance=bundle, schema=schema)
    except Exception as exc:
        errors.append(f"bundle_schema_validation_failed:{exc}")

    return _step(
        status="passed" if not errors else "failed",
        ok=not errors,
        errors=errors,
        schema_path=str(schema_path),
    )


def _inventory_check(
    bundle_dir: Path, bundle: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    errors: list[str] = []
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, list):
        return (
            _step(
                status="failed",
                ok=False,
                errors=["artifacts_not_array"],
                artifact_count=0,
                hash_verified_count=0,
                byte_size_verified_count=0,
                missing_roles=list(REQUIRED_ROLES),
                duplicate_roles=[],
                missing_files=[],
                sha256_mismatches=[],
                byte_mismatches=[],
            ),
            {},
        )

    by_role: dict[str, dict[str, Any]] = {}
    duplicate_roles: list[str] = []
    missing_files: list[str] = []
    sha256_mismatches: list[str] = []
    byte_mismatches: list[str] = []
    hash_verified_count = 0
    byte_verified_count = 0

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            errors.append("artifact_entry_not_object")
            continue
        role = artifact.get("role")
        bundle_path_value = artifact.get("bundle_path")
        if not isinstance(role, str) or not role:
            errors.append("artifact_missing_role")
            continue
        if role in by_role:
            duplicate_roles.append(role)
        else:
            by_role[role] = artifact
        if not isinstance(bundle_path_value, str) or not bundle_path_value:
            errors.append(f"artifact_missing_bundle_path:{role}")
            continue
        target_path = bundle_dir / bundle_path_value
        if not target_path.exists():
            missing_files.append(f"{role}:{bundle_path_value}")
            continue
        actual_sha256 = sha256_path(target_path)
        expected_sha256 = artifact.get("sha256")
        if actual_sha256 != expected_sha256:
            sha256_mismatches.append(role)
        else:
            hash_verified_count += 1
        actual_bytes = target_path.stat().st_size
        expected_bytes = artifact.get("bytes")
        if actual_bytes != expected_bytes:
            byte_mismatches.append(role)
        else:
            byte_verified_count += 1

    missing_roles = [role for role in REQUIRED_ROLES if role not in by_role]
    errors.extend(f"duplicate_role:{role}" for role in duplicate_roles)
    errors.extend(f"missing_required_role:{role}" for role in missing_roles)
    errors.extend(f"missing_bundle_file:{item}" for item in missing_files)
    errors.extend(f"sha256_mismatch:{role}" for role in sha256_mismatches)
    errors.extend(f"byte_size_mismatch:{role}" for role in byte_mismatches)

    return (
        _step(
            status="passed" if not errors else "failed",
            ok=not errors,
            errors=errors,
            artifact_count=len(artifacts),
            hash_verified_count=hash_verified_count,
            byte_size_verified_count=byte_verified_count,
            missing_roles=missing_roles,
            duplicate_roles=duplicate_roles,
            missing_files=missing_files,
            sha256_mismatches=sha256_mismatches,
            byte_mismatches=byte_mismatches,
        ),
        by_role,
    )


def _consistency_check(bundle: dict[str, Any], by_role: dict[str, dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    artifacts = bundle.get("artifacts") if isinstance(bundle.get("artifacts"), list) else []
    summary = bundle.get("summary") if isinstance(bundle.get("summary"), dict) else {}
    claims = bundle.get("promotion_claims") if isinstance(bundle.get("promotion_claims"), dict) else {}
    source_snapshot = (
        bundle.get("source_snapshot") if isinstance(bundle.get("source_snapshot"), dict) else {}
    )
    source_summary = (
        source_snapshot.get("summary")
        if isinstance(source_snapshot.get("summary"), dict)
        else {}
    )
    benchmark_artifact = (
        source_snapshot.get("benchmark_artifact")
        if isinstance(source_snapshot.get("benchmark_artifact"), dict)
        else {}
    )
    snapshot_report = (
        source_snapshot.get("snapshot_report")
        if isinstance(source_snapshot.get("snapshot_report"), dict)
        else {}
    )
    snapshot_index = (
        source_snapshot.get("snapshot_index")
        if isinstance(source_snapshot.get("snapshot_index"), dict)
        else {}
    )

    artifact_count = len(artifacts)
    required_artifact_count = sum(
        1 for artifact in artifacts if isinstance(artifact, dict) and artifact.get("required") is True
    )

    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        errors.append(f"unexpected_bundle_schema:{bundle.get('schema_version')}")
    if bundle.get("support_class") != "research-grade":
        errors.append(f"unexpected_support_class:{bundle.get('support_class')}")
    if summary.get("artifact_count") != artifact_count:
        errors.append("artifact_count_mismatch")
    if summary.get("required_artifact_count") != required_artifact_count:
        errors.append("required_artifact_count_mismatch")

    def _check_source(role: str, payload: dict[str, Any]) -> None:
        artifact = by_role.get(role, {})
        if artifact:
            if payload.get("bundle_path") != artifact.get("bundle_path"):
                errors.append(f"{role}_bundle_path_mismatch")
            if payload.get("sha256") != artifact.get("sha256"):
                errors.append(f"{role}_sha256_mismatch")
            if payload.get("bytes") != artifact.get("bytes"):
                errors.append(f"{role}_bytes_mismatch")

    _check_source("benchmark_artifact", benchmark_artifact)
    _check_source("current_snapshot_report", snapshot_report)
    _check_source("current_snapshot_index", snapshot_index)

    expected_status = "ok" if source_summary.get("status") == "ok" else "fail"
    if summary.get("status") != expected_status:
        errors.append("summary_status_mismatch")
    if summary.get("supports_future_stable_review") != claims.get("future_stable_review_ready"):
        errors.append("supports_future_stable_review_mismatch")
    if summary.get("supports_committed_snapshot") != claims.get("committed_snapshot_supported"):
        errors.append("supports_committed_snapshot_mismatch")

    return _step(
        status="passed" if not errors else "failed",
        ok=not errors,
        errors=errors,
        artifact_count=artifact_count,
        required_artifact_count=required_artifact_count,
        expected_summary_status=expected_status,
    )


def _promotion_readiness_check(bundle: dict[str, Any], require_promotion_ready: bool) -> dict[str, Any]:
    summary = bundle.get("summary") if isinstance(bundle.get("summary"), dict) else {}
    claims = bundle.get("promotion_claims") if isinstance(bundle.get("promotion_claims"), dict) else {}
    source_snapshot = (
        bundle.get("source_snapshot") if isinstance(bundle.get("source_snapshot"), dict) else {}
    )
    source_summary = (
        source_snapshot.get("summary")
        if isinstance(source_snapshot.get("summary"), dict)
        else {}
    )

    actual_host = str(source_summary.get("benchmark_host", "unknown"))
    actual_min_net = float(source_summary.get("min_net_end_to_end_upper_limit_speedup", 0.0))
    actual_overall_min_net = float(
        source_summary.get(
            "overall_min_net_end_to_end_upper_limit_speedup",
            source_summary.get("min_net_end_to_end_upper_limit_speedup", 0.0),
        )
    )
    actual_public_min_net = float(
        source_summary.get(
            "public_reinterpretation_style_min_net_end_to_end_upper_limit_speedup", 0.0
        )
    )
    actual_max_abs_q_mu_diff = float(source_summary.get("max_abs_q_mu_diff", 0.0))
    actual_max_upper_limit_ratio_deviation = float(
        source_summary.get("max_upper_limit_ratio_deviation", 0.0)
    )
    actual_acceptance_supported = bool(
        claims.get("research_grade_acceptance_supported", False)
    )
    actual_future_review_ready = bool(claims.get("future_stable_review_ready", False))
    actual_committed_snapshot_supported = bool(
        claims.get("committed_snapshot_supported", False)
    )

    errors: list[str] = []
    status = "not_requested"
    ok = True
    if require_promotion_ready:
        status = "passed"
        if actual_host != REQUIRED_BENCHMARK_HOST:
            errors.append(f"unexpected_benchmark_host:{actual_host}")
        if summary.get("status") != "ok":
            errors.append(f"bundle_summary_not_ok:{summary.get('status')}")
        if not actual_acceptance_supported:
            errors.append("research_grade_acceptance_not_supported")
        if not actual_future_review_ready:
            errors.append("future_stable_review_not_ready")
        if not actual_committed_snapshot_supported:
            errors.append("committed_snapshot_not_supported")
        if actual_min_net < REQUIRED_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP:
            errors.append(
                "insufficient_net_end_to_end_upper_limit_speedup:"
                f"{actual_min_net:.6f}<{REQUIRED_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP:.1f}"
            )
        if actual_max_abs_q_mu_diff > REQUIRED_MAX_ABS_Q_MU_DIFF:
            errors.append(
                "max_abs_q_mu_diff_exceeds_threshold:"
                f"{actual_max_abs_q_mu_diff:.6f}>{REQUIRED_MAX_ABS_Q_MU_DIFF:.1f}"
            )
        if actual_max_upper_limit_ratio_deviation > REQUIRED_MAX_UPPER_LIMIT_RATIO_DEVIATION:
            errors.append(
                "max_upper_limit_ratio_deviation_exceeds_threshold:"
                f"{actual_max_upper_limit_ratio_deviation:.6f}>{REQUIRED_MAX_UPPER_LIMIT_RATIO_DEVIATION:.2f}"
            )
        ok = not errors
        status = "passed" if ok else "failed"

    return _step(
        status=status,
        ok=ok,
        errors=errors,
        required_benchmark_host=REQUIRED_BENCHMARK_HOST,
        required_min_net_end_to_end_upper_limit_speedup=REQUIRED_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP,
        required_max_abs_q_mu_diff=REQUIRED_MAX_ABS_Q_MU_DIFF,
        required_max_upper_limit_ratio_deviation=REQUIRED_MAX_UPPER_LIMIT_RATIO_DEVIATION,
        actual_benchmark_host=actual_host,
        actual_min_net_end_to_end_upper_limit_speedup=actual_min_net,
        actual_overall_min_net_end_to_end_upper_limit_speedup=actual_overall_min_net,
        actual_public_reinterpretation_style_min_net_end_to_end_upper_limit_speedup=actual_public_min_net,
        actual_max_abs_q_mu_diff=actual_max_abs_q_mu_diff,
        actual_max_upper_limit_ratio_deviation=actual_max_upper_limit_ratio_deviation,
        actual_research_grade_acceptance_supported=actual_acceptance_supported,
        actual_future_stable_review_ready=actual_future_review_ready,
        actual_committed_snapshot_supported=actual_committed_snapshot_supported,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-promotion-ready", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    bundle_dir = args.bundle_dir.resolve()
    bundle_json_path = bundle_dir / "promotion_evidence.json"
    bundle = load_json(bundle_json_path)

    schema_validation = _schema_check(bundle)
    inventory, by_role = _inventory_check(bundle_dir, bundle)
    consistency = _consistency_check(bundle, by_role)
    promotion_readiness = _promotion_readiness_check(bundle, args.require_promotion_ready)

    top_level_errors = [
        *schema_validation["errors"],
        *inventory["errors"],
        *consistency["errors"],
        *promotion_readiness["errors"],
    ]
    ok = not top_level_errors
    status = "passed" if ok else "failed"
    report = {
        "schema_version": CHECK_SCHEMA_VERSION,
        "surface": "simplified_likelihood_exporter",
        "checked_at_utc": now_utc(bool(args.deterministic)),
        "status": status,
        "ok": ok,
        "require_promotion_ready": bool(args.require_promotion_ready),
        "bundle_dir": str(bundle_dir),
        "bundle_json_path": str(bundle_json_path),
        "bundle_summary": _bundle_summary(bundle),
        "checks": {
            "schema_validation": schema_validation,
            "inventory": inventory,
            "consistency": consistency,
            "promotion_readiness": promotion_readiness,
        },
        "summary": {
            "top_level_errors": top_level_errors,
        },
    }
    _write_json(args.out.resolve(), report)
    print(f"Exporter promotion evidence check: status={status}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
