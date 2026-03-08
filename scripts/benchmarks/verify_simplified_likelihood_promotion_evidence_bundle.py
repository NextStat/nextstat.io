#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _simplified_likelihood_promotion_bundle import (
    BUNDLE_SCHEMA_VERSION,
    CHECK_SCHEMA_VERSION,
    REPO_ROOT,
    REQUIRED_BENCHMARK_HOST,
    REQUIRED_MIN_END_TO_END_UPPER_LIMIT_SPEEDUP,
    REQUIRED_ROLES,
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


def _schema_check(bundle: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    schema_path = (
        REPO_ROOT
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_promotion_evidence_bundle_v0.schema.json"
    )
    try:
        import jsonschema  # type: ignore
    except Exception as exc:
        errors.append(f"missing_jsonschema:{exc}")
        return _step(
            status="failed",
            ok=False,
            errors=errors,
            schema_path=str(schema_path),
        )

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


def _inventory_check(bundle_dir: Path, bundle: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
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
    benchmark = (
        bundle.get("benchmark_evidence") if isinstance(bundle.get("benchmark_evidence"), dict) else {}
    )
    benchmark_summary = benchmark.get("summary") if isinstance(benchmark.get("summary"), dict) else {}
    public_summary = (
        benchmark.get("public_fixture_matrix")
        if isinstance(benchmark.get("public_fixture_matrix"), dict)
        else {}
    )
    benchmark_artifact = by_role.get("benchmark_artifact", {})

    artifact_count = len(artifacts)
    required_artifact_count = sum(
        1 for artifact in artifacts if isinstance(artifact, dict) and artifact.get("required") is True
    )

    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        errors.append(f"unexpected_bundle_schema:{bundle.get('schema_version')}")
    if summary.get("artifact_count") != artifact_count:
        errors.append("artifact_count_mismatch")
    if summary.get("required_artifact_count") != required_artifact_count:
        errors.append("required_artifact_count_mismatch")
    if benchmark_artifact:
        if benchmark.get("bundle_path") != benchmark_artifact.get("bundle_path"):
            errors.append("benchmark_bundle_path_mismatch")
        if benchmark.get("sha256") != benchmark_artifact.get("sha256"):
            errors.append("benchmark_sha256_mismatch")
        if benchmark.get("bytes") != benchmark_artifact.get("bytes"):
            errors.append("benchmark_bytes_mismatch")

    expected_status = (
        "ok"
        if benchmark_summary.get("status") == "ok" and public_summary.get("status") == "ok"
        else "fail"
    )
    if summary.get("status") != expected_status:
        errors.append("summary_status_mismatch")
    if summary.get("supports_speedup_claim") != claims.get("speedup_target_supported"):
        errors.append("supports_speedup_claim_mismatch")
    if summary.get("supports_public_fixture_matrix") != claims.get("public_fixture_matrix_supported"):
        errors.append("supports_public_fixture_matrix_mismatch")
    if claims.get("stable_surface_supported") != (benchmark_summary.get("status") == "ok"):
        errors.append("stable_surface_supported_mismatch")

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
    benchmark = (
        bundle.get("benchmark_evidence") if isinstance(bundle.get("benchmark_evidence"), dict) else {}
    )
    benchmark_summary = benchmark.get("summary") if isinstance(benchmark.get("summary"), dict) else {}
    public_summary = (
        benchmark.get("public_fixture_matrix")
        if isinstance(benchmark.get("public_fixture_matrix"), dict)
        else {}
    )

    actual_host = str(summary.get("benchmark_host", "unknown"))
    actual_min_e2e = float(benchmark_summary.get("min_speedup_end_to_end_upper_limit", 0.0))
    actual_public_status = str(public_summary.get("status", "fail"))
    actual_stable_surface_supported = bool(claims.get("stable_surface_supported", False))
    actual_speedup_target_supported = bool(claims.get("speedup_target_supported", False))

    errors: list[str] = []
    status = "not_requested"
    ok = True
    if require_promotion_ready:
        status = "passed"
        if actual_host != REQUIRED_BENCHMARK_HOST:
            errors.append(f"unexpected_benchmark_host:{actual_host}")
        if summary.get("status") != "ok":
            errors.append(f"bundle_summary_not_ok:{summary.get('status')}")
        if not actual_stable_surface_supported:
            errors.append("stable_surface_not_supported")
        if actual_public_status != "ok":
            errors.append(f"public_fixture_matrix_not_ok:{actual_public_status}")
        if actual_min_e2e < REQUIRED_MIN_END_TO_END_UPPER_LIMIT_SPEEDUP:
            errors.append(
                "insufficient_end_to_end_upper_limit_speedup:"
                f"{actual_min_e2e:.6f}<{REQUIRED_MIN_END_TO_END_UPPER_LIMIT_SPEEDUP:.1f}"
            )
        if not actual_speedup_target_supported:
            errors.append("speedup_target_not_supported")
        ok = not errors
        status = "passed" if ok else "failed"

    return _step(
        status=status,
        ok=ok,
        errors=errors,
        required_benchmark_host=REQUIRED_BENCHMARK_HOST,
        actual_benchmark_host=actual_host,
        required_min_end_to_end_upper_limit_speedup=REQUIRED_MIN_END_TO_END_UPPER_LIMIT_SPEEDUP,
        actual_min_end_to_end_upper_limit_speedup=actual_min_e2e,
        actual_public_fixture_matrix_status=actual_public_status,
        actual_stable_surface_supported=actual_stable_surface_supported,
        actual_speedup_target_supported=actual_speedup_target_supported,
    )


def _bundle_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    summary = bundle.get("summary") if isinstance(bundle.get("summary"), dict) else {}
    return {
        "schema_version": bundle.get("schema_version"),
        "status": summary.get("status"),
        "benchmark_host": summary.get("benchmark_host"),
        "supports_speedup_claim": bool(summary.get("supports_speedup_claim", False)),
        "supports_public_fixture_matrix": bool(
            summary.get("supports_public_fixture_matrix", False)
        ),
        "artifact_count": int(summary.get("artifact_count", 0)),
        "required_artifact_count": int(summary.get("required_artifact_count", 0)),
    }


def _build_report(
    *,
    bundle_dir: Path,
    bundle_json_path: Path,
    require_promotion_ready: bool,
    deterministic: bool,
) -> dict[str, Any]:
    bundle = load_json(bundle_json_path)
    schema_check = _schema_check(bundle)
    inventory_check, by_role = _inventory_check(bundle_dir, bundle)
    consistency_check = _consistency_check(bundle, by_role)
    promotion_readiness = _promotion_readiness_check(bundle, require_promotion_ready)

    top_level_errors = [
        *schema_check["errors"],
        *inventory_check["errors"],
        *consistency_check["errors"],
        *promotion_readiness["errors"],
    ]
    ok = (
        schema_check["ok"]
        and inventory_check["ok"]
        and consistency_check["ok"]
        and promotion_readiness["ok"]
    )
    status = "passed" if ok else "failed"

    return {
        "schema_version": CHECK_SCHEMA_VERSION,
        "surface": "simplified_likelihood",
        "checked_at_utc": now_utc(deterministic),
        "status": status,
        "ok": ok,
        "require_promotion_ready": require_promotion_ready,
        "bundle_dir": str(bundle_dir),
        "bundle_json_path": str(bundle_json_path),
        "bundle_summary": _bundle_summary(bundle),
        "checks": {
            "schema_validation": schema_check,
            "inventory": inventory_check,
            "consistency": consistency_check,
            "promotion_readiness": promotion_readiness,
        },
        "summary": {
            "top_level_errors": top_level_errors,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--bundle-json",
        type=Path,
        default=None,
        help="Override promotion_evidence.json path; defaults to <bundle-dir>/promotion_evidence.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output report path; defaults to <bundle-dir>/promotion_evidence_check.json",
    )
    parser.add_argument("--require-promotion-ready", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    bundle_dir = args.bundle_dir
    bundle_json_path = args.bundle_json or (bundle_dir / "promotion_evidence.json")
    out_path = args.out or (bundle_dir / "promotion_evidence_check.json")

    report = _build_report(
        bundle_dir=bundle_dir,
        bundle_json_path=bundle_json_path,
        require_promotion_ready=bool(args.require_promotion_ready),
        deterministic=bool(args.deterministic),
    )
    _write_json(out_path, report)
    print(
        "Simplified-likelihood promotion evidence check:",
        f"status={report['status']}",
        f"require_promotion_ready={str(report['require_promotion_ready']).lower()}",
        f"errors={len(report['summary']['top_level_errors'])}",
        sep=" ",
    )
    print(f"Report written to {out_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
