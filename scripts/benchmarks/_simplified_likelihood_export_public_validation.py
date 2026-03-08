from __future__ import annotations

from pathlib import Path
from typing import Any

from _simplified_likelihood_export_benchmark import relative_or_absolute
from _simplified_likelihood_exporter_promotion_bundle import (
    REQUIRED_MAX_ABS_Q_MU_DIFF,
    REQUIRED_MAX_UPPER_LIMIT_RATIO_DEVIATION,
)
from _simplified_likelihood_promotion_bundle import REPO_ROOT, now_utc


REPORT_SCHEMA_VERSION = "nextstat_simplified_likelihood_export_public_validation_report_v0"
CURRENT_REPORT_FILENAME = "export_public_validation_report.json"
DEFAULT_CATALOG_PATH = (
    REPO_ROOT / "docs" / "specs" / "apex2_simplified_likelihood_export_public_case_catalog_v0.example.json"
)
PROMOTED_RUNTIME_CONSTRAINT_COVARIANCE_SOURCE = "source_model_constraints"
PUBLIC_VALIDATION_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP = 0.75
NON_GAUSSIAN_SOURCE_MODIFIER_TYPES = {"shapefactor", "shapesys", "staterror"}

def _resolve_workspace_path(catalog_case: dict[str, Any]) -> Path:
    raw_path = Path(str(catalog_case.get("workspace_json_path", "")))
    return raw_path if raw_path.is_absolute() else (REPO_ROOT / raw_path).resolve()


def _workspace_modifier_types(workspace_path: Path) -> list[str]:
    import json

    payload = json.loads(workspace_path.read_text(encoding="utf-8"))
    modifier_types = {
        str(modifier["type"])
        for channel in payload.get("channels", [])
        for sample in channel.get("samples", [])
        for modifier in sample.get("modifiers", [])
    }
    return sorted(modifier_types)


def _case_is_within_promoted_runtime_boundary(
    catalog_case: dict[str, Any],
) -> tuple[bool, str | None, list[str], bool]:
    observed_source = str(catalog_case.get("constraint_covariance_source", ""))
    if observed_source != PROMOTED_RUNTIME_CONSTRAINT_COVARIANCE_SOURCE:
        return (
            False,
            "constraint_covariance_source="
            f"{observed_source} stays outside the promoted stable exporter runtime boundary",
            [],
            False,
        )
    workspace_path = _resolve_workspace_path(catalog_case)
    modifier_types = _workspace_modifier_types(workspace_path)
    gaussian_only = not any(
        modifier_type in NON_GAUSSIAN_SOURCE_MODIFIER_TYPES for modifier_type in modifier_types
    )
    if not gaussian_only:
        return (
            False,
            "workspace modifier surface includes non-Gaussian or unsupported source constraints: "
            + ", ".join(modifier_types),
            modifier_types,
            False,
        )
    return True, None, modifier_types, True


def _matched_public_case(
    *,
    catalog_case: dict[str, Any],
    benchmark_cases_by_name: dict[str, dict[str, Any]],
    benchmark_cases_by_analysis_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    case_id = str(catalog_case.get("case_id", ""))
    analysis_id = str(catalog_case.get("analysis_id", ""))
    if case_id and case_id in benchmark_cases_by_name:
        return benchmark_cases_by_name[case_id]
    if analysis_id and analysis_id in benchmark_cases_by_analysis_id:
        return benchmark_cases_by_analysis_id[analysis_id]
    return None


def build_public_validation_report(
    *,
    benchmark_artifact_path: Path,
    benchmark: dict[str, Any],
    catalog_path: Path,
    catalog: dict[str, Any],
    deterministic: bool,
) -> dict[str, Any]:
    export_matrix = benchmark.get("export_matrix") if isinstance(benchmark.get("export_matrix"), dict) else {}
    export_summary = (
        export_matrix.get("summary") if isinstance(export_matrix.get("summary"), dict) else {}
    )
    benchmark_summary = (
        benchmark.get("summary") if isinstance(benchmark.get("summary"), dict) else {}
    )
    environment = (
        benchmark.get("environment") if isinstance(benchmark.get("environment"), dict) else {}
    )
    benchmark_cases = (
        export_matrix.get("cases") if isinstance(export_matrix.get("cases"), list) else []
    )
    public_benchmark_cases = [
        case
        for case in benchmark_cases
        if isinstance(case, dict) and case.get("case_kind") == "public_reinterpretation_style"
    ]
    benchmark_cases_by_name = {
        str(case.get("name")): case for case in public_benchmark_cases if case.get("name")
    }
    benchmark_cases_by_analysis_id = {
        str(case.get("analysis_id")): case
        for case in public_benchmark_cases
        if case.get("analysis_id")
    }

    catalog_cases = catalog.get("cases") if isinstance(catalog.get("cases"), list) else []
    matched_cases: list[dict[str, Any]] = []
    missing_catalog_case_ids: list[str] = []
    matched_names: set[str] = set()

    for raw_catalog_case in catalog_cases:
        if not isinstance(raw_catalog_case, dict):
            continue
        matched_case = _matched_public_case(
            catalog_case=raw_catalog_case,
            benchmark_cases_by_name=benchmark_cases_by_name,
            benchmark_cases_by_analysis_id=benchmark_cases_by_analysis_id,
        )
        case_id = str(raw_catalog_case.get("case_id", ""))
        if matched_case is None:
            if case_id:
                missing_catalog_case_ids.append(case_id)
            continue

        matched_names.add(str(matched_case.get("name", "")))
        workspace_path = _resolve_workspace_path(raw_catalog_case)
        within_boundary, boundary_reason, modifier_types, gaussian_only = (
            _case_is_within_promoted_runtime_boundary(raw_catalog_case)
        )
        fidelity = matched_case.get("fidelity") if isinstance(matched_case.get("fidelity"), dict) else {}
        gates = matched_case.get("gates") if isinstance(matched_case.get("gates"), dict) else {}
        fidelity_gates = gates.get("fidelity") if isinstance(gates.get("fidelity"), dict) else {}
        performance_gates = (
            gates.get("performance") if isinstance(gates.get("performance"), dict) else {}
        )
        output = matched_case.get("output") if isinstance(matched_case.get("output"), dict) else {}
        validation = (
            matched_case.get("validation") if isinstance(matched_case.get("validation"), dict) else {}
        )
        speedup = (
            matched_case.get("bench", {})
            .get("speedup", {})
            .get("net_end_to_end_upper_limit")
        )
        matched_cases.append(
            {
                "case_id": case_id,
                "name": str(matched_case.get("name", "")),
                "title": str(raw_catalog_case.get("title", "")),
                "analysis_id": str(matched_case.get("analysis_id", "")),
                "experiment": str(matched_case.get("experiment", "")),
                "reference": str(matched_case.get("reference", "")),
                "source_workspace_path": relative_or_absolute(workspace_path),
                "source_workspace_modifier_types": modifier_types,
                "gaussian_constrained_source_workspace": gaussian_only,
                "constraint_covariance_source": str(
                    raw_catalog_case.get("constraint_covariance_source", "")
                ),
                "within_promoted_stable_runtime_boundary": within_boundary,
                "outside_promoted_stable_runtime_boundary_reason": boundary_reason,
                "validation": {
                    "schema_valid": bool(validation.get("schema_valid", False)),
                    "input_schema_valid": bool(validation.get("input_schema_valid", False)),
                    "audit_schema_valid": bool(validation.get("audit_schema_valid", False)),
                    "export_report_schema_valid": bool(
                        validation.get("export_report_schema_valid", False)
                    ),
                    "runtime_export_ok": bool(validation.get("runtime_export_ok", False)),
                },
                "fidelity": {
                    "delta_mu_hat_over_sigma_full": float(
                        fidelity.get("delta_mu_hat_over_sigma_full", 0.0)
                    ),
                    "max_abs_q_mu_diff": float(fidelity.get("max_abs_q_mu_diff", 0.0)),
                    "upper_limit_ratio": float(fidelity.get("upper_limit_ratio", 0.0)),
                    "passes": {
                        "mu_hat": bool(fidelity_gates.get("mu_hat", False)),
                        "q_mu": bool(fidelity_gates.get("q_mu", False)),
                        "upper_limit": bool(fidelity_gates.get("upper_limit", False)),
                    },
                },
                "performance": {
                    "net_end_to_end_upper_limit_speedup": float(speedup or 0.0),
                    "passes": {
                        "net_end_to_end_upper_limit_speedup": float(speedup or 0.0)
                        >= PUBLIC_VALIDATION_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP
                    },
                },
                "output": {
                    "full_nuisance_count": int(output.get("full_nuisance_count", 0)),
                    "reduced_nuisance_count": int(output.get("reduced_nuisance_count", 0)),
                    "reduction_ratio": float(output.get("reduction_ratio", 0.0)),
                    "uncertainty_model_kind": str(output.get("uncertainty_model_kind", "")),
                },
            }
        )

    unexpected_public_case_names = sorted(
        name for name in benchmark_cases_by_name if name and name not in matched_names
    )
    observed_constraint_covariance_sources = sorted(
        {
            str(case.get("constraint_covariance_source", ""))
            for case in matched_cases
            if case.get("constraint_covariance_source")
        }
    )
    observed_source_workspace_modifier_types = sorted(
        {
            modifier_type
            for case in matched_cases
            for modifier_type in case.get("source_workspace_modifier_types", [])
        }
    )
    within_boundary_count = sum(
        1 for case in matched_cases if case["within_promoted_stable_runtime_boundary"]
    )
    outside_boundary_count = len(matched_cases) - within_boundary_count
    all_cases_within_boundary = bool(matched_cases) and outside_boundary_count == 0
    all_cases_gaussian_constrained_source_workspaces = all(
        bool(case["gaussian_constrained_source_workspace"]) for case in matched_cases
    )
    all_schema_valid = all(
        case["validation"]["schema_valid"]
        and case["validation"]["input_schema_valid"]
        and case["validation"]["audit_schema_valid"]
        and case["validation"]["export_report_schema_valid"]
        and case["validation"]["runtime_export_ok"]
        for case in matched_cases
    )
    all_fidelity_gates_pass = all(
        case["fidelity"]["passes"]["mu_hat"]
        and case["fidelity"]["passes"]["q_mu"]
        and case["fidelity"]["passes"]["upper_limit"]
        for case in matched_cases
    )
    all_performance_gates_pass = all(
        case["performance"]["passes"]["net_end_to_end_upper_limit_speedup"]
        for case in matched_cases
    )
    max_abs_q_mu_diff = max(
        (case["fidelity"]["max_abs_q_mu_diff"] for case in matched_cases),
        default=0.0,
    )
    max_upper_limit_ratio_deviation = max(
        (abs(case["fidelity"]["upper_limit_ratio"] - 1.0) for case in matched_cases),
        default=0.0,
    )
    min_net_speedup = min(
        (case["performance"]["net_end_to_end_upper_limit_speedup"] for case in matched_cases),
        default=0.0,
    )
    status = (
        "ok"
        if matched_cases
        and not missing_catalog_case_ids
        and not unexpected_public_case_names
        and all_cases_within_boundary
        and all_cases_gaussian_constrained_source_workspaces
        and all_schema_valid
        and all_fidelity_gates_pass
        and all_performance_gates_pass
        else "fail"
    )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "surface": "simplified_likelihood_export_public_validation",
        "status": status,
        "generated_at_utc": now_utc(deterministic),
        "benchmark_artifact_source_path": relative_or_absolute(benchmark_artifact_path),
        "benchmark_artifact_schema_version": str(benchmark.get("schema_version", "")),
        "catalog_path": relative_or_absolute(catalog_path),
        "catalog_schema_version": str(catalog.get("schema_version", "")),
        "boundary": {
            "surface_support_class": "stable-evidence",
            "does_not_expand_promoted_runtime_claim": True,
            "promoted_runtime_constraint_covariance_source": PROMOTED_RUNTIME_CONSTRAINT_COVARIANCE_SOURCE,
            "public_cases_outside_promoted_runtime_boundary_allowed": False,
        },
        "summary": {
            "benchmark_host": str(environment.get("hostname", "")),
            "benchmark_status": str(benchmark_summary.get("status", "")),
            "export_matrix_status": str(export_summary.get("status", "")),
            "catalog_case_count": len(catalog_cases),
            "public_case_count": len(matched_cases),
            "public_case_names": [str(case["name"]) for case in matched_cases],
            "missing_catalog_case_ids": missing_catalog_case_ids,
            "unexpected_public_case_names": unexpected_public_case_names,
            "all_schema_valid": all_schema_valid,
            "all_fidelity_gates_pass": all_fidelity_gates_pass,
            "all_performance_gates_pass": all_performance_gates_pass,
            "all_cases_within_promoted_stable_runtime_boundary": all_cases_within_boundary,
            "all_cases_gaussian_constrained_source_workspaces": all_cases_gaussian_constrained_source_workspaces,
            "max_abs_q_mu_diff": max_abs_q_mu_diff,
            "max_upper_limit_ratio_deviation": max_upper_limit_ratio_deviation,
            "min_net_end_to_end_upper_limit_speedup": min_net_speedup,
            "fidelity_thresholds": {
                "max_abs_q_mu_diff": REQUIRED_MAX_ABS_Q_MU_DIFF,
                "max_upper_limit_ratio_deviation": REQUIRED_MAX_UPPER_LIMIT_RATIO_DEVIATION,
            },
            "performance_thresholds": {
                "min_net_end_to_end_upper_limit_speedup": PUBLIC_VALIDATION_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP
            },
            "cases_within_promoted_stable_runtime_boundary": within_boundary_count,
            "cases_outside_promoted_stable_runtime_boundary": outside_boundary_count,
            "observed_constraint_covariance_sources": observed_constraint_covariance_sources,
            "observed_source_workspace_modifier_types": observed_source_workspace_modifier_types,
        },
        "cases": matched_cases,
    }
