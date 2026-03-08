#!/usr/bin/env python3
"""Apex2 runner: simplified-likelihood fidelity and reinterpretation speedup report.

Methodology: Planning -> Exploration -> Execution -> Verification

This report compares paired deterministic synthetic cases:
  1. full HistFactory workspace with many Gaussian-constrained histosys nuisances
  2. covariance-form simplified likelihood derived from the same latent uncertainty modes

The output summarizes:
  - schema/runtime validation
  - factorization diagnostics
  - full-vs-simplified statistical fidelity
  - reduction and speedup gates
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

if os.environ.get("NEXTSTAT_PREFER_INSTALLED") == "1":
    _repo_root = Path(__file__).resolve().parents[1]
    _source_bindings = (_repo_root / "bindings" / "ns-py" / "python").resolve()
    sys.path[:] = [
        entry
        for entry in sys.path
        if not entry
        or (
            Path(entry).resolve() != _source_bindings
            if isinstance(entry, str)
            else True
        )
    ]

import nextstat

from _apex2_json import write_report_json

PY_TESTS_DIR = Path(__file__).resolve().parent / "python"
if str(PY_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(PY_TESTS_DIR))

from _simplified_likelihood_case_zoo import available_suite_names, make_suite  # noqa: E402
from _simplified_likelihood_export_public_case_catalog import (  # noqa: E402
    catalog_example_path as export_public_case_catalog_example_path,
    catalog_schema_path as export_public_case_catalog_schema_path,
    load_catalog as load_export_public_case_catalog,
    resolve_workspace_path as resolve_export_public_workspace_path,
)
from _simplified_likelihood_public_fixture_catalog import (  # noqa: E402
    catalog_example_path,
    catalog_schema_path,
    resolve_workspace_path,
)


REPORT_SCHEMA_VERSION = "nextstat_apex2_simplified_likelihood_report_v0"
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_SCHEMA_PATH = (
    REPO_ROOT
    / "docs"
    / "schemas"
    / "hep"
    / "simplified_likelihood_v0.schema.json"
)
AUDIT_SCHEMA_PATH = (
    REPO_ROOT
    / "docs"
    / "schemas"
    / "hep"
    / "simplified_likelihood_audit_v0.schema.json"
)
EXPORT_REPORT_SCHEMA_PATH = (
    REPO_ROOT
    / "docs"
    / "schemas"
    / "hep"
    / "simplified_likelihood_export_report_v0.schema.json"
)
PUBLIC_FIXTURE_CATALOG_PATH = catalog_example_path()
PUBLIC_FIXTURE_CATALOG_SCHEMA_PATH = catalog_schema_path()
EXPORT_PUBLIC_CASE_CATALOG_PATH = export_public_case_catalog_example_path()
EXPORT_PUBLIC_CASE_CATALOG_SCHEMA_PATH = export_public_case_catalog_schema_path()


def _measure_wall_s(fn, *, repeat: int) -> float:
    times: list[float] = []
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def _load_model(workspace_json: str) -> nextstat.HistFactoryModel:
    return nextstat.HistFactoryModel.from_workspace(workspace_json)


def _fit_metrics(model: nextstat.HistFactoryModel) -> dict[str, Any]:
    fit = nextstat.fit(model)
    poi_names = model.parameter_names()
    poi_index = poi_names.index("mu")
    return {
        "mu_hat": float(fit.bestfit[poi_index]),
        "sigma_mu": float(fit.uncertainties[poi_index]),
        "success": bool(fit.success),
        "nll": float(fit.nll),
        "n_iter": int(fit.n_iter),
    }


def _scan_metrics(model: nextstat.HistFactoryModel, *, mu_values: list[float]) -> dict[str, Any]:
    scan = nextstat.profile_scan(model, mu_values)
    q_mu = [float(point["q_mu"]) for point in scan["points"]]
    return {
        "mu_values": [float(mu) for mu in mu_values],
        "q_mu": q_mu,
        "mu_hat": float(scan["mu_hat"]),
        "nll_hat": float(scan["nll_hat"]),
    }


def _upper_limit_metrics(
    model: nextstat.HistFactoryModel,
    *,
    alpha: float,
    lo: float,
    hi: float,
    rtol: float,
    max_iter: int,
) -> dict[str, Any]:
    result = nextstat.upper_limit(
        model,
        method="root",
        alpha=alpha,
        lo=lo,
        hi=hi,
        rtol=rtol,
        max_iter=max_iter,
    )
    if isinstance(result, tuple):
        observed, expected = result
        return {
            "observed": float(observed),
            "expected": [float(value) for value in expected],
        }
    return {"observed": float(result), "expected": None}


def _validate_schema(
    *,
    instance: dict[str, Any],
    schema_path: Path,
) -> tuple[bool | None, bool]:
    try:
        import jsonschema  # type: ignore
    except Exception:
        return None, False

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=instance, schema=schema)
    return True, True


def _max_abs_diff(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return float("inf")
    return max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=0.0)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("inf")
    return float(numerator) / float(denominator)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_nextstat_cli_binary(cli_arg: str | None) -> Path | None:
    candidates: list[Path] = []
    if cli_arg:
        candidates.append(Path(cli_arg))
    env_binary = os.environ.get("NEXTSTAT_CLI_BINARY")
    if env_binary:
        candidates.append(Path(env_binary))
    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", "target"))
    candidates.extend(
        [
            target_dir / "debug" / "nextstat",
            target_dir / "release" / "nextstat",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _run_nextstat(cli_binary: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(cli_binary), *args],
        cwd=Path.cwd(),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=True,
    )


def _derive_config_for_case(case) -> dict[str, Any]:
    channels = [str(channel["name"]) for channel in case.workspace["channels"]]
    bins = [
        f"{observation['name']}/bin{idx}"
        for observation in case.workspace["observations"]
        for idx, _ in enumerate(observation["data"])
    ]
    return {
        "schema_version": "nextstat_simplified_likelihood_derive_v0",
        "source_workspace": {
            "format": "pyhf",
            "schema_version": "pyhf_workspace_v1",
            "poi_name": "mu",
        },
        "fit_result": {
            "schema_version": "nextstat_fit_result_v0",
            "background_state": "postfit_background",
        },
        "selection": {
            "channels": channels,
            "bins": bins,
        },
        "reduction": {
            "output_uncertainty_model": "basis",
            "basis_method": "eigen",
            "explained_variance_target": 0.999,
            "constraint_covariance_source": "source_model_constraints",
            "max_components": int(case.latent_rank),
            "split_stat_covariance": True,
        },
        "jacobian": {
            "method": "finite_difference",
            "relative_step": 0.01,
            "absolute_step_floor": 1e-6,
        },
        "fidelity_smoke": {
            "random_draws": 8,
            "qmu_test_mu": 1.0,
            "upper_limit_cl": 0.95,
        },
        "output_contract": {
            "schema_version": "nextstat_simplified_likelihood_v0",
            "require_factorization_diagnostics": True,
            "require_fidelity_diagnostics": True,
        },
    }


def _derive_config_for_public_export_case(case: dict[str, Any]) -> dict[str, Any]:
    reduction: dict[str, Any] = {
        "output_uncertainty_model": str(case["output_uncertainty_model"]),
        "basis_method": str(case["basis_method"]),
        "explained_variance_target": float(case["explained_variance_target"]),
        "constraint_covariance_source": str(case["constraint_covariance_source"]),
        "split_stat_covariance": bool(case["split_stat_covariance"]),
    }
    max_components = case.get("max_components")
    if max_components is not None:
        reduction["max_components"] = int(max_components)

    return {
        "schema_version": "nextstat_simplified_likelihood_derive_v0",
        "source_workspace": {
            "format": str(case["source_workspace_format"]),
            "schema_version": str(case["source_workspace_schema_version"]),
            "poi_name": str(case["poi_name"]),
        },
        "fit_result": {
            "schema_version": "nextstat_fit_result_v0",
            "background_state": "postfit_background",
        },
        "selection": {
            "channels": list(case["selection"]["channels"]),
            "bins": list(case["selection"]["bins"]),
        },
        "reduction": reduction,
        "jacobian": {
            "method": "finite_difference",
            "relative_step": 0.01,
            "absolute_step_floor": 1e-6,
        },
        "fidelity_smoke": {
            "random_draws": 8,
            "qmu_test_mu": 1.0,
            "upper_limit_cl": 0.95,
        },
        "output_contract": {
            "schema_version": "nextstat_simplified_likelihood_v0",
            "require_factorization_diagnostics": True,
            "require_fidelity_diagnostics": True,
        },
    }


def _synthetic_export_case_spec(case) -> dict[str, Any]:
    return {
        "name": str(case.name),
        "case_kind": "synthetic",
        "experiment": "Synthetic",
        "analysis_id": f"{case.name}.export_matrix",
        "reference": "internal-apex2-export-matrix",
        "source_workspace_path": None,
        "source_workspace_format": "pyhf",
        "source_workspace_schema_version": "pyhf_workspace_v1",
        "workspace": case.workspace,
        "derive_config": _derive_config_for_case(case),
    }


def _load_public_export_case_specs(*, catalog_path: Path) -> list[dict[str, Any]]:
    catalog = load_export_public_case_catalog()
    if catalog_path.resolve() != EXPORT_PUBLIC_CASE_CATALOG_PATH.resolve():
        catalog = _load_json(catalog_path)
    catalog_schema_valid, _jsonschema_available = _validate_schema(
        instance=catalog,
        schema_path=EXPORT_PUBLIC_CASE_CATALOG_SCHEMA_PATH,
    )
    if catalog_schema_valid is False:
        raise ValueError(
            f"public export case catalog failed schema validation: {catalog_path}"
        )

    specs: list[dict[str, Any]] = []
    for case in catalog["cases"]:
        workspace_path = (
            resolve_export_public_workspace_path(case)
            if catalog_path.resolve() == EXPORT_PUBLIC_CASE_CATALOG_PATH.resolve()
            else REPO_ROOT / case["workspace_json_path"]
        )
        specs.append(
            {
                "name": str(case["case_id"]),
                "case_kind": str(case["case_kind"]),
                "experiment": str(case["experiment"]),
                "analysis_id": str(case["analysis_id"]),
                "reference": str(case["reference"]),
                "source_workspace_path": str(case["workspace_json_path"]),
                "source_workspace_format": str(case["source_workspace_format"]),
                "source_workspace_schema_version": str(case["source_workspace_schema_version"]),
                "workspace": _load_json(workspace_path),
                "derive_config": _derive_config_for_public_export_case(case),
            }
        )
    return specs


def _all_finite(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _derived_fidelity_gate(fidelity: dict[str, Any] | None) -> bool:
    if fidelity is None:
        return False

    try:
        qmu_delta_smoke = float(fidelity["qmu_delta_smoke"])
        upper_limit_ratio_smoke = float(fidelity["upper_limit_ratio_smoke"])
        nuisance_count_full = float(fidelity["nuisance_count_full"])
        nuisance_count_reduced = float(fidelity["nuisance_count_reduced"])
    except (KeyError, TypeError, ValueError):
        return False

    reduced_fraction = _safe_ratio(nuisance_count_reduced, nuisance_count_full)
    return (
        math.isfinite(qmu_delta_smoke)
        and qmu_delta_smoke <= 0.1
        and math.isfinite(upper_limit_ratio_smoke)
        and 0.95 <= upper_limit_ratio_smoke <= 1.05
        and math.isfinite(reduced_fraction)
        and reduced_fraction <= 0.25
    )


def _public_fixture_case_report(
    *,
    fixture: dict[str, Any],
    mu_values: list[float],
    alpha: float,
    limit_hi: float,
    limit_rtol: float,
    limit_max_iter: int,
) -> dict[str, Any]:
    workspace_path = resolve_workspace_path(fixture)
    workspace = _load_json(workspace_path)
    workspace_json = json.dumps(workspace, sort_keys=True)

    input_schema_valid, jsonschema_available = _validate_schema(
        instance=workspace,
        schema_path=INPUT_SCHEMA_PATH,
    )

    audit = nextstat.workspace_audit(workspace_json)
    audit_schema_valid, audit_jsonschema_available = _validate_schema(
        instance=audit,
        schema_path=AUDIT_SCHEMA_PATH,
    )
    assert jsonschema_available == audit_jsonschema_available or not audit_jsonschema_available

    model = _load_model(workspace_json)
    fit = nextstat.fit(model)
    cls = float(nextstat.hypotest(1.0, model))
    upper_limit = _upper_limit_metrics(
        model,
        alpha=alpha,
        lo=0.0,
        hi=limit_hi,
        rtol=limit_rtol,
        max_iter=limit_max_iter,
    )
    scan = _scan_metrics(model, mu_values=mu_values)

    diagnostics = audit.get("diagnostics", {})
    embedded_diagnostics = (
        workspace.get("diagnostics") if isinstance(workspace.get("diagnostics"), dict) else {}
    )
    embedded_diagnostics = embedded_diagnostics if isinstance(embedded_diagnostics, dict) else {}
    factorization = embedded_diagnostics.get("factorization", diagnostics.get("factorization"))
    fidelity = embedded_diagnostics.get("fidelity", diagnostics.get("fidelity"))
    derived_fixture = fixture["source_format"] == "derived_from_workspace"
    runtime_gate = (
        bool(fit.success)
        and math.isfinite(float(fit.nll))
        and 0.0 <= cls <= 1.0
        and math.isfinite(float(upper_limit["observed"]))
        and math.isfinite(float(scan["mu_hat"]))
        and len(scan["q_mu"]) == len(mu_values)
        and _all_finite(scan["q_mu"])
    )
    derived_fidelity = _derived_fidelity_gate(fidelity) if derived_fixture else True
    schema_valid = input_schema_valid is not False and audit_schema_valid is not False
    stable_surface_ok = fixture["stable_commands"] == [
        "audit",
        "fit",
        "hypotest",
        "upper-limit",
        "scan",
    ]

    return {
        "fixture_id": fixture["fixture_id"],
        "title": fixture["title"],
        "experiment": fixture["experiment"],
        "analysis_id": fixture["analysis_id"],
        "reference": fixture["reference"],
        "source_format": fixture["source_format"],
        "provenance_kind": fixture["provenance_kind"],
        "ranking_semantics": fixture["ranking_semantics"],
        "workspace_json_path": fixture["workspace_json_path"],
        "stable_commands": list(fixture["stable_commands"]),
        "research_grade_commands": list(fixture["research_grade_commands"]),
        "validation": {
            "runtime_audit_ok": True,
            "jsonschema_available": bool(jsonschema_available),
            "input_schema_valid": input_schema_valid,
            "audit_schema_valid": audit_schema_valid,
            "schema_valid": schema_valid,
        },
        "runtime": {
            "fit_success": bool(fit.success),
            "hypotest_pvalue": cls,
            "upper_limit_observed": float(upper_limit["observed"]),
            "scan_mu_values": list(mu_values),
            "scan_q_mu": list(scan["q_mu"]),
            "scan_mu_hat": float(scan["mu_hat"]),
        },
        "diagnostics": {
            "factorization": factorization,
            "fidelity": fidelity,
        },
        "evidence": {
            "full_vs_simplified_fidelity_supported": derived_fixture,
            "embedded_fidelity_diagnostics_present": fidelity is not None,
        },
        "gates": {
            "schema": schema_valid,
            "runtime": runtime_gate,
            "stable_surface": stable_surface_ok,
            "derived_fidelity": derived_fidelity,
        },
        "status": (
            "ok"
            if schema_valid and runtime_gate and stable_surface_ok and derived_fidelity
            else "fail"
        ),
    }


def _public_fixture_matrix_report(
    *,
    catalog_path: Path,
    mu_values: list[float],
    alpha: float,
    limit_hi: float,
    limit_rtol: float,
    limit_max_iter: int,
) -> dict[str, Any]:
    catalog = _load_json(catalog_path)
    catalog_schema_valid, jsonschema_available = _validate_schema(
        instance=catalog,
        schema_path=PUBLIC_FIXTURE_CATALOG_SCHEMA_PATH,
    )

    cases = [
        _public_fixture_case_report(
            fixture=fixture,
            mu_values=mu_values,
            alpha=alpha,
            limit_hi=limit_hi,
            limit_rtol=limit_rtol,
            limit_max_iter=limit_max_iter,
        )
        for fixture in catalog["fixtures"]
    ]

    derived_cases = [case for case in cases if case["source_format"] == "derived_from_workspace"]
    summary = {
        "status": "ok" if all(case["status"] == "ok" for case in cases) else "fail",
        "catalog_schema_valid": catalog_schema_valid is not False,
        "jsonschema_available": bool(jsonschema_available),
        "fixture_count": len(cases),
        "all_schema_valid": all(case["validation"]["schema_valid"] for case in cases),
        "all_runtime_gates_pass": all(case["gates"]["runtime"] for case in cases),
        "all_stable_surface_gates_pass": all(case["gates"]["stable_surface"] for case in cases),
        "all_derived_fidelity_gates_pass": all(
            case["gates"]["derived_fidelity"] for case in derived_cases
        ),
        "derived_fixture_count": len(derived_cases),
        "fixtures_with_embedded_fidelity_evidence": sum(
            1
            for case in cases
            if case["evidence"]["embedded_fidelity_diagnostics_present"]
        ),
        "source_formats": sorted({str(case["source_format"]) for case in cases}),
    }

    return {
        "catalog_schema_version": catalog["schema_version"],
        "catalog_path": os.path.relpath(catalog_path, Path.cwd()),
        "cases": cases,
        "summary": summary,
    }


def _case_report(
    *,
    case,
    mu_values: list[float],
    fit_repeat: int,
    upper_limit_repeat: int,
    alpha: float,
    limit_hi: float,
    limit_rtol: float,
    limit_max_iter: int,
) -> dict[str, Any]:
    full_json = json.dumps(case.workspace, sort_keys=True)
    simplified_json = json.dumps(case.simplified_workspace, sort_keys=True)

    input_schema_valid, jsonschema_available = _validate_schema(
        instance=case.simplified_workspace,
        schema_path=INPUT_SCHEMA_PATH,
    )

    audit = nextstat.workspace_audit(simplified_json)
    audit_schema_valid, audit_jsonschema_available = _validate_schema(
        instance=audit,
        schema_path=AUDIT_SCHEMA_PATH,
    )
    assert jsonschema_available == audit_jsonschema_available or not audit_jsonschema_available

    full_model = _load_model(full_json)
    simplified_model = _load_model(simplified_json)

    full_fit = _fit_metrics(full_model)
    simplified_fit = _fit_metrics(simplified_model)
    full_scan = _scan_metrics(full_model, mu_values=mu_values)
    simplified_scan = _scan_metrics(simplified_model, mu_values=mu_values)
    full_limit = _upper_limit_metrics(
        full_model,
        alpha=alpha,
        lo=0.0,
        hi=limit_hi,
        rtol=limit_rtol,
        max_iter=limit_max_iter,
    )
    simplified_limit = _upper_limit_metrics(
        simplified_model,
        alpha=alpha,
        lo=0.0,
        hi=limit_hi,
        rtol=limit_rtol,
        max_iter=limit_max_iter,
    )

    delta_mu_hat = simplified_fit["mu_hat"] - full_fit["mu_hat"]
    sigma_mu_full = max(abs(full_fit["sigma_mu"]), 1e-12)
    delta_mu_hat_over_sigma_full = abs(delta_mu_hat) / sigma_mu_full
    max_abs_q_mu_diff = _max_abs_diff(full_scan["q_mu"], simplified_scan["q_mu"])
    upper_limit_ratio = _safe_ratio(
        simplified_limit["observed"],
        full_limit["observed"],
    )
    reduced_fraction = _safe_ratio(
        float(audit["reduced_nuisance_count"]),
        float(case.full_nuisance_count),
    )
    json_size_fraction = _safe_ratio(
        float(len(simplified_json.encode("utf-8"))),
        float(len(full_json.encode("utf-8"))),
    )

    bench = {
        "build_wall_s": {
            "full": _measure_wall_s(lambda: _load_model(full_json), repeat=1),
            "simplified": _measure_wall_s(lambda: _load_model(simplified_json), repeat=1),
        },
        "fit_wall_s": {
            "full": _measure_wall_s(lambda: nextstat.fit(full_model), repeat=fit_repeat),
            "simplified": _measure_wall_s(
                lambda: nextstat.fit(simplified_model), repeat=fit_repeat
            ),
        },
        "upper_limit_wall_s": {
            "full": _measure_wall_s(
                lambda: _upper_limit_metrics(
                    full_model,
                    alpha=alpha,
                    lo=0.0,
                    hi=limit_hi,
                    rtol=limit_rtol,
                    max_iter=limit_max_iter,
                ),
                repeat=upper_limit_repeat,
            ),
            "simplified": _measure_wall_s(
                lambda: _upper_limit_metrics(
                    simplified_model,
                    alpha=alpha,
                    lo=0.0,
                    hi=limit_hi,
                    rtol=limit_rtol,
                    max_iter=limit_max_iter,
                ),
                repeat=upper_limit_repeat,
            ),
        },
        "end_to_end_upper_limit_wall_s": {
            "full": _measure_wall_s(
                lambda: _upper_limit_metrics(
                    _load_model(full_json),
                    alpha=alpha,
                    lo=0.0,
                    hi=limit_hi,
                    rtol=limit_rtol,
                    max_iter=limit_max_iter,
                ),
                repeat=upper_limit_repeat,
            ),
            "simplified": _measure_wall_s(
                lambda: _upper_limit_metrics(
                    _load_model(simplified_json),
                    alpha=alpha,
                    lo=0.0,
                    hi=limit_hi,
                    rtol=limit_rtol,
                    max_iter=limit_max_iter,
                ),
                repeat=upper_limit_repeat,
            ),
        },
    }
    bench["speedup"] = {
        "build": _safe_ratio(bench["build_wall_s"]["full"], bench["build_wall_s"]["simplified"]),
        "fit": _safe_ratio(bench["fit_wall_s"]["full"], bench["fit_wall_s"]["simplified"]),
        "upper_limit": _safe_ratio(
            bench["upper_limit_wall_s"]["full"],
            bench["upper_limit_wall_s"]["simplified"],
        ),
        "end_to_end_upper_limit": _safe_ratio(
            bench["end_to_end_upper_limit_wall_s"]["full"],
            bench["end_to_end_upper_limit_wall_s"]["simplified"],
        ),
    }

    fidelity_passes = {
        "mu_hat": delta_mu_hat_over_sigma_full <= 0.05,
        "q_mu": max_abs_q_mu_diff <= 0.1,
        "upper_limit": 0.95 <= upper_limit_ratio <= 1.05,
    }
    performance_passes = {
        "reduced_nuisance_fraction": reduced_fraction <= 0.25,
        "json_size_fraction": json_size_fraction <= 0.35,
        "end_to_end_upper_limit_speedup": bench["speedup"]["end_to_end_upper_limit"] >= 3.0,
    }

    return {
        "name": case.name,
        "full": {
            "measurement_name": case.measurement,
            "channel_count": len(case.workspace["channels"]),
            "total_bins": sum(len(obs["data"]) for obs in case.workspace["observations"]),
            "parameter_count": int(full_model.n_params()),
            "nuisance_count": int(case.full_nuisance_count),
            "json_bytes": len(full_json.encode("utf-8")),
        },
        "simplified": {
            "schema_version": case.simplified_workspace["schema_version"],
            "channel_count": int(audit["channel_count"]),
            "total_bins": int(audit["total_bins"]),
            "parameter_count": int(simplified_model.n_params()),
            "reduced_nuisance_count": int(audit["reduced_nuisance_count"]),
            "latent_rank_target": int(case.latent_rank),
            "json_bytes": len(simplified_json.encode("utf-8")),
            "reduced_nuisance_fraction": reduced_fraction,
            "json_size_fraction": json_size_fraction,
        },
        "validation": {
            "runtime_audit_ok": True,
            "jsonschema_available": bool(jsonschema_available),
            "input_schema_valid": input_schema_valid,
            "audit_schema_valid": audit_schema_valid,
            "schema_valid": input_schema_valid is not False and audit_schema_valid is not False,
        },
        "factorization": audit["diagnostics"]["factorization"],
        "fidelity": {
            "mu_hat_full": full_fit["mu_hat"],
            "mu_hat_simplified": simplified_fit["mu_hat"],
            "sigma_mu_full": full_fit["sigma_mu"],
            "delta_mu_hat": delta_mu_hat,
            "delta_mu_hat_over_sigma_full": delta_mu_hat_over_sigma_full,
            "scan_mu_values": full_scan["mu_values"],
            "q_mu_full": full_scan["q_mu"],
            "q_mu_simplified": simplified_scan["q_mu"],
            "max_abs_q_mu_diff": max_abs_q_mu_diff,
            "upper_limit_full": full_limit["observed"],
            "upper_limit_simplified": simplified_limit["observed"],
            "upper_limit_ratio": upper_limit_ratio,
            "passes": fidelity_passes,
        },
        "gates": {
            "fidelity": fidelity_passes,
            "performance": performance_passes,
        },
        "bench": bench,
        "status": "ok" if all(fidelity_passes.values()) and all(performance_passes.values()) else "fail",
    }


def _export_matrix_case_report(
    *,
    export_case: dict[str, Any],
    nextstat_cli: Path,
    mu_values: list[float],
    alpha: float,
    limit_hi: float,
    limit_rtol: float,
    limit_max_iter: int,
    export_repeat: int,
    upper_limit_repeat: int,
    full_fit: dict[str, Any],
    full_scan: dict[str, Any],
    full_limit: dict[str, Any],
    full_end_to_end_upper_limit_wall_s: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"nextstat_export_matrix_{export_case['name']}_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        workspace_path = tmpdir / "workspace.json"
        fit_path = tmpdir / "fit.json"
        derive_config_path = tmpdir / "derive.json"
        simplified_path = tmpdir / "simplified.json"
        export_report_path = tmpdir / "export_report.json"

        workspace_path.write_text(
            json.dumps(export_case["workspace"], sort_keys=True, indent=2),
            encoding="utf-8",
        )
        derive_config_path.write_text(
            json.dumps(export_case["derive_config"], sort_keys=True, indent=2),
            encoding="utf-8",
        )
        _run_nextstat(
            nextstat_cli,
            "fit",
            "--input",
            str(workspace_path),
            "--output",
            str(fit_path),
            "--threads",
            "1",
        )

        def _run_export() -> None:
            _run_nextstat(
                nextstat_cli,
                "simplify",
                "workspace",
                "--input",
                str(workspace_path),
                "--fit",
                str(fit_path),
                "--derive-config",
                str(derive_config_path),
                "--experiment",
                str(export_case["experiment"]),
                "--analysis-id",
                str(export_case["analysis_id"]),
                "--reference",
                str(export_case["reference"]),
                "--output",
                str(simplified_path),
                "--report",
                str(export_report_path),
                "--threads",
                "1",
            )

        export_wall_s = _measure_wall_s(_run_export, repeat=export_repeat)
        exported_workspace = _load_json(simplified_path)
        exported_report = _load_json(export_report_path)
        exported_json = json.dumps(exported_workspace, sort_keys=True)
        exported_model = _load_model(exported_json)
        exported_audit = nextstat.workspace_audit(exported_json)

        input_schema_valid, jsonschema_available = _validate_schema(
            instance=exported_workspace,
            schema_path=INPUT_SCHEMA_PATH,
        )
        audit_schema_valid, audit_jsonschema_available = _validate_schema(
            instance=exported_audit,
            schema_path=AUDIT_SCHEMA_PATH,
        )
        export_report_schema_valid, export_report_jsonschema_available = _validate_schema(
            instance=exported_report,
            schema_path=EXPORT_REPORT_SCHEMA_PATH,
        )
        assert (
            jsonschema_available == audit_jsonschema_available or not audit_jsonschema_available
        )
        assert (
            jsonschema_available == export_report_jsonschema_available
            or not export_report_jsonschema_available
        )

        exported_fit = _fit_metrics(exported_model)
        exported_scan = _scan_metrics(exported_model, mu_values=mu_values)
        exported_limit = _upper_limit_metrics(
            exported_model,
            alpha=alpha,
            lo=0.0,
            hi=limit_hi,
            rtol=limit_rtol,
            max_iter=limit_max_iter,
        )
        exported_end_to_end_upper_limit_wall_s = _measure_wall_s(
            lambda: _upper_limit_metrics(
                _load_model(exported_json),
                alpha=alpha,
                lo=0.0,
                hi=limit_hi,
                rtol=limit_rtol,
                max_iter=limit_max_iter,
            ),
            repeat=upper_limit_repeat,
        )

        delta_mu_hat = exported_fit["mu_hat"] - full_fit["mu_hat"]
        sigma_mu_full = max(abs(full_fit["sigma_mu"]), 1e-12)
        delta_mu_hat_over_sigma_full = abs(delta_mu_hat) / sigma_mu_full
        max_abs_q_mu_diff = _max_abs_diff(full_scan["q_mu"], exported_scan["q_mu"])
        upper_limit_ratio = _safe_ratio(exported_limit["observed"], full_limit["observed"])

        total_end_to_end_upper_limit_wall_s = export_wall_s + exported_end_to_end_upper_limit_wall_s
        total_end_to_end_upper_limit_speedup = _safe_ratio(
            full_end_to_end_upper_limit_wall_s,
            total_end_to_end_upper_limit_wall_s,
        )
        output_summary = exported_report["output"]
        report_fidelity = exported_report["diagnostics"]["fidelity"]
        report_factorization = exported_report["diagnostics"]["factorization"]
        fidelity_passes = {
            "mu_hat": delta_mu_hat_over_sigma_full <= 0.05,
            "q_mu": max_abs_q_mu_diff <= 0.1,
            "upper_limit": 0.95 <= upper_limit_ratio <= 1.05,
        }
        performance_passes = {
            "net_end_to_end_upper_limit_speedup": (
                math.isfinite(total_end_to_end_upper_limit_speedup)
                and total_end_to_end_upper_limit_speedup >= 0.0
            ),
        }
        schema_valid = (
            input_schema_valid is not False
            and audit_schema_valid is not False
            and export_report_schema_valid is not False
        )

        return {
            "name": str(export_case["name"]),
            "case_kind": str(export_case["case_kind"]),
            "experiment": str(export_case["experiment"]),
            "analysis_id": str(export_case["analysis_id"]),
            "reference": str(export_case["reference"]),
            "source_workspace_path": export_case["source_workspace_path"],
            "source_workspace_format": str(export_case["source_workspace_format"]),
            "source_workspace_schema_version": str(export_case["source_workspace_schema_version"]),
            "source_fit_schema_version": "nextstat_fit_result_v0",
            "export_report_schema_version": exported_report["schema_version"],
            "validation": {
                "runtime_export_ok": True,
                "jsonschema_available": bool(jsonschema_available),
                "input_schema_valid": input_schema_valid,
                "audit_schema_valid": audit_schema_valid,
                "export_report_schema_valid": export_report_schema_valid,
                "schema_valid": schema_valid,
            },
            "output": {
                "schema_version": output_summary["schema_version"],
                "uncertainty_model_kind": output_summary["uncertainty_model_kind"],
                "bins_count": int(output_summary["bins_count"]),
                "full_nuisance_count": int(output_summary["full_nuisance_count"]),
                "reduced_nuisance_count": int(output_summary["reduced_nuisance_count"]),
                "reduction_ratio": float(output_summary["reduction_ratio"]),
                "json_bytes": int(output_summary["json_bytes"]),
                "json_sha256": str(output_summary["json_sha256"]),
            },
            "factorization": report_factorization,
            "report_fidelity": report_fidelity,
            "fidelity": {
                "mu_hat_full": full_fit["mu_hat"],
                "mu_hat_exported": exported_fit["mu_hat"],
                "sigma_mu_full": full_fit["sigma_mu"],
                "delta_mu_hat": delta_mu_hat,
                "delta_mu_hat_over_sigma_full": delta_mu_hat_over_sigma_full,
                "scan_mu_values": full_scan["mu_values"],
                "q_mu_full": full_scan["q_mu"],
                "q_mu_exported": exported_scan["q_mu"],
                "max_abs_q_mu_diff": max_abs_q_mu_diff,
                "upper_limit_full": full_limit["observed"],
                "upper_limit_exported": exported_limit["observed"],
                "upper_limit_ratio": upper_limit_ratio,
                "passes": fidelity_passes,
            },
            "bench": {
                "export_wall_s": export_wall_s,
                "exported_end_to_end_upper_limit_wall_s": exported_end_to_end_upper_limit_wall_s,
                "net_end_to_end_upper_limit_wall_s": total_end_to_end_upper_limit_wall_s,
                "speedup": {
                    "net_end_to_end_upper_limit": total_end_to_end_upper_limit_speedup,
                },
            },
            "gates": {
                "fidelity": fidelity_passes,
                "performance": performance_passes,
            },
            "status": (
                "ok"
                if schema_valid and all(fidelity_passes.values()) and all(performance_passes.values())
                else "fail"
            ),
        }


def _export_matrix_report(
    *,
    export_cases: list[dict[str, Any]],
    nextstat_cli: Path,
    mu_values: list[float],
    alpha: float,
    limit_hi: float,
    limit_rtol: float,
    limit_max_iter: int,
    export_repeat: int,
    upper_limit_repeat: int,
) -> dict[str, Any]:
    reports = []
    for export_case in export_cases:
        full_json = json.dumps(export_case["workspace"], sort_keys=True)
        full_model = _load_model(full_json)
        full_fit = _fit_metrics(full_model)
        full_scan = _scan_metrics(full_model, mu_values=mu_values)
        full_limit = _upper_limit_metrics(
            full_model,
            alpha=alpha,
            lo=0.0,
            hi=limit_hi,
            rtol=limit_rtol,
            max_iter=limit_max_iter,
        )
        full_end_to_end_upper_limit_wall_s = _measure_wall_s(
            lambda: _upper_limit_metrics(
                _load_model(full_json),
                alpha=alpha,
                lo=0.0,
                hi=limit_hi,
                rtol=limit_rtol,
                max_iter=limit_max_iter,
            ),
            repeat=upper_limit_repeat,
        )
        reports.append(
            _export_matrix_case_report(
                export_case=export_case,
                nextstat_cli=nextstat_cli,
                mu_values=mu_values,
                alpha=alpha,
                limit_hi=limit_hi,
                limit_rtol=limit_rtol,
                limit_max_iter=limit_max_iter,
                export_repeat=export_repeat,
                upper_limit_repeat=upper_limit_repeat,
                full_fit=full_fit,
                full_scan=full_scan,
                full_limit=full_limit,
                full_end_to_end_upper_limit_wall_s=full_end_to_end_upper_limit_wall_s,
            )
        )

    summary = {
        "status": "ok" if all(case["status"] == "ok" for case in reports) else "fail",
        "case_count": len(reports),
        "case_kinds": sorted({str(case["case_kind"]) for case in reports}),
        "synthetic_case_count": sum(
            1 for case in reports if case["case_kind"] == "synthetic"
        ),
        "public_reinterpretation_style_case_count": sum(
            1
            for case in reports
            if case["case_kind"] == "public_reinterpretation_style"
        ),
        "public_reinterpretation_style_case_names": [
            str(case["name"])
            for case in reports
            if case["case_kind"] == "public_reinterpretation_style"
        ],
        "all_schema_valid": all(case["validation"]["schema_valid"] for case in reports),
        "all_fidelity_gates_pass": all(all(case["gates"]["fidelity"].values()) for case in reports),
        "all_performance_gates_pass": all(
            all(case["gates"]["performance"].values()) for case in reports
        ),
        "max_abs_q_mu_diff": max(case["fidelity"]["max_abs_q_mu_diff"] for case in reports),
        "max_upper_limit_ratio_deviation": max(
            abs(case["fidelity"]["upper_limit_ratio"] - 1.0) for case in reports
        ),
        "min_net_end_to_end_upper_limit_speedup": min(
            case["bench"]["speedup"]["net_end_to_end_upper_limit"] for case in reports
        ),
    }
    return {
        "cli_binary": os.path.relpath(nextstat_cli, Path.cwd()),
        "cases": reports,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        default="ci",
        choices=available_suite_names(),
        help="Deterministic synthetic suite preset to run.",
    )
    parser.add_argument(
        "--scan",
        default="0.0,0.5,1.0,1.5,2.0,2.5,3.0",
        help="Comma-separated POI values for q_mu fidelity checks.",
    )
    parser.add_argument(
        "--fit-repeat",
        type=int,
        default=3,
        help="Number of fit timing repeats per case.",
    )
    parser.add_argument(
        "--upper-limit-repeat",
        type=int,
        default=2,
        help="Number of upper-limit timing repeats per case.",
    )
    parser.add_argument(
        "--export-repeat",
        type=int,
        default=1,
        help="Number of simplify/export timing repeats per case when the export matrix is enabled.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--limit-hi", type=float, default=5.0)
    parser.add_argument("--limit-rtol", type=float, default=1e-4)
    parser.add_argument("--limit-max-iter", type=int, default=80)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tmp/apex2_simplified_likelihood_report.json"),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Omit environment-specific metadata and write canonicalized JSON.",
    )
    parser.add_argument(
        "--include-public-fixtures",
        action="store_true",
        help="Append the curated public-style consume matrix to the Apex2 report.",
    )
    parser.add_argument(
        "--include-export-matrix",
        action="store_true",
        help="Append full -> derived -> reinterpret exporter measurements using the nextstat CLI binary.",
    )
    parser.add_argument(
        "--include-export-public-cases",
        action="store_true",
        help="Append curated public reinterpretation-style export cases to the exporter matrix.",
    )
    parser.add_argument(
        "--public-fixture-catalog",
        type=Path,
        default=PUBLIC_FIXTURE_CATALOG_PATH,
        help="Path to the curated public-style simplified-likelihood fixture catalog JSON.",
    )
    parser.add_argument(
        "--export-public-case-catalog",
        type=Path,
        default=EXPORT_PUBLIC_CASE_CATALOG_PATH,
        help="Path to the curated public-style exporter case catalog JSON.",
    )
    parser.add_argument(
        "--nextstat-cli",
        default=None,
        help="Path to the nextstat CLI binary used for export-matrix runs. Defaults to NEXTSTAT_CLI_BINARY or CARGO_TARGET_DIR lookups.",
    )
    args = parser.parse_args()

    mu_values = [float(value.strip()) for value in args.scan.split(",") if value.strip()]
    cases = make_suite(args.suite)
    nextstat_cli = None
    if args.include_export_matrix:
        nextstat_cli = _resolve_nextstat_cli_binary(args.nextstat_cli)
        if nextstat_cli is None:
            parser.error(
                "--include-export-matrix requires --nextstat-cli, NEXTSTAT_CLI_BINARY, or a discovered nextstat binary under CARGO_TARGET_DIR"
            )
    elif args.include_export_public_cases:
        parser.error("--include-export-public-cases requires --include-export-matrix")

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "stability": "apex2",
        "params": {
            "suite": args.suite,
            "scan": mu_values,
            "fit_repeat": int(args.fit_repeat),
            "upper_limit_repeat": int(args.upper_limit_repeat),
            "export_repeat": int(args.export_repeat),
            "alpha": float(args.alpha),
            "limit_hi": float(args.limit_hi),
            "limit_rtol": float(args.limit_rtol),
            "limit_max_iter": int(args.limit_max_iter),
            "gates": {
                "delta_mu_hat_over_sigma_full_max": 0.05,
                "max_abs_q_mu_diff": 0.1,
                "upper_limit_ratio_min": 0.95,
                "upper_limit_ratio_max": 1.05,
                "reduced_nuisance_fraction_max": 0.25,
                "json_size_fraction_max": 0.35,
                "end_to_end_upper_limit_speedup_min": 3.0,
                "export_net_end_to_end_upper_limit_speedup_min": 0.0,
            },
        },
        "cases": [],
        "summary": {},
    }
    if not args.deterministic:
        report["environment"] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "cwd": os.getcwd(),
            "nextstat_version": nextstat.__version__,
            "timestamp": int(time.time()),
        }

    for case in cases:
        report["cases"].append(
            _case_report(
                case=case,
                mu_values=mu_values,
                fit_repeat=args.fit_repeat,
                upper_limit_repeat=args.upper_limit_repeat,
                alpha=args.alpha,
                limit_hi=args.limit_hi,
                limit_rtol=args.limit_rtol,
                limit_max_iter=args.limit_max_iter,
            )
        )

    all_cases = report["cases"]
    summary_bench = {
        "min_speedup_end_to_end_upper_limit": min(
            case["bench"]["speedup"]["end_to_end_upper_limit"] for case in all_cases
        ),
        "min_speedup_fit": min(case["bench"]["speedup"]["fit"] for case in all_cases),
        "min_speedup_upper_limit": min(
            case["bench"]["speedup"]["upper_limit"] for case in all_cases
        ),
    }
    report["summary"] = {
        "status": "ok" if all(case["status"] == "ok" for case in all_cases) else "fail",
        "case_count": len(all_cases),
        "all_schema_valid": all(case["validation"]["schema_valid"] for case in all_cases),
        "all_fidelity_gates_pass": all(
            all(case["gates"]["fidelity"].values()) for case in all_cases
        ),
        "all_performance_gates_pass": all(
            all(case["gates"]["performance"].values()) for case in all_cases
        ),
        "max_delta_mu_hat_over_sigma_full": max(
            case["fidelity"]["delta_mu_hat_over_sigma_full"] for case in all_cases
        ),
        "max_abs_q_mu_diff": max(case["fidelity"]["max_abs_q_mu_diff"] for case in all_cases),
        "max_upper_limit_ratio_deviation": max(
            abs(case["fidelity"]["upper_limit_ratio"] - 1.0) for case in all_cases
        ),
        "max_reduced_nuisance_fraction": max(
            case["simplified"]["reduced_nuisance_fraction"] for case in all_cases
        ),
        "max_json_size_fraction": max(
            case["simplified"]["json_size_fraction"] for case in all_cases
        ),
        "bench": summary_bench,
    }

    if args.include_public_fixtures:
        public_fixture_matrix = _public_fixture_matrix_report(
            catalog_path=args.public_fixture_catalog,
            mu_values=mu_values,
            alpha=args.alpha,
            limit_hi=args.limit_hi,
            limit_rtol=args.limit_rtol,
            limit_max_iter=args.limit_max_iter,
        )
        report["public_fixture_matrix"] = public_fixture_matrix
        report["summary"]["public_fixture_matrix_included"] = True
        report["summary"]["public_fixture_matrix_status"] = public_fixture_matrix["summary"][
            "status"
        ]
        report["summary"]["public_fixture_matrix_fixture_count"] = public_fixture_matrix[
            "summary"
        ]["fixture_count"]
        if public_fixture_matrix["summary"]["status"] != "ok":
            report["summary"]["status"] = "fail"

    if args.include_export_matrix:
        assert nextstat_cli is not None
        export_cases = [_synthetic_export_case_spec(case) for case in cases]
        if args.include_export_public_cases:
            export_cases.extend(
                _load_public_export_case_specs(catalog_path=args.export_public_case_catalog)
            )
        export_matrix = _export_matrix_report(
            export_cases=export_cases,
            nextstat_cli=nextstat_cli,
            mu_values=mu_values,
            alpha=args.alpha,
            limit_hi=args.limit_hi,
            limit_rtol=args.limit_rtol,
            limit_max_iter=args.limit_max_iter,
            export_repeat=args.export_repeat,
            upper_limit_repeat=args.upper_limit_repeat,
        )
        report["export_matrix"] = export_matrix
        report["summary"]["export_matrix_included"] = True
        report["summary"]["export_matrix_status"] = export_matrix["summary"]["status"]
        report["summary"]["export_matrix_case_count"] = export_matrix["summary"]["case_count"]
        report["summary"]["export_matrix_case_kinds"] = export_matrix["summary"]["case_kinds"]
        report["summary"]["export_matrix_public_reinterpretation_style_case_count"] = export_matrix[
            "summary"
        ]["public_reinterpretation_style_case_count"]
        report["summary"]["export_matrix_min_net_end_to_end_upper_limit_speedup"] = export_matrix[
            "summary"
        ]["min_net_end_to_end_upper_limit_speedup"]
        if export_matrix["summary"]["status"] != "ok":
            report["summary"]["status"] = "fail"

    write_report_json(args.out, report, deterministic=bool(args.deterministic))

    print_args = [
        "Apex2 simplified-likelihood report:",
        f"suite={args.suite}",
        f"cases={report['summary']['case_count']}",
        f"status={report['summary']['status']}",
        f"max_delta_mu/sigma={report['summary']['max_delta_mu_hat_over_sigma_full']:.4f}",
        f"max_qmu_diff={report['summary']['max_abs_q_mu_diff']:.4f}",
        f"min_e2e_ul_speedup={summary_bench['min_speedup_end_to_end_upper_limit']:.2f}x",
    ]
    if args.include_public_fixtures:
        print_args.append(
            f"public_matrix_status={report['summary']['public_fixture_matrix_status']}"
        )
    if args.include_export_matrix:
        print_args.append(f"export_matrix_status={report['summary']['export_matrix_status']}")
        print_args.append(
            "export_net_e2e_ul_speedup="
            f"{report['summary']['export_matrix_min_net_end_to_end_upper_limit_speedup']:.2f}x"
        )
        if args.include_export_public_cases:
            print_args.append(
                "export_public_cases="
                f"{report['summary']['export_matrix_public_reinterpretation_style_case_count']}"
            )

    print(
        *print_args,
        sep=" ",
    )
    print(f"Report written to {args.out}")
    return 0 if report["summary"]["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
