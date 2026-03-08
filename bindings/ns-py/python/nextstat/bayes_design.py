"""Bayesian clinical trial design helpers."""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
import random
from statistics import NormalDist
import tempfile
import textwrap
from typing import Any

try:
    from . import _core as _core  # type: ignore
except ImportError:  # pragma: no cover
    _core = None  # type: ignore

from . import audit as _audit

_beta_binomial_design_analyze_json = getattr(
    _core, "beta_binomial_design_analyze_json", None
)
_beta_binomial_design_operating_characteristics_json = getattr(
    _core, "beta_binomial_design_operating_characteristics_json", None
)
_beta_binomial_design_posterior_predictive_json = getattr(
    _core, "beta_binomial_design_posterior_predictive_json", None
)
_beta_binomial_design_prior_sensitivity_json = getattr(
    _core, "beta_binomial_design_prior_sensitivity_json", None
)
_beta_binomial_design_report_json = getattr(
    _core, "beta_binomial_design_report_json", None
)
_beta_binomial_design_report_markdown = getattr(
    _core, "beta_binomial_design_report_markdown", None
)
_normal_normal_design_analyze_json = getattr(
    _core, "normal_normal_design_analyze_json", None
)
_normal_normal_design_operating_characteristics_json = getattr(
    _core, "normal_normal_design_operating_characteristics_json", None
)
_normal_normal_design_posterior_predictive_json = getattr(
    _core, "normal_normal_design_posterior_predictive_json", None
)
_normal_normal_design_prior_sensitivity_json = getattr(
    _core, "normal_normal_design_prior_sensitivity_json", None
)
_normal_normal_design_report_json = getattr(
    _core, "normal_normal_design_report_json", None
)
_normal_normal_design_report_markdown = getattr(
    _core, "normal_normal_design_report_markdown", None
)

_REGULATORY_APPENDIX_SECTION_ORDER = [
    "design_summary",
    "prior_specification",
    "decision_rules",
    "current_analysis",
    "operating_characteristics",
    "posterior_predictive",
    "prior_sensitivity",
    "provenance",
]

_REGULATORY_APPENDIX_ENDPOINT_SUMMARY = {
    "beta_binomial": "binary endpoint with exact beta-binomial conjugate updating",
    "normal_normal": "continuous endpoint with exact normal-normal conjugate updating",
}

_CONFLICT_SEVERITY_ORDER = {"low": 0, "moderate": 1, "high": 2}
_STANDARD_NORMAL = NormalDist()


def _coerce_json_payload(payload_or_path: dict[str, Any] | str | Path) -> str:
    if isinstance(payload_or_path, dict):
        return json.dumps(payload_or_path)
    if isinstance(payload_or_path, Path):
        return payload_or_path.read_text(encoding="utf-8")
    text = str(payload_or_path)
    path = Path(text)
    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
    except OSError:
        pass
    return text


def _coerce_json_object(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    name: str,
) -> dict[str, Any]:
    payload = json.loads(_coerce_json_payload(payload_or_path))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must decode to a JSON object")
    return payload


def _coerce_json_artifact(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    inline_name: str,
) -> tuple[str, str]:
    if isinstance(payload_or_path, dict):
        return (json.dumps(payload_or_path, indent=2) + "\n", inline_name)
    if isinstance(payload_or_path, Path):
        return (payload_or_path.read_text(encoding="utf-8"), str(payload_or_path))
    text = str(payload_or_path)
    path = Path(text)
    try:
        if path.exists():
            return (path.read_text(encoding="utf-8"), str(path))
    except OSError:
        pass
    parsed = json.loads(text)
    return (json.dumps(parsed, indent=2) + "\n", inline_name)


def _validate_design_report(
    report: dict[str, Any],
    *,
    expected_family: str,
    expected_schema_version: str,
) -> None:
    if report.get("design_family") != expected_family:
        raise ValueError(f"report.design_family must be {expected_family}")
    if report.get("schema_version") != expected_schema_version:
        raise ValueError(f"report.schema_version must be {expected_schema_version}")
    for key in (
        "design_spec",
        "current_analysis",
        "operating_characteristics",
        "posterior_predictive",
        "prior_sensitivity",
        "provenance",
    ):
        if key not in report:
            raise ValueError(f"report.{key} is required")


def _validate_frozen_design_report_contract(report: dict[str, Any]) -> None:
    """Fail fast on stale or cross-wired frozen report payloads.

    These builders operate on a frozen design report, not a bag of loosely-related
    sections. If a caller mutates or mixes `current_analysis`, `prior_sensitivity`,
    or `provenance` from different runs, we prefer an explicit validation error over
    a deterministic but misleading downstream artifact.
    """

    design_spec = report["design_spec"]
    current_analysis = report["current_analysis"]
    prior_sensitivity = report["prior_sensitivity"]
    provenance = report["provenance"]

    design_id = str(design_spec["design_id"])
    current_design_id = str(current_analysis["design_id"])
    prior_design_id = str(prior_sensitivity["design_id"])
    if current_design_id != design_id:
        raise ValueError(
            "report.current_analysis.design_id must match report.design_spec.design_id"
        )
    if prior_design_id != design_id:
        raise ValueError(
            "report.prior_sensitivity.design_id must match report.design_spec.design_id"
        )

    current_look = current_analysis["look"]
    prior_look = prior_sensitivity["look"]
    if _json_clone(current_look) != _json_clone(prior_look):
        raise ValueError(
            "report.prior_sensitivity.look must match report.current_analysis.look"
        )

    current_observed = current_analysis["observed"]
    prior_observed = prior_sensitivity["observed"]
    if _json_clone(current_observed) != _json_clone(prior_observed):
        raise ValueError(
            "report.prior_sensitivity.observed must match report.current_analysis.observed"
        )

    current_look_id = str(current_look["id"])
    prior_look_id = str(prior_look["id"])
    if str(current_observed["look_id"]) != current_look_id:
        raise ValueError(
            "report.current_analysis.observed.look_id must match report.current_analysis.look.id"
        )
    if str(prior_observed["look_id"]) != prior_look_id:
        raise ValueError(
            "report.prior_sensitivity.observed.look_id must match report.prior_sensitivity.look.id"
        )

    analysis_schema_version = provenance.get("analysis_schema_version")
    if str(current_analysis["schema_version"]) != str(analysis_schema_version):
        raise ValueError(
            "report.current_analysis.schema_version must match provenance.analysis_schema_version"
        )

    prior_sensitivity_schema_version = provenance.get(
        "prior_sensitivity_report_schema_version"
    )
    if str(prior_sensitivity["schema_version"]) != str(
        prior_sensitivity_schema_version
    ):
        raise ValueError(
            "report.prior_sensitivity.schema_version must match provenance.prior_sensitivity_report_schema_version"
        )


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _planned_total_sample_size(looks: Any) -> int:
    if not isinstance(looks, list) or not looks:
        raise ValueError("report.design_spec.looks must contain at least one look")
    last_look = looks[-1]
    if not isinstance(last_look, dict):
        raise ValueError("report.design_spec.looks entries must be JSON objects")
    return int(last_look["n_control"]) + int(last_look["n_treatment"])


def _build_regulatory_appendix(
    report_or_path: dict[str, Any] | str | Path,
    *,
    expected_family: str,
    expected_schema_version: str,
) -> dict[str, Any]:
    report_json = _coerce_json_payload(report_or_path)
    report = json.loads(report_json)
    if not isinstance(report, dict):
        raise ValueError("report must decode to a JSON object")
    _validate_design_report(
        report,
        expected_family=expected_family,
        expected_schema_version=expected_schema_version,
    )

    design_spec = report["design_spec"]
    current_analysis = report["current_analysis"]
    operating_characteristics = report["operating_characteristics"]
    posterior_predictive = report["posterior_predictive"]
    prior_sensitivity = report["prior_sensitivity"]
    provenance = report["provenance"]

    looks = design_spec["looks"]
    planned_looks = [
        {
            "id": str(look["id"]),
            "n_control": int(look["n_control"]),
            "n_treatment": int(look["n_treatment"]),
        }
        for look in looks
    ]
    prior_variants = prior_sensitivity.get("variants", [])
    baseline_variant_id = "baseline"
    for variant in prior_variants:
        if variant.get("is_baseline"):
            baseline_variant_id = str(variant["variant_id"])
            break

    sections = {
        "design_summary": {
            "design_id": str(design_spec["design_id"]),
            "design_family": expected_family,
            "endpoint_summary": _REGULATORY_APPENDIX_ENDPOINT_SUMMARY[expected_family],
            "planned_looks": planned_looks,
            "total_planned_sample_size": _planned_total_sample_size(looks),
            "current_look_id": str(current_analysis["look"]["id"]),
        },
        "prior_specification": {
            "control_prior": _json_clone(design_spec["control_prior"]),
            "treatment_prior": _json_clone(design_spec["treatment_prior"]),
            "likelihood": _json_clone(design_spec.get("likelihood")),
        },
        "decision_rules": _json_clone(design_spec["decision_rules"]),
        "current_analysis": {
            "look_id": str(current_analysis["look"]["id"]),
            "recommended_action": str(
                current_analysis["decision"]["recommended_action"]
            ),
            "posterior_probability_gt_margin": float(
                current_analysis["decision"]["posterior_probability_gt_margin"]
            ),
            "treatment_effect_margin": float(current_analysis["decision"]["margin"]),
            "posterior_effect_summary": _json_clone(
                current_analysis["posterior"]["effect_difference"]
            ),
        },
        "operating_characteristics": {
            "n_replicates": int(operating_characteristics["n_replicates"]),
            "seed": int(operating_characteristics["seed"]),
            "scenario_summaries": [
                {
                    "scenario_id": str(scenario["scenario_id"]),
                    "success_rate": float(scenario["success_rate"]),
                    "futility_rate": float(scenario["futility_rate"]),
                    "no_decision_rate": float(scenario["no_decision_rate"]),
                    "expected_total_sample_size": float(
                        scenario["expected_total_sample_size"]
                    ),
                }
                for scenario in operating_characteristics["scenarios"]
            ],
        },
        "posterior_predictive": {
            "n_replicates": int(posterior_predictive["n_replicates"]),
            "seed": int(posterior_predictive["seed"]),
            "eventual_success_probability": float(
                posterior_predictive["eventual_success_probability"]
            ),
            "eventual_futility_probability": float(
                posterior_predictive["eventual_futility_probability"]
            ),
            "eventual_no_decision_probability": float(
                posterior_predictive["eventual_no_decision_probability"]
            ),
            "expected_total_sample_size": float(
                posterior_predictive["expected_total_sample_size"]
            ),
            "future_look_summaries": [
                {
                    "look_id": str(item["look_id"]),
                    "conditional_stop_probability": float(
                        item["conditional_stop_probability"]
                    ),
                    "conditional_success_probability": float(
                        item["conditional_success_probability"]
                    ),
                    "conditional_futility_probability": float(
                        item["conditional_futility_probability"]
                    ),
                }
                for item in posterior_predictive["future_look_summaries"]
            ],
        },
        "prior_sensitivity": {
            "n_replicates": int(prior_sensitivity["n_replicates"]),
            "seed": int(prior_sensitivity["seed"]),
            "baseline_variant_id": baseline_variant_id,
            "variant_summaries": [
                {
                    "variant_id": str(variant["variant_id"]),
                    "recommended_action": str(variant["recommended_action"]),
                    "posterior_probability_gt_margin": float(
                        variant["posterior_probability_gt_margin"]
                    ),
                    "eventual_success_probability": float(
                        variant["eventual_success_probability"]
                    ),
                    "eventual_futility_probability": float(
                        variant["eventual_futility_probability"]
                    ),
                    "eventual_no_decision_probability": float(
                        variant["eventual_no_decision_probability"]
                    ),
                    "expected_total_sample_size": float(
                        variant["expected_total_sample_size"]
                    ),
                    "posterior_probability_delta_vs_baseline": float(
                        variant["posterior_probability_delta_vs_baseline"]
                    ),
                    "eventual_success_probability_delta_vs_baseline": float(
                        variant["eventual_success_probability_delta_vs_baseline"]
                    ),
                }
                for variant in prior_variants
            ],
        },
        "provenance": _json_clone(provenance),
    }

    return {
        "schema_version": "nextstat_bayesian_design_regulatory_appendix_v0",
        "stability": str(report.get("stability", "research-grade")),
        "appendix_id": f"{design_spec['design_id']}_regulatory_appendix_v0",
        "design_family": expected_family,
        "design_id": str(design_spec["design_id"]),
        "source_report_schema_version": str(report["schema_version"]),
        "generated_from_frozen_report": True,
        "required_sections": list(_REGULATORY_APPENDIX_SECTION_ORDER),
        "section_order": list(_REGULATORY_APPENDIX_SECTION_ORDER),
        "sections": sections,
    }


def _validate_regulatory_appendix(appendix: dict[str, Any]) -> None:
    if (
        appendix.get("schema_version")
        != "nextstat_bayesian_design_regulatory_appendix_v0"
    ):
        raise ValueError(
            "appendix.schema_version must be nextstat_bayesian_design_regulatory_appendix_v0"
        )
    design_family = appendix.get("design_family")
    if design_family not in _REGULATORY_APPENDIX_ENDPOINT_SUMMARY:
        raise ValueError(
            "appendix.design_family must be beta_binomial or normal_normal"
        )
    for key in (
        "appendix_id",
        "design_id",
        "source_report_schema_version",
        "generated_from_frozen_report",
        "required_sections",
        "section_order",
        "sections",
    ):
        if key not in appendix:
            raise ValueError(f"appendix.{key} is required")

    required_sections = appendix["required_sections"]
    section_order = appendix["section_order"]
    sections = appendix["sections"]
    if required_sections != _REGULATORY_APPENDIX_SECTION_ORDER:
        raise ValueError(
            "appendix.required_sections must match the published v0 section order"
        )
    if section_order != _REGULATORY_APPENDIX_SECTION_ORDER:
        raise ValueError(
            "appendix.section_order must match the published v0 section order"
        )
    if not isinstance(sections, dict):
        raise ValueError("appendix.sections must be a JSON object")
    for section_name in _REGULATORY_APPENDIX_SECTION_ORDER:
        if section_name not in sections:
            raise ValueError(f"appendix.sections.{section_name} is required")


def _coerce_regulatory_appendix(
    appendix_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    appendix_json = _coerce_json_payload(appendix_or_path)
    appendix = json.loads(appendix_json)
    if not isinstance(appendix, dict):
        raise ValueError("appendix must decode to a JSON object")
    _validate_regulatory_appendix(appendix)
    return appendix


def _build_prior_conflict_diagnostic(
    report_or_path: dict[str, Any] | str | Path,
    *,
    expected_family: str,
    expected_schema_version: str,
) -> dict[str, Any]:
    report_json = _coerce_json_payload(report_or_path)
    report = json.loads(report_json)
    if not isinstance(report, dict):
        raise ValueError("report must decode to a JSON object")
    _validate_design_report(
        report,
        expected_family=expected_family,
        expected_schema_version=expected_schema_version,
    )
    _validate_frozen_design_report_contract(report)

    prior_sensitivity = report["prior_sensitivity"]
    variants = prior_sensitivity.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("report.prior_sensitivity.variants must be non-empty")

    baseline_candidates = [
        variant for variant in variants if bool(variant.get("is_baseline"))
    ]
    if not baseline_candidates:
        baseline_candidates = [
            variant
            for variant in variants
            if str(variant.get("variant_id")) == "baseline"
        ]
    if len(baseline_candidates) != 1:
        raise ValueError(
            "report.prior_sensitivity.variants must contain exactly one baseline variant"
        )
    baseline_variant = baseline_candidates[0]
    baseline_variant_id = str(baseline_variant["variant_id"])
    baseline_action = str(baseline_variant["recommended_action"])

    posterior_probabilities = [
        float(variant["posterior_probability_gt_margin"]) for variant in variants
    ]
    eventual_success_probabilities = [
        float(variant["eventual_success_probability"]) for variant in variants
    ]
    expected_total_sample_sizes = [
        float(variant["expected_total_sample_size"]) for variant in variants
    ]

    baseline_probability = float(baseline_variant["posterior_probability_gt_margin"])
    success_threshold = float(
        report["current_analysis"]["decision"]["success_threshold"]
    )
    futility_threshold = float(
        report["current_analysis"]["decision"]["futility_threshold"]
    )
    nearest_threshold_margin = min(
        abs(baseline_probability - success_threshold),
        abs(baseline_probability - futility_threshold),
    )

    posterior_probability_range = max(posterior_probabilities) - min(
        posterior_probabilities
    )
    eventual_success_probability_range = max(eventual_success_probabilities) - min(
        eventual_success_probabilities
    )
    expected_total_sample_size_range = max(expected_total_sample_sizes) - min(
        expected_total_sample_sizes
    )
    planned_total_sample_size = float(
        _planned_total_sample_size(report["design_spec"]["looks"])
    )
    expected_total_sample_size_range_fraction = (
        expected_total_sample_size_range / planned_total_sample_size
        if planned_total_sample_size > 0.0
        else 0.0
    )

    max_abs_probability_delta = max(
        abs(float(variant["posterior_probability_gt_margin"]) - baseline_probability)
        for variant in variants
    )
    baseline_success_probability = float(
        baseline_variant["eventual_success_probability"]
    )
    max_abs_eventual_success_delta = max(
        abs(
            float(variant["eventual_success_probability"])
            - baseline_success_probability
        )
        for variant in variants
    )
    baseline_expected_total_sample_size = float(
        baseline_variant["expected_total_sample_size"]
    )
    max_abs_expected_total_sample_size_delta = max(
        abs(
            float(variant["expected_total_sample_size"])
            - baseline_expected_total_sample_size
        )
        for variant in variants
    )

    action_flip_variant_ids = [
        str(variant["variant_id"])
        for variant in variants
        if str(variant["variant_id"]) != baseline_variant_id
        and str(variant["recommended_action"]) != baseline_action
    ]
    recommended_action_flip_count = len(action_flip_variant_ids)
    decision_margin_ratio = (
        posterior_probability_range / nearest_threshold_margin
        if nearest_threshold_margin > 0.0
        else None
    )

    moderate_decision_margin_ratio_threshold = 0.5
    moderate_eventual_success_probability_range_threshold = 0.25
    moderate_expected_total_sample_size_range_fraction_threshold = 0.25

    if recommended_action_flip_count > 0:
        conflict_severity = "high"
        rationale = [
            "Recommended action flips for at least one published prior variant.",
            (
                "Flip variants: "
                + ", ".join(f"`{variant_id}`" for variant_id in action_flip_variant_ids)
                + "."
            ),
        ]
    elif nearest_threshold_margin <= 0.0 and posterior_probability_range > 0.0:
        conflict_severity = "high"
        rationale = [
            "Baseline posterior probability is already on a decision threshold.",
            "Any non-zero prior-driven posterior-probability spread is decision-critical.",
        ]
    elif (
        decision_margin_ratio is not None
        and decision_margin_ratio >= moderate_decision_margin_ratio_threshold
    ):
        conflict_severity = "moderate"
        rationale = [
            "Prior-driven posterior-probability spread consumes at least half of the baseline distance to the nearest decision threshold.",
            (
                "Posterior-probability spread is "
                f"`{posterior_probability_range:.6f}` with a nearest decision margin of "
                f"`{nearest_threshold_margin:.6f}`."
            ),
        ]
    elif (
        eventual_success_probability_range
        >= moderate_eventual_success_probability_range_threshold
    ):
        conflict_severity = "moderate"
        rationale = [
            "Prior-driven eventual-success spread exceeds the published moderate threshold.",
            (f"Eventual-success spread is `{eventual_success_probability_range:.6f}`."),
        ]
    elif (
        expected_total_sample_size_range_fraction
        >= moderate_expected_total_sample_size_range_fraction_threshold
    ):
        conflict_severity = "moderate"
        rationale = [
            "Prior-driven expected sample-size spread exceeds the published moderate threshold relative to planned enrollment.",
            (
                "Expected sample-size spread fraction is "
                f"`{expected_total_sample_size_range_fraction:.6f}` of planned enrollment."
            ),
        ]
    else:
        conflict_severity = "low"
        rationale = [
            "Published prior variants do not flip the baseline recommended action.",
            (
                "Posterior-probability spread is "
                f"`{posterior_probability_range:.6f}` against a nearest decision margin of "
                f"`{nearest_threshold_margin:.6f}`."
            ),
        ]

    return {
        "schema_version": "nextstat_bayesian_prior_conflict_diagnostic_v0",
        "stability": str(report.get("stability", "research-grade")),
        "diagnostic_id": f"{report['design_spec']['design_id']}_prior_conflict_diagnostic_v0",
        "design_family": expected_family,
        "design_id": str(report["design_spec"]["design_id"]),
        "source_report_schema_version": str(report["schema_version"]),
        "source_prior_sensitivity_schema_version": str(
            prior_sensitivity["schema_version"]
        ),
        "generated_from_frozen_report": True,
        "baseline_variant_id": baseline_variant_id,
        "baseline_recommended_action": baseline_action,
        "reported_variant_count": len(variants),
        "conflict_severity": conflict_severity,
        "decision_instability": recommended_action_flip_count > 0,
        "thresholds": {
            "high_recommended_action_flip_count_threshold": 1,
            "moderate_decision_margin_ratio_threshold": moderate_decision_margin_ratio_threshold,
            "moderate_eventual_success_probability_range_threshold": moderate_eventual_success_probability_range_threshold,
            "moderate_expected_total_sample_size_range_fraction_threshold": moderate_expected_total_sample_size_range_fraction_threshold,
        },
        "metrics": {
            "baseline_posterior_probability_gt_margin": baseline_probability,
            "success_threshold": success_threshold,
            "futility_threshold": futility_threshold,
            "nearest_decision_threshold_margin": nearest_threshold_margin,
            "decision_margin_ratio": decision_margin_ratio,
            "posterior_probability_range": posterior_probability_range,
            "eventual_success_probability_range": eventual_success_probability_range,
            "expected_total_sample_size_range": expected_total_sample_size_range,
            "expected_total_sample_size_range_fraction_of_plan": expected_total_sample_size_range_fraction,
            "max_abs_posterior_probability_delta_vs_baseline": max_abs_probability_delta,
            "max_abs_eventual_success_probability_delta_vs_baseline": max_abs_eventual_success_delta,
            "max_abs_expected_total_sample_size_delta_vs_baseline": max_abs_expected_total_sample_size_delta,
            "recommended_action_flip_count": recommended_action_flip_count,
            "recommended_action_flip_variant_ids": action_flip_variant_ids,
        },
        "rationale": rationale,
        "variant_summaries": [
            {
                "variant_id": str(variant["variant_id"]),
                "is_baseline": bool(variant.get("is_baseline", False)),
                "recommended_action": str(variant["recommended_action"]),
                "posterior_probability_gt_margin": float(
                    variant["posterior_probability_gt_margin"]
                ),
                "eventual_success_probability": float(
                    variant["eventual_success_probability"]
                ),
                "expected_total_sample_size": float(
                    variant["expected_total_sample_size"]
                ),
                "posterior_probability_delta_vs_baseline": float(
                    float(variant["posterior_probability_gt_margin"])
                    - baseline_probability
                ),
                "eventual_success_probability_delta_vs_baseline": float(
                    float(variant["eventual_success_probability"])
                    - baseline_success_probability
                ),
                "expected_total_sample_size_delta_vs_baseline": float(
                    float(variant["expected_total_sample_size"])
                    - baseline_expected_total_sample_size
                ),
            }
            for variant in variants
        ],
    }


def _validate_historical_control_borrowing_policy(
    policy: dict[str, Any],
    *,
    expected_family: str,
) -> None:
    if (
        policy.get("schema_version")
        != "nextstat_bayesian_historical_control_borrowing_policy_v0"
    ):
        raise ValueError(
            "policy.schema_version must be "
            "nextstat_bayesian_historical_control_borrowing_policy_v0"
        )
    if policy.get("design_family") != expected_family:
        raise ValueError(f"policy.design_family must be {expected_family}")
    policy_id = policy.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("policy.policy_id must be a non-empty string")

    borrowing_model = policy.get("borrowing_model")
    expected_model = (
        "power_prior" if expected_family == "beta_binomial" else "commensurate"
    )
    if borrowing_model != expected_model:
        raise ValueError(
            f"policy.borrowing_model must be {expected_model} for {expected_family}"
        )

    historical_sources = policy.get("historical_sources")
    if not isinstance(historical_sources, list) or not historical_sources:
        raise ValueError("policy.historical_sources must be a non-empty array")
    for idx, source in enumerate(historical_sources):
        if not isinstance(source, dict):
            raise ValueError(f"policy.historical_sources[{idx}] must be a JSON object")
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(
                f"policy.historical_sources[{idx}].source_id must be a non-empty string"
            )
        source_role = source.get("source_role")
        if source_role not in {"external_control_arm", "legacy_internal_control"}:
            raise ValueError(
                f"policy.historical_sources[{idx}].source_role must be "
                "external_control_arm or legacy_internal_control"
            )
        planned_size = source.get("planned_control_sample_size")
        if not isinstance(planned_size, (int, float)) or float(planned_size) <= 0.0:
            raise ValueError(
                f"policy.historical_sources[{idx}].planned_control_sample_size must be > 0"
            )
        exchangeability = source.get("exchangeability_assessment")
        if exchangeability not in _CONFLICT_SEVERITY_ORDER:
            raise ValueError(
                f"policy.historical_sources[{idx}].exchangeability_assessment must be low, moderate, or high"
            )
        data_cut_label = source.get("data_cut_label")
        if not isinstance(data_cut_label, str) or not data_cut_label:
            raise ValueError(
                f"policy.historical_sources[{idx}].data_cut_label must be a non-empty string"
            )

    eligibility = policy.get("eligibility")
    if not isinstance(eligibility, dict):
        raise ValueError("policy.eligibility must be a JSON object")
    minimum_current_control_sample_size = eligibility.get(
        "minimum_current_control_sample_size"
    )
    if (
        not isinstance(minimum_current_control_sample_size, (int, float))
        or float(minimum_current_control_sample_size) <= 0.0
    ):
        raise ValueError(
            "policy.eligibility.minimum_current_control_sample_size must be > 0"
        )
    minimum_control_information_fraction = eligibility.get(
        "minimum_control_information_fraction"
    )
    if not isinstance(minimum_control_information_fraction, (int, float)) or not (
        0.0 <= float(minimum_control_information_fraction) <= 1.0
    ):
        raise ValueError(
            "policy.eligibility.minimum_control_information_fraction must be in [0, 1]"
        )
    if not isinstance(eligibility.get("disallow_recommended_action_flip"), bool):
        raise ValueError(
            "policy.eligibility.disallow_recommended_action_flip must be boolean"
        )

    borrowing_strength = policy.get("borrowing_strength")
    if not isinstance(borrowing_strength, dict):
        raise ValueError("policy.borrowing_strength must be a JSON object")
    full_fraction = borrowing_strength.get("full_borrowing_fraction")
    tapered_fraction = borrowing_strength.get("tapered_borrowing_fraction")
    suspended_fraction = borrowing_strength.get("suspended_borrowing_fraction")
    for key, value in (
        ("full_borrowing_fraction", full_fraction),
        ("tapered_borrowing_fraction", tapered_fraction),
        ("suspended_borrowing_fraction", suspended_fraction),
    ):
        if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"policy.borrowing_strength.{key} must be in [0, 1]")
    if not (
        float(full_fraction) >= float(tapered_fraction) >= float(suspended_fraction)
    ):
        raise ValueError(
            "policy.borrowing_strength must satisfy full >= tapered >= suspended"
        )

    conflict_response = policy.get("conflict_response")
    if not isinstance(conflict_response, dict):
        raise ValueError("policy.conflict_response must be a JSON object")
    full_max_conflict = conflict_response.get("full_borrowing_max_conflict_severity")
    tapered_max_conflict = conflict_response.get(
        "tapered_borrowing_max_conflict_severity"
    )
    if full_max_conflict not in _CONFLICT_SEVERITY_ORDER:
        raise ValueError(
            "policy.conflict_response.full_borrowing_max_conflict_severity must be low, moderate, or high"
        )
    if tapered_max_conflict not in _CONFLICT_SEVERITY_ORDER:
        raise ValueError(
            "policy.conflict_response.tapered_borrowing_max_conflict_severity must be low, moderate, or high"
        )
    if (
        _CONFLICT_SEVERITY_ORDER[str(full_max_conflict)]
        > _CONFLICT_SEVERITY_ORDER[str(tapered_max_conflict)]
    ):
        raise ValueError(
            "policy.conflict_response must satisfy full_borrowing_max_conflict_severity <= tapered_borrowing_max_conflict_severity"
        )
    for key in (
        "max_eventual_success_probability_range_for_full_borrowing",
        "max_expected_total_sample_size_range_fraction_for_full_borrowing",
    ):
        value = conflict_response.get(key)
        if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"policy.conflict_response.{key} must be in [0, 1]")


def _build_historical_control_borrowing_review(
    report_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
    *,
    expected_family: str,
    expected_schema_version: str,
) -> dict[str, Any]:
    report_json = _coerce_json_payload(report_or_path)
    report = json.loads(report_json)
    if not isinstance(report, dict):
        raise ValueError("report must decode to a JSON object")
    _validate_design_report(
        report,
        expected_family=expected_family,
        expected_schema_version=expected_schema_version,
    )

    policy_json = _coerce_json_payload(policy_or_path)
    policy = json.loads(policy_json)
    if not isinstance(policy, dict):
        raise ValueError("policy must decode to a JSON object")
    _validate_historical_control_borrowing_policy(
        policy, expected_family=expected_family
    )

    prior_conflict = _build_prior_conflict_diagnostic(
        report,
        expected_family=expected_family,
        expected_schema_version=expected_schema_version,
    )
    conflict_metrics = prior_conflict["metrics"]

    current_control_sample_size = float(report["current_analysis"]["look"]["n_control"])
    planned_control_sample_size = float(report["design_spec"]["looks"][-1]["n_control"])
    current_control_information_fraction = (
        current_control_sample_size / planned_control_sample_size
        if planned_control_sample_size > 0.0
        else 0.0
    )
    historical_sources = [
        {
            "source_id": str(source["source_id"]),
            "source_role": str(source["source_role"]),
            "planned_control_sample_size": float(source["planned_control_sample_size"]),
            "exchangeability_assessment": str(source["exchangeability_assessment"]),
            "data_cut_label": str(source["data_cut_label"]),
        }
        for source in policy["historical_sources"]
    ]
    total_planned_historical_control_sample_size = sum(
        source["planned_control_sample_size"] for source in historical_sources
    )

    eligibility = policy["eligibility"]
    borrowing_strength = policy["borrowing_strength"]
    conflict_response = policy["conflict_response"]
    meets_minimum_current_control_sample_size = current_control_sample_size >= float(
        eligibility["minimum_current_control_sample_size"]
    )
    meets_minimum_control_information_fraction = (
        current_control_information_fraction
        >= float(eligibility["minimum_control_information_fraction"])
    )
    passes_action_flip_gate = not bool(
        eligibility["disallow_recommended_action_flip"]
    ) or not bool(prior_conflict["decision_instability"])

    conflict_severity = str(prior_conflict["conflict_severity"])
    conflict_rank = _CONFLICT_SEVERITY_ORDER[conflict_severity]
    full_rank = _CONFLICT_SEVERITY_ORDER[
        str(conflict_response["full_borrowing_max_conflict_severity"])
    ]
    tapered_rank = _CONFLICT_SEVERITY_ORDER[
        str(conflict_response["tapered_borrowing_max_conflict_severity"])
    ]
    within_full_eventual_success_range = float(
        conflict_metrics["eventual_success_probability_range"]
    ) <= float(
        conflict_response["max_eventual_success_probability_range_for_full_borrowing"]
    )
    within_full_expected_total_sample_size_range_fraction = float(
        conflict_metrics["expected_total_sample_size_range_fraction_of_plan"]
    ) <= float(
        conflict_response[
            "max_expected_total_sample_size_range_fraction_for_full_borrowing"
        ]
    )

    rationale: list[str] = []
    if not meets_minimum_current_control_sample_size:
        recommended_borrowing_state = "suspend"
        rationale.append(
            "Current concurrent control sample size is below the published borrowing-activation minimum."
        )
    elif not meets_minimum_control_information_fraction:
        recommended_borrowing_state = "suspend"
        rationale.append(
            "Current concurrent control information fraction is below the published borrowing-activation minimum."
        )
    elif not passes_action_flip_gate:
        recommended_borrowing_state = "suspend"
        rationale.append(
            "Published prior-variant action flips block historical borrowing under the current policy."
        )
    elif (
        conflict_rank <= full_rank
        and within_full_eventual_success_range
        and within_full_expected_total_sample_size_range_fraction
    ):
        recommended_borrowing_state = "retain"
        rationale.append(
            "Current prior-conflict diagnostics stay within the published full-borrowing thresholds."
        )
    elif conflict_rank <= tapered_rank:
        recommended_borrowing_state = "taper"
        rationale.append(
            "Historical borrowing remains conditionally acceptable, but the current conflict diagnostics exceed the full-borrowing limits."
        )
    else:
        recommended_borrowing_state = "suspend"
        rationale.append(
            "Current prior-conflict severity exceeds the published tapered-borrowing ceiling."
        )

    if recommended_borrowing_state == "retain":
        current_effective_borrowing_fraction = float(
            borrowing_strength["full_borrowing_fraction"]
        )
        borrowing_eligible = True
        rationale.append(
            "The review keeps the configured full borrowing fraction for the current frozen report."
        )
    elif recommended_borrowing_state == "taper":
        current_effective_borrowing_fraction = float(
            borrowing_strength["tapered_borrowing_fraction"]
        )
        borrowing_eligible = True
        rationale.append(
            "The review downgrades to the published tapered borrowing fraction while preserving deterministic traceability."
        )
    else:
        current_effective_borrowing_fraction = float(
            borrowing_strength["suspended_borrowing_fraction"]
        )
        borrowing_eligible = False
        rationale.append(
            "The review falls back to the published suspended borrowing fraction for the current frozen report."
        )

    current_effective_historical_control_sample_size = (
        current_effective_borrowing_fraction
        * total_planned_historical_control_sample_size
    )

    return {
        "schema_version": "nextstat_bayesian_historical_control_borrowing_review_v0",
        "stability": str(report.get("stability", "research-grade")),
        "review_id": (
            f"{report['design_spec']['design_id']}_{policy['policy_id']}"
            "_historical_control_borrowing_review_v0"
        ),
        "design_family": expected_family,
        "design_id": str(report["design_spec"]["design_id"]),
        "policy_id": str(policy["policy_id"]),
        "borrowing_model": str(policy["borrowing_model"]),
        "source_report_schema_version": str(report["schema_version"]),
        "source_policy_schema_version": str(policy["schema_version"]),
        "source_prior_conflict_schema_version": str(prior_conflict["schema_version"]),
        "generated_from_frozen_report": True,
        "recommended_borrowing_state": recommended_borrowing_state,
        "borrowing_eligible": borrowing_eligible,
        "current_effective_borrowing_fraction": current_effective_borrowing_fraction,
        "current_effective_historical_control_sample_size": (
            current_effective_historical_control_sample_size
        ),
        "borrowing_strength": {
            "full_borrowing_fraction": float(
                borrowing_strength["full_borrowing_fraction"]
            ),
            "tapered_borrowing_fraction": float(
                borrowing_strength["tapered_borrowing_fraction"]
            ),
            "suspended_borrowing_fraction": float(
                borrowing_strength["suspended_borrowing_fraction"]
            ),
        },
        "gating": {
            "meets_minimum_current_control_sample_size": (
                meets_minimum_current_control_sample_size
            ),
            "meets_minimum_control_information_fraction": (
                meets_minimum_control_information_fraction
            ),
            "passes_action_flip_gate": passes_action_flip_gate,
            "within_full_eventual_success_probability_range": (
                within_full_eventual_success_range
            ),
            "within_full_expected_total_sample_size_range_fraction": (
                within_full_expected_total_sample_size_range_fraction
            ),
        },
        "diagnostics": {
            "prior_conflict_severity": conflict_severity,
            "decision_instability": bool(prior_conflict["decision_instability"]),
            "recommended_action_flip_count": int(
                conflict_metrics["recommended_action_flip_count"]
            ),
            "current_control_sample_size": current_control_sample_size,
            "planned_control_sample_size": planned_control_sample_size,
            "current_control_information_fraction": (
                current_control_information_fraction
            ),
            "historical_source_count": len(historical_sources),
            "total_planned_historical_control_sample_size": (
                total_planned_historical_control_sample_size
            ),
            "eventual_success_probability_range": float(
                conflict_metrics["eventual_success_probability_range"]
            ),
            "expected_total_sample_size_range_fraction_of_plan": float(
                conflict_metrics["expected_total_sample_size_range_fraction_of_plan"]
            ),
        },
        "historical_sources": historical_sources,
        "rationale": rationale,
    }


def _target_prior_sample_sizes(
    report: dict[str, Any],
    *,
    prior_target: str,
) -> tuple[float, float, float]:
    current_look = report["current_analysis"]["look"]
    planned_look = report["design_spec"]["looks"][-1]
    if prior_target == "control_prior":
        current_sample_size = float(current_look["n_control"])
        planned_sample_size = float(planned_look["n_control"])
    else:
        current_sample_size = float(current_look["n_treatment"])
        planned_sample_size = float(planned_look["n_treatment"])
    information_fraction = (
        current_sample_size / planned_sample_size if planned_sample_size > 0.0 else 0.0
    )
    return current_sample_size, planned_sample_size, information_fraction


def _validate_robust_mixture_prior_policy(
    policy: dict[str, Any],
    *,
    expected_family: str,
) -> None:
    if (
        policy.get("schema_version")
        != "nextstat_bayesian_robust_mixture_prior_policy_v0"
    ):
        raise ValueError(
            "policy.schema_version must be "
            "nextstat_bayesian_robust_mixture_prior_policy_v0"
        )
    if policy.get("design_family") != expected_family:
        raise ValueError(f"policy.design_family must be {expected_family}")
    policy_id = policy.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("policy.policy_id must be a non-empty string")

    expected_model = (
        "robust_mixture_beta"
        if expected_family == "beta_binomial"
        else "robust_mixture_normal"
    )
    if policy.get("mixture_model") != expected_model:
        raise ValueError(
            f"policy.mixture_model must be {expected_model} for {expected_family}"
        )

    prior_target = policy.get("prior_target")
    if prior_target not in {"control_prior", "treatment_prior"}:
        raise ValueError("policy.prior_target must be control_prior or treatment_prior")

    mixture_components = policy.get("mixture_components")
    if not isinstance(mixture_components, list) or len(mixture_components) != 2:
        raise ValueError(
            "policy.mixture_components must contain exactly two components"
        )

    seen_roles: set[str] = set()
    total_base_weight = 0.0
    informative_base_weight = None
    for idx, component in enumerate(mixture_components):
        if not isinstance(component, dict):
            raise ValueError(f"policy.mixture_components[{idx}] must be a JSON object")
        component_id = component.get("component_id")
        if not isinstance(component_id, str) or not component_id:
            raise ValueError(
                f"policy.mixture_components[{idx}].component_id must be a non-empty string"
            )
        component_role = component.get("component_role")
        if component_role not in {"informative", "weak_reference"}:
            raise ValueError(
                f"policy.mixture_components[{idx}].component_role must be informative or weak_reference"
            )
        if component_role in seen_roles:
            raise ValueError(
                "policy.mixture_components must include each component_role exactly once"
            )
        seen_roles.add(str(component_role))

        base_weight = component.get("base_weight")
        if not isinstance(base_weight, (int, float)) or not (
            0.0 <= float(base_weight) <= 1.0
        ):
            raise ValueError(
                f"policy.mixture_components[{idx}].base_weight must be in [0, 1]"
            )
        total_base_weight += float(base_weight)
        if component_role == "informative":
            informative_base_weight = float(base_weight)

        if expected_family == "beta_binomial":
            for key in ("alpha", "beta"):
                value = component.get(key)
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                ):
                    raise ValueError(
                        f"policy.mixture_components[{idx}].{key} must be finite and > 0"
                    )
        else:
            mean = component.get("mean")
            if not isinstance(mean, (int, float)) or not math.isfinite(float(mean)):
                raise ValueError(
                    f"policy.mixture_components[{idx}].mean must be finite for {expected_family}"
                )
            sd = component.get("sd")
            if (
                not isinstance(sd, (int, float))
                or not math.isfinite(float(sd))
                or float(sd) <= 0.0
            ):
                raise ValueError(
                    f"policy.mixture_components[{idx}].sd must be finite and > 0"
                )

    if seen_roles != {"informative", "weak_reference"}:
        raise ValueError(
            "policy.mixture_components must include exactly one informative and one weak_reference component"
        )
    if not math.isclose(total_base_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("policy.mixture_components base weights must sum to 1")

    eligibility = policy.get("eligibility")
    if not isinstance(eligibility, dict):
        raise ValueError("policy.eligibility must be a JSON object")
    minimum_information_fraction = eligibility.get("minimum_information_fraction")
    if not isinstance(minimum_information_fraction, (int, float)) or not (
        0.0 <= float(minimum_information_fraction) <= 1.0
    ):
        raise ValueError(
            "policy.eligibility.minimum_information_fraction must be in [0, 1]"
        )
    if not isinstance(eligibility.get("disallow_recommended_action_flip"), bool):
        raise ValueError(
            "policy.eligibility.disallow_recommended_action_flip must be boolean"
        )

    weight_schedule = policy.get("weight_schedule")
    if not isinstance(weight_schedule, dict):
        raise ValueError("policy.weight_schedule must be a JSON object")
    retain_informative_weight = weight_schedule.get("retain_informative_weight")
    tapered_informative_weight = weight_schedule.get("tapered_informative_weight")
    fallback_informative_weight = weight_schedule.get("fallback_informative_weight")
    for key, value in (
        ("retain_informative_weight", retain_informative_weight),
        ("tapered_informative_weight", tapered_informative_weight),
        ("fallback_informative_weight", fallback_informative_weight),
    ):
        if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"policy.weight_schedule.{key} must be in [0, 1]")
    if not (
        float(retain_informative_weight)
        >= float(tapered_informative_weight)
        >= float(fallback_informative_weight)
    ):
        raise ValueError(
            "policy.weight_schedule must satisfy retain >= tapered >= fallback"
        )
    if not math.isclose(
        float(fallback_informative_weight), 0.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            "policy.weight_schedule.fallback_informative_weight must be 0 for weak-only fallback"
        )
    if informative_base_weight is None or not math.isclose(
        float(retain_informative_weight),
        informative_base_weight,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "policy.weight_schedule.retain_informative_weight must match the informative component base_weight"
        )

    conflict_response = policy.get("conflict_response")
    if not isinstance(conflict_response, dict):
        raise ValueError("policy.conflict_response must be a JSON object")
    retain_max_conflict = conflict_response.get("retain_max_conflict_severity")
    tapered_max_conflict = conflict_response.get("tapered_max_conflict_severity")
    if retain_max_conflict not in _CONFLICT_SEVERITY_ORDER:
        raise ValueError(
            "policy.conflict_response.retain_max_conflict_severity must be low, moderate, or high"
        )
    if tapered_max_conflict not in _CONFLICT_SEVERITY_ORDER:
        raise ValueError(
            "policy.conflict_response.tapered_max_conflict_severity must be low, moderate, or high"
        )
    if (
        _CONFLICT_SEVERITY_ORDER[str(retain_max_conflict)]
        > _CONFLICT_SEVERITY_ORDER[str(tapered_max_conflict)]
    ):
        raise ValueError(
            "policy.conflict_response must satisfy retain_max_conflict_severity <= tapered_max_conflict_severity"
        )
    for key in (
        "max_eventual_success_probability_range_for_retain",
        "max_expected_total_sample_size_range_fraction_for_retain",
    ):
        value = conflict_response.get(key)
        if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"policy.conflict_response.{key} must be in [0, 1]")


def _robust_mixture_component_prior(
    component: dict[str, Any],
    *,
    expected_family: str,
) -> dict[str, float]:
    if expected_family == "beta_binomial":
        return {
            "alpha": float(component["alpha"]),
            "beta": float(component["beta"]),
        }
    return {
        "mean": float(component["mean"]),
        "sd": float(component["sd"]),
    }


def _build_robust_mixture_prior_review(
    report_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
    *,
    expected_family: str,
    expected_schema_version: str,
) -> dict[str, Any]:
    report_json = _coerce_json_payload(report_or_path)
    report = json.loads(report_json)
    if not isinstance(report, dict):
        raise ValueError("report must decode to a JSON object")
    _validate_design_report(
        report,
        expected_family=expected_family,
        expected_schema_version=expected_schema_version,
    )

    policy_json = _coerce_json_payload(policy_or_path)
    policy = json.loads(policy_json)
    if not isinstance(policy, dict):
        raise ValueError("policy must decode to a JSON object")
    _validate_robust_mixture_prior_policy(policy, expected_family=expected_family)

    prior_conflict = _build_prior_conflict_diagnostic(
        report,
        expected_family=expected_family,
        expected_schema_version=expected_schema_version,
    )
    conflict_metrics = prior_conflict["metrics"]
    prior_target = str(policy["prior_target"])
    current_sample_size, planned_sample_size, current_information_fraction = (
        _target_prior_sample_sizes(report, prior_target=prior_target)
    )
    components_by_role = {
        str(component["component_role"]): component
        for component in policy["mixture_components"]
    }
    informative_component = components_by_role["informative"]
    weak_component = components_by_role["weak_reference"]

    eligibility = policy["eligibility"]
    weight_schedule = policy["weight_schedule"]
    conflict_response = policy["conflict_response"]
    meets_minimum_information_fraction = current_information_fraction >= float(
        eligibility["minimum_information_fraction"]
    )
    passes_action_flip_gate = not bool(
        eligibility["disallow_recommended_action_flip"]
    ) or not bool(prior_conflict["decision_instability"])

    conflict_severity = str(prior_conflict["conflict_severity"])
    conflict_rank = _CONFLICT_SEVERITY_ORDER[conflict_severity]
    retain_rank = _CONFLICT_SEVERITY_ORDER[
        str(conflict_response["retain_max_conflict_severity"])
    ]
    tapered_rank = _CONFLICT_SEVERITY_ORDER[
        str(conflict_response["tapered_max_conflict_severity"])
    ]
    within_retain_eventual_success_range = float(
        conflict_metrics["eventual_success_probability_range"]
    ) <= float(conflict_response["max_eventual_success_probability_range_for_retain"])
    within_retain_expected_total_sample_size_range_fraction = float(
        conflict_metrics["expected_total_sample_size_range_fraction_of_plan"]
    ) <= float(
        conflict_response["max_expected_total_sample_size_range_fraction_for_retain"]
    )

    rationale: list[str] = []
    if not meets_minimum_information_fraction:
        recommended_mixture_state = "fallback_to_weak"
        rationale.append(
            "Current concurrent information fraction is below the published robust-mixture activation minimum."
        )
    elif not passes_action_flip_gate:
        recommended_mixture_state = "fallback_to_weak"
        rationale.append(
            "Published prior-variant action flips block the informative mixture component under the current policy."
        )
    elif (
        conflict_rank <= retain_rank
        and within_retain_eventual_success_range
        and within_retain_expected_total_sample_size_range_fraction
    ):
        recommended_mixture_state = "retain"
        rationale.append(
            "Current prior-conflict diagnostics stay within the published retain thresholds for the informative mixture component."
        )
    elif conflict_rank <= tapered_rank:
        recommended_mixture_state = "taper"
        rationale.append(
            "The informative mixture component remains conditionally acceptable, but the current conflict diagnostics exceed the retain limits."
        )
    else:
        recommended_mixture_state = "fallback_to_weak"
        rationale.append(
            "Current prior-conflict severity exceeds the published tapered-mixture ceiling."
        )

    if recommended_mixture_state == "retain":
        current_informative_weight = float(weight_schedule["retain_informative_weight"])
        mixture_eligible = True
        rationale.append(
            "The review keeps the published informative-component base weight for the frozen report."
        )
    elif recommended_mixture_state == "taper":
        current_informative_weight = float(
            weight_schedule["tapered_informative_weight"]
        )
        mixture_eligible = True
        rationale.append(
            "The review deterministically tapers the informative-component weight while preserving the weak-reference guardrail."
        )
    else:
        current_informative_weight = float(
            weight_schedule["fallback_informative_weight"]
        )
        mixture_eligible = False
        rationale.append(
            "The review falls back to the published weak-reference-only state for the frozen report."
        )

    effective_component_weights = []
    for component in policy["mixture_components"]:
        component_role = str(component["component_role"])
        effective_weight = (
            current_informative_weight
            if component_role == "informative"
            else 1.0 - current_informative_weight
        )
        effective_component_weights.append(
            {
                "component_id": str(component["component_id"]),
                "component_role": component_role,
                "base_weight": float(component["base_weight"]),
                "effective_weight": effective_weight,
                "prior": _robust_mixture_component_prior(
                    component,
                    expected_family=expected_family,
                ),
            }
        )

    return {
        "schema_version": "nextstat_bayesian_robust_mixture_prior_review_v0",
        "stability": str(report.get("stability", "research-grade")),
        "review_id": (
            f"{report['design_spec']['design_id']}_{policy['policy_id']}"
            "_robust_mixture_prior_review_v0"
        ),
        "design_family": expected_family,
        "design_id": str(report["design_spec"]["design_id"]),
        "policy_id": str(policy["policy_id"]),
        "mixture_model": str(policy["mixture_model"]),
        "prior_target": prior_target,
        "source_report_schema_version": str(report["schema_version"]),
        "source_policy_schema_version": str(policy["schema_version"]),
        "source_prior_conflict_schema_version": str(prior_conflict["schema_version"]),
        "generated_from_frozen_report": True,
        "recommended_mixture_state": recommended_mixture_state,
        "mixture_eligible": mixture_eligible,
        "current_informative_weight": current_informative_weight,
        "effective_component_weights": effective_component_weights,
        "gating": {
            "meets_minimum_information_fraction": meets_minimum_information_fraction,
            "passes_action_flip_gate": passes_action_flip_gate,
            "within_retain_eventual_success_probability_range": (
                within_retain_eventual_success_range
            ),
            "within_retain_expected_total_sample_size_range_fraction": (
                within_retain_expected_total_sample_size_range_fraction
            ),
        },
        "diagnostics": {
            "prior_conflict_severity": conflict_severity,
            "decision_instability": bool(prior_conflict["decision_instability"]),
            "recommended_action_flip_count": int(
                conflict_metrics["recommended_action_flip_count"]
            ),
            "current_target_sample_size": current_sample_size,
            "planned_target_sample_size": planned_sample_size,
            "current_information_fraction": current_information_fraction,
            "posterior_probability_range": float(
                conflict_metrics["posterior_probability_range"]
            ),
            "eventual_success_probability_range": float(
                conflict_metrics["eventual_success_probability_range"]
            ),
            "expected_total_sample_size_range_fraction_of_plan": float(
                conflict_metrics["expected_total_sample_size_range_fraction_of_plan"]
            ),
        },
        "rationale": rationale,
    }


def _expected_design_schema_version(expected_family: str) -> str:
    if expected_family == "beta_binomial":
        return "nextstat_beta_binomial_design_v0"
    return "nextstat_normal_normal_design_v0"


def _expected_report_schema_version(expected_family: str) -> str:
    if expected_family == "beta_binomial":
        return "nextstat_beta_binomial_design_report_v0"
    return "nextstat_normal_normal_design_report_v0"


def _expected_campaign_schema_version(expected_family: str) -> str:
    if expected_family == "beta_binomial":
        return "nextstat_beta_binomial_prior_sensitivity_campaign_v0"
    return "nextstat_normal_normal_prior_sensitivity_campaign_v0"


def _analysis_stops_trial(analysis: dict[str, Any]) -> bool:
    return str(analysis["decision"]["recommended_action"]) != "continue"


def _open_unit_interval(rng: random.Random) -> float:
    return min(1.0 - 1e-12, max(1e-12, rng.random()))


def _sample_normal(rng: random.Random, *, mean: float, sd: float) -> float:
    if sd <= 0.0:
        return mean
    return mean + sd * _STANDARD_NORMAL.inv_cdf(_open_unit_interval(rng))


def _simulate_beta_binomial_observed_sequence(
    spec: dict[str, Any],
    scenario: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    observed_sequence: list[dict[str, Any]] = []
    control_successes = 0
    treatment_successes = 0
    previous_n_control = 0
    previous_n_treatment = 0
    p_control = float(scenario["p_control"])
    p_treatment = float(scenario["p_treatment"])

    for look in spec["looks"]:
        n_control = int(look["n_control"])
        n_treatment = int(look["n_treatment"])
        additional_n_control = n_control - previous_n_control
        additional_n_treatment = n_treatment - previous_n_treatment
        if additional_n_control < 0 or additional_n_treatment < 0:
            raise ValueError("spec.looks must have non-decreasing cumulative sample sizes")
        for _ in range(additional_n_control):
            control_successes += int(rng.random() < p_control)
        for _ in range(additional_n_treatment):
            treatment_successes += int(rng.random() < p_treatment)
        observed_sequence.append(
            {
                "look_id": str(look["id"]),
                "control_successes": control_successes,
                "treatment_successes": treatment_successes,
            }
        )
        previous_n_control = n_control
        previous_n_treatment = n_treatment

    return observed_sequence


def _simulate_normal_normal_observed_sequence(
    spec: dict[str, Any],
    scenario: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    observed_sequence: list[dict[str, Any]] = []
    control_sum = 0.0
    treatment_sum = 0.0
    previous_n_control = 0
    previous_n_treatment = 0
    mean_control = float(scenario["mean_control"])
    mean_treatment = float(scenario["mean_treatment"])
    likelihood = spec["likelihood"]
    known_sd_control = float(likelihood["known_sd_control"])
    known_sd_treatment = float(likelihood["known_sd_treatment"])

    for look in spec["looks"]:
        n_control = int(look["n_control"])
        n_treatment = int(look["n_treatment"])
        additional_n_control = n_control - previous_n_control
        additional_n_treatment = n_treatment - previous_n_treatment
        if additional_n_control < 0 or additional_n_treatment < 0:
            raise ValueError("spec.looks must have non-decreasing cumulative sample sizes")
        for _ in range(additional_n_control):
            control_sum += _sample_normal(
                rng,
                mean=mean_control,
                sd=known_sd_control,
            )
        for _ in range(additional_n_treatment):
            treatment_sum += _sample_normal(
                rng,
                mean=mean_treatment,
                sd=known_sd_treatment,
            )
        observed_sequence.append(
            {
                "look_id": str(look["id"]),
                "control_sample_mean": control_sum / float(n_control),
                "treatment_sample_mean": treatment_sum / float(n_treatment),
            }
        )
        previous_n_control = n_control
        previous_n_treatment = n_treatment

    return observed_sequence


def _expected_prior_sensitivity_report_schema_version(expected_family: str) -> str:
    if expected_family == "beta_binomial":
        return "nextstat_beta_binomial_prior_sensitivity_report_v0"
    return "nextstat_normal_normal_prior_sensitivity_report_v0"


def _validate_extension_prior(
    prior: Any,
    *,
    expected_family: str,
    field_name: str,
) -> None:
    if not isinstance(prior, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    if expected_family == "beta_binomial":
        for key in ("alpha", "beta"):
            value = prior.get(key)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{field_name}.{key} must be finite and > 0")
        return

    mean = prior.get("mean")
    if not isinstance(mean, (int, float)) or not math.isfinite(float(mean)):
        raise ValueError(f"{field_name}.mean must be finite")
    sd = prior.get("sd")
    if (
        not isinstance(sd, (int, float))
        or not math.isfinite(float(sd))
        or float(sd) <= 0.0
    ):
        raise ValueError(f"{field_name}.sd must be finite and > 0")


def _validate_extension_spec(
    spec: dict[str, Any],
    *,
    expected_family: str,
) -> None:
    if spec.get("schema_version") != _expected_design_schema_version(expected_family):
        raise ValueError(
            "spec.schema_version must be "
            f"{_expected_design_schema_version(expected_family)}"
        )
    design_id = spec.get("design_id")
    if not isinstance(design_id, str) or not design_id:
        raise ValueError("spec.design_id must be a non-empty string")

    _validate_extension_prior(
        spec.get("control_prior"),
        expected_family=expected_family,
        field_name="spec.control_prior",
    )
    _validate_extension_prior(
        spec.get("treatment_prior"),
        expected_family=expected_family,
        field_name="spec.treatment_prior",
    )

    looks = spec.get("looks")
    if not isinstance(looks, list) or not looks:
        raise ValueError("spec.looks must contain at least one look")
    previous_n_control = 0
    previous_n_treatment = 0
    for idx, look in enumerate(looks):
        if not isinstance(look, dict):
            raise ValueError(f"spec.looks[{idx}] must be a JSON object")
        look_id = look.get("id")
        if not isinstance(look_id, str) or not look_id:
            raise ValueError(f"spec.looks[{idx}].id must be a non-empty string")
        n_control = look.get("n_control")
        n_treatment = look.get("n_treatment")
        if not isinstance(n_control, int) or n_control <= 0:
            raise ValueError(f"spec.looks[{idx}].n_control must be a positive integer")
        if not isinstance(n_treatment, int) or n_treatment <= 0:
            raise ValueError(
                f"spec.looks[{idx}].n_treatment must be a positive integer"
            )
        if n_control < previous_n_control or n_treatment < previous_n_treatment:
            raise ValueError("spec.looks must have non-decreasing cumulative sample sizes")
        previous_n_control = n_control
        previous_n_treatment = n_treatment

    if expected_family == "normal_normal":
        likelihood = spec.get("likelihood")
        if not isinstance(likelihood, dict):
            raise ValueError("spec.likelihood must be a JSON object")
        for key in ("known_sd_control", "known_sd_treatment"):
            value = likelihood.get(key)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"spec.likelihood.{key} must be finite and > 0")

    simulation = spec.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError("spec.simulation must be a JSON object")
    n_replicates = simulation.get("n_replicates")
    seed = simulation.get("seed")
    if not isinstance(n_replicates, int) or n_replicates <= 0:
        raise ValueError("spec.simulation.n_replicates must be a positive integer")
    if not isinstance(seed, int):
        raise ValueError("spec.simulation.seed must be an integer")
    scenarios = simulation.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("spec.simulation.scenarios must be a non-empty array")
    for idx, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(
                f"spec.simulation.scenarios[{idx}] must be a JSON object"
            )
        scenario_id = scenario.get("id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(
                f"spec.simulation.scenarios[{idx}].id must be a non-empty string"
            )
        if expected_family == "beta_binomial":
            for key in ("p_control", "p_treatment"):
                value = scenario.get(key)
                if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
                    raise ValueError(
                        f"spec.simulation.scenarios[{idx}].{key} must be in [0, 1]"
                    )
        else:
            for key in ("mean_control", "mean_treatment"):
                value = scenario.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(
                        f"spec.simulation.scenarios[{idx}].{key} must be finite"
                    )


def _validate_extension_campaign(
    campaign: dict[str, Any],
    *,
    expected_family: str,
) -> None:
    if campaign.get("schema_version") != _expected_campaign_schema_version(
        expected_family
    ):
        raise ValueError(
            "campaign.schema_version must be "
            f"{_expected_campaign_schema_version(expected_family)}"
        )
    variants = campaign.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("campaign.variants must be a non-empty array")
    seen_variant_ids = {"baseline"}
    for idx, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"campaign.variants[{idx}] must be a JSON object")
        variant_id = variant.get("id")
        if not isinstance(variant_id, str) or not variant_id:
            raise ValueError(
                f"campaign.variants[{idx}].id must be a non-empty string"
            )
        if variant_id in seen_variant_ids:
            raise ValueError("campaign.variants ids must be unique and must not reuse baseline")
        seen_variant_ids.add(variant_id)
        _validate_extension_prior(
            variant.get("control_prior"),
            expected_family=expected_family,
            field_name=f"campaign.variants[{idx}].control_prior",
        )
        _validate_extension_prior(
            variant.get("treatment_prior"),
            expected_family=expected_family,
            field_name=f"campaign.variants[{idx}].treatment_prior",
        )


def _collect_extension_variant_specs(
    spec: dict[str, Any],
    campaign: dict[str, Any],
) -> list[dict[str, Any]]:
    variant_specs = [
        {
            "variant_id": "baseline",
            "is_baseline": True,
            "spec": _json_clone(spec),
        }
    ]
    for variant in campaign["variants"]:
        variant_spec = _json_clone(spec)
        variant_spec["control_prior"] = _json_clone(variant["control_prior"])
        variant_spec["treatment_prior"] = _json_clone(variant["treatment_prior"])
        variant_specs.append(
            {
                "variant_id": str(variant["id"]),
                "is_baseline": False,
                "spec": variant_spec,
            }
        )
    return variant_specs


def _terminal_path_outcome(
    analyses: list[dict[str, Any]],
    *,
    start_index: int,
) -> tuple[str, float]:
    for analysis in analyses[start_index:]:
        action = str(analysis["decision"]["recommended_action"])
        if action != "continue":
            look = analysis["look"]
            return action, float(look["n_control"]) + float(look["n_treatment"])
    final_look = analyses[-1]["look"]
    return (
        "no_decision",
        float(final_look["n_control"]) + float(final_look["n_treatment"]),
    )


def _compose_review_snapshot_report(
    *,
    spec: dict[str, Any],
    variant_specs: list[dict[str, Any]],
    analyses_by_variant: dict[str, list[dict[str, Any]]],
    current_look_index: int,
    expected_family: str,
    stability: str,
) -> dict[str, Any]:
    current_analysis = _json_clone(analyses_by_variant["baseline"][current_look_index])
    baseline_summary: dict[str, float] | None = None
    prior_sensitivity_variants = []
    for variant in variant_specs:
        variant_id = str(variant["variant_id"])
        analysis = analyses_by_variant[variant_id][current_look_index]
        terminal_action, terminal_total_sample_size = _terminal_path_outcome(
            analyses_by_variant[variant_id],
            start_index=current_look_index,
        )
        summary = {
            "variant_id": variant_id,
            "is_baseline": bool(variant["is_baseline"]),
            "recommended_action": str(analysis["decision"]["recommended_action"]),
            "posterior_probability_gt_margin": float(
                analysis["decision"]["posterior_probability_gt_margin"]
            ),
            "eventual_success_probability": (
                1.0 if terminal_action == "stop_for_success" else 0.0
            ),
            "expected_total_sample_size": terminal_total_sample_size,
        }
        if bool(variant["is_baseline"]):
            baseline_summary = {
                "posterior_probability_gt_margin": summary[
                    "posterior_probability_gt_margin"
                ],
                "eventual_success_probability": summary[
                    "eventual_success_probability"
                ],
                "expected_total_sample_size": summary["expected_total_sample_size"],
            }
        prior_sensitivity_variants.append(summary)

    if baseline_summary is None:
        raise ValueError("baseline variant is required for pathwise prior sensitivity")

    for summary in prior_sensitivity_variants:
        summary["posterior_probability_delta_vs_baseline"] = (
            float(summary["posterior_probability_gt_margin"])
            - baseline_summary["posterior_probability_gt_margin"]
        )
        summary["eventual_success_probability_delta_vs_baseline"] = (
            float(summary["eventual_success_probability"])
            - baseline_summary["eventual_success_probability"]
        )
        summary["expected_total_sample_size_delta_vs_baseline"] = (
            float(summary["expected_total_sample_size"])
            - baseline_summary["expected_total_sample_size"]
        )

    prior_sensitivity = {
        "schema_version": _expected_prior_sensitivity_report_schema_version(
            expected_family
        ),
        "stability": stability,
        "design_id": str(spec["design_id"]),
        "look": _json_clone(current_analysis["look"]),
        "observed": _json_clone(current_analysis["observed"]),
        "n_replicates": 1,
        "seed": 0,
        "variants": prior_sensitivity_variants,
    }
    return {
        "schema_version": _expected_report_schema_version(expected_family),
        "stability": stability,
        "design_family": expected_family,
        "design_spec": _json_clone(spec),
        "current_analysis": current_analysis,
        "operating_characteristics": {},
        "posterior_predictive": {},
        "prior_sensitivity": prior_sensitivity,
        "provenance": {
            "analysis_schema_version": str(current_analysis["schema_version"]),
            "prior_sensitivity_report_schema_version": str(
                prior_sensitivity["schema_version"]
            ),
        },
    }


def _simulate_review_operating_characteristics(
    spec_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
    *,
    expected_family: str,
    output_schema_version: str,
    derived_review_schema_version: str,
    state_key: str,
    states: tuple[str, ...],
    eligible_key: str,
    mean_fields: tuple[tuple[str, str, str], ...],
    scenario_param_keys: tuple[str, ...],
    analyze_design: Any,
    build_review: Any,
    validate_policy: Any,
    simulate_observed_sequence: Any,
) -> dict[str, Any]:
    spec = _coerce_json_object(spec_or_path, name="spec")
    campaign = _coerce_json_object(campaign_or_path, name="campaign")
    policy = _coerce_json_object(policy_or_path, name="policy")

    _validate_extension_spec(spec, expected_family=expected_family)
    _validate_extension_campaign(campaign, expected_family=expected_family)
    validate_policy(policy, expected_family=expected_family)

    n_replicates = int(spec["simulation"]["n_replicates"])
    seed = int(spec["simulation"]["seed"])
    looks = spec["looks"]
    stability = str(policy.get("stability", "research-grade"))
    variant_specs = _collect_extension_variant_specs(spec, campaign)
    scenarios_out: list[dict[str, Any]] = []

    for scenario_index, raw_scenario in enumerate(spec["simulation"]["scenarios"]):
        scenario = _json_clone(raw_scenario)
        if not isinstance(scenario, dict):
            raise ValueError("spec.simulation.scenarios entries must be JSON objects")

        terminal_state_counts = {state: 0 for state in states}
        terminal_eligible_count = 0
        terminal_decision_instability_count = 0
        terminal_high_conflict_count = 0
        terminal_mean_sums = {
            terminal_output_name: 0.0
            for _, terminal_output_name, _ in mean_fields
        }
        look_totals: dict[str, dict[str, float]] = {}
        for look in looks:
            look_id = str(look["id"])
            look_totals[look_id] = {
                "review_count": 0.0,
                "eligible_count": 0.0,
                "decision_instability_count": 0.0,
                "high_conflict_count": 0.0,
            }
            for state in states:
                look_totals[look_id][f"{state}_count"] = 0.0
            for _, _, look_output_name in mean_fields:
                look_totals[look_id][look_output_name] = 0.0

        scenario_id = str(scenario["id"])
        rng = random.Random(f"{seed}:{scenario_id}")
        for _ in range(n_replicates):
            observed_sequence = simulate_observed_sequence(spec, scenario, rng)
            analyses_by_variant = {
                str(variant["variant_id"]): [
                    analyze_design(variant["spec"], observed)
                    for observed in observed_sequence
                ]
                for variant in variant_specs
            }
            terminal_review: dict[str, Any] | None = None

            for look_index, observed in enumerate(observed_sequence):
                report = _compose_review_snapshot_report(
                    spec=spec,
                    variant_specs=variant_specs,
                    analyses_by_variant=analyses_by_variant,
                    current_look_index=look_index,
                    expected_family=expected_family,
                    stability=stability,
                )
                review = build_review(report, policy)
                look_totals[str(observed["look_id"])]["review_count"] += 1.0
                look_totals[str(observed["look_id"])][
                    f"{review[state_key]}_count"
                ] += 1.0
                if bool(review[eligible_key]):
                    look_totals[str(observed["look_id"])]["eligible_count"] += 1.0
                if bool(review["diagnostics"]["decision_instability"]):
                    look_totals[str(observed["look_id"])][
                        "decision_instability_count"
                    ] += 1.0
                if (
                    str(review["diagnostics"]["prior_conflict_severity"]) == "high"
                ):
                    look_totals[str(observed["look_id"])]["high_conflict_count"] += 1.0
                for input_field, _, look_output_name in mean_fields:
                    look_totals[str(observed["look_id"])][look_output_name] += float(
                        review[input_field]
                    )
                terminal_review = review
                if _analysis_stops_trial(analyses_by_variant["baseline"][look_index]):
                    break

            if terminal_review is None:
                raise ValueError("simulation produced no evaluable review states")

            terminal_state_counts[str(terminal_review[state_key])] += 1
            if bool(terminal_review[eligible_key]):
                terminal_eligible_count += 1
            if bool(terminal_review["diagnostics"]["decision_instability"]):
                terminal_decision_instability_count += 1
            if str(terminal_review["diagnostics"]["prior_conflict_severity"]) == "high":
                terminal_high_conflict_count += 1
            for input_field, terminal_output_name, _ in mean_fields:
                terminal_mean_sums[terminal_output_name] += float(
                    terminal_review[input_field]
                )

        scenario_out: dict[str, Any] = {
            "scenario_id": scenario_id,
            f"{states[0]}_rate": terminal_state_counts[states[0]] / float(n_replicates),
            f"{states[1]}_rate": terminal_state_counts[states[1]] / float(n_replicates),
            f"{states[2]}_rate": terminal_state_counts[states[2]] / float(n_replicates),
            f"{eligible_key}_rate": terminal_eligible_count / float(n_replicates),
            "decision_instability_rate": (
                terminal_decision_instability_count / float(n_replicates)
            ),
            "high_conflict_rate": terminal_high_conflict_count / float(n_replicates),
            "look_summaries": [],
        }
        for param_key in scenario_param_keys:
            scenario_out[param_key] = float(scenario[param_key])
        for terminal_output_name, total in terminal_mean_sums.items():
            scenario_out[terminal_output_name] = total / float(n_replicates)

        for look in looks:
            look_id = str(look["id"])
            totals = look_totals[look_id]
            review_count = float(totals["review_count"])
            look_summary: dict[str, Any] = {
                "look_id": look_id,
                "review_probability": review_count / float(n_replicates),
                f"{states[0]}_probability": (
                    float(totals[f"{states[0]}_count"]) / float(n_replicates)
                ),
                f"{states[1]}_probability": (
                    float(totals[f"{states[1]}_count"]) / float(n_replicates)
                ),
                f"{states[2]}_probability": (
                    float(totals[f"{states[2]}_count"]) / float(n_replicates)
                ),
                f"{eligible_key}_probability": (
                    float(totals["eligible_count"]) / float(n_replicates)
                ),
                "decision_instability_probability": (
                    float(totals["decision_instability_count"])
                    / float(n_replicates)
                ),
                "high_conflict_probability": (
                    float(totals["high_conflict_count"]) / float(n_replicates)
                ),
            }
            for _, _, look_output_name in mean_fields:
                look_summary[look_output_name] = (
                    float(totals[look_output_name]) / review_count
                    if review_count > 0.0
                    else 0.0
                )
            scenario_out["look_summaries"].append(look_summary)
        scenarios_out.append(scenario_out)

    return {
        "schema_version": output_schema_version,
        "stability": stability,
        "design_family": expected_family,
        "design_id": str(spec["design_id"]),
        "policy_id": str(policy["policy_id"]),
        "source_design_schema_version": _expected_design_schema_version(
            expected_family
        ),
        "source_campaign_schema_version": str(campaign["schema_version"]),
        "source_policy_schema_version": str(policy["schema_version"]),
        "derived_review_schema_version": derived_review_schema_version,
        "generated_from_seeded_simulation": True,
        "n_replicates": n_replicates,
        "seed": seed,
        "scenarios": scenarios_out,
    }


def _simulate_historical_control_borrowing_operating_characteristics(
    spec_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
    *,
    expected_family: str,
    analyze_design: Any,
    simulate_observed_sequence: Any,
) -> dict[str, Any]:
    return _simulate_review_operating_characteristics(
        spec_or_path,
        campaign_or_path,
        policy_or_path,
        expected_family=expected_family,
        output_schema_version=(
            "nextstat_bayesian_historical_control_borrowing_operating_characteristics_v0"
        ),
        derived_review_schema_version=(
            "nextstat_bayesian_historical_control_borrowing_review_v0"
        ),
        state_key="recommended_borrowing_state",
        states=("retain", "taper", "suspend"),
        eligible_key="borrowing_eligible",
        mean_fields=(
            (
                "current_effective_borrowing_fraction",
                "mean_terminal_effective_borrowing_fraction",
                "mean_effective_borrowing_fraction_when_reviewed",
            ),
            (
                "current_effective_historical_control_sample_size",
                "mean_terminal_effective_historical_control_sample_size",
                "mean_effective_historical_control_sample_size_when_reviewed",
            ),
        ),
        scenario_param_keys=(
            ("p_control", "p_treatment")
            if expected_family == "beta_binomial"
            else ("mean_control", "mean_treatment")
        ),
        analyze_design=analyze_design,
        build_review=lambda report, policy: _build_historical_control_borrowing_review(
            report,
            policy,
            expected_family=expected_family,
            expected_schema_version=_expected_report_schema_version(expected_family),
        ),
        validate_policy=_validate_historical_control_borrowing_policy,
        simulate_observed_sequence=simulate_observed_sequence,
    )


def _simulate_robust_mixture_prior_operating_characteristics(
    spec_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
    *,
    expected_family: str,
    analyze_design: Any,
    simulate_observed_sequence: Any,
) -> dict[str, Any]:
    return _simulate_review_operating_characteristics(
        spec_or_path,
        campaign_or_path,
        policy_or_path,
        expected_family=expected_family,
        output_schema_version=(
            "nextstat_bayesian_robust_mixture_prior_operating_characteristics_v0"
        ),
        derived_review_schema_version=(
            "nextstat_bayesian_robust_mixture_prior_review_v0"
        ),
        state_key="recommended_mixture_state",
        states=("retain", "taper", "fallback_to_weak"),
        eligible_key="mixture_eligible",
        mean_fields=(
            (
                "current_informative_weight",
                "mean_terminal_informative_weight",
                "mean_informative_weight_when_reviewed",
            ),
        ),
        scenario_param_keys=(
            ("p_control", "p_treatment")
            if expected_family == "beta_binomial"
            else ("mean_control", "mean_treatment")
        ),
        analyze_design=analyze_design,
        build_review=lambda report, policy: _build_robust_mixture_prior_review(
            report,
            policy,
            expected_family=expected_family,
            expected_schema_version=_expected_report_schema_version(expected_family),
        ),
        validate_policy=_validate_robust_mixture_prior_policy,
        simulate_observed_sequence=simulate_observed_sequence,
    )


def _humanize_appendix_section(section_name: str) -> str:
    return section_name.replace("_", " ").title()


def _appendix_json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True) + "\n```"


def _appendix_markdown_section(section_name: str, section_value: dict[str, Any]) -> str:
    heading = f"## {_humanize_appendix_section(section_name)}"
    if section_name == "design_summary":
        looks = section_value.get("planned_looks", [])
        lines = [
            heading,
            "",
            f"- Design ID: `{section_value['design_id']}`",
            f"- Design family: `{section_value['design_family']}`",
            f"- Endpoint summary: {section_value['endpoint_summary']}",
            f"- Current look ID: `{section_value['current_look_id']}`",
            f"- Total planned sample size: `{section_value['total_planned_sample_size']}`",
            "",
            "### Planned Looks",
            "",
            "| Look ID | N Control | N Treatment |",
            "| --- | ---: | ---: |",
        ]
        for look in looks:
            lines.append(
                f"| `{look['id']}` | `{look['n_control']}` | `{look['n_treatment']}` |"
            )
        return "\n".join(lines)

    if section_name == "prior_specification":
        lines = [
            heading,
            "",
            "### Control Prior",
            "",
            _appendix_json_block(section_value["control_prior"]),
            "",
            "### Treatment Prior",
            "",
            _appendix_json_block(section_value["treatment_prior"]),
        ]
        likelihood = section_value.get("likelihood")
        if likelihood is not None:
            lines.extend(
                [
                    "",
                    "### Likelihood",
                    "",
                    _appendix_json_block(likelihood),
                ]
            )
        return "\n".join(lines)

    if section_name == "current_analysis":
        lines = [
            heading,
            "",
            f"- Look ID: `{section_value['look_id']}`",
            f"- Recommended action: `{section_value['recommended_action']}`",
            f"- Posterior probability > margin: `{section_value['posterior_probability_gt_margin']}`",
            f"- Treatment effect margin: `{section_value['treatment_effect_margin']}`",
            "",
            "### Posterior Effect Summary",
            "",
            _appendix_json_block(section_value["posterior_effect_summary"]),
        ]
        return "\n".join(lines)

    if section_name == "operating_characteristics":
        lines = [
            heading,
            "",
            f"- Replicates: `{section_value['n_replicates']}`",
            f"- Seed: `{section_value['seed']}`",
            "",
            "### Scenario Summaries",
            "",
            _appendix_json_block(section_value["scenario_summaries"]),
        ]
        return "\n".join(lines)

    if section_name == "posterior_predictive":
        lines = [
            heading,
            "",
            f"- Replicates: `{section_value['n_replicates']}`",
            f"- Seed: `{section_value['seed']}`",
            f"- Eventual success probability: `{section_value['eventual_success_probability']}`",
            f"- Eventual futility probability: `{section_value['eventual_futility_probability']}`",
            f"- Eventual no-decision probability: `{section_value['eventual_no_decision_probability']}`",
            f"- Expected total sample size: `{section_value['expected_total_sample_size']}`",
            "",
            "### Future Look Summaries",
            "",
            _appendix_json_block(section_value["future_look_summaries"]),
        ]
        return "\n".join(lines)

    if section_name == "prior_sensitivity":
        lines = [
            heading,
            "",
            f"- Replicates: `{section_value['n_replicates']}`",
            f"- Seed: `{section_value['seed']}`",
            f"- Baseline variant ID: `{section_value['baseline_variant_id']}`",
            "",
            "### Variant Summaries",
            "",
            _appendix_json_block(section_value["variant_summaries"]),
        ]
        return "\n".join(lines)

    return "\n".join([heading, "", _appendix_json_block(section_value)])


def render_bayesian_regulatory_appendix_markdown(
    appendix_or_path: dict[str, Any] | str | Path,
) -> str:
    """Render a frozen Bayesian regulatory appendix artifact as deterministic Markdown."""
    appendix = _coerce_regulatory_appendix(appendix_or_path)
    lines = [
        "# Bayesian Regulatory Appendix",
        "",
        f"- Appendix ID: `{appendix['appendix_id']}`",
        f"- Design family: `{appendix['design_family']}`",
        f"- Design ID: `{appendix['design_id']}`",
        f"- Source report schema version: `{appendix['source_report_schema_version']}`",
        f"- Generated from frozen report: `{appendix['generated_from_frozen_report']}`",
        f"- Stability: `{appendix.get('stability', 'research-grade')}`",
        "",
    ]
    sections = appendix["sections"]
    for section_name in appendix["section_order"]:
        lines.append(_appendix_markdown_section(section_name, sections[section_name]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Missing dependency: matplotlib. Install via `pip install nextstat[viz]`."
        ) from e


def _apply_appendix_pub_style() -> None:
    from . import report as _report_style
    import matplotlib as mpl

    _report_style._apply_pub_style()
    mpl.rcParams.update(
        {
            "figure.constrained_layout.use": True,
            "savefig.bbox": "tight",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "font.size": 10.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _new_appendix_page(figsize: tuple[float, float] = (8.27, 11.69)):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.axis("off")
    return fig, ax


def _chunk_lines(lines: list[str], *, chunk_size: int) -> list[list[str]]:
    if not lines:
        return [[]]
    return [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]


def _appendix_section_pdf_lines(
    section_name: str, section_value: dict[str, Any]
) -> list[str]:
    lines = [f"{_humanize_appendix_section(section_name)}", ""]
    pretty = json.dumps(section_value, indent=2, sort_keys=True).splitlines()
    for raw_line in pretty:
        wrapped = textwrap.wrap(raw_line, width=96) or [""]
        lines.extend(wrapped)
    return lines


def _appendix_footer(
    ax: Any, page: int, total: int, *, appendix_id: str, design_family: str
) -> None:
    ax.text(
        0.0,
        0.02,
        f"appendix={appendix_id}  family={design_family}",
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
        color="#6B7280",
        transform=ax.transAxes,
    )
    ax.text(
        1.0,
        0.02,
        f"Page {page}/{total}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#6B7280",
        transform=ax.transAxes,
    )


def write_bayesian_regulatory_appendix_pdf(
    pdf_path: str | Path,
    appendix_or_path: dict[str, Any] | str | Path,
) -> None:
    """Render a frozen Bayesian regulatory appendix artifact to a deterministic PDF."""
    appendix = _coerce_regulatory_appendix(appendix_or_path)
    _require_matplotlib()
    _apply_appendix_pub_style()

    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    output_path = Path(pdf_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    section_pages: list[tuple[str, list[str]]] = []
    for section_name in appendix["section_order"]:
        section_lines = _appendix_section_pdf_lines(
            section_name,
            appendix["sections"][section_name],
        )
        for idx, chunk in enumerate(
            _chunk_lines(section_lines, chunk_size=40), start=1
        ):
            title = _humanize_appendix_section(section_name)
            if idx > 1:
                title = f"{title} (cont.)"
            section_pages.append((title, chunk))

    total_pages = 1 + len(section_pages)
    fixed_dt = datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    metadata = {
        "Title": "NextStat Bayesian Regulatory Appendix",
        "Creator": "nextstat.bayes_design",
        "Producer": "NextStat (matplotlib)",
        "CreationDate": fixed_dt,
        "ModDate": fixed_dt,
    }

    with PdfPages(output_path, metadata=metadata) as pages:
        fig, ax = _new_appendix_page()
        ax.text(
            0.0,
            0.97,
            "Bayesian Regulatory Appendix",
            ha="left",
            va="top",
            fontsize=18,
            weight="bold",
            color="#111827",
            transform=ax.transAxes,
        )
        cover_lines = [
            f"Appendix ID: {appendix['appendix_id']}",
            f"Design family: {appendix['design_family']}",
            f"Design ID: {appendix['design_id']}",
            f"Source report schema version: {appendix['source_report_schema_version']}",
            f"Generated from frozen report: {appendix['generated_from_frozen_report']}",
            f"Stability: {appendix.get('stability', 'research-grade')}",
            "",
            "Sections:",
            *[
                f"- {_humanize_appendix_section(section)}"
                for section in appendix["section_order"]
            ],
        ]
        ax.text(
            0.0,
            0.88,
            "\n".join(cover_lines),
            ha="left",
            va="top",
            fontsize=11,
            family="monospace",
            color="#111827",
            transform=ax.transAxes,
            linespacing=1.35,
        )
        _appendix_footer(
            ax,
            1,
            total_pages,
            appendix_id=str(appendix["appendix_id"]),
            design_family=str(appendix["design_family"]),
        )
        pages.savefig(fig)
        plt.close(fig)

        for page_idx, (title, lines) in enumerate(section_pages, start=2):
            fig, ax = _new_appendix_page()
            ax.text(
                0.0,
                0.97,
                title,
                ha="left",
                va="top",
                fontsize=15,
                weight="bold",
                color="#111827",
                transform=ax.transAxes,
            )
            ax.text(
                0.0,
                0.91,
                "\n".join(lines),
                ha="left",
                va="top",
                fontsize=9.5,
                family="monospace",
                color="#111827",
                transform=ax.transAxes,
                linespacing=1.22,
            )
            _appendix_footer(
                ax,
                page_idx,
                total_pages,
                appendix_id=str(appendix["appendix_id"]),
                design_family=str(appendix["design_family"]),
            )
            pages.savefig(fig)
            plt.close(fig)


def _write_design_report_bundle(
    bundle_dir: str | Path,
    report_or_path: dict[str, Any] | str | Path,
    *,
    expected_family: str,
    expected_schema_version: str,
    render_report: Any,
    command: str,
) -> dict[str, Any]:
    report_text, original_path = _coerce_json_artifact(
        report_or_path,
        inline_name=f"<inline-{expected_family}-design-report.json>",
    )
    report = json.loads(report_text)
    if not isinstance(report, dict):
        raise ValueError("report must decode to a JSON object")
    _validate_design_report(
        report,
        expected_family=expected_family,
        expected_schema_version=expected_schema_version,
    )

    markdown = render_report(report_text)
    artifact_paths = {
        "run_bundle_meta": "meta.json",
        "run_bundle_manifest": "manifest.json",
        "frozen_report_json": "inputs/input.json",
        "design_report_markdown": "outputs/design_report.md",
        "design_spec": "outputs/design_spec.json",
        "current_analysis": "outputs/current_analysis.json",
        "operating_characteristics": "outputs/operating_characteristics.json",
        "posterior_predictive": "outputs/posterior_predictive.json",
        "prior_sensitivity": "outputs/prior_sensitivity.json",
        "provenance": "outputs/provenance.json",
    }
    summary = {
        "schema_version": "nextstat_bayesian_design_report_bundle_v0",
        "stability": "research-grade",
        "design_family": expected_family,
        "report_schema_version": report["schema_version"],
        "deterministic": True,
        "artifact_paths": artifact_paths,
    }

    bundle_path = Path(bundle_dir)
    source_path = Path(original_path)
    use_source_path = source_path.exists()
    provenance = report.get("provenance", {})
    tool_version = None
    if isinstance(provenance, dict):
        software_version = provenance.get("software_version")
        if software_version is not None:
            tool_version = str(software_version)

    with tempfile.TemporaryDirectory() as td:
        staged_report_path = Path(td) / "design_report.json"
        if not use_source_path:
            staged_report_path.write_text(report_text, encoding="utf-8")
        input_path = source_path if use_source_path else staged_report_path

        _audit.write_bundle(
            bundle_path,
            command=command,
            args={
                "design_family": expected_family,
                "report_schema_version": report["schema_version"],
                "bundle_schema_version": "nextstat_bayesian_design_report_bundle_v0",
            },
            input_path=input_path,
            output_value=summary,
            tool_version=tool_version,
            deterministic=True,
        )

    if original_path != str(input_path):
        meta_path = bundle_path / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["input"]["original_path"] = original_path
        meta_path.write_bytes(_audit._json_pretty_bytes(meta))

    outputs_dir = bundle_path / "outputs"
    output_jsons = {
        "design_spec.json": report["design_spec"],
        "current_analysis.json": report["current_analysis"],
        "operating_characteristics.json": report["operating_characteristics"],
        "posterior_predictive.json": report["posterior_predictive"],
        "prior_sensitivity.json": report["prior_sensitivity"],
        "provenance.json": report["provenance"],
    }
    for filename, payload in output_jsons.items():
        (outputs_dir / filename).write_bytes(_audit._json_pretty_bytes(payload))
    (outputs_dir / "design_report.md").write_text(markdown, encoding="utf-8")
    _audit._write_manifest(bundle_path)
    return summary


def analyze_beta_binomial_design(
    spec_or_path: dict[str, Any] | str | Path,
    observed_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Analyze a beta-binomial Bayesian design at a named look."""
    if _beta_binomial_design_analyze_json is None:
        raise ImportError(
            "nextstat._core.beta_binomial_design_analyze_json is unavailable; rebuild/reinstall "
            "the NextStat Python extension to use nextstat.bayes_design"
        )
    spec_json = _coerce_json_payload(spec_or_path)
    observed_json = _coerce_json_payload(observed_or_path)
    return json.loads(_beta_binomial_design_analyze_json(spec_json, observed_json))


def simulate_beta_binomial_design(
    spec_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Compute seeded operating characteristics for a beta-binomial design."""
    if _beta_binomial_design_operating_characteristics_json is None:
        raise ImportError(
            "nextstat._core.beta_binomial_design_operating_characteristics_json is unavailable; "
            "rebuild/reinstall the NextStat Python extension to use nextstat.bayes_design"
        )
    spec_json = _coerce_json_payload(spec_or_path)
    return json.loads(_beta_binomial_design_operating_characteristics_json(spec_json))


def forecast_beta_binomial_design(
    spec_or_path: dict[str, Any] | str | Path,
    observed_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Forecast conditional design outcomes from current beta-binomial data."""
    if _beta_binomial_design_posterior_predictive_json is None:
        raise ImportError(
            "nextstat._core.beta_binomial_design_posterior_predictive_json is unavailable; "
            "rebuild/reinstall the NextStat Python extension to use nextstat.bayes_design"
        )
    spec_json = _coerce_json_payload(spec_or_path)
    observed_json = _coerce_json_payload(observed_or_path)
    return json.loads(
        _beta_binomial_design_posterior_predictive_json(spec_json, observed_json)
    )


def analyze_beta_binomial_prior_sensitivity(
    spec_or_path: dict[str, Any] | str | Path,
    observed_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Compare beta-binomial design conclusions across explicit prior variants."""
    if _beta_binomial_design_prior_sensitivity_json is None:
        raise ImportError(
            "nextstat._core.beta_binomial_design_prior_sensitivity_json is unavailable; "
            "rebuild/reinstall the NextStat Python extension to use nextstat.bayes_design"
        )
    spec_json = _coerce_json_payload(spec_or_path)
    observed_json = _coerce_json_payload(observed_or_path)
    campaign_json = _coerce_json_payload(campaign_or_path)
    return json.loads(
        _beta_binomial_design_prior_sensitivity_json(
            spec_json, observed_json, campaign_json
        )
    )


def build_beta_binomial_design_report(
    spec_or_path: dict[str, Any] | str | Path,
    observed_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Build the stable beta-binomial design report artifact."""
    if _beta_binomial_design_report_json is None:
        raise ImportError(
            "nextstat._core.beta_binomial_design_report_json is unavailable; rebuild/reinstall "
            "the NextStat Python extension to use nextstat.bayes_design"
        )
    spec_json = _coerce_json_payload(spec_or_path)
    observed_json = _coerce_json_payload(observed_or_path)
    campaign_json = _coerce_json_payload(campaign_or_path)
    return json.loads(
        _beta_binomial_design_report_json(spec_json, observed_json, campaign_json)
    )


def render_beta_binomial_design_report(
    report_or_path: dict[str, Any] | str | Path,
) -> str:
    """Render a beta-binomial design report artifact as deterministic Markdown."""
    if _beta_binomial_design_report_markdown is None:
        raise ImportError(
            "nextstat._core.beta_binomial_design_report_markdown is unavailable; rebuild/"
            "reinstall the NextStat Python extension to use nextstat.bayes_design"
        )
    report_json = _coerce_json_payload(report_or_path)
    return _beta_binomial_design_report_markdown(report_json)


def write_beta_binomial_design_report_bundle(
    bundle_dir: str | Path,
    report_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Package a frozen beta-binomial design report into a deterministic run bundle."""
    return _write_design_report_bundle(
        bundle_dir,
        report_or_path,
        expected_family="beta_binomial",
        expected_schema_version="nextstat_beta_binomial_design_report_v0",
        render_report=render_beta_binomial_design_report,
        command="bayes_design.write_beta_binomial_design_report_bundle",
    )


def build_beta_binomial_regulatory_appendix(
    report_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Build a frozen beta-binomial regulatory appendix artifact from a design report."""
    return _build_regulatory_appendix(
        report_or_path,
        expected_family="beta_binomial",
        expected_schema_version="nextstat_beta_binomial_design_report_v0",
    )


def build_beta_binomial_prior_conflict_diagnostic(
    report_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Build a frozen beta-binomial prior-data conflict diagnostic from a design report."""
    return _build_prior_conflict_diagnostic(
        report_or_path,
        expected_family="beta_binomial",
        expected_schema_version="nextstat_beta_binomial_design_report_v0",
    )


def build_beta_binomial_historical_control_borrowing_review(
    report_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Build a frozen beta-binomial historical-control borrowing review from a design report and policy."""
    return _build_historical_control_borrowing_review(
        report_or_path,
        policy_or_path,
        expected_family="beta_binomial",
        expected_schema_version="nextstat_beta_binomial_design_report_v0",
    )


def simulate_beta_binomial_historical_control_borrowing_operating_characteristics(
    spec_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Compute seeded borrowing-review operating characteristics for a beta-binomial design."""
    return _simulate_historical_control_borrowing_operating_characteristics(
        spec_or_path,
        campaign_or_path,
        policy_or_path,
        expected_family="beta_binomial",
        analyze_design=analyze_beta_binomial_design,
        simulate_observed_sequence=_simulate_beta_binomial_observed_sequence,
    )


def build_beta_binomial_robust_mixture_prior_review(
    report_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Build a frozen beta-binomial robust-mixture prior review from a design report and policy."""
    return _build_robust_mixture_prior_review(
        report_or_path,
        policy_or_path,
        expected_family="beta_binomial",
        expected_schema_version="nextstat_beta_binomial_design_report_v0",
    )


def simulate_beta_binomial_robust_mixture_prior_operating_characteristics(
    spec_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Compute seeded robust-mixture-review operating characteristics for a beta-binomial design."""
    return _simulate_robust_mixture_prior_operating_characteristics(
        spec_or_path,
        campaign_or_path,
        policy_or_path,
        expected_family="beta_binomial",
        analyze_design=analyze_beta_binomial_design,
        simulate_observed_sequence=_simulate_beta_binomial_observed_sequence,
    )


def analyze_normal_normal_design(
    spec_or_path: dict[str, Any] | str | Path,
    observed_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Analyze a normal-normal Bayesian design at a named look."""
    if _normal_normal_design_analyze_json is None:
        raise ImportError(
            "nextstat._core.normal_normal_design_analyze_json is unavailable; rebuild/reinstall "
            "the NextStat Python extension to use nextstat.bayes_design"
        )
    spec_json = _coerce_json_payload(spec_or_path)
    observed_json = _coerce_json_payload(observed_or_path)
    return json.loads(_normal_normal_design_analyze_json(spec_json, observed_json))


def simulate_normal_normal_design(
    spec_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Compute seeded operating characteristics for a normal-normal design."""
    if _normal_normal_design_operating_characteristics_json is None:
        raise ImportError(
            "nextstat._core.normal_normal_design_operating_characteristics_json is unavailable; "
            "rebuild/reinstall the NextStat Python extension to use nextstat.bayes_design"
        )
    spec_json = _coerce_json_payload(spec_or_path)
    return json.loads(_normal_normal_design_operating_characteristics_json(spec_json))


def forecast_normal_normal_design(
    spec_or_path: dict[str, Any] | str | Path,
    observed_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Forecast conditional design outcomes from current normal-normal data."""
    if _normal_normal_design_posterior_predictive_json is None:
        raise ImportError(
            "nextstat._core.normal_normal_design_posterior_predictive_json is unavailable; "
            "rebuild/reinstall the NextStat Python extension to use nextstat.bayes_design"
        )
    spec_json = _coerce_json_payload(spec_or_path)
    observed_json = _coerce_json_payload(observed_or_path)
    return json.loads(
        _normal_normal_design_posterior_predictive_json(spec_json, observed_json)
    )


def analyze_normal_normal_prior_sensitivity(
    spec_or_path: dict[str, Any] | str | Path,
    observed_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Compare normal-normal design conclusions across explicit prior variants."""
    if _normal_normal_design_prior_sensitivity_json is None:
        raise ImportError(
            "nextstat._core.normal_normal_design_prior_sensitivity_json is unavailable; "
            "rebuild/reinstall the NextStat Python extension to use nextstat.bayes_design"
        )
    spec_json = _coerce_json_payload(spec_or_path)
    observed_json = _coerce_json_payload(observed_or_path)
    campaign_json = _coerce_json_payload(campaign_or_path)
    return json.loads(
        _normal_normal_design_prior_sensitivity_json(
            spec_json, observed_json, campaign_json
        )
    )


def build_normal_normal_design_report(
    spec_or_path: dict[str, Any] | str | Path,
    observed_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Build the stable normal-normal design report artifact."""
    if _normal_normal_design_report_json is None:
        raise ImportError(
            "nextstat._core.normal_normal_design_report_json is unavailable; rebuild/reinstall "
            "the NextStat Python extension to use nextstat.bayes_design"
        )
    spec_json = _coerce_json_payload(spec_or_path)
    observed_json = _coerce_json_payload(observed_or_path)
    campaign_json = _coerce_json_payload(campaign_or_path)
    return json.loads(
        _normal_normal_design_report_json(spec_json, observed_json, campaign_json)
    )


def render_normal_normal_design_report(
    report_or_path: dict[str, Any] | str | Path,
) -> str:
    """Render a normal-normal design report artifact as deterministic Markdown."""
    if _normal_normal_design_report_markdown is None:
        raise ImportError(
            "nextstat._core.normal_normal_design_report_markdown is unavailable; rebuild/"
            "reinstall the NextStat Python extension to use nextstat.bayes_design"
        )
    report_json = _coerce_json_payload(report_or_path)
    return _normal_normal_design_report_markdown(report_json)


def write_normal_normal_design_report_bundle(
    bundle_dir: str | Path,
    report_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Package a frozen normal-normal design report into a deterministic run bundle."""
    return _write_design_report_bundle(
        bundle_dir,
        report_or_path,
        expected_family="normal_normal",
        expected_schema_version="nextstat_normal_normal_design_report_v0",
        render_report=render_normal_normal_design_report,
        command="bayes_design.write_normal_normal_design_report_bundle",
    )


def build_normal_normal_regulatory_appendix(
    report_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Build a frozen normal-normal regulatory appendix artifact from a design report."""
    return _build_regulatory_appendix(
        report_or_path,
        expected_family="normal_normal",
        expected_schema_version="nextstat_normal_normal_design_report_v0",
    )


def build_normal_normal_prior_conflict_diagnostic(
    report_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Build a frozen normal-normal prior-data conflict diagnostic from a design report."""
    return _build_prior_conflict_diagnostic(
        report_or_path,
        expected_family="normal_normal",
        expected_schema_version="nextstat_normal_normal_design_report_v0",
    )


def build_normal_normal_historical_control_borrowing_review(
    report_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Build a frozen normal-normal historical-control borrowing review from a design report and policy."""
    return _build_historical_control_borrowing_review(
        report_or_path,
        policy_or_path,
        expected_family="normal_normal",
        expected_schema_version="nextstat_normal_normal_design_report_v0",
    )


def simulate_normal_normal_historical_control_borrowing_operating_characteristics(
    spec_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Compute seeded borrowing-review operating characteristics for a normal-normal design."""
    return _simulate_historical_control_borrowing_operating_characteristics(
        spec_or_path,
        campaign_or_path,
        policy_or_path,
        expected_family="normal_normal",
        analyze_design=analyze_normal_normal_design,
        simulate_observed_sequence=_simulate_normal_normal_observed_sequence,
    )


def build_normal_normal_robust_mixture_prior_review(
    report_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Build a frozen normal-normal robust-mixture prior review from a design report and policy."""
    return _build_robust_mixture_prior_review(
        report_or_path,
        policy_or_path,
        expected_family="normal_normal",
        expected_schema_version="nextstat_normal_normal_design_report_v0",
    )


def simulate_normal_normal_robust_mixture_prior_operating_characteristics(
    spec_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Compute seeded robust-mixture-review operating characteristics for a normal-normal design."""
    return _simulate_robust_mixture_prior_operating_characteristics(
        spec_or_path,
        campaign_or_path,
        policy_or_path,
        expected_family="normal_normal",
        analyze_design=analyze_normal_normal_design,
        simulate_observed_sequence=_simulate_normal_normal_observed_sequence,
    )


__all__ = [
    "analyze_beta_binomial_design",
    "analyze_beta_binomial_prior_sensitivity",
    "analyze_normal_normal_design",
    "analyze_normal_normal_prior_sensitivity",
    "build_beta_binomial_design_report",
    "build_beta_binomial_historical_control_borrowing_review",
    "build_beta_binomial_prior_conflict_diagnostic",
    "build_beta_binomial_robust_mixture_prior_review",
    "build_beta_binomial_regulatory_appendix",
    "build_normal_normal_design_report",
    "build_normal_normal_historical_control_borrowing_review",
    "build_normal_normal_prior_conflict_diagnostic",
    "build_normal_normal_robust_mixture_prior_review",
    "build_normal_normal_regulatory_appendix",
    "forecast_beta_binomial_design",
    "forecast_normal_normal_design",
    "render_bayesian_regulatory_appendix_markdown",
    "render_beta_binomial_design_report",
    "render_normal_normal_design_report",
    "simulate_beta_binomial_design",
    "simulate_beta_binomial_historical_control_borrowing_operating_characteristics",
    "simulate_beta_binomial_robust_mixture_prior_operating_characteristics",
    "simulate_normal_normal_design",
    "simulate_normal_normal_historical_control_borrowing_operating_characteristics",
    "simulate_normal_normal_robust_mixture_prior_operating_characteristics",
    "write_bayesian_regulatory_appendix_pdf",
    "write_beta_binomial_design_report_bundle",
    "write_normal_normal_design_report_bundle",
]
