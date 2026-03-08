"""Contract tests for beta-binomial Bayesian trial design helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from statistics import NormalDist

import pytest

import nextstat.bayes_design as ns_bayes_design


def _design_spec() -> dict[str, object]:
    return {
        "schema_version": "nextstat_beta_binomial_design_v0",
        "design_id": "binary_superiority_demo",
        "control_prior": {"alpha": 1.0, "beta": 1.0},
        "treatment_prior": {"alpha": 1.0, "beta": 1.0},
        "looks": [
            {"id": "interim", "n_control": 20, "n_treatment": 20},
            {"id": "final", "n_control": 40, "n_treatment": 40},
        ],
        "decision_rules": {
            "success": {
                "posterior_probability_threshold": 0.95,
                "treatment_effect_margin": 0.0,
            },
            "futility": {
                "posterior_probability_threshold": 0.20,
                "treatment_effect_margin": 0.0,
            },
        },
        "analysis": {"credible_interval_level": 0.95},
        "simulation": {
            "n_replicates": 32,
            "seed": 123,
            "scenarios": [
                {"id": "null", "p_control": 0.40, "p_treatment": 0.40},
                {"id": "alt", "p_control": 0.40, "p_treatment": 0.60},
            ],
        },
    }


def _normal_design_spec() -> dict[str, object]:
    return {
        "schema_version": "nextstat_normal_normal_design_v0",
        "design_id": "continuous_superiority_demo",
        "control_prior": {"mean": 0.0, "sd": 10.0},
        "treatment_prior": {"mean": 0.0, "sd": 10.0},
        "likelihood": {"known_sd_control": 1.0, "known_sd_treatment": 1.0},
        "looks": [
            {"id": "interim", "n_control": 10, "n_treatment": 10},
            {"id": "final", "n_control": 20, "n_treatment": 20},
        ],
        "decision_rules": {
            "success": {
                "posterior_probability_threshold": 0.975,
                "treatment_effect_margin": 0.0,
            },
            "futility": {
                "posterior_probability_threshold": 0.10,
                "treatment_effect_margin": 0.0,
            },
        },
        "analysis": {"credible_interval_level": 0.95},
        "simulation": {
            "n_replicates": 32,
            "seed": 456,
            "scenarios": [
                {"id": "null", "mean_control": 0.0, "mean_treatment": 0.0},
                {"id": "alt", "mean_control": 0.0, "mean_treatment": 0.75},
            ],
        },
    }


def _beta_prior_sensitivity_campaign() -> dict[str, object]:
    return {
        "schema_version": "nextstat_beta_binomial_prior_sensitivity_campaign_v0",
        "variants": [
            {
                "id": "skeptical",
                "control_prior": {"alpha": 1.0, "beta": 1.0},
                "treatment_prior": {"alpha": 1.0, "beta": 8.0},
            },
            {
                "id": "enthusiastic",
                "control_prior": {"alpha": 1.0, "beta": 1.0},
                "treatment_prior": {"alpha": 8.0, "beta": 1.0},
            },
        ],
    }


def _beta_extension_prior_sensitivity_campaign() -> dict[str, object]:
    return {
        "schema_version": "nextstat_beta_binomial_prior_sensitivity_campaign_v0",
        "variants": [
            {
                "id": "skeptical",
                "control_prior": {"alpha": 1.0, "beta": 1.0},
                "treatment_prior": {"alpha": 1.0, "beta": 8.0},
            }
        ],
    }


def _normal_prior_sensitivity_campaign() -> dict[str, object]:
    return {
        "schema_version": "nextstat_normal_normal_prior_sensitivity_campaign_v0",
        "variants": [
            {
                "id": "skeptical",
                "control_prior": {"mean": 0.0, "sd": 10.0},
                "treatment_prior": {"mean": -1.0, "sd": 0.2},
            },
            {
                "id": "enthusiastic",
                "control_prior": {"mean": 0.0, "sd": 10.0},
                "treatment_prior": {"mean": 1.0, "sd": 0.2},
            },
        ],
    }


def _normal_extension_prior_sensitivity_campaign() -> dict[str, object]:
    return {
        "schema_version": "nextstat_normal_normal_prior_sensitivity_campaign_v0",
        "variants": [
            {
                "id": "skeptical",
                "control_prior": {"mean": 0.0, "sd": 10.0},
                "treatment_prior": {"mean": -1.0, "sd": 0.2},
            }
        ],
    }


def _extension_design_spec() -> dict[str, object]:
    spec = json.loads(json.dumps(_design_spec()))
    spec["simulation"] = {
        "n_replicates": 1,
        "seed": 321,
        "scenarios": [
            {"id": "alt", "p_control": 0.40, "p_treatment": 0.60},
        ],
    }
    return spec


def _extension_normal_design_spec() -> dict[str, object]:
    spec = json.loads(json.dumps(_normal_design_spec()))
    spec["simulation"] = {
        "n_replicates": 1,
        "seed": 654,
        "scenarios": [
            {"id": "alt", "mean_control": 0.0, "mean_treatment": 0.75},
        ],
    }
    return spec


def _beta_historical_borrowing_policy() -> dict[str, object]:
    return {
        "schema_version": "nextstat_bayesian_historical_control_borrowing_policy_v0",
        "stability": "research-grade",
        "policy_id": "binary_superiority_demo_historical_control_policy_v0",
        "design_family": "beta_binomial",
        "borrowing_model": "power_prior",
        "historical_sources": [
            {
                "source_id": "study_2019_control",
                "source_role": "external_control_arm",
                "planned_control_sample_size": 120,
                "exchangeability_assessment": "moderate",
                "data_cut_label": "csr-final",
            }
        ],
        "eligibility": {
            "minimum_current_control_sample_size": 20,
            "minimum_control_information_fraction": 0.5,
            "disallow_recommended_action_flip": True,
        },
        "borrowing_strength": {
            "full_borrowing_fraction": 0.35,
            "tapered_borrowing_fraction": 0.15,
            "suspended_borrowing_fraction": 0.0,
        },
        "conflict_response": {
            "full_borrowing_max_conflict_severity": "low",
            "tapered_borrowing_max_conflict_severity": "moderate",
            "max_eventual_success_probability_range_for_full_borrowing": 0.15,
            "max_expected_total_sample_size_range_fraction_for_full_borrowing": 0.10,
        },
    }


def _normal_historical_borrowing_policy() -> dict[str, object]:
    return {
        "schema_version": "nextstat_bayesian_historical_control_borrowing_policy_v0",
        "stability": "research-grade",
        "policy_id": "continuous_superiority_demo_historical_control_policy_v0",
        "design_family": "normal_normal",
        "borrowing_model": "commensurate",
        "historical_sources": [
            {
                "source_id": "study_2022_control",
                "source_role": "external_control_arm",
                "planned_control_sample_size": 80,
                "exchangeability_assessment": "moderate",
                "data_cut_label": "db-lock",
            }
        ],
        "eligibility": {
            "minimum_current_control_sample_size": 10,
            "minimum_control_information_fraction": 0.5,
            "disallow_recommended_action_flip": True,
        },
        "borrowing_strength": {
            "full_borrowing_fraction": 0.30,
            "tapered_borrowing_fraction": 0.10,
            "suspended_borrowing_fraction": 0.0,
        },
        "conflict_response": {
            "full_borrowing_max_conflict_severity": "low",
            "tapered_borrowing_max_conflict_severity": "moderate",
            "max_eventual_success_probability_range_for_full_borrowing": 0.12,
            "max_expected_total_sample_size_range_fraction_for_full_borrowing": 0.10,
        },
    }


def _beta_robust_mixture_policy() -> dict[str, object]:
    return {
        "schema_version": "nextstat_bayesian_robust_mixture_prior_policy_v0",
        "stability": "research-grade",
        "policy_id": "binary_superiority_demo_robust_mixture_policy_v0",
        "design_family": "beta_binomial",
        "mixture_model": "robust_mixture_beta",
        "prior_target": "control_prior",
        "mixture_components": [
            {
                "component_id": "historical_informative",
                "component_role": "informative",
                "base_weight": 0.70,
                "alpha": 18.0,
                "beta": 42.0,
            },
            {
                "component_id": "weak_reference",
                "component_role": "weak_reference",
                "base_weight": 0.30,
                "alpha": 1.0,
                "beta": 1.0,
            },
        ],
        "eligibility": {
            "minimum_information_fraction": 0.5,
            "disallow_recommended_action_flip": True,
        },
        "weight_schedule": {
            "retain_informative_weight": 0.70,
            "tapered_informative_weight": 0.35,
            "fallback_informative_weight": 0.0,
        },
        "conflict_response": {
            "retain_max_conflict_severity": "low",
            "tapered_max_conflict_severity": "moderate",
            "max_eventual_success_probability_range_for_retain": 0.15,
            "max_expected_total_sample_size_range_fraction_for_retain": 0.10,
        },
    }


def _normal_robust_mixture_policy() -> dict[str, object]:
    return {
        "schema_version": "nextstat_bayesian_robust_mixture_prior_policy_v0",
        "stability": "research-grade",
        "policy_id": "continuous_superiority_demo_robust_mixture_policy_v0",
        "design_family": "normal_normal",
        "mixture_model": "robust_mixture_normal",
        "prior_target": "treatment_prior",
        "mixture_components": [
            {
                "component_id": "historical_informative",
                "component_role": "informative",
                "base_weight": 0.65,
                "mean": 0.60,
                "sd": 0.30,
            },
            {
                "component_id": "weak_reference",
                "component_role": "weak_reference",
                "base_weight": 0.35,
                "mean": 0.0,
                "sd": 10.0,
            },
        ],
        "eligibility": {
            "minimum_information_fraction": 0.5,
            "disallow_recommended_action_flip": True,
        },
        "weight_schedule": {
            "retain_informative_weight": 0.65,
            "tapered_informative_weight": 0.25,
            "fallback_informative_weight": 0.0,
        },
        "conflict_response": {
            "retain_max_conflict_severity": "low",
            "tapered_max_conflict_severity": "moderate",
            "max_eventual_success_probability_range_for_retain": 0.12,
            "max_expected_total_sample_size_range_fraction_for_retain": 0.10,
        },
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bundle_file_bytes(bundle_dir: Path) -> dict[str, bytes]:
    return {
        path.relative_to(bundle_dir).as_posix(): path.read_bytes()
        for path in sorted(bundle_dir.rglob("*"))
        if path.is_file()
    }


def test_beta_binomial_analyze_exact_posterior_update_contract() -> None:
    result = ns_bayes_design.analyze_beta_binomial_design(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 14,
        },
    )

    assert result["schema_version"] == "nextstat_beta_binomial_design_analysis_v0"
    assert result["design_id"] == "binary_superiority_demo"
    assert result["look"]["id"] == "interim"

    control = result["posterior"]["control"]
    treatment = result["posterior"]["treatment"]
    effect = result["posterior"]["effect_difference"]

    assert control["alpha"] == pytest.approx(9.0, rel=0.0, abs=1e-12)
    assert control["beta"] == pytest.approx(13.0, rel=0.0, abs=1e-12)
    assert control["mean"] == pytest.approx(9.0 / 22.0, rel=0.0, abs=1e-12)
    assert treatment["alpha"] == pytest.approx(15.0, rel=0.0, abs=1e-12)
    assert treatment["beta"] == pytest.approx(7.0, rel=0.0, abs=1e-12)
    assert treatment["mean"] == pytest.approx(15.0 / 22.0, rel=0.0, abs=1e-12)
    assert effect["posterior_mean"] == pytest.approx(6.0 / 22.0, rel=0.0, abs=1e-12)
    assert effect["posterior_probability_gt_margin"] > 0.95

    decision = result["decision"]
    assert decision["success"] is True
    assert decision["futility"] is False
    assert decision["recommended_action"] == "stop_for_success"


def test_beta_binomial_operating_characteristics_are_seed_deterministic() -> None:
    first = ns_bayes_design.simulate_beta_binomial_design(_design_spec())
    second = ns_bayes_design.simulate_beta_binomial_design(_design_spec())

    assert first == second
    assert (
        first["schema_version"] == "nextstat_beta_binomial_operating_characteristics_v0"
    )
    assert first["stability"] == "research-grade"

    null_scenario = next(
        item for item in first["scenarios"] if item["scenario_id"] == "null"
    )
    alt_scenario = next(
        item for item in first["scenarios"] if item["scenario_id"] == "alt"
    )

    assert (
        null_scenario["success_rate"]
        + null_scenario["futility_rate"]
        + null_scenario["no_decision_rate"]
    ) == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert alt_scenario["success_rate"] > null_scenario["success_rate"]
    assert null_scenario["expected_total_sample_size"] <= 80.0


def test_beta_binomial_analyze_rejects_success_counts_above_enrollment() -> None:
    with pytest.raises(ValueError, match="control_successes must be <= n_control"):
        ns_bayes_design.analyze_beta_binomial_design(
            _design_spec(),
            {
                "look_id": "interim",
                "control_successes": 21,
                "treatment_successes": 14,
            },
        )


def test_beta_binomial_forecast_is_seed_deterministic() -> None:
    observed = {
        "look_id": "interim",
        "control_successes": 8,
        "treatment_successes": 9,
    }

    first = ns_bayes_design.forecast_beta_binomial_design(_design_spec(), observed)
    second = ns_bayes_design.forecast_beta_binomial_design(_design_spec(), observed)

    assert first == second
    assert first["schema_version"] == "nextstat_beta_binomial_posterior_predictive_v0"
    assert first["current_analysis"]["look"]["id"] == "interim"
    assert first["eventual_success_probability"] == pytest.approx(
        first["future_look_summaries"][0]["conditional_success_probability"],
        rel=0.0,
        abs=1e-12,
    )
    assert (
        first["eventual_success_probability"]
        + first["eventual_futility_probability"]
        + first["eventual_no_decision_probability"]
    ) == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert first["expected_total_sample_size"] >= first["current_total_sample_size"]
    assert first["expected_total_sample_size"] <= 80.0


def test_beta_binomial_prior_sensitivity_orders_predictive_success_by_prior() -> None:
    report = ns_bayes_design.analyze_beta_binomial_prior_sensitivity(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 9,
        },
        _beta_prior_sensitivity_campaign(),
    )

    assert (
        report["schema_version"] == "nextstat_beta_binomial_prior_sensitivity_report_v0"
    )

    baseline = next(
        item for item in report["variants"] if item["variant_id"] == "baseline"
    )
    skeptical = next(
        item for item in report["variants"] if item["variant_id"] == "skeptical"
    )
    enthusiastic = next(
        item for item in report["variants"] if item["variant_id"] == "enthusiastic"
    )

    assert baseline["posterior_probability_delta_vs_baseline"] == pytest.approx(
        0.0, rel=0.0, abs=1e-12
    )
    assert baseline["eventual_success_probability_delta_vs_baseline"] == pytest.approx(
        0.0, rel=0.0, abs=1e-12
    )
    assert (
        skeptical["posterior_probability_gt_margin"]
        < baseline["posterior_probability_gt_margin"]
    )
    assert (
        skeptical["eventual_success_probability"]
        < baseline["eventual_success_probability"]
    )
    assert (
        enthusiastic["eventual_success_probability"]
        > baseline["eventual_success_probability"]
    )


def test_beta_binomial_prior_sensitivity_rejects_duplicate_variant_ids() -> None:
    with pytest.raises(
        ValueError, match="duplicate prior sensitivity variant id 'skeptical'"
    ):
        ns_bayes_design.analyze_beta_binomial_prior_sensitivity(
            _design_spec(),
            {
                "look_id": "interim",
                "control_successes": 8,
                "treatment_successes": 9,
            },
            {
                "schema_version": "nextstat_beta_binomial_prior_sensitivity_campaign_v0",
                "variants": [
                    {
                        "id": "skeptical",
                        "control_prior": {"alpha": 1.0, "beta": 1.0},
                        "treatment_prior": {"alpha": 1.0, "beta": 8.0},
                    },
                    {
                        "id": "skeptical",
                        "control_prior": {"alpha": 1.0, "beta": 1.0},
                        "treatment_prior": {"alpha": 8.0, "beta": 1.0},
                    },
                ],
            },
        )


def test_beta_binomial_prior_conflict_diagnostic_is_deterministic_and_flags_action_flip() -> (
    None
):
    report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 14,
        },
        _beta_prior_sensitivity_campaign(),
    )

    first = ns_bayes_design.build_beta_binomial_prior_conflict_diagnostic(report)
    second = ns_bayes_design.build_beta_binomial_prior_conflict_diagnostic(
        json.dumps(report)
    )

    assert first == second
    assert first["schema_version"] == "nextstat_bayesian_prior_conflict_diagnostic_v0"
    assert first["generated_from_frozen_report"] is True
    assert first["conflict_severity"] == "high"
    assert first["decision_instability"] is True
    assert first["baseline_variant_id"] == "baseline"
    assert first["baseline_recommended_action"] == "stop_for_success"
    assert first["metrics"]["recommended_action_flip_count"] == 1
    assert first["metrics"]["recommended_action_flip_variant_ids"] == ["skeptical"]
    assert first["metrics"]["posterior_probability_range"] == pytest.approx(
        0.9952713491957896 - 0.7834445847056318,
        rel=0.0,
        abs=1e-12,
    )
    assert first["variant_summaries"][0]["variant_id"] == "baseline"
    assert first["variant_summaries"][1]["variant_id"] == "skeptical"
    assert first["variant_summaries"][2]["variant_id"] == "enthusiastic"


def test_beta_binomial_prior_conflict_diagnostic_can_classify_low_conflict() -> None:
    report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 14,
        },
        _beta_prior_sensitivity_campaign(),
    )
    baseline_variant = json.loads(
        json.dumps(report["prior_sensitivity"]["variants"][0])
    )
    nearby_variant = json.loads(json.dumps(baseline_variant))
    nearby_variant["variant_id"] = "nearby"
    nearby_variant["is_baseline"] = False
    nearby_variant["posterior_probability_gt_margin"] = (
        baseline_variant["posterior_probability_gt_margin"] - 0.005
    )
    nearby_variant["posterior_probability_delta_vs_baseline"] = -0.005
    nearby_variant["eventual_success_probability"] = 0.96
    nearby_variant["eventual_success_probability_delta_vs_baseline"] = -0.04
    nearby_variant["expected_total_sample_size"] = 44.0
    nearby_variant["expected_total_sample_size_delta_vs_baseline"] = 4.0
    report["prior_sensitivity"]["variants"] = [baseline_variant, nearby_variant]

    diagnostic = ns_bayes_design.build_beta_binomial_prior_conflict_diagnostic(report)

    assert diagnostic["conflict_severity"] == "low"
    assert diagnostic["decision_instability"] is False
    assert diagnostic["metrics"]["recommended_action_flip_count"] == 0
    assert diagnostic["metrics"]["decision_margin_ratio"] < 0.5


def test_beta_binomial_prior_conflict_diagnostic_rejects_missing_prior_variants() -> (
    None
):
    report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 14,
        },
        _beta_prior_sensitivity_campaign(),
    )
    report["prior_sensitivity"]["variants"] = []

    with pytest.raises(
        ValueError, match="report.prior_sensitivity.variants must be non-empty"
    ):
        ns_bayes_design.build_beta_binomial_prior_conflict_diagnostic(report)


def test_beta_binomial_prior_conflict_diagnostic_rejects_stale_prior_sensitivity_snapshot() -> (
    None
):
    report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 14,
        },
        _beta_prior_sensitivity_campaign(),
    )
    report["prior_sensitivity"]["observed"]["control_successes"] += 1

    with pytest.raises(
        ValueError,
        match="report.prior_sensitivity.observed must match report.current_analysis.observed",
    ):
        ns_bayes_design.build_beta_binomial_prior_conflict_diagnostic(report)


def test_beta_binomial_prior_conflict_diagnostic_rejects_provenance_schema_mismatch() -> (
    None
):
    report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 14,
        },
        _beta_prior_sensitivity_campaign(),
    )
    report["provenance"]["prior_sensitivity_report_schema_version"] = (
        "corrupted_schema_version"
    )

    with pytest.raises(
        ValueError,
        match="report.prior_sensitivity.schema_version must match provenance.prior_sensitivity_report_schema_version",
    ):
        ns_bayes_design.build_beta_binomial_prior_conflict_diagnostic(report)


def test_beta_binomial_historical_control_borrowing_review_suspends_on_high_conflict() -> (
    None
):
    report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 14,
        },
        _beta_prior_sensitivity_campaign(),
    )

    first = ns_bayes_design.build_beta_binomial_historical_control_borrowing_review(
        report,
        _beta_historical_borrowing_policy(),
    )
    second = ns_bayes_design.build_beta_binomial_historical_control_borrowing_review(
        json.dumps(report),
        json.dumps(_beta_historical_borrowing_policy()),
    )

    assert first == second
    assert (
        first["schema_version"]
        == "nextstat_bayesian_historical_control_borrowing_review_v0"
    )
    assert first["generated_from_frozen_report"] is True
    assert first["recommended_borrowing_state"] == "suspend"
    assert first["borrowing_eligible"] is False
    assert first["current_effective_borrowing_fraction"] == pytest.approx(0.0)
    assert first["diagnostics"]["prior_conflict_severity"] == "high"
    assert first["diagnostics"]["recommended_action_flip_count"] == 1
    assert first["diagnostics"][
        "current_control_information_fraction"
    ] == pytest.approx(0.5)
    assert first["diagnostics"][
        "total_planned_historical_control_sample_size"
    ] == pytest.approx(120.0)


def test_beta_binomial_historical_control_borrowing_review_can_taper_on_moderate_conflict() -> (
    None
):
    report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 14,
        },
        _beta_prior_sensitivity_campaign(),
    )
    baseline_variant = json.loads(
        json.dumps(report["prior_sensitivity"]["variants"][0])
    )
    moderate_variant = json.loads(json.dumps(baseline_variant))
    moderate_variant["variant_id"] = "moderate"
    moderate_variant["is_baseline"] = False
    moderate_variant["posterior_probability_gt_margin"] = 0.956
    moderate_variant["posterior_probability_delta_vs_baseline"] = -0.009
    moderate_variant["eventual_success_probability"] = 0.96
    moderate_variant["eventual_success_probability_delta_vs_baseline"] = -0.04
    moderate_variant["expected_total_sample_size"] = 44.0
    moderate_variant["expected_total_sample_size_delta_vs_baseline"] = 4.0
    report["prior_sensitivity"]["variants"] = [baseline_variant, moderate_variant]

    review = ns_bayes_design.build_beta_binomial_historical_control_borrowing_review(
        report,
        _beta_historical_borrowing_policy(),
    )

    assert review["recommended_borrowing_state"] == "taper"
    assert review["borrowing_eligible"] is True
    assert review["current_effective_borrowing_fraction"] == pytest.approx(0.15)
    assert review["diagnostics"]["prior_conflict_severity"] == "moderate"
    assert review["diagnostics"]["recommended_action_flip_count"] == 0


def test_beta_binomial_historical_control_borrowing_review_can_retain_on_low_conflict() -> (
    None
):
    report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 14,
        },
        _beta_prior_sensitivity_campaign(),
    )
    baseline_variant = json.loads(
        json.dumps(report["prior_sensitivity"]["variants"][0])
    )
    nearby_variant = json.loads(json.dumps(baseline_variant))
    nearby_variant["variant_id"] = "nearby"
    nearby_variant["is_baseline"] = False
    nearby_variant["posterior_probability_gt_margin"] = 0.961
    nearby_variant["posterior_probability_delta_vs_baseline"] = -0.004
    nearby_variant["eventual_success_probability"] = 0.98
    nearby_variant["eventual_success_probability_delta_vs_baseline"] = -0.02
    nearby_variant["expected_total_sample_size"] = 42.0
    nearby_variant["expected_total_sample_size_delta_vs_baseline"] = 2.0
    report["prior_sensitivity"]["variants"] = [baseline_variant, nearby_variant]

    review = ns_bayes_design.build_beta_binomial_historical_control_borrowing_review(
        report,
        _beta_historical_borrowing_policy(),
    )

    assert review["recommended_borrowing_state"] == "retain"
    assert review["borrowing_eligible"] is True
    assert review["current_effective_borrowing_fraction"] == pytest.approx(0.35)
    assert review["diagnostics"]["prior_conflict_severity"] == "low"


def test_beta_binomial_robust_mixture_prior_review_falls_back_to_weak_on_high_conflict() -> (
    None
):
    report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 14,
        },
        _beta_prior_sensitivity_campaign(),
    )

    first = ns_bayes_design.build_beta_binomial_robust_mixture_prior_review(
        report,
        _beta_robust_mixture_policy(),
    )
    second = ns_bayes_design.build_beta_binomial_robust_mixture_prior_review(
        json.dumps(report),
        json.dumps(_beta_robust_mixture_policy()),
    )

    assert first == second
    assert first["schema_version"] == "nextstat_bayesian_robust_mixture_prior_review_v0"
    assert first["generated_from_frozen_report"] is True
    assert first["recommended_mixture_state"] == "fallback_to_weak"
    assert first["mixture_eligible"] is False
    assert first["current_informative_weight"] == pytest.approx(0.0)
    assert first["diagnostics"]["prior_conflict_severity"] == "high"
    assert first["diagnostics"]["decision_instability"] is True
    assert first["diagnostics"]["current_information_fraction"] == pytest.approx(0.5)
    assert first["effective_component_weights"][0]["effective_weight"] == pytest.approx(
        0.0
    )
    assert first["effective_component_weights"][1]["effective_weight"] == pytest.approx(
        1.0
    )


def test_beta_binomial_robust_mixture_prior_review_can_taper_on_moderate_conflict() -> (
    None
):
    report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 14,
        },
        _beta_prior_sensitivity_campaign(),
    )
    baseline_variant = json.loads(
        json.dumps(report["prior_sensitivity"]["variants"][0])
    )
    moderate_variant = json.loads(json.dumps(baseline_variant))
    moderate_variant["variant_id"] = "moderate"
    moderate_variant["is_baseline"] = False
    moderate_variant["posterior_probability_gt_margin"] = 0.956
    moderate_variant["posterior_probability_delta_vs_baseline"] = -0.009
    moderate_variant["eventual_success_probability"] = 0.96
    moderate_variant["eventual_success_probability_delta_vs_baseline"] = -0.04
    moderate_variant["expected_total_sample_size"] = 44.0
    moderate_variant["expected_total_sample_size_delta_vs_baseline"] = 4.0
    report["prior_sensitivity"]["variants"] = [baseline_variant, moderate_variant]

    review = ns_bayes_design.build_beta_binomial_robust_mixture_prior_review(
        report,
        _beta_robust_mixture_policy(),
    )

    assert review["recommended_mixture_state"] == "taper"
    assert review["mixture_eligible"] is True
    assert review["current_informative_weight"] == pytest.approx(0.35)
    assert review["effective_component_weights"][0][
        "effective_weight"
    ] == pytest.approx(0.35)
    assert review["effective_component_weights"][1][
        "effective_weight"
    ] == pytest.approx(0.65)
    assert review["diagnostics"]["prior_conflict_severity"] == "moderate"


def test_beta_binomial_robust_mixture_prior_review_can_retain_on_low_conflict() -> None:
    report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 14,
        },
        _beta_prior_sensitivity_campaign(),
    )
    baseline_variant = json.loads(
        json.dumps(report["prior_sensitivity"]["variants"][0])
    )
    nearby_variant = json.loads(json.dumps(baseline_variant))
    nearby_variant["variant_id"] = "nearby"
    nearby_variant["is_baseline"] = False
    nearby_variant["posterior_probability_gt_margin"] = 0.961
    nearby_variant["posterior_probability_delta_vs_baseline"] = -0.004
    nearby_variant["eventual_success_probability"] = 0.98
    nearby_variant["eventual_success_probability_delta_vs_baseline"] = -0.02
    nearby_variant["expected_total_sample_size"] = 42.0
    nearby_variant["expected_total_sample_size_delta_vs_baseline"] = 2.0
    report["prior_sensitivity"]["variants"] = [baseline_variant, nearby_variant]

    review = ns_bayes_design.build_beta_binomial_robust_mixture_prior_review(
        report,
        _beta_robust_mixture_policy(),
    )

    assert review["recommended_mixture_state"] == "retain"
    assert review["mixture_eligible"] is True
    assert review["current_informative_weight"] == pytest.approx(0.70)
    assert review["effective_component_weights"][0][
        "effective_weight"
    ] == pytest.approx(0.70)
    assert review["effective_component_weights"][1][
        "effective_weight"
    ] == pytest.approx(0.30)
    assert review["diagnostics"]["prior_conflict_severity"] == "low"


def test_normal_normal_robust_mixture_prior_review_rejects_family_mismatch() -> None:
    report = ns_bayes_design.build_normal_normal_design_report(
        _normal_design_spec(),
        {
            "look_id": "interim",
            "control_sample_mean": 0.1,
            "treatment_sample_mean": 1.0,
        },
        _normal_prior_sensitivity_campaign(),
    )

    with pytest.raises(ValueError, match="policy.design_family must be normal_normal"):
        ns_bayes_design.build_normal_normal_robust_mixture_prior_review(
            report,
            _beta_robust_mixture_policy(),
        )


def test_robust_mixture_prior_reviews_match_committed_examples() -> None:
    beta_report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 14,
        },
        _beta_prior_sensitivity_campaign(),
    )
    normal_report = ns_bayes_design.build_normal_normal_design_report(
        _normal_design_spec(),
        {
            "look_id": "interim",
            "control_sample_mean": 0.1,
            "treatment_sample_mean": 1.0,
        },
        _normal_prior_sensitivity_campaign(),
    )

    beta_review = ns_bayes_design.build_beta_binomial_robust_mixture_prior_review(
        beta_report,
        _beta_robust_mixture_policy(),
    )
    normal_review = ns_bayes_design.build_normal_normal_robust_mixture_prior_review(
        normal_report,
        _normal_robust_mixture_policy(),
    )

    expected_beta = json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/bayesian_robust_mixture_prior_review_beta_binomial_v0.example.json"
        ).read_text(encoding="utf-8")
    )
    expected_normal = json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/bayesian_robust_mixture_prior_review_normal_normal_v0.example.json"
        ).read_text(encoding="utf-8")
    )

    assert beta_review == expected_beta
    assert normal_review == expected_normal


def test_beta_binomial_regulatory_appendix_is_deterministic_from_frozen_report() -> (
    None
):
    report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 9,
        },
        _beta_prior_sensitivity_campaign(),
    )

    first = ns_bayes_design.build_beta_binomial_regulatory_appendix(report)
    second = ns_bayes_design.build_beta_binomial_regulatory_appendix(json.dumps(report))

    assert first == second
    assert first["schema_version"] == "nextstat_bayesian_design_regulatory_appendix_v0"
    assert first["generated_from_frozen_report"] is True
    assert (
        first["source_report_schema_version"]
        == "nextstat_beta_binomial_design_report_v0"
    )
    assert first["sections"]["design_summary"]["planned_looks"][-1]["id"] == "final"
    assert first["sections"]["current_analysis"]["recommended_action"] in {
        "continue",
        "stop_for_success",
        "stop_for_futility",
    }
    assert first["sections"]["prior_sensitivity"]["baseline_variant_id"] == "baseline"
    assert len(first["sections"]["prior_sensitivity"]["variant_summaries"]) == 3


def test_beta_binomial_regulatory_appendix_rejects_incomplete_frozen_report() -> None:
    report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 9,
        },
        _beta_prior_sensitivity_campaign(),
    )
    del report["provenance"]

    with pytest.raises(ValueError, match="report.provenance is required"):
        ns_bayes_design.build_beta_binomial_regulatory_appendix(report)


def test_beta_binomial_design_report_matches_committed_examples() -> None:
    report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 9,
        },
        _beta_prior_sensitivity_campaign(),
    )

    expected_report = json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/beta_binomial_design_report_v0.example.json"
        ).read_text(encoding="utf-8")
    )
    assert report == expected_report

    markdown = ns_bayes_design.render_beta_binomial_design_report(report)
    expected_markdown = (
        _repo_root() / "docs/specs/pharma/beta_binomial_design_report_v0.example.md"
    ).read_text(encoding="utf-8")
    assert markdown == expected_markdown


def test_normal_normal_analyze_exact_posterior_update_contract() -> None:
    result = ns_bayes_design.analyze_normal_normal_design(
        _normal_design_spec(),
        {
            "look_id": "interim",
            "control_sample_mean": 0.1,
            "treatment_sample_mean": 1.0,
        },
    )

    posterior_variance = 1.0 / (1.0 / 100.0 + 10.0)
    posterior_sd = posterior_variance**0.5
    control_mean = posterior_variance * 10.0 * 0.1
    treatment_mean = posterior_variance * 10.0 * 1.0
    diff_mean = treatment_mean - control_mean
    diff_sd = (2.0 * posterior_variance) ** 0.5
    diff_prob = 1.0 - NormalDist(mu=diff_mean, sigma=diff_sd).cdf(0.0)

    assert result["schema_version"] == "nextstat_normal_normal_design_analysis_v0"
    assert result["design_id"] == "continuous_superiority_demo"
    assert result["look"]["id"] == "interim"

    control = result["posterior"]["control"]
    treatment = result["posterior"]["treatment"]
    effect = result["posterior"]["effect_difference"]

    assert control["posterior_mean"] == pytest.approx(control_mean, rel=0.0, abs=1e-12)
    assert control["posterior_sd"] == pytest.approx(posterior_sd, rel=0.0, abs=1e-12)
    assert treatment["posterior_mean"] == pytest.approx(
        treatment_mean, rel=0.0, abs=1e-12
    )
    assert treatment["posterior_sd"] == pytest.approx(posterior_sd, rel=0.0, abs=1e-12)
    assert effect["posterior_mean"] == pytest.approx(diff_mean, rel=0.0, abs=1e-12)
    assert effect["posterior_sd"] == pytest.approx(diff_sd, rel=0.0, abs=1e-12)
    assert effect["posterior_probability_gt_margin"] == pytest.approx(
        diff_prob, rel=0.0, abs=1e-12
    )

    decision = result["decision"]
    assert decision["success"] is True
    assert decision["futility"] is False
    assert decision["recommended_action"] == "stop_for_success"


def test_normal_normal_operating_characteristics_are_seed_deterministic() -> None:
    first = ns_bayes_design.simulate_normal_normal_design(_normal_design_spec())
    second = ns_bayes_design.simulate_normal_normal_design(_normal_design_spec())

    assert first == second
    assert (
        first["schema_version"] == "nextstat_normal_normal_operating_characteristics_v0"
    )
    assert first["stability"] == "research-grade"

    null_scenario = next(
        item for item in first["scenarios"] if item["scenario_id"] == "null"
    )
    alt_scenario = next(
        item for item in first["scenarios"] if item["scenario_id"] == "alt"
    )

    assert (
        null_scenario["success_rate"]
        + null_scenario["futility_rate"]
        + null_scenario["no_decision_rate"]
    ) == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert alt_scenario["success_rate"] > null_scenario["success_rate"]
    assert null_scenario["expected_total_sample_size"] <= 40.0


def test_normal_normal_analyze_rejects_nonpositive_known_sd() -> None:
    spec = _normal_design_spec()
    spec["likelihood"] = {"known_sd_control": 0.0, "known_sd_treatment": 1.0}

    with pytest.raises(
        ValueError, match="likelihood.known_sd_control must be finite and > 0"
    ):
        ns_bayes_design.analyze_normal_normal_design(
            spec,
            {
                "look_id": "interim",
                "control_sample_mean": 0.1,
                "treatment_sample_mean": 1.0,
            },
        )


def test_normal_normal_forecast_is_seed_deterministic() -> None:
    observed = {
        "look_id": "interim",
        "control_sample_mean": 0.1,
        "treatment_sample_mean": 0.3,
    }

    first = ns_bayes_design.forecast_normal_normal_design(
        _normal_design_spec(), observed
    )
    second = ns_bayes_design.forecast_normal_normal_design(
        _normal_design_spec(), observed
    )

    assert first == second
    assert first["schema_version"] == "nextstat_normal_normal_posterior_predictive_v0"
    assert first["current_analysis"]["look"]["id"] == "interim"
    assert first["eventual_success_probability"] == pytest.approx(
        first["future_look_summaries"][0]["conditional_success_probability"],
        rel=0.0,
        abs=1e-12,
    )
    assert (
        first["eventual_success_probability"]
        + first["eventual_futility_probability"]
        + first["eventual_no_decision_probability"]
    ) == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert first["expected_total_sample_size"] >= first["current_total_sample_size"]
    assert first["expected_total_sample_size"] <= 40.0


def test_normal_normal_prior_sensitivity_orders_predictive_success_by_prior() -> None:
    report = ns_bayes_design.analyze_normal_normal_prior_sensitivity(
        _normal_design_spec(),
        {
            "look_id": "interim",
            "control_sample_mean": 0.1,
            "treatment_sample_mean": 0.3,
        },
        _normal_prior_sensitivity_campaign(),
    )

    assert (
        report["schema_version"] == "nextstat_normal_normal_prior_sensitivity_report_v0"
    )

    baseline = next(
        item for item in report["variants"] if item["variant_id"] == "baseline"
    )
    skeptical = next(
        item for item in report["variants"] if item["variant_id"] == "skeptical"
    )
    enthusiastic = next(
        item for item in report["variants"] if item["variant_id"] == "enthusiastic"
    )

    assert baseline["posterior_probability_delta_vs_baseline"] == pytest.approx(
        0.0, rel=0.0, abs=1e-12
    )
    assert baseline["eventual_success_probability_delta_vs_baseline"] == pytest.approx(
        0.0, rel=0.0, abs=1e-12
    )
    assert (
        skeptical["posterior_probability_gt_margin"]
        < baseline["posterior_probability_gt_margin"]
    )
    assert (
        skeptical["eventual_success_probability"]
        < baseline["eventual_success_probability"]
    )
    assert (
        enthusiastic["eventual_success_probability"]
        > baseline["eventual_success_probability"]
    )


def test_normal_normal_prior_sensitivity_rejects_reserved_baseline_id() -> None:
    with pytest.raises(
        ValueError, match="prior sensitivity variant id 'baseline' is reserved"
    ):
        ns_bayes_design.analyze_normal_normal_prior_sensitivity(
            _normal_design_spec(),
            {
                "look_id": "interim",
                "control_sample_mean": 0.1,
                "treatment_sample_mean": 0.3,
            },
            {
                "schema_version": "nextstat_normal_normal_prior_sensitivity_campaign_v0",
                "variants": [
                    {
                        "id": "baseline",
                        "control_prior": {"mean": 0.0, "sd": 10.0},
                        "treatment_prior": {"mean": -1.0, "sd": 0.2},
                    }
                ],
            },
        )


def test_normal_normal_prior_conflict_diagnostic_flags_futility_flip() -> None:
    report = ns_bayes_design.build_normal_normal_design_report(
        _normal_design_spec(),
        {
            "look_id": "interim",
            "control_sample_mean": 0.1,
            "treatment_sample_mean": 1.0,
        },
        _normal_prior_sensitivity_campaign(),
    )

    diagnostic = ns_bayes_design.build_normal_normal_prior_conflict_diagnostic(report)

    assert (
        diagnostic["schema_version"] == "nextstat_bayesian_prior_conflict_diagnostic_v0"
    )
    assert diagnostic["conflict_severity"] == "high"
    assert diagnostic["decision_instability"] is True
    assert diagnostic["baseline_recommended_action"] == "stop_for_success"
    assert diagnostic["metrics"]["recommended_action_flip_count"] == 1
    assert diagnostic["metrics"]["recommended_action_flip_variant_ids"] == ["skeptical"]
    assert (
        diagnostic["metrics"]["max_abs_posterior_probability_delta_vs_baseline"] > 0.9
    )
    assert diagnostic["metrics"]["posterior_probability_range"] > 0.9


def test_normal_normal_historical_control_borrowing_review_rejects_family_mismatch() -> (
    None
):
    report = ns_bayes_design.build_normal_normal_design_report(
        _normal_design_spec(),
        {
            "look_id": "interim",
            "control_sample_mean": 0.1,
            "treatment_sample_mean": 1.0,
        },
        _normal_prior_sensitivity_campaign(),
    )

    with pytest.raises(ValueError, match="policy.design_family must be normal_normal"):
        ns_bayes_design.build_normal_normal_historical_control_borrowing_review(
            report,
            _beta_historical_borrowing_policy(),
        )


def test_prior_conflict_diagnostics_match_committed_examples() -> None:
    beta_report = json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/beta_binomial_design_report_v0.example.json"
        ).read_text(encoding="utf-8")
    )
    normal_report = json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/normal_normal_design_report_v0.example.json"
        ).read_text(encoding="utf-8")
    )

    beta_diagnostic = ns_bayes_design.build_beta_binomial_prior_conflict_diagnostic(
        beta_report
    )
    normal_diagnostic = ns_bayes_design.build_normal_normal_prior_conflict_diagnostic(
        normal_report
    )

    assert beta_diagnostic == json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/beta_binomial_prior_conflict_diagnostic_v0.example.json"
        ).read_text(encoding="utf-8")
    )
    assert normal_diagnostic == json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/normal_normal_prior_conflict_diagnostic_v0.example.json"
        ).read_text(encoding="utf-8")
    )


def test_historical_control_borrowing_reviews_match_committed_examples() -> None:
    beta_report = json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/beta_binomial_design_report_v0.example.json"
        ).read_text(encoding="utf-8")
    )
    normal_report = json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/normal_normal_design_report_v0.example.json"
        ).read_text(encoding="utf-8")
    )
    beta_policy = json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/bayesian_historical_control_borrowing_policy_beta_binomial_v0.example.json"
        ).read_text(encoding="utf-8")
    )
    normal_policy = json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/bayesian_historical_control_borrowing_policy_normal_normal_v0.example.json"
        ).read_text(encoding="utf-8")
    )

    beta_review = (
        ns_bayes_design.build_beta_binomial_historical_control_borrowing_review(
            beta_report,
            beta_policy,
        )
    )
    normal_review = (
        ns_bayes_design.build_normal_normal_historical_control_borrowing_review(
            normal_report,
            normal_policy,
        )
    )

    assert beta_review == json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/bayesian_historical_control_borrowing_review_beta_binomial_v0.example.json"
        ).read_text(encoding="utf-8")
    )
    assert normal_review == json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/bayesian_historical_control_borrowing_review_normal_normal_v0.example.json"
        ).read_text(encoding="utf-8")
    )


def test_beta_binomial_historical_control_borrowing_operating_characteristics_are_deterministic() -> (
    None
):
    first = (
        ns_bayes_design.simulate_beta_binomial_historical_control_borrowing_operating_characteristics(
            _extension_design_spec(),
            _beta_extension_prior_sensitivity_campaign(),
            _beta_historical_borrowing_policy(),
        )
    )
    second = (
        ns_bayes_design.simulate_beta_binomial_historical_control_borrowing_operating_characteristics(
            json.dumps(_extension_design_spec()),
            json.dumps(_beta_extension_prior_sensitivity_campaign()),
            json.dumps(_beta_historical_borrowing_policy()),
        )
    )

    assert first == second
    assert (
        first["schema_version"]
        == "nextstat_bayesian_historical_control_borrowing_operating_characteristics_v0"
    )
    assert first["design_family"] == "beta_binomial"
    assert first["n_replicates"] == 1
    assert first["seed"] == 321
    assert len(first["scenarios"]) == 1

    for scenario in first["scenarios"]:
        assert scenario["retain_rate"] == pytest.approx(
            1.0 - scenario["taper_rate"] - scenario["suspend_rate"]
        )
        assert 0.0 <= scenario["borrowing_eligible_rate"] <= 1.0
        assert 0.0 <= scenario["decision_instability_rate"] <= 1.0
        assert 0.0 <= scenario["high_conflict_rate"] <= 1.0
        assert 0.0 <= scenario["mean_terminal_effective_borrowing_fraction"] <= 0.35
        assert (
            0.0
            <= scenario["mean_terminal_effective_historical_control_sample_size"]
            <= 42.0
        )
        for look_summary in scenario["look_summaries"]:
            assert 0.0 <= look_summary["review_probability"] <= 1.0
            assert look_summary["review_probability"] == pytest.approx(
                look_summary["retain_probability"]
                + look_summary["taper_probability"]
                + look_summary["suspend_probability"]
            )
            assert 0.0 <= look_summary["borrowing_eligible_probability"] <= 1.0
            assert 0.0 <= look_summary["decision_instability_probability"] <= 1.0
            assert 0.0 <= look_summary["high_conflict_probability"] <= 1.0
            assert (
                0.0
                <= look_summary["mean_effective_borrowing_fraction_when_reviewed"]
                <= 0.35
            )
            assert (
                0.0
                <= look_summary[
                    "mean_effective_historical_control_sample_size_when_reviewed"
                ]
                <= 42.0
            )


def test_beta_binomial_robust_mixture_prior_operating_characteristics_are_deterministic() -> (
    None
):
    first = (
        ns_bayes_design.simulate_beta_binomial_robust_mixture_prior_operating_characteristics(
            _extension_design_spec(),
            _beta_extension_prior_sensitivity_campaign(),
            _beta_robust_mixture_policy(),
        )
    )
    second = (
        ns_bayes_design.simulate_beta_binomial_robust_mixture_prior_operating_characteristics(
            json.dumps(_extension_design_spec()),
            json.dumps(_beta_extension_prior_sensitivity_campaign()),
            json.dumps(_beta_robust_mixture_policy()),
        )
    )

    assert first == second
    assert (
        first["schema_version"]
        == "nextstat_bayesian_robust_mixture_prior_operating_characteristics_v0"
    )
    assert first["design_family"] == "beta_binomial"
    assert first["n_replicates"] == 1
    assert first["seed"] == 321
    assert len(first["scenarios"]) == 1

    for scenario in first["scenarios"]:
        assert scenario["retain_rate"] == pytest.approx(
            1.0 - scenario["taper_rate"] - scenario["fallback_to_weak_rate"]
        )
        assert 0.0 <= scenario["mixture_eligible_rate"] <= 1.0
        assert 0.0 <= scenario["decision_instability_rate"] <= 1.0
        assert 0.0 <= scenario["high_conflict_rate"] <= 1.0
        assert 0.0 <= scenario["mean_terminal_informative_weight"] <= 0.70
        for look_summary in scenario["look_summaries"]:
            assert 0.0 <= look_summary["review_probability"] <= 1.0
            assert look_summary["review_probability"] == pytest.approx(
                look_summary["retain_probability"]
                + look_summary["taper_probability"]
                + look_summary["fallback_to_weak_probability"]
            )
            assert 0.0 <= look_summary["mixture_eligible_probability"] <= 1.0
            assert 0.0 <= look_summary["decision_instability_probability"] <= 1.0
            assert 0.0 <= look_summary["high_conflict_probability"] <= 1.0
            assert (
                0.0
                <= look_summary["mean_informative_weight_when_reviewed"]
                <= 0.70
            )


def test_borrowing_and_robust_prior_operating_characteristics_match_committed_examples() -> (
    None
):
    beta_historical = (
        ns_bayes_design.simulate_beta_binomial_historical_control_borrowing_operating_characteristics(
            _extension_design_spec(),
            _beta_extension_prior_sensitivity_campaign(),
            _beta_historical_borrowing_policy(),
        )
    )
    normal_historical = (
        ns_bayes_design.simulate_normal_normal_historical_control_borrowing_operating_characteristics(
            _extension_normal_design_spec(),
            _normal_extension_prior_sensitivity_campaign(),
            _normal_historical_borrowing_policy(),
        )
    )
    beta_robust = (
        ns_bayes_design.simulate_beta_binomial_robust_mixture_prior_operating_characteristics(
            _extension_design_spec(),
            _beta_extension_prior_sensitivity_campaign(),
            _beta_robust_mixture_policy(),
        )
    )
    normal_robust = (
        ns_bayes_design.simulate_normal_normal_robust_mixture_prior_operating_characteristics(
            _extension_normal_design_spec(),
            _normal_extension_prior_sensitivity_campaign(),
            _normal_robust_mixture_policy(),
        )
    )

    assert beta_historical == json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/bayesian_historical_control_borrowing_operating_characteristics_beta_binomial_v0.example.json"
        ).read_text(encoding="utf-8")
    )
    assert normal_historical == json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/bayesian_historical_control_borrowing_operating_characteristics_normal_normal_v0.example.json"
        ).read_text(encoding="utf-8")
    )
    assert beta_robust == json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/bayesian_robust_mixture_prior_operating_characteristics_beta_binomial_v0.example.json"
        ).read_text(encoding="utf-8")
    )
    assert normal_robust == json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/bayesian_robust_mixture_prior_operating_characteristics_normal_normal_v0.example.json"
        ).read_text(encoding="utf-8")
    )


def test_normal_normal_design_report_matches_committed_examples() -> None:
    report = ns_bayes_design.build_normal_normal_design_report(
        _normal_design_spec(),
        {
            "look_id": "interim",
            "control_sample_mean": 0.1,
            "treatment_sample_mean": 0.3,
        },
        _normal_prior_sensitivity_campaign(),
    )

    expected_report = json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/normal_normal_design_report_v0.example.json"
        ).read_text(encoding="utf-8")
    )
    assert report == expected_report

    markdown = ns_bayes_design.render_normal_normal_design_report(report)
    expected_markdown = (
        _repo_root() / "docs/specs/pharma/normal_normal_design_report_v0.example.md"
    ).read_text(encoding="utf-8")
    assert markdown == expected_markdown


def test_normal_normal_regulatory_appendix_carries_likelihood_and_operating_characteristics() -> (
    None
):
    report = ns_bayes_design.build_normal_normal_design_report(
        _normal_design_spec(),
        {
            "look_id": "interim",
            "control_sample_mean": 0.1,
            "treatment_sample_mean": 0.3,
        },
        _normal_prior_sensitivity_campaign(),
    )

    appendix = ns_bayes_design.build_normal_normal_regulatory_appendix(report)

    assert (
        appendix["schema_version"] == "nextstat_bayesian_design_regulatory_appendix_v0"
    )
    assert appendix["design_family"] == "normal_normal"
    assert appendix["sections"]["prior_specification"]["likelihood"] == {
        "known_sd_control": 1.0,
        "known_sd_treatment": 1.0,
    }
    scenarios = appendix["sections"]["operating_characteristics"]["scenario_summaries"]
    assert {item["scenario_id"] for item in scenarios} == {"null", "alt"}
    assert appendix["sections"]["posterior_predictive"]["future_look_summaries"][0][
        "look_id"
    ] == ("final")


def test_regulatory_appendix_markdown_matches_committed_examples() -> None:
    beta_report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 9,
        },
        _beta_prior_sensitivity_campaign(),
    )
    normal_report = ns_bayes_design.build_normal_normal_design_report(
        _normal_design_spec(),
        {
            "look_id": "interim",
            "control_sample_mean": 0.1,
            "treatment_sample_mean": 0.3,
        },
        _normal_prior_sensitivity_campaign(),
    )

    beta_markdown = ns_bayes_design.render_bayesian_regulatory_appendix_markdown(
        ns_bayes_design.build_beta_binomial_regulatory_appendix(beta_report)
    )
    normal_markdown = ns_bayes_design.render_bayesian_regulatory_appendix_markdown(
        ns_bayes_design.build_normal_normal_regulatory_appendix(normal_report)
    )

    assert beta_markdown == (
        _repo_root()
        / "docs/specs/pharma/beta_binomial_regulatory_appendix_v0.example.md"
    ).read_text(encoding="utf-8")
    assert normal_markdown == (
        _repo_root()
        / "docs/specs/pharma/normal_normal_regulatory_appendix_v0.example.md"
    ).read_text(encoding="utf-8")


def test_regulatory_appendix_pdf_is_deterministic(tmp_path: Path) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not installed")

    report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 9,
        },
        _beta_prior_sensitivity_campaign(),
    )
    appendix = ns_bayes_design.build_beta_binomial_regulatory_appendix(report)
    first_pdf = tmp_path / "first.pdf"
    second_pdf = tmp_path / "second.pdf"

    ns_bayes_design.write_bayesian_regulatory_appendix_pdf(first_pdf, appendix)
    ns_bayes_design.write_bayesian_regulatory_appendix_pdf(
        second_pdf, json.dumps(appendix)
    )

    assert first_pdf.read_bytes() == second_pdf.read_bytes()
    assert first_pdf.read_bytes().startswith(b"%PDF-")


def test_beta_binomial_design_report_bundle_is_deterministic_and_complete(
    tmp_path: Path,
) -> None:
    report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 9,
        },
        _beta_prior_sensitivity_campaign(),
    )

    first_dir = tmp_path / "bundle_first"
    second_dir = tmp_path / "bundle_second"
    first = ns_bayes_design.write_beta_binomial_design_report_bundle(first_dir, report)
    second = ns_bayes_design.write_beta_binomial_design_report_bundle(
        second_dir, report
    )

    assert first == second
    assert _bundle_file_bytes(first_dir) == _bundle_file_bytes(second_dir)
    expected_summary = json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/beta_binomial_design_report_bundle_v0.example.json"
        ).read_text(encoding="utf-8")
    )
    assert first == expected_summary

    expected_paths = {
        "run_bundle_meta": "meta.json",
        "run_bundle_manifest": "manifest.json",
        "current_analysis": "outputs/current_analysis.json",
        "design_report_markdown": "outputs/design_report.md",
        "design_spec": "outputs/design_spec.json",
        "frozen_report_json": "inputs/input.json",
        "operating_characteristics": "outputs/operating_characteristics.json",
        "posterior_predictive": "outputs/posterior_predictive.json",
        "prior_sensitivity": "outputs/prior_sensitivity.json",
        "provenance": "outputs/provenance.json",
    }
    assert first["schema_version"] == "nextstat_bayesian_design_report_bundle_v0"
    assert first["deterministic"] is True
    assert first["artifact_paths"] == expected_paths

    meta = json.loads((first_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["created_unix_ms"] == 0
    assert meta["input"]["original_path"] == "<inline-beta_binomial-design-report.json>"

    assert (
        json.loads((first_dir / "inputs" / "input.json").read_text(encoding="utf-8"))
        == report
    )
    assert (
        json.loads(
            (first_dir / "outputs" / "design_spec.json").read_text(encoding="utf-8")
        )
        == (report["design_spec"])
    )
    assert (
        json.loads(
            (first_dir / "outputs" / "current_analysis.json").read_text(
                encoding="utf-8"
            )
        )
        == report["current_analysis"]
    )
    assert (
        json.loads(
            (first_dir / "outputs" / "operating_characteristics.json").read_text(
                encoding="utf-8"
            )
        )
        == report["operating_characteristics"]
    )
    assert (
        json.loads(
            (first_dir / "outputs" / "posterior_predictive.json").read_text(
                encoding="utf-8"
            )
        )
        == report["posterior_predictive"]
    )
    assert (
        json.loads(
            (first_dir / "outputs" / "prior_sensitivity.json").read_text(
                encoding="utf-8"
            )
        )
        == report["prior_sensitivity"]
    )
    assert (
        json.loads(
            (first_dir / "outputs" / "provenance.json").read_text(encoding="utf-8")
        )
        == (report["provenance"])
    )

    markdown = (first_dir / "outputs" / "design_report.md").read_text(encoding="utf-8")
    assert markdown == ns_bayes_design.render_beta_binomial_design_report(report)

    summary = json.loads(
        (first_dir / "outputs" / "result.json").read_text(encoding="utf-8")
    )
    assert summary == first

    manifest = json.loads((first_dir / "manifest.json").read_text(encoding="utf-8"))
    assert sorted(entry["path"] for entry in manifest["files"]) == sorted(
        [
            "inputs/input.json",
            "meta.json",
            "outputs/current_analysis.json",
            "outputs/design_report.md",
            "outputs/design_spec.json",
            "outputs/operating_characteristics.json",
            "outputs/posterior_predictive.json",
            "outputs/prior_sensitivity.json",
            "outputs/provenance.json",
            "outputs/result.json",
        ]
    )


def test_normal_normal_design_report_bundle_preserves_report_path(
    tmp_path: Path,
) -> None:
    report = ns_bayes_design.build_normal_normal_design_report(
        _normal_design_spec(),
        {
            "look_id": "interim",
            "control_sample_mean": 0.1,
            "treatment_sample_mean": 1.0,
        },
        _normal_prior_sensitivity_campaign(),
    )
    report_path = tmp_path / "normal_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    summary = ns_bayes_design.write_normal_normal_design_report_bundle(
        bundle_dir, report_path
    )

    expected_summary = json.loads(
        (
            _repo_root()
            / "docs/specs/pharma/normal_normal_design_report_bundle_v0.example.json"
        ).read_text(encoding="utf-8")
    )
    assert summary == expected_summary
    assert summary["design_family"] == "normal_normal"
    meta = json.loads((bundle_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["input"]["original_path"] == str(report_path)
    assert meta["created_unix_ms"] == 0

    markdown = (bundle_dir / "outputs" / "design_report.md").read_text(encoding="utf-8")
    assert markdown == ns_bayes_design.render_normal_normal_design_report(report)


def test_beta_binomial_design_report_bundle_rejects_non_empty_dir(
    tmp_path: Path,
) -> None:
    report = ns_bayes_design.build_beta_binomial_design_report(
        _design_spec(),
        {
            "look_id": "interim",
            "control_successes": 8,
            "treatment_successes": 9,
        },
        _beta_prior_sensitivity_campaign(),
    )

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    junk_path = bundle_dir / "junk.txt"
    junk_path.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="bundle directory must be empty"):
        ns_bayes_design.write_beta_binomial_design_report_bundle(bundle_dir, report)

    assert junk_path.read_text(encoding="utf-8") == "keep"
    assert sorted(path.name for path in bundle_dir.iterdir()) == ["junk.txt"]


def test_normal_normal_design_report_bundle_rejects_missing_required_section(
    tmp_path: Path,
) -> None:
    report = ns_bayes_design.build_normal_normal_design_report(
        _normal_design_spec(),
        {
            "look_id": "interim",
            "control_sample_mean": 0.1,
            "treatment_sample_mean": 1.0,
        },
        _normal_prior_sensitivity_campaign(),
    )
    report.pop("provenance")

    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ValueError, match="report.provenance is required"):
        ns_bayes_design.write_normal_normal_design_report_bundle(bundle_dir, report)

    assert not bundle_dir.exists()
