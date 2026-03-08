from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import nextstat.bayes_design as ns_bayes_design


def _raise_hidden_execution(*_args: object, **_kwargs: object) -> str:
    raise AssertionError(
        "hidden execution path must not run for frozen report surfaces"
    )


def _beta_design_spec() -> dict[str, object]:
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


def _beta_observed() -> dict[str, object]:
    return {
        "look_id": "interim",
        "control_successes": 8,
        "treatment_successes": 14,
    }


def _beta_campaign() -> dict[str, object]:
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


def _beta_report() -> dict[str, object]:
    return {
        "schema_version": "nextstat_beta_binomial_design_report_v0",
        "stability": "research-grade",
        "design_family": "beta_binomial",
        "design_spec": _beta_design_spec(),
        "current_analysis": {
            "schema_version": "nextstat_beta_binomial_design_analysis_v0",
            "stability": "research-grade",
            "design_id": "binary_superiority_demo",
            "look": {"id": "interim", "n_control": 20, "n_treatment": 20},
            "observed": _beta_observed(),
            "posterior": {
                "control": {
                    "alpha": 9.0,
                    "beta": 13.0,
                    "mean": 0.409091,
                    "ci_lower": 0.22,
                    "ci_upper": 0.61,
                },
                "treatment": {
                    "alpha": 15.0,
                    "beta": 7.0,
                    "mean": 0.681818,
                    "ci_lower": 0.47,
                    "ci_upper": 0.84,
                },
                "effect_difference": {
                    "margin": 0.0,
                    "posterior_mean": 0.272727,
                    "posterior_probability_gt_margin": 0.965,
                },
            },
            "decision": {
                "success": True,
                "futility": False,
                "recommended_action": "stop_for_success",
                "posterior_probability_gt_margin": 0.965,
                "success_threshold": 0.95,
                "futility_threshold": 0.20,
                "margin": 0.0,
            },
        },
        "operating_characteristics": {
            "schema_version": "nextstat_beta_binomial_operating_characteristics_v0",
            "stability": "research-grade",
            "design_id": "binary_superiority_demo",
            "n_replicates": 32,
            "seed": 123,
            "scenarios": [],
        },
        "posterior_predictive": {
            "schema_version": "nextstat_beta_binomial_posterior_predictive_v0",
            "stability": "research-grade",
            "design_id": "binary_superiority_demo",
            "current_analysis": {
                "schema_version": "nextstat_beta_binomial_design_analysis_v0",
                "decision": {"recommended_action": "stop_for_success"},
            },
            "n_replicates": 32,
            "seed": 123,
            "current_total_sample_size": 40.0,
            "expected_total_sample_size": 40.0,
            "expected_remaining_sample_size": 0.0,
            "eventual_success_probability": 1.0,
            "eventual_futility_probability": 0.0,
            "eventual_no_decision_probability": 0.0,
            "future_look_summaries": [],
        },
        "prior_sensitivity": {
            "schema_version": "nextstat_beta_binomial_prior_sensitivity_report_v0",
            "stability": "research-grade",
            "design_id": "binary_superiority_demo",
            "look": {"id": "interim", "n_control": 20, "n_treatment": 20},
            "observed": _beta_observed(),
            "n_replicates": 32,
            "seed": 123,
            "variants": [
                {
                    "variant_id": "baseline",
                    "is_baseline": True,
                    "control_prior": {"alpha": 1.0, "beta": 1.0},
                    "treatment_prior": {"alpha": 1.0, "beta": 1.0},
                    "posterior_mean": 0.272727,
                    "posterior_probability_gt_margin": 0.965,
                    "recommended_action": "stop_for_success",
                    "eventual_success_probability": 1.0,
                    "eventual_futility_probability": 0.0,
                    "eventual_no_decision_probability": 0.0,
                    "expected_total_sample_size": 40.0,
                    "expected_remaining_sample_size": 0.0,
                    "future_look_summaries": [],
                    "posterior_probability_delta_vs_baseline": 0.0,
                    "eventual_success_probability_delta_vs_baseline": 0.0,
                    "expected_total_sample_size_delta_vs_baseline": 0.0,
                },
                {
                    "variant_id": "skeptical",
                    "is_baseline": False,
                    "control_prior": {"alpha": 1.0, "beta": 1.0},
                    "treatment_prior": {"alpha": 1.0, "beta": 8.0},
                    "posterior_mean": 0.10815,
                    "posterior_probability_gt_margin": 0.783445,
                    "recommended_action": "continue",
                    "eventual_success_probability": 0.15625,
                    "eventual_futility_probability": 0.0,
                    "eventual_no_decision_probability": 0.84375,
                    "expected_total_sample_size": 80.0,
                    "expected_remaining_sample_size": 40.0,
                    "future_look_summaries": [],
                    "posterior_probability_delta_vs_baseline": -0.181555,
                    "eventual_success_probability_delta_vs_baseline": -0.84375,
                    "expected_total_sample_size_delta_vs_baseline": 40.0,
                },
            ],
        },
        "provenance": {
            "software_name": "nextstat",
            "software_version": "0.0.0",
            "design_schema_version": "nextstat_beta_binomial_design_v0",
            "analysis_schema_version": "nextstat_beta_binomial_design_analysis_v0",
            "operating_characteristics_schema_version": "nextstat_beta_binomial_operating_characteristics_v0",
            "posterior_predictive_schema_version": "nextstat_beta_binomial_posterior_predictive_v0",
            "prior_sensitivity_campaign_schema_version": "nextstat_beta_binomial_prior_sensitivity_campaign_v0",
            "prior_sensitivity_report_schema_version": "nextstat_beta_binomial_prior_sensitivity_report_v0",
            "simulation_seed": 123,
            "n_replicates": 32,
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


def _normal_observed() -> dict[str, object]:
    return {
        "look_id": "interim",
        "control_sample_mean": 0.1,
        "treatment_sample_mean": 1.0,
    }


def _normal_campaign() -> dict[str, object]:
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


def _normal_report() -> dict[str, object]:
    return {
        "schema_version": "nextstat_normal_normal_design_report_v0",
        "stability": "research-grade",
        "design_family": "normal_normal",
        "design_spec": _normal_design_spec(),
        "current_analysis": {
            "schema_version": "nextstat_normal_normal_design_analysis_v0",
            "stability": "research-grade",
            "design_id": "continuous_superiority_demo",
            "look": {"id": "interim", "n_control": 10, "n_treatment": 10},
            "observed": _normal_observed(),
            "posterior": {
                "control": {
                    "posterior_mean": 0.1,
                    "posterior_sd": 0.2,
                    "ci_lower": -0.3,
                    "ci_upper": 0.5,
                },
                "treatment": {
                    "posterior_mean": 1.0,
                    "posterior_sd": 0.2,
                    "ci_lower": 0.6,
                    "ci_upper": 1.4,
                },
                "effect_difference": {
                    "margin": 0.0,
                    "posterior_mean": 0.9,
                    "posterior_sd": 0.28,
                    "ci_lower": 0.35,
                    "ci_upper": 1.45,
                    "posterior_probability_gt_margin": 0.998,
                },
            },
            "decision": {
                "success": True,
                "futility": False,
                "recommended_action": "stop_for_success",
                "posterior_probability_gt_margin": 0.998,
                "success_threshold": 0.975,
                "futility_threshold": 0.10,
                "margin": 0.0,
            },
        },
        "operating_characteristics": {
            "schema_version": "nextstat_normal_normal_operating_characteristics_v0",
            "stability": "research-grade",
            "design_id": "continuous_superiority_demo",
            "n_replicates": 32,
            "seed": 456,
            "scenarios": [],
        },
        "posterior_predictive": {
            "schema_version": "nextstat_normal_normal_posterior_predictive_v0",
            "stability": "research-grade",
            "design_id": "continuous_superiority_demo",
            "current_analysis": {
                "schema_version": "nextstat_normal_normal_design_analysis_v0",
                "decision": {"recommended_action": "stop_for_success"},
            },
            "n_replicates": 32,
            "seed": 456,
            "current_total_sample_size": 20.0,
            "expected_total_sample_size": 20.0,
            "expected_remaining_sample_size": 0.0,
            "eventual_success_probability": 1.0,
            "eventual_futility_probability": 0.0,
            "eventual_no_decision_probability": 0.0,
            "future_look_summaries": [],
        },
        "prior_sensitivity": {
            "schema_version": "nextstat_normal_normal_prior_sensitivity_report_v0",
            "stability": "research-grade",
            "design_id": "continuous_superiority_demo",
            "look": {"id": "interim", "n_control": 10, "n_treatment": 10},
            "observed": _normal_observed(),
            "n_replicates": 32,
            "seed": 456,
            "variants": [
                {
                    "variant_id": "baseline",
                    "is_baseline": True,
                    "control_prior": {"mean": 0.0, "sd": 10.0},
                    "treatment_prior": {"mean": 0.0, "sd": 10.0},
                    "posterior_mean": 0.9,
                    "posterior_probability_gt_margin": 0.998,
                    "recommended_action": "stop_for_success",
                    "eventual_success_probability": 1.0,
                    "eventual_futility_probability": 0.0,
                    "eventual_no_decision_probability": 0.0,
                    "expected_total_sample_size": 20.0,
                    "expected_remaining_sample_size": 0.0,
                    "future_look_summaries": [],
                    "posterior_probability_delta_vs_baseline": 0.0,
                    "eventual_success_probability_delta_vs_baseline": 0.0,
                    "expected_total_sample_size_delta_vs_baseline": 0.0,
                },
                {
                    "variant_id": "skeptical",
                    "is_baseline": False,
                    "control_prior": {"mean": 0.0, "sd": 10.0},
                    "treatment_prior": {"mean": -1.0, "sd": 0.2},
                    "posterior_mean": -0.528472,
                    "posterior_probability_gt_margin": 0.070186,
                    "recommended_action": "stop_for_futility",
                    "eventual_success_probability": 0.0,
                    "eventual_futility_probability": 1.0,
                    "eventual_no_decision_probability": 0.0,
                    "expected_total_sample_size": 20.0,
                    "expected_remaining_sample_size": 0.0,
                    "future_look_summaries": [],
                    "posterior_probability_delta_vs_baseline": -0.927814,
                    "eventual_success_probability_delta_vs_baseline": -1.0,
                    "expected_total_sample_size_delta_vs_baseline": 0.0,
                },
            ],
        },
        "provenance": {
            "software_name": "nextstat",
            "software_version": "0.0.0",
            "design_schema_version": "nextstat_normal_normal_design_v0",
            "analysis_schema_version": "nextstat_normal_normal_design_analysis_v0",
            "operating_characteristics_schema_version": "nextstat_normal_normal_operating_characteristics_v0",
            "posterior_predictive_schema_version": "nextstat_normal_normal_posterior_predictive_v0",
            "prior_sensitivity_campaign_schema_version": "nextstat_normal_normal_prior_sensitivity_campaign_v0",
            "prior_sensitivity_report_schema_version": "nextstat_normal_normal_prior_sensitivity_report_v0",
            "simulation_seed": 456,
            "n_replicates": 32,
        },
    }


def test_build_beta_binomial_design_report_accepts_json_strings(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake(spec_json: str, observed_json: str, campaign_json: str) -> str:
        captured["spec"] = json.loads(spec_json)
        captured["observed"] = json.loads(observed_json)
        captured["campaign"] = json.loads(campaign_json)
        return json.dumps(
            {
                "schema_version": "nextstat_beta_binomial_design_report_v0",
                "design_family": "beta_binomial",
            }
        )

    monkeypatch.setattr(ns_bayes_design, "_beta_binomial_design_report_json", _fake)

    out = ns_bayes_design.build_beta_binomial_design_report(
        json.dumps(_beta_design_spec()),
        json.dumps(_beta_observed()),
        json.dumps(_beta_campaign()),
    )

    assert out["schema_version"] == "nextstat_beta_binomial_design_report_v0"
    assert captured["spec"] == _beta_design_spec()
    assert captured["observed"] == _beta_observed()
    assert captured["campaign"] == _beta_campaign()


def test_build_normal_normal_design_report_accepts_json_strings(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake(spec_json: str, observed_json: str, campaign_json: str) -> str:
        captured["spec"] = json.loads(spec_json)
        captured["observed"] = json.loads(observed_json)
        captured["campaign"] = json.loads(campaign_json)
        return json.dumps(
            {
                "schema_version": "nextstat_normal_normal_design_report_v0",
                "design_family": "normal_normal",
            }
        )

    monkeypatch.setattr(ns_bayes_design, "_normal_normal_design_report_json", _fake)

    out = ns_bayes_design.build_normal_normal_design_report(
        json.dumps(_normal_design_spec()),
        json.dumps(_normal_observed()),
        json.dumps(_normal_campaign()),
    )

    assert out["schema_version"] == "nextstat_normal_normal_design_report_v0"
    assert captured["spec"] == _normal_design_spec()
    assert captured["observed"] == _normal_observed()
    assert captured["campaign"] == _normal_campaign()


def test_render_beta_binomial_design_report_accepts_json_string_without_hidden_execution(
    monkeypatch,
) -> None:
    report = _beta_report()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        ns_bayes_design, "_beta_binomial_design_report_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design, "_beta_binomial_design_analyze_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_operating_characteristics_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_posterior_predictive_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_prior_sensitivity_json",
        _raise_hidden_execution,
    )

    def _fake(report_json: str) -> str:
        captured["report"] = json.loads(report_json)
        return "# Frozen binary report\n"

    monkeypatch.setattr(ns_bayes_design, "_beta_binomial_design_report_markdown", _fake)

    out = ns_bayes_design.render_beta_binomial_design_report(json.dumps(report))

    assert out == "# Frozen binary report\n"
    assert captured["report"] == report


def test_render_normal_normal_design_report_accepts_json_string_without_hidden_execution(
    monkeypatch,
) -> None:
    report = _normal_report()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        ns_bayes_design, "_normal_normal_design_report_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design, "_normal_normal_design_analyze_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_operating_characteristics_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_posterior_predictive_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_prior_sensitivity_json",
        _raise_hidden_execution,
    )

    def _fake(report_json: str) -> str:
        captured["report"] = json.loads(report_json)
        return "# Frozen continuous report\n"

    monkeypatch.setattr(ns_bayes_design, "_normal_normal_design_report_markdown", _fake)

    out = ns_bayes_design.render_normal_normal_design_report(json.dumps(report))

    assert out == "# Frozen continuous report\n"
    assert captured["report"] == report


def test_write_beta_binomial_design_report_bundle_accepts_json_string_without_hidden_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    report = _beta_report()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        ns_bayes_design, "_beta_binomial_design_report_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design, "_beta_binomial_design_analyze_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_operating_characteristics_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_posterior_predictive_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_prior_sensitivity_json",
        _raise_hidden_execution,
    )

    def _fake(report_json: str) -> str:
        captured["report"] = json.loads(report_json)
        return "# Frozen binary report\n"

    monkeypatch.setattr(ns_bayes_design, "render_beta_binomial_design_report", _fake)

    summary = ns_bayes_design.write_beta_binomial_design_report_bundle(
        tmp_path / "bundle",
        json.dumps(report),
    )

    assert summary["schema_version"] == "nextstat_bayesian_design_report_bundle_v0"
    assert summary["design_family"] == "beta_binomial"
    assert captured["report"] == report


def test_write_normal_normal_design_report_bundle_accepts_json_string_without_hidden_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    report = _normal_report()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        ns_bayes_design, "_normal_normal_design_report_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design, "_normal_normal_design_analyze_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_operating_characteristics_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_posterior_predictive_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_prior_sensitivity_json",
        _raise_hidden_execution,
    )

    def _fake(report_json: str) -> str:
        captured["report"] = json.loads(report_json)
        return "# Frozen continuous report\n"

    monkeypatch.setattr(ns_bayes_design, "render_normal_normal_design_report", _fake)

    summary = ns_bayes_design.write_normal_normal_design_report_bundle(
        tmp_path / "bundle",
        json.dumps(report),
    )

    assert summary["schema_version"] == "nextstat_bayesian_design_report_bundle_v0"
    assert summary["design_family"] == "normal_normal"
    assert captured["report"] == report


def test_build_beta_binomial_regulatory_appendix_accepts_json_string_without_hidden_execution(
    monkeypatch,
) -> None:
    report = _beta_report()

    monkeypatch.setattr(
        ns_bayes_design, "_beta_binomial_design_report_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design, "_beta_binomial_design_analyze_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_operating_characteristics_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_posterior_predictive_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_prior_sensitivity_json",
        _raise_hidden_execution,
    )

    appendix = ns_bayes_design.build_beta_binomial_regulatory_appendix(
        json.dumps(report)
    )

    assert (
        appendix["schema_version"] == "nextstat_bayesian_design_regulatory_appendix_v0"
    )
    assert appendix["design_family"] == "beta_binomial"


def test_build_normal_normal_regulatory_appendix_accepts_json_string_without_hidden_execution(
    monkeypatch,
) -> None:
    report = _normal_report()

    monkeypatch.setattr(
        ns_bayes_design, "_normal_normal_design_report_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design, "_normal_normal_design_analyze_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_operating_characteristics_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_posterior_predictive_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_prior_sensitivity_json",
        _raise_hidden_execution,
    )

    appendix = ns_bayes_design.build_normal_normal_regulatory_appendix(
        json.dumps(report)
    )

    assert (
        appendix["schema_version"] == "nextstat_bayesian_design_regulatory_appendix_v0"
    )
    assert appendix["design_family"] == "normal_normal"


def test_render_bayesian_regulatory_appendix_markdown_accepts_json_string_without_hidden_execution(
    monkeypatch,
) -> None:
    appendix = ns_bayes_design.build_beta_binomial_regulatory_appendix(_beta_report())

    monkeypatch.setattr(
        ns_bayes_design, "_beta_binomial_design_report_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design, "_beta_binomial_design_analyze_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_operating_characteristics_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_posterior_predictive_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_prior_sensitivity_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "build_beta_binomial_regulatory_appendix",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "build_normal_normal_regulatory_appendix",
        _raise_hidden_execution,
    )

    out = ns_bayes_design.render_bayesian_regulatory_appendix_markdown(
        json.dumps(appendix)
    )

    assert out.startswith("# Bayesian Regulatory Appendix")
    assert "binary_superiority_demo" in out


def test_write_bayesian_regulatory_appendix_pdf_accepts_json_string_without_hidden_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        return

    appendix = ns_bayes_design.build_normal_normal_regulatory_appendix(_normal_report())

    monkeypatch.setattr(
        ns_bayes_design, "_normal_normal_design_report_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design, "_normal_normal_design_analyze_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_operating_characteristics_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_posterior_predictive_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_prior_sensitivity_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "build_beta_binomial_regulatory_appendix",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "build_normal_normal_regulatory_appendix",
        _raise_hidden_execution,
    )

    out_pdf = tmp_path / "appendix.pdf"
    ns_bayes_design.write_bayesian_regulatory_appendix_pdf(
        out_pdf, json.dumps(appendix)
    )

    assert out_pdf.exists()
    assert out_pdf.read_bytes().startswith(b"%PDF-")


def test_build_beta_binomial_prior_conflict_diagnostic_accepts_json_string_without_hidden_execution(
    monkeypatch,
) -> None:
    report = _beta_report()

    monkeypatch.setattr(
        ns_bayes_design, "_beta_binomial_design_report_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design, "_beta_binomial_design_analyze_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_operating_characteristics_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_posterior_predictive_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_prior_sensitivity_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design, "build_beta_binomial_design_report", _raise_hidden_execution
    )

    diagnostic = ns_bayes_design.build_beta_binomial_prior_conflict_diagnostic(
        json.dumps(report)
    )

    assert (
        diagnostic["schema_version"] == "nextstat_bayesian_prior_conflict_diagnostic_v0"
    )
    assert diagnostic["design_family"] == "beta_binomial"


def test_build_normal_normal_prior_conflict_diagnostic_accepts_json_string_without_hidden_execution(
    monkeypatch,
) -> None:
    report = _normal_report()

    monkeypatch.setattr(
        ns_bayes_design, "_normal_normal_design_report_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design, "_normal_normal_design_analyze_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_operating_characteristics_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_posterior_predictive_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_prior_sensitivity_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design, "build_normal_normal_design_report", _raise_hidden_execution
    )

    diagnostic = ns_bayes_design.build_normal_normal_prior_conflict_diagnostic(
        json.dumps(report)
    )

    assert (
        diagnostic["schema_version"] == "nextstat_bayesian_prior_conflict_diagnostic_v0"
    )
    assert diagnostic["design_family"] == "normal_normal"


def test_build_beta_binomial_historical_control_borrowing_review_accepts_json_string_without_hidden_execution(
    monkeypatch,
) -> None:
    report = _beta_report()
    policy = _beta_historical_borrowing_policy()

    monkeypatch.setattr(
        ns_bayes_design, "_beta_binomial_design_report_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design, "_beta_binomial_design_analyze_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_operating_characteristics_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_posterior_predictive_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_prior_sensitivity_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design, "build_beta_binomial_design_report", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "build_beta_binomial_prior_conflict_diagnostic",
        _raise_hidden_execution,
    )

    review = ns_bayes_design.build_beta_binomial_historical_control_borrowing_review(
        json.dumps(report),
        json.dumps(policy),
    )

    assert (
        review["schema_version"]
        == "nextstat_bayesian_historical_control_borrowing_review_v0"
    )
    assert review["design_family"] == "beta_binomial"


def test_build_normal_normal_historical_control_borrowing_review_accepts_json_string_without_hidden_execution(
    monkeypatch,
) -> None:
    report = _normal_report()
    policy = _normal_historical_borrowing_policy()

    monkeypatch.setattr(
        ns_bayes_design, "_normal_normal_design_report_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design, "_normal_normal_design_analyze_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_operating_characteristics_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_posterior_predictive_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_prior_sensitivity_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design, "build_normal_normal_design_report", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "build_normal_normal_prior_conflict_diagnostic",
        _raise_hidden_execution,
    )

    review = ns_bayes_design.build_normal_normal_historical_control_borrowing_review(
        json.dumps(report),
        json.dumps(policy),
    )

    assert (
        review["schema_version"]
        == "nextstat_bayesian_historical_control_borrowing_review_v0"
    )
    assert review["design_family"] == "normal_normal"


def test_build_beta_binomial_robust_mixture_prior_review_accepts_json_string_without_hidden_execution(
    monkeypatch,
) -> None:
    report = _beta_report()
    policy = _beta_robust_mixture_policy()

    monkeypatch.setattr(
        ns_bayes_design, "_beta_binomial_design_report_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design, "_beta_binomial_design_analyze_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_operating_characteristics_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_posterior_predictive_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_beta_binomial_design_prior_sensitivity_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design, "build_beta_binomial_design_report", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "build_beta_binomial_prior_conflict_diagnostic",
        _raise_hidden_execution,
    )

    review = ns_bayes_design.build_beta_binomial_robust_mixture_prior_review(
        json.dumps(report),
        json.dumps(policy),
    )

    assert (
        review["schema_version"] == "nextstat_bayesian_robust_mixture_prior_review_v0"
    )
    assert review["design_family"] == "beta_binomial"


def test_build_normal_normal_robust_mixture_prior_review_accepts_json_string_without_hidden_execution(
    monkeypatch,
) -> None:
    report = _normal_report()
    policy = _normal_robust_mixture_policy()

    monkeypatch.setattr(
        ns_bayes_design, "_normal_normal_design_report_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design, "_normal_normal_design_analyze_json", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_operating_characteristics_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_posterior_predictive_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "_normal_normal_design_prior_sensitivity_json",
        _raise_hidden_execution,
    )
    monkeypatch.setattr(
        ns_bayes_design, "build_normal_normal_design_report", _raise_hidden_execution
    )
    monkeypatch.setattr(
        ns_bayes_design,
        "build_normal_normal_prior_conflict_diagnostic",
        _raise_hidden_execution,
    )

    review = ns_bayes_design.build_normal_normal_robust_mixture_prior_review(
        json.dumps(report),
        json.dumps(policy),
    )

    assert (
        review["schema_version"] == "nextstat_bayesian_robust_mixture_prior_review_v0"
    )
    assert review["design_family"] == "normal_normal"
