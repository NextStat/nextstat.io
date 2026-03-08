import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_beta_binomial_design_schemas_and_examples_smoke():
    cases = [
        (
            "docs/schemas/pharma/beta_binomial_design_v0.schema.json",
            "docs/specs/pharma/beta_binomial_design_v0.example.json",
            "nextstat_beta_binomial_design_v0",
        ),
        (
            "docs/schemas/pharma/beta_binomial_design_analysis_v0.schema.json",
            "docs/specs/pharma/beta_binomial_design_analysis_v0.example.json",
            "nextstat_beta_binomial_design_analysis_v0",
        ),
        (
            "docs/schemas/pharma/beta_binomial_operating_characteristics_v0.schema.json",
            "docs/specs/pharma/beta_binomial_operating_characteristics_v0.example.json",
            "nextstat_beta_binomial_operating_characteristics_v0",
        ),
        (
            "docs/schemas/pharma/normal_normal_design_v0.schema.json",
            "docs/specs/pharma/normal_normal_design_v0.example.json",
            "nextstat_normal_normal_design_v0",
        ),
        (
            "docs/schemas/pharma/normal_normal_design_analysis_v0.schema.json",
            "docs/specs/pharma/normal_normal_design_analysis_v0.example.json",
            "nextstat_normal_normal_design_analysis_v0",
        ),
        (
            "docs/schemas/pharma/normal_normal_operating_characteristics_v0.schema.json",
            "docs/specs/pharma/normal_normal_operating_characteristics_v0.example.json",
            "nextstat_normal_normal_operating_characteristics_v0",
        ),
        (
            "docs/schemas/pharma/beta_binomial_posterior_predictive_v0.schema.json",
            "docs/specs/pharma/beta_binomial_posterior_predictive_v0.example.json",
            "nextstat_beta_binomial_posterior_predictive_v0",
        ),
        (
            "docs/schemas/pharma/normal_normal_posterior_predictive_v0.schema.json",
            "docs/specs/pharma/normal_normal_posterior_predictive_v0.example.json",
            "nextstat_normal_normal_posterior_predictive_v0",
        ),
        (
            "docs/schemas/pharma/beta_binomial_prior_sensitivity_campaign_v0.schema.json",
            "docs/specs/pharma/beta_binomial_prior_sensitivity_campaign_v0.example.json",
            "nextstat_beta_binomial_prior_sensitivity_campaign_v0",
        ),
        (
            "docs/schemas/pharma/beta_binomial_prior_sensitivity_report_v0.schema.json",
            "docs/specs/pharma/beta_binomial_prior_sensitivity_report_v0.example.json",
            "nextstat_beta_binomial_prior_sensitivity_report_v0",
        ),
        (
            "docs/schemas/pharma/beta_binomial_design_report_v0.schema.json",
            "docs/specs/pharma/beta_binomial_design_report_v0.example.json",
            "nextstat_beta_binomial_design_report_v0",
        ),
        (
            "docs/schemas/pharma/normal_normal_prior_sensitivity_campaign_v0.schema.json",
            "docs/specs/pharma/normal_normal_prior_sensitivity_campaign_v0.example.json",
            "nextstat_normal_normal_prior_sensitivity_campaign_v0",
        ),
        (
            "docs/schemas/pharma/normal_normal_prior_sensitivity_report_v0.schema.json",
            "docs/specs/pharma/normal_normal_prior_sensitivity_report_v0.example.json",
            "nextstat_normal_normal_prior_sensitivity_report_v0",
        ),
        (
            "docs/schemas/pharma/normal_normal_design_report_v0.schema.json",
            "docs/specs/pharma/normal_normal_design_report_v0.example.json",
            "nextstat_normal_normal_design_report_v0",
        ),
        (
            "docs/schemas/pharma/bayesian_design_report_bundle_v0.schema.json",
            "docs/specs/pharma/beta_binomial_design_report_bundle_v0.example.json",
            "nextstat_bayesian_design_report_bundle_v0",
        ),
        (
            "docs/schemas/pharma/bayesian_design_report_bundle_v0.schema.json",
            "docs/specs/pharma/normal_normal_design_report_bundle_v0.example.json",
            "nextstat_bayesian_design_report_bundle_v0",
        ),
        (
            "docs/schemas/pharma/bayesian_design_regulatory_appendix_v0.schema.json",
            "docs/specs/pharma/beta_binomial_regulatory_appendix_v0.example.json",
            "nextstat_bayesian_design_regulatory_appendix_v0",
        ),
        (
            "docs/schemas/pharma/bayesian_design_regulatory_appendix_v0.schema.json",
            "docs/specs/pharma/normal_normal_regulatory_appendix_v0.example.json",
            "nextstat_bayesian_design_regulatory_appendix_v0",
        ),
        (
            "docs/schemas/pharma/bayesian_prior_conflict_diagnostic_v0.schema.json",
            "docs/specs/pharma/beta_binomial_prior_conflict_diagnostic_v0.example.json",
            "nextstat_bayesian_prior_conflict_diagnostic_v0",
        ),
        (
            "docs/schemas/pharma/bayesian_prior_conflict_diagnostic_v0.schema.json",
            "docs/specs/pharma/normal_normal_prior_conflict_diagnostic_v0.example.json",
            "nextstat_bayesian_prior_conflict_diagnostic_v0",
        ),
        (
            "docs/schemas/pharma/bayesian_historical_control_borrowing_policy_v0.schema.json",
            "docs/specs/pharma/bayesian_historical_control_borrowing_policy_beta_binomial_v0.example.json",
            "nextstat_bayesian_historical_control_borrowing_policy_v0",
        ),
        (
            "docs/schemas/pharma/bayesian_historical_control_borrowing_policy_v0.schema.json",
            "docs/specs/pharma/bayesian_historical_control_borrowing_policy_normal_normal_v0.example.json",
            "nextstat_bayesian_historical_control_borrowing_policy_v0",
        ),
        (
            "docs/schemas/pharma/bayesian_historical_control_borrowing_review_v0.schema.json",
            "docs/specs/pharma/bayesian_historical_control_borrowing_review_beta_binomial_v0.example.json",
            "nextstat_bayesian_historical_control_borrowing_review_v0",
        ),
        (
            "docs/schemas/pharma/bayesian_historical_control_borrowing_operating_characteristics_v0.schema.json",
            "docs/specs/pharma/bayesian_historical_control_borrowing_operating_characteristics_beta_binomial_v0.example.json",
            "nextstat_bayesian_historical_control_borrowing_operating_characteristics_v0",
        ),
        (
            "docs/schemas/pharma/bayesian_historical_control_borrowing_review_v0.schema.json",
            "docs/specs/pharma/bayesian_historical_control_borrowing_review_normal_normal_v0.example.json",
            "nextstat_bayesian_historical_control_borrowing_review_v0",
        ),
        (
            "docs/schemas/pharma/bayesian_historical_control_borrowing_operating_characteristics_v0.schema.json",
            "docs/specs/pharma/bayesian_historical_control_borrowing_operating_characteristics_normal_normal_v0.example.json",
            "nextstat_bayesian_historical_control_borrowing_operating_characteristics_v0",
        ),
        (
            "docs/schemas/pharma/bayesian_robust_mixture_prior_policy_v0.schema.json",
            "docs/specs/pharma/bayesian_robust_mixture_prior_policy_beta_binomial_v0.example.json",
            "nextstat_bayesian_robust_mixture_prior_policy_v0",
        ),
        (
            "docs/schemas/pharma/bayesian_robust_mixture_prior_policy_v0.schema.json",
            "docs/specs/pharma/bayesian_robust_mixture_prior_policy_normal_normal_v0.example.json",
            "nextstat_bayesian_robust_mixture_prior_policy_v0",
        ),
        (
            "docs/schemas/pharma/bayesian_robust_mixture_prior_review_v0.schema.json",
            "docs/specs/pharma/bayesian_robust_mixture_prior_review_beta_binomial_v0.example.json",
            "nextstat_bayesian_robust_mixture_prior_review_v0",
        ),
        (
            "docs/schemas/pharma/bayesian_robust_mixture_prior_operating_characteristics_v0.schema.json",
            "docs/specs/pharma/bayesian_robust_mixture_prior_operating_characteristics_beta_binomial_v0.example.json",
            "nextstat_bayesian_robust_mixture_prior_operating_characteristics_v0",
        ),
        (
            "docs/schemas/pharma/bayesian_robust_mixture_prior_review_v0.schema.json",
            "docs/specs/pharma/bayesian_robust_mixture_prior_review_normal_normal_v0.example.json",
            "nextstat_bayesian_robust_mixture_prior_review_v0",
        ),
        (
            "docs/schemas/pharma/bayesian_robust_mixture_prior_operating_characteristics_v0.schema.json",
            "docs/specs/pharma/bayesian_robust_mixture_prior_operating_characteristics_normal_normal_v0.example.json",
            "nextstat_bayesian_robust_mixture_prior_operating_characteristics_v0",
        ),
    ]

    try:
        import jsonschema  # type: ignore
    except Exception:
        jsonschema = None

    for schema_rel, example_rel, expected_version in cases:
        schema_path = _repo_root() / schema_rel
        assert schema_path.exists(), f"missing schema: {schema_path}"
        schema = json.loads(schema_path.read_text())
        assert schema.get("$schema"), "schema must declare $schema"
        assert schema.get("$id"), "schema must declare $id"
        assert schema.get("type") == "object"

        example_path = _repo_root() / example_rel
        assert example_path.exists(), f"missing example: {example_path}"
        example = json.loads(example_path.read_text())
        assert example.get("schema_version") == expected_version

        if jsonschema is not None:
            jsonschema.validate(instance=example, schema=schema)
