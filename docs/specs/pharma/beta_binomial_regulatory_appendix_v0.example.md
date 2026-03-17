# Bayesian Regulatory Appendix

- Appendix ID: `binary_superiority_demo_regulatory_appendix_v0`
- Design family: `beta_binomial`
- Design ID: `binary_superiority_demo`
- Source report schema version: `nextstat_beta_binomial_design_report_v0`
- Generated from frozen report: `True`
- Stability: `research-grade`

## Design Summary

- Design ID: `binary_superiority_demo`
- Design family: `beta_binomial`
- Endpoint summary: binary endpoint with exact beta-binomial conjugate updating
- Current look ID: `interim`
- Total planned sample size: `80`

### Planned Looks

| Look ID | N Control | N Treatment |
| --- | ---: | ---: |
| `interim` | `20` | `20` |
| `final` | `40` | `40` |

## Prior Specification

### Control Prior

```json
{
  "alpha": 1.0,
  "beta": 1.0
}
```

### Treatment Prior

```json
{
  "alpha": 1.0,
  "beta": 1.0
}
```

## Decision Rules

```json
{
  "futility": {
    "posterior_probability_threshold": 0.2,
    "treatment_effect_margin": 0.0
  },
  "success": {
    "posterior_probability_threshold": 0.95,
    "treatment_effect_margin": 0.0
  }
}
```

## Current Analysis

- Look ID: `interim`
- Recommended action: `continue`
- Posterior probability > margin: `0.6221311262168687`
- Treatment effect margin: `0.0`

### Posterior Effect Summary

```json
{
  "margin": 0.0,
  "posterior_mean": 0.045454545454545414,
  "posterior_probability_gt_margin": 0.6221311262168687
}
```

## Operating Characteristics

- Replicates: `32`
- Seed: `123`

### Scenario Summaries

```json
[
  {
    "expected_total_sample_size": 71.25,
    "futility_rate": 0.28125,
    "no_decision_rate": 0.65625,
    "scenario_id": "null",
    "success_rate": 0.0625
  },
  {
    "expected_total_sample_size": 66.25,
    "futility_rate": 0.0,
    "no_decision_rate": 0.34375,
    "scenario_id": "alt",
    "success_rate": 0.65625
  }
]
```

## Posterior Predictive

- Replicates: `32`
- Seed: `123`
- Eventual success probability: `0.125`
- Eventual futility probability: `0.0625`
- Eventual no-decision probability: `0.8125`
- Expected total sample size: `80.0`

### Future Look Summaries

```json
[
  {
    "conditional_futility_probability": 0.0625,
    "conditional_stop_probability": 0.1875,
    "conditional_success_probability": 0.125,
    "look_id": "final"
  }
]
```

## Prior Sensitivity

- Replicates: `32`
- Seed: `123`
- Baseline variant ID: `baseline`

### Variant Summaries

```json
[
  {
    "eventual_futility_probability": 0.0625,
    "eventual_no_decision_probability": 0.8125,
    "eventual_success_probability": 0.125,
    "eventual_success_probability_delta_vs_baseline": 0.0,
    "expected_total_sample_size": 80.0,
    "posterior_probability_delta_vs_baseline": 0.0,
    "posterior_probability_gt_margin": 0.6221311262168687,
    "recommended_action": "continue",
    "variant_id": "baseline"
  },
  {
    "eventual_futility_probability": 0.28125,
    "eventual_no_decision_probability": 0.71875,
    "eventual_success_probability": 0.0,
    "eventual_success_probability_delta_vs_baseline": -0.125,
    "expected_total_sample_size": 80.0,
    "posterior_probability_delta_vs_baseline": -0.30386295637778804,
    "posterior_probability_gt_margin": 0.3182681698390807,
    "recommended_action": "continue",
    "variant_id": "skeptical"
  },
  {
    "eventual_futility_probability": 0.0,
    "eventual_no_decision_probability": 0.46875,
    "eventual_success_probability": 0.53125,
    "eventual_success_probability_delta_vs_baseline": 0.40625,
    "expected_total_sample_size": 80.0,
    "posterior_probability_delta_vs_baseline": 0.277540332930438,
    "posterior_probability_gt_margin": 0.8996714591473067,
    "recommended_action": "continue",
    "variant_id": "enthusiastic"
  }
]
```

## Provenance

```json
{
  "analysis_schema_version": "nextstat_beta_binomial_design_analysis_v0",
  "design_schema_version": "nextstat_beta_binomial_design_v0",
  "n_replicates": 32,
  "operating_characteristics_schema_version": "nextstat_beta_binomial_operating_characteristics_v0",
  "posterior_predictive_schema_version": "nextstat_beta_binomial_posterior_predictive_v0",
  "prior_sensitivity_campaign_schema_version": "nextstat_beta_binomial_prior_sensitivity_campaign_v0",
  "prior_sensitivity_report_schema_version": "nextstat_beta_binomial_prior_sensitivity_report_v0",
  "simulation_seed": 123,
  "software_name": "nextstat",
  "software_version": "0.10.1"
}
```
