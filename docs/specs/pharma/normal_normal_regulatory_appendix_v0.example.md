# Bayesian Regulatory Appendix

- Appendix ID: `continuous_superiority_demo_regulatory_appendix_v0`
- Design family: `normal_normal`
- Design ID: `continuous_superiority_demo`
- Source report schema version: `nextstat_normal_normal_design_report_v0`
- Generated from frozen report: `True`
- Stability: `research-grade`

## Design Summary

- Design ID: `continuous_superiority_demo`
- Design family: `normal_normal`
- Endpoint summary: continuous endpoint with exact normal-normal conjugate updating
- Current look ID: `interim`
- Total planned sample size: `40`

### Planned Looks

| Look ID | N Control | N Treatment |
| --- | ---: | ---: |
| `interim` | `10` | `10` |
| `final` | `20` | `20` |

## Prior Specification

### Control Prior

```json
{
  "mean": 0.0,
  "sd": 10.0
}
```

### Treatment Prior

```json
{
  "mean": 0.0,
  "sd": 10.0
}
```

### Likelihood

```json
{
  "known_sd_control": 1.0,
  "known_sd_treatment": 1.0
}
```

## Decision Rules

```json
{
  "futility": {
    "posterior_probability_threshold": 0.1,
    "treatment_effect_margin": 0.0
  },
  "success": {
    "posterior_probability_threshold": 0.975,
    "treatment_effect_margin": 0.0
  }
}
```

## Current Analysis

- Look ID: `interim`
- Recommended action: `continue`
- Posterior probability > margin: `0.6725589163359341`
- Treatment effect margin: `0.0`

### Posterior Effect Summary

```json
{
  "ci_lower": -0.6762844079283722,
  "ci_upper": 1.0758848075287717,
  "margin": 0.0,
  "posterior_mean": 0.1998001998001998,
  "posterior_probability_gt_margin": 0.6725589163359341,
  "posterior_sd": 0.4469901562676742
}
```

## Operating Characteristics

- Replicates: `32`
- Seed: `456`

### Scenario Summaries

```json
[
  {
    "expected_total_sample_size": 36.875,
    "futility_rate": 0.1875,
    "no_decision_rate": 0.78125,
    "scenario_id": "null",
    "success_rate": 0.03125
  },
  {
    "expected_total_sample_size": 33.125,
    "futility_rate": 0.0,
    "no_decision_rate": 0.375,
    "scenario_id": "alt",
    "success_rate": 0.625
  }
]
```

## Posterior Predictive

- Replicates: `32`
- Seed: `456`
- Eventual success probability: `0.125`
- Eventual futility probability: `0.0`
- Eventual no-decision probability: `0.875`
- Expected total sample size: `40.0`

### Future Look Summaries

```json
[
  {
    "conditional_futility_probability": 0.0,
    "conditional_stop_probability": 0.125,
    "conditional_success_probability": 0.125,
    "look_id": "final"
  }
]
```

## Prior Sensitivity

- Replicates: `32`
- Seed: `456`
- Baseline variant ID: `baseline`

### Variant Summaries

```json
[
  {
    "eventual_futility_probability": 0.0,
    "eventual_no_decision_probability": 0.875,
    "eventual_success_probability": 0.125,
    "eventual_success_probability_delta_vs_baseline": 0.0,
    "expected_total_sample_size": 40.0,
    "posterior_probability_delta_vs_baseline": 0.0,
    "posterior_probability_gt_margin": 0.6725589163359341,
    "recommended_action": "continue",
    "variant_id": "baseline"
  },
  {
    "eventual_futility_probability": 1.0,
    "eventual_no_decision_probability": 0.0,
    "eventual_success_probability": 0.0,
    "eventual_success_probability_delta_vs_baseline": -0.125,
    "expected_total_sample_size": 20.0,
    "posterior_probability_delta_vs_baseline": -0.6515023156200955,
    "posterior_probability_gt_margin": 0.02105660071583859,
    "recommended_action": "stop_for_futility",
    "variant_id": "skeptical"
  },
  {
    "eventual_futility_probability": 0.0,
    "eventual_no_decision_probability": 0.25,
    "eventual_success_probability": 0.75,
    "eventual_success_probability_delta_vs_baseline": 0.625,
    "expected_total_sample_size": 40.0,
    "posterior_probability_delta_vs_baseline": 0.30204575781838905,
    "posterior_probability_gt_margin": 0.9746046741543232,
    "recommended_action": "continue",
    "variant_id": "enthusiastic"
  }
]
```

## Provenance

```json
{
  "analysis_schema_version": "nextstat_normal_normal_design_analysis_v0",
  "design_schema_version": "nextstat_normal_normal_design_v0",
  "n_replicates": 32,
  "operating_characteristics_schema_version": "nextstat_normal_normal_operating_characteristics_v0",
  "posterior_predictive_schema_version": "nextstat_normal_normal_posterior_predictive_v0",
  "prior_sensitivity_campaign_schema_version": "nextstat_normal_normal_prior_sensitivity_campaign_v0",
  "prior_sensitivity_report_schema_version": "nextstat_normal_normal_prior_sensitivity_report_v0",
  "simulation_seed": 456,
  "software_name": "nextstat",
  "software_version": "0.9.9"
}
```
