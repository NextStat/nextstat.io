# Bayesian Trial Design Report

- Family: beta_binomial
- Design ID: binary_superiority_demo
- Stability: research-grade
- Software: nextstat 0.9.9
- Simulation: 32 replicates, seed 123

## Provenance

- Report schema: nextstat_beta_binomial_design_report_v0
- Design schema: nextstat_beta_binomial_design_v0
- Analysis schema: nextstat_beta_binomial_design_analysis_v0
- Operating characteristics schema: nextstat_beta_binomial_operating_characteristics_v0
- Posterior predictive schema: nextstat_beta_binomial_posterior_predictive_v0
- Prior sensitivity campaign schema: nextstat_beta_binomial_prior_sensitivity_campaign_v0
- Prior sensitivity report schema: nextstat_beta_binomial_prior_sensitivity_report_v0

## Design Spec

### Priors

| Arm | Prior | Prior mean |
| --- | --- | ---: |
| Control | Beta(1.000, 1.000) | 0.500 |
| Treatment | Beta(1.000, 1.000) | 0.500 |

### Looks

| Look | N control | N treatment |
| --- | ---: | ---: |
| interim | 20 | 20 |
| final | 40 | 40 |

### Decision Criteria

| Rule | Posterior threshold | Margin | Credible interval level |
| --- | ---: | ---: | ---: |
| Success | 0.950 | 0.000 | 0.950 |
| Futility | 0.200 | 0.000 | 0.950 |

### Simulation Scenarios

| Scenario | Control rate | Treatment rate |
| --- | ---: | ---: |
| null | 0.400 | 0.400 |
| alt | 0.400 | 0.600 |

## Current Analysis

- Look: interim (N control = 20, N treatment = 20)
- Observed successes: control = 8, treatment = 9
- Recommended action: continue
- Posterior mean treatment effect: 0.045
- Posterior Pr(effect > margin): 0.622

## Operating Characteristics

| Scenario | Success | Futility | No decision | Expected total N |
| --- | ---: | ---: | ---: | ---: |
| null | 0.062 | 0.281 | 0.656 | 71.250 |
| alt | 0.656 | 0.000 | 0.344 | 66.250 |

## Posterior Predictive Forecast

| Eventual success | Eventual futility | Eventual no decision | Expected total N | Expected remaining N |
| ---: | ---: | ---: | ---: | ---: |
| 0.125 | 0.062 | 0.812 | 80.000 | 40.000 |

### Future Looks

| Future look | Stop | Success | Futility |
| --- | ---: | ---: | ---: |
| final | 0.188 | 0.125 | 0.062 |

## Prior Sensitivity

| Variant | Baseline | Action | Posterior Pr(effect > margin) | Delta vs baseline | Eventual success | Delta vs baseline | Expected total N | Delta vs baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | yes | continue | 0.622 | +0.000 | 0.125 | +0.000 | 80.000 | +0.000 |
| skeptical | no | continue | 0.318 | -0.304 | 0.000 | -0.125 | 80.000 | +0.000 |
| enthusiastic | no | continue | 0.900 | +0.278 | 0.531 | +0.406 | 80.000 | +0.000 |
