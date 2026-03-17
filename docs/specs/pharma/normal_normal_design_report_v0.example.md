# Bayesian Trial Design Report

- Family: normal_normal
- Design ID: continuous_superiority_demo
- Stability: research-grade
- Software: nextstat 0.10.1
- Simulation: 32 replicates, seed 456

## Provenance

- Report schema: nextstat_normal_normal_design_report_v0
- Design schema: nextstat_normal_normal_design_v0
- Analysis schema: nextstat_normal_normal_design_analysis_v0
- Operating characteristics schema: nextstat_normal_normal_operating_characteristics_v0
- Posterior predictive schema: nextstat_normal_normal_posterior_predictive_v0
- Prior sensitivity campaign schema: nextstat_normal_normal_prior_sensitivity_campaign_v0
- Prior sensitivity report schema: nextstat_normal_normal_prior_sensitivity_report_v0

## Design Spec

### Priors

| Arm | Prior |
| --- | --- |
| Control | Normal(mean=0.000, sd=10.000) |
| Treatment | Normal(mean=0.000, sd=10.000) |

### Likelihood

| Control known sd | Treatment known sd | Credible interval level |
| ---: | ---: | ---: |
| 1.000 | 1.000 | 0.950 |

### Looks

| Look | N control | N treatment |
| --- | ---: | ---: |
| interim | 10 | 10 |
| final | 20 | 20 |

### Decision Criteria

| Rule | Posterior threshold | Margin |
| --- | ---: | ---: |
| Success | 0.975 | 0.000 |
| Futility | 0.100 | 0.000 |

### Simulation Scenarios

| Scenario | Control mean | Treatment mean |
| --- | ---: | ---: |
| null | 0.000 | 0.000 |
| alt | 0.000 | 0.750 |

## Current Analysis

- Look: interim (N control = 10, N treatment = 10)
- Observed means: control = 0.100, treatment = 0.300
- Recommended action: continue
- Posterior mean treatment effect: 0.200
- Posterior Pr(effect > margin): 0.673

## Operating Characteristics

| Scenario | Success | Futility | No decision | Expected total N |
| --- | ---: | ---: | ---: | ---: |
| null | 0.031 | 0.188 | 0.781 | 36.875 |
| alt | 0.625 | 0.000 | 0.375 | 33.125 |

## Posterior Predictive Forecast

| Eventual success | Eventual futility | Eventual no decision | Expected total N | Expected remaining N |
| ---: | ---: | ---: | ---: | ---: |
| 0.125 | 0.000 | 0.875 | 40.000 | 20.000 |

### Future Looks

| Future look | Stop | Success | Futility |
| --- | ---: | ---: | ---: |
| final | 0.125 | 0.125 | 0.000 |

## Prior Sensitivity

| Variant | Baseline | Action | Posterior Pr(effect > margin) | Delta vs baseline | Eventual success | Delta vs baseline | Expected total N | Delta vs baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | yes | continue | 0.673 | +0.000 | 0.125 | +0.000 | 40.000 | +0.000 |
| skeptical | no | stop_for_futility | 0.021 | -0.652 | 0.000 | -0.125 | 20.000 | -20.000 |
| enthusiastic | no | continue | 0.975 | +0.302 | 0.750 | +0.625 | 40.000 | +0.000 |
