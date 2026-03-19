---
title: "Benchmark Suite: Econometrics (Panel / Causal Inference)"
description: "Econometrics benchmark suite for NextStat: Panel FE fit scaling, DiD TWFE + event study wall-time, IV/2SLS two-stage cost, and AIPW doubly-robust estimator performance with cluster-count scaling."
status: stable
last_updated: 2026-03-19
keywords:
  - panel fixed effects benchmark
  - difference-in-differences performance
  - IV 2SLS benchmark
  - AIPW doubly robust
  - causal inference software
  - econometrics performance
  - cluster robust SE
  - NextStat econometrics
---

# Econometrics Benchmark Suite (Panel / Causal Inference)

This suite benchmarks NextStat's econometrics and causal inference infrastructure:

- Panel fixed effects (1-way / 2-way cluster SE)
- Difference-in-Differences (TWFE + event study + staggered-adoption DiD)
- Wild cluster bootstrap inference for DiD TWFE (Webb 6-point)
- Instrumental Variables (2SLS; HC1 + HAC/Newey-West)
- AIPW doubly-robust estimator

This page is a **runbook + methodology**. Results are published as benchmark snapshots (see [Public Benchmarks](/docs/public-benchmarks)).

## Named competitor baselines

| Case | NextStat function | Competitor | Library version |
| --- | --- | --- | --- |
| Panel FE | `panel_fe_fit()` | statsmodels OLS (cluster) | statsmodels ≥ 0.14 |
| DiD TWFE | `did_twfe_fit()` | statsmodels OLS (cluster) | statsmodels ≥ 0.14 |
| DiD Wild Bootstrap | `did_twfe_wild_cluster_bootstrap()` | pyfixest `wildboottest()` | pyfixest ≥ 0.40 |
| DiD Staggered | `staggered_did_fit()` | pyfixest `lpdid()` | pyfixest ≥ 0.40 |
| Event Study TWFE | `event_study_twfe_fit()` | statsmodels OLS (cluster) | statsmodels ≥ 0.14 |
| IV 2SLS (HC1) | `iv_2sls_fit(cov="hc1")` | linearmodels `IV2SLS` | linearmodels ≥ 6.1 |
| IV 2SLS (HAC) | `iv_2sls_fit(cov="hac")` | linearmodels `IV2SLS` (Bartlett) | linearmodels ≥ 6.1 |
| AIPW | `aipw_fit()` | (no external parity yet) | — |

## Estimator boundary: staggered-adoption DiD

NextStat's `staggered_did_fit()` uses a **group-time ATT estimator** with
not-yet-treated controls and within-group demeaning. This is algorithmically
different from pyfixest's `lpdid()`, which implements the
Callaway-Sant'Anna (2021) doubly-robust framework with propensity score
reweighting.

**Expected coefficient differences**: up to ~20% relative on the aggregate
ATT, because the two estimators weight cohort-time cells differently and
use different control group construction. This is an algorithm choice, not
a bug.

**Where parity holds (machine-precision)**:
- Panel FE, DiD TWFE, event study TWFE — coefficient and SE differences
  < 1e-14 relative to statsmodels with identical demeaning
- IV 2SLS (HC1) — coefficient parity < 1e-10 relative to linearmodels

**Where parity is approximate**:
- DiD staggered (~20% coef diff vs pyfixest lpdid) — different algorithm
- IV HAC (Newey-West) — kernel bandwidth selection differences
- Wild cluster bootstrap — stochastic by nature, p-value agreement checked

## What is compared

- **NextStat (Rust core)** vs **statsmodels** (OLS/cluster robust on transformed designs)
- **NextStat** vs **linearmodels** (`PanelOLS`, `IV2SLS`, kernel/HAC covariance)
- **NextStat** vs **pyfixest** (`feols`, `lpdid`, `wildboottest`)
- Optional external parity: **R fixest** / **R did** for extended staggered-adoption sensitivity

## What is measured

### 1) Panel FE fit wall-time

Measures:

- wall-time scaling with entity count (100, 1K, 10K, 100K entities)
- wall-time scaling with cluster count for cluster-robust SE
- coefficient and SE parity vs reference (statsmodels or R fixest)

### 2) DiD TWFE + event study

Measures:

- TWFE estimator wall-time at varying treatment/control group sizes
- event study dynamic effects computation cost
- pre-trend test overhead

Additional case:

- staggered-adoption DiD (group-time ATT with not-yet-treated controls)
- wild cluster bootstrap on ATT (Webb 6-point weights) for few-cluster robustness

### 3) IV/2SLS two-stage cost

Measures:

- first-stage fit wall-time
- second-stage fit wall-time
- total IV estimator wall-time vs OLS baseline
- Hausman test overhead
- HAC/Newey-West covariance overhead vs HC1

### 4) AIPW doubly-robust estimator

Measures:

- propensity model fit cost
- outcome model fit cost
- ATE/ATT estimation wall-time
- comparison vs naive OLS treatment effect

## Scaling axes (what we vary)

- Number of entities / observations (100 to 100K)
- Number of clusters (10, 100, 1K)
- Number of covariates (5, 20, 100)
- Treatment group fraction (10%, 50%)

## How to run locally (current)

Python API:

```python
import nextstat

# Panel FE
result = nextstat.panel_fe(y, X, entity_id, cluster_id)

# DiD
result = nextstat.did_twfe(y, treated, post, X)

# IV/2SLS
result = nextstat.iv_2sls(y, X_exog, X_endog, Z)

# AIPW
result = nextstat.aipw(y, treatment, X)
```

CLI:

```bash
nextstat econometrics panel-fe --input panel_data.csv --entity entity_id --cluster cluster_id
nextstat econometrics did --input did_data.csv --treated treated --post post
```

## Related reading

- [Public Benchmarks Specification](/docs/public-benchmarks) — canonical spec.
- [Validation Report Artifacts](/docs/validation-report) — validation pack for published snapshots.
