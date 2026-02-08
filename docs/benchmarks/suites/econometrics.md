---
title: "Benchmark Suite: Econometrics (Panel / Causal Inference)"
description: "Econometrics benchmark suite for NextStat: Panel FE fit scaling, DiD TWFE + event study wall-time, IV/2SLS two-stage cost, and AIPW doubly-robust estimator performance with cluster-count scaling."
status: draft
last_updated: 2026-02-08
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

- Panel fixed effects (1-way cluster SE)
- Difference-in-Differences (TWFE + event study)
- Instrumental Variables (2SLS)
- AIPW doubly-robust estimator

This page is a **runbook + methodology**. Results are published as benchmark snapshots (see [Public Benchmarks](/docs/benchmarks/public-benchmarks)).

## What is compared

Planned comparisons include:

- **NextStat (Rust core)** vs **statsmodels** (`PanelOLS`, `IV2SLS`)
- Optional: **linearmodels** (Python) for panel FE and IV
- Optional: **R fixest** for high-dimensional FE benchmarks

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

### 3) IV/2SLS two-stage cost

Measures:

- first-stage fit wall-time
- second-stage fit wall-time
- total IV estimator wall-time vs OLS baseline
- Hausman test overhead

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

- [Public Benchmarks Specification](/docs/benchmarks/public-benchmarks) — canonical spec.
- [Validation Report Artifacts](/docs/references/validation-report) — validation pack for published snapshots.
