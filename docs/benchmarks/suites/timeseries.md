---
title: "Benchmark Suite: Time Series (Kalman / State Space / GARCH)"
description: "Time series benchmark suite for NextStat: Kalman filter/smoother throughput, EM convergence cost, GARCH parameter estimation, and forecasting latency with named competitor baselines (pykalman, statsmodels, arch)."
status: stable
last_updated: 2026-03-19
keywords:
  - Kalman filter benchmark
  - state space model performance
  - EM algorithm convergence
  - GARCH benchmark
  - time series forecasting latency
  - Kalman smoother throughput
  - statsmodels comparison
  - pykalman comparison
  - arch comparison
  - scientific time series
  - NextStat time series
---

# Time Series Benchmark Suite (Kalman / State Space / GARCH)

This suite benchmarks NextStat's time series and state-space model infrastructure:

- Kalman filter forward pass throughput
- Kalman smoother (RTS) throughput
- EM convergence cost (iterations × NLL evaluations)
- Forecasting latency per horizon step
- GARCH(1,1) parameter estimation

This page is a **runbook + methodology**. Results are published as benchmark snapshots (see [Public Benchmarks](/docs/public-benchmarks)).

## Current promoted public proof subset

The current committed public proof subset is anchored by:

- `benchmarks/nextstat-public-benchmarks/manifests/snapshots/timeseries-publisher-20260319T195703Z/timeseries/timeseries_suite.json`

That snapshot contains four committed competitor-backed cases:

- `kalman_local_level_500`
- `kalman_local_level_5000`
- `garch11_1000`
- `garch11_5000`

with `4/4 ok`, `0 warn`, `0 failed`, and `parity_status="ok"` on every
committed case.

The current promoted subset uses `pykalman` for the Kalman local-level proof
path and `arch` for the GARCH proof path. `statsmodels` remains a documented
reference and methodology fallback for the Kalman MLE / EM boundary.

## Named competitor baselines

| Case | NextStat function | Competitor | Library version |
| --- | --- | --- | --- |
| Kalman filter (local level) | `kalman_filter()` | pykalman `KalmanFilter` | pykalman ≥ 0.9.7 |
| Kalman smoother (RTS) | `kalman_smooth()` | pykalman `KalmanFilter.smooth()` | pykalman ≥ 0.9.7 |
| Kalman EM | `kalman_em()` | statsmodels `UnobservedComponents` | statsmodels ≥ 0.14 |
| GARCH(1,1) | `garch_fit()` | arch `arch_model()` | arch ≥ 7.0 |

## What is compared

- **NextStat (Rust core)** vs **pykalman** (Kalman filter/smoother/EM parity)
- **NextStat** vs **statsmodels** (`UnobservedComponents`, `KalmanFilter` — MLE fallback)
- **NextStat** vs **arch** (GARCH(1,1) parameter estimation)

## What is measured

### 1) Kalman filter throughput (states/sec)

Measures the cost of a single forward pass through the Kalman filter at varying state dimensions and observation counts.

Correctness gating:

- Verify filtered state estimates and log-likelihood vs reference (statsmodels or analytic) within tolerance.

### 2) Kalman smoother throughput (states/sec)

Measures the cost of the RTS (Rauch-Tung-Striebel) backward pass.

### 3) EM convergence cost

Measures:

- wall-time per EM iteration
- number of iterations to convergence under declared tolerance
- total NLL evaluations (filter passes)

### 4) Forecasting latency

Measures:

- per-step forecast cost at varying horizon lengths
- confidence interval computation overhead

### 5) GARCH(1,1) parameter estimation

Measures:

- MLE fit wall-time for varying return series lengths
- parameter parity (omega, alpha, beta) vs arch
- log-likelihood parity vs arch

## Parity methodology

**Where parity holds (machine-precision)**:
- Kalman filter log-likelihood and state estimates — < 1e-6 relative to
  pykalman/statsmodels with identical state-space specification
- Kalman EM estimated Q and R — < 1e-6 relative to pykalman

**Where parity is approximate**:
- GARCH(1,1) — parameter differences depend on optimizer convergence
  tolerance and starting values. Log-likelihood parity < 1e-4 relative to arch.
- EM convergence iteration count — sensitive to tolerance and initialization

## Scaling axes (what we vary)

- State dimension (1D, 3D, 10D, 50D)
- Observation count (100, 1K, 10K, 100K time steps)
- Missing data fraction (0%, 10%, 50%)

## How to run locally (current)

Rust microbenchmarks:

```bash
cargo bench -p ns-inference --bench kalman_benchmark
```

CLI commands:

```bash
nextstat timeseries kalman-filter --input kalman_1d.json
nextstat timeseries kalman-smooth --input kalman_1d.json
nextstat timeseries kalman-em --input kalman_1d.json --max-iter 50
nextstat timeseries kalman-forecast --input kalman_1d.json --forecast-steps 20
```

## Apex2 time series suite

The time series validation results are included in the Apex2 master report under the `timeseries` key.

## Related reading

- [Public Benchmarks Specification](/docs/public-benchmarks) — canonical spec.
- [Time Series Tutorial](/docs/tutorials/phase-8-timeseries) — Kalman filter/smoother/EM walkthrough.
- [Validation Report Artifacts](/docs/validation-report) — validation pack for published snapshots.
