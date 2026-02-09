---
title: "Benchmark Suite: Time Series (Kalman / State Space)"
description: "Time series benchmark suite for NextStat: Kalman filter/smoother throughput, EM convergence cost, forecasting latency, and state-dimension scaling for state-space models."
status: draft
last_updated: 2026-02-08
keywords:
  - Kalman filter benchmark
  - state space model performance
  - EM algorithm convergence
  - time series forecasting latency
  - Kalman smoother throughput
  - statsmodels comparison
  - scientific time series
  - NextStat time series
---

# Time Series Benchmark Suite (Kalman / State Space)

This suite benchmarks NextStat's time series and state-space model infrastructure:

- Kalman filter forward pass throughput
- Kalman smoother (RTS) throughput
- EM convergence cost (iterations × NLL evaluations)
- Forecasting latency per horizon step

This page is a **runbook + methodology**. Results are published as benchmark snapshots (see [Public Benchmarks](/docs/benchmarks/public-benchmarks)).

## What is compared

Planned comparisons include:

- **NextStat (Rust core)** vs **statsmodels** (`UnobservedComponents`, `KalmanFilter`)
- Optional: **PyKalman** and **filterpy** for filter/smoother throughput

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

- [Public Benchmarks Specification](/docs/benchmarks/public-benchmarks) — canonical spec.
- [Time Series Tutorial](/docs/tutorials/phase-8-timeseries) — Kalman filter/smoother/EM walkthrough.
- [Validation Report Artifacts](/docs/references/validation-report) — validation pack for published snapshots.
