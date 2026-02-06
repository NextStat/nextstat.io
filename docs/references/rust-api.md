---
title: "Rust API Reference"
status: stable
---

# Rust API Reference

This page summarizes the main Rust crates and their public entry points.

## Crates

- `ns-core`: core types and traits (model interface, `FitResult`, error handling).
- `ns-inference`: inference algorithms and model packs (MLE, NUTS, CLs, time series, PK/NLME, etc.).
- `ns-translate`: ingestion and translation layers (pyhf/HistFactory).
- `ns-viz`: plot-friendly artifacts (CLs curves, profile scans).
- `ns-cli`: `nextstat` CLI.

## `ns-core`

Key exports:
- `ns_core::traits::{Model, LogDensityModel, PreparedNll, ComputeBackend}`
- `ns_core::{FitResult, Error, Result}`

The stable integration point for new models is `LogDensityModel` (NLL + gradient + metadata).

## `ns-inference`

Key exports (see `crates/ns-inference/src/lib.rs`):
- MLE: `MaximumLikelihoodEstimator`, `OptimizerConfig`, `OptimizationResult`
- Posterior + priors: `Posterior`, `Prior`
- NUTS: `NutsConfig`, `sample_nuts`, `sample_nuts_multichain`
- Frequentist: `AsymptoticCLsContext`, `HypotestResult`
- Profile likelihood: `ProfileLikelihoodScan`, `ProfilePoint`
- Laplace: `laplace_log_marginal`, `LaplaceResult`
- Time series: Kalman / EM / forecasting utilities (see `ns_inference::timeseries::*`)
- Pharmacometrics: `OneCompartmentOralPkModel`, `OneCompartmentOralPkNlmeModel`, `LloqPolicy`

Minimal MLE example:

```rust
use ns_inference::MaximumLikelihoodEstimator;
use ns_translate::pyhf::{HistFactoryModel, Workspace};

let json = std::fs::read_to_string("workspace.json")?;
let ws: Workspace = serde_json::from_str(&json)?;
let model = HistFactoryModel::from_workspace(&ws)?;

let mle = MaximumLikelihoodEstimator::new();
let fit = mle.fit(&model)?;
println!("nll={}", fit.nll);
```

## CLI

The CLI is implemented in `crates/ns-cli` and wraps `ns-inference` surfaces.
See `docs/references/cli.md`.

