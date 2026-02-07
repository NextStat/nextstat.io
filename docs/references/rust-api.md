---
title: "Rust API Reference"
status: stable
---

# Rust API Reference

This page summarizes the main Rust crates and their public entry points.

## Crates

- `ns-core`: core types and traits (model interface, `FitResult`, error handling).
- `ns-root`: native ROOT file I/O (TH1 histograms, TTree columnar access, expression engine, histogram filler).
- `ns-inference`: inference algorithms and model packs (MLE, NUTS, CLs, time series, PK/NLME, etc.).
- `ns-translate`: ingestion and translation layers (pyhf JSON, HistFactory XML, ntuple-to-workspace builder).
- `ns-viz`: plot-friendly artifacts (CLs curves, profile scans).
- `ns-cli`: `nextstat` CLI.
- `ns-wasm` (in `bindings/ns-wasm`): `wasm-bindgen` bindings used by the static `playground/` demo.

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

## `ns-root`

Native ROOT file reader — zero dependency on ROOT C++.

Key exports:
- `ns_root::RootFile` — open ROOT files (mmap or owned bytes), read TH1 histograms and TTrees.
- `ns_root::{Tree, BranchInfo, LeafType}` — TTree metadata and branch descriptors.
- `ns_root::BranchReader` — columnar data extraction with parallel basket decompression (rayon).
- `ns_root::CompiledExpr` — expression engine for selections/weights (`compile()` → `eval_row()` / `eval_bulk()`). Supports ternary `cond ? a : b`; parse errors include `line/col`.
- `ns_root::{HistogramSpec, FilledHistogram, fill_histograms}` — single-pass histogram filling.

TTree example:

```rust
use ns_root::RootFile;

let file = RootFile::open("data.root")?;
let tree = file.get_tree("events")?;

// Read a branch as Vec<f64> (type conversion from any numeric leaf type)
let pt: Vec<f64> = file.branch_data(&tree, "pt")?;

// Expression engine
let sel = ns_root::CompiledExpr::compile("njet >= 4 && pt > 25.0")?;
let mask = sel.eval_bulk(&[&njet_col, &pt_col]);

// Fill histogram with selection + weight
let spec = ns_root::HistogramSpec {
    name: "mbb".into(),
    variable: ns_root::CompiledExpr::compile("mbb")?,
    weight: Some(ns_root::CompiledExpr::compile("weight_mc")?),
    selection: Some(sel),
    bin_edges: vec![0., 50., 100., 150., 200., 300.],
    flow_policy: ns_root::FlowPolicy::Drop,
    negative_weight_policy: ns_root::NegativeWeightPolicy::Allow,
};
let histos = ns_root::fill_histograms(&[spec], &columns)?;
```

## `ns-translate`

Key exports (in addition to existing pyhf/HistFactory):
- `ns_translate::NtupleWorkspaceBuilder` — fluent builder: ROOT ntuples → HistFactory `Workspace`.
- `ns_translate::ntuple::{ChannelConfig, SampleConfig, NtupleModifier}` — configuration types.

Ntuple-to-workspace example:

```rust
use ns_translate::NtupleWorkspaceBuilder;

let ws = NtupleWorkspaceBuilder::new()
    .ntuple_path("ntuples/")
    .tree_name("events")
    .measurement("meas", "mu")
    .add_channel("SR", |ch| {
        ch.variable("mbb")
          .binning(&[0., 50., 100., 150., 200., 300.])
          .selection("njet >= 4 && pt > 25.0")
          .add_sample("signal", |s| {
              s.file("ttH.root").weight("weight_mc").normfactor("mu")
          })
          .add_sample("background", |s| {
              s.file("ttbar.root")
               .weight("weight_mc")
               .normsys("bkg_norm", 0.9, 1.1)
               .staterror()
          })
    })
    .build()?;  // → Workspace
```

## `ns-compute` — GPU backends

In addition to the SIMD Poisson NLL path, `ns-compute` provides optional GPU backends:

- **`accelerate` feature** (macOS): Apple Accelerate (vDSP/vForce) for vectorized `ln()` and subtraction.
  - Compile-time: build `ns-cli` (or your crate) with `--features accelerate`.
  - Runtime: enabled by default, but can be disabled for strict parity/determinism via:
    - programmatic API: `ns_compute::set_accelerate_enabled(false)`
    - env var: `NEXTSTAT_DISABLE_ACCELERATE=1`
    - CLI: `--threads 1` (auto-disables Accelerate)
- **`cuda` feature** (Linux/NVIDIA): CUDA batch NLL + analytical gradient via cudarc 0.19 (dynamic loading — binary works without CUDA installed).

Key CUDA exports (feature-gated):
- `ns_compute::cuda_types::{GpuModelData, GpuModifierEntry, GpuModifierDesc, ...}` — `#[repr(C)]` structs for GPU data layout (always available, no feature gate).
- `ns_compute::cuda_batch::CudaBatchAccelerator` — GPU orchestrator: model upload, batch NLL+gradient kernel launch, result download.
  - `single_nll_grad(params)` — convenience wrapper for single-model NLL+gradient (n_active=1).
  - `single_nll(params)` — NLL-only for single model.
  - `upload_observed_single(obs, ln_facts, mask)` — upload observed data for one model.

Key inference exports — **batch** (feature-gated):
- `ns_inference::gpu_batch::fit_toys_batch_gpu(model, params, n_toys, seed, config)` — Lockstep L-BFGS-B batch optimizer using GPU kernel.
- `ns_inference::batch::is_cuda_batch_available()` — Runtime CUDA availability check.

Key inference exports — **single-model** (feature-gated):
- `ns_inference::gpu_single::GpuSession` — Shared GPU state for single-model fits. Serializes and uploads model data once; reuses across multiple fits (profile scan, ranking).
  - `GpuSession::new(model)` — Create session, upload model + observed data.
  - `upload_observed(model)` — Re-upload observed data (e.g. after `with_observed_main()`).
  - `nll_grad(params)` — Single NLL+gradient evaluation on GPU.
  - `fit_minimum(model, config)` — Run L-BFGS-B optimizer with GPU objective.
  - `fit_minimum_from_with_bounds(model, init, bounds, config)` — Warm-start + custom bounds.
- `ns_inference::gpu_single::GpuObjective` (internal) — `ObjectiveFunction` with fused NLL+gradient caching. Exploits argmin's cost-then-gradient contract: 1 GPU kernel launch per iteration instead of 2.

MLE GPU methods (on `MaximumLikelihoodEstimator`, feature-gated):
- `fit_gpu(model)` — Full fit with Hessian (GPU minimization + CPU Hessian via finite differences of GPU gradient).
- `fit_minimum_gpu(model)` — NLL minimization only (no Hessian).
- `fit_minimum_gpu_from(model, init)` — With warm-start.
- `fit_minimum_gpu_from_with_bounds(model, init, bounds)` — With custom bounds (for fixed-param fits).

Profile likelihood GPU:
- `ns_inference::profile_likelihood::scan_gpu(mle, model, mu_values)` — GPU-accelerated profile scan with shared `GpuSession` and warm-start between mu values.

Model serialization:
- `HistFactoryModel::serialize_for_gpu() -> GpuModelData` — Converts HistFactory model to flat GPU-friendly buffers (nominal counts, CSR modifiers, constraints).

## CLI

The CLI is implemented in `crates/ns-cli` and wraps `ns-inference` surfaces.
See `docs/references/cli.md`.
