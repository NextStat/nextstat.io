---
title: "Rust API Reference"
status: stable
---

# Rust API Reference

This page summarizes the main Rust crates and their public entry points.

## Crates

| Crate | Purpose |
|-------|---------|
| `ns-core` | Core types and traits (model interface, `FitResult`, error handling) |
| `ns-ad` | Automatic differentiation primitives (forward-mode Dual, reverse-mode Tape, `Scalar` trait) |
| `ns-prob` | Probability distributions and math (logpdf, cdf, transforms) |
| `ns-compute` | Compute backends: SIMD, Apple Accelerate, CUDA, Metal |
| `ns-root` | Native ROOT file I/O (TH1, TTree, expression engine, histogram filler) |
| `ns-translate` | Format translators: pyhf JSON, HistFactory XML, TRExFitter config, ntuple builder |
| `ns-inference` | Inference algorithms and model packs (MLE, NUTS, CLs, time series, PK/NLME, etc.) |
| `ns-viz` | Plot-friendly artifacts (CLs curves, profile scans) |
| `ns-cli` | `nextstat` CLI |
| `ns-server` | `nextstat-server` REST API for shared GPU inference (axum) |
| `ns-wasm` | WebAssembly bindings for the browser playground |
| `ns-py` | Python bindings via PyO3/maturin |

---

## `ns-core`

Key exports:
- `ns_core::traits::{Model, LogDensityModel, PreparedNll, ComputeBackend, PoiModel, FixedParamModel}`
- `ns_core::{FitResult, Error, Result}`

`FitResult` fields:
- `nll`, `params`, `uncertainties`, `converged`, `n_iter`, `n_fev`, `n_gev`
- `termination_reason: String` — optimizer termination reason
- `final_grad_norm: f64` — final gradient norm
- `initial_nll: f64` — NLL at start
- `n_active_bounds: usize` — number of active box constraints

The stable integration point for new models is `LogDensityModel` (NLL + gradient + metadata).

---

## `ns-ad`

Automatic differentiation primitives. `#![warn(missing_docs)]`.

Key exports:
- `ns_ad::dual::Dual` — forward-mode dual number (`value + epsilon * derivative`). Implements all standard math ops (`exp`, `ln`, `powi`, `powf`, `sqrt`, `abs`, `max`).
- `ns_ad::dual32::Dual32` — f32 variant for Metal GPU validation.
- `ns_ad::tape::Tape` — reverse-mode computation tape for efficient gradient computation with many parameters. `gradient_reverse_reuse()` clears and reuses a tape across optimizer iterations.
- `ns_ad::scalar::Scalar` — trait abstracting `f64`, `Dual`, and tape variables. Enables writing generic code that works for both evaluation and AD:

```rust
use ns_ad::scalar::Scalar;

fn poisson_nll<T: Scalar>(expected: T, observed: f64) -> T {
    expected - T::from_f64(observed) * expected.ln()
}
```

---

## `ns-prob`

Probability building blocks. Reusable across all inference domains.

Modules:
- `ns_prob::normal` — `logpdf(x, mu, sigma)`, `cdf(x, mu, sigma)`, `quantile(p, mu, sigma)`
- `ns_prob::poisson` — `logpmf(k, lambda)`
- `ns_prob::exponential` — `logpdf(x, rate)`, `cdf(x, rate)`
- `ns_prob::gamma` — `logpdf(x, shape, rate)`
- `ns_prob::beta` — `logpdf(x, alpha, beta)`
- `ns_prob::weibull` — `logpdf(x, k, lambda)`
- `ns_prob::student_t` — `logpdf(x, df, mu, sigma)`
- `ns_prob::bernoulli` — `logpmf(k, p)`
- `ns_prob::binomial` — `logpmf(k, n, p)`
- `ns_prob::neg_binomial` — `logpmf(k, r, p)`
- `ns_prob::math` — stable numeric helpers (`log1pexp`, `logsumexp`, `sigmoid`)
- `ns_prob::transforms` — bijective transforms for unconstrained parameterization (`log`, `logit`, `softplus`)
- `ns_prob::distributions` — trait-based distribution interface

---

## `ns-inference`

Key exports (see `crates/ns-inference/src/lib.rs`):
- MLE: `MaximumLikelihoodEstimator`, `OptimizerConfig`, `OptimizationResult`, `RankingEntry`
- Posterior + priors: `Posterior`, `Prior`
- NUTS: `NutsConfig`, `sample_nuts`, `sample_nuts_multichain`
- Frequentist: `AsymptoticCLsContext`, `HypotestResult`
- Toy-based: `hypotest_qtilde_toys`, `hypotest_qtilde_toys_expected_set`, `ToyHypotestResult`
- Profile likelihood: `ProfileLikelihoodScan`, `ProfilePoint`, `scan_histfactory`
- Laplace: `laplace_log_marginal`, `LaplaceResult`
- Model builder: `ModelBuilder`, `ComposedGlmModel` (hierarchical GLMs)
- Regression: `LinearRegressionModel`, `LogisticRegressionModel`, `PoissonRegressionModel`, `ols_fit`
- Ordinal: `OrderedLogitModel`, `OrderedProbitModel`
- LMM: `LmmMarginalModel`, `LmmRandomEffects`
- Survival: `ExponentialSurvivalModel`, `WeibullSurvivalModel`, `LogNormalAftModel`, `CoxPhModel`
- Time series: Kalman / EM / forecasting utilities (see `ns_inference::timeseries::*`)
- PK/NLME: `OneCompartmentOralPkModel`, `OneCompartmentOralPkNlmeModel`, `LloqPolicy`
- ODE: `rk4_linear`, `OdeSolution`
- Optimizer: `LbfgsbOptimizer`, `ObjectiveFunction`
- Batch: `fit_toys_batch`, `is_accelerate_available`
- Transforms: `ParameterTransform`

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

### Differentiable Layer (CUDA, feature-gated)

Zero-copy PyTorch integration for ML workflows:

- `ns_inference::differentiable::DifferentiableSession` — GPU session for differentiable NLL evaluation.
  - `new(model, signal_sample_name)` — upload model to GPU, identify signal sample.
  - `nll_grad_signal(params, d_signal, d_grad_signal)` — compute NLL and write `dNLL/d(signal)` gradient directly into a PyTorch CUDA tensor (zero-copy via raw device pointers).
  - `signal_n_bins()`, `n_params()`, `parameter_init()` — metadata accessors.

- `ns_inference::differentiable::ProfiledDifferentiableSession` — GPU session for profiled test statistics.
  - `new(model, signal_sample_name)` — upload model, require POI defined.
  - `profiled_q0_and_grad(d_signal) -> (f64, Vec<f64>)` — discovery test statistic q0 and its gradient w.r.t. signal bins. Runs two GPU L-BFGS-B fits (null + unconditional) and applies the envelope theorem: `dq0/ds = 2*(dNLL/ds|_{theta_hat_0} - dNLL/ds|_{theta_hat})`.
  - `profiled_qmu_and_grad(mu_test, d_signal) -> (f64, Vec<f64>)` — exclusion test statistic qmu.

---

## `ns-root`

Native ROOT file reader — zero dependency on ROOT C++.

Key exports:
- `ns_root::RootFile` — open ROOT files (mmap or owned bytes), read TH1 histograms and TTrees.
- `ns_root::{Tree, BranchInfo, LeafType}` — TTree metadata and branch descriptors.
- `ns_root::BranchReader` — columnar data extraction with parallel basket decompression (rayon).
- `ns_root::CompiledExpr` — expression engine for selections/weights (`compile()` -> `eval_row()` / `eval_bulk()`). Supports ternary `cond ? a : b`; parse errors include `line/col`.
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

---

## `ns-translate`

Format translators. Ingestion from multiple HEP/analysis formats into the internal model.

Key exports:
- `ns_translate::pyhf::{Workspace, HistFactoryModel, NllScratch, PreparedModel}` — pyhf JSON format (full HistFactory probability model).
- `ns_translate::histfactory` — HistFactory XML format (ROOT-style workspace definition).
- `ns_translate::trex` — TRExFitter configuration files (`.txt` and `.yaml` formats). Parses Region/Sample/Systematic/NormFactor blocks and converts to `Workspace`.
- `ns_translate::NtupleWorkspaceBuilder` — fluent builder: ROOT ntuples -> HistFactory `Workspace`.
- `ns_translate::ntuple::{ChannelConfig, SampleConfig, NtupleModifier}` — configuration types.
- `ns_translate::hs3` — HS3 (HEP Statistics Serialization Standard) v0.2 format. See below.

### HS3 module (`ns_translate::hs3`)

Native HS3 JSON support — direct conversion to `HistFactoryModel` without going through the pyhf intermediate format.

Sub-modules:
- `ns_translate::hs3::schema` — Serde types for HS3 JSON. Custom `Deserialize` for tagged-union distributions and modifiers. Unknown types preserved as raw `serde_json::Value` (forward-compatible).
- `ns_translate::hs3::resolve` — Two-pass reference resolver. Pass 1: index all named objects. Pass 2: resolve analysis → likelihood → channels → samples → modifiers. Produces `ResolvedWorkspace`.
- `ns_translate::hs3::convert` — Convert `ResolvedWorkspace` → `HistFactoryModel`. Handles parameter registration, staterror sigma_rel computation, constraint mapping.
- `ns_translate::hs3::detect` — Fast format auto-detection (`WorkspaceFormat::Hs3 | Pyhf | Unknown`). Prefix scan + full-parse fallback.
- `ns_translate::hs3::export` — Export `HistFactoryModel` → HS3 JSON for roundtrip workflows (load → fit → export with bestfit params).

Key public functions:
- `ns_translate::hs3::convert::from_hs3_default(json) -> Result<HistFactoryModel>` — parse HS3 with ROOT HistFactory defaults (Code1 NormSys, Code0 HistoSys).
- `ns_translate::hs3::convert::from_hs3(json, analysis, param_points, normsys_interp, histosys_interp) -> Result<HistFactoryModel>` — full-control HS3 parsing.
- `ns_translate::hs3::detect::detect_format(json) -> WorkspaceFormat` — instant format detection.
- `ns_translate::hs3::export::export_hs3(model, name, bestfit, original) -> Hs3Workspace` — export to HS3.
- `ns_translate::hs3::export::export_hs3_json(model, name, bestfit, original) -> Result<String>` — export to JSON string.

Example:

```rust
use ns_translate::hs3::convert::from_hs3_default;
use ns_translate::hs3::detect::{detect_format, WorkspaceFormat};
use ns_translate::hs3::export::export_hs3_json;

let json = std::fs::read_to_string("workspace-postFit_PTV.json")?;

// Auto-detect format
match detect_format(&json) {
    WorkspaceFormat::Hs3 => {
        let model = from_hs3_default(&json)?;
        println!("HS3: {} params, {} channels", model.n_params(), model.n_channels());

        // Roundtrip: export back to HS3 JSON
        let exported = export_hs3_json(&model, "analysis", None, None)?;
        std::fs::write("exported.json", exported)?;
    }
    WorkspaceFormat::Pyhf => { /* pyhf path */ }
    WorkspaceFormat::Unknown => { /* error */ }
}
```

Supported modifier types: `normfactor`, `normsys`, `histosys`, `staterror`, `shapesys`, `shapefactor`, `lumi`. Unknown types are silently skipped.

### Arrow / Parquet module (`ns_translate::arrow`)

Zero-copy columnar data interchange with the Arrow ecosystem. Feature-gated behind `arrow-io`.

Sub-modules:
- `ns_translate::arrow::ingest` — Arrow IPC / RecordBatch → pyhf `Workspace`.
  - `ArrowIngestConfig { poi, observations, measurement_name }` — ingestion configuration.
  - `from_arrow_ipc(ipc_bytes, config) -> Result<Workspace>` — ingest Arrow IPC bytes.
  - `from_record_batches(batches, config) -> Result<Workspace>` — ingest RecordBatch slice.
  - `read_ipc_batches(ipc_bytes) -> Result<Vec<RecordBatch>>` — deserialize IPC to batches.
- `ns_translate::arrow::export` — `HistFactoryModel` → Arrow RecordBatch / IPC bytes.
  - `yields_to_ipc(model, params) -> Result<Vec<u8>>` — export expected yields.
  - `parameters_to_ipc(model, params) -> Result<Vec<u8>>` — export parameter metadata.
  - `yields_to_record_batch(model, params) -> Result<RecordBatch>` — yields as RecordBatch.
  - `parameters_to_record_batch(model, params) -> Result<RecordBatch>` — params as RecordBatch.
  - `record_batch_to_ipc(batch) -> Result<Vec<u8>>` — generic batch → IPC serialization.
- `ns_translate::arrow::parquet` — Parquet read/write (Zstd compression).
  - `from_parquet(path, config) -> Result<Workspace>` — read Parquet file to Workspace.
  - `from_parquet_bytes(data, config) -> Result<Workspace>` — read Parquet bytes.
  - `write_parquet(path, batches) -> Result<()>` — write RecordBatches to Parquet.

Schema: `channel` (Utf8), `sample` (Utf8), `yields` (List\<Float64\>), optional `stat_error` (List\<Float64\>).

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
    .build()?;  // -> Workspace
```

---

## `ns-compute` — Evaluation Modes and GPU Backends

### EvalMode: Parity vs Fast

`ns-compute` provides a process-wide evaluation mode that controls the trade-off between
numerical precision and speed:

```rust
use ns_compute::{EvalMode, set_eval_mode, eval_mode};

// Parity mode: Kahan summation, Accelerate disabled, single-thread recommended.
// Used for deterministic pyhf parity validation.
set_eval_mode(EvalMode::Parity);

// Fast mode (default): naive summation, SIMD/Accelerate/CUDA enabled.
set_eval_mode(EvalMode::Fast);

assert_eq!(eval_mode(), EvalMode::Fast);
```

| Mode | Summation | Backend | Threads | Use Case |
|------|-----------|---------|---------|----------|
| `Fast` | Naive `+=` | SIMD / Accelerate / CUDA | Rayon (auto) | Production inference |
| `Parity` | Kahan compensated | SIMD only (Accelerate OFF) | 1 (forced) | CI, pyhf parity validation |

When `Parity` mode is activated:
1. `EvalMode::Parity` is set via atomic flag (zero-cost read)
2. Apple Accelerate is automatically disabled
3. Kahan compensated summation is used in `PreparedModel::nll()` dispatch
4. Thread count should be set to 1 for full determinism

Kahan summation variants (in `ns_compute::simd`):
- `poisson_nll_simd_kahan()` — SIMD f64x4 accumulator + f64x4 compensation
- `poisson_nll_simd_sparse_kahan()` — sparse variant (skips zero-obs bins)
- `poisson_nll_scalar_kahan()` — scalar loop with compensation
- `poisson_nll_accelerate_kahan()` — Accelerate vvlog + Kahan scalar sum

**Measured overhead:** <5% (Kahan vs naive at same thread count).

**Tolerance contract:** See `docs/pyhf-parity-contract.md` for 7 tolerance tiers from
per-bin 1e-12 to toy ensemble 0.05.

### PreparedModel NLL Dispatch

`PreparedModel::nll()` dispatches to the appropriate backend via `dispatch_poisson_nll()`:

```
EvalMode::Parity ->
  has_zero_obs? -> poisson_nll_simd_sparse_kahan()
  otherwise    -> poisson_nll_simd_kahan()

EvalMode::Fast ->
  accelerate_enabled()? -> poisson_nll_accelerate()
  has_zero_obs?         -> poisson_nll_simd_sparse()
  otherwise             -> poisson_nll_simd()
```

### Batch Toy Fitting (CPU)

```rust
use ns_inference::batch::fit_toys_batch;

let results = fit_toys_batch(&model, &params, /*n_toys=*/1000, /*seed=*/42, None);
// Uses par_iter().map_init() — one Tape per Rayon thread (~12), not per toy (1000+)
```

### GPU Backends

In addition to the SIMD Poisson NLL path, `ns-compute` provides optional GPU backends:

- **`accelerate` feature** (macOS): Apple Accelerate (vDSP/vForce) for vectorized `ln()` and subtraction.
  - Compile-time: build `ns-cli` (or your crate) with `--features accelerate`.
  - Runtime: enabled by default, but can be disabled for strict parity/determinism via:
    - programmatic API: `ns_compute::set_accelerate_enabled(false)`
    - env var: `NEXTSTAT_DISABLE_ACCELERATE=1`
    - CLI: `--threads 1` (auto-disables Accelerate)
- **`cuda` feature** (Linux/NVIDIA): CUDA batch NLL + analytical gradient via cudarc 0.19 (dynamic loading — binary works without CUDA installed).
- **`metal` feature** (macOS/Apple Silicon): Metal GPU batch NLL + analytical gradient in f32. Runtime MSL compilation, `StorageModeShared` (zero-copy unified memory). Requires Apple GPU family 7+ (M1+).

Key CUDA exports (feature-gated):
- `ns_compute::cuda_types::{GpuModelData, GpuModifierEntry, GpuModifierDesc, ...}` — `#[repr(C)]` structs for GPU data layout (always available, no feature gate).
- `ns_compute::cuda_batch::CudaBatchAccelerator` — GPU orchestrator: model upload, batch NLL+gradient kernel launch, result download.
  - `single_nll_grad(params)` — convenience wrapper for single-model NLL+gradient (n_active=1).
  - `single_nll(params)` — NLL-only for single model.
  - `upload_observed_single(obs, ln_facts, mask)` — upload observed data for one model.
- `ns_compute::differentiable::DifferentiableAccelerator` — CUDA accelerator for differentiable NLL with per-signal-bin gradients (zero-copy PyTorch integration).

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
- `fit_gpu_from(model, init)` — Full fit with warm-start + Hessian (GPU minimization from explicit starting point + CPU Hessian).
- `fit_minimum_gpu(model)` — NLL minimization only (no Hessian).
- `fit_minimum_gpu_from(model, init)` — With warm-start.
- `fit_minimum_gpu_from_with_bounds(model, init, bounds)` — With custom bounds (for fixed-param fits).

Profile likelihood GPU:
- `ns_inference::profile_likelihood::scan_gpu(mle, model, mu_values)` — GPU-accelerated profile scan with shared `GpuSession` and warm-start between mu values.

Key Metal exports (feature-gated):
- `ns_compute::metal_types::{MetalModelData, MetalAuxPoissonEntry, MetalGaussConstraintEntry}` — f32 `#[repr(C)]` structs for Metal GPU data layout (always available, no feature gate).
- `ns_compute::metal_batch::MetalBatchAccelerator` — Metal GPU orchestrator: MSL compilation, buffer management, kernel dispatch. f64<->f32 conversion at API boundary.
  - `from_metal_data(data, max_batch)` — Compile MSL, upload static model buffers, pre-allocate dynamic buffers.
  - `batch_nll_grad(params, n_active) -> (Vec<f64>, Vec<f64>)` — Fused NLL + gradient for all active toys.
  - `batch_nll(params, n_active) -> Vec<f64>` — NLL-only (for line search).
  - `single_nll_grad(params)`, `single_nll(params)` — Convenience wrappers for n_active=1.
  - `upload_observed(obs, ln_facts, mask, n_toys)` — Upload toy observed data (f64->f32).
- `ns_inference::metal_batch::fit_toys_batch_metal(model, params, n_toys, seed, config)` — Lockstep L-BFGS-B batch optimizer using Metal kernel. Tolerance clamped to max(tol, 1e-3) for f32.
- `ns_inference::batch::is_metal_batch_available()` — Runtime Metal availability check.
- `ns_inference::lbfgs::LbfgsState` — Shared standalone L-BFGS-B state machine (used by both CUDA and Metal batch fitters).

Model serialization:
- `HistFactoryModel::serialize_for_gpu() -> Result<GpuModelData>` — Converts HistFactory model to flat GPU-friendly buffers (nominal counts, CSR modifiers, constraints). Returns `Err(Validation)` if any NormSys modifier has non-positive hi/lo factors (the GPU polynomial kernel cannot represent the CPU piecewise-linear fallback).
- `MetalModelData::from_gpu_data(&GpuModelData)` — Converts f64 GPU data to f32 Metal data with pre-computed `lgamma` for auxiliary Poisson constraints.

---

## `ns-viz`

Lightweight, dependency-free artifacts for plot-friendly output. Models return plain structs that can be serialized to JSON or consumed directly.

---

## `ns-server`

Self-hosted REST API for shared GPU inference. Built on axum 0.8.

Key exports:
- `ns_server::state::AppState` — server state with GPU lock, atomic counters, model cache.
- `ns_server::pool::ModelPool` — LRU model cache keyed by SHA-256 hash of workspace JSON.
- `ns_server::pool::ModelInfo` — cached model metadata (id, name, n_params, n_channels, age, hit count).

Endpoints: `/v1/fit`, `/v1/ranking`, `/v1/batch/fit`, `/v1/batch/toys`, `/v1/models`, `/v1/health`.

See `docs/references/cli.md` for full endpoint documentation and CLI arguments.

---

## CLI

The CLI is implemented in `crates/ns-cli` and wraps `ns-inference` surfaces.
See `docs/references/cli.md`.
