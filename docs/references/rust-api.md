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
| `ns-unbinned` | Event-level (unbinned) PDFs, normalizing flows, DCR surrogates, EventStore |
| `ns-compute` | Compute backends: SIMD, Apple Accelerate, CUDA, Metal |
| `ns-root` | Native ROOT file I/O (TH1, TTree, expression engine, histogram filler) |
| `ns-translate` | Format translators: pyhf JSON, HistFactory XML, TRExFitter config, ntuple builder |
| `ns-inference` | Inference algorithms and model packs (MLE, NUTS, CLs, time series, PK/NLME, etc.) |
| `ns-viz` | Plot-friendly artifacts (CLs curves, profile scans) |
| `ns-cli` | `nextstat` CLI |
| `ns-server` | `nextstat-server` REST API for shared GPU inference (axum) |
| `ns-wasm` | WebAssembly bindings for the browser playground |
| `ns-zstd` | Pure Rust Zstd decoder (fork of ruzstd, optimized for ROOT decompression) |
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

Probability building blocks. Reusable across all inference domains. Each distribution module exports `logpdf`/`logpmf` and `nll` (negative log-likelihood).

Modules:
- `ns_prob::normal` — `logpdf(x, mu, sigma)`, `nll(x, mu, sigma)`
- `ns_prob::poisson` — `logpmf(k, lambda)`, `nll(k, lambda)`
- `ns_prob::exponential` — `logpdf(x, rate)`, `nll(x, rate)`
- `ns_prob::gamma` — `logpdf(x, shape, rate)`, `logpdf_shape_scale(x, shape, scale)`, `nll_shape_rate(x, shape, rate)`
- `ns_prob::beta` — `logpdf(x, a, b)`, `nll(x, a, b)`
- `ns_prob::weibull` — `logpdf(x, k, lambda)`, `nll(x, k, lambda)`
- `ns_prob::student_t` — `logpdf(x, mu, sigma, nu)`, `nll(x, mu, sigma, nu)`
- `ns_prob::bernoulli` — `logpmf(k, p)`, `logpmf_logit(k, logit_p)`, `nll(k, p)`
- `ns_prob::binomial` — `logpmf(k, n, p)`, `logpmf_logit(k, n, logit_p)`, `nll(k, n, p)`
- `ns_prob::neg_binomial` — `logpmf_r_p(k, r, p)`, `logpmf_mean_disp(k, mu, alpha)` (NB2), `nll_mean_disp(k, mu, alpha)`
- `ns_prob::math` — stable numeric helpers: `log1pexp(x)`, `sigmoid(x)`, `log_sigmoid(x)`, `softplus(x)`, `exp_clamped(x)`
- `ns_prob::transforms` — bijective transforms for unconstrained parameterization (see below)
- `ns_prob::distributions` — convenience wrapper functions: `normal_logpdf`, `student_t_logpdf`, `bernoulli_logpmf_logit`, `binomial_logpmf_logit`, `poisson_logpmf`, `negbinom_logpmf`, `gamma_logpdf`, `lognormal_logpdf`, `logit_to_prob`

### Transforms (`ns_prob::transforms`)

Provides a `Bijector` trait and concrete implementations for mapping constrained parameters to unconstrained space:

- `Bijector` trait: `forward(z) -> theta`, `inverse(theta) -> z`, `log_abs_det_jacobian(z)`, `grad_log_abs_det_jacobian(z)`
- Concrete bijectors: `IdentityBijector`, `ExpBijector`, `SoftplusBijector`, `SigmoidBijector`, `LowerBoundedBijector::new(lo)`, `UpperBoundedBijector::new(hi)`
- `ParameterTransform` — composite transform from parameter bounds:
  - `from_bounds(bounds) -> Self` — auto-select bijectors (Exp for `(0,∞)`, Sigmoid for bounded, etc.)
  - `from_bounds_softplus(bounds) -> Self` — prefer Softplus over Exp for lower-bounded params
  - `forward(z)`, `inverse(theta)`, `log_abs_det_jacobian(z)`, `jacobian_diag(z)`, `dim()`

---

## `ns-inference`

Key exports (see `crates/ns-inference/src/lib.rs`):
- MLE: `MaximumLikelihoodEstimator`, `OptimizerConfig`, `OptimizationResult`, `RankingEntry`
- Posterior + priors: `Posterior`, `Prior`
- NUTS: `NutsConfig`, `sample_nuts`, `sample_nuts_multichain`
- Frequentist: `AsymptoticCLsContext`, `HypotestResult`
- Toy-based: `hypotest_qtilde_toys`, `hypotest_qtilde_toys_expected_set`, `ToyHypotestResult`
- Toy-based GPU: `hypotest_qtilde_toys_gpu`, `hypotest_qtilde_toys_expected_set_gpu` (feature `cuda` or `metal`)
- Profile likelihood: `ProfileLikelihoodScan`, `ProfilePoint`, `scan_histfactory`, `scan_histfactory_diag`
- Profile GPU: `scan_gpu` (feature `cuda`), `scan_metal` (feature `metal`)
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
- Batch GPU: `fit_toys_batch_gpu` (feature `cuda`), `fit_toys_batch_metal` (feature `metal`)
- GPU sessions: `CudaGpuSession`, `cuda_session` (feature `cuda`); `MetalGpuSession`, `metal_session` (feature `metal`)
- Ranking GPU: `ranking_gpu` (feature `cuda`), `ranking_metal` (feature `metal`)
- Differentiable: `DifferentiableSession`, `ProfiledDifferentiableSession` (feature `cuda`); `MetalProfiledDifferentiableSession` (feature `metal`)
- Transforms: `ParameterTransform`
- Hybrid: `HybridLikelihood`, `SharedParameterMap` (combine binned + unbinned models with shared parameters)
- Toys: `asimov_main`, `poisson_main_from_expected`, `poisson_main_toys`
- Diagnostics: `DiagnosticsResult`, `QualityGates`, `compute_diagnostics`, `quality_summary` (MCMC split R-hat, bulk/tail ESS)
- Regression extended: `GammaRegressionModel`, `TweedieRegressionModel` (Phase 9)
- EVT: `GevModel`, `GpdModel` (extreme value theory — block maxima and peaks-over-threshold)
- Meta-analysis: `meta_fixed`, `meta_random`, `MetaAnalysisResult`, `ForestRow`, `Heterogeneity`
- Competing risks: `cumulative_incidence`, `gray_test`, `fine_gray_fit`, `CifEstimate`, `FineGrayResult`
- Sequential: `group_sequential_design`, `alpha_spending_design`, `sequential_test`, `SequentialDesign`
- Chain ladder: `chain_ladder_fit`, `mack_chain_ladder`, `bootstrap_reserves`, `ChainLadderResult`, `MackResult`
- Churn: `churn_risk_model`, `churn_uplift`, `churn_retention`, `bootstrap_hazard_ratios`, `cohort_retention_matrix`, and 15+ types
- Econometrics: `panel_fe_fit`, `did_canonical`, `event_study`, `iv_2sls`, `aipw_ate`, `rosenbaum_bounds`, `cluster_robust_se`
- PK extended: `TwoCompartmentIvPkModel`, `TwoCompartmentOralPkModel`, `ErrorModel` (analytical gradients)
- PD: `EmaxModel`, `SigmoidEmaxModel`, `IndirectResponseModel`, `PkPdLink`
- Dosing: `DoseEvent`, `DoseRoute`, `DosingRegimen`
- FOCE: `FoceConfig`, `FoceEstimator`, `FoceResult`, `OmegaMatrix`
- SAEM: `SaemConfig`, `SaemEstimator`, `SaemDiagnostics`
- SCM: `ScmConfig`, `ScmEstimator`, `ScmResult`, `CovariateCandidate`
- VPC: `VpcConfig`, `VpcResult`, `gof_1cpt_oral`, `vpc_1cpt_oral`
- ODE adaptive: `OdeSystem`, `OdeOptions`, `rk45`, `esdirk4`, `solve_at_times`
- NONMEM: `NonmemDataset`, `NonmemRecord`
- Artifacts: `NlmeArtifact`, `RunBundle`, `SCHEMA_VERSION`

Minimal MLE example:

```rust
use ns_inference::MaximumLikelihoodEstimator;
use ns_translate::pyhf::{HistFactoryModel, HistoSysInterpCode, NormSysInterpCode, Workspace};

let json = std::fs::read_to_string("workspace.json")?;
let ws: Workspace = serde_json::from_str(&json)?;

// Default interpolation (NextStat "smooth" defaults): NormSys=Code4, HistoSys=Code4p.
// For strict HistFactory/pyhf defaults, use Code1/Code0:
let model = HistFactoryModel::from_workspace_with_settings(
    &ws,
    NormSysInterpCode::Code1,
    HistoSysInterpCode::Code0,
)?;

let mle = MaximumLikelihoodEstimator::new();
let fit = mle.fit(&model)?;
println!("nll={}", fit.nll);
```

### GPU Flow Session (CUDA, feature-gated)

Orchestrates flow PDF evaluation + GPU NLL reduction for unbinned models with neural PDFs:

- `ns_inference::gpu_flow_session::GpuFlowSession` — session managing flow eval (CPU or CUDA EP) + GPU NLL reduction.
  - `new(config: GpuFlowSessionConfig)` — create session, allocate GPU buffers.
  - `nll(logp_flat, params) -> f64` — compute NLL from pre-computed `logp_flat[n_procs × n_events]`. Yields derived from `params` via process descriptors.
  - `nll_grad(params, eval_logp) -> (f64, Vec<f64>)` — NLL + gradient via central finite differences. `eval_logp` is called `2·n_params + 1` times.
  - `compute_yields(params) -> Vec<f64>` — yield computation from parameter vector.
- `ns_inference::gpu_flow_session::GpuFlowSessionConfig` — `processes`, `n_events`, `n_params`, `gauss_constraints`, `constraint_const`.
- `ns_inference::gpu_flow_session::FlowProcessDesc` — per-process descriptor: `base_yield`, `yield_param_idx`, `yield_is_scaled`.

```rust
use ns_inference::gpu_flow_session::{GpuFlowSession, GpuFlowSessionConfig, FlowProcessDesc};

let config = GpuFlowSessionConfig {
    processes: vec![
        FlowProcessDesc {
            process_index: 0,
            base_yield: 100.0,
            yield_param_idx: Some(0),
            yield_is_scaled: true,
        },
    ],
    n_events: 50_000,
    n_params: 1,
    gauss_constraints: vec![],
    constraint_const: 0.0,
};

let mut session = GpuFlowSession::new(config)?;
let nll = session.nll(&logp_flat, &params)?;
```

### Volatility Models (GARCH / Stochastic Volatility)

Financial time series volatility estimation in `ns_inference::timeseries::volatility`:

- `garch11_fit(y, config) -> Garch11Fit` — fit Gaussian GARCH(1,1) by MLE (L-BFGS-B).
  - `Garch11Params { mu, omega, alpha, beta }` — model parameters.
  - `Garch11Config { optimizer, alpha_beta_max, init, min_var }` — stationarity constraint + optimizer config.
  - `Garch11Fit { params, log_likelihood, conditional_variance, optimization }` — result with per-observation h_t.

- `sv_logchi2_fit(y, config) -> SvLogChi2Fit` — approximate stochastic volatility via Gaussian approximation for log(χ²₁) + Kalman MLE.
  - `SvLogChi2Params { mu, phi, sigma }` — log-variance mean, AR(1) persistence, vol-of-vol.
  - `SvLogChi2Config { optimizer, log_eps, init }` — configuration.
  - `SvLogChi2Fit { params, log_likelihood, smoothed_h, smoothed_sigma, optimization }` — result with RTS-smoothed log-variance series.

```rust
use ns_inference::timeseries::volatility::{garch11_fit, Garch11Config};

let returns = vec![0.01, -0.02, 0.005, 0.03, -0.015];
let fit = garch11_fit(&returns, Garch11Config::default())?;
println!("omega={:.4} alpha={:.4} beta={:.4}", fit.params.omega, fit.params.alpha, fit.params.beta);
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
- `ns_root::{Tree, BranchInfo, LeafType}` — TTree metadata and branch descriptors. **Compound leaf-list branches** (multiple scalars packed per entry, e.g. `"x/F:y/F:z/F"`) are now parsed — each scalar is exposed as a separate `BranchInfo`.
- `ns_root::BranchReader` — columnar data extraction with parallel basket decompression (rayon).
- `ns_root::BasketCache` — per-`RootFile` LRU cache of decompressed basket payloads. Byte-bounded eviction (default 256 MiB). Keyed by basket seek position; values are `Arc<Vec<u8>>` (shared ownership, zero-copy on cache hit). `RootFile::basket_cache()` returns cache stats; `RootFile::set_cache_config()` tunes capacity or disables caching.
- `ns_root::LazyBranchReader` — on-demand branch reader that decompresses only the baskets needed for the requested entries. Created via `RootFile::lazy_branch_reader()`. Methods: `read_f64_at(entry)` (single entry, one basket), `read_f64_range(start, end)` (range, only overlapping baskets), `read_all_f64()` (all entries), `load_all_chained()` (returns `ChainedSlice`).
- `ns_root::ChainedSlice` — zero-copy concatenation of multiple decompressed basket payloads via `Arc` sharing. O(log n) random access across non-contiguous segments via binary search on cumulative offsets. Methods: `locate(pos)`, `read_array::<N>(pos)`, `decode_f64_at(pos, leaf_type)`.
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
- `ns_translate::pyhf::{Workspace, HistFactoryModel, NllScratch, PreparedModel, NormSysInterpCode, HistoSysInterpCode}` — pyhf JSON format (full HistFactory probability model).
- `ns_translate::histfactory::{parse_combination_xml, HistFactoryXmlWorkspace}` — HistFactory XML format (ROOT-style workspace definition). Feature-gated behind `root-io`.
- `ns_translate::trex::{parse_trex_config, TrexConfig}` — TRExFitter configuration files (`.txt` and `.yaml` formats). Parses Region/Sample/Systematic/NormFactor blocks and converts to `Workspace`. Feature-gated behind `root-io`.
- `ns_translate::NtupleWorkspaceBuilder` — fluent builder: ROOT ntuples -> HistFactory `Workspace`.
- `ns_translate::ntuple::{ChannelConfig, SampleConfig, NtupleModifier}` — configuration types.
- `ns_translate::pyhf::audit::workspace_audit(json) -> Result<WorkspaceAudit>` — inspect a workspace: channel/sample/modifier counts, unsupported features.
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
- `ns_translate::hs3::export::import_unbinned_hs3(workspace, analysis, param_points) -> Result<UnbinnedSpecV0>` — import `nextstat_unbinned_dist` channels from HS3 into unbinned spec (`feature = "unbinned"`).
- `ns_translate::hs3::export::import_unbinned_hs3_json(json, analysis, param_points) -> Result<UnbinnedSpecV0>` — JSON convenience wrapper (`feature = "unbinned"`).
- `ns_translate::hs3::export::import_hybrid_hs3(workspace, analysis, param_points) -> Result<ImportedHybridHs3>` — split hybrid HS3 into binned `HistFactoryModel` + unbinned spec with shared parameter names (`feature = "unbinned"`).

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

**Tolerance contract:** 7 tiers from per-bin NLL 1e-12 to toy ensemble 0.05.
Canonical values: `tests/python/_tolerances.py`.

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

Key CUDA exports — **flow NLL reduction** (feature-gated):
- `ns_compute::cuda_flow_nll::CudaFlowNllAccelerator` — GPU NLL reduction from externally-computed log-prob values (flow PDFs). Separates PDF evaluation from likelihood reduction, enabling mixed parametric+flow models.
  - `new(config: &FlowNllConfig)` — allocate GPU buffers, load PTX kernel.
  - `nll(logp_flat, yields, params) -> f64` — host-upload path: `logp_flat[n_procs × n_events]` evaluated on CPU, uploaded to GPU for NLL reduction.
  - `nll_device(d_logp_flat, yields, params) -> f64` — device-resident path: `CudaSlice<f64>` from ONNX CUDA EP stays on GPU (zero host↔device copy for log-prob).
  - `is_available() -> bool` — runtime CUDA check.
- `ns_compute::cuda_flow_nll::FlowNllConfig` — configuration: `n_events`, `n_procs`, `n_params`, `gauss_constraints`, `constraint_const`.

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
- `ns_inference::metal_batch::fit_toys_from_data_metal(model, expected_main, n_toys, seed, init, bounds, config)` — Lower-level entry with custom expected data and init params.
- `ns_inference::batch::is_metal_batch_available()` — Runtime Metal availability check.
- `ns_inference::lbfgs::LbfgsState` — Shared standalone L-BFGS-B state machine (used by both CUDA and Metal batch fitters).

Key Metal exports — **single-model** (feature-gated):
- `ns_inference::gpu_session::MetalGpuSession` — Type alias for `GpuSession<MetalBatchAccelerator>`. Same API as CUDA `GpuSession` but all GPU compute in f32.
  - `metal_session(model) -> Result<MetalGpuSession>` — Create session, upload model + observed data.
  - `is_metal_single_available() -> bool` — Runtime Metal availability check.
  - `nll_grad(params)`, `fit_minimum(model, config)`, `fit_minimum_from_with_bounds(model, init, bounds, config)` — same methods as CUDA GpuSession.

MLE Metal methods (on `MaximumLikelihoodEstimator`, feature-gated):
- `fit_metal(model)` — Full fit with Hessian (Metal minimization + CPU Hessian via FD of Metal gradient).
- `fit_metal_from(model, init)` — Full fit with warm-start + Hessian.
- `fit_minimum_metal(model)` — NLL minimization only (no Hessian).
- `fit_minimum_metal_from(model, init)` — With warm-start.
- `fit_minimum_metal_from_with_bounds(model, init, bounds)` — With custom bounds.

Profile likelihood Metal:
- `ns_inference::profile_likelihood::scan_metal(mle, model, mu_values)` — Metal-accelerated profile scan. Same contract as `scan_gpu` but f32 precision.

Ranking Metal:
- `ns_inference::mle::ranking_metal(mle, model)` — NP ranking with Metal GPU. Nominal fit on CPU (needs Hessian for pull/constraint), per-NP ±1σ refits on Metal with shared session and warm-start.

Metal differentiable (profiled test statistics):
- `ns_inference::metal_differentiable::MetalProfiledDifferentiableSession` — Metal session for profiled q₀/qμ with envelope-theorem gradients.
  - `new(model, signal_sample_name)` — Upload model, require POI defined.
  - `upload_signal(signal)` — Upload signal histogram to GPU.
  - `profiled_q0_and_grad() -> (f64, Vec<f64>)` — Discovery q₀ + gradient w.r.t. signal bins.
  - `profiled_qmu_and_grad(mu_test) -> (f64, Vec<f64>)` — Exclusion qμ + gradient.
  - `batch_profiled_qmu(mu_values) -> Vec<(f64, Vec<f64>)>` — Multiple mu values, session reuse.
  - `signal_n_bins()`, `n_params()`, `parameter_init()`

Model serialization:
- `HistFactoryModel::serialize_for_gpu() -> Result<GpuModelData>` — Converts HistFactory model to flat GPU-friendly buffers (nominal counts, CSR modifiers, constraints). Returns `Err(Validation)` if any NormSys modifier has non-positive hi/lo factors (the GPU polynomial kernel cannot represent the CPU piecewise-linear fallback).
- `MetalModelData::from_gpu_data(&GpuModelData)` — Converts f64 GPU data to f32 Metal data with pre-computed `lgamma` for auxiliary Poisson constraints.

---

## `ns-unbinned`

Event-level (unbinned) likelihood PDFs. Feature-gated: `neural` enables ONNX-backed normalizing flows.

Key exports:
- `ns_unbinned::UnbinnedPdf` — trait for normalized PDFs used in event-level likelihoods.
- `ns_unbinned::EventStore` — columnar event storage (observable columns + optional weights). Key weight methods: `sum_weights()`, `effective_sample_size()`, `weight_summary() -> Option<WeightSummary>`.
- `ns_unbinned::WeightSummary` — weight diagnostics: `n_events`, `sum_weights`, `effective_sample_size` `(Σw)²/Σw²`, `min_weight`, `max_weight`, `mean_weight`, `n_zero`.
- `ns_unbinned::ObservableSpec` — observable metadata (name, bounds).
- `ns_unbinned::UnbinnedModel` — full unbinned model (channels, processes, parameters, constraints). `channel_weight_summaries()` returns per-channel `(name, n_events, sum_weights, ESS)` tuples.
- `ns_unbinned::UnbinnedChannel` — single channel with processes and observed data.
- `ns_unbinned::Parameter`, `ns_unbinned::Constraint` — parameter definitions and constraint types.

### Parametric PDFs

| Struct | Description | Shape params |
|--------|-------------|------|
| `GaussianPdf` | Gaussian (Normal) | `mu`, `sigma` |
| `ExponentialPdf` | Exponential decay | `lambda` |
| `CrystalBallPdf` | Crystal Ball (Gaussian core + power-law tail) | `mu`, `sigma`, `alpha`, `n` |
| `DoubleCrystalBallPdf` | Double-sided Crystal Ball | `mu`, `sigma`, `alpha_lo`, `n_lo`, `alpha_hi`, `n_hi` |
| `ChebyshevPdf` | Chebyshev polynomial | coefficients `c0..cN` |
| `ArgusPdf` | ARGUS function (B-physics) | `m0`, `c`, `p` |
| `VoigtianPdf` | Voigt profile (Gaussian ⊗ Breit-Wigner) | `mu`, `sigma`, `gamma` |
| `SplinePdf` | Monotonic cubic interpolation | knot values |

### Non-parametric PDFs

| Struct | Description |
|--------|-------------|
| `HistogramPdf` | Fixed histogram PDF |
| `KdePdf` | 1-D Kernel Density Estimate |
| `KdeNdPdf` | N-D Kernel Density Estimate (Silverman bandwidth) |
| `MorphingHistogramPdf` | Template morphing with HistoSys (code0/code4p) |
| `MorphingKdePdf` | KDE with weight-based systematics |
| `HorizontalMorphingKdePdf` | KDE with horizontal (shift/scale) systematics |
| `ProductPdf` | Product of independent PDFs: `log p = Σ log pᵢ` |

### Neural PDFs (feature `neural`)

Requires `--features neural`. Uses ONNX Runtime via the `ort` crate.

| Struct | Description |
|--------|-------------|
| `FlowPdf` | ONNX-backed normalizing flow. Loads `flow_manifest.json` + ONNX models. Supports unconditional and conditional (context params) flows. |
| `DcrSurrogate` | Neural DCR surrogate replacing binned template morphing. Wraps conditional `FlowPdf` trained via FAIR-HUC protocol. Drop-in for `MorphingHistogramPdf`. |
| `FlowManifest` | Deserialized `flow_manifest.json` (schema `nextstat_flow_v0`). |

#### `FlowPdf` — ONNX normalizing flow

```rust
use ns_unbinned::{FlowPdf, UnbinnedPdf};
use ns_unbinned::event_store::{EventStore, ObservableSpec};

// Unconditional flow
let flow = FlowPdf::from_manifest("models/flow_manifest.json".as_ref(), &[])?;
assert_eq!(flow.observables(), &["mass"]);

// Conditional flow: context_param_indices maps global param indices → context vector
let flow = FlowPdf::from_manifest("models/flow_manifest.json".as_ref(), &[3, 7])?;
assert_eq!(flow.n_params(), 2); // 2 context parameters

// Evaluate
let mut logp = vec![0.0; events.n_events()];
flow.log_prob_batch(&events, &params, &mut logp)?;

// Sample (requires sample ONNX model in manifest)
let sampled = flow.sample(&params, 1000, &[(-6.0, 6.0)], &mut rng)?;
```

#### `DcrSurrogate` — neural template morphing replacement

```rust
use ns_unbinned::{DcrSurrogate, UnbinnedPdf};

let dcr = DcrSurrogate::from_manifest(
    "models/bkg_dcr/flow_manifest.json".as_ref(),
    &[3, 7],  // systematic param indices in global model
    vec!["jes_alpha".into(), "jer_alpha".into()],
    "background".into(),
)?;

// Same UnbinnedPdf interface as any other PDF
let mut logp = vec![0.0; events.n_events()];
dcr.log_prob_batch(&events, &params, &mut logp)?;
```

#### `FlowManifest` schema (`nextstat_flow_v0`)

```json
{
  "schema_version": "nextstat_flow_v0",
  "flow_type": "nsf",
  "features": 1,
  "context_features": 2,
  "observable_names": ["mass"],
  "context_names": ["jes_alpha", "jer_alpha"],
  "support": [[5000.0, 6000.0]],
  "base_distribution": "standard_normal",
  "models": {
    "log_prob": "log_prob.onnx",
    "sample": "sample.onnx"
  }
}
```

### Normalization

- `ns_unbinned::normalize::QuadratureGrid` — Gauss-Legendre quadrature grid for 1-D numerical normalization.
- `ns_unbinned::normalize::log_normalize_quadrature(pdf, grid, params)` — compute `log ∫ p(x|θ) dx`.
- `ns_unbinned::normalize::QuadratureOrder` — quadrature orders: `N32`, `N64`, `N128`.

Well-trained flows produce ∫p≈1 by construction; quadrature verifies and optionally corrects.

### Spec (YAML)

`PdfSpec` enum for declarative model specification:

```yaml
# Unconditional flow
pdf:
  type: flow
  manifest: models/signal_flow/flow_manifest.json

# Conditional flow
pdf:
  type: conditional_flow
  manifest: models/signal_flow/flow_manifest.json
  context_params: [alpha_syst1, alpha_syst2]

# DCR surrogate (replaces binned template morphing)
pdf:
  type: dcr_surrogate
  manifest: models/bkg_dcr/flow_manifest.json
  systematics: [jes_alpha, jer_alpha]
```

### Feature gates

| Feature | Effect |
|---------|--------|
| `neural` | Enables `ort` + `download-binaries` for ONNX Runtime |
| `neural-cuda` | Adds CUDA Execution Provider |
| `neural-tensorrt` | Adds TensorRT Execution Provider |

---

## `ns-viz`

Lightweight, dependency-free artifacts for plot-friendly output. All structs derive `Serialize` and include a `schema_version` field for forward compatibility. No external plotting dependencies — artifacts are plain JSON-serializable data consumed by Python matplotlib, web frontends, or custom renderers.

Top-level re-exports:

```rust
pub use ns_viz::{
    ClsCurveArtifact, ClsCurvePoint, NsSigmaOrder,
    ProfileCurveArtifact, ProfileCurvePoint,
    RankingArtifact, PullsArtifact, PullEntry,
    CorrArtifact, DistributionsArtifact, DistributionsChannelArtifact, RatioPolicy,
    GammasArtifact, YieldsArtifact, SeparationArtifact,
    SummaryArtifact, PieArtifact, UncertaintyBreakdownArtifact,
};
```

### `cls` — CLs Brazil band curves

- `ClsCurveArtifact` — full CLs exclusion curve with observed and expected limits.
  - `alpha: f64`, `nsigma_order: [i32; 5]` (canonical `[2, 1, 0, -1, -2]`)
  - `points: Vec<ClsCurvePoint>`, `mu_values`, `cls_obs`, `cls_exp: [Vec<f64>; 5]`
  - `obs_limit: f64`, `exp_limits: [f64; 5]`
  - `from_scan(ctx, mle, alpha, scan) -> Result<Self>`
- `ClsCurvePoint` — per-mu-value: `mu`, `cls`, `expected: [f64; 5]`

### `profile` — Profile likelihood scans

- `ProfileCurveArtifact` — profile likelihood scan result.
  - `poi_index`, `mu_hat`, `nll_hat`
  - `points: Vec<ProfileCurvePoint>`, `mu_values`, `q_mu_values`, `twice_delta_nll`
  - `impl From<ProfileLikelihoodScan>`
- `ProfileCurvePoint` — per-point: `mu`, `q_mu`, `nll_mu`, `converged`, `n_iter`

### `ranking` — Nuisance parameter impact

- `RankingArtifact` — NP impact on POI.
  - `names`, `delta_mu_up`, `delta_mu_down`, `pull`, `constraint` (all `Vec`)
  - `impl From<Vec<RankingEntry>>`

### `pulls` — Pull/constraint plots

- `PullsArtifact` — pulls and constraints for all NPs and POI.
  - `entries: Vec<PullEntry>`, plus metadata
  - `pulls_artifact(model, fit, threads) -> Result<PullsArtifact>`
- `PullEntry` — `name`, `kind` ("poi"/"nuisance"), `pull`, `constraint`, `prefit_center/sigma`, `postfit_center/sigma`

### `corr` — Correlation matrix

- `CorrArtifact` — correlation (and optionally covariance) matrix.
  - `parameter_names`, `corr: Vec<Vec<f64>>`, `covariance: Option<Vec<Vec<f64>>>`
  - `corr_artifact(model, fit, threads, include_covariance) -> Result<CorrArtifact>`

### `distributions` — Stacked pre/post-fit histograms

- `DistributionsArtifact` — per-channel stacked distributions with ratio panel.
  - `channels: Vec<DistributionsChannelArtifact>`
  - `distributions_artifact(model, data_by_channel, bin_edges_by_channel, params_prefit, params_postfit, threads, blinded_channels) -> Result<Self>`
- `DistributionsChannelArtifact` — `bin_edges`, `data_y/yerr`, `samples: Vec<DistributionsSampleSeries>`, `total_prefit_y/postfit_y`, `ratio_y/yerr`, `ratio_policy`

### `yields` — Yield tables

- `YieldsArtifact` — per-channel yield summary (pre/post-fit sums).
  - `channels: Vec<YieldsChannel>` with `data`, `samples: Vec<YieldsSample>`, `total_prefit/postfit`
  - `yields_artifact(model, params_prefit, params_postfit, threads, blinded_channels) -> Result<Self>`

### `gammas` — Staterror (Barlow-Beeston) parameters

- `GammasArtifact` — gamma parameter pre/post-fit values and uncertainties.
  - `entries: Vec<GammaEntry>` with `name`, `channel`, `bin_index`, `prefit/postfit_value/sigma`
  - `gammas_artifact(model, fit) -> Result<Self>`

### `separation` — Signal vs background shape comparison

- `SeparationArtifact` — per-channel S/B separation metric.
  - `channels: Vec<SeparationChannelArtifact>` with `signal_shape`, `background_shape`, `separation: f64` (0=identical, 1=fully separated)
  - `separation_artifact(model, params, signal_samples, bin_edges_by_channel) -> Result<Self>`

### `summary` — Multi-fit mu comparison

- `SummaryArtifact` — comparison of POI estimates across analyses/channels.
  - `entries: Vec<SummaryEntry>` with `label`, `mu_hat`, `sigma`, `nll`, `converged`
  - `summary_artifact(poi_name, entries) -> Result<Self>`

### `pie` — Sample composition

- `PieArtifact` — per-channel sample yield fractions.
  - `channels: Vec<PieChannelArtifact>` with `total_yield`, `slices: Vec<PieSlice>` (`sample_name`, `yield_sum`, `fraction`)
  - `pie_artifact(model, params) -> Result<Self>`

### `uncertainty` — Uncertainty breakdown (grouped)

- `UncertaintyBreakdownArtifact` — grouped NP impact breakdown.
  - `grouping_policy`, `groups: Vec<UncertaintyGroup>` (sorted by impact), `total: f64`
  - `UncertaintyGroup` — `name`, `impact` (quadrature sum), `n_parameters`
  - `uncertainty_breakdown_from_ranking(ranking, grouping_policy, threads) -> Result<Self>`

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

## `ns-zstd`

Pure Rust Zstd decoder, forked from ruzstd 0.8.2 with performance optimizations for ROOT file decompression. Used internally by `ns-root` for `.root` files with Zstd compression (ROOT 6.20+).

Key optimizations over upstream ruzstd:
- Fused decode+execute: single-pass (eliminates intermediate `Vec<Sequence>`)
- Exponential match copy: doubles copy size each iteration (log₂ iterations)
- Static Huffman lookup tables, inline FSE/Huffman, pre-sized decode buffers

Performance: ~820 MB/s median throughput (2x original ruzstd).

Public API:
- `ns_zstd::decoding::StreamingDecoder` — incremental Zstd frame decoder.
- `ns_zstd::io::Read`-based adapter for integration with `std::io` pipelines.

This crate is an internal dependency of `ns-root` and not typically used directly.

---

## CLI

The CLI is implemented in `crates/ns-cli` and wraps `ns-inference` surfaces.
See `docs/references/cli.md`.
