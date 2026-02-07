# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Phase 2C — Compute Backends & Determinism

**EvalMode: Parity vs Fast (deterministic validation system)**
- `EvalMode` enum (`ns-compute`): process-wide atomic flag controlling summation strategy and backend dispatch.
- **Parity mode**: Kahan compensated summation, Apple Accelerate disabled, threads forced to 1.
  Produces bit-exact reproducible NLL/gradient across runs.
- **Fast mode** (default): naive summation, SIMD/Accelerate/CUDA enabled, Rayon multi-threaded.
- `set_eval_mode()` / `eval_mode()` Rust API; `dispatch_poisson_nll()` routes to Kahan or naive at runtime.
- Kahan summation variants: `poisson_nll_simd_kahan()`, `poisson_nll_simd_sparse_kahan()`,
  `poisson_nll_scalar_kahan()`, `poisson_nll_accelerate_kahan()` — all use f64x4 compensated accumulation.
- **CLI**: `--parity` flag on `fit`, `scan`, `hypotest`, `hypotest-toys` commands.
- **Python**: `nextstat.set_eval_mode("parity")` / `nextstat.get_eval_mode()`.
- **Kahan summation overhead**: <5% vs naive summation *at the same thread count and backend*
  (confirmed across simple/complex/tHu/tttt). Note: full Parity mode also disables Accelerate,
  so Parity-vs-Fast wall time on macOS may differ more than 5%.
- 7-tier tolerance contract vs pyhf NumPy: per-bin ~1e-14 worst-case (well below 1e-12 threshold)
  → toy ensemble 0.05 (`docs/pyhf-parity-contract.md`).
- 45+ new parity tests: gradient parity, per-bin golden, batch toys, eval mode, fast-vs-parity tolerance.

**Apple Accelerate backend (macOS)**
- Feature `accelerate` enables Apple vDSP/vForce for Poisson NLL computation.
- `vvlog()` vectorized ln() for entire arrays; `vDSP_vsubD`, `vDSP_vmulD`, `vDSP_sveD` for arithmetic.
- FFI via `unsafe extern "C"`, linked via `build.rs` (`cargo:rustc-link-lib=framework=Accelerate`).
- Three-layer disable gate: compile-time feature, `EvalMode::Parity`, env var `NEXTSTAT_DISABLE_ACCELERATE=1`.
- `nextstat.has_accelerate() -> bool` Python binding.
- Feature chain: `ns-compute/accelerate` → `ns-translate` → `ns-inference` → `ns-cli`, `ns-py`.

**CUDA GPU batch backend**
- **CUDA fused NLL+Gradient kernel** (`crates/ns-compute/kernels/batch_nll_grad.cu`):
  All 7 modifier types (NormFactor, ShapeSys, ShapeFactor, NormSys/Code4, HistoSys/Code4p, StatError, Lumi)
  with analytical gradient in a single kernel launch. 1 block = 1 toy, threads = bins, shared memory for params.
- **cudarc 0.19 integration** (dynamic loading — binary works without CUDA installed):
  `CudaBatchAccelerator` in `ns-compute::cuda_batch` manages GPU buffers, kernel launches, and H↔D transfers.
- **GPU model serialization**: `HistFactoryModel::serialize_for_gpu() -> GpuModelData` converts HistFactory model
  to flat GPU-friendly buffers (nominal counts, modifier descriptors, per-bin param indices, constraints).
- **Lockstep batch optimizer** (`ns-inference::gpu_batch`): standalone `LbfgsState` L-BFGS-B stepper,
  all toys at same iteration with convergence masking. `fit_toys_batch_gpu()` entry point.
- **Feature chain**: `ns-compute/cuda` → `ns-translate/cuda` → `ns-inference/cuda` → `ns-cli/cuda`, `ns-py/cuda`.
- **PTX build system**: `build.rs` compiles `.cu` kernel via `nvcc --ptx -arch=sm_70`, embedded via `include_str!`.

**CUDA single-model fit path**
- `GpuSession` (`ns-inference::gpu_single`): shared GPU state — serializes model once, reuses for
  multiple fits (profile scans, ranking). `upload_observed_single()` for observed data.
- `GpuObjective`: `ObjectiveFunction` with fused NLL+gradient caching via `RefCell<GpuCache>`.
  argmin calls `cost(x)` then `gradient(x)` with same x → 1 GPU launch per L-BFGS iteration, not 2.
- `fit_gpu()`, `fit_minimum_gpu()`, `fit_minimum_gpu_from()`, `fit_minimum_gpu_from_with_bounds()` MLE methods.
- `compute_hessian_gpu()`: finite differences of GPU gradient (N+1 kernel launches).
- `scan_gpu()`: shared GpuSession across all scan points, warm-start between mu values.
- **CLI**: `--gpu` flag on `fit` and `scan` commands.
- **Python**: `nextstat.fit(model, device="cuda")`, `nextstat.profile_scan(model, ..., device="cuda")`.
- **Python**: `nextstat.has_cuda()`, `nextstat.fit_toys_batch_gpu(model, params, device="cuda")`.

**Metal/f32 GPU types (infrastructure)**
- `MetalModelData` (`ns-compute::metal_types`): f32-precision flat buffers for Apple Metal GPU.
- `MetalAuxPoissonEntry`, `MetalGaussConstraintEntry` with pre-computed `lgamma` (Metal has no `lgamma()`).
- `MetalBackend` stub (feature-gated `--features metal`); kernel implementation pending.

**f32 / Dual32 precision PoC (Metal feasibility study)**
- `impl Scalar for f32` in `ns-ad` — enables `nll_generic::<f32>()` for precision analysis.
- `Dual32` (f32-based dual numbers) in `ns-ad` — enables analytical gradient in f32.
- Validated on tHu (184 params): NLL rel error 3.4e-7, analytical gradient max error 3.2e-4, zero sign flips.
- **Verdict**: f32 analytical gradients are viable for L-BFGS-B on large models (Metal path confirmed feasible).

**Batch toy fitting (CPU)**
- `fit_toys_batch()` in `ns-inference::batch`: Rayon parallel toy fitting with per-thread AD Tape reuse.
- `par_iter().map_init()` pattern: one Tape per Rayon worker thread (~12), not per toy (1000+).
- Skips Hessian/covariance for speed; seed-based reproducibility (`toy_seed = base_seed + toy_index`).
- **Python**: `nextstat.fit_toys_batch(model, params, n_toys=1000, seed=42)`.

**Reverse-mode tape memory optimization**
- `gradient_reverse_reuse(&self, params, &mut tape)`: clears + reuses tape capacity (zero realloc).
- `fit_minimum_histfactory_with_tape()`: takes external `&mut Tape` for caller-controlled lifetime.
- All tape ops (`var`, `constant`, `add`, `mul`, `ln`, etc.) marked `#[inline]`.

#### HistoSys Interpolation Code 0 (Piecewise Linear)
- `HistoSysInterpCode` enum: `Code0` (piecewise linear) and `Code4p` (polynomial+linear extrapolation).
- Default for HistoSys changed to Code 0, matching pyhf default.
- SIMD kernel `histosys_code0_delta_accumulate()` in ns-compute.
- Generic `histosys_code0_delta<T: Scalar>()` for forward-mode AD.
- Tape AD path supports both codes via `interp_code` dispatch.
- 5 new unit tests: scalar parity, zero-delta, Code0/Code4p agreement at α=0 and α=±1.

#### Phase 4 — Native TTree Reader & Ntuple-to-Workspace Pipeline

**ns-root: TTree/TBranch binary reader**
- Memory-mapped file access via `DataSource::Mmap` (memmap2) — no full-file RAM copy for GB+ ntuples.
- Native TTree/TBranch binary deserialization with ROOT class reference system
  (kNewClassTag, kClassMask, kByteCountMask), TObjArray dispatch, TLeaf type detection.
- Basket decompression (zlib/LZ4/ZSTD) with rayon-parallel columnar extraction (`BranchReader`).
- 9 leaf types: `f32`, `f64`, `i32`, `i64`, `u32`, `u64`, `i16`, `i8`, `bool`.
- `RootFile::get_tree()`, `branch_reader()`, `branch_data()` public API.

**ns-root: Expression engine (bytecode-compiled, vectorized)**
- Recursive descent parser for string-based selections, weights, and variable expressions.
- Full grammar: arithmetic (`+`, `-`, `*`, `/`), comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`),
  boolean logic (`&&`, `||`, `!`), ternary (`cond ? a : b`),
  built-in functions (`abs`, `sqrt`, `log`, `exp`, `pow`, `min`, `max`).
- Bytecode compilation: `CompiledExpr::compile()` → stack-based VM for efficient evaluation.
- `eval_row()` for single-row evaluation; `eval_bulk()` for vectorized column-wise evaluation (all rows at once).
- Span-aware error reporting with line/column positions.
- Variables resolved by branch name; bulk evaluation over columnar data.

**ns-root: Histogram filler**
- Single-pass histogram filling with selection cuts, weights, and variable binning.
- `HistogramSpec` (variable, weight, selection as `CompiledExpr`) + `FilledHistogram` output with `sumw2`.
- `fill_histograms()` fills multiple histograms in one pass over the data.
- `From<FilledHistogram> for Histogram` conversion.

**ns-translate: NtupleWorkspaceBuilder**
- High-level fluent builder API: ntuple ROOT files → HistFactory `Workspace`.
- `NtupleWorkspaceBuilder::new().ntuple_path(...).tree_name(...).measurement(...).add_channel(...)`.
- Per-sample modifier support: `NormFactor`, `NormSys`, `WeightSys` (up/down weight expressions),
  `TreeSys` (up/down ROOT files), `StatError` (auto `sqrt(sumw2)`).
- ROOT file caching across samples; Asimov data when no data file specified.
- Produces the same `Workspace` struct as the pyhf JSON and HistFactory XML paths.

**Performance (1000 entries, 7 branches, release build):**

| Operation | NextStat (Rust) | uproot + numpy | Speedup |
|---|---:|---:|---:|
| File open (mmap) | 75 µs | 215 µs | ~3x |
| TTree metadata parse | 50 µs | 1,400 µs | ~28x |
| Read 1 branch | 65 µs | 675 µs | ~10x |
| Read all 7 branches | 200 µs | 1,300 µs | ~6.5x |
| Selection eval | 15 µs | 26 µs | ~1.7x |
| Histogram fill | 28 µs | 96 µs | ~3.4x |
| **Total pipeline** | **~430 µs** | **~3,700 µs** | **~8.5x** |

**Bug fixes:**
- HistFactory XML parser: strip `<!DOCTYPE>` declarations before parsing (roxmltree rejects DTD by default).

#### Systematics Preprocessing: Smoothing + Pruning

**Smoothing (`nextstat.analysis.preprocess.smooth`)**
- 353QH,twice algorithm (ROOT `TH1::Smooth` equivalent): running median(3→5→3) + Hanning + residual smooth, applied twice.
- Gaussian kernel smoothing with configurable bandwidth (`sigma` in bin units).
- MAXVARIATION cap: `|smoothed_delta[i]| <= max(|original_delta|)`.
- Top-level `smooth_variation(nominal, up, down)` API working on deltas.
- `SmoothHistoSysStep` pipeline step for workspace-wide smoothing.

**Pruning (`nextstat.analysis.preprocess.prune`)**
- Shape pruning: prune if `max |delta/nominal| < threshold` for both up/down.
- Norm pruning: prune if `|hi - 1| < threshold` and `|lo - 1| < threshold`.
- Overall pruning: decompose histosys into norm (integral ratio) + shape (residual), prune if both negligible.
- `PruneSystematicsStep` pipeline step with two-pass approach (collect decisions, then remove modifiers).
- `PruneDecision` dataclass with human-readable `reason` string for audit.

**Recommended pipeline order:** `hygiene → symmetrize → smooth → prune`.

#### Phase 4.1 — TRExFitter Interop (Config Import + Analysis Spec)

**ns-cli: TRExFitter config importers (NTUP subset)**
- `nextstat import trex-config` imports a TRExFitter-style config subset (`ReadFrom: NTUP`) into a pyhf JSON `Workspace`.
- `nextstat build-hists` runs the NTUP pipeline and writes `workspace.json` into `--out-dir` (deterministic for the same inputs).
- `nextstat trex import-config` converts a TRExFitter `.config` file into an analysis spec v0 YAML (`inputs.mode=trex_config_yaml`) and a mapping report (`*.mapping.json`).
- `--base-dir` controls relative-path resolution for `File:` entries (default: config file directory).
- Runnable minimal example: `docs/examples/trex_config_ntup_minimal.txt`.

**TREx Analysis Spec v0 (YAML + JSON Schema)**
- Spec + examples: `docs/specs/trex/analysis_spec_v0.yaml`, `docs/specs/trex/examples/*.yaml`.
- JSON Schema (IDE autocomplete): `docs/schemas/trex/analysis_spec_v0.schema.json`.
- Validator + runner: `scripts/trex/validate_analysis_spec.py`, `scripts/trex/run_analysis_spec.py` (supports `--dry-run`).
- `nextstat run <spec.yaml>` supports analysis spec v0 orchestration (import/fit/scan/report).
- Numbers-first baselines for spec runs: `tests/record_trex_analysis_spec_baseline.py` and `tests/compare_trex_analysis_spec_with_latest_baseline.py`.
- Tutorial: `docs/tutorials/trex-analysis-spec.md`.

#### Report System (Numbers-First Artifacts + Publication-Ready Rendering)

- `nextstat report` master command: generates all numeric artifacts from a workspace + optional fit.
- **Artifact outputs** (versioned JSON schemas under `docs/schemas/trex/`):
  - `distributions.json` — prefit/postfit expected yields per sample per region, bin edges, data, Garwood errors, ratio.
  - `pulls.json` — nuisance parameter pulls and constraints (postfit sigma / prefit sigma).
  - `corr.json` — correlation matrix derived from inverse Hessian (optional raw covariance via `--include-covariance`).
  - `yields.json` — per-region per-sample prefit/postfit yield tables; also `yields.csv` and `yields.tex`.
  - `uncertainty.json` — ranking-based uncertainty breakdown (skippable via `--skip-uncertainty`).
- `nextstat viz distributions`, `viz pulls`, `viz corr`, `viz ranking` subcommands for individual artifacts.
- **Python rendering**: `python -m nextstat.report render` → multi-page PDF + per-plot SVGs (requires `matplotlib`).
- `--render` flag on `nextstat report` to invoke rendering automatically.
- `--uncertainty-grouping` policy (`prefix_1`, etc.) for systematic grouping in ranking.
- `--deterministic` flag for stable JSON key ordering.
- If `--fit` is omitted, `nextstat report` runs an MLE fit and writes `fit.json` into `--out-dir`.

#### Patchset Support (HEPData Interop)

- `nextstat import patchset --workspace BkgOnly.json --patchset patchset.json [--patch-name ...]`:
  applies pyhf PatchSet (HEPData signal patch format) to a base workspace.
- Python: `nextstat.apply_patchset(workspace_json_str, patchset_json_str, patch_name=None) -> str`.

#### Phase 9 — Pharma & Social Sciences Domain Packs

**Pack A: Survival Analysis**
- Parametric survival models with right-censoring (`ns-inference::survival`):
  Exponential (`log_rate`), Weibull (`log_k`, `log_lambda`), LogNormal AFT (`mu`, `log_sigma`).
- Cox Proportional Hazards model with Efron/Breslow ties handling (`CoxPhModel`).
- **High-level Python API** (`nextstat.survival`): callable builders with `.fit(...)` method.
  - `nextstat.survival.exponential.fit(times, events, x)` → `ParametricSurvivalFit`
  - `nextstat.survival.weibull.fit(times, events, x)` → `ParametricSurvivalFit`
  - `nextstat.survival.lognormal_aft.fit(times, events, x)` → `ParametricSurvivalFit`
  - `nextstat.survival.cox_ph.fit(times, events, x, ties="efron", robust=True)` → `CoxPhFit`
- **Cox PH robust SE + CI**: sandwich covariance estimator (I⁻¹ B I⁻¹), tie-aware score residuals.
  - `CoxPhFit.robust_se`, `.robust_cov`, `.confint(robust=True)`, `.hazard_ratio_confint(robust=True)`.
  - Hessian via finite-diff of `grad_nll`; B via sum of outer-product score residuals (Breslow/Efron).
- All survival models integrate with `nextstat.fit()` and `nextstat.sample()`.

**Pack B: Longitudinal / Mixed-Effects**
- Linear Mixed Model with analytic marginal likelihood (`LmmMarginalModel`):
  random intercept, or random intercept + one random slope (diagonal RE covariance).
- Laplace approximation utilities for approximate posteriors (`ns-inference::laplace`).
- Python surface: `nextstat.LmmMarginalModel(...)` with `fit()` / `sample()` integration.
- LMM parameter recovery smoke tests and tutorial (`docs/tutorials/phase-9-lmm.md`).

**Pack C: Ordinal Models**
- Ordered logit/probit models (`OrderedLogitModel`, `OrderedProbitModel`), with a stable cutpoint parameterization.
- Python surface: `nextstat.ordinal.ordered_logit.fit(...)`, `nextstat.ordinal.ordered_probit.fit(...)`.

**Pack D: Missing Data**
- Explicit missing-data policies for `(X, y)` (`drop_rows`, `impute_mean`) via `nextstat.missing.apply_policy(...)`.

**Pack E: Causal Convenience Helpers**
- Propensity score estimation, IPW weights, and overlap/balance diagnostics via `nextstat.causal.propensity`.

#### Phase 11 — Applied Statistics API (Formula, Summary, Robust SE, sklearn)

**Formula + design matrices**
- Minimal formula parsing + deterministic design matrices: `nextstat.formula.parse_formula()` and `design_matrices()`.
- Tabular adapter: `nextstat.formula.to_columnar()` supports dict-of-columns, list-of-dicts, and pandas DataFrame (if installed).
- Categoricals: explicit `categorical=[...]` one-hot encoding with deterministic column naming/order.

**Regression surfaces**
- `from_formula` wrappers for GLM fits: `nextstat.glm.linear.from_formula`, `.logistic.from_formula`, `.poisson.from_formula`, `.negbin.from_formula`.
- Formula helpers for hierarchical random-intercept builders: `nextstat.hier.linear_random_intercept_from_formula`, `.logistic_random_intercept_from_formula`, `.poisson_random_intercept_from_formula`.

**Summaries + robust covariance**
- Dependency-light Wald summaries (coef/SE/CI/p-values): `nextstat.summary.fit_summary(...)` + `summary_to_str(...)`.
- Robust covariance estimators for OLS (HC0–HC3, 1-way cluster) + baseline GLM sandwich: `nextstat.robust`.

**Interoperability**
- Optional scikit-learn adapters: `nextstat.sklearn.NextStatLinearRegression`, `NextStatLogisticRegression`, `NextStatPoissonRegressor`.

#### Phase 12 — Econometrics & Causal Inference Pack

**Panel + DiD + IV**
- Panel FE (within estimator) baseline + 1-way cluster SE: `nextstat.econometrics.panel_fe_fit(...)` and `panel_fe_from_formula(...)`.
- DiD TWFE + event-study helpers: `nextstat.econometrics.did_twfe_fit/from_formula`, `event_study_twfe_fit/from_formula`.
- IV / 2SLS baseline with weak-IV diagnostics (first-stage F, partial R²): `nextstat.econometrics.iv_2sls_fit/from_formula`
  with covariance options `homoskedastic|hc1|cluster`.

**Doubly robust**
- AIPW baseline for ATE/ATT + E-value helper: `nextstat.causal.aipw`.

#### Phase 13 — Pharmacometrics Pack (ODE + PK/NLME)

**ODE baseline**
- Deterministic RK4 integrator for linear systems `dy/dt = A y`: `nextstat.rk4_linear(...)` and `nextstat.ode.rk4_linear(...)`.

**PK / NLME**
- One-compartment oral PK model (`OneCompartmentOralPkModel`) with LLOQ handling (censoring).
- NLME extension with per-subject random effects (`OneCompartmentOralPkNlmeModel`).

#### Infrastructure
- Release pipeline hardening: test gate, version validation, multi-arch wheels, sdist,
  CHANGELOG-based release notes, optional PyPI publish.
- Structured logging with `tracing` and `--log-level` CLI flag.
- `ns-cli`: reproducible run bundle (`--bundle` flag).
- Apex2 baseline recorder (`tests/record_baseline.py`) with full environment fingerprint
  (hostname, Python/pyhf/nextstat/numpy versions, git commit, CPU, platform).
- Apex2 baseline comparison (`tests/compare_with_latest_baseline.py`) with per-type manifests
  and `--require-same-host` strict mode.
- NUTS quality report (`tests/apex2_nuts_quality_report.py`) integrated into master report.
- Bias/pulls report hardened: per-parameter output, `--params poi|all`, `--min-used-abs/frac`,
  try/except around toy-loop fits, `skipped` status on insufficient valid toys.
- Slow test ergonomics: `sbc` pytest marker, `NS_RUN_SBC=1` gate, reduced default toy counts.
- **CI parity gate** (`.github/workflows/pyhf-parity.yml`): runs full parity test suite on push/PR
  to main/develop; uploads golden reports as artifacts.
- **TREx baseline refresh** (`.github/workflows/trex-baseline-refresh.yml`): self-hosted runner
  workflow for recording TREx parity baselines (manual dispatch + nightly schedule).
- **HEPData workspace tests** (`tests/python/test_hepdata_workspaces.py`): opt-in NLL parity checks
  on real HEPData pyhf workspaces (requires manual download via `tests/hepdata/fetch_workspaces.py`).
- **TREx baseline recorder** (`tests/record_trex_baseline.py`): captures fit + expected_data baseline
  from a real TRExFitter/HistFactory export directory; compare via `tests/compare_trex_baseline_files.py`.
- `fit()` now supports `init_pars=` for warm-start MLE (Rust `fit_from()` + Python binding).

#### Documentation
- `docs/pyhf-parity-contract.md`: 7-tier tolerance hierarchy with rationale, architecture diagram,
  batch toys section, performance characteristics, CI integration guide.
- `docs/references/trex_replacement_parity_contract.md`: TREx replacement parity contract (v0)
  with EvalMode, reporting schemas, determinism policy, and baseline management.
- `docs/references/optimizer-convergence.md`: L-BFGS-B vs SLSQP analysis on large models (184–249 params).
- `docs/plans/metal-batch-gpu.md`: Metal f32 GPU feasibility study with PoC results.

#### Visualization
- `plot_cls_curve()` now respects `nsigma_order` + `cls_exp` dynamically.
- `plot_brazil_limits()` for observed/expected upper-limit Brazil band.
- `plot_profile_curve()` supports `y="q_mu"` (default) and `y="twice_delta_nll"`.

### Fixed
- Optimizer early-stop bug: removed `target_cost(0.0)` that broke models with NLL < 0.
- `kalman_simulate()` now supports `init="sample|mean"` and `x0=...` for custom start.

## [0.1.0] - 2026-02-05

### Added

#### Phase 1 — Core Engine
- `ns-core`: workspace data model (Channel, Sample, Parameter, Modifier, Measurement).
- `ns-core`: Poisson + modifier evaluation (histosys, normsys, shapesys, staterror, lumi).
- `ns-core`: negative log-likelihood (NLL) computation with constraint terms.
- `ns-translate`: pyhf JSON workspace parser with full HistFactory support.
- `ns-compute`: SIMD-accelerated Poisson NLL via `wide::f64x4`.
- `ns-py`: Python bindings (`nextstat` package) with `Model`, `FitResult` classes.
- `ns-cli`: `nextstat` CLI with `fit`, `hypotest`, `upper-limit`, `scan`, `version` commands.
- CI workflows for Rust (test + clippy) and Python (maturin build + pytest).
- GitHub release workflow with wheel and CLI binary artifacts.

#### Phase 2A — SIMD & PreparedModel
- `PreparedModel` with cached observed data, ln-factorials, and observation masks.
- SIMD Poisson NLL path (f64x4) with lane-by-lane scalar `ln()` for accuracy.
- `fit_minimum()` extracted for fast repeated minimizations (profile, toys).

#### Phase 2B — Automatic Differentiation
- `ns-ad`: dual-number forward-mode AD (`Dual<f64>`).
- `ns-ad`: reverse-mode tape AD (`Tape`, `TapeVar`).
- Generic `nll_generic<T: Scalar>` path supporting both AD backends.

#### Phase 3.1 — Frequentist Inference
- `ns-inference`: MLE fitting via `argmin` (L-BFGS-B with Hessian uncertainties).
- `ns-inference`: asymptotic CLs hypothesis testing (q-tilde test statistic).
- `ns-inference`: profile likelihood scan over POI values.
- `ns-inference`: CLs upper limits via bisection and linear scan.
- `ns-inference`: expected CLs band (Brazil band) computation.
- `ns-inference`: batch MLE fitting (`fit_batch`), toy studies (`fit_toys`), NP ranking.
- `ns-py`: `mle.fit_batch()`, `mle.fit_toys()`, `mle.ranking()` Python bindings.

#### Phase 3.2 — Bayesian Sampling
- `ns-inference`: No-U-Turn Sampler (NUTS) with dual averaging.
- `ns-inference`: HMC diagnostics (divergences, tree depth, step size).
- Rank-normalized folded R-hat + improved ESS (Geyer IMS + variogram).
- E-BFMI energy diagnostic.
- Overdispersed chain initialization option.
- Non-slow sampling quality summary.
- `ns-py`: `sample()` Python binding returning ArviZ-compatible dict.

#### Phase 3.4 — Visualization Artifacts
- `ns-viz`: `ProfileCurveArtifact` (q_mu vs mu, with sigma lines and best-fit).
- `ns-viz`: `ClsCurveArtifact` (observed + expected CLs with Brazil band).
- `ns-cli`: `viz profile` and `viz cls` subcommands.
- `ns-py`: `viz_profile_curve()` and `viz_cls_curve()` Python helpers.

#### Phase 5 — Universal Model API
- `ns-core`: `LogDensityModel` trait for generic inference across all model types.
- `ns-prob` crate: probability distribution library (Normal, StudentT, Bernoulli, Binomial,
  Poisson, NegativeBinomial, Gamma, Exponential, Weibull, Beta).
- `ns-prob`: bijector/transform layer (Identity, Exp, Softplus, Sigmoid, Affine).
- `ns-inference`: `ComposedGlmModel` builder for composing GLM-style models.
- `ns-inference`: generic posterior, HMC, and NUTS over any `LogDensityModel`.
- `ns-inference`: parameter transforms with log-Jacobian for constrained parameters.
- `ns-py`: `GaussianMeanModel` for non-HEP Bayesian estimation.
- `ns-py`: `GlmSpec` data adapter (`nextstat.data`) for design matrices and responses.

#### Phase 6 — Regression & GLM Pack
- Linear, logistic, Poisson, and negative binomial regression models (`ns-inference::regression`).
- Ridge regression via Normal prior on coefficients (MAP/L2).
- Separation detection and stabilization warnings for logistic regression.
- Exposure/offset support for Poisson regression.
- Python GLM surface: `nextstat.glm.linear`, `.logistic`, `.poisson`, `.negbin`.
- Cross-validation utilities: `kfold_indices()`, `cross_val_score()` (`nextstat.glm.cv`).
- Metrics: RMSE, log-loss, Poisson deviance (`nextstat.glm.metrics`).
- Golden regression test fixtures with statsmodels parity (`tests/fixtures/regression/`).
- Criterion benchmarks: `regression_benchmark.rs`, `glm_fit_predict_benchmark.rs`.

#### Phase 7 — Hierarchical / Multilevel Models
- Random intercept support in `ComposedGlmModel` (Normal prior, partial pooling).
- Random slopes / varying coefficients.
- Non-centered parameterization for improved NUTS mixing.
- Correlated random effects via LKJ prior + Cholesky decomposition.
- Python surface: `nextstat.hier` with `linear_random_intercept()`, `linear_random_slope()`,
  `logistic_random_intercept()`, `logistic_correlated_intercept_slope()`, etc.
- Posterior Predictive Checks: `nextstat.ppc.ppc_glm_from_sample()` with replicate generation.
- Criterion benchmark: `hier_benchmark.rs`.

#### Phase 8 — Time Series & State Space Models
- Linear-Gaussian Kalman filter and RTS smoother (`ns-inference::timeseries`).
- EM parameter estimation for state-space models (Q, R, optionally F, H).
- Multi-step-ahead Kalman forecasting with prediction intervals.
- Standard model builders: local-level, local-trend.
- AR(1) parameter transform (softplus + bounds).
- Missing observation handling (skip update, correct likelihood).
- State simulation from model with controllable seed.
- Python surface: `nextstat.timeseries` with `kalman_filter()`, `kalman_smooth()`,
  `kalman_em()`, `kalman_fit()`, `kalman_forecast()`, `kalman_simulate()`.
- Visualization: `plot_kalman_states()`, `plot_forecast_bands()`.
- Criterion benchmarks: `kalman_benchmark.rs`, `em_benchmark.rs`.

#### Apex2 Validation System
- Master report aggregator (`tests/apex2_master_report.py`) with exit code policy.
- pyhf NLL/expected_data parity runner.
- P6 GLM benchmark runner with baseline comparison and slowdown detection.
- Bias/pulls regression runner (NextStat vs pyhf).
- SBC posterior calibration runner (NUTS).
- NUTS quality runner (divergence/R-hat/ESS/E-BFMI gates).
- ROOT/TRExFitter parity runner (optional).
- Nightly slow CI workflow (`.github/workflows/apex2-nightly-slow.yml`).
