# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Phase 4 — Native TTree Reader & Ntuple-to-Workspace Pipeline

**ns-root: TTree/TBranch binary reader**
- Memory-mapped file access via `DataSource::Mmap` (memmap2) — no full-file RAM copy for GB+ ntuples.
- Native TTree/TBranch binary deserialization with ROOT class reference system
  (kNewClassTag, kClassMask, kByteCountMask), TObjArray dispatch, TLeaf type detection.
- Basket decompression (zlib/LZ4/ZSTD) with rayon-parallel columnar extraction (`BranchReader`).
- 9 leaf types: `f32`, `f64`, `i32`, `i64`, `u32`, `u64`, `i16`, `i8`, `bool`.
- `RootFile::get_tree()`, `branch_reader()`, `branch_data()` public API.

**ns-root: Expression engine**
- Recursive descent parser for string-based selections, weights, and variable expressions.
- Full grammar: arithmetic (`+`, `-`, `*`, `/`), comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`),
  boolean logic (`&&`, `||`, `!`), built-in functions (`abs`, `sqrt`, `log`, `exp`, `pow`, `min`, `max`).
- `CompiledExpr::compile()` → `eval_row()` / `eval_bulk()` API.
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

#### Phase 9 — Pharma & Social Sciences Domain Packs

**Pack A: Survival Analysis**
- Parametric survival models with right-censoring (`ns-inference::survival`):
  Exponential (`log_rate`), Weibull (`log_k`, `log_lambda`), LogNormal AFT (`mu`, `log_sigma`).
- Cox Proportional Hazards model with Efron/Breslow ties handling (`CoxPhModel`).
- Python surface: `nextstat.survival.exponential()`, `.weibull()`, `.lognormal_aft()`, `.cox_ph()`.
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
