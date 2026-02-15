# NextStat Roadmap

Last updated: 2026-02-12 · Current version: **0.9.0** (unreleased)

## Current Status

NextStat is a production-ready statistical inference framework with:

- **HistFactory fitting** — full pyhf JSON + HS3 v0.2 compatibility, all modifier types, asymptotic CLs, profile likelihood, toys.
- **GPU acceleration** — CUDA (NVIDIA, f64) and Metal (Apple Silicon, f32) fused NLL+gradient kernels, batch toy pipeline.
- **Unbinned likelihood** — 10+ PDF types including neural (FlowPdf, DcrSurrogate via ONNX), product, spline, KDE.
- **Bayesian sampling** — NUTS with dual averaging, HMC diagnostics, ArviZ integration.
- **Time series** — Kalman filter/smoother, EM, GARCH(1,1), stochastic volatility.
- **Regression/GLM** — linear, logistic, Poisson, negative binomial, ordinal, LMM, survival (Cox PH).
- **Pharmacometrics** — 1-compartment oral PK (individual + population NLME), LLOQ handling, Laplace approximation.
- **Causal inference** — propensity scores, IPW (ATE/ATT/ATC), balance diagnostics.
- **Visualization** — full TRExFitter plot parity (18 plot types): distributions, pulls, correlations, ranking, CLs, profile, yields, separation, pie, summary, uncertainty, gammas, Brazil band, significance scan, 2D contours, unfolding, morphing, injection.
- **Bindings** — Python (PyO3), R (extendr, experimental), WASM (browser playground).
- **Differentiable analysis** — zero-copy PyTorch integration for NN → signal → profiled CLs → loss.

## v1.0.0 Milestone

Target: **stable public API**. Breaking changes to public Rust/Python APIs are expected before 1.0.

| Area | Status | Remaining |
|------|--------|-----------|
| HistFactory pyhf parity | ✅ Done | Maintenance only |
| Asymptotic CLs / profile | ✅ Done | — |
| GPU CUDA/Metal | ✅ Done | — |
| Unbinned likelihood | ✅ Feature-complete | Stabilize spec YAML |
| Neural PDFs (FlowPdf, DCR) | ✅ Feature-complete | Stabilize ONNX manifest format |
| NUTS / Bayesian | ✅ Done | — |
| Python API surface | 🔶 Mostly stable | Finalize `nextstat.volatility`, `nextstat.causal` |
| R bindings | 🔶 Experimental | Expand beyond `ns_normal_logpdf` / `ns_ols_fit` |
| WASM playground | 🔶 Working | UI polish |
| Documentation | 🔶 Good | API coverage gaps being closed |
| Test coverage reporting | ✅ CI added | cargo-llvm-cov → Codecov workflow |

## Near-term (v0.10–v0.12)

### Parquet / Arrow event-level pipeline ✅
- `EventStore::from_record_batches()` / `from_parquet()` — unbinned fits from Parquet without ROOT.
- Unbinned event-level Parquet schema v1 (`docs/references/unbinned-parquet-schema.md`).
- `nextstat convert` CLI — ROOT TTree → Parquet conversion.
- Fit results / toys / scan points → Parquet export (`ns-translate/arrow/results.rs`).

### High-performance Parquet reads + GPU-direct ✅
- **mmap zero-copy** — `memmap2`-backed reader, no heap allocation for file I/O.
- **Column projection** — read only requested columns (35–50% speedup).
- **Row group predicate pushdown** — prune row groups via min/max statistics (up to 5× on selective filters).
- **Parallel row group decode** — rayon-based, benefits >1M events with many row groups.
- **Direct-to-GPU upload** — `read_parquet_events_soa()` → SoA f64 → `CudaSlice<f64>` (CUDA) or `MTLBuffer` (Metal, f32).
- Criterion benchmarks: 100k events/4 cols decoded in **557 µs**, 10% pushdown filter in **159 µs**.
- See `docs/gpu-contract.md` § Direct-to-GPU Parquet Pipeline for full numbers.

### R bindings expansion ✅
- `nextstat_fit()`, `nextstat_hypotest()`, `nextstat_upper_limit()` for HistFactory workspaces.
- GLM wrappers: `nextstat_glm_logistic()`, `nextstat_glm_poisson()`, `nextstat_glm_negbin()`.
- Time series: `nextstat_kalman()`, `nextstat_garch()`, `nextstat_sv()`.
- CRAN preparation: roxygen2 docs (11 functions), testthat (48 assertions), configure script, getting-started vignette, NEWS.md.
- Tarball self-containment is handled via vendored Rust crates (`make nsr-vendor-sync` / `make nsr-vendor-check`).
- Clean local check path: `make nsr-cran-check-clean` (fails fast if `pandoc` / `checkbashisms` / R suggests are missing).

### CI hardening
- ✅ Test coverage reporting (cargo-llvm-cov → Codecov).
- GPU test CI via self-hosted runners (CUDA).
- Clippy `deny(warnings)` gate in CI.

### ns-zstd audit + encoder optimizations ✅
- Decoder path: zero `unimplemented!()`; all `panic!()` are defensive invariant assertions.
- Encoder path: 20+ hot-path optimizations (FSE state transitions, packed sequence bit writes, hash-chain collision reduction via head tags and u64 reject, common-prefix u128 comparison, fast-path depth-1 search, no-match skip heuristic).
- **`zstd-shim`** crate: transparent backend selection — native `libzstd` on desktop for maximum throughput, pure-Rust `ns-zstd` on WASM and embedded targets.

### Documentation
- Complete API coverage for all public exports.
- ✅ Interactive examples in WASM playground (HistFactory + GLM regression, guided descriptions, UI polish).

### Unbinned toy convergence & robustness ✅

Production-grade toy pipeline for pharma (≥95% convergence required) and HEP.

- **Fail-fast CB/DCB validation** — `alpha > 0`, `n > 1` (strictly) enforced at spec parse time (`spec.rs:533-603`). Benchmark specs fixed: `n_cb` lower bound `1.0 → 1.01`.
- **Warm-start from MLE θ̂** — toy fits initialize from observed-data MLE instead of nominal. Eliminates cold-start divergence.
- **Retry non-converged with jittered init** — automatic re-attempt with perturbed starting point.
- **Hessian skip** — `q̃(μ)` does not require Hessian; auto-enabled under pull guardrails.
- **Granular error classification** — `n_error` split into `n_validation_error + n_computation_error + n_nonconverged`. All 4 error handlers (CPU batch, Metal, CUDA flow, CUDA analytical) updated. JSON backward-compatible (`n_error` = sum).

### CPU-farm shard mode ✅

Embarrassingly-parallel toy distribution across CPU nodes (SLURM, HTCondor, bare SSH).

- **`--shard INDEX/TOTAL`** — `parse_shard()` for `N/M` format, uniform remainder distribution, deterministic seed offset (`seed + shard_start`). Incompatible with `--gpu` (fail-fast).
- **`unbinned-merge-toys`** subcommand — reads shard JSON files, validates generation config match, sorts by `shard_start`, aggregates all metrics, merges per-toy arrays (`poi_hat`, `poi_sigma`, `nll`, `converged`). Outputs `overall_convergence` and `fittable_convergence` with per-shard detail in `shards[]`.

### Cross-vertical benchmark suite ⬜

9 verticals have no published competitor benchmarks. Without speed claims we cannot
make credible sales pitches. All benchmarks run on dedicated CPU server (EPYC 7502P,
64 threads, 128 GB). Competitors installed and verified.

| Benchmark | Competitors | Priority | Effort | Status |
|-----------|-------------|----------|--------|--------|
| Bayesian NUTS ESS/sec | PyMC, NumPyro, CmdStan | **P0** | 1 day | ⬜ |
| GLM fit speed (Linear/Logistic/Poisson/Tweedie) | statsmodels, glum, scikit-learn | **P0** | 0.5 day | ⬜ |
| Econometrics (Panel FE, DiD, IV/2SLS) | statsmodels, linearmodels | **P0** | 0.5 day | ⬜ |
| Time Series (Kalman, GARCH, EGARCH) | statsmodels, arch | P1 | 1 day | ⬜ |
| Pharma NLME (Theophylline/Warfarin) | NONMEM (literature), nlmixr2 | P1 | 1 day | ⬜ |
| Survival (Cox PH, C-index) | scikit-survival | P1 | 0.5 day | ⬜ |
| EVT (GEV/GPD) | pyextremes | P2 | 0.5 day | ⬜ |
| Meta-analysis (FE/RE, I²/Q) | pymare | P2 | 0.5 day | ⬜ |
| Insurance (Chain Ladder, Mack, Bootstrap) | chainladder-python | P2 | 0.5 day | ⬜ |

Output: `docs/benchmarks/<vertical>-speed-2026.md` + JSON artifacts for each.

### Econometrics inference improvements ⬜

Current econometrics vertical has 1-way cluster SE only and naive TWFE DiD —
not credible for applied researchers. These features bring us to fixest-level
capability.

| Feature | Reference | Priority | Effort | Status |
|---------|-----------|----------|--------|--------|
| Two-way clustering (Cameron–Gelbach–Miller multiway) | CGM (2011) JBES | **P0** | 2 days | ⬜ |
| HAC / Newey–West standard errors | Newey–West (1987), Andrews (1991) | **P0** | 1 day | ⬜ |
| Staggered-adoption DiD | Callaway–Sant'Anna (2021) or Sun–Abraham (2021) | **P1** | 3 days | ⬜ |
| Wild cluster bootstrap | Cameron–Gelbach–Miller (2008), Webb 6-point | P1 | 2 days | ⬜ |
| Iterative two-way demeaning | Gaure (2013) lfe, Correia (2016) reghdfe | P1 | 2 days | ⬜ |

## Medium-term (v1.0)

### API stabilization
- Freeze public Rust API for `ns-core`, `ns-inference`, `ns-compute`.
- Semantic versioning compliance for all crates.
- Python API stability guarantee.

### Performance
- ✅ Analytical Jacobian gradients for FlowPdf (replace finite differences).
- ✅ TensorRT execution provider for neural PDFs (FP16, engine cache, dynamic batch profiles).
- ✅ CUDA EP f32 zero-copy NLL reduction (`nll_device_ptr_f32`) — 57× faster than f64 host-upload.
- ✅ Multi-GPU batch toy pipeline for flow PDFs (`CudaFlowBatchNllAccelerator`, `shard_flow_toys`).
- ✅ Fused single-pass CPU NLL kernel + `wide::f64x4` SIMD for Gaussian+Exponential topology — 4–12× speedup on x86 AVX2, 2–5× on ARM NEON. ~770 M events/s NLL throughput.
- ✅ Multi-GPU parallel dispatch (`std::thread::scope`) — both GPUs at 99–100% utilization.
- ✅ Comprehensive GPU benchmark suite (2×H100 SXM, 2×RTX 4090) — see `docs/benchmarks/unbinned-benchmark-suite.md` §3.3.

### Ecosystem
- HistFactory XML v2 (native, no ROOT dependency).
- HS3 roundtrip fidelity improvements.
- Broader TRExFitter config coverage.

### TRExFitter visualization parity ✅

Full plot-type coverage matching all TRExFitter output types (`ns-viz` crate):

| Plot Type | Module | TRExFitter Step |
|-----------|--------|-----------------|
| Pre/post-fit distributions | `distributions` | `d` |
| NP pulls + constraints | `pulls` | `p` |
| Correlation matrix | `corr` | `p` |
| NP ranking (impact on POI) | `ranking` | `r` |
| CLs upper limit (Brazil band) | `cls` | `l` |
| 1D likelihood profile | `profile` | `l`/`s` |
| Yield tables | `yields` | `w`/`d` |
| Separation (S vs B) | `separation` | `d` |
| Pie charts (composition) | `pie` | `d` |
| Summary / multi-fit | `summary` | `m` |
| Uncertainty breakdown | `uncertainty` | `i` |
| Gammas / staterror | `gammas` | `p` |
| Significance scan (p₀ vs parameter) | `significance` | `s` |
| 2D likelihood contours (2-POI) | `contour` | `l` (2D) |
| Unfolding (response matrix + spectrum) | `unfolding` | `x` |
| Morphing validation | `morphing` | config |
| Signal injection / linearity | `injection` | config |

## GPU Acceleration Roadmap

Benchmarks on 2×H100 SXM (Feb 2026) show CPU Rayon dominates all CUDA
workloads due to host-side optimizer roundtrip overhead. Metal wins on
Apple Silicon (unified memory). Full results: `docs/benchmarks/unbinned-benchmark-suite.md` §3.3.

### Current recommendation

| Workload | Use | Flag |
|----------|-----|------|
| Simple PDFs, any CPU | Rayon (default) | `--threads auto` |
| Simple PDFs, Apple Silicon | Metal | `--gpu metal` |
| Batch toys, CUDA | Correct but slower than CPU | `--gpu cuda --gpu-devices 0,1` |

### G1: GPU-native optimizer ✅

Persistent mega-kernel with on-device L-BFGS — eliminates host↔device roundtrip.
Single CUDA block per toy, entire `NLL → grad → param update → convergence check`
cycle runs without returning to host. Supports single- and multi-channel models.

Benchmarks (RTX 4000 SFF Ada, Gauss+Exp):
- 480 ev × 1K toys: lockstep 21.85 s → GPU-native 0.50 s (**44×**), convergence 70% → **100%**
- 10K ev × 200 toys: lockstep 98 s → GPU-native 1.23 s (**80×**), convergence 78% → **100%**
- Numerical parity: Max |ΔNLL| = 4.55e-13 (machine precision)

### G2: CUDA EP for ONNX flows ✅

ONNX Runtime CUDA Execution Provider — evaluate flow PDFs directly on GPU,
output stays device-resident (zero H2D). f32 zero-copy batch NLL path implemented
(`flow_batch_nll_reduce_f32`).

**G2-R1: CLI `--gpu cuda` for flow PDFs** ✅ — `nextstat unbinned-fit-toys --gpu cuda`
now auto-detects flow/conditional_flow/dcr_surrogate PDFs via `spec_has_flow_pdfs()`,
routes to the flow batch CUDA path: CPU toy sampling → CPU logp eval →
`flow_batch_nll_reduce` kernel → lockstep L-BFGS-B. `build_flow_batch_config()`
maps yield specs + Gaussian constraints to `FlowBatchNllConfig`.

**G2-R2: GPU-native optimizer for flows** — cancelled. The G1 mega-kernel evaluates
analytical PDFs inside CUDA kernels; ONNX Runtime cannot be called from device code.
G3 (analytical gradients) provides the practical optimization: 1 forward pass instead
of 2·n+1 finite-difference evaluations.

**G2-R3: Full f64 device path** — cancelled. ONNX models output f32 by design;
f32 NLL parity is 4.43e-10 (sufficient). f64 host-upload path already exists.

### G3: Analytical gradients for flows ✅

Replace finite differences (`2·n_context + 1` ONNX evaluations per optimizer step)
with a single forward pass through a `log_prob_grad` ONNX model that outputs both
`log_prob [batch]` and `∂log_prob/∂context [batch, n_context]`.

Stack: `FlowPdf::log_prob_grad_batch_cuda()` (CUDA EP I/O binding, 2 device-resident
outputs) → `GpuFlowSession::nll_grad_analytical_device_f32()` (launches
`flow_nll_grad_reduce_f32` kernel, assembles full gradient on CPU).

Verified on RTX 4000 SFF Ada: NLL parity f32↔f64 = 4.43e-10, gradient parity
at machine precision. 3 CPU tests + 3 GPU tests.

### G4: >1M events, complex multi-dimensional PDFs ⬜

When single NLL evaluation costs >1 ms (vs ~0.1 ms for simple PDFs), GPU
compute finally dominates launch overhead. Multi-dimensional convolutions,
mixture models with 10+ components. Automatically benefits from G1.

## Pharma / Biotech Roadmap

NextStat's core engine (L-BFGS + analytical gradients + SIMD + Rayon) is directly
applicable to pharmacometrics and clinical biostatistics. CPU-only speed is a
competitive advantage: pharma workloads are small-N (50–5K subjects), latency-bound,
and GPU is irrelevant. See `.internal/pharma-biotech-strategy.md` for full analysis.

### Phase 1: Foundation ⬜ (Q1 2026, ~8–10 weeks)

Feature parity for basic population PK workflows.

- **FOCE/FOCEI** — first-order conditional estimation for NLME (FDA gold standard).
- **2-compartment PK** — IV bolus, IV infusion, oral (CL, V1, Q, V2, KA).
- **Rich dosing regimens** — multi-dose, infusions, dose adjustments, NONMEM EVID/AMT.
- **Proportional + combined error models** — beyond additive Normal for NLME.
- **NONMEM dataset reader** — .csv with EVID/MDV/AMT/DV columns.

Validation target: reproduce Warfarin + Theophylline reference results from NONMEM.

### Phase 2: Credibility ⬜ (Q2 2026)

- **Correlated random effects** — full Omega matrix (Cholesky parameterization).
- **Stepwise covariate modeling** — forward addition + backward elimination with LRT.
- **VPC + GOF diagnostics** — visual predictive checks, CWRES plots.
- **Pharma benchmark suite** — Warfarin, Theophylline, Phenobarbital parity reports.
- **PAGE 2026 poster** — benchmark comparison vs NONMEM/Monolix.

### Phase 3: Market Entry ⬜ (Q3–Q4 2026)

- **SAEM algorithm** — stochastic approximation EM (Monolix's core method).
- **PD models** — Emax, sigmoid Emax, indirect response.
- **Stiff ODE solver** — transit compartments, Michaelis-Menten, TMDD.
- **IQ/OQ/PQ validation protocols** — pharma GxP documentation.
- **R CRAN package** with NLME wrappers.

## Cross-Vertical Statistical Features

Features that unlock multiple market verticals simultaneously (insurance,
epidemiology, quant finance, genomics, reliability engineering, A/B testing).
See `.internal/market-verticals-analysis.md` for full analysis.

### High-ROI cross-vertical features ⬜

| Feature | Effort | Verticals Unlocked |
|---------|--------|--------------------|
| **Kaplan-Meier + log-rank test** | 1 week | Pharma, insurance, epi, reliability |
| **Gamma + Tweedie GLM** | 2 weeks | Insurance (pricing), pharma |
| **Chain Ladder + Mack** (reserving) | 2 weeks | Insurance |
| **Competing risks (Fine-Gray)** | 2 weeks | Pharma, insurance, epi |
| **Meta-analysis** (fixed + random effects) | 2 weeks | Pharma, epi |
| **GEV + GPD** (extreme value) | 2 weeks | Reinsurance, quant finance |
| **GARCH variants** (EGARCH, GJR) | 2 weeks | Quant finance |
| **Sequential testing** (α-spending) | 2 weeks | Pharma (adaptive trials), A/B testing |

## Long-term (post-1.0)

- **Combine-format ingestion** — read CMS Combine datacards natively.
- **Distributed fitting** — multi-node toy generation and fitting.
- **WebGPU backend** — GPU acceleration in the browser via WASM + WebGPU.
- **Julia bindings** — via C FFI or BinaryBuilder.
- **Neural ODE PK** — deep PK models via ONNX, leveraging existing flow infrastructure.
- **PBPK** — physiologically-based PK (multi-organ compartment models).
- **Clinical trial simulation platform** — virtual trials, adaptive designs.
- **Copula models** — Gaussian, t, Clayton, Frank for joint distributions.

## Known Limitations

- **ns-zstd** — forked from ruzstd; decoder is stable and fuzz-tested. Encoder has received 20+ hot-path optimizations but remains less mature than the decoder. The `zstd-shim` crate transparently selects native `libzstd` on desktop and falls back to pure-Rust `ns-zstd` on WASM/embedded.
- **GPU tests** — require physical hardware (NVIDIA for CUDA, Apple Silicon for Metal). Not all GPU paths are exercised in CI.
- **Neural PDFs** — require `--features neural` and ONNX Runtime. Not included in default builds.
- **R bindings** — experimental, minimal API surface. No GPU support yet.
- **WASM** — no GPU, no ONNX, no file I/O. Subset of core inference only.

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development setup and PR guidelines.
Feature requests and bug reports: [GitHub Issues](https://github.com/nextstat/nextstat/issues).
