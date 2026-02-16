# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.6] — 2026-02-17

### Added

- **Unified Python API** — merged model-type and device variants into single functions with runtime dispatch: `ranking()` (was 3 functions), `hypotest()`/`hypotest_toys()` (was 4), `profile_scan()` (was 3), `fit_toys()` (was 4), `upper_limit()` (was 2). All accept `device="cpu"|"cuda"|"metal"` and dispatch on `HistFactoryModel` vs `UnbinnedModel` automatically. Old `unbinned_*`, `*_gpu`, `*_batch_gpu` variants removed.
- **TypedDict return types** — added ~25 structured `TypedDict` definitions (`RankingEntry`, `ProfileScanResult`, `HypotestToysMetaResult`, `PanelFeResult`, `KalmanFilterResult`, `MetaAnalysisResult`, `ClsCurveResult`, etc.) replacing opaque `Dict[str, Any]` returns. IDE autocomplete now works for all inference functions.
- **Internal API conventions document** — `docs/internal/api-conventions.md` defining rules for parameter ordering, device handling, model-type dispatch, return types, and new function checklist.
- **`profile_scan(return_curve=True)`** — merges former `profile_curve()` into `profile_scan()` with plot-friendly arrays (`mu_values`, `q_mu_values`, `twice_delta_nll`).
- **`upper_limit(method="root")`** — merges former `upper_limits_root()` into `upper_limit()` with `method="bisect"|"root"`.

- **LAPS Metal backend** — GPU-accelerated MAMS sampler on Apple Silicon (M1–M5) via Metal Shading Language. Built-in models (`StdNormal`, `EightSchools`, `NealFunnel`, `NealFunnelNcp`, `NealFunnelRiemannian`, `GlmLogistic`) with f32 compute, fused multi-step kernel, and SIMD-group cooperative kernel for data-heavy GLM. Automatic fallback: CUDA (f64) > Metal (f32) > error.
- **LAPS windowed mass adaptation** — warmup Phase 2+3 now uses Stan-style doubling windows (default 3) for `inv_mass` estimation. Each window resets Welford statistics and dual averaging, improving convergence on multi-scale models. Configurable via `n_mass_windows` in `LapsConfig` (set to 1 for single-pass legacy behavior).
- **LAPS Neal Funnel NCP** — non-centered parametrization of Neal's funnel (`model="neal_funnel_ncp"`). Removes position-dependent geometry, allowing MAMS to sample efficiently. R-hat < 1.02, ESS/s > 40,000 on M5 GPU (4096 chains).
- **LAPS Riemannian MAMS for Neal Funnel** — experimental `model="neal_funnel_riemannian"` with hybrid position-dependent metric: x-components use Fisher metric `exp(v/2)` scaling (Riemannian), v-component uses learned diagonal mass (standard). 19x ESS/s improvement over centered parametrization for x-dimensions. Known limitation: systematic v-bias (v_mean ≈ 3.5 vs expected 0) due to isokinetic dynamics — use NCP for unbiased sampling. Available on both Metal (f32) and CUDA (f64) backends.

### Fixed

- **`panel_fe()` parameter order** — changed from `(entity_ids, x, y, p)` to `(y, x, entity_ids, p)` to match econometrics module convention (`y` first).
- **BCa confidence interval engine** — reusable bootstrap CI utilities in `ns-inference` (`percentile` + `bca`) with diagnostics (`z0`, acceleration, adjusted alphas, sample counts).
- **HEP toy-summary CI controls** — `nextstat unbinned-fit-toys` now supports opt-in summary CI computation for `summary.mean`:
  - `--summary-ci-method percentile|bca`
  - `--summary-ci-level`
  - `--summary-ci-bootstrap`
  Output includes `summary.mean_ci` with requested/effective method and fallback diagnostics.
- **Churn bootstrap CI method selection** — `nextstat churn bootstrap-hr` now supports:
  - `--ci-method percentile|bca` (default `percentile`)
  - `--n-jackknife` for BCa acceleration estimation.
  Output includes method metadata and per-coefficient diagnostics (`ci_diagnostics`) with fallback reason.
- **Python churn parity for CI methods** — `nextstat.churn_bootstrap_hr(...)` now accepts `ci_method` and `n_jackknife` and returns per-coefficient effective method/diagnostics.
- **BCa benchmark harnesses**:
  - `scripts/benchmarks/bench_unbinned_summary_ci.py` (HEP `unbinned-fit-toys`)
  - `scripts/benchmarks/bench_churn_bootstrap_ci_methods.py` (churn `bootstrap-hr`)
  - runbook: `docs/benchmarks/bca-hep-churn-ci-methods.md`.
- **BCa CI release-gate automation**:
  - `scripts/benchmarks/check_bca_ci_gates.py` validates HEP/churn overhead, fallback-rate, and effective-BCa-rate thresholds.
  - `.github/workflows/apex2-nightly-slow.yml` now runs BCa benchmark matrix + gate check and publishes gate artifacts.
- **Benchmark artifact retention policy** — for `bench_results/*` BCa snapshots, repository keeps only `summary.json`; generated intermediates (`raw_runs.json`, `summary.md`, `churn_data.json`, `unbinned_spec_summary_ci.json`) are ignored.
- **Benchmark retention policy guard test** — added `tests/python/test_bca_bench_results_policy.py` to fail CI if live tracked files under `bench_results/*` include anything except `summary.json`.
- **Controlled skew calibration harness (HEP + churn)** — added `scripts/benchmarks/bench_bca_skew_calibration.py` for BCa vs percentile coverage/width calibration in asymmetric scenarios (HEP boundary POI and churn heavy-censoring small-sample regime), with runbook updates in `docs/benchmarks/bca-hep-churn-ci-methods.md`.
- **Skew calibration baseline snapshot (2026-02-16)** — published matrix summary artifact at `bench_results/bca_skew_calibration_2026-02-16/summary.json`.
- **Apex2 nightly skew-calibration run** — `.github/workflows/apex2-nightly-slow.yml` now executes `bench_bca_skew_calibration.py` and uploads informational skew-scenario artifacts.
- **Benchmark binary selection hardening** — BCa benchmark scripts now accept `--nextstat-bin` and prioritize `release` artifacts in auto-discovery to avoid accidental stale-binary runs on remote stands.
- **HEP BCa diagnostic enrichment** — HEP benchmark summaries now include CI-center bias metrics (`median_center_minus_poi_true`, `mean_center_minus_poi_true`) for correct interpretation of inclusion rates.
- **HEP dataset-level BCa calibration benchmark (BCA-14)** — added `scripts/benchmarks/bench_hep_dataset_bootstrap_ci.py` with per-run observed data regeneration, bootstrap CI comparison (percentile vs BCa), and baseline artifact `bench_results/hep_dataset_bootstrap_ci_2026-02-16/summary.json`. Added smoke regression `tests/python/test_bca_hep_dataset_bootstrap_ci_smoke.py`.
- **BCA-14 nextstat-bench full-matrix snapshot** — provisioned benchmark runtime on `nextstat-bench` (ROOT + Python deps), built native Linux `nextstat`, and published adult-run summary artifact `bench_results/hep_dataset_bootstrap_ci_nextstat_bench_2026-02-16_full/summary.json` (`runs=32`, `n_bootstrap=1500`, `threads=56`). Coverage is `0.96875` for both percentile and BCa in both HEP scenarios, with `fallback_count=0`.
- **BCA-14 nextstat-bench long-run calibration snapshot** — published extended matrix artifact `bench_results/hep_dataset_bootstrap_ci_nextstat_bench_2026-02-16_longrun/summary.json` (`runs=128`, `n_bootstrap=1500`, `threads=56`). Coverage remained high (`0.97656` mid-POI for both methods; boundary scenario `0.96875` percentile vs `0.97656` BCa), BCa remained fallback-free and slightly narrower.
- **Churn benchmark coverage diagnostics + dataset-level mode** — `scripts/benchmarks/bench_churn_bootstrap_ci_methods.py` now supports:
  - `--use-default-truth` (coverage fields in summary: `coverage_vs_true_hr`, `coverage_hits`, `coverage_total`)
  - `--regenerate-data-per-run` (frequentist dataset-level calibration)
  - generator controls (`--n-cohorts`, `--max-time`, `--treatment-fraction`)
  - per-coefficient aggregates (`per_coefficient` coverage/mean point/mean width).
- **Churn nextstat-bench dataset-level calibration snapshots** — published:
  - `bench_results/churn_bootstrap_ci_methods_nextstat_bench_2026-02-16_datasetlevel_treat30/summary.json`
  - `bench_results/churn_bootstrap_ci_methods_nextstat_bench_2026-02-16_datasetlevel_treat00/summary.json`
  using `runs=128`, `n_customers=4000`, `n_bootstrap=800`, `n_jackknife=240`. Coverage is near nominal in both methods (treat=0.3: percentile `0.9453`, BCa `0.9512`; treat=0.0: both `0.9297`), all with zero fallbacks.
- **Apex2 nightly churn benchmark switched to dataset-level calibration default (BCA-20)** — `.github/workflows/apex2-nightly-slow.yml` now runs churn BCa benchmark with `--regenerate-data-per-run --use-default-truth` and pinned generator controls (`--n-cohorts 48 --max-time 12 --treatment-fraction 0.3`) so nightly coverage reflects dataset-level calibration instead of fixed-dataset conditional diagnostics.
- **Rootless HEP dataset generator path (BCA-15)** — `scripts/benchmarks/bench_hep_dataset_bootstrap_ci.py` now supports `--root-writer auto|uproot|root-cli` and defaults to rootless `uproot` when available (with `root` CLI fallback). Smoke coverage in `tests/python/test_bca_hep_dataset_bootstrap_ci_smoke.py` now accepts either backend and skips only when both are unavailable.
- **Single-artifact visualization renderer (`nextstat viz render`)** — added direct rendering of JSON viz artifacts to image/document outputs (`pulls`, `corr`, `ranking`) without full report generation. Supports title/DPI options and correlation filters (`--corr-include`, `--corr-exclude`, `--corr-top-n`) via Python `nextstat.viz_render` entrypoint.

### Fixed

- **HS3/pyhf CLI scope clarity for workspace utilities** — corrected command help/docs and added explicit fail-fast diagnostics for pyhf-only commands (`audit`, `report`, `viz distributions`, `export histfactory`, `preprocess smooth/prune`) when HS3 input is provided. `viz separation` now reports a clear HS3-specific requirement to pass `--signal-samples` explicitly.
- **Arrow/Parquet `from_arrow` large-offset compatibility** — fixed ingestion for `Utf8/LargeUtf8` and `List<Float64>/LargeList<Float64>` in core paths, so Arrow tables from Polars/DuckDB are accepted without pre-normalization. Added Rust and Python regression coverage (including Polars/DuckDB round-trips) and removed the temporary normalization workaround from the Route C quick-start flow.

## [0.9.5] — 2026-02-15

### Fixed

- **PyPI wheel coverage** — `pip install nextstat` now works out of the box on all major platforms without requiring Rust. Pre-built wheels are published for:
  - Linux x86_64 (manylinux_2_17): Python 3.11, 3.12, 3.13
  - Linux aarch64 (manylinux_2_17): Python 3.11, 3.12, 3.13
  - macOS arm64 (Apple Silicon): Python 3.11, 3.12, 3.13
  - macOS x86_64 (Intel): Python 3.11, 3.12, 3.13
  - Windows x86_64: Python 3.11, 3.12, 3.13
- **Linux x86_64 glibc compatibility** — wheels now target manylinux_2_17 (glibc 2.17+) instead of manylinux_2_35, fixing installation on CentOS 7/8, RHEL 7+, Ubuntu 18.04+, and most CI/Docker images.
- **macOS Intel support** — added `x86_64-apple-darwin` target to the release matrix. Intel Mac users no longer fall back to source builds.
- **Multi-interpreter wheel builds** — Linux wheels are built inside the manylinux Docker container (all interpreters pre-installed); macOS/Windows use explicit `setup-python` with 3.11/3.12/3.13 before `--find-interpreter`.

### Added

- **HEP Full Workflow Tutorial** — comprehensive 1200-line tutorial (`docs/tutorials/hep-full-workflow.md`) covering workspace construction, all modifier types, MLE fitting, CLs hypothesis testing, upper limits (Brazil band), NP ranking, pulls, correlation matrix, profile likelihood scans, workspace combination, mass scans, GPU acceleration, preprocessing, and automated reports. Available on both English and Russian documentation sites.
- **Detailed Installation & Quickstart guides** — rewritten with step-by-step instructions, expected outputs, troubleshooting sections, and GPU acceleration flags.

## [0.9.4] — 2026-02-15

### Added

#### Python API

- **Population PK estimation** — `nlme_foce()` (FOCE/FOCEI) and `nlme_saem()` (SAEM) population pharmacokinetic estimators for 1-compartment oral models. Returns fixed effects (theta), random effects (eta), covariance (omega), OFV, and convergence diagnostics.
- **2-compartment PK models** — `TwoCompartmentIvPkModel` (IV bolus, 4 params: CL, V1, V2, Q) and `TwoCompartmentOralPkModel` (oral, 5 params: CL, V1, V2, Q, Ka) with additive/proportional/combined error models, analytical gradients, LLOQ handling, and predict.
- **PK model diagnostics** — `pk_vpc()` (Visual Predictive Check) and `pk_gof()` (Goodness of Fit with PRED, IPRED, IWRES, CWRES) for post-estimation diagnostics. `read_nonmem()` parses NONMEM-format CSV datasets.
- **pyhf workspace operations** — `workspace_combine`, `workspace_prune`, `workspace_rename`, `workspace_sorted`, `workspace_digest`, `workspace_to_xml` now available as top-level Python functions. Full parity with pyhf workspace manipulation API.
- **Simple model builders** — `simplemodel_uncorrelated(signal, bkg, bkg_uncertainty)` and `simplemodel_correlated(signal, bkg, bkg_up, bkg_down)` for quick workspace construction with shapesys/histosys uncertainties.
- **EDM (Estimated Distance to Minimum)** — `FitResult.edm` and `FitMinimumResult.edm` expose the Minuit-compatible convergence metric (g^T H^{-1} g) using the L-BFGS inverse Hessian approximation.
- **Unified `nextstat.sample()` dispatcher** — single entry point for all MCMC methods: `nextstat.sample(model, method="nuts")`, `nextstat.sample(model, method="mams")`, `nextstat.sample(model, method="laps")`. Supports `return_idata=True` for direct ArviZ `InferenceData` return and `out="trace.json"` for saving results to disk. Per-method aliases (`sample_nuts`, `sample_mams`, `sample_laps`) remain available for direct access.
- **`sample_laps` exported** — LAPS GPU sampler is now available as `nextstat.sample_laps()` (previously required `nextstat._core.sample_laps()`). Reports informative error when CUDA is not available.
- **`nextstat.bayes.sample()` supports all methods** — the ArviZ convenience wrapper now accepts `method="nuts"/"mams"/"laps"` and delegates to the unified dispatcher.

#### MAMS Sampler

- **Adaptive warmup phases** — when `init_strategy="pathfinder"` provides a Hessian-derived inverse mass matrix, MAMS warmup phase durations are rebalanced (10%/15%/10%/65% vs default 15%/40%/15%/30%) — less Welford variance collection, more equilibration. Default remains `init_strategy="random"` since Pathfinder can degrade on funnel-like geometries.

#### GPU Acceleration

- **LAPS multi-GPU** — `sample_laps()` now supports multiple GPUs via `device_ids` parameter. Chains are split across devices with synchronized warmup adaptation (barrier every 50 iterations for global step size and inverse mass matrix) and fully independent sampling. Auto-detects all available GPUs when `device_ids=None`. Python: `nextstat.sample_laps(model, n_chains=65536, device_ids=[0,1,2,3])`. Rust: `LapsConfig { device_ids: Some(vec![0,1,2,3]), .. }`.
- **LAPS (Late-Adjusted Parallel Sampler)** — GPU-accelerated MAMS sampler on CUDA. Runs 4096+ chains simultaneously (1 thread = 1 chain), zero warp divergence from fixed trajectory length. Two-phase warmup: Phase 1 unadjusted MCLMC for fast posterior exploration, Phase 2 MH-corrected for exact sampling. Built-in model gradients computed inline on device (std_normal, eight_schools, neal_funnel, glm_logistic). Python: `nextstat.sample_laps("std_normal", model_data={"dim": 10}, n_chains=4096, n_samples=2000)`. Requires NVIDIA GPU with CUDA.
- **LAPS user-defined models (NVRTC JIT)** — any user model can now run on GPU via `nextstat.RawCudaModel(dim, cuda_src, data=...)`. The user provides CUDA C source defining `user_nll()` and `user_grad()` device functions; NextStat JIT-compiles them via NVRTC with the MAMS engine, caches PTX to disk. Python: `model = nextstat.RawCudaModel(dim=10, cuda_src=..., data=[1.0]); nextstat.sample_laps(model)`. Rust: `LapsModel::Custom { cuda_src, ... }`. Requires NVIDIA GPU with CUDA.
- **LAPS fused kernel** — single kernel launch executes N transitions, keeping chain state in registers throughout. Eliminates per-transition kernel launch overhead (~5-10μs/launch on H100). Configurable via `fused_transitions` parameter: `nextstat.sample_laps(model, fused_transitions=1000)`. For fast models (dim≤100), reduces 2000 launches to 2, yielding significant speedup on high-SM GPUs.
- **LAPS H100 tuning** — configurable `sync_interval` (default 100, was hardcoded 50), `welford_chains` (default 256, was 64 for multi-GPU), `batch_size` (default 1000, was 500) for large-scale GPU sampling. Python: `nextstat.sample_laps(model, sync_interval=100, welford_chains=256, batch_size=1000)`.
- **PF3.4 Metal telemetry + matrix harness (phase 1)** — added fine-grained Metal toy-sampler timing surfaces (`prepare/counts/readback/prefix/sample/host-convert`) and CLI metrics fields for unbinned Metal toy workflows (`sampler_init_s`, `sample_phase_detail`, Metal hypotest `sample/ensemble` phase timing). Added local benchmark runner `scripts/benchmarks/pf34_metal_matrix.py` with matrix + schema (`benchmarks/unbinned/matrices/pf34_metal_v1.json`, `benchmarks/unbinned/schemas/pf34_metal_matrix_v1.schema.json`), artifact summaries, and fail-fast Metal runtime preflight (`preflight.json`) for reproducible PF3.4 baselines.
- **PF3.4 GPU lockstep policy** — curvature-based `poi_sigma` pass is now policy-gated (`poi_sigma_enabled`) and runs only when pull guardrails are requested, eliminating extra default NLL passes in GPU toy workflows.
- **PF3.4 Apple M5 benchmark snapshot (2026-02-13)** — archived local Metal matrix bundle at `benchmarks/unbinned/artifacts/2026-02-13/pf34_metal_20260213T194850Z` (`16/16` runs `rc=0`). At 10k toys, `metal_device` outperformed `metal_host` by ~`2.9x` (Gauss+Exp fit-toys) and ~`5.0x` (CB+Exp fit-toys); hypotest-toys improved by ~`1.55x` (Gauss+Exp) and ~`1.99x` (CB+Exp).
- **PF3.4 M8 hypotest orchestration cleanup (Metal)** — refactored `unbinned-hypotest-toys` Metal path to build toy samplers once and reuse a shared `sample_device` closure across observed/expected-set branches (removed duplicated sampler construction blocks). Local validation bundle `benchmarks/unbinned/artifacts/2026-02-13/pf34_m8_local_20260213T204135Z` shows performance-neutral behavior within noise (Gauss+Exp +1.2%, CB+Exp -0.7% vs PF3.4 baseline).
- **PF3.4 M8 Metal hypotest timing split** — `unbinned-hypotest-toys` Metal metrics now expose per-ensemble fit stages (`build_s`, `free_fit_s`, `fixed_fit_s`) under `timing.breakdown.toys.{b,sb}`. Validation artifact `benchmarks/unbinned/artifacts/2026-02-13/pf34_m8_timing_20260213T211006Z` confirms fixed-fit is the dominant stage in B-only ensemble.
- **PF3.4 M8 Metal fixed-fit optimization (Step 3)** — tuned Metal hypotest fixed-fit stage to use a dedicated toy config (`max_iter=400`, `tol=3e-6`) for constrained B/S+B toy fits. Validation artifact `benchmarks/unbinned/artifacts/2026-02-13/pf34_m8_step3_20260213T212122Z`: Gauss+Exp `hypotest_device` 10k improved `15.16s -> 3.26s` (~`4.65x` vs M8 timing pass), CB+Exp improved `8.43s -> 5.27s` (~`1.60x`). Quality stayed stable (`cls/clb/clsb` unchanged, `n_error=0`, `n_nonconverged=0`).
- **PF3.4 M7 Metal u32 toy-offset guard + auto-chunking** — `unbinned-fit-toys --gpu metal` and `unbinned-hypotest-toys --gpu metal` now preflight expected events-per-toy against the 32-bit toy-offset budget and automatically split large toy workloads into contiguous Metal batches (`n_batches`, `max_toys_per_batch`) instead of aborting. CLI now fails fast only when a single toy itself would exceed the 32-bit budget, with actionable guidance to use CPU shard mode (`--shard INDEX/TOTAL`) or reduce dataset/toy size. Added unit coverage for chunk planner safe/oversized cases.
- **PF3.4 M3 Metal hot-path allocation/copy trim** — `MetalBatchAccelerator::upload_observed()` now reuses preallocated f32 scratch buffers for `observed/ln_facts/obs_mask` conversion (`max_batch * n_main_bins`) instead of allocating three temporary `Vec<f32>` on every call. In both `MetalBatchAccelerator` and `MetalUnbinnedBatchAccelerator`, gradient-output zeroing now uses direct shared-buffer memset (`write_bytes`) rather than copying a host zero-vector each step. `MetalUnbinnedToySampler::sample_toys_1d_inner()` also removed per-call `params_f32` and `counts.to_vec()` allocations by writing params directly into a shared buffer and reading counts as a shared slice for prefix-sum. This reduces host-side allocation/copy churn in lockstep toy loops.
- **PF3.4 M3 local timing check (Apple M5, 2026-02-13)** — quick repeat set (`n=3`) for 10k-toy `metal_device` cases after M3 slices, artifact: `benchmarks/unbinned/artifacts/2026-02-13/pf34_m3_slice2_repeats_20260213T232535Z`. Median wall-times: Gauss+Exp fit-toys `1.51s`, CB+Exp fit-toys `1.72s`, Gauss+Exp hypotest-toys `2.94s`, CB+Exp hypotest-toys `5.02s`.
- **PF3.4 full matrix rerun after M3 slices (Apple M5, 2026-02-13)** — full local matrix artifact: `benchmarks/unbinned/artifacts/2026-02-13/pf34_m3_full_20260213T232729Z` (`16/16` runs `rc=0`). Key 10k-toy `metal_device` wall-times: Gauss+Exp fit-toys `1.37s`, CB+Exp fit-toys `1.76s`, Gauss+Exp hypotest-toys `2.71s`, CB+Exp hypotest-toys `4.74s`.
- **PF3.4 M5 toy-sampler kernel step (Apple M5, 2026-02-13)** — `unbinned_toy_sample_obs_1d` now caches per-toy process CDF in threadgroup memory (`TOY_MAX_PROC_CACHE=256`) to avoid per-event recomputation of process yields in common low-process channels. Explored GPU counts→scan→sample offsets path but did not keep it in default route after local measurements showed launch overhead without end-to-end benefit; current default keeps CPU prefix-sum and the kernel-side CDF cache only. Validation artifacts: `benchmarks/unbinned/artifacts/2026-02-13/pf34_m5_post_20260213T234528Z`, `benchmarks/unbinned/artifacts/2026-02-13/pf34_m5_post_repeats_20260213T234748Z`, `benchmarks/unbinned/artifacts/2026-02-13/pf34_m5_final_repeats_20260213T235051Z`.
- **PF3.4 M6 gradient-atomic contention reduction (Apple M5, 2026-02-14)** — `unbinned_batch_nll_grad` now stages gradient contributions in threadgroup memory for the first `min(n_params, 24)` parameters and flushes them once per parameter/threadgroup, reducing global `atomic_float` pressure in the toy event loop hot path. Validation artifacts: `benchmarks/unbinned/artifacts/2026-02-14/pf34_m6_repeats_20260214T000844Z` and `benchmarks/unbinned/artifacts/2026-02-14/pf34_m6_full_20260214T000926Z`. On controlled 10k-toy repeats (`n=3`, `metal_device`) vs pre-M6 local baseline (`pf34_m5_final_repeats_20260213T235051Z`): Gauss+Exp fit speedup `+8.05%`, CB+Exp fit speedup `+8.83%`, Gauss+Exp hypotest speedup `+1.39%`, CB+Exp hypotest speedup `+28.73%`.
- **PF3.4 M6.3 process-yield cache in Metal batch kernels (Apple M5, 2026-02-14)** — `unbinned_batch_nll_grad` and `unbinned_batch_nll_only` now precompute per-process `nu/log(nu)/dnu` once per toy in threadgroup memory and reuse them across event loops. This removes repeated yield/modifier recomputation from the hot path. Validation artifacts: `benchmarks/unbinned/artifacts/2026-02-14/pf34_m6_3_proc_cache_repeats_20260214T005418Z` and `benchmarks/unbinned/artifacts/2026-02-14/pf34_m6_3_proc_cache_repeats5_20260214T005553Z`. On stabilized 10k-toy repeats (`n=5`, `metal_device`) vs `pf34_m6_2_repeats_20260214T004701Z`: Gauss+Exp fit speedup `+18.81%`, CB+Exp fit speedup `+3.03%`, Gauss+Exp hypotest speedup `+26.68%`, CB+Exp hypotest speedup `+16.70%`.
- **PF3.4 M6.5 rate-modifier derivative cache in Metal batch gradients (Apple M5, 2026-02-14)** — `unbinned_batch_nll_grad` now caches per-modifier derivative coefficients `dnu_m = nu * dlogf` once per toy (when `total_rate_mods <= 256`) and reuses them in the per-event gradient loop, avoiding repeated `rate_modifier_factor_dlogf` calls in the hot path. Validation artifacts: cache-on `benchmarks/unbinned/artifacts/2026-02-14/pf34_m6_5_rate_dnu_cache_repeats_20260214T093849Z`, same-session control (cache disabled) `benchmarks/unbinned/artifacts/2026-02-14/pf34_m6_5_ab_control_nocache_repeats_20260214T094256Z`, final keep-run `benchmarks/unbinned/artifacts/2026-02-14/pf34_m6_5_final_repeats_20260214T094544Z`. A/B medians at 10k toys show speedup `+5.63%` to `+6.80%`; conservative keep-run deltas vs control remain positive (`+0.91%` to `+2.11%`).
- **PF3.4 M6.6 process metadata cache in Metal batch gradients (Apple M5, 2026-02-14)** — `unbinned_batch_nll_grad` now caches per-process `rate_mod_offset` and sanitized `n_rate_mods` in threadgroup memory (`s_proc_mod_off`, `s_proc_nmods`) once per toy and reuses them in hot loops, reducing repeated process descriptor loads and bounds checks. Validation artifact: `benchmarks/unbinned/artifacts/2026-02-14/pf34_m6_6_proc_meta_cache_repeats_20260214T100036Z` (`n=3`, `metal_device`, 10k toys). Medians vs M6.5 final baseline (`pf34_m6_5_final_repeats_20260214T094544Z`): Gauss+Exp fit `speedup +6.85%`, CB+Exp fit `speedup +4.59%`, Gauss+Exp hypotest `speedup +6.21%`, CB+Exp hypotest `speedup +5.91%`. Numerical quality unchanged (`n_error=0`, `n_nonconverged=0`, `cls/clb/clsb=1.0`).
- **PF3.1 remote stand preflight gate (CUDA)** — added `scripts/benchmarks/pf31_remote_preflight.sh` to fail fast before expensive matrix runs: captures remote env (`nvidia-smi/topology/compute-apps`, toolchain, binary version), validates minimum GPU count, runs smoke checks for host and device-resident toy paths, and syncs artifacts immediately to local bundle (`summary.json`, `summary.md`, smoke outputs/metrics). Runbook/docs updated in `docs/benchmarks/unbinned-publication-runbook.md` and `benchmarks/unbinned/README.md`.
- **PF3.1 CUDA4 saturation spot-check protocol + snapshots** — documented and validated a `cuda4-only` stress check path for temporary stands (`--gpu-devices 0,1,2,3 --gpu-sample-toys --gpu-shards 16`) with mandatory 1s `nvidia-smi` sampling and archived util traces. New artifacts: `benchmarks/unbinned/artifacts/2026-02-14/pf31_cuda4_only_20260214T121023Z` (`100k toys`) and `benchmarks/unbinned/artifacts/2026-02-14/pf31_cuda4_only_20260214T121214Z_t300k` (`300k toys`). Added explicit guardrail for toy-offset overflow budget (`n_toys * expected_events_per_toy <= u32::MAX`).
- **PF3.3 CUDA gpu-native line-search + gradient hot-path** — the persistent unbinned CUDA L-BFGS kernel (`unbinned_batch_lbfgs_fit`) now computes the Armijo directional derivative using the *projected* clamped step (bounds-aware) and falls back to steepest descent (dropping history) when the projected step is not a descent direction. Gradient accumulation moved into a per-toy shared buffer. On GEX44 (`pf31_gauss_exp_2m`, `n_toys=10000`, `--gpu cuda --gpu-native`): `batch_fit_s` `1390.99s` → `592.28s` (speedup `2.35x`). Artifact: `benchmarks/unbinned/artifacts/2026-02-14/pf33_t6_sharedgrad_rerun_20260214T185831Z/metrics.json`.
- **PF3.3 CUDA gpu-native multi-channel pointer descriptors** — multi-channel native fits no longer download and re-upload per-channel static buffers to concatenate them on the host. `GpuChannelDesc` now carries device pointers to channel-local buffers, and host-toy multi-channel build uses a shared CUDA context via `CudaUnbinnedBatchAccelerator::from_unbinned_static_and_toys_in_ctx`. GEX44 2-channel smoke (`n_toys=2000`, `--gpu cuda --gpu-native`): `batch_build_s=0.0073s`, `n_error=0`. Artifact: `benchmarks/unbinned/artifacts/2026-02-14/pf33_t4_multich_smoke_20260214T192847Z/metrics.json`.

#### Visualization

- **TRExFitter plot parity** — `ns-viz` now covers all 18 TRExFitter output plot types. Five new artifact modules added:
- **Significance scan** (`significance`) — `SignificanceScanArtifact` with p₀ / Z vs mass or signal-strength parameter. Supports single-workspace and multi-mass-point scans. Comparable to TRExFitter step `s` (`GetSignificance`).
- **2D likelihood contours** (`contour`) — `ContourArtifact` with 2-POI Δ(2NLL) grid and marching-squares contour extraction at 68% / 95% CL thresholds. Comparable to TRExFitter `GetContour`.
- **Unfolding plots** (`unfolding`) — `ResponseMatrixArtifact` (migration matrix with purity/stability metrics, optional column normalisation) and `UnfoldedSpectrumArtifact` (unfolded differential distribution with stat/syst errors and truth overlay). Comparable to TRExFitter step `x`.
- **Morphing validation** (`morphing`) — `MorphingArtifact` showing discrete anchor templates alongside interpolated result at an arbitrary parameter value.
- **Signal injection / linearity** (`injection`) — `InjectionArtifact` with μ̂ vs μ_inj linearity plot data, per-point pull statistics, and automatic least-squares slope/intercept fit.

#### Survival Analysis

- **Interval-censored Weibull AFT with covariates** — accelerated failure time model `S(t|x) = exp(-(t/λ(x))^k)` where `log(λ_i) = x_i^T β`. Supports exact, right, left, and interval censoring with analytical gradients. Python: `nextstat.IntervalCensoredWeibullAftModel(time_lower, time_upper, censor_type, covariates)`.

#### Bayesian Inference (MCMC)

- **MAMS sampler (Metropolis-Adjusted Microcanonical Sampler)** — exact MCMC sampler based on isokinetic dynamics on the unit velocity sphere (arXiv:2503.01707). 1.3–1.7x better ESS/gradient than NUTS on standard benchmarks, 5x better than BlackJAX adjusted_mclmc. Stan-style 4-phase DualAveraging warmup, diagonal preconditioning, early termination for divergences. Python: `nextstat.sample_mams(model, n_chains=4, n_warmup=1000, n_samples=1000)`.
- **Pathfinder init strategy for NUTS and MAMS** — `init_strategy="pathfinder"` runs a full L-BFGS fit with Hessian to find the posterior mode and extracts the diagonal inverse Hessian as an initial mass matrix. This gives warmup a head-start on the preconditioner, allowing shorter warmup (e.g. 200 vs 1000 iterations). Falls back to random init on fit failure. Python: `nextstat.sample(model, init_strategy="pathfinder")` / `nextstat.sample_mams(model, init_strategy="pathfinder")`.
- **N-dimensional Neal's funnel** — `FunnelModel(dim=N)` generalizes the classic 2D funnel to arbitrary dimensions for MCMC stress testing. Default `dim=2` for backwards compatibility.

#### Cross-Vertical Statistical Features (ns-inference)

- **Cross-Entropy Importance Sampling for fault trees** — iterative proposal optimization for rare-event probability estimation using orders of magnitude fewer scenarios than vanilla MC. Multi-level adaptive biasing with soft importance function handles probabilities down to ~1e-16 (AND-of-8 at p_each=0.01). Exponential smoothing (α=0.7) on proposal updates prevents oscillation. Supports all failure modes: Bernoulli, WeibullMission (reduced to effective p), and BernoulliUncertain (CE on latent Z proposal). Python: `nextstat.fault_tree_mc_ce_is(spec, n_per_level=10000, seed=42)`.
- **Metal fault tree MC** — GPU-accelerated Monte Carlo fault tree simulation on Apple Silicon via MSL kernel dispatch. Supports all failure modes (Bernoulli, BernoulliUncertain, WeibullMission). Python: `nextstat.fault_tree_mc(spec, n_scenarios, device='metal')`.
- **Profile CI warm-start** — bisection steps in `profile_ci()` now carry forward the optimized nuisance parameters from the previous step, reducing redundant optimizer iterations for multi-step profiles.
- **Profile likelihood confidence intervals** — generic bisection-based profile CI for any `LogDensityModel`. Computes `{θ : 2*(NLL(θ) - NLL_min) ≤ χ²(1, α)}` for one or all parameters. Python: `nextstat.profile_ci(model, fit_result)`.
- **Identifiability warnings** — `FitResult.warnings` reports potential identifiability issues: near-singular Hessian (condition number > 1e8), non-finite uncertainties, near-zero Hessian diagonal.

### Fixed

- **NB GLM gradient robustness** — hardened `NegativeBinomialRegressionModel::grad_nll()` against non-finite intermediate terms in dispersion derivative accumulation (skip non-finite per-event `d_logp/dtheta`; sanitize non-finite gradient outputs to finite zeros), preventing first-step line-search breakdowns in benchmark-like NB2 synthetic fits.
- **Regression guardrail for public GLM workload shape** — added `test_mle_negbin_benchmark_like_dataset_converges` (`n=1000`, `p=10`, `seed=42`) to lock in convergence behavior for benchmark-style negative-binomial data generation.
- **GLM NegativeBinomial parity alignment in public benchmark harness** — `suites/glm/run.py` now aligns coefficient vectors for NB parity by trimming the extra NextStat dispersion parameter when comparing against `statsmodels` GLM with fixed `alpha`; NB coefficient parity is now evaluated on matched beta terms, with explicit `coef_alignment` metadata.
- **GLM deterministic regression gate for NB first-step failures** — `suites/glm/suite.py` now hard-fails NB cases on non-convergence or first-step abort pattern (`n_evaluations <= 1`) to prevent silent reintroduction of `MoreThuente NaN/Inf` regressions in benchmark snapshots.
- **GLM deterministic rerun snapshot (2026-02-14)** — post-fix rerun archive: `benchmarks/nextstat-public-benchmarks/out/glm_suite_rerun_20260214T142830Z` (`12/12 ok`, `0 warn`, `0 failed`), with `negbin_{1k,10k,100k}` converged and parity `ok`.
- **LAPS kernel first-call rejection** — fixed CUDA kernel bug where all chains were rejected on the very first transition because `potential_old = NaN` triggered the divergence check (`!isfinite(current_w)` where `current_w = NaN - NaN = NaN`). Now skips early termination when `potential_old` is not finite and unconditionally accepts the first transition to initialize chain state. Affected all three kernel variants (non-fused, fused, warp).
- **LAPS data-aware chain initialization** — chains now start within ~2σ of the typical set via `init_scale = L / (π√dim)` instead of fixed `U(-2, 2)`. For data-heavy GLM models (n≥1000), this prevents chains from starting 10–30 posterior σ away from the mode. GLM logistic n=1000 p=20: warmup time 271s → 183s.
- **LAPS warmup cold-start eps collapse** — fixed step size adaptation when chains start from random initialization. Full-trajectory calibration from random init caused eps collapse (→ 0.0003) and capped leapfrog steps (64/transition), making data-heavy GPU models 10x slower. Now uses burn-in transitions + short calibration for initial search, and clamps eps ≤ L/2 to prevent Phase 3 Dual Averaging over-adaptation. GLM logistic (n=200, p=6, 4096 chains): 288s → 31s.
- **WASM Bayesian GLM-logistic stability hardening** — browser `run_bayes_sample` now validates sampler inputs (`target_accept` in `(0,1)`, finite, and `max_treedepth` bounds), standardizes logistic design matrix columns (z-score), enforces both outcome classes, and uses weakly-informative priors with penalized intercept in the Bayesian logistic demo path to prevent runaway coefficients/divergence spikes on near-separable inputs.
- **CUDA device-sharded toy failure diagnostics** — `unbinned-fit-toys --gpu cuda --gpu-sample-toys` now emits phase-aware shard error context (`sample/build/fit` with shard range and device id), records `shard_progress` checkpoints in timing breakdown, and avoids silent thread panic loss by replacing `join().unwrap()` with structured panic/error propagation including last shard checkpoints.
- **CUDA lockstep retry scheduling fix (PF3.3-T2)** — fixed adaptive retry flow in `unbinned_gpu_batch` to avoid double-running the initial warm-start attempt with mismatched `max_iter` budgets. Added CUDA-scoped unit coverage for adaptive retry schedule monotonicity/full-budget restore and smooth-jitter bounds safety.
- **GPU toy-fit optimizer diagnostics (PF3.3-T3)** — `unbinned-fit-toys` now reports optimizer telemetry for GPU toy batches: iteration median/p95, fraction at max-iter, retry usage, line-search exhaustion counts/fraction, and status/error histograms. Added `results.optimizer_diagnostics` in output JSON and `timing.breakdown.toys.optimizer` in metrics JSON (backward-compatible extension).
- **CUDA native status taxonomy alignment (kernel ↔ Rust)** — unified status mapping for GPU-native L-BFGS (`0=MaxIterReached`, `1=Converged`, `2=ComputationFailed`) and propagated explicit termination reasons/warnings to `FitResult` (`retry_attempts_used`, `line_search_exhaustions`, raw `cuda_status`). Added per-toy line-search exhaustion counter in CUDA kernel output.
- **CUDA lockstep active-toy compaction (PF3.3-T5)** — unbinned lockstep now evaluates only active (non-converged) toys via compacted parameter buffers and active-slot→toy mapping in CUDA batch kernels (`unbinned_batch_nll_grad`/`unbinned_batch_nll_only`), while preserving deterministic output order. Added active-subset fallback adapter for non-CUDA accelerators and unit coverage for non-contiguous active index ordering + shrinking active batches.

## [0.9.3] — 2026-02-13

### Fixed

- **PyPI wheel coverage** — release CI now builds wheels for all supported Python versions (3.11–3.14) on every platform. Previously, `--find-interpreter` was only set for Linux aarch64, causing macOS, Linux x86_64, and Windows to ship only the runner-default Python wheel (cp314/cp312). All four targets now use `--find-interpreter` in the maturin build matrix.

## [0.9.2] — 2026-02-13

### Added

#### Native ROOT I/O

- **LRU basket cache** — decompressed TTree basket payloads are cached per-`RootFile` with byte-bounded LRU eviction (default 256 MiB). Eliminates redundant decompression on repeated branch reads (e.g. multi-branch formula evaluation). `RootFile::basket_cache()` for stats, `RootFile::set_cache_config()` to tune capacity or disable.
- **Lazy branch reader** — `RootFile::lazy_branch_reader()` returns a `LazyBranchReader` that decompresses only the baskets needed for the requested entries. `read_f64_at(entry)` touches one basket; `read_f64_range(start, end)` touches only overlapping baskets. Backed by the LRU cache for repeated access.
- **ChainedSlice** — zero-copy concatenation of multiple decompressed basket payloads via `Arc` sharing. Presents non-contiguous segments as a single logical byte slice with O(log n) random access and cross-segment reads. `LazyBranchReader::load_all_chained()` returns a `ChainedSlice` for custom decode pipelines.
- **ROOT leaflist parsing** — `ns-root` now parses compound leaf-list branches (multiple scalars packed per entry).

#### ns-zstd Performance

- **Encoder hot-path optimizations** — 20+ targeted improvements to the pure-Rust Zstd encoder: faster FSE state transitions, packed sequence bit writes, hash-chain collision reduction via head tags and u64 reject, common-prefix u128 comparison, fast-path depth-1 search, lazy/history check reduction, no-match skip heuristic, and match-length encoding fast-path.
- **Compression Level 3 (Default) complete** — hash-chain match finder (depth-2, packed u64 heads, tag rejection), lazy matching at P+1, offset history rep-codes (1/2/3), 256KiB window, FSE `RepeatOrBuild` strategy. Ratio within 5% of C zstd level 3 on ROOT-like data. All Parquet/WASM use cases now fully supported via pure Rust.
- **`zstd-shim`** — transparent backend selection crate: uses native `libzstd` (via `zstd-safe`) on desktop for maximum throughput, falls back to pure-Rust `ns-zstd` on WASM and embedded targets.

#### CPU Farm Orchestration

- **CPU farm tooling for unbinned toys (HEP cluster prep)** — added `scripts/farm/preflight_cluster.py`, `scripts/farm/run_unbinned_fit_toys_cluster.py`, and `scripts/farm/merge_unbinned_toys_results.py` for CPU-only multi-host toy sharding, deterministic seed/toy_start scheduling, and merged output reconstruction.

#### GPU Acceleration

- **Multi-channel GPU batch toys (Metal + CUDA)** — `ns-inference` batch toy fitter now handles workspaces with multiple channels on both Metal and CUDA backends.
- **Unbinned CUDA multi-GPU batch toys (`--gpu-devices`)** — `nextstat unbinned-fit-toys` and `nextstat unbinned-hypotest-toys` can shard host-sampled toys across multiple CUDA devices (manual stream/context-per-device orchestration, merged in toy order). Device-resident `--gpu-sample-toys` remains single-GPU for now.
- **Unbinned CUDA device-resident shard orchestration (`--gpu-shards`)** — `nextstat unbinned-fit-toys` and `nextstat unbinned-hypotest-toys` now support sharded `--gpu-sample-toys` execution. Shards are mapped to `--gpu-devices` round-robin and can be used for single-GPU emulation (`--gpu cuda --gpu-sample-toys --gpu-shards N`) before validating on 2+ GPU hardware.
- **Unbinned CUDA host-toy shard orchestration (`--gpu-shards` without `--gpu-sample-toys`)** — CUDA toy workflows now also support sharded host-toy execution (`pipeline: cuda_host_sharded`) for `unbinned-fit-toys` and `unbinned-hypotest-toys`, with shard plan exposed in metrics.
- **Unbinned CUDA sharded toy-path metrics/tests** — added CUDA integration coverage for `--gpu-sample-toys --gpu-shards` in both `unbinned-fit-toys` and `unbinned-hypotest-toys`, including metrics contract checks for `pipeline: cuda_device_sharded` and `device_shard_plan`.
- **PF3.1-OPT2: Parallel device-resident multi-GPU** — the `cuda_device_sharded` pipeline now runs each shard (sample → build → fit) on its own thread with its own CUDA context via `std::thread::scope`, enabling true multi-GPU concurrency. Previously, shards were processed sequentially in a `for` loop (zero overlap, flat scaling). Fixed in 3 locations: `cmd_unbinned_fit_toys`, `count_q_ge_ensemble_cuda_device_sharded`, `generate_q_ensemble_cuda_device_sharded`. Per-device timing (`shard_detail`) emitted in metrics JSON. Validated on 4× A40: 2 GPU = 1.97× (near-linear), 3 GPU = 2.9×. Before fix: flat 1.00×.
- **PF3.1 multi-GPU health-check note** — added an explicit infrastructure gate for 2+ GPU benchmark stands: verify CUDA Driver API initialization (`cuInit`) before running sharded toy matrices. Some virtualized H100 nodes can show GPUs in `nvidia-smi` while failing runtime init (`cuInit=802`), which makes `--gpu cuda` unavailable despite visible hardware.
- **PF3.1 2x H100 benchmark matrix snapshot** — completed and archived a full unbinned toy matrix (10k/50k/100k) for CPU, 1-GPU/2-GPU host-toy, and 1-GPU/2-GPU device-resident sharded paths (`benchmarks/unbinned/artifacts/2026-02-11/pf31_matrix_20260211T182402Z`). On this stand, host-toy multi-GPU scales close to 2x (e.g. 1345s -> 701s at 100k), while device-resident sharded path is near-flat (1318s -> 1316s).
- **PF3.1 single-GPU runtime snapshot (Gex44, RTX 4000 Ada)** — added 10k/20k validation artifacts for worker-per-device orchestration (`benchmarks/unbinned/artifacts/2026-02-12/pf31_opt2_valid_20260212T024003Z`, `benchmarks/unbinned/artifacts/2026-02-12/pf31_opt2_focus_20260212T031632Z`). For this workload, single-GPU toy pipelines remain much slower than CPU fused path, while `--gpu-sample-toys` still reduces sampling time versus host-toy.
- **CUDA `--gpu-native` routing policy hardening** — `unbinned-fit-toys --gpu cuda` no longer auto-enables `--gpu-native`; persistent on-device L-BFGS is now explicit opt-in only. Added CUDA CLI integration coverage to assert default pipeline stays non-native unless `--gpu-native` is passed.
- **CUDA toy-fit warm-start + iteration budget parity (phase 1)** — analytical `unbinned-fit-toys --gpu cuda` now warm-starts from observed-data MLE `θ̂` when available (fallback to spec init), and uses `max_iter=5000` for both lockstep and native batch fit calls (previously 1000).
- **GEX44 post-rebuild CUDA snapshot (2026-02-13)** — archived fresh single-GPU artifacts at `benchmarks/unbinned/artifacts/2026-02-13/gex44_pf33_rebuild_20260212T234615Z/summary.json`. On RTX 4000 Ada, both `cuda_device` and `cuda_gpu_native` toy paths remain drastically slower than CPU fused path for Gauss+Exp/CrystalBall toy workloads, with near-identical CUDA wall-time between default and native routes. Follow-up optimization tasks were opened in BMCP.
- **CUDA toy-fit optimizer stabilization (pass 1)** — added relative objective-decrease stopping and non-finite fail-fast in the GPU-native unbinned L-BFGS kernel (`unbinned_batch_lbfgs_fit`), and mirrored relative objective early-stop in lockstep `LbfgsState` to reduce max-iter tails on toy workloads. Includes new unit tests for relative-objective stop behavior.
- **GEX44 CUDA recovery snapshot (2026-02-13)** — archived new single-GPU artifacts at `benchmarks/unbinned/artifacts/2026-02-13/gex44_cuda_opt1_20260213T091253Z` and `benchmarks/unbinned/artifacts/2026-02-13/gex44_cuda_opt1_scale_20260213T091410Z`. On this branch snapshot, analytical toy fits route to `cuda_gpu_native` and outperform CPU for Gauss+Exp: ~6.3x at 1k toys, ~5.0x at 10k toys, ~4.6x at 50k toys, and ~2.9x on a ~2M-events/toy stress case (50 toys), with 100% convergence in listed runs.
- **GEX44 CB/DCB CUDA scale snapshot (2026-02-13)** — added CrystalBall/DoubleCrystalBall 10k-event toy matrix artifacts and summary at `benchmarks/unbinned/artifacts/2026-02-13/gex44_cuda_opt1_scale_20260213T091410Z/summary_cb_dcb_scale.json`. Measured 100% convergence for all listed runs (1k and 10k toys): CB CUDA (`cuda_gpu_native`) ~15-17x faster than CPU; DCB CUDA host path already ~28-33x faster than CPU, and explicit `--gpu-native` further improves DCB throughput by ~12% (1k toys) to ~56% (10k toys) vs DCB CUDA host mode.
- **PF3.1-OPT4 sharding estimation fix** — auto-shard VRAM estimation for Poisson toys now uses the expected yield at the toy generation point (sum of yield expressions) rather than the observed dataset size, preventing under-sharding on large-yield studies (e.g. O(2M) events/toy with O(10k) toys). Added `YieldExpr::value()` public helper for tooling.
- **PF3.1 benchmark harness mode split (`host` vs `native`)** — `scripts/benchmarks/pf31_remote_matrix.sh` now supports `PF31_HOST_FIT_MODES` (default `host,native`) and emits explicit host lockstep (`..._host_t...`) and explicit `--gpu-native` (`..._native_t...`) cases for analytical CUDA toy fits. `summary.json` rows now include `fit_mode` classification (`cpu|host|native|host_sharded|device_sharded`) to avoid manual post-processing when comparing DCB host/native routes.
- **PF3.3 unbinned CPU vs CUDA gate matrix** — added `benchmarks/unbinned/matrices/pf33_gate_v1.json`, new 100k-events/toy specs (`benchmarks/unbinned/specs/pf33_*_100k.json`), and strict report writer `scripts/benchmarks/pf33_gate_report.py` emitting the mandatory numbers-only table format from `.claude/benchmark-protocol.md`.
- **PF3.3 unbinned runtime policy gate (`--gpu auto`)** — `unbinned-fit-toys` and `unbinned-hypotest-toys` now accept `--gpu auto` to policy-select CPU vs CUDA based on model topology and estimated events/toy, and log the decision reason. `--gpu auto` rejects backend-specific knobs (`--gpu-devices/--gpu-shards/--gpu-sample-toys/--gpu-native`) to keep behavior explicit; use `--gpu cuda|metal` for overrides.
- **PF3.1 large-scale multi-GPU snapshot (4x A40, 2026-02-13)** — added artifact `benchmarks/unbinned/artifacts/2026-02-13/pf31_2m10k_multigpu_20260213T125921Z/summary.json` for `2M events/toy × 10k toys` (Gauss+Exp). All successful runs converged 100% (`n_error=0`). Measured wall-times: `cuda_device_sharded` 1GPU(sh8)=210.78s, 2GPU(sh8)=106.72s, 4GPU(sh16)=58.79s (scaling ~1.97x and ~3.58x vs 1GPU). `cuda_gpu_native_sharded` 1GPU=479.09s, 2GPU=238.42s, 4GPU=121.49s (scaling ~2.01x and ~3.94x vs 1GPU), but slower absolute throughput than device-sharded on this workload.
- **PF3.1 large-shard guardrail finding (u32 event-offset overflow)** — on the same 4x A40 run, single-GPU `cuda_device_sharded` with `--gpu-shards 4` failed fast with `Validation error: total toy events overflow u32` (`total_events=4295980781`, `n_toys=2500` per shard). Using more shards (`--gpu-shards 8`) resolves the overflow; follow-up bug task opened in BMCP for 64-bit offsets/guardrail policy.
- **PF3.1 apples-to-apples CPU baseline snapshot (4x A40 stand, 2026-02-13)** — added CPU runs for `pf31_gauss_exp_2m` on the same host (`194.68.245.147`) used for the 4x A40 GPU matrix: `100 toys = 12.11s`, `500 toys = 47.72s`, `1000 toys = 94.10s`, all `n_error=0`. Extrapolated from steady-state (500/1000 toys), CPU throughput is ~`10.55 toys/s` (`~947.6s` for 10k toys), giving measured GPU speedups of ~`4.50x` (`cuda1_device_sh8`), `8.88x` (`cuda2_device_sh8`), and `16.12x` (`cuda4_device_sh16`) for the same workload family.
- **PF3.1 P0 shard preflight hardening (implemented)** — CUDA toy commands now preflight estimated events-per-toy against 32-bit toy-offset limits before sampling/fitting. If user-specified `--gpu-shards` is too small, CLI fails fast with a deterministic error. In auto mode, shard count is increased as needed (and host-toy path auto-enables sharding when required) to avoid runtime offset overflow.
- **PF3.1 P0 guardrail verified on fresh 2x A40 run** — archived post-fix artifact `benchmarks/unbinned/artifacts/2026-02-13/pf31_p0_2gpu_2m10k_20260213T143409Z` (`2M events/toy × 10k toys`, `--gpu-devices 0,1 --gpu-shards 8 --gpu-sample-toys`). Result: `pipeline=cuda_device_sharded`, `wall=106.71s`, `10000/10000 converged`, `n_error=0`, shard plan `[0,1,0,1,0,1,0,1]`.
- **PF3.1 publication benchmark prep kit** — added publication-grade orchestration and reporting for unbinned GPU snapshots: matrix file `benchmarks/unbinned/matrices/pf31_publication_v1.json`, orchestrator `scripts/benchmarks/pf31_publication_matrix.sh` (dry-run/preflight + case execution + snapshot index), and aggregator `scripts/benchmarks/pf31_publication_report.py` (`publication_summary.json` / `.md` with gate evaluation). Added runbook: `docs/benchmarks/unbinned-publication-runbook.md`.
- **Metal `--gpu-sample-toys`** — device-resident toy sampling on Apple Silicon (previously CUDA-only).
- **Parquet observed data for unbinned `--gpu`** — unbinned GPU path can now ingest observed data directly from Parquet files.
- **TensorRT execution provider for neural PDFs** — `--features neural-tensorrt` enables TensorRT EP with FP16 inference, engine caching (`~/.cache/nextstat/tensorrt/`), and dynamic batch-size optimization profiles. Automatic fallback chain: TensorRT → CUDA EP → CPU. `FlowGpuConfig` for custom TRT settings; `FlowPdf::from_manifest_with_config()` constructor; `FlowPdf::gpu_ep_kind()` for runtime introspection.
- **Analytical Jacobian gradients for flow PDFs (G3)** — when the manifest includes a `log_prob_grad` ONNX model, `FlowPdf::log_prob_grad_batch()` computes exact `∂ log p / ∂ context` in a single forward pass instead of `2 × n_context` finite-difference evaluations. CUDA kernel `flow_nll_grad_reduce_f32` computes NLL + gradient intermediates in one launch; `GpuFlowSession::nll_grad_analytical_device_f32()` assembles the full gradient from device-resident f32 buffers. `FlowPdf::log_prob_grad_batch_cuda()` runs the grad ONNX model on CUDA EP with I/O binding, returning `FlowCudaLogProbGrad` (two device-resident tensors: log_prob + Jacobian). Verified on RTX 4000 SFF Ada: NLL parity f32↔f64 = 4.43e-10, gradient parity at machine precision.
- **Multi-GPU batch toy fitting for flow PDFs** — `fit_flow_toys_batch_cuda()` runs lockstep L-BFGS-B across thousands of toys using the `flow_batch_nll_reduce` kernel (1 block = 1 toy). `shard_flow_toys()` partitions toys evenly across devices; `fit_flow_toys_batch_multi_gpu()` drives all GPUs and merges results. Finite-difference gradients via `2 × n_params + 1` batch NLL launches.
- **CLI `--gpu cuda` for flow PDFs in toy pipeline (G2-R1)** — `nextstat unbinned-fit-toys --gpu cuda` now supports flow/conditional_flow/dcr_surrogate PDFs. CPU toy sampling → CPU logp evaluation → CUDA NLL reduction via `flow_batch_nll_reduce` kernel → lockstep L-BFGS-B. `spec_has_flow_pdfs()` auto-detects neural PDFs and routes to the flow batch path; `build_flow_batch_config()` maps yield specs to `FlowBatchProcessDesc`. Supports Gaussian constraints. Limitations: single included channel, no rate modifiers, no `--gpu-sample-toys`/`--gpu-native` (CPU sampling only, lockstep only).
- **Normalization grid cache for all dimensionalities** — `QuadratureGrid::auto()` selects the optimal strategy per dimension count: Gauss-Legendre (1-3D), low-order tensor product N16/N8 (4-5D), Halton quasi-Monte Carlo (6D+). `NormalizationCache` avoids recomputation when parameters haven't changed. `FlowPdf` now normalizes at all dimensionalities (previously skipped 4D+).
- **CUDA EP f32 zero-copy NLL reduction** — `GpuFlowSession::nll_device_ptr_f32()` accepts a raw CUDA device pointer to float log-probs from ONNX Runtime CUDA EP, eliminating the host-to-device memcpy. Up to 57× faster than the f64 host-upload path at typical event counts (1K). Python binding: `GpuFlowSession.nll_device_ptr_f32(ptr, params)`.

#### Hybrid Likelihood (Phase 4)

- **`HybridLikelihood<A, B>`** — generic combined likelihood that sums NLLs from two `LogDensityModel` implementations with shared parameters. Implements `LogDensityModel`, `PoiModel`, and `FixedParamModel` so it works with all existing MLE, profile scans, CLs, and toy infrastructure.
- **`SharedParameterMap`** — merges parameter vectors from two models by name. Shared parameters get intersected bounds; model-specific parameters are appended. `extract_a()` / `extract_b()` for parameter slicing, `scatter_grad_a()` / `scatter_grad_b()` for gradient accumulation. `with_poi_from_a()` / `with_poi_from_b()` for POI resolution across models.
- **CLI `nextstat hybrid-fit`** — `--binned` (pyhf/HS3 JSON) + `--unbinned` (YAML/JSON spec). Prints summary of shared/total parameters, runs MLE via `HybridLikelihood`, outputs JSON with `hybrid: true`, `n_shared`, bestfit, uncertainties, covariance.
- **`WeightSummary` diagnostics** — `EventStore::weight_summary()` returns `WeightSummary` with ESS `(Σw)²/Σw²`, sum/min/max/mean weights, n_zero. `EventStore::sum_weights()` and `effective_sample_size()` accessors. `UnbinnedModel::channel_weight_summaries()` for per-channel reporting.
- **HS3 unbinned extension** — `Hs3UnbinnedDist` (`"nextstat_unbinned_dist"` type tag) extends the HS3 v0.2 schema with event-level channels. Other HS3 consumers see it as `Unknown` and skip gracefully. `export_unbinned_hs3()` serializes `UnbinnedSpecV0` + `UnbinnedModel` to HS3 JSON. `export_hybrid_hs3()` merges binned `HistFactoryModel` + unbinned into a single workspace with shared parameters.
- **Fused single-pass CPU NLL kernel** — `fused_kernel::fused_gauss_exp_nll` computes `log(Σ νp · fp(x))` per event inline for the Gaussian+Exponential topology, eliminating intermediate `Vec<f64>` allocations and multi-pass memory traffic. Topology detection via `UnbinnedPdf::pdf_tag()` string matching; unsupported topologies fall back to the generic multi-pass path. Adaptive parallelism: sequential for N < 8k, rayon `par_chunks(1024)` for N ≥ 8k.
- **Unbinned CPU toy parallelism heuristic** — batch toy fitting now avoids nested Rayon parallelism (toys-parallel outer loop plus events-parallel inner loop). For small `n_toys` the batch loop runs sequentially and uses event-level Rayon inside NLL; for large `n_toys` it parallelises across toys and runs the per-event loop sequentially on each worker for better cache locality and lower scheduling overhead.
- **SIMD-vectorized fused kernel (`wide::f64x4`)** — the fused event loop processes 4 events per iteration using `wide::f64x4` (AVX2 on x86, NEON on ARM). Vectorises `exp()`, `ln()`, `max()`, and FMA operations for both NLL and gradient accumulation. Scalar remainder loop for `n_events % 4`. **4–12× speedup** over generic path on x86 AVX2; **2–5×** on ARM NEON. ~770 M events/s NLL throughput at 100k events on Hetzner x86 (i5-13500).
- **Unbinned benchmark suite** — Criterion benchmarks in `ns-unbinned/benches/unbinned_nll.rs`: NLL eval, gradient, and full MLE fit for Gaussian+Exponential and Crystal Ball models at 1k/10k/100k events. ~770 M events/s NLL throughput on x86 (fused+SIMD); full 5-param fit in 637 µs at 10k events. Reference comparison doc at `docs/benchmarks/unbinned-benchmark-suite.md`.
- **Fused CrystalBall+Exponential kernel** — `fused_kernel::fused_cb_exp_nll` extends the fused single-pass path to the CB+Exp topology (second most common in HEP unbinned analyses). Scalar event loop with rayon adaptive parallelism; handles the piecewise tail/core junction with analytical gradients for all 5 shape parameters (mu, sigma, alpha, n, lambda). Topology detection via `CrystalBallPdf::pdf_tag() == "crystal_ball"`. **2.7–5.8× speedup** over generic multi-pass on Apple M5; 3.5× at 100k NLL+grad (478 µs vs 1.66 ms).
- **Unbinned fused-vs-generic benchmark mode** — `UnbinnedModel::nll_generic()` / `UnbinnedModel::grad_nll_generic()` force the generic multi-pass path for apples-to-apples CPU comparison against the default fused path. Criterion now emits paired entries (`sig_bkg_gaussian_exp` vs `sig_bkg_gaussian_exp_generic`) for both NLL and NLL+grad.

#### Cross-Vertical Statistical Features (ns-inference)

- **Gamma GLM** (`GammaRegressionModel`) — Gamma distribution with log link, shared shape parameter α. Analytical NLL and gradient. For insurance claim amounts, hospital costs, strictly positive continuous responses.
- **Tweedie GLM** (`TweedieRegressionModel`) — compound Poisson-Gamma with power `p ∈ (1, 2)`, log link. Saddle-point NLL approximation (Dunn & Smyth 2005). Handles exact zeros. For insurance aggregate claims, rainfall, zero-inflated positive continuous data.
- **GEV distribution** (`GevModel`) — Generalized Extreme Value for block maxima (Fréchet ξ>0, Gumbel ξ≈0, Weibull ξ<0). MLE via L-BFGS-B with analytical gradient. `return_level(T)` for T-block return levels (e.g. 100-year flood). For reinsurance, hydrology, climate extremes.
- **GPD distribution** (`GpdModel`) — Generalized Pareto for peaks-over-threshold. MLE with analytical gradient. `quantile(p)` for excess quantiles (VaR/ES). For tail risk in finance, reinsurance pricing.
- **Meta-analysis** (`meta_fixed`, `meta_random`) — fixed-effects (inverse-variance) and random-effects (DerSimonian–Laird) pooling. Heterogeneity: Cochran's Q, I², H², τ². Forest plot data with per-study weights and CIs. For pharma (clinical trial pooling), epidemiology, social science.
- **Competing risks** (`competing_risks`) — Aalen–Johansen cumulative incidence function (CIF) estimator with delta-method SE and confidence bands. Gray's K-sample test for comparing CIF across groups (competing-risks analogue of log-rank). Fine–Gray subdistribution hazard regression via Newton–Raphson with IPCW weights, standard errors, z-statistics, p-values. For pharma (death from disease vs other causes), insurance (lapse vs death vs disability), epidemiology.
- **EGARCH(1,1)** (`egarch11_fit`) — Nelson (1991) exponential GARCH with log-variance formulation guaranteeing h_t > 0 without parameter constraints. Magnitude + sign (leverage) decomposition. MLE via L-BFGS-B. For finance (asymmetric volatility), risk management.
- **GJR-GARCH(1,1)** (`gjr_garch11_fit`) — Glosten–Jagannathan–Runkle (1993) threshold GARCH with indicator-based leverage term (γ·ε²·I(ε<0)). Stationarity guard (α + β + γ/2 < 1). MLE via L-BFGS-B. For finance (bad-news volatility amplification), VaR/ES.
- **Group sequential testing** (`sequential`) — O'Brien–Fleming and Pocock classical boundaries via bisection. Lan–DeMets alpha-spending with OF-like, Pocock-like, and Hwang–Shih–DeCani(γ) spending functions. `sequential_test()` evaluates observed z-statistics against a design with adjusted p-values. For clinical trials (interim DSMB analyses), A/B testing (valid early stopping).
- **Chain Ladder + Mack** (`chain_ladder`) — deterministic Chain Ladder with volume-weighted development factors, cumulative factors, and projected triangle. Mack (1993) distribution-free stochastic model: per-origin and total prediction SE, coefficient of variation, normal-approximation prediction intervals, correlated total variance. Bootstrap IBNR via Pearson residual resampling with percentile CIs. For insurance/reinsurance (IBNR reserves), Solvency II / IFRS 17 compliance.

#### WASM Playground

- **Slim 454 KB binary** — stripped Arrow/Parquet from WASM build (`ns-translate` without `arrow-io`), custom `[profile.release-wasm]` with `opt-level = "z"`, `lto = "fat"`, `strip = true`, plus `wasm-opt -Oz`. Down from 5.7 MB to 454 KB.
- **UI polish** — standard site header with logo, compact single-screen layout, loading spinner on Run button, `⌘+Enter` keyboard shortcut, auto-load simple example on first visit, converged/failed badges in MLE Fit results, subtle fade-in animations on tab switch.
- **Guided examples** — 4 HistFactory examples (simple counting, shape analysis, multi-channel, discovery) and 3 GLM examples (linear, logistic, Poisson regression) with contextual descriptions. Dropdown filtered by active operation tab — only compatible examples shown.
- **Auto-run on tab switch** — switching between workspace operations (Brazil Band ↔ Profile Scan ↔ MLE Fit ↔ Hypo Test) auto-runs the new operation on the loaded workspace. GLM ↔ workspace transitions clear the editor to prevent format mismatches.
- **GLM Regression tab** — new `run_glm()` WASM endpoint exposing linear, logistic, and Poisson regression via L-BFGS-B optimizer. Model/intercept selectors in the UI, parameter table in results. Backed by `ns-inference` `LogDensityModel` trait.
- **Mass Scan (Type B Brazil Band)** — ATLAS/CMS-style exclusion plot: 95% CL upper limit on μ vs signal peak position with ±1σ/±2σ expected bands. Auto-generates mass hypotheses by redistributing signal across bins via Gaussian kernel, runs full asymptotic CLs at each point. Enable via "Mass Scan (Type B)" checkbox in Brazil Band mode.

#### Inference Server (ns-server)

- **API key authentication** — `--api-keys <file>` or `NS_API_KEYS` env var. Bearer token validation on all endpoints except `GET /v1/health`. Open mode when unconfigured (dev-friendly).
- **Per-IP rate limiting** — `--rate-limit <N>` (requests/second/IP). Token-bucket with lazy prune. Health endpoint always exempt.
- **`POST /v1/unbinned/fit`** — unbinned MLE fit endpoint. Accepts `nextstat_unbinned_spec_v0` JSON + `data_root` path, compiles model via `ns-unbinned`, runs L-BFGS fit, returns bestfit/uncertainties/NLL/covariance.
- **`POST /v1/nlme/fit`** — NLME / PK population fit endpoint. Supports `pk_1cpt` (individual 1-compartment oral PK) and `nlme_1cpt` (population NLME with log-normal random effects, per-subject eta parameters). LLOQ policies: ignore, replace_half, censored.
- **Async job system** — `POST /v1/jobs/submit` (submit long-running task → `job_id`), `GET /v1/jobs/{id}` (poll status), `DELETE /v1/jobs/{id}` (cancel), `GET /v1/jobs` (list). In-memory store with TTL pruning, cancellation tokens. Currently supports `batch_toys` task type.
- **`GET /v1/openapi.json`** — OpenAPI 3.1 specification covering all 16 endpoints with schemas, security definitions, and tags. Zero extra dependencies.

#### Survival Analysis (Non-parametric)

- **Kaplan-Meier estimator** — `nextstat.kaplan_meier(times, events, conf_level=0.95)`: non-parametric survival curve with Greenwood variance, log-log transformed confidence intervals, median survival, and number-at-risk table. Validated against R `survival::survfit`.
- **Log-rank test** — `nextstat.log_rank_test(times, events, groups)`: Mantel-Cox chi-squared test comparing survival distributions of 2+ groups with hypergeometric variance. Validated against R `survival::survdiff`.
- **CLI: `nextstat survival km`** — Kaplan-Meier from JSON input, `--conf-level` option.
- **CLI: `nextstat survival log-rank-test`** — Log-rank test from JSON input with per-group observed/expected output.

#### Subscription / Churn Vertical

- **Synthetic SaaS churn dataset** — `nextstat.churn_generate_data()`: deterministic, seeded cohort data with right-censored churn times, plan/region/usage covariates, and treatment assignment. Exponential proportional hazards DGP.
- **Cohort retention analysis** — `nextstat.churn_retention()`: stratified Kaplan-Meier curves per group + log-rank comparison in a single call.
- **Churn risk model** — `nextstat.churn_risk_model()`: Cox PH workflow returning hazard ratios with CIs.
- **Causal uplift** — `nextstat.churn_uplift()`: AIPW-based intervention impact estimation on churn with Rosenbaum sensitivity.
- **CLI: `nextstat churn generate-data`** — synthetic data generator with `--n-customers`, `--seed`, etc.
- **CLI: `nextstat churn retention`** — cohort retention analysis from JSON.
- **CLI: `nextstat churn risk-model`** — Cox PH hazard ratios from JSON.
- **CLI: `nextstat churn uplift`** — causal uplift from JSON with `--horizon`.

#### CLI

- **`nextstat mass-scan`** — batch asymptotic CLs upper limits across multiple workspaces (Type B Brazil Band). Reads a directory of workspace JSONs (one per mass/signal hypothesis), computes observed + expected (±1σ/±2σ) limits for each, outputs a single JSON with all mass points. `--labels` for custom X-axis labels. Comparable to TRExFitter `Limit` action.
- **`nextstat significance`** — discovery significance (p₀ and Z-value). Tests the background-only hypothesis (μ=0), reports observed Z, expected Z from Asimov, p₀, and q₀. Comparable to TRExFitter `GetSignificance`.
- **`nextstat goodness-of-fit`** — saturated-model goodness-of-fit test. Fits the model, computes Poisson deviance χ² between best-fit expected and observed yields, reports χ²/ndof and p-value. Comparable to TRExFitter saturated GoF.
- **`nextstat combine`** — merge multiple pyhf JSON workspaces into a single combined workspace. Channels are unioned, systematics with the same name are automatically correlated (shared NPs), measurement parameter configs are merged. `--prefix-channels` for auto-prefixing channel names on conflict. Comparable to TRExFitter `MultiFit` combination and `pyhf combine`.
- **`nextstat fit --asimov`** — blind fit on Asimov (expected) data. Generates expected yields at nominal parameters and replaces observed data before fitting. Comparable to TRExFitter `FitBlind`.
- **`nextstat viz gammas`** — gammas (staterror / Barlow-Beeston) artifact. Shows postfit γ parameter values with prefit/postfit uncertainties, channel and bin labels. Comparable to TRExFitter gammas plot.
- **`nextstat viz summary`** — multi-fit μ summary artifact. Takes multiple fit result JSONs and produces POI central values + uncertainties for each. `--labels` for custom labels. Comparable to TRExFitter summary plot in combination papers.
- **`nextstat viz pie`** — sample composition pie chart artifact per channel. Shows fraction of total expected yield per process. Works at prefit or postfit parameters. Comparable to TRExFitter pie chart.
- **`nextstat viz separation`** — signal vs background shape comparison per channel. Normalises signal and background to unit area, computes separation metric ∈ [0,1]. Auto-detects signal samples from POI normfactor, or `--signal-samples` for explicit control. Optional `--histfactory-xml` for bin edges. Comparable to TRExFitter separation plot.
- **`nextstat preprocess smooth`** — native Rust 353QH,twice smoothing for HistoSys templates. Smooths deltas (variation − nominal) preserving nominal shape, optional `--max-variation` cap. No Python dependency. Comparable to ROOT `TH1::Smooth`.
- **`nextstat preprocess prune`** — native Rust pruning of negligible systematics. Removes HistoSys/NormSys modifiers with max |δ/nominal| below `--threshold` (default 0.5%). No Python dependency. Comparable to TRExFitter pruning.

#### pyhf Feature Parity

- **`Workspace::prune()`** — remove channels, samples, modifiers, or measurement POIs by name. Mirrors `pyhf.Workspace.prune()`.
- **`Workspace::rename()`** — rename channels, samples, modifiers, or measurement POIs. Mirrors `pyhf.Workspace.rename()`.
- **`Workspace::sorted()`** — return a workspace with channels, samples, and modifiers sorted by name. Mirrors `pyhf.Workspace.sorted()`.
- **`Workspace::digest()`** — SHA-256 content digest of the canonicalised workspace JSON. Mirrors `pyhf.Workspace.digest()`.
- **`Workspace::combine()`** — merge two workspaces with configurable channel join semantics (`None`, `Outer`, `LeftOuter`, `RightOuter`). Mirrors `pyhf.Workspace.combine()`.
- **`pyhf::simplemodels::uncorrelated_background()`** — quick workspace builder for signal + background with uncorrelated (shapesys) uncertainties. Mirrors `pyhf.simplemodels.uncorrelated_background()`.
- **`pyhf::simplemodels::correlated_background()`** — quick workspace builder for signal + background with correlated (histosys) uncertainties. Mirrors `pyhf.simplemodels.correlated_background()`.
- **`pyhf::xml_export::workspace_to_xml()`** — export a pyhf workspace to HistFactory XML format (structural; ROOT histogram file writing is a future enhancement). Mirrors `pyhf json2xml`.
- **HistoSys interpolation code2** — quadratic interpolation with linear extrapolation for HistoSys modifiers. Scalar, SIMD (`histosys_code2_delta_accumulate`), and tape-based AD paths. Completes code0/code2/code4p coverage.
- **Test statistics `t_μ` and `t̃_μ`** — `TestStatistic::TMu` (Eq. 8, arXiv:1007.1727) and `TestStatistic::TMuTilde` (Eq. 11) added alongside existing `q_μ` / `q̃_μ`.
- **`OptimizerStrategy` presets** — `Default` (scipy-like), `MinuitLike` (smooth logistic bounds, higher precision), `HighPrecision` (tightest tolerances). `OptimizerConfig::from_strategy()` constructor.
- **`docs/pyhf-parity.md`** — comprehensive feature matrix documenting NextStat vs pyhf parity status across all workspace operations, modifier types, interpolation codes, test statistics, optimizer backends, and beyond-pyhf capabilities.

#### Econometrics & Causal Inference (Phase 12)

- **Panel fixed-effects regression** — entity-demeaned ("within") OLS with Liang–Zeger cluster-robust (HC0 sandwich) standard errors. Small-sample correction `G/(G-1) × (N-1)/(N-K)`. `panel_fe_fit()` in Rust, `nextstat.panel_fe()` in Python.
- **Difference-in-Differences (DiD)** — canonical 2×2 estimator (`did_canonical`) and multi-period event-study specification with leads/lags (`event_study`). Two-way FE demeaning, cluster-robust SE, 95% CI. Python: `nextstat.did()`, `nextstat.event_study()`.
- **Instrumental Variables / 2SLS** — standard two-stage least squares with first-stage F-statistic, partial R², and Stock–Yogo 10% weak-instrument test. Supports cluster-robust SE. Python: `nextstat.iv_2sls()`.
- **AIPW (Doubly Robust)** — Augmented Inverse Probability Weighting estimator for ATE. Influence-function SE, propensity score trimming. Python: `nextstat.aipw_ate()`.
- **Rosenbaum sensitivity analysis** — Wilcoxon signed-rank bounds for matched-pair sensitivity to unobserved confounding. Reports critical Γ at which significance is lost. Python: `nextstat.rosenbaum_bounds()`.
- **`docs/references/econometrics.md`** — reference documentation with code examples, assumptions table, and limitations.

#### API Stabilization

- **ns-core re-exports** — `LogDensityModel`, `PoiModel`, `FixedParamModel`, `PreparedNll`, `PreparedModelRef` now re-exported from crate root (`ns_core::LogDensityModel` works without `ns_core::traits::*`).
- **Deprecated `Model` trait** — superseded by `LogDensityModel`. Zero external users; will be removed in 1.0.
- **Deprecated `FitResult::n_evaluations()`** — use the `n_iter` field directly. Zero callers.
- **ns-inference re-exports** — added `scan`, `scan_metal`, `NegativeBinomialRegressionModel`, `QualityGates`, `compute_diagnostics`, `quality_summary` to crate root.
- **`nextstat.unbinned.UnbinnedAnalysis`** — high-level workflow wrapper over `UnbinnedModel`. `UnbinnedAnalysis.from_config(path)` compiles the model; `.fit()`, `.scan(mu_values)`, `.hypotest(mu_test)`, `.hypotest_toys(poi_test)`, `.ranking()`, `.summary()` delegate to the underlying inference functions. `.with_fixed_param()` and `.parameter_index()` for parameter manipulation.
- **Python `__all__` completeness** — added `volatility`, `UnbinnedModel`, `HybridModel`, `unbinned_hypotest`, `unbinned_hypotest_toys`, `unbinned_profile_scan`, `unbinned_ranking`, `set_threads`, `from_parquet_with_modifiers` to `nextstat.__all__`.

### Fixed

- **L-BFGS steepest-descent fallback** — optimizer now correctly falls back to steepest descent when the L-BFGS update produces a non-descent direction, preventing convergence failures on ill-conditioned problems.

## [0.9.0] — 2026-02-09

### Added

#### Neural Density Estimation

- **Flow PDF** — ONNX-backed normalizing flow as an unbinned PDF. Loads pre-trained flows from `flow_manifest.json` + ONNX models. Supports unconditional and conditional flows with nuisance parameters as context. Spec YAML: `type: flow` / `type: conditional_flow`. Feature-gated: `--features neural`.
- **DCR Surrogate** — neural Direct Classifier Ratio surrogate replacing binned template morphing. Drop-in replacement for morphing histograms — smooth, continuous, bin-free systematic morphing trained via FAIR-HUC protocol. Spec YAML: `type: dcr_surrogate`.
- Unbinned spec YAML supports `flow`, `conditional_flow`, and `dcr_surrogate` PDF types with automatic feature gating.
- **Normalization verification** — Gauss-Legendre quadrature (orders 32–128) for normalization verification and correction of neural PDFs.
- **Training helpers** — Python scripts for flow training (zuko NSF + ONNX export), DCR distillation from HistFactory templates, and validation (normalization, PIT/KS, closure checks).
- **Python bindings for FlowPdf / DcrSurrogate** — `nextstat.FlowPdf` and `nextstat.DcrSurrogate` classes exposed in `ns-py` behind `--features neural`. Standalone ONNX flow evaluation from Python: `from_manifest()`, `log_prob_batch()`, `update_normalization()`, `validate_nominal_normalization()`.

#### TRExFitter Importer

- **Full config surface** — `ns-translate` TRExFitter importer now parses the complete config surface:
  - New block types: `Fit` (FitType/FitRegion/FitBlind/NumCPU/POIAsimov/UseMinos), `Limit` (LimitType/LimitBlind/ConfidenceLevel), `Significance` (SignificanceBlind).
  - Job-level: `Lumi`, `LumiRelErr`, `MCstatThreshold`, `SystPruningShape`, `SystPruningNorm`, `DebugLevel`, `BlindingType`/`BlindingThreshold`.
  - Region: `Type` (SIGNAL/CONTROL/VALIDATION), `Label`, `ShortLabel`, `TexLabel`, `LogScale`, `Rebin`, `MCweight`, `AutomaticDropBins`.
  - Sample: `Title`, `Group`, `NormalizedByTheory`, `LumiScale`, `Exclude`, `IgnoreSelection`, `FillColor`/`LineColor`, `SeparateGammas`, `UseSystematic`.
  - Systematic: `NuisanceParameter` (custom NP name), `Symmetrisation`, `IsFreeParameter`, `Decorrelate`, `Exclude`/`ExcludeRegion`, `Category`/`SubCategory`, `ReferenceSample`, `ScaleUp`/`ScaleDown`, `PreSmoothing`, `SmoothingOption`.
  - NormFactor: `Regions` (region-scoped), `Nominal`/`Min`/`Max`, `Constant`, `Expression`, `Category`.
- **Workspace building** — `NuisanceParameter` wires custom NP names into modifiers; `Exclude`/`ExcludeRegion` on Systematic and `Exclude` on Sample are respected during workspace construction.
- **Coverage report** — expanded `known_global` and `known_in_block` recognize 150+ TREx config keys including cosmetic/presentation attributes, eliminating false-positive unknown-attr warnings on real-world configs.

#### Documentation

- **Unbinned spec reference** — `docs/references/unbinned-spec.md`: dedicated human-readable reference for `nextstat_unbinned_spec_v0` covering all PDF types, yield expressions, rate modifiers (NormSys + WeightSys), per-event systematics, neural PDFs, and GPU acceleration constraints.

#### GPU Acceleration

- **CUDA (NVIDIA, f64)** — fused NLL+gradient kernel covering all 7 HistFactory modifier types in a single launch. Lockstep batch optimizer fits thousands of toys in parallel. Dynamic loading via cudarc — binary works without CUDA installed.
  - `nextstat.fit(model, device="cuda")`, `--gpu cuda` CLI flag
- **Metal (Apple Silicon, f32)** — same fused kernel in MSL. Zero-copy unified memory. NLL parity vs CPU f64: 1.27e-6 relative diff.
  - `nextstat.fit_toys_batch_gpu(model, ..., device="metal")`, `--gpu metal` CLI flag
- **Apple Accelerate** — vDSP/vForce vectorized NLL on macOS. <5% overhead vs naive summation.
- **GPU-resident toy pipeline (CUDA)** — `--gpu-sample-toys` now keeps sampled events on the GPU device, eliminating the D2H+H2D round-trip of the large `obs_flat` buffer between sampler and batch fitter.
- **Flow CUDA EP + zero-copy reduction (CUDA)** — Flow `log_prob` can run on ONNX Runtime CUDA EP with I/O binding so the `float` output stays device-resident and is consumed directly by the GPU NLL reducer (`flow_nll_reduce_f32`, `CudaFlowNllAccelerator::nll_device_ptr_f32`).
- **Unbinned GPU WeightSys** — `weightsys` rate modifier now lowered to CUDA/Metal kernels (code0/code4p interpolation). Spec YAML: `type: weightsys`, `param`, `lo`, `hi`, optional `interp_code`.
- **Unbinned observed-data weights (CPU/GPU)** — `channels[].data.weight` can provide finite, non-negative per-event frequency weights (multiply each event’s `-log L` contribution) for both CPU and CUDA/Metal `--gpu` paths.
- **CPU batch toys** — Rayon-parallel toy fitting with per-thread tape reuse, seed-based reproducibility.
- **Reverse-mode tape** — faster gradient computation with reduced memory allocation.

#### Differentiable Analysis (PyTorch)

- Zero-copy CUDA kernel reads signal histogram from a PyTorch tensor and writes dNLL/dsignal directly into the grad buffer — no device-host roundtrip.
- `DifferentiableSession`: NLL + signal gradient at fixed nuisance parameters.
- `ProfiledDifferentiableSession`: profiled test statistics with envelope-theorem gradients — enables NN → signal histogram → profiled CLs → loss.
- `nextstat.torch` Python module: `NextStatNLLFunction`, `NextStatProfiledQ0Function` (autograd), `NextStatLayer(nn.Module)`.
- `profiled_zmu_loss()` — Zμ loss wrapper (sqrt(qμ) with numerical stability) for signal-strength optimization.
- `SignificanceLoss(model)` — ML-friendly class wrapping profiled −Z₀. Init once, call per-batch: `loss_fn(signal_hist).backward()`.
- `SoftHistogram` — differentiable binning (Gaussian KDE / sigmoid): NN classifier scores → soft histogram → `SignificanceLoss`.
- `batch_profiled_q0_loss()` — profiled q₀ for a batch of signal histograms (ensemble training).
- `signal_jacobian()`, `signal_jacobian_numpy()` — direct ∂q₀/∂signal without autograd for SciPy bridge and fast pruning.
- `as_tensor()` — DLPack array-API bridge: JAX, CuPy, Arrow, NumPy → `torch.Tensor`.
- `nextstat.mlops` — fit metrics extraction for W&B / MLflow / Neptune: `metrics_dict(result)`, `significance_metrics(z0)`, `StepTimer`.
- `nextstat.interpret` — systematic-impact ranking as Feature Importance: `rank_impact(model)`, `rank_impact_df()`, `plot_rank_impact()`.
- **`nextstat.tools`** — LLM tool definitions (OpenAI function calling, LangChain, MCP) for 9 operations: fit, hypotest, hypotest_toys, upper_limit, ranking, discovery_asymptotic, scan, workspace_audit, read_root_histogram. `get_toolkit()` returns JSON Schema; `execute_tool(name, args)` bridges agent calls to NextStat.
- **`nextstat.distill`** — surrogate training dataset generator. `generate_dataset(model, n_samples=100k, method="sobol")` produces `(params, NLL, gradient)` tuples. Export to PyTorch `TensorDataset`, `.npz`, or Parquet. Built-in `train_mlp_surrogate()` with Sobolev loss.
- Fit convergence check: returns error if GPU profile fit fails to converge.

#### Gymnasium RL Environment

- `nextstat.gym` — optional Gymnasium/Gym wrapper treating a HistFactory workspace as an RL/DOE environment.
- Propose updates to a sample's nominal yields, receive a NextStat metric as reward (NLL, q₀, Z₀, qμ, Zμ).
- `make_histfactory_env()` factory with configurable `reward_metric`, `action_mode` (additive/logmul), `action_scale`.
- Compatible with `gymnasium` (preferred) and legacy `gym`.

#### Deterministic Validation

- **EvalMode** — process-wide flag: **Parity** (Kahan summation, single-threaded, bit-exact) vs **Fast** (default, SIMD/GPU, multi-threaded).
- CLI: `--parity` · Python: `nextstat.set_eval_mode("parity")`.
- 7-tier tolerance contract vs pyhf (per-bin ~1e-14 worst case).

#### Native ROOT I/O

- **TTree reader** — mmap file access, native binary deserialization, basket decompression (zlib/LZ4/ZSTD) with rayon-parallel extraction. 9 leaf types + jagged branches.
- **Expression engine** — bytecode-compiled, vectorized. Full grammar: arithmetic, comparisons, boolean logic, ternary, builtins. Dynamic jagged indexing (`jet_pt[idx]`) follows ROOT/TTreeFormula convention. Python wrapper: `nextstat.analysis.expr_eval`.
- **Histogram filler** — single-pass with selection cuts, weights, variable binning.
- **Unsplit vector branch decoding** — best-effort decoding for `std::vector<T>` branches without offset tables.
- **~8.5× faster** than uproot+numpy on the full pipeline.

#### Ntuple-to-Workspace Pipeline

- `NtupleWorkspaceBuilder`: ROOT ntuples → HistFactory `Workspace` via fluent Rust API.
- Per-sample modifiers: NormFactor, NormSys, WeightSys, TreeSys, HistoSys, StatError.
- Produces the same `Workspace` struct as the pyhf JSON path — no ROOT C++ dependency.

#### TRExFitter Interop

- `nextstat import trex-config` — import TRExFitter `.config` into pyhf JSON workspace.
- `nextstat build-hists` — run NTUP pipeline, write `workspace.json`.
- **HIST mode** — read pre-built ROOT histograms (`ReadFrom: HIST`) alongside NTUP.
- **Analysis Spec v0** (YAML + JSON Schema) — `nextstat run <spec.yaml>` orchestrates import/fit/scan/report.
- Jagged column support and TRExFitter-style expression compatibility.

#### Systematics Preprocessing

- **Smoothing**: 353QH,twice algorithm (ROOT `TH1::Smooth` equivalent) + Gaussian kernel.
- **Pruning**: shape, norm, and overall pruning with audit trail.
- **`nextstat preprocess`** CLI with declarative YAML config and content-hash caching.
- Recommended order: hygiene → symmetrize → smooth → prune.

#### HistFactory Enhancements

- **HS3 v0.2 ingestion** — load HS3 JSON workspaces (ROOT 6.37+) natively. Auto-detects format (pyhf vs HS3) at load time.
- **HS3 roundtrip export** — export `HistFactoryModel` back to HS3 JSON with bestfit parameter points.
- Python: `HistFactoryModel.from_workspace()` (auto-detect), `HistFactoryModel.from_hs3(json_str)`. CLI: auto-detection in `nextstat fit`, `nextstat scan`.
- HS3 inputs use ROOT HistFactory defaults (NormSys Code1, HistoSys Code0). For pyhf JSON inputs, NextStat defaults to smooth interpolation (NormSys Code4, HistoSys Code4p); use `--interp-defaults pyhf` (CLI) or `from_workspace_with_settings(Code1, Code0)` (Rust) for strict pyhf defaults.
- HEPData patchset support: `nextstat import patchset`, Python `nextstat.apply_patchset()`.
- **Arrow / Polars ingestion** — `nextstat.from_arrow(table)` creates a HistFactoryModel from PyArrow Table, RecordBatch, or any Arrow-compatible source (Polars, DuckDB, Spark). `nextstat.from_parquet(path)` reads Parquet directly.
- **Arrow export** — `nextstat.to_arrow(model, what="yields"|"params")` exports expected yields or parameter metadata as a PyArrow Table. Uses Arrow IPC bridge (zero pyo3 version conflicts).
- **`nextstat convert`** — ROOT TTree → Parquet CLI. Observable bounds (`--observable mass:100:180`), selection expressions, per-event weights, and `--max-events` truncation. Writes `nextstat_unbinned_events_v1` schema with Zstd compression.
- **Modifier table schema (v2)** — binned Parquet now round-trips full HistFactory workspaces: yields table + modifiers table. All 7 modifier types (normfactor, normsys, histosys, shapesys, shapefactor, staterror, lumi). `workspace_to_parquet()` / `from_parquet_with_modifiers()`.
- **Fit results → Parquet** — `toy_results_to_parquet()` and `scan_points_to_parquet()` export batch toy fits and profile scan points as compact Parquet files queryable via DuckDB/Polars.
- **EventStore Parquet I/O** — `EventStore::from_parquet(path)` / `EventStore::to_parquet(path)` for unbinned event data with observable metadata preserved in Parquet footer key-value pairs.
- **High-performance Parquet reads** — `memmap2`-backed zero-copy reader (`read_parquet_mmap`), row group predicate pushdown via min/max statistics (`ColumnBound` + `read_parquet_mmap_filtered`), and parallel row group decode with rayon (`read_parquet_mmap_parallel`). Criterion benchmark suite in `benches/parquet_read.rs`.
- **Direct-to-GPU Parquet** — `read_parquet_events_soa()` produces GPU-ready SoA f64 layout combining mmap + pushdown + parallel decode. `cuda_parquet::upload_events_to_cuda()` uploads SoA directly to `CudaSlice<f64>`. `metal_parquet::upload_events_to_metal()` converts f64→f32 and uploads to `MTLBuffer`. Both support per-event weights and observable bounds.
- **ConstraintTerm semantics** — LogNormal alpha-transform (`normsys_alpha_effective`), Gamma constraint for ShapeSys, Uniform and NoConstraint handling. Parsed from `<ConstraintTerm>` metadata in HistFactory XML.

#### Unbinned Likelihood

- **Product PDF** — joint likelihood over independent observables: `log p(x,y) = log p₁(x) + log p₂(y)`. Enables multi-observable unbinned fits without manual factorization.
- **Spline PDF** — monotonic cubic Hermite (Fritsch–Carlson) interpolation from user-specified knot positions and density values. Analytically normalized, inverse-CDF sampling for toys.
- **Multi-dimensional KDE** — 2-D/3-D Gaussian kernel density estimator with Silverman bandwidth, truncated on bounded observable support.
- **ARGUS PDF** — ARGUS background shape for B-meson spectroscopy. Gauss-Legendre normalization on bounded support.
- **Voigtian PDF** — pseudo-Voigt (Thompson–Cox–Hastings) resonance line shape. Gaussian ⊗ Breit-Wigner convolution for resonance + detector resolution modeling.
- Normalization integrals are cached across optimizer iterations, avoiding redundant quadrature when parameters haven't changed.
- **Flow PDF integration tests** — ONNX-backed normalizing flow verified against analytical standard normal: log-prob, sampling, normalization, and per-event baseline comparison with parametric Gaussian.
- **GPU flow NLL reduction** — CUDA kernel for extended unbinned likelihood from externally-computed log-prob values (flow PDFs). Supports multi-process logsumexp reduction, Gaussian constraints, and both host-upload and device-resident (ONNX CUDA EP) input paths.
- **GPU flow session** — orchestrates flow PDF evaluation (CPU or CUDA EP) with GPU NLL reduction. Central finite-difference gradient, yield computation from parameter vector, and Gaussian constraint handling.

#### Report System

- `nextstat report` — generates distributions, pulls, correlations, yields (.json/.csv/.tex), and uncertainty ranking from a workspace.
- Python rendering: multi-page PDF + per-plot SVGs via matplotlib.
- `--blind` flag masks observed data for unblinded regions.
- `--deterministic` for stable JSON key ordering.
- **`nextstat validation-report`** — unified validation artifact combining Apex2 results with workspace fingerprints. Outputs `validation_report.json` (schema `validation_report_v1`) with dataset SHA-256, model spec, environment, regulated-review notes, and per-suite pass/fail summary. Optional `--pdf` renders an 8-page audit-ready PDF via matplotlib, including a dedicated Pharma PK/NLME page with parameter recovery and prediction error worst-case tables.

#### Survival Analysis

- Parametric models: Exponential, Weibull, LogNormal AFT (with right-censoring).
- Cox Proportional Hazards: Efron/Breslow ties, robust sandwich SE, Schoenfeld residuals.
- Python: `nextstat.survival.{exponential,weibull,lognormal_aft,cox_ph}.fit(...)`.
- CLI: `nextstat survival fit`, `nextstat survival predict`.

#### Linear Mixed Models

- Analytic marginal likelihood (random intercept, random intercept + slope).
- Laplace approximation for approximate posteriors.
- Python: `nextstat.LmmMarginalModel(...)`.

#### Ordinal Models

- Ordered logit/probit with stable cutpoint parameterization.
- Python: `nextstat.ordinal.ordered_logit.fit(...)`, `nextstat.ordinal.ordered_probit.fit(...)`.

#### Econometrics & Causal Inference

- **Panel FE** with 1-way cluster SE.
- **DiD TWFE** + event-study helpers.
- **IV / 2SLS** with weak-IV diagnostics (first-stage F, partial R²).
- **AIPW** for ATE/ATT + E-value helper. Propensity scores, IPW weights, overlap diagnostics. Python: `nextstat.causal.aipw()`, `nextstat.causal.propensity_scores()`.
- **GARCH / Stochastic Volatility** — GARCH(1,1) and stochastic volatility models for financial time series. CLI: `nextstat volatility fit`, `nextstat volatility forecast`. Python: `nextstat.volatility.garch()`, `nextstat.volatility.sv()`.

#### Pharmacometrics

- RK4 integrator for linear ODE systems.
- One-compartment oral PK model with LLOQ censoring.
- NLME extension with per-subject random effects.
- **Error model enum** — `ErrorModel::Additive`, `Proportional`, `Combined(σ_add, σ_prop)` with variance, NLL, and gradient helpers. All PK models now accept an error model instead of a fixed σ. Backward-compatible constructors preserved.
- **2-compartment PK models** — `TwoCompartmentIvPkModel` (IV bolus, 4 params: CL, V1, V2, Q) and `TwoCompartmentOralPkModel` (oral first-order absorption, 5 params: CL, V1, V2, Q, Ka). Analytical bi/tri-exponential solutions with eigenvalue decomposition. LLOQ handling and all three error models supported.
- **Dosing regimen** — `DosingRegimen` struct supporting IV bolus, oral, and IV infusion dose events. Single-dose, repeated-dose, and mixed-route schedules. Concentration-time profiles via superposition for 1-compartment and 2-compartment models. Closed-form infusion solutions (during + post-infusion phases).
- **NONMEM dataset reader** — `NonmemDataset::from_csv()` parses standard NONMEM-format CSV files (ID, TIME, DV, AMT, EVID, MDV, CMT, RATE columns). Auto-infers EVID/MDV when omitted. Converts dosing records to `DosingRegimen` per subject (CMT=1 → oral, CMT=2 → IV bolus/infusion). Extracts observation data for model fitting.
- **FOCE/FOCEI estimation** — `FoceEstimator` implements First-Order Conditional Estimation for population PK. Two-level optimization: per-subject ETA optimization (damped Newton-Raphson) + population parameter updates (EM-like alternation with analytical Ω update). Laplace approximation with ridge-regularized Hessian. `FoceEstimator::focei()` / `FoceEstimator::foce()` constructors.
- **Correlated random effects** — `OmegaMatrix` stores the full Ω variance–covariance matrix via Cholesky factor L (Ω = L·Lᵀ, always positive-definite). Constructors: `from_diagonal()`, `from_correlation()`, `from_covariance()`, `from_cholesky()`. Efficient `inv_quadratic()` and `log_det()` via forward-substitution. `FoceEstimator::fit_1cpt_oral_correlated()` fits with full Ω; `FoceResult` now includes `omega_matrix` and `correlation` fields.
- **Stepwise Covariate Modeling (SCM)** — `ScmEstimator` implements forward selection + backward elimination of covariate–parameter relationships using ΔOFV (χ²(1) likelihood ratio test). Power, proportional, and exponential covariate relationships on CL/V/Ka. Configurable α thresholds (default: forward 0.05, backward 0.01). Returns full audit trace with per-step ΔOFV, p-values, and coefficients.
- **VPC and GOF diagnostics** — `vpc_1cpt_oral()` runs Visual Predictive Checks: simulates N replicates from fitted model, bins by time, computes observed vs simulated quantile prediction intervals. `gof_1cpt_oral()` computes PRED, IPRED, IWRES, and CWRES per observation. Configurable quantiles, bin count, and PI level.
- **Pharma benchmark suite** — synthetic datasets mimicking Warfarin (32 subjects, rich sampling), Theophylline (12 subjects, sparse), and Phenobarbital (40 neonates). Parameter recovery validation with FOCE fit, GOF diagnostics, and VPC for each. Includes correlated-Ω Warfarin variant (CL–V correlation). `cargo test --test pharma_benchmark`.
- **NLME artifact schema (v2.0.0)** — `NlmeArtifact` wraps all estimation results (fixed effects, random effects covariance + correlation, SCM trace, VPC/GOF) into a single JSON-serializable structure. CSV exports for fixed effects, random effects, GOF records, VPC bins, and SCM trace. Optional sections omitted when unused.
- **Run bundle (provenance)** — `RunBundle` captures NextStat version, git revision, Rust toolchain, OS/CPU, random seeds, dataset provenance (label + hash + counts), and reference tool versions. Attached to `NlmeArtifact` for reproducible benchmark runs.
- **SAEM algorithm** — `SaemEstimator` implements Stochastic Approximation EM for NLME (Monolix-class). Metropolis-Hastings E-step with adaptive proposal variance, stochastic approximation with configurable burn-in/estimation phases, closed-form M-step for θ and Ω. Returns `FoceResult`-compatible output plus `SaemDiagnostics` (acceptance rates, OFV trace). Supports diagonal and correlated Ω.
- **PD models** — `EmaxModel` (direct effect), `SigmoidEmaxModel` (Hill equation with configurable γ), and `IndirectResponseModel` (Types I–IV: inhibit/stimulate production/loss). ODE-based IDR models use the adaptive RK45 solver. `PkPdLink` interpolates PK concentration profiles for PD integration. All models include `predict()`, `gradient()`, and `nll()` methods.
- **Adaptive ODE solvers** — `rk45()` (Dormand–Prince 4(5) with PI step-size control) for non-stiff PK/PD systems and `esdirk4()` (L-stable SDIRK2 with Newton iteration) for stiff systems (transit compartments with ktr > 100). Generic `OdeSystem` trait for user-defined RHS. `solve_at_times()` convenience for output interpolation.

#### Applied Statistics API

- Formula parsing + deterministic design matrices (`nextstat.formula`).
- `from_formula` wrappers for all GLM and hierarchical builders.
- Wald summaries + robust covariance (HC0-HC3, 1-way cluster).
- scikit-learn adapters: `NextStatLinearRegression`, `NextStatLogisticRegression`, `NextStatPoissonRegressor`.
- Missing-data policies: `drop_rows`, `impute_mean`.

#### WASM Playground

- Browser-based inference via `wasm-bindgen`: `fit_json()`, `hypotest_json()`, `upper_limit_json()`.
- Drag-and-drop `workspace.json` → asymptotic CLs Brazil bands. No Python, no server.

#### Visualization

- `plot_cls_curve()`, `plot_brazil_limits()`, `plot_profile_curve()`.
- `nextstat viz distributions`, `viz pulls`, `viz corr`, `viz ranking` subcommands.
- Kalman: `plot_kalman_states()`, `plot_forecast_bands()`.

#### Pure-Rust Zstd Codec (`ns-zstd`)

- **`ns-zstd` crate** — pure-Rust Zstd decompressor and compressor for ROOT file I/O. Zero C dependency — enables WASM and embedded targets. Supports compression levels 1–19 with FSE (Finite State Entropy) and Huffman entropy coding. Decode output matches `libzstd` byte-for-byte (verified via fixture tests). Hash-chain match finder with configurable search depth.

#### R Bindings

- **`nextstat` R package** — native R interface via `extendr` (`bindings/ns-r/`). 11 exported functions: `nextstat_fit()`, `nextstat_hypotest()`, `nextstat_upper_limit()` (HistFactory), `nextstat_glm_logistic()`, `nextstat_glm_poisson()`, `nextstat_glm_negbin()` (GLM), `nextstat_kalman()`, `nextstat_garch()`, `nextstat_sv()` (time series), `ns_normal_logpdf()`, `ns_ols_fit()` (core).
- **CRAN preparation** — roxygen2 documentation for all 11 functions, testthat test suite (48 assertions), `configure` script (Rust ≥ 1.85 detection), getting-started vignette, NEWS.md, CRAN-compliant LICENSE and DESCRIPTION.
- **Pharmacometrics R functions** — `ns_foce()` (FOCE/FOCEI population estimation), `ns_saem()` (SAEM with acceptance rates and OFV trace), `ns_vpc()` (Visual Predictive Check), `ns_gof()` (Goodness of Fit with PRED/IPRED/IWRES/CWRES), `ns_idr()` (indirect response model simulation). All backed by Rust via extendr.

#### CLI & Infrastructure

- Structured logging (`--log-level`), reproducible run bundles (`--bundle`).
- `fit()` supports `init_pars=` for warm-start MLE.
- CI: pyhf parity gate on push/PR, TREx baseline refresh (nightly), HEPData workspace tests.
- Apex2 validation: NLL parity, bias/pulls regression, SBC calibration, NUTS quality gates.
- **`nextstat-server`** — self-hosted REST API for shared GPU inference. `POST /v1/fit` (workspace → FitResult), `POST /v1/ranking` (NP impacts), `GET /v1/health`. `--gpu cuda|metal`, `--port`, `--host`, `--threads`.
- **`nextstat.remote`** — pure-Python thin client (httpx). `client = nextstat.remote.connect("http://gpu-server:3742")`, then `client.fit(workspace)`, `client.ranking(workspace)`, `client.health()`. Typed dataclass results.
- **Batch API** — `POST /v1/batch/fit` fits up to 100 workspaces in one request; `POST /v1/batch/toys` runs GPU-accelerated toy fitting (CUDA/Metal/CPU). `client.batch_fit(workspaces)`, `client.batch_toys(workspace, n_toys=1000)`.
- **Model cache** — `POST /v1/models` uploads a workspace and returns a `model_id` (SHA-256); subsequent `/v1/fit` and `/v1/ranking` calls accept `model_id=` to skip re-parsing. `GET /v1/models`, `DELETE /v1/models/:id`. LRU eviction (64 models).
- **Docker & Helm** — multi-stage `Dockerfile` for CPU and CUDA builds, Helm chart with health probes, GPU resource requests, configurable replicas.

### Fixed

- End-to-end discovery script (`e2e_discovery.py`): fixed `--no-deterministic` flag handling. Script now correctly writes `summary.json` and `summary.md`.
- CUDA batch toys (`--gpu cuda`) crash when some toys converge before others.
- GPU profiled session (`ProfiledDifferentiableSession`) convergence failure near parameter bounds.
- Optimizer early-stop with negative NLL (`target_cost(0.0)` removed).
- `kalman_simulate()`: `init="sample|mean"` and `x0=...` support.
- StatError: incorrect `sqrt(sumw2)` propagation with zero nominal counts.
- Metal GPU: scratch buffer reuse (~40% less allocation overhead).
- HistFactory XML: strip `<!DOCTYPE>` declarations before parsing.
- CUDA/Metal signal gradient race condition: incorrect accumulation when multiple samples contribute to the same bin.
- 10 missing Python re-exports in `__init__.py`: `has_metal`, `read_root_histogram`, `workspace_audit`, `cls_curve`, `profile_curve`, `kalman_filter/smooth/em/forecast/simulate`.

---

## [0.1.0] — 2026-02-05

Initial public release.

### Core Engine

- HistFactory workspace data model with full pyhf JSON compatibility.
- Poisson NLL with all modifier types (histosys, normsys, shapesys, staterror, lumi) + Barlow-Beeston.
- SIMD-accelerated NLL via `wide::f64x4`.
- Automatic differentiation: forward-mode (dual numbers) and reverse-mode (tape AD).

### Frequentist Inference

- MLE via L-BFGS-B with Hessian-based uncertainties.
- Asymptotic CLs hypothesis testing (q-tilde test statistic).
- Profile likelihood scans, CLs upper limits (bisection + linear scan), Brazil bands.
- Batch MLE, toy studies, nuisance parameter ranking.

### Bayesian Sampling

- No-U-Turn Sampler (NUTS) with dual averaging.
- HMC diagnostics: divergences, tree depth, step size, E-BFMI.
- Rank-normalized folded R-hat + improved ESS (Geyer IMS + variogram).
- Python: `sample()` returning ArviZ-compatible dict.

### Regression & GLM

- Linear, logistic, Poisson, negative binomial regression.
- Ridge regression (MAP/L2), separation detection, exposure/offset support.
- Cross-validation (`kfold_indices`, `cross_val_score`) and metrics (RMSE, log-loss, Poisson deviance).

### Hierarchical Models

- Random intercepts/slopes, correlated effects (LKJ + Cholesky), non-centered parameterization.
- Posterior Predictive Checks.

### Time Series

- Linear-Gaussian Kalman filter + RTS smoother.
- EM parameter estimation, multi-step-ahead forecasting with prediction intervals.
- Local-level, local-trend, AR(1) builders. Missing observation handling.

### Probability Distributions

- Normal, StudentT, Bernoulli, Binomial, Poisson, NegativeBinomial, Gamma, Exponential, Weibull, Beta.
- Bijector/transform layer: Identity, Exp, Softplus, Sigmoid, Affine.

### Visualization

- Profile likelihood curves and CLs Brazil band plots.
- CLI: `viz profile`, `viz cls`. Python: `viz_profile_curve()`, `viz_cls_curve()`.

### Python Bindings & CLI

- `nextstat` Python package (PyO3/maturin) with `Model`, `FitResult` classes.
- `nextstat` CLI: `fit`, `hypotest`, `upper-limit`, `scan`, `version`.
- CI workflows + GitHub release pipeline (multi-arch wheels + CLI binary).

### Validation (Apex2)

- Master report aggregator with NLL parity, GLM benchmarks, bias/pulls regression, SBC calibration, NUTS quality gates.
- Nightly slow CI workflow.
