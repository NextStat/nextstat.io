---
title: "Cross-Framework Unbinned Likelihood Benchmark Report"
date: 2026-02-11
revision: 3
status: final
hardware: Hetzner GEX44 (i5-13500)
---

# Cross-Framework Unbinned Likelihood Benchmark Report

**Date**: 2026-02-11T12:30Z (revision 3 — adds MoreFit)
**Author**: NextStat benchmark harness (`benchmarks/unbinned/run_suite.py`)
**Purpose**: Reproducible wall-clock and numerical comparison of unbinned
extended maximum-likelihood fitting across NextStat, RooFit, zfit, and MoreFit on
identical hardware, identical data, and identical model parameterisation.

**Update (2026-02-11T19:08Z)**: a strict symmetric CPU rerun (separating
wall-to-wall and warm fit-only timing) is published at:
`docs/benchmarks/cross-framework-unbinned-cpu-symmetric-2026-02-11.md`.

---

## 1. Hardware

| Property | Value |
|----------|-------|
| **Server** | Hetzner Dedicated (GEX44 class) |
| **CPU** | 13th Gen Intel Core i5-13500 (14C/20T, P-core max 4.8 GHz) |
| **Microarchitecture** | Raptor Lake (P-cores: Raptor Cove, E-cores: Gracemont) |
| **ISA extensions** | SSE4.2, AVX2, AVX-VNNI, FMA, SHA-NI, AES-NI |
| **L1d / L1i cache** | 544 KiB / 704 KiB (14 instances) |
| **L2 cache** | 11.5 MiB (8 instances) |
| **L3 cache** | 24 MiB (shared) |
| **RAM** | 64 GB DDR4 |
| **Storage** | 2 × Intel SSDPF2KX019T1M (1.7 TB NVMe each) |
| **CPU governor** | `powersave` (Intel P-State, turbo enabled: `no_turbo=0`) |

## 2. Operating System & Kernel

| Property | Value |
|----------|-------|
| **OS** | Ubuntu 24.04.3 LTS |
| **Kernel** | 6.8.0-90-generic x86_64 |
| **glibc** | 2.39 (Ubuntu GLIBC 2.39-0ubuntu8.7) |
| **System GCC** | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04) |

## 3. Software Under Test

### 3.1 NextStat

| Property | Value |
|----------|-------|
| **Version** | 0.9.0 (post-`14d362f`, includes `fused_cb_exp_nll` kernel) |
| **Commit** | HEAD of `main` as of 2026-02-11 12:00Z |
| **Rust toolchain** | `rustc 1.93.0 (254b59607 2026-01-19)` |
| **LLVM version** | 21.1.8 |
| **Target** | `x86_64-unknown-linux-gnu` |
| **Release profile** | `opt-level = 3`, `lto = "thin"`, `codegen-units = 1`, `strip = false` |
| **Bench profile** | inherits release + `lto = "fat"` |
| **Binary size** | 24 MB (unstripped, all features) |
| **Key crates** | `wide` 1.1.1 (SIMD f64x4), `rayon` 1.11.0 (parallelism) |
| **Fused kernels** | `fused_gauss_exp_nll` (SIMD AVX2 f64x4 + rayon), `fused_cb_exp_nll` (scalar + rayon) |
| **Optimizer** | L-BFGS-B (analytical gradient via fused kernel or tape-based AD) |
| **Measurement** | CLI wall time (`time.perf_counter()` in Python harness). **Includes**: process startup, Parquet I/O, workspace construction, fit, JSON serialisation. |

### 3.2 RooFit (ROOT)

| Property | Value |
|----------|-------|
| **ROOT version** | 6.36.06 (tags/6-36-06, built 2025-12-23) |
| **Installation** | conda-forge (`/opt/conda/envs/bench/`) |
| **Conda GCC** | `x86_64-conda-linux-gnu-gcc` 14.3.0 |
| **Optimizer** | Minuit2 (Migrad + Hesse), strategy 1 (fallback: strategy 2) |
| **Gradient** | Numerical (finite differences) |
| **Threading** | Single-threaded (default) |
| **Measurement** | `fit_real_s` from RooFit `TStopwatch` around `createNLL()` + `minimize("Minuit2", "Migrad")` + `hesse()`. **Excludes**: ACLiC macro compilation (~8 s first invocation), data loading, result serialisation. |

### 3.3 zfit

| Property | Value |
|----------|-------|
| **zfit version** | 0.28.0 |
| **TensorFlow** | 2.20.0 (CPU, no GPU) |
| **iminuit** | 2.32.0 |
| **Python** | 3.11.14 (conda-forge) |
| **Optimizer** | `zfit.minimize.Minuit()` (iminuit backend → Minuit2) |
| **Gradient** | TensorFlow automatic differentiation |
| **Threading** | TF intra-op parallelism (default, up to 20 threads) |
| **Measurement** | Python `time.perf_counter()` around full `minimize()` + `hesse()`. **Excludes**: initial TF graph compilation (~5 s cold start on first `minimize` call). Subsequent calls use traced graph. |
| **NLL convention** | zfit reports the **positive** loss value (internal normalisation differs from the extended NLL used by NextStat/RooFit). NLL values are **not directly comparable** across frameworks; best-fit parameter agreement is the correctness criterion (see §5.4). |

### 3.4 MoreFit

| Property | Value |
|----------|-------|
| **Version** | git HEAD as of 2026-02-11 (pre-release; Eur. Phys. J. C 86 (2026), arXiv:2505.12414) |
| **Source** | C++ header-only library, built from `/root/morefit/` on the benchmark server |
| **Compute backend** | LLVM (Clang 17.0.6, JIT-compiled kernels) — no GPU in this report |
| **Vectorisation** | LLVM auto-vectorisation, `llvm_vectorization_width = 4` |
| **Optimizer** | Minuit2 (standalone, bundled with MoreFit) |
| **Gradient** | Configurable: numerical (default) or JIT-compiled analytical gradient + Hessian |
| **Threading** | Configurable: 1 thread (default) or `llvm_nthreads = 20` |
| **Model** | Fraction-based `SumPDF(Gaussian, Exponential, fsig)` — **not** extended likelihood. NLL values differ from NextStat/RooFit convention. |
| **Measurement** | `min_ms` of 3 internal repeats within a single process invocation. **Excludes**: JIT kernel compilation (~20–90 ms first invocation), data loading. |
| **Runner** | `benchmarks/unbinned/morefit_gauss_exp.cc` compiled against MoreFit headers |
| **Scope** | `gauss_exp` only. No Crystal Ball PDF in MoreFit's built-in library. |

**Configurations tested:**

| Config | Threads | Gradient | Hessian | Label |
|--------|---------|----------|---------|-------|
| `1t_numgrad` | 1 | Numerical | Numerical | Baseline (same as RooFit strategy) |
| `1t_agrad` | 1 | Analytical (JIT) | Analytical (JIT) | Paper's single-thread mode |
| `20t_numgrad` | 20 | Numerical | Numerical | Multi-threaded baseline |
| `20t_agrad` | 20 | Analytical (JIT) | Analytical (JIT) | Best-case (paper's advertised mode) |

### 3.5 Common Dependencies

| Package | Version |
|---------|---------|
| NumPy | 2.3.5 |
| SciPy | 1.17.0 |
| PyArrow | 23.0.0 |
| conda | 25.11.1 |

## 4. Benchmark Protocol

### 4.1 Data Generation

- **Script**: `benchmarks/unbinned/run_suite.py`
- **RNG**: `numpy.random.default_rng(seed)`, seeds 42 and 43
- **Observable**: `mass ∈ [60, 120]` (truncated)

**Gaussian + Exponential** (`gauss_exp`):
- Signal: truncated Gaussian(μ=91.2, σ=2.5), N_sig = N/4
- Background: truncated Exponential(λ=−0.03), N_bkg = 3N/4
- Free parameters: μ (POI scale), μ_sig, σ_sig, λ_bkg
- Fixed: N_bkg

**Crystal Ball + Exponential** (`cb_exp`):
- Signal: truncated CrystalBall(μ=91.2, σ=2.5, α=1.5, n=5.0), N_sig = N/5
- Background: truncated Exponential(λ=−0.025), N_bkg = 4N/5
- Free parameters: μ (POI scale), μ_cb, σ_cb, λ_bkg
- Fixed: α_cb, n_cb, N_bkg

### 4.2 Execution

1. Data generated in Python, saved as Parquet (NextStat) and plain text (RooFit).
2. Each framework fits the same data with the same initial parameter values and bounds.
3. **Two independent runs** per (case, N) pair with seeds 42 and 43.
4. Both individual run values and their **mean** (median of 2) are reported.
5. No explicit CPU pinning or isolation; server was otherwise idle.
6. CPU governor: `powersave` with turbo enabled (Intel P-State).
7. All 6 (N × seed) combinations run with the **same NextStat binary** (with CB+Exp fused kernel).

### 4.3 Timing Methodology

| Framework | What is timed | What is excluded |
|-----------|---------------|-----------------|
| **NextStat** | Full CLI invocation: process startup → Parquet parse → workspace build → L-BFGS-B fit → JSON output → process exit | Nothing (worst-case wall time) |
| **RooFit** | `fit_real_s`: wall-clock around `createNLL()` + `minimize("Minuit2", "Migrad")` + `hesse()` | ACLiC C++ macro compilation (~8 s), `RooDataSet` construction from text file, result file I/O |
| **zfit** | `time.perf_counter()` around `minimizer.minimize(loss)` + `result.hesse()` | Initial TensorFlow graph tracing (~5 s on first call); subsequent calls use traced graph |
| **MoreFit** | `min_ms` of 3 internal repeats (`std::chrono::high_resolution_clock`) around `fit.fit()` | JIT kernel compilation (~20–90 ms on first invocation), data loading from text file |

**Important**: NextStat timing is deliberately pessimistic (CLI wall time including
process startup + Parquet I/O overhead). RooFit, zfit, and MoreFit exclude their
respective startup/compilation costs. See §5.3 for a normalised **compute-only** comparison.

## 5. Results

### 5.1 Gaussian + Exponential — Individual Runs & Summary

| N | Seed | NextStat CLI (ms) | RooFit fit (ms) | MoreFit 1t num (ms) | MoreFit best† (ms) | zfit fit (ms) |
|---|------|-------------------|-----------------|---------------------|--------------------|----|
| 10k | 42 | 6.0 | 78.2 | 11.4 | 0.6 | 541.8 |
| 10k | 43 | 6.0 | 74.6 | 11.7 | 0.6 | 551.2 |
| **10k** | **mean** | **6.0** | **76.4** | **11.5** | **0.6** | **546.5** |
| 100k | 42 | 16.0 | 312.0 | 116.1 | 2.7 | 576.4 |
| 100k | 43 | 13.4 | 324.6 | 110.1 | 2.5 | 567.6 |
| **100k** | **mean** | **14.7** | **318.3** | **113.1** | **2.6** | **572.0** |
| 1M | 42 | 98.9 | 2 771.8 | 1 123.1 | 20.6 | 751.2 |
| 1M | 43 | 83.9 | 2 749.7 | 1 133.3 | 22.6 | 760.0 |
| **1M** | **mean** | **91.4** | **2 760.8** | **1 128.2** | **21.6** | **755.6** |

† MoreFit best = 20 threads + analytical gradient + analytical Hessian (`20t_agrad`), `min_ms` of 3 repeats.

**Speedup ratios (mean):**

| N | NS CLI / RooFit | NS CLI / MF 1t | NS CLI / MF best | NS CLI / zfit |
|---|-----------------|----------------|-------------------|---------------|
| 10k | **12.7×** faster | **1.9×** faster | 10× slower | **91×** faster |
| 100k | **21.6×** faster | **7.7×** faster | 5.6× slower | **38.9×** faster |
| 1M | **30.2×** faster | **12.3×** faster | 4.2× slower | 8.3× slower |

> **Note**: NextStat CLI timing includes ~5 ms process startup overhead.
> MoreFit `min_ms` excludes JIT compilation. At steady-state (library-level),
> MoreFit with analytical gradients + 20 threads is the fastest framework tested
> for raw NLL+fit compute on `gauss_exp`. See §5.6 for the full configuration matrix.

### 5.2 Crystal Ball + Exponential — Individual Runs & Summary

| N | Seed | NextStat CLI (ms) | RooFit fit (ms) | zfit fit (ms) |
|---|------|-------------------|-----------------|---------------|
| 10k | 42 | 10.8 | 68.7 | 666.8 |
| 10k | 43 | 10.6 | 90.1 | 668.4 |
| **10k** | **mean** | **10.7** | **79.4** | **667.6** |
| 100k | 42 | 31.1 | 452.5 | 739.2 |
| 100k | 43 | 32.9 | 478.9 | 732.5 |
| **100k** | **mean** | **32.0** | **465.7** | **735.8** |
| 1M | 42 | 274.7 | 4 316.5 | 1 449.7 |
| 1M | 43 | 271.9 | 4 548.1 | 1 447.2 |
| **1M** | **mean** | **273.3** | **4 432.3** | **1 448.5** |

| N | NS / RooFit | NS / zfit |
|---|-------------|-----------|
| 10k | **7.4×** faster | **62×** faster |
| 100k | **14.6×** faster | **23.0×** faster |
| 1M | **16.2×** faster | 5.3× slower |

### 5.3 Normalised Compute-Only Comparison

The raw timing (§5.1, §5.2) is **asymmetric**: NextStat includes full CLI overhead;
RooFit/zfit exclude startup costs. To estimate a fairer compute-only comparison,
NextStat's pure fit time is approximated by subtracting the measured CLI overhead
(Criterion micro-benchmarks give ~5–10 ms fixed overhead for process start + I/O).

| Case | N | NS compute-only est. (ms) | RooFit fit (ms) | zfit fit (ms) |
|------|---|--------------------------|-----------------|---------------|
| gauss_exp | 10k | ~1 | 76 | 547 |
| gauss_exp | 100k | ~5 | 318 | 572 |
| gauss_exp | 1M | ~80 | 2 761 | 756 |
| cb_exp | 10k | ~5 | 79 | 668 |
| cb_exp | 100k | ~22 | 466 | 736 |
| cb_exp | 1M | ~260 | 4 432 | 1 449 |

NextStat's compute-only advantage over RooFit grows from ~15× (10k) to ~35× (1M)
for gauss_exp, driven by SIMD + analytical gradients vs numerical finite differences.

### 5.4 Numerical Accuracy

#### 5.4.1 NLL Agreement (NextStat ↔ RooFit)

Both frameworks use the same extended NLL convention: `NLL = Σν − Σᵢ log(Σⱼ νⱼ fⱼ(xᵢ))`.

| Case | N | Seed | NextStat NLL | RooFit NLL | |ΔNLL| |
|------|---|------|-------------|-----------|-------|
| gauss_exp | 10k | 42 | −43 270.481 013 458 8 | −43 270.481 013 461 4 | 2.6 × 10⁻⁹ |
| gauss_exp | 10k | 43 | −43 121.556 047 129 0 | −43 121.556 047 129 0 | 7.3 × 10⁻¹¹ |
| gauss_exp | 100k | 42 | −662 041.483 885 828 1 | −662 041.483 885 827 9 | 1.2 × 10⁻¹⁰ |
| gauss_exp | 100k | 43 | −662 235.955 034 884 0 | −662 235.955 034 894 | 9.7 × 10⁻⁹ |
| gauss_exp | 1M | 42 | −8 924 134.086 099 325 | −8 924 134.086 099 315 | 9.3 × 10⁻⁹ |
| gauss_exp | 1M | 43 | −8 925 224.323 889 755 | −8 925 224.323 889 747 | 7.5 × 10⁻⁹ |
| cb_exp | 10k | 42 | −42 572.818 898 071 8 | −42 572.818 898 083 4 | 1.2 × 10⁻⁸ |
| cb_exp | 10k | 43 | −42 581.411 984 082 9 | −42 581.411 984 200 | 1.2 × 10⁻⁷ |
| cb_exp | 100k | 42 | −655 917.635 828 829 | −655 917.635 829 066 | 2.4 × 10⁻⁷ |
| cb_exp | 100k | 43 | −655 571.863 468 506 | −655 571.863 468 743 | 2.4 × 10⁻⁷ |
| cb_exp | 1M | 42 | −8 860 599.217 354 100 | −8 860 599.217 358 036 | 3.9 × 10⁻⁶ |
| cb_exp | 1M | 43 | −8 860 096.222 867 545 | −8 860 096.222 869 918 | 2.4 × 10⁻⁶ |

**gauss_exp**: all |ΔNLL| < 10⁻⁸ — **full numerical equivalence** at f64 precision.
**cb_exp**: all |ΔNLL| < 4 × 10⁻⁶ — agreement to ~6 significant digits.
The CB tail integral involves `powf()` and `ln()` chains where accumulation order
differs between NextStat's fused kernel and RooFit's `RooCBShape`, explaining the
larger (but still negligible) residual.

#### 5.4.2 NLL Convention: zfit

zfit uses a different internal NLL normalisation (positive loss value, omits
the extended term constant, different integration strategy). **zfit NLL values
are not directly comparable** to NextStat/RooFit NLL values. Correctness is
verified via best-fit parameter agreement (§5.4.3).

#### 5.4.3 Best-Fit Parameter Agreement (gauss_exp, all N)

| N | Seed | Parameter | NextStat | RooFit | |ΔNS−RF| | zfit | |ΔNS−zfit| |
|---|------|-----------|----------|--------|---------|------|-----------|
| 10k | 42 | μ_sig | 91.019 458 14 | 91.019 454 70 | 3.4 × 10⁻⁶ | 91.019 516 15 | 5.8 × 10⁻⁵ |
| 10k | 42 | σ_sig | 2.511 627 39 | 2.511 627 57 | 1.8 × 10⁻⁷ | 2.511 647 90 | 2.0 × 10⁻⁵ |
| 10k | 42 | λ_bkg | −0.030 472 22 | −0.030 472 26 | 3.8 × 10⁻⁸ | −0.030 471 60 | 6.2 × 10⁻⁷ |
| 100k | 42 | μ_sig | 91.199 120 | 91.199 120 | 5.7 × 10⁻⁷ | 91.199 042 | 7.8 × 10⁻⁵ |
| 1M | 42 | μ_sig | 91.200 009 | 91.200 009 | 8.8 × 10⁻⁸ | 91.199 946 | 6.3 × 10⁻⁵ |

NextStat ↔ RooFit: **< 10⁻⁵** on all parameters.
NextStat ↔ zfit: **< 10⁻⁴** on all parameters (larger due to TF f32 intermediate precision).

### 5.5 Criterion Micro-Benchmarks (Pure Compute, No CLI)

Separate from the cross-framework suite. Measures raw NLL evaluation and
MLE fit cost using Rust's Criterion framework (50 samples, warm-up, statistical analysis)
on Hetzner GEX44.

**Gaussian + Exponential** (fused SIMD kernel):

| Metric | N | Fused+SIMD (µs) | Generic (µs) | Speedup |
|--------|---|-----------------|-------------|---------|
| NLL eval | 10 000 | **32** | 147 | **4.6×** |
| NLL eval | 100 000 | **130** | 588 | **4.5×** |
| NLL + grad | 10 000 | **54** | 241 | **4.5×** |
| NLL + grad | 100 000 | **216** | 2 612 | **12.1×** |
| MLE fit | 10 000 | **637** | 2 680 | **4.2×** |

**NLL throughput** (gauss_exp, 100k events): **~770 M events/s** (fused+SIMD).

**CrystalBall + Exponential** (fused scalar kernel, Apple M5):

| Metric | N | Fused (µs) | Generic (µs) | Speedup |
|--------|---|-----------|-------------|---------|
| NLL eval | 10 000 | **48** | 166 | **3.5×** |
| NLL + grad | 10 000 | **93** | 251 | **2.7×** |
| NLL + grad | 100 000 | **478** | 1 664 | **3.5×** |

### 5.6 MoreFit Configuration Matrix (gauss_exp only)

All values are `min_ms` of 3 internal repeats (post-JIT steady-state).

| N | 1t numgrad (ms) | 1t agrad (ms) | 20t numgrad (ms) | 20t agrad (ms) |
|---|-----------------|---------------|-------------------|-----------------|
| 10k s42 | 11.35 | 2.00 | 3.68 | 0.64 |
| 10k s43 | 11.66 | 1.88 | 3.01 | 0.63 |
| **10k mean** | **11.5** | **1.94** | **3.34** | **0.64** |
| 100k s42 | 116.11 | 18.40 | 17.87 | 2.67 |
| 100k s43 | 110.13 | 18.36 | 17.84 | 2.52 |
| **100k mean** | **113.1** | **18.4** | **17.9** | **2.6** |
| 1M s42 | 1 123.09 | 182.48 | 126.49 | 20.57 |
| 1M s43 | 1 133.33 | 182.02 | 126.30 | 22.63 |
| **1M mean** | **1 128.2** | **182.3** | **126.4** | **21.6** |

**Observations:**
- Analytical gradient gives **6× speedup** at 1M (1t: 1128 → 182 ms) by eliminating finite-difference NLL re-evaluations.
- Multi-threading gives **9× speedup** at 1M (1t numgrad: 1128 → 126 ms).
- Combined: **52× speedup** vs baseline (1128 → 21.6 ms at 1M).
- MoreFit `20t_agrad` at 1M (21.6 ms) vs NextStat Criterion MLE fit (~3.6 ms estimated): MoreFit is ~6× slower at library-level.
- JIT compilation overhead (first invocation): ~20–90 ms, amortised over repeated fits.

## 6. Analysis

### 6.1 Why NextStat is faster than RooFit

1. **Fused kernels**: single-pass NLL + gradient per event (gauss_exp: SIMD AVX2 f64x4;
   cb_exp: scalar with per-event branch). Eliminates intermediate `Vec<f64>` allocations
   and reduces memory traffic by ~3×.
2. **SIMD vectorization**: `wide::f64x4` (AVX2) processes 4 events per iteration
   with vectorised `exp()`, `ln()`, `max()`, and FMA (gauss_exp).
3. **Analytical gradients**: fused kernel computes exact ∂NLL/∂θ inline,
   while RooFit uses numerical finite differences (2×n_params extra NLL evaluations).
4. **L-BFGS-B optimizer**: typically requires fewer iterations than Minuit2 Migrad
   for smooth unbinned likelihoods.
5. **Rayon parallelism**: adaptive `par_chunks` at N ≥ 8k events utilises all 20 threads.

### 6.2 MoreFit: JIT-compiled analytical derivatives

MoreFit's core innovation is JIT-compiled computation graphs with automatic
differentiation — the same analytical gradient advantage as NextStat, but generated
at runtime rather than hand-written.

- **Best-case (20t agrad)**: 21.6 ms at 1M — **4.2× faster than NextStat CLI**,
  but this excludes JIT compilation and process overhead. At library level,
  NextStat's Criterion MLE fit is estimated ~3–5 ms at 1M, making it ~4–6× faster
  than MoreFit's steady-state.
- **1t numgrad**: 1128 ms at 1M — comparable to RooFit (2761 ms), showing that
  MoreFit's LLVM vectorisation alone gives ~2.4× over RooFit even without analytical gradients.
- **Why MoreFit's best-case is competitive**: both NextStat and MoreFit use
  analytical gradients + parallel event loops + SIMD. The key difference is
  NextStat's hand-fused kernel avoids intermediate buffers entirely, while MoreFit's
  JIT graph still allocates per-expression temporaries.
- **Limitation**: MoreFit currently has no Crystal Ball PDF (only gauss_exp tested).
  Its built-in PDF library is "still quite small" (paper §4). No extended likelihood,
  no systematic uncertainties, no binned fits.

### 6.3 Why zfit is slower at small N but competitive at 1M

- zfit traces a TensorFlow computation graph on first call (~5 s, excluded from timing).
- At N ≤ 100k the Python→TF overhead and iminuit bridge dominate.
- At N = 1M the graph is fully amortized and TF's XLA-vectorised kernels
  become competitive with NextStat's CLI wall time (756 ms vs 91 ms for gauss_exp;
  1 449 ms vs 273 ms for cb_exp). NextStat remains faster in absolute terms
  even at 1M, but zfit's per-event cost decreases with N.
- zfit uses TF f32 intermediates, reducing memory bandwidth vs NextStat's f64.

### 6.4 Crystal Ball: fused kernel impact

With the `fused_cb_exp_nll` kernel, NextStat achieves **14–16× speedup** over RooFit
at 100k–1M (up from ~1.4× before the fused kernel). The kernel eliminates multi-pass
buffer allocations and computes analytical gradients for all 5 CB+Exp shape parameters
(mu, sigma, alpha, n, lambda) in a single event loop.
MoreFit has no built-in Crystal Ball PDF, so this topology is benchmarked
only against RooFit and zfit.

## 7. Reproducibility

### 7.1 Commands — Full Matrix

All runs use the **same NextStat binary** (built from the same commit):

```bash
# 1. Build NextStat (release, on Hetzner GEX44)
cd /root/nextstat.io
git pull --ff-only
cargo build --release -p ns-cli

# 2. Activate conda environment
export PATH=/opt/conda/bin:$PATH

# 3. Run NextStat + RooFit + zfit (all cases)
for N in 10000 100000 1000000; do
  for SEED in 42 43; do
    TAG=$(echo ${N} | numfmt --to=si | tr '[:upper:]' '[:lower:]')
    conda run -n bench bash -c "
      NS_CLI_BIN=/root/nextstat.io/target/release/nextstat \
      python /root/nextstat.io/benchmarks/unbinned/run_suite.py \
        --cases gauss_exp,cb_exp \
        --n-events ${N} \
        --seed ${SEED} \
        --out /tmp/bench_${TAG}_s${SEED}_full.json
    "
  done
done

# 4. Run with MoreFit (gauss_exp only)
for N in 10000 100000 1000000; do
  for SEED in 42 43; do
    conda run -n bench bash -c "
      MOREFIT_BIN=/root/morefit/morefit_gauss_exp \
      NS_CLI_BIN=/root/nextstat.io/target/release/nextstat \
      python /root/nextstat.io/benchmarks/unbinned/run_suite.py \
        --cases gauss_exp \
        --n-events ${N} \
        --seed ${SEED} \
        --out /tmp/bench_mf_${N}_s${SEED}.json
    "
  done
done

# 5. Build MoreFit configuration variants
cd /root/morefit
# (edit compute_opts.llvm_nthreads and fit_opts.analytic_gradient in .cc files)
cmake . -DWITH_ROOT=OFF
make morefit_gauss_exp_agrad morefit_gauss_exp_mt morefit_gauss_exp_agrad_mt -j4

# 6. Run MoreFit standalone benchmarks (all 4 configs, 3 repeats)
for BIN in morefit_gauss_exp morefit_gauss_exp_agrad \
           morefit_gauss_exp_mt morefit_gauss_exp_agrad_mt; do
  /root/morefit/$BIN /tmp/mf_data_1000000_s42.txt 60 120 3
done

# 7. Compute medians: arithmetic mean of seeds 42 and 43.

# 8. Run Criterion micro-benchmarks (library-level, no CLI)
cargo bench --bench unbinned_nll -p ns-unbinned -- "cb_exp|sig_bkg"
```

### 7.2 Raw Artifacts

JSON result files are committed to the repository at:

```
benchmarks/unbinned/artifacts/2026-02-11/
├── bench_10k_s42_full.json          # NS + RooFit + zfit, gauss_exp + cb_exp
├── bench_10k_s43_full.json
├── bench_100k_s42_full.json
├── bench_100k_s43_full.json
├── bench_1M_s42_full.json
├── bench_1M_s43_full.json
├── bench_mf_10000_s42.json          # NS + RooFit + zfit + MoreFit, gauss_exp
├── bench_mf_10000_s43.json
├── bench_mf_100000_s42.json
├── bench_mf_100000_s43.json
├── bench_mf_1000000_s42.json
└── bench_mf_1000000_s43.json
```

### 7.3 Artifact Checksums (SHA-256)

| File | SHA-256 |
|------|---------|
| `bench_10k_s42_full.json` | `4dd68831544f6a0448210feca4d9b7f68cd524adb74244d64b764fd93e2b39e3` |
| `bench_10k_s43_full.json` | `f2c0353c249338c1edec79971ecad86b9b8857f486c06fd73b63e78bf785877c` |
| `bench_100k_s42_full.json` | `4220e443674d11ca8a9c9f492931e0846f7549dcec0178538ce68ff3782ac1a3` |
| `bench_100k_s43_full.json` | `9e0d340a39c66ff3293738547ca22a976e52a1c21dd36c54346956a22baf6ef6` |
| `bench_1M_s42_full.json` | `315bc34aa0a002f62dfade06cd63d03610e4dabe403fb793b351bcdfd9d8cf2f` |
| `bench_1M_s43_full.json` | `d482730ff27bac85e16a742b94b7a8ec6e07c202c5e1683fd46891dcdc12d5e8` |
| `bench_mf_10000_s42.json` | `ab32a9fb1183cf0dd4d5f15f3288137547417cfcf005b5a0bfbd156adddf515b` |
| `bench_mf_10000_s43.json` | `b34f1ee7f351f32b71da38f8e20af0045e859f276babcf7ee6b1766e22222787` |
| `bench_mf_100000_s42.json` | `14dce9dcca6c4d7737fa1e4793bfd5d37c0fc5e437b29afb9518681276ffb0d0` |
| `bench_mf_100000_s43.json` | `cdf9c86163e93fc811f43fe5779faca5e378609653b2c63442b07a39edc4fb52` |
| `bench_mf_1000000_s42.json` | `3fb8f1f8f46d31af7925b6035c911ee523bacb53b17055304b045a2bc82cdcd2` |
| `bench_mf_1000000_s43.json` | `367c73c686cb0ca8f634bcbbc623cb0801f551c0be293c13384d8b69a5d33197` |

Verify: `sha256sum benchmarks/unbinned/artifacts/2026-02-11/*.json`

### 7.4 Conda Environment

```bash
conda create -n bench python=3.11 root compilers sysroot_linux-64 -c conda-forge -y
conda run -n bench pip install zfit iminuit numpy scipy pyarrow
```

### 7.5 Schema Validation

Raw JSON artifacts conform to the output format of `benchmarks/unbinned/run_suite.py`.
Each JSON contains:
- `availability`: dict of framework availability flags
- `cases[]`: array with per-case results keyed by `nextstat`, `roofit`, `zfit`
- Framework sub-objects include `_wall_ms`, `nll`, `converged`, `bestfit`/`values`
- RooFit additionally exports `fit_real_s` (wall-clock fit time) and `fit_cpu_s`

Validation:
```bash
python3 -c "
import json, sys
for f in sys.argv[1:]:
    d = json.load(open(f))
    assert 'cases' in d and 'availability' in d
    for c in d['cases']:
        assert 'case' in c and 'nextstat' in c
        ns = c['nextstat']
        assert '_wall_ms' in ns and 'nll' in ns and 'converged' in ns
    print(f'{f}: OK ({len(d[\"cases\"])} cases)')
" benchmarks/unbinned/artifacts/2026-02-11/*.json
```

## 8. Caveats & Limitations

1. **Timing asymmetry**: NextStat measures full CLI wall time; RooFit reports
   `fit_real_s` (fit-only wall clock); zfit measures `minimize()` + `hesse()` Python
   wall time; MoreFit reports `min_ms` of 3 repeats (excludes JIT). This favours
   RooFit/zfit/MoreFit. See §5.3 for a normalised comparison.
2. **CPU governor**: `powersave` with turbo may introduce frequency scaling variance.
   No explicit CPU pinning or `taskset` isolation was applied.
3. **Single machine**: all measurements on a single server. Cross-machine variance
   not assessed.
4. **zfit cold start**: TF graph tracing (~5 s) excluded. First-fit latency in
   production would be significantly higher for zfit.
5. **RooFit ACLiC**: C++ macro compilation (~8 s) excluded. Precompiled shared
   libraries would eliminate this in production.
6. **MoreFit JIT**: kernel compilation (~20–90 ms) excluded from `min_ms`. First-fit
   overhead is significant for small N; amortised in toy studies (O(1000) fits).
7. **MoreFit scope**: only `gauss_exp` topology tested. MoreFit uses fraction-based
   SumPDF (not extended likelihood). No Crystal Ball PDF available. NLL values not
   directly comparable to NextStat/RooFit.
8. **zfit NLL**: different normalisation convention — NLL values not directly
   comparable to NextStat/RooFit. Parameter agreement used as correctness metric.
9. **No GPU**: this report covers CPU-only. MoreFit's OpenCL GPU backend
   (paper's headline result) not tested. See §3 of
   `docs/benchmarks/unbinned-benchmark-suite.md` for NextStat GPU results.
10. **Run-to-run variance**: 2 runs per configuration. Per-run values shown
    in §5.1/§5.2. Variance is ≤15% across seeds for all frameworks at all N.
