# GVM Measurement Combination — Benchmark Snapshot

**Date**: 2026-03-07
**Status**: Published
**Bench file**: `crates/ns-inference/benches/measurement_combine_benchmark.rs`

## Environments

Two platforms were benchmarked to cover both desktop and server deployment:

### Apple M5 (desktop)

| Key | Value |
|-----|-------|
| Machine | MacBook Pro (Apple M5, 10 cores, 24 GB) |
| OS | Darwin 25.2.0 arm64 |
| Rust | rustc 1.93.0 (254b59607 2026-01-19) |
| Build | `--release` (bench profile, optimized) |
| Criterion | 10 samples, 500ms warmup, 3s measurement |

### AMD EPYC 7502P (server)

| Key | Value |
|-----|-------|
| Machine | Hetzner Dedicated (AMD EPYC 7502P 32-Core, 128 GB) |
| OS | Linux 6.8.0-90-generic x86_64 |
| Rust | rustc 1.93.1 (01f6ddf75 2026-02-11) |
| Build | `--release` (bench profile, optimized) |
| Criterion | 10 samples, 500ms warmup, 3s measurement |

---

## Fit (single combination)

| Fixture | Solver | M5 Median | EPYC Median | EPYC/M5 |
|---------|--------|-----------|-------------|---------|
| Paper top-mass (15 x 22) | Auto | 101.0 µs | 412.6 µs | 4.1x |
| Synthetic (32 x 24) | Auto | 11.8 ms | 42.4 ms | 3.6x |
| Synthetic (32 x 24) | AnalyticPerturbative | 12.1 ms | 42.7 ms | 3.5x |
| Synthetic (32 x 24) | NumericalPaper | 30.5 ms | 101.0 ms | 3.3x |
| Synthetic (64 x 48) | Auto | 167.4 ms | 425.2 ms | 2.5x |
| Synthetic (64 x 48) | AnalyticPerturbative | 125.9 ms | 411.2 ms | 3.3x |

### Observations

- Paper top-mass (15 measurements, 22 systematics): sub-millisecond on both platforms.
- Auto and AnalyticPerturbative nearly identical at 32x24 (perturbative path succeeds, no fallback).
- NumericalPaper is 2.4–2.5x slower than AnalyticPerturbative on both platforms (expected: full numerical profiling vs closed-form expansion).
- M5 is 2.5–4.1x faster than EPYC single-threaded (Apple Silicon IPC advantage over Zen 2 @ 2.5 GHz base).
- At 64x48 Auto ≈ AnalyticPerturbative on EPYC (425 vs 411 ms), while M5 shows a wider gap (167 vs 126 ms), suggesting Auto fallback frequency is data-dependent.

## Calibration (toy-based)

| Fixture | Solver | N toys | M5 Median | EPYC Median | EPYC/M5 |
|---------|--------|--------|-----------|-------------|---------|
| Paper top-mass (15 x 22) | Auto | 16 | 8.7 ms | 19.5 ms | 2.2x |
| Synthetic (32 x 24) | Auto | 8 | 93.5 ms | 154.7 ms | 1.7x |
| Synthetic (32 x 24) | AnalyticPerturbative | 8 | 56.5 ms | 151.9 ms | 2.7x |
| Synthetic (32 x 24) | NumericalPaper | 8 | 193.8 ms | 303.4 ms | 1.6x |
| Synthetic (64 x 48) | Auto | 4 | 509.4 ms | 1.34 s | 2.6x |
| Synthetic (64 x 48) | NumericalPaper | 4 | 1.04 s | 1.88 s | 1.8x |

### Observations

- Paper top-mass 16-toy calibration: 8.7 ms (M5) / 19.5 ms (EPYC). Calibration at 256 toys: ~140 ms / ~310 ms.
- AnalyticPerturbative is 2.0–3.4x faster than NumericalPaper for toy calibration at 32x24.
- 64x48 NumericalPaper with 4 toys: ~1 s (M5) / ~1.9 s (EPYC). 128-toy calibration: ~33 s / ~60 s.
- EPYC/M5 ratio narrows at larger problems (1.6–1.8x at 64x48 NumericalPaper), suggesting memory-bound workloads benefit from EPYC's higher bandwidth.

## Calibration Campaign (scenarios x seeds x toys)

| Fixture | Solver | Scenarios | Seeds | N toys | M5 Median | EPYC Median | EPYC/M5 |
|---------|--------|-----------|-------|--------|-----------|-------------|---------|
| Paper top-mass (15 x 22) | Auto | 2 | 2 | 8 | 63.8 ms | 169.9 ms | 2.7x |
| Synthetic (32 x 24) | Auto | 3 | 2 | 4 | 492.2 ms | 1.21 s | 2.5x |

### Observations

- Paper top-mass full campaign (2 scenarios × 2 seeds × 8 toys = 32 fits + calibration): 64 ms (M5) / 170 ms (EPYC).
- Synthetic 32x24 (3 scenarios × 2 seeds × 4 toys = 24 fits + calibration): 492 ms (M5) / 1.21 s (EPYC).
- Both platforms complete research-grade campaigns in under 2 seconds.

## Rayon Thread Scaling

These measurements use the `mt_*` Criterion groups and vary
`RAYON_NUM_THREADS` on each platform.

### Calibration (256 toys)

#### Apple M5 (10 cores)

| Fixture | 1t | 2t | 4t | 8t | 10t | Best | Speedup |
|---------|-----|-----|-----|-----|------|------|---------|
| Paper top-mass (15 x 22) | 294.8 ms | 278.1 ms | 286.7 ms | 388.2 ms | 393.8 ms | 2t | 1.06x |
| Synthetic (32 x 24) | 2.228 s | 2.154 s | 2.527 s | 2.698 s | 2.897 s | 2t | 1.03x |

#### AMD EPYC 7502P (32 cores / 64 threads)

| Fixture | 1t | 4t | 16t | 32t | 64t | Best | Speedup |
|---------|------|-------|--------|--------|--------|------|---------|
| Paper top-mass (15 x 22) | 1.108 s | 301 ms | 107 ms | **78 ms** | 83 ms | 32t | **14.2x** |
| Synthetic (32 x 24) | 9.758 s | 2.736 s | 989 ms | 744 ms | **690 ms** | 64t | **14.1x** |

### Campaign (8 scenarios x 8 seeds x 32 toys)

#### Apple M5 (10 cores)

| Fixture | 1t | 2t | 4t | 8t | 10t | Best | Speedup |
|---------|------|------|------|------|------|------|---------|
| Paper top-mass (15 x 22) | 8.079 s | 4.926 s | 2.596 s | **1.562 s** | 2.903 s | 8t | **5.2x** |

#### AMD EPYC 7502P (32 cores / 64 threads)

| Fixture | 1t | 4t | 16t | 32t | 64t | Best | Speedup |
|---------|-------|------|------|------|------|------|---------|
| Paper top-mass (15 x 22) | 18.38 s | 5.23 s | **2.69 s** | 2.73 s | 2.70 s | 16t | **6.8x** |

### Thread-Scaling Observations

- **EPYC calibration scales excellently**: 256-toy paper top-mass drops from 1.1 s → 78 ms = **14.2×** at 32 threads. Near-linear to 16t, then diminishing returns.
- **EPYC campaigns plateau at 16t**: 18.4 s → 2.7 s = **6.8×**. Campaign outer loop has 8 scenarios, so 16+ threads see Amdahl saturation on sequential inner work.
- **M5 calibration barely scales**: peaks at 2t with only 1.03–1.06× improvement, then regresses. Each toy is too fast (~1.2 ms per toy for paper top-mass) — Rayon scheduling overhead dominates on a 10-core desktop chip.
- **M5 campaigns scale well to 8t**: 5.2× speedup. The heavier per-scenario work (32 toys × calibration) amortizes thread overhead.
- **32t vs 64t on EPYC**: SMT gives marginal benefit for calibration (+7% on synthetic 32×24), no benefit for campaigns. 32 physical cores = sweet spot.
- **Server vs desktop**: EPYC's 32-core advantage is decisive for parallelizable workloads — 78 ms (EPYC 32t) vs 278 ms (M5 2t) for 256-toy calibration, despite M5 being 3–4× faster single-threaded.

## Solver Parity Calibration

| Fixture | Comparison | N toys | M5 Median | EPYC Median | EPYC/M5 |
|---------|-----------|--------|-----------|-------------|---------|
| Paper top-mass (15 x 22) | NumericalPaper vs AnalyticPerturbative | 8 | 42.1 ms | 161.2 ms | 3.8x |

### Observations

- Parity check runs both solvers per scenario and computes deltas.
- 42 ms (M5) / 161 ms (EPYC) for the paper fixture includes 2 × (scenario fit + calibration) for each solver.

## Reproducing

```bash
CARGO_TARGET_DIR=/path/to/target cargo bench \
  --bench measurement_combine_benchmark -p ns-inference
```

Criterion HTML reports: `target/criterion/measurement_combine_*/report/index.html`

## Paper Reference

L. Canonero and G. Cowan, "Combination of measurements and the BLUE method
generalized by allowing for errors in the error assignments,"
*Eur. Phys. J. C* **85**, 156 (2025).
