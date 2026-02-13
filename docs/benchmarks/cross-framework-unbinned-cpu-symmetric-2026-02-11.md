---
title: "Cross-Framework Unbinned CPU Benchmark (Symmetric Timing)"
status: published
last_updated: 2026-02-11
---

# Cross-Framework Unbinned CPU Benchmark (Symmetric Timing)

This report is the apples-to-apples CPU rerun for `gauss_exp` on:

- host: Hetzner GEX44 (`i5-13500`, 20 threads)
- event counts: `10k`, `100k`, `1M`
- seeds: `42`, `43` (median of seeds reported)

Artifacts:
- Baseline (default optimizer): `benchmarks/unbinned/artifacts/2026-02-11/bench_symm_cpu_gex44_diag2_20260211T225229Z.json`
- Tuned (`--opt-m 20 --opt-smooth-bounds`): `benchmarks/unbinned/artifacts/2026-02-11/bench_symm_cpu_gex44_tune2_m20smooth_20260211T225807Z.json`

## 1) Protocol

Runner:
- `benchmarks/unbinned/bench_cpu_symmetry.py`

Measured separately:

1. **External wall time**
   - NextStat: `nextstat unbinned-fit` process wall
   - MoreFit: binary process wall, `repeats=1` (includes startup/JIT)
2. **Warm in-process fit time**
   - NextStat: Python bindings (`nextstat.fit()`), warmup=2, repeats=9
   - MoreFit: `repeats=9`, first sample treated as cold, median over warm samples

MoreFit variants:
- `mf_1t_num`, `mf_1t_agrad`, `mf_20t_num`, `mf_20t_agrad`

## 2) External wall (ms, median-of-seeds, tuned)

| N | NextStat CLI | MF 1t num | MF 1t agrad | MF 20t num | MF 20t agrad |
|---|-------------:|----------:|------------:|-----------:|-------------:|
| 10k | 4.406 | 45.039 | 98.272 | 35.654 | 101.151 |
| 100k | 9.264 | 175.210 | 128.622 | 62.838 | 116.918 |
| 1M | 46.929 | 1476.186 | 441.613 | 320.809 | 272.778 |

Result: with startup/JIT included, NextStat CLI is faster than all tested MoreFit CPU variants at all N.

## 3) Warm fit-only (ms, median-of-seeds, tuned)

| N | NextStat library | MF 1t num | MF 1t agrad | MF 20t num | MF 20t agrad |
|---|-----------------:|----------:|------------:|-----------:|-------------:|
| 10k | 0.835 | 13.046 | 1.935 | 2.905 | 0.602 |
| 100k | 3.592 | 127.399 | 19.883 | 18.094 | 2.719 |
| 1M | 31.827 | 1289.420 | 199.855 | 164.474 | 24.963 |

Result: in steady-state compute, `mf_20t_agrad` remains faster, but the tuned NextStat gap shrinks to ~`1.27–1.39×` (from ~`3×` baseline).

## 4) Tuning impact (default → tuned)

| N | NS CLI (ms) | Speedup | NS lib full (ms) | Speedup | NS lib minimum (ms) | Speedup |
|---|------------:|--------:|-----------------:|--------:|--------------------:|--------:|
| 10k | 5.970 → 4.406 | 1.36× | 2.180 → 0.835 | 2.61× | 1.934 → 0.567 | 3.41× |
| 100k | 14.543 → 9.264 | 1.57× | 8.800 → 3.592 | 2.45× | 8.408 → 2.698 | 3.12× |
| 1M | 93.268 → 46.929 | 1.99× | 79.168 → 31.827 | 2.49× | 72.315 → 24.965 | 2.90× |

The dominant effect is optimizer convergence speed, not likelihood value changes.
For all `(N, seed)` pairs, tuned and baseline `nll` agree within numerical noise (`|ΔNLL| ≤ 1.1e-8`).

## 5) Optimizer diagnostics (NextStat, default vs tuned)

| N | NS n_iter (default → tuned) | NS n_fev (default → tuned) | NS n_gev (default → tuned) |
|---|----------------------------:|---------------------------:|---------------------------:|
| 10k | 20 → 7 | 41 → 10 | 62 → 17 |
| 100k | 28 → 8 | 49 → 14 | 78 → 22 |
| 1M | 28 → 7 | 56 → 16 | 85 → 24 |

Interpretation:
- For `gauss_exp` with box constraints, `--opt-smooth-bounds` is high impact.
- `m=20` and smooth bounds together reduce line-search/evaluation churn by ~3–5×.
- Remaining gap to MoreFit warm-fit is now mostly kernel-level compute (~25 ms vs ~32 ms at 1M), no longer a convergence-count issue.

## 6) Practical recommendation

For production unbinned fits with constrained parameters:

- use `--opt-m 20 --opt-smooth-bounds` for `nextstat unbinned-fit`
- keep default tolerances unless you need stricter convergence criteria
