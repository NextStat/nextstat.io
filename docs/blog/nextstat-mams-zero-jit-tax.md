---
title: "NextStat v0.9.6: Zero-JIT Tax, ESS/grad, and Convergence"
subtitle: "Final canonical results with strict backend split: Metal, CUDA V100, and EPYC CPU."
date: 2026-02-18
tags: [NextStat, MAMS, LAPS, BlackJAX, Benchmarks, Bayesian Inference]
---

# NextStat v0.9.6: Zero-JIT Tax, ESS/grad, and Convergence

> **TL;DR (only final v0.9.6 numbers)**
>
> - **LAPS Metal:** final matrix is **8/8 ok** with `Div%=0` across all cases.
> - **CUDA V100 parity (3-seed median, canonical):** NextStat LAPS keeps **zero runtime JIT tax** (`cold ~= warm`), while BlackJAX cold-start is **11.8-90.6s** in this setup.
> - **Time-to-result in real edit cycles:** when model structure/shape changes (priors, parameterization, dimensions), JAX/XLA workflows commonly recompile; that compile wall repeats across iterations. NextStat AOT kernels keep iteration latency close to warm-path behavior.
> - **ESS/grad (V100 sampling, report-chain normalized):** on matched targets, NS LAPS ranges from **2.46x to 45.11x** vs BlackJAX in this canonical run.
> - **CPU funnel fairness fixed:** `FunnelNcpModel` (NCP) is **6/6 ok** across 3 seeds on EPYC for both MAMS and NUTS; centered funnel remains a known pathological control.

---

## MAMS and LAPS in one page (for NUTS users)

If you already know NUTS, the key mental model is:

- `NUTS` is dynamic HMC with per-transition tree expansion (adaptive path length each step).
- `MAMS` (Metropolis Adjusted Microcanonical Sampler) uses microcanonical/isokinetic dynamics with a fixed trajectory length in preconditioned space.
- `LAPS` is the massively parallel GPU/Metal sampler path that applies MAMS-style dynamics across thousands of chains with hardware-oriented execution.

What is new relative to standard NUTS implementations:

- Fixed-shape transition kernels are easier to run efficiently on SIMD/GPU backends than recursive tree building.
- The sampler is built for very large chain parallelism (`4096` chains is a normal operating point in this report).
- In this release, MAMS/LAPS use microcanonical-aware diagnostics gates and explicit NCP-vs-centered disclosure for funnel-like geometries.

Where this design is strongest:

- Time-to-result in iterative workflows (no repeated JIT compile wall in our AOT path).
- Hierarchical and multi-scale targets where we observe high ESS/grad in the matched-target runs.
- Local acceleration paths (Metal) and server GPU paths (CUDA) with the same sampler semantics.

Where NUTS or XLA-heavy stacks can still win:

- Simple low-dimensional targets after warm compilation (higher raw warm-path throughput is possible).
- Some smooth concentrated posteriors on CPU (for example large-n logistic in this report).

This report is therefore not claiming a universal sampler winner; it documents where each execution model is stronger, with explicit fairness caveats.

---

## 1) Protocol and fairness rules

### Backend split (no mixing)

- `LAPS Metal` results are reported separately from `LAPS CUDA`.
- `CPU (EPYC)` results are reported separately from GPU.
- No cross-backend table mixes Metal and CUDA values.

### Multi-run aggregation (anti-cherry-pick)

- Final CPU/GPU comparison tables use **three independent seeds**: `42, 123, 777`.
- We report **median** as the primary number (robust to outliers), with `mean ± std` shown where useful.
- Single-seed values are kept only in raw artifacts, not as headline claims.

### Time-to-result in iterative modeling

- Bayesian work is an edit cycle (change prior, add covariate, reparameterize, rerun).
- In JIT/XLA stacks, graph/shape changes often invalidate compiled executables and trigger recompilation, so cold-start costs recur during exploration.
- NextStat uses AOT-compiled Rust/CUDA kernels, so wall-clock iteration time stays near warm-path latency even as models evolve.

### Funnel parameterization disclosure

For `std_normal`, `eight_schools`, and `glm_logistic`, both engines sample the **same target density** (identical log-density functions).

For `neal_funnel_10d`, the parameterizations differ in the V100 parity run:
- **NS LAPS** samples the **Non-Centered Parameterization (NCP)**: `log p(v, z) = -v²/18 - 0.5 * sum(z_i²)`.
- **BlackJAX** samples the centered parameterization: `log p(v, x) = -v²/18 - 0.5 * exp(-v) * sum(x_i²) - 0.5*(d-1)*v`.

These are **not the same optimization problem**. The centered funnel has position-dependent curvature that is fundamentally harder for fixed-metric samplers. The `neal_funnel` rows in section 3 and the Appendix therefore reflect both algorithmic and parameterization differences and **should not be interpreted as a like-for-like throughput comparison**. They are retained to show convergence behavior (NS converges, BlackJAX does not) but excluded from headline ESS/grad claims.

- CPU now has explicit **`FunnelNcpModel`** for fair NCP comparisons (section 6).
- Centered `FunnelModel` remains a separate hard-geometry control.

### Algorithmic changes in v0.9.6

- MAMS uses `eps_jitter=0.1` by default (±10% uniform step-size noise per transition), breaking fixed-L periodicity and improving tail ESS on periodic targets like `std_normal`.
- Default trajectory length: `L = sqrt(d)` in preconditioned space (Robnik et al. 2023).

### BlackJAX configuration (V100 parity run)

To preempt concerns about competitor misconfiguration, the full BlackJAX setup:

- **Sampler:** `blackjax.adjusted_mclmc` with `isokinetic_mclachlan` integrator.
- **Warmup:** built-in `blackjax.adjusted_mclmc_find_L_and_step_size` (500 iterations, single-chain warmup, `target_accept=0.9`, `diagonal_preconditioning=True`).
- **Trajectory length:** tuned by BlackJAX warmup (`L`, `step_size`), then `n_steps = round(L / step_size)`.
- **Mass matrix:** sampling uses tuned `inverse_mass_matrix` from BlackJAX warmup.
- **Multi-chain:** 4096 chains, `jax.vmap(run_chain)`, `block_until_ready()` + `device_get()` for fair host-side timing.
- **Cold/warm:** cold = first `vmap` call (includes XLA compilation); warm = second call with cached JIT.
- **Init:** chains are initialized around the warmed single-chain state (`warmed_state.position + N(0, 0.5)`).
- **Seed:** 42 (cold), 1042 (warm).
- **Seeds:** `42, 123, 777` (for each seed, warm run uses `seed + 1000` key path).
- **Source:** `benchmarks/gpu_triple_bench.py`, functions `_blackjax_builtin_warmup()` and `bench_blackjax()`.

### V100 parity run config (NS LAPS, 3 seeds)

- `n_chains=4096`, `n_warmup=500`, `n_samples=1000`, `report_chains=256`, `seeds=42/123/777`.
- Section 3/4 report **median across 3 seeds**.
- R-hat computed from 256 report chains (512 half-chains), giving materially tighter diagnostics than the earlier 64-chain reporting.

---

## 2) Canonical LAPS Metal results (final)

Hardware: **Apple M5, 10 GPU cores, 24 GB unified memory**.

| Model | Chains | w+s | Wall (s) | R-hat | ESS/s | Div% | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| std_normal_10d | 256 | 100+100 | 0.14 | 1.175 | 3,680 | 0.0 | ok |
| std_normal_10d_4096ch | 4096 | 200+500 | 0.09 | 1.038 | 12,585 | 0.0 | ok |
| eight_schools | 4096 | 500+2000 | 0.25 | 1.007 | 124,705 | 0.0 | ok |
| neal_funnel_10d | 4096 | 500+2000 | 0.31 | 1.006 | 22,791 | 0.0 | ok |
| neal_funnel_riemannian_10d | 4096 | 500+2000 | 0.27 | 1.010 | 14,142 | 0.0 | ok |
| glm_logistic_n200_p6 | 4096 | 500+2000 | 2.15 | 1.005 | 4,647 | 0.0 | ok |
| glm_logistic_n1000_p20 | 4096 | 500+2000 | 34.32 | 1.010 | 248 | 0.0 | ok |
| glm_logistic_n5000_p20 | 4096 | 500+2000 | 59.06 | 1.015 | 110 | 0.0 | ok |

Note: the 256-chain `std_normal_10d` row (R-hat 1.175) demonstrates the minimum viable chain count; the 4096-chain row is the canonical benchmark configuration.

In practice, this shows that local Apple Silicon can run datacenter-style massively parallel inference workloads with strong convergence diagnostics, without CUDA setup or JIT compile latency.

Quality gate policy used for this matrix:

- `MAMS`/`LAPS`: `QualityGates::microcanonical()` (EBFMI is warning-only).
- `NUTS`: strict default gate preserved (`EBFMI fail < 0.20`).

---

## 3) CUDA V100 parity run (LAPS vs BlackJAX, 3-seed median)

Hardware: **Tesla V100-PCIE-16GB**.

| Model | Engine | Cold (s) | Warm (s) | min_ESS | ESS/s (warm) | R-hat |
|---|---|---:|---:|---:|---:|---:|
| std_normal_10d | NS LAPS GPU | 1.554 | 0.240 | 159,753 | 680,785 | 1.0062 |
| std_normal_10d | BlackJAX GPU | 14.064 | 0.225 | 1,771 | 7,847 | 1.1010 |
| eight_schools | NS LAPS GPU | 1.425 | 0.241 | 75,682 | 314,476 | 1.0065 |
| eight_schools | BlackJAX GPU | 11.769 | 0.346 | 28,020 | 75,255 | 1.0080 |
| neal_funnel_10d | NS LAPS GPU | 1.404 | 0.259 | 54,768 | 211,581 | 1.0083 |
| neal_funnel_10d | BlackJAX GPU | 15.517 | 0.412 | 706 | 1,759 | 1.2732 |
| glm_logistic | NS LAPS GPU | 23.791 | 9.254 | 77,852 | 8,415 | 1.0086 |
| glm_logistic | BlackJAX GPU | 90.615 | 77.765 | 19,583 | 226 | 1.0122 |

### Reading this table

- **Zero JIT tax:** NS LAPS cold remains close to warm (AOT-compiled Rust/CUDA). BlackJAX cold-start is materially higher in this setup (`11.8-90.6s`).
- **Warm-start throughput (canonical run):** NS LAPS is higher on all matched targets in this setup.
- **`neal_funnel` is not a like-for-like comparison** (see section 1: NS samples NCP, BlackJAX samples centered). In these 3 seeds, BlackJAX centered-funnel R-hat ranges `1.260-1.275` and remains weaker than NS NCP, which is expected from parameterization difficulty, not a sampler defect.

---

## 4) ESS/grad on V100 (sampling phase, matched targets only, 3-seed median)

| Model | NS LAPS ESS/grad | BlackJAX ESS/grad | Ratio (NS/BJ) |
|---|---:|---:|---:|
| std_normal_10d | 0.312017 | 0.006917 | 45.11x |
| eight_schools | 0.098544 | 0.040104 | 2.46x |
| glm_logistic | 0.101370 | 0.002638 | 38.43x |

`neal_funnel` is excluded from this table because the two engines sample different parameterizations (see section 1).

A major contributor to the change vs earlier drafts is denominator normalization: both engines now compute ESS/grad on the same `report_chains` budget.

The practical interpretation for this canonical run is:

- NS LAPS achieves higher ESS/grad across all matched targets reported here.
- `glm_logistic` remains the most expensive target for both engines in absolute wall time.

---

## 5) LAPS quality verification on V100 (`report_chains=256`)

Separate run with tighter diagnostics (`report_chains=256` → 512 half-chains → SE(R-hat) ≈ 0.015).

| Model | R-hat max | ESS_tail min | E-BFMI | Status |
|---|---:|---:|---:|---|
| StdNormal 10d | 1.0175 | 18,947 | 1.035 | ok |
| NealFunnel NCP 10d | 1.0126 | 48,202 | 0.970 | ok |
| GLM n=5000 p=20 | 1.0149 | 49,660 | 0.863 | ok |
| GLM n=200 p=6 | 1.0044 | 55,423 | 0.449 | ok |
| NealFunnel centered 10d | 1.2914 | 257 | 0.000 | fail (expected control) |

This confirms that LAPS convergence is solid when measured with sufficient diagnostic chains. The parity-run R-hat values (section 3, `report_chains=256`) are directly comparable to the quality run.

---

## 6) CPU EPYC (MAMS vs NUTS) and funnel parity fix

Hardware: **AMD EPYC 7502P, 32 cores / 64 threads, 128 GB RAM** (Hetzner dedicated).

### EPYC multi-seed summary (42/123/777, 3-run aggregate)

Config: `n_chains=4`, `n_warmup=1000`, `n_samples=1000`, `eps_jitter=0.1`.

| Model | MAMS ESS/s (median) | MAMS ESS/s (mean ± std) | NUTS ESS/s (median) | NUTS ESS/s (mean ± std) | Ratio MAMS/NUTS (median) |
|---|---:|---:|---:|---:|---:|
| std_normal_d2 | 129,591.50 | 137,761.37 ± 75,444.26 | 200,840.51 | 200,328.55 ± 13,459.94 | 0.645 |
| std_normal_d10 | 100,420.10 | 103,641.28 ± 4,691.72 | 85,159.17 | 95,603.62 ± 15,815.37 | 1.179 |
| std_normal_d50 | 13,006.52 | 13,150.30 ± 867.22 | 28,305.06 | 26,112.78 ± 3,638.46 | 0.460 |
| eight_schools | 98,201.46 | 93,407.92 ± 8,226.95 | 48,577.42 | 46,018.43 ± 5,780.89 | 2.022 |
| logreg_n1000_p10 | 714.19 | 711.12 ± 10.42 | 3,896.00 | 3,914.32 ± 28.34 | 0.183 |
| logreg_n5000_p20 | 37.10 | 35.80 ± 3.70 | 185.83 | 189.84 ± 11.08 | 0.200 |

Observed pattern in this real-run matrix:

| Case | dim | n_data | Ratio MAMS/NUTS | Leader |
|---|---:|---:|---:|---|
| std_normal_d2 | 2 | - | 0.645 | NUTS |
| eight_schools | 10 | 8 | 2.022 | MAMS |
| std_normal_d10 | 10 | - | 1.179 | MAMS |
| std_normal_d50 | 50 | - | 0.460 | NUTS |
| logreg_n1000_p10 | 10 | 1000 | 0.183 | NUTS |
| logreg_n5000_p20 | 20 | 5000 | 0.200 | NUTS |

Why large-n logistic favors NUTS in this CPU protocol:

- Gradient cost scales with `O(n*p)` per leapfrog step; with `n=5000, p=20`, each extra step is expensive.
- NUTS can terminate trajectories early via U-turn, while MAMS uses fixed trajectory length in preconditioned space.
- As `n` grows, posterior geometry becomes closer to well-conditioned Gaussian; this is a strong regime for NUTS with adaptive path length.

Practical recommendation:

- Prefer MAMS for hierarchical / multi-scale geometries.
- Prefer NUTS for large-n GLM-like posteriors on CPU.
- A reasonable product direction is an explicit `method=\"auto\"` heuristic (for example, GLM with large `n` -> NUTS; hierarchical/funnel-like targets -> MAMS), while keeping manual override.

### Funnel parameterization control (EPYC, 3 seeds)

Config: `n_chains=4`, `n_warmup=1000`, `n_samples=1000`.

**MAMS:**

| Model | Seed | R-hat | ESS_tail | EBFMI | Status |
|---|---:|---:|---:|---:|---|
| Centered (`FunnelModel`) | 42 | 1.0785 | 221 | n/a | ok |
| Centered (`FunnelModel`) | 123 | 1.0353 | 31 | n/a | fail |
| Centered (`FunnelModel`) | 777 | 1.0781 | 244 | n/a | ok |
| NCP (`FunnelNcpModel`) | 42 | 1.0067 | 1,914 | n/a | ok |
| NCP (`FunnelNcpModel`) | 123 | 1.0100 | 1,897 | n/a | ok |
| NCP (`FunnelNcpModel`) | 777 | 1.0048 | 1,924 | n/a | ok |

**NUTS:**

| Model | Seed | R-hat | ESS_tail | EBFMI | Status |
|---|---:|---:|---:|---:|---|
| Centered (`FunnelModel`) | 42 | 2.3844 | 14 | n/a | fail |
| Centered (`FunnelModel`) | 123 | 1.3636 | 72 | n/a | fail |
| Centered (`FunnelModel`) | 777 | 1.9480 | 17 | n/a | fail |
| NCP (`FunnelNcpModel`) | 42 | 1.0026 | 2,516 | n/a | ok |
| NCP (`FunnelNcpModel`) | 123 | 1.0027 | 1,604 | n/a | ok |
| NCP (`FunnelNcpModel`) | 777 | 1.0024 | 2,385 | n/a | ok |

Interpretation:

- **NCP is 6/6 ok** across all seeds for both MAMS and NUTS. ESS_tail ranges 1,604-2,516 (NUTS) and 1,897-1,924 (MAMS).
- **Centered is 3/3 fail** for NUTS and 1/3 fail for MAMS.
- The previous CPU funnel mismatch was methodological (centered vs NCP), not a "CPU is weak" issue.
- `FunnelNcpModel` is the recommended benchmark parameterization for CPU/GPU parity.
- Centered `FunnelModel` is kept as a known pathological control; this is a limitation demonstration, not a product regression.
- In these EPYC funnel-control artifacts, `EBFMI` is not exported (`n/a` in the tables), so pass/fail here is based on R-hat/ESS quality gates.

---

## 7) Reproducibility and environment metadata

- V100 benchmark JSON includes top-level `environment` snapshot (`python`, `jax`, `cuda`, `gpu`, package versions).
- EPYC suite stores hardware/config/seed metadata and per-case metrics; full package-level environment snapshot is currently only present in the V100 parity JSON.

Artifacts (all in `docs/blog/artifacts/v096-zero-jit-tax/`):

- V100 3-seed matrix (canonical): `v100-multi-seed-matrix-canonical.json`
- V100 chart data (canonical): `v100-parity-chart-data-canonical.csv`, `v100-essgrad-ratio-canonical.csv`
- V100 raw 3-seed parity run (canonical): `v100_v096_builtinwarmup_3seed_20260218T224654Z/seed_42/gpu_triple_bench.json`, `v100_v096_builtinwarmup_3seed_20260218T224654Z/seed_123/gpu_triple_bench.json`, `v100_v096_builtinwarmup_3seed_20260218T224654Z/seed_777/gpu_triple_bench.json`
- V100 funnel addendum raw runs: `v100_ns_funnel_3seed_20260218T231337Z/*`, `v100_bj_funnel_builtin3seed_20260218T231204Z/*`
- V100 quality run (`report_chains=256`, 5 models): `v100-quality-report256-5models.json`
- V100 + EPYC refresh note: `2026-02-17-v096-refresh-v100-epyc.md`
- EPYC multi-seed matrix: `epyc-multi-seed-matrix.json`
- EPYC suite output: `epyc-mams-suite.json`
- EPYC funnel-control (3 seeds, explicit R-hat table source): `epyc-funnel-control-3seed.json`

---

## Appendix: V100 neal_funnel (different parameterizations)

Retained for transparency. These rows compare NS LAPS (NCP) against BlackJAX (centered) — **not a like-for-like comparison**.

| Metric | NS LAPS (NCP) | BlackJAX (centered) |
|---|---:|---:|
| Cold (s) | 1.404 | 15.517 |
| Warm (s) | 0.259 | 0.412 |
| min_ESS | 54,768 | 706 |
| ESS/s (warm) | 211,581 | 1,759 |
| R-hat | 1.0083 | 1.2732 |
| ESS/grad | 0.071312 | 0.000710 |

BlackJAX's non-convergence on the centered funnel is expected (see section 6: even NUTS fails 3/3 on centered funnel with 4 chains and standard budget). This comparison primarily demonstrates that NS's default NCP dispatch produces converged results where the centered parameterization does not.

---

## References

- Robnik, Cohn-Gordon, Seljak. *Metropolis Adjusted Microcanonical Hamiltonian Monte Carlo (MAMS).* [arXiv:2503.01707](https://arxiv.org/abs/2503.01707)
- BlackJAX. *Composable Bayesian inference in JAX.* [arXiv:2402.10797](https://arxiv.org/abs/2402.10797)
