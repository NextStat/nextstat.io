---
title: "NUTS Sampler — Algorithm & Benchmark Report"
status: stable
---

# NUTS Sampler

NextStat implements the No-U-Turn Sampler (NUTS) with multinomial trajectory
sampling and the generalized U-turn criterion (Betancourt 2017).  The
implementation targets algorithmic parity with CmdStan 2.38 (released 2026-01-13).

## Algorithm

### Core

- **Multinomial NUTS**: leaf weights are `exp(-energy_error)`, proposal selected
  via multinomial sampling across the full trajectory tree.
- **Generalized U-turn criterion**: three checks per subtree merge (see below).
- **Tree doubling**: `depth < max_treedepth` (default 10 &rarr; up to 1024
  leapfrog steps).

### Stan-exact 3-criterion U-turn

When two subtrees are merged (both inside `build_tree` recursion and in the
outer `nuts_transition` loop), three independent U-turn conditions are
evaluated:

| # | Name | Formula | Purpose |
|---|------|---------|---------|
| 1 | Full tree | `is_turning(rho_total, p_left, p_right)` | Overall trajectory progress |
| 2 | Init-to-junction | `is_turning(rho_init + p_final_junction, p_start, p_final_junction)` | Catches U-turns at the boundary between subtrees |
| 3 | Junction-to-final | `is_turning(rho_final + p_init_junction, p_init_junction, p_end)` | Symmetric check from the other side |

If **any** of the three checks detects a U-turn, the tree stops growing.
This matches Stan's `base_nuts.hpp` ("Demand satisfaction around/between
merged subtrees").

**Impact**: eliminates a long tail of unnecessarily deep trees (depth 4-7)
that a single-check implementation produces.  Mean tree depth for the GLM
benchmark dropped from 3.02 to 2.68 (CmdStan: 2.59).

### Mass matrix adaptation

- **Windowed warmup** (Stan-style 3-phase schedule): init &rarr; slow windows
  (25, 50, 100, 200, &hellip;) &rarr; last window absorbs remainder.
- **Metric types**: `Diagonal` (default, matches CmdStan `diag_e`), `Dense`,
  `Auto` (dense for dim &le; 32).
- **Regularization** (Stan-exact):
  - Diagonal: `alpha * var + 1e-3 * (1 - alpha) * ones`
  - Dense: `alpha * cov + 1e-3 * (1 - alpha) * I`
  - where `alpha = n / (n + 5)`, `n` = sample count in the window.

### Step-size re-search after metric update

When the mass matrix is updated at a slow window boundary, NextStat
re-searches for a reasonable step size using the new metric (matching Stan's
`init_stepsize()` + `set_mu()` + `restart()` in `adapt_diag_e_nuts`).
Dual averaging is only reset on these metric-update boundaries &mdash; not
at the init or terminal window boundaries.

**Impact**: +14% ESS/sec on GLM vs the previous implementation (v8 &rarr; v9).

### Step-size jittering (optional)

After warmup, each transition can optionally use a jittered step size:
`eps * (1 + jitter * U(-1,1))`.  This is controlled by the `stepsize_jitter`
parameter (default 0, matching Stan's default).

Benchmark results show jittering does not improve ESS on well-conditioned
models (GLM: &minus;7% at j=0.1, &minus;18% at j=0.5) but may help on
models with periodic trajectory patterns.

### Other Stan-parity details

| Feature | NextStat | CmdStan |
|---------|----------|---------|
| Step-size search | `eps_0 = 1.0`, double/halve | Same |
| Dual averaging | `gamma=0.05, t0=10, kappa=0.75` | Same |
| Proposal selection | Within-tree multinomial; top-level progressive sampling | Same |
| Init strategy | `Uniform(-2, 2)` unconstrained, 100 retries | Same |
| Divergence threshold | `abs(delta_H) > 1000` | Same |
| Jacobian corrections | `log|J|` added to log-density, chain rule in gradient | Same |
| Step-size re-search | After every metric update | Same |

## Python API

```python
import nextstat

result = nextstat.sample(
    model,
    n_chains=4,           # parallel via Rayon (GIL released)
    n_warmup=1000,
    n_samples=2000,
    seed=42,
    metric="diagonal",    # "diagonal" | "dense" | "auto"
    target_accept=0.8,
    max_treedepth=10,
    init_strategy="random",
    stepsize_jitter=0.0,  # optional: 0.0-1.0
)
```

Returns a dict with keys: `draws`, `sample_stats` (tree depths, accept probs,
step sizes, energies, divergences), `diagnostics` (R-hat, ESS bulk/tail, E-BFMI,
divergence rate).

## Benchmarks

### Setup

- **Server**: AMD EPYC 7502P (32C/64T), 128 GB RAM
- **Config**: 4 chains, 1000 warmup, 2000 samples, diagonal metric,
  `target_accept = 0.8`
- **Competitors**: CmdStan 2.38.0 (via `cmdstanpy`). Other optional baselines
  (PyMC, NumPyro) are supported by the harness but not included in the v10
  publication artifacts.
- **Metric**: min ESS (bulk) / wall time across all parameters
- **Seeds**: dataset fixed (`dataset_seed = 12345`), chain seeds `{42, 0, 123}`
- **Artifacts**: `benchmarks/nextstat-public-benchmarks/suites/bayesian/results_v10/`

### Results (ESS/sec, avg 3 seeds; min ESS_bulk / wall time)

All results use 4 chains, 1000 warmup, 2000 samples, diagonal metric, and are
averaged across seeds `{42, 0, 123}` with a fixed dataset seed for generated
cases.

| Model | NextStat | CmdStan | NS / CmdStan |
|-------|----------|---------|--------------|
| GLM logistic (6p) | **29,895** | 29,600 | **1.01x** |
| Hier random intercept (22p) | **3,255** | 1,015 | **3.21x** |
| Eight Schools NCP (10p) | **54,083** | 26,644 | **2.03x** |

Algorithmic efficiency (proposal selection quality) measured as ESS/leapfrog on
GLM logistic:

| Metric | NextStat | CmdStan | NS / CmdStan |
|--------|----------|---------|--------------|
| ESS/leapfrog | **0.228** | 0.195 | **116.7%** |

### Reproducibility

Seed variance depends on both dataset generation and chain seeds; for
publishable snapshots we keep dataset seeds fixed and vary chain seeds.

| Model | NextStat min ESS_bulk range | CmdStan min ESS_bulk range |
|-------|-------------------|--------------------|
| GLM logistic (3 seeds) | 8000.0 -- 8000.0 | 7927.6 -- 8172.2 |
| Hier random intercept (3 seeds) | 2871.8 -- 3646.8 | 2051.8 -- 2774.5 |
| Eight Schools (3 seeds) | 5068.2 -- 5811.4 | 4445.4 -- 5498.7 |

NextStat's ESS/sec is typically stable to within ~1% across repeated runs; the
largest observed variance is usually due to external backends (process startup,
OS scheduling).

### Tree depth distribution (all models)

| Model | NextStat | CmdStan |
|-------|----------|---------|
| GLM (6p) | d2: 29%, d3: 71% | d2: 36%, d3: 64% |
| Hier (22p) | d3: 41%, d4: 58%, d5: 1% | d3: 77%, d4: 23% |
| 8Schools (10p) | d3: 16%, d4: 84%, d5: 0.3% | d3: 33%, d4: 65%, d5: 2% |

### Diagnostic quality

All benchmarks pass standard diagnostic thresholds:

| Criterion | Threshold | NextStat (all models) |
|-----------|-----------|----------------------|
| R-hat | < 1.01 | Pass (max 1.004) |
| ESS bulk | > 400 per chain | Pass |
| E-BFMI | > 0.3 | Pass (min 0.94) |
| Divergence rate | < 1% | Pass (0.0%) |

## Stan-parity fix: progressive sampling

The remaining GLM gap in v9 was not PRNG-driven.  It came from a top-level
proposal-selection mismatch with Stan.

Within `build_tree()`, both Stan and NextStat use multinomial selection
proportional to subtree weights: `W_outer / (W_inner + W_outer)`.

However, at the **top level** (joining a new subtree into the running tree),
Stan uses **progressive sampling**:

- Stan: accept subtree proposal with probability `min(1, W_subtree / W_existing)`
  (computed **before** `W_existing` is updated by `W_subtree`).
- NextStat v9: used multinomial `W_subtree / (W_existing + W_subtree)` at the
  top level, which keeps the initial point in the proposal pool too often on
  well-conditioned models.

Switching NextStat to Stan-style progressive sampling doubles ESS/leapfrog on
GLM logistic (0.112 → 0.228) and improves sampler efficiency across all tested
models.

## Potential improvements

1. **PRNG exploration**: evaluate alternative engines (Philox, Xoshiro) for
   potentially better trajectory diversity on simple geometries.
2. **Warmup schedule tuning**: experiment with longer final windows or
   different window growth rates for improved mass matrix estimates.

## Source files

| File | Purpose |
|------|---------|
| `crates/ns-inference/src/nuts.rs` | NUTS sampler, build_tree, nuts_transition |
| `crates/ns-inference/src/adapt.rs` | Windowed adaptation, dual averaging, Welford estimators |
| `crates/ns-inference/src/hmc.rs` | Leapfrog integrator, Metric (Diag/DenseCholesky) |
| `crates/ns-inference/src/posterior.rs` | Bijector transforms, Jacobian corrections |
| `crates/ns-inference/src/chain.rs` | Multi-chain orchestration (Rayon parallel) |
| `crates/ns-inference/src/eight_schools.rs` | Eight Schools model (non-centered) |
| `crates/ns-inference/src/diagnostics.rs` | R-hat, ESS, E-BFMI computation |
