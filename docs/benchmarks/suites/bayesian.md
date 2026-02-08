---
title: "Benchmark Suite: Bayesian (ESS/sec)"
status: draft
last_updated: 2026-02-08
---

# Bayesian Benchmark Suite (ESS/sec vs Stan + PyMC)

This suite benchmarks Bayesian inference workflows with metrics that matter:

- **ESS/sec** (bulk + tail) and wall-time per effective draw
- diagnostic stability under clearly specified settings

The primary goal is to avoid “apples vs oranges” comparisons by publishing:

- the exact model + priors
- the exact inference settings
- the exact diagnostics and ESS computation policy

## What is compared

Planned comparisons:

- NextStat NUTS implementation (Rust core) vs Stan vs PyMC (where feasible)

## What is measured

### ESS/sec

Report:

- bulk ESS/sec and tail ESS/sec (per parameter group)
- wall-time per effective draw

### Warmup + adaptation behavior

Publish:

- warmup length
- target acceptance
- step-size adaptation policy
- mass matrix policy (diag vs dense) and update schedule

## Protocol requirements (to keep the comparison honest)

- Same model and prior parameterization across frameworks.
- Same effective warmup/sampling budgets (or publish both and justify).
- Same RNG seeding policy (where supported) and deterministic preprocessing.
- Diagnostics must be computed with the same method/version (or explicitly noted).

## Harness entry points (current)

NextStat provides Criterion benches for NUTS in:

- `crates/ns-inference/benches/nuts_benchmark.rs`

Run locally:

```bash
cargo bench -p ns-inference --bench nuts_benchmark
```

For public benchmarks, we will wrap these benches (and external-framework runs) into a single harness that produces:

- raw draws (or summary traces)
- ESS metrics
- environment manifests

## Threats to validity (things we will document explicitly)

- Different gradient implementations (analytical vs AD) change the cost model.
- Parameterization differences (centered vs non-centered) dominate sampler efficiency.
- BLAS backend differences can dominate linear algebra-heavy models.

