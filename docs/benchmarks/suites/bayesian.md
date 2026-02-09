---
title: "Benchmark Suite: Bayesian (ESS/sec)"
description: "Bayesian inference benchmark suite for NextStat: ESS/sec (bulk + tail), wall-time per effective draw, warmup/adaptation behavior, and SBC calibration — with honest comparisons vs Stan and PyMC under fully specified settings."
status: draft
last_updated: 2026-02-09
keywords:
  - NUTS sampler benchmark
  - ESS per second
  - Bayesian inference performance
  - Stan comparison
  - PyMC comparison
  - MCMC benchmark
  - simulation-based calibration
  - Hamiltonian Monte Carlo
  - NextStat NUTS
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

## Initial public baseline set (recommended)

For the first public Bayesian snapshots, we recommend a small set of models that cover distinct NUTS regimes:

1. **Simple HistFactory** (~8 parameters): fast, validates basic ESS/sec and R-hat convergence in an inference-like likelihood.
2. **Logistic regression (GLM)**: classic benchmark, straightforward to reproduce in Stan/PyMC.
3. **Hierarchical random intercepts** (non-centered): exercises funnel-like geometry and parameterization sensitivity.

Large HEP workspaces (e.g. `tHu`, 184 parameters) are valuable but typically too slow for nightly runs; keep them behind `#[ignore]` or run them as release-only smoke tests.

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

Standalone public benchmarks harness (seed repo) status:

- runnable implementation: **seed (NextStat + optional Stan/PyMC)**
- suite directory: `benchmarks/nextstat-public-benchmarks/suites/bayesian/`
- CLI: `python3 suites/bayesian/suite.py --out-dir ... --backends nextstat,cmdstanpy,pymc`

Dependency note (seed repo):

- Core harness deps are pinned and minimal.
- Optional backends require extra deps (and for Stan, **CmdStan** binaries). Missing optional deps are reported as `status="warn"` with an actionable `reason`, rather than failing the whole snapshot.

## SBC calibration suite (shipped)

The SBC (Simulation-Based Calibration) suite validates posterior correctness independently of performance:

- Generates synthetic datasets from the prior
- Runs NUTS sampling on each
- Checks rank uniformity of posterior draws vs true parameters

This is included in the Apex2 master report under the `sbc` key.

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_sbc_report.py \
  --out tmp/apex2_sbc_report.json
```

The NUTS quality smoke suite additionally checks divergence rate, R-hat, ESS, and E-BFMI floors:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_nuts_quality_report.py \
  --out tmp/apex2_nuts_quality_report.json
```

## Threats to validity (things we will document explicitly)

- Different gradient implementations (analytical vs AD) change the cost model.
- Parameterization differences (centered vs non-centered) dominate sampler efficiency.
- BLAS backend differences can dominate linear algebra-heavy models.

## Related reading

- [Public Benchmarks Specification](/docs/benchmarks/public-benchmarks) — canonical spec.
- [Validation Report Artifacts](/docs/references/validation-report) — validation pack for published snapshots.
