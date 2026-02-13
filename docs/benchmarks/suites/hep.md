---
title: "Benchmark Suite: HEP (HistFactory)"
description: "HEP benchmark suite for NextStat: NLL evaluation, gradients, MLE fits, profile scans, toy ensembles, and GPU batch benchmarks against pyhf and ROOT/RooFit with correctness-gated protocols."
status: draft
last_updated: 2026-02-08
keywords:
  - HistFactory benchmark
  - pyhf comparison
  - ROOT RooFit benchmark
  - NLL evaluation performance
  - profile likelihood scan
  - toy ensemble throughput
  - CUDA GPU batch fit
  - Metal GPU benchmark
  - HEP statistical inference
  - NextStat
---

# HEP Benchmark Suite (HistFactory / pyhf / ROOT)

This suite benchmarks NextStat on the workflows that dominate binned-likelihood HEP analyses:

- NLL evaluation and gradients
- MLE fits (unconditional and conditional)
- profile likelihood scans
- hypothesis tests / toy ensembles

This page is a **runbook + methodology**. Results are published as benchmark snapshots (see [Public Benchmarks](/docs/public-benchmarks)).

## What is compared

Planned comparisons include:

- **NextStat (Rust core + Python bindings)** vs **pyhf** (reference implementation)
- Optional: **ROOT/RooFit/RooStats** baselines for selected workflows (where feasible to automate)

## Inputs (workspaces)

We benchmark on two categories of inputs:

1. **Curated real(istic) workspaces** (multi-channel, high-parameter-count), e.g. fixtures under `tests/fixtures/` such as:
   - `simple_workspace.json` (small sanity case)
   - `complex_workspace.json` (systematics-heavy)
   - `workspace_tHu.json`, `tttt-prod_workspace.json`, etc. (large parameter vectors)
2. **Synthetic scaling workspaces** to measure asymptotics (e.g., shapesys-per-bin scaling).

## Measured tasks

### 1) NLL time / call

Measures the cost of one likelihood evaluation at a fixed parameter vector.

Correctness gating:

- Verify `|NLL_nextstat - NLL_pyhf|` within tolerance before measuring.

Existing harness (today, in-repo):

- `tests/benchmark_pyhf_vs_nextstat.py` (Python end-to-end, includes correctness sanity before timing)

Seed harness (public benchmarks repo bootstrap):

- `benchmarks/nextstat-public-benchmarks/suites/hep/` (single-case `run.py` and multi-case `suite.py`)

Publishable artifacts under pinned schemas:

- `nextstat.benchmark_result.v1` per case
- `nextstat.benchmark_suite_result.v1` index

Optional baseline artifacts (seed):

- `nextstat.hep_root_baseline_result.v1` for ROOT baseline runs

The seed harness also supports optional full MLE fit timing via `--fit --fit-repeat N` and records results in the per-case JSON under the `fit` block.

Profile-level metrics (seed):

- `--profile` computes discovery-style profiled `q0` / `Z0` (conditional fit at `mu=0`) and writes them under a `profile` block.
- The public harness treats missing optional backends as best-effort: if conditional fits are not available, it emits `profile.status="skipped"` with an actionable `reason`.

Baseline templates (optional): `benchmarks/nextstat-public-benchmarks/suites/hep/baselines/`

### 2) Gradient time / call

Measures the cost of gradient evaluation required for optimizers/samplers.

Notes:

- Comparisons must be explicit about *which* gradients (w.r.t. which parameterization) are evaluated and whether the framework exposes analytical gradients or uses AD/FD.

### 3) MLE fits (wall-time + convergence)

Measures:

- wall-time to convergence under a declared stopping rule
- number of objective/gradient evaluations (when exposed)
- stability across warm-start policies

Correctness gating:

- verify that best-fit POI and `ΔNLL` are consistent within tolerance (reference-dependent).

### 4) Profile likelihood scans

Measures:

- scan time for a fixed set of POI values
- warm-start policy (e.g., reuse previous scan point)
- tail-point behavior (local minima risk)

### 5) Toy ensembles / pseudo-experiments

Measures:

- toys/sec at fixed model size and toy count
- scaling with parameter count
- CPU multi-thread vs GPU batch (where supported)

## Protocol (timing + reporting)

All suite scripts must declare:

- warmup policy and what is excluded from timing (imports/JIT/kernel load)
- repetition policy (N repeats, report median and distribution)
- environment manifest capture (toolchain + deps + hardware)

## How to run locally (current)

Python end-to-end benchmark vs pyhf:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/benchmark_pyhf_vs_nextstat.py --fit
```

Rust microbenchmarks (Criterion):

```bash
cargo bench -p ns-inference --bench mle_benchmark
```

## Known pitfalls (what we will not "benchmark away")

- Reference mismatches (wrong interpolation codes, constraints, masking) invalidate the benchmark; correctness gates must fail fast.
- Cold-start vs warm-start must be declared; mixing them produces misleading scan numbers.
- For Python comparisons, "Python overhead included" vs "core-only" must be separated and both published.

## Related reading

- [Public Benchmarks Specification](/docs/public-benchmarks) — canonical spec.
- [Benchmarks — GPU section](/docs/benchmarks#gpu-benchmarks-cuda) — full GPU benchmark tables.
- [Validation Report Artifacts](/docs/validation-report) — validation pack for published snapshots.
- [Numerical Accuracy](/blog/numerical-accuracy) — ROOT/pyhf/NextStat 3-way profile scan comparison.
