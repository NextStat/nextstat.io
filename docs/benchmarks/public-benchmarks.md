---
title: "Public Benchmarks"
status: draft
last_updated: 2026-02-08
---

# Public Benchmarks (Trust Offensive)

This page is the **canonical specification** for how NextStat benchmarks are designed, executed, and published. The goal is not “a fast number” — it is **trustworthy, reproducible evidence** that other people can rerun and audit.

If you want the narrative version (why we’re doing this and what it changes), see the blog posts in `docs/blog/`.

## Scope

We benchmark **end-to-end user workflows** rather than isolated micro-kernels:

- HEP / HistFactory: NLL evaluation, gradients, MLE fits, profile scans, toy ensembles.
- Pharma: PK/NLME likelihood evaluation + fitting loops.
- Bayesian: gradient-based samplers (NUTS) with ESS/sec and wall-time.
- ML infra: compilation vs execution time (e.g., JAX compile latency vs steady-state throughput) where relevant to scientific pipelines.

We also publish “fast path vs reference path” comparisons (e.g., parity mode vs fast mode) when correctness contracts are part of the product.

## Non-goals

- “Hero” runs on one cherry-picked machine.
- Benchmarks that depend on undocumented caches, hidden warmups, or hand-tuned flags.
- Performance claims without a reproducibility story (exact versions, inputs, and scripts).

## Trust model (what you should be able to verify)

For every published snapshot you should be able to answer, from artifacts alone:

1. **What was measured?** (definition of tasks + metrics)
2. **On what data?** (dataset ID + hash + license)
3. **Under what environment?** (OS, CPU/GPU, compiler, Python, dependency versions)
4. **From what code?** (NextStat commit hash, dependency lockfiles, build flags)
5. **Does it still match reference?** (sanity/parity checks before timing)
6. **How stable is the number?** (repeat strategy, distributions, and reporting)

## Reproducibility contract

### Environment pinning

Published runs must include:

- `rust-toolchain.toml` and `Cargo.lock` (Rust toolchain + dependencies)
- Python version + dependency lock (e.g., uv/pip-tools/poetry lock)
- GPU runtime details when used (CUDA version / Metal / driver)

### Benchmark harness is open source

The harness scripts are part of the repo, not copy-pasted from blog posts.

Today (in this repo) the main entry points are:

- Python end-to-end comparisons vs pyhf: `tests/benchmark_pyhf_vs_nextstat.py`
- Rust-only microbenchmarks: `cargo bench --workspace` (Criterion)

### Correctness gating before timing

Before recording performance numbers, the harness must validate that the output is sane and (when applicable) matches a reference implementation within a stated tolerance.

Example: the pyhf-vs-NextStat harness verifies NLL agreement before it prints timings (`tests/benchmark_pyhf_vs_nextstat.py`).

## What we publish (artifacts)

Each benchmark snapshot should publish:

- Raw per-test measurements (not just a final table)
- Summary tables (median/best-of-N policy must be explicit)
- A **baseline manifest**: code SHA, env versions, dataset hashes, and run configuration
- Any correctness/parity reports used as gating

## Suites (planned structure)

### HEP suite (pyhf + ROOT/RooFit harness)

**Docs:** this site (how to run, what’s measured, what is gated).  
**Blog:** results + interpretation once snapshots are public.

Measurements:

- NLL time / call (CPU parity mode and fast mode)
- Gradient time / call (where exposed)
- MLE fit wall-time and convergence behavior
- Profile scan wall-time with warm-start policy
- Toy ensemble throughput (toys/sec) for CPU and GPU batch modes

Correctness gates:

- NLL parity vs pyhf at representative parameter points
- Fit-level checks (POI estimates, likelihood differences within tolerance)

### Pharma suite (PK/NLME plan + datasets)

Measurements:

- Likelihood + gradient time for standard models
- Fit wall-time (fixed iteration protocols to avoid “stopping rule” ambiguity)
- Scaling with subject count / observation count

### Bayesian suite (ESS/sec vs Stan + PyMC)

Primary metrics:

- ESS/sec (bulk ESS and tail ESS) per parameter group
- Wall-time per effective draw

Notes:

- ESS is only meaningful with matched model, priors, and diagnostics settings.
- We must publish the exact inference settings (step size adaptation, target accept, mass matrix policy).

### ML suite (JAX compile vs execution)

Primary metrics:

- Cold-start latency (compile time)
- Warm throughput (steady-state execution)
- Memory footprint (where measurable and stable)

## Publishing (CI artifacts + baselines)

We publish benchmark snapshots via CI:

- Every run has a unique, immutable identifier.
- Artifacts include raw results + baseline manifest.
- Baseline comparisons are opt-in and versioned (no silent “moving targets”).

## DOI + citation

Benchmark snapshots that are stable enough to cite should be published with a DOI (e.g., Zenodo) and a machine-readable citation file (e.g., `CITATION.cff`).

## Third-party replication

The strongest trust signal is an independent rerun. The replication process should produce:

- A rerun log with the same harness
- The baseline manifest of the rerun environment
- A signed report comparing the rerun vs the published snapshot

## Blog posts (narrative)

- Trust Offensive: Public Benchmarks — why this exists and how to interpret it.
- “The End of the Scripting Era” — how reproducible benchmarking changes how we build scientific software.

