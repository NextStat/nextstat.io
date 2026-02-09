---
title: "Public Benchmarks"
description: "Canonical specification for NextStat's public benchmark program: protocols, correctness gates, environment pinning, artifacts, and suite structure for reproducible performance evidence across HEP, pharma, Bayesian, ML, time series, and econometrics."
status: shipped
last_updated: 2026-02-08
keywords:
  - public benchmarks
  - reproducible benchmarks
  - scientific software validation
  - HistFactory benchmark
  - pyhf comparison
  - NLME benchmark
  - NUTS sampler benchmark
  - benchmark protocol
  - NextStat
---

# Public Benchmarks (Trust Offensive)

This page is the **canonical specification** for how NextStat benchmarks are designed, executed, and published. The goal is not "a fast number" — it is **trustworthy, reproducible evidence** that other people can rerun and audit.

If you want the narrative version (why we're doing this and what it changes), see the blog posts:

- [Trust Offensive: Public Benchmarks](/blog/trust-offensive-public-benchmarks)
- [The End of the Scripting Era](/blog/end-of-scripting-era-benchmarks)
- [Benchmark Snapshots as Products](/blog/benchmark-snapshots-ci-artifacts)
- [Third-Party Replication: Signed Reports](/blog/third-party-replication-signed-report)
- [Building a Trustworthy HEP Benchmark Harness](/blog/hep-benchmark-harness)
- [Numerical Accuracy](/blog/numerical-accuracy)
- [Differentiable HistFactory in PyTorch](/blog/differentiable-layer-nextstat-pytorch)
- [Bayesian Benchmarks: ESS/sec](/blog/bayesian-benchmarks-ess-per-sec)
- [Pharma Benchmarks: PK/NLME](/blog/pharma-benchmarks-pk-nlme)
- [JAX Compile vs Execution](/blog/jax-compile-vs-execution)

## What goes in docs vs blog

**Docs (this site) are canonical**: protocols, contracts, runbooks, and “how to rerun” instructions.

**Blog posts are narrative**: motivation, design rationale, interpretation of results, and “what this changes” framing.

Rule of thumb:

- If a reader needs to *execute* the benchmark, it belongs in **docs**.
- If a reader needs to *understand why* the benchmark program exists, it belongs in the **blog**.

## Scope

We benchmark **end-to-end user workflows** rather than isolated micro-kernels:

- HEP / HistFactory: NLL evaluation, gradients, MLE fits, profile scans, toy ensembles.
- Pharma: PK/NLME likelihood evaluation + fitting loops.
- Bayesian: gradient-based samplers (NUTS) with ESS/sec and wall-time.
- ML infra: compilation vs execution time (e.g., JAX compile latency vs steady-state throughput) where relevant to scientific pipelines.
- Time Series: Kalman filter/smoother throughput, EM convergence cost, forecasting latency.
- Econometrics: panel FE fit wall-time, DiD TWFE + event study wall-time, IV/2SLS first-stage + second-stage cost, AIPW doubly-robust estimator vs naive OLS.

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
- A **validation report** (`validation_report.json` + optional `validation_report.pdf`) produced by [`nextstat validation-report`](/docs/references/validation-report), containing dataset SHA-256 fingerprint, model spec, environment, and per-suite pass/fail summary

## Suites

### HEP suite (pyhf + ROOT/RooFit harness)

**Docs:** this site (how to run, what’s measured, what is gated).  
**Blog:** results + interpretation once snapshots are public.

Suite doc: [HEP Benchmark Suite](/docs/benchmarks/suites/hep).

Measurements:

- NLL time / call (CPU parity mode and fast mode)
- Gradient time / call (where exposed)
- MLE fit wall-time and convergence behavior
- Profile scan wall-time with warm-start policy
- Toy ensemble throughput (toys/sec) for CPU and GPU batch modes

Correctness gates:

- NLL parity vs pyhf at representative parameter points
- Fit-level checks (POI estimates, likelihood differences within tolerance)

GPU measurements (shipped):

- CPU vs CUDA vs Metal batch toy throughput (toys/sec at 100–5000 toys)
- Profile scan crossover analysis (~150+ parameters for GPU advantage)
- Differentiable layer latency (NLL + signal gradient, profiled q₀)

### Pharma suite (PK/NLME + analytic reference baselines)

Suite doc: [Pharma Benchmark Suite](/docs/benchmarks/suites/pharma).

Measurements:

- Likelihood + gradient time for standard models
- Fit wall-time (fixed iteration protocols to avoid “stopping rule” ambiguity)
- Scaling with subject count / observation count

### Bayesian suite (ESS/sec vs Stan + PyMC)

Suite doc: [Bayesian Benchmark Suite](/docs/benchmarks/suites/bayesian).

Primary metrics:

- ESS/sec (bulk ESS and tail ESS) per parameter group
- Wall-time per effective draw

Notes:

- ESS is only meaningful with matched model, priors, and diagnostics settings.
- We must publish the exact inference settings (step size adaptation, target accept, mass matrix policy).

### ML suite (compile vs execution + differentiable pipelines)

Suite doc: [ML Benchmark Suite](/docs/benchmarks/suites/ml).

Primary metrics:

- Cold-start latency (compile time)
- Warm throughput (steady-state execution)
- Memory footprint (where measurable and stable)

### Time Series suite (Kalman + state space)

Suite doc: [Time Series Benchmark Suite](/docs/benchmarks/suites/timeseries).

Measurements:

- Kalman filter/smoother throughput (states/sec) at varying state dimension
- EM convergence cost (iterations × NLL evaluations)
- Forecasting latency per horizon step

### Econometrics suite (panel + causal inference)

Suite doc: [Econometrics Benchmark Suite](/docs/benchmarks/suites/econometrics).

Measurements:

- Panel FE fit wall-time scaling with entity count and cluster count
- DiD TWFE + event study wall-time
- IV/2SLS first-stage + second-stage cost
- AIPW doubly-robust estimator vs naive OLS

## Publishing (CI artifacts + baselines)

We publish benchmark snapshots via CI:

- Every run has a unique, immutable identifier.
- Artifacts include raw results + baseline manifest + `validation_report.json`.
- Baseline comparisons are opt-in and versioned (no silent "moving targets").

Publishing + replication doc: [Publishing Benchmarks](/docs/benchmarks/publishing).

Benchmarks repo skeleton (pinned envs, manifests): [Benchmarks Repo Skeleton](/docs/benchmarks/repo-skeleton).

First run playbook (step-by-step): [First Public Benchmark Snapshot (Playbook)](/docs/benchmarks/first-public-snapshot).

Seed harness directory (in this repo, to bootstrap the standalone benchmarks repo): `benchmarks/nextstat-public-benchmarks/`.

Seed publishing helper (in this repo): `benchmarks/nextstat-public-benchmarks/scripts/publish_snapshot.py` can generate a local snapshot directory under `benchmarks/nextstat-public-benchmarks/manifests/snapshots/<snapshot_id>/` and schema-validate the produced artifacts.

## Live pages (published artifacts)

For the live, user-facing registry of published artifacts on nextstat.io:

- Benchmark Results: [/docs/benchmark-results](/docs/benchmark-results)
- Snapshot Registry: [/docs/snapshot-registry](/docs/snapshot-registry)

Machine-readable registry:

- The standalone public benchmarks repo also maintains a commit-backed `snapshot_registry.json` (used to drive/verify the live pages and automation).

## DOI + citation

Benchmark snapshots that are stable enough to cite should be published with a DOI (e.g., Zenodo) and a machine-readable citation file (e.g., `CITATION.cff`).

Production DOI note: first production record is published at DOI `10.5281/zenodo.18542624` (https://zenodo.org/records/18542624).

Pipeline validation note: we also have a Zenodo **sandbox** record published (DOI `10.5072/zenodo.437330`). Sandbox DOIs are not intended for real citation.

## Third-party replication

The strongest trust signal is an independent rerun. The replication process should produce:

- A rerun log with the same harness
- The baseline manifest of the rerun environment
- A signed report comparing the rerun vs the published snapshot

Replication bundle (production DOI): `10.5281/zenodo.18543606` (https://zenodo.org/records/18543606). This replication record links back to the published snapshot DOI `10.5281/zenodo.18542624`.

Runbook: [Third-Party Replication Runbook](/docs/benchmarks/replication).

## Blog posts (narrative)

- [Trust Offensive: Public Benchmarks](/blog/trust-offensive-public-benchmarks) — why this exists and how to interpret it.
- [The End of the Scripting Era](/blog/end-of-scripting-era-benchmarks) — how reproducible benchmarking changes how we build scientific software.
- [Third-Party Replication: Signed Reports](/blog/third-party-replication-signed-report) — external reruns + signed validation reports as the ultimate trust signal.
