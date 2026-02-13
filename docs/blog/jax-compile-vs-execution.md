---
title: "JAX Compile vs Execution: The Benchmark You Actually Need"
slug: jax-compile-vs-execution
description: "Why compile latency matters in scientific ML pipelines and how NextStat plans to benchmark cold-start vs warm throughput with reproducible harnesses and published artifacts."
date: 2026-02-08
author: NextStat Team
status: draft
keywords:
  - JAX compile latency
  - compile vs execution benchmark
  - cold start latency
  - warm throughput
  - reproducible benchmarks
  - scientific ML pipelines
  - NextStat
category: ml
---

<!--
  Blog draft (technical).
  Suite runbook: docs/benchmarks/suites/ml.md
-->

# JAX Compile vs Execution: The Benchmark You Actually Need

**Trust Offensive series:** [Index](/blog/trust-offensive) · **Prev:** [Pharma Benchmarks: PK/NLME](/blog/pharma-benchmarks-pk-nlme) · **Next:** [Trust Offensive series index](/blog/trust-offensive)

Many ML “benchmarks” measure only steady-state throughput.

That’s the right metric if you run a model for hours.

But in scientific pipelines, a lot of work happens in short loops:

- hyperparameter sweeps
- repeated small fits
- interactive analysis iterations
- short training runs for ablations

In those settings, **compile latency** can dominate the total cost.

This post explains how we benchmark “compile vs execution” as a first-class measurement — and how we publish it as a reproducible artifact instead of a one-off anecdote.

Runbook/spec:

- [Public Benchmarks specification (protocol + artifacts)](/docs/public-benchmarks)
- Suite runbook (repo path): `docs/benchmarks/suites/ml.md`

---

## Abstract

In scientific ML pipelines, a large fraction of wall-time is often **latency**, not FLOPs:

- import and runtime initialization,
- tracing / graph building,
- compilation,
- kernel loading / caching,
- and the first real execution on real-shaped inputs.

So the benchmark we actually need is not “examples/sec at steady state”. It is a *two-regime* measurement:

1. **Time-to-first-result (TTFR)** in a fresh process (cold start).
2. **Warm throughput** once compilation and caches are populated.

To make that publishable, we treat each run as a snapshot with raw distributions, a pinned environment, and an explicit cache policy.

---

## 1) Two regimes, two metrics

### Regime A: cold-start / time-to-first-result

This includes:

- import time
- graph tracing
- compilation
- first execution

This is the metric that matters for short runs.

### Regime B: warm throughput

This measures steady-state execution when:

- compilation caches are populated,
- kernels are loaded,
- and the process is already running.

This matters for long runs.

Publishing one number without specifying the regime is not meaningful.

---

## 2) Definitions: what we time

For publishable runs we report *component* timings, not just a single aggregate:

- `t_import`: importing the runtime stack (best-effort proxy for “startup cost”)
- `t_first_call`: first call that triggers tracing + compilation + first execution
- `t_second_call`: second call on the same shapes (warm execution proxy)
- `t_steady_state`: distribution over repeated warm calls (with a declared sync policy)

For GPU-backed runtimes, the benchmark must explicitly synchronize (or “block until ready”) to avoid timing only CPU dispatch.

---

## 3) Benchmark protocol (what must be specified)

To make the result interpretable, the harness must specify:

- whether the process is fresh (new process) or persistent
- cache state (clean cache vs warmed cache)
- dataset sizes and data layout (to avoid accidentally benchmarking input conversion)
- what is included/excluded from the measurement window

For cold-start benchmarks, the only honest baseline is a fresh process with a declared cache policy.

---

## 4) Cache policy: “cold” has multiple meanings

Compile-latency results are extremely sensitive to caching.

So instead of pretending there is one “cold start”, we publish explicit modes:

- **cold process, warm cache**: new Python process, but persistent compilation cache allowed
- **cold process, cold cache**: new Python process, and compilation cache directory is empty (when feasible)
- **warm process**: same long-lived process (typical for interactive analysis)

If we can’t reliably clear a cache (because the runtime stores it outside our control), we treat that as a constraint and publish the limitation.

---

## 5) What we publish (artifacts)

For each snapshot:

- cold-start distributions (not just a single timing)
- warm-throughput distributions
- baseline manifest (versions, hardware, settings)
- cache policy and harness version

Published artifact contracts (ML suite):

- per-case results: `nextstat.ml_benchmark_result.v1`
- suite index: `nextstat.ml_benchmark_suite_result.v1`

Publishing contract: [Public Benchmarks](/docs/public-benchmarks). Validation pack artifact: [Validation Report](/docs/validation-report).

---

## 6) Why this is part of the NextStat benchmark program

NextStat’s core value proposition is not “wins a microbenchmark”.

It’s that entire scientific pipelines become:

- faster,
- more reproducible,
- and easier to audit.

Compile-vs-execution tradeoffs are part of that story when ML is inside the loop.

## Appendix: seed harness status (today)

The public benchmarks seed repo includes a runnable ML suite under `benchmarks/nextstat-public-benchmarks/suites/ml/`:

- it measures cold-start TTFR (import + first call) using multiple **fresh processes**
- it measures warm-call throughput as a per-call distribution
- it runs with NumPy by default and includes optional `jax_jit_*` cases:
  - `warn` with `reason="missing_dependency: jax"` if JAX is not installed
  - GPU-intended cases report `warn` with `reason="gpu_unavailable"` when a CUDA backend is not available

Reproducible seed run (suite runner writes per-case JSON + suite index):

```bash
python benchmarks/nextstat-public-benchmarks/suites/ml/suite.py \
  --deterministic \
  --out-dir benchmarks/nextstat-public-benchmarks/out/ml
```

Published artifact schemas (seed harness writes `schema_version` fields):

- per-case: `nextstat.ml_benchmark_result.v1`
- suite index: `nextstat.ml_benchmark_suite_result.v1`
