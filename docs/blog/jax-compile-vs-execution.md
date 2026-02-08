<!--
  Blog draft (technical).
  Suite runbook: docs/benchmarks/suites/ml.md
-->

# JAX Compile vs Execution: The Benchmark You Actually Need

**Last updated:** 2026-02-08  
**Status:** Blog draft (technical)

Many ML “benchmarks” measure only steady-state throughput.

That’s the right metric if you run a model for hours.

But in scientific pipelines, a lot of work happens in short loops:

- hyperparameter sweeps
- repeated small fits
- interactive analysis iterations
- short training runs for ablations

In those settings, **compile latency** can dominate the total cost.

This post explains how we plan to benchmark “compile vs execution” as a first-class measurement — and how we’ll publish it as a reproducible artifact instead of a one-off anecdote.

Runbook/spec:

- `docs/benchmarks/suites/ml.md`

---

## 1. Two regimes, two metrics

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

## 2. Benchmark protocol (what must be specified)

To make the result interpretable, the harness must specify:

- whether the process is fresh (new process) or persistent
- cache state (clean cache vs warmed cache)
- dataset sizes and data layout (to avoid accidentally benchmarking input conversion)
- what is included/excluded from the measurement window

For cold-start benchmarks, the only honest baseline is a fresh process with a declared cache policy.

---

## 3. What we will publish

For each snapshot:

- cold-start distributions (not just a single timing)
- warm-throughput distributions
- baseline manifest (versions, hardware, settings)
- cache policy and harness version

Publishing spec: `docs/benchmarks/publishing.md`.

---

## 4. Why this is part of the NextStat benchmark program

NextStat’s core value proposition is not “wins a microbenchmark”.

It’s that entire scientific pipelines become:

- faster,
- more reproducible,
- and easier to audit.

Compile-vs-execution tradeoffs are part of that story when ML is inside the loop.

