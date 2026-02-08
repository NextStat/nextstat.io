---
title: "Benchmark Suite: ML Infrastructure (Compile vs Execution)"
status: draft
last_updated: 2026-02-08
---

# ML Benchmark Suite (Compile vs Execution)

Many scientific ML pipelines are bottlenecked not by FLOPs, but by *latency*:

- cold-start compilation (JIT)
- graph tracing
- kernel loading and caching
- data movement and conversions across runtimes

This suite measures compile vs execution tradeoffs in a way that is relevant to analysis pipelines (not just toy NN inference).

## What is compared

Planned comparisons include:

- JAX cold-start compile latency vs steady-state throughput
- end-to-end “analysis step time” where compilation is in the loop (e.g., short training runs, hyperparameter sweeps)

## What is measured

### 1) Cold-start latency

Time to first result, including:

- import + tracing
- compilation
- first execution

### 2) Warm throughput

Steady-state execution time after caches are populated.

### 3) Memory footprint (best-effort)

Where stable and measurable, report:

- peak device memory
- host memory allocations (if tracked)

## Protocol requirements

Every benchmark must specify:

- what counts as “compile” vs “execution”
- warmup policy and cache state (fresh process vs persistent)
- dataset sizes and data layout (to avoid “benchmarking the input pipeline” unintentionally)

## Notes

This suite is primarily meant to support downstream benchmarks where ML is part of a scientific inference loop (e.g., differentiable pipelines). We will publish it alongside the public benchmark snapshots rather than as standalone “JAX vs X” marketing numbers.

