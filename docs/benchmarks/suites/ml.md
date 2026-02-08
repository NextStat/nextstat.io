---
title: "Benchmark Suite: ML Infrastructure (Compile vs Execution)"
description: "ML infrastructure benchmark suite for NextStat: cold-start compilation latency vs steady-state throughput, differentiable HistFactory pipeline benchmarks, and memory footprint measurement for scientific ML workflows."
status: draft
last_updated: 2026-02-08
keywords:
  - ML benchmark latency
  - JAX compile benchmark
  - differentiable HistFactory
  - PyTorch inference pipeline
  - scientific ML performance
  - SignificanceLoss benchmark
  - differentiable analysis
  - NextStat ML
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

## Differentiable pipeline benchmarks (shipped)

NextStat's differentiable HistFactory layer enables end-to-end gradient flow from signal histograms through profiled q₀ to PyTorch optimizers. Relevant benchmarks:

| Metric | Value (CUDA, tHu 184p) | Value (CUDA, complex 8p) |
|---|---|---|
| NLL + signal gradient | 3.66 ms | 0.12 ms |
| Profiled q₀ | — | 3.0 ms |
| 20-step training loop | 2.4 ms/step | — |
| Signal gradient accuracy vs FD | 2.07e-9 max error | — |

These benchmarks are in `docs/benchmarks.md` under "Differentiable Layer" and "Neural Network Training".

The `SignificanceLoss` class and `SoftHistogram` enable direct optimization of discovery significance in PyTorch training loops. Benchmark coverage includes:

- Forward pass cost (NLL evaluation via Rust core)
- Backward pass cost (gradient via CUDA zero-copy or CPU tape)
- End-to-end training step wall-time

## Notes

This suite is primarily meant to support downstream benchmarks where ML is part of a scientific inference loop (e.g., differentiable pipelines). We will publish it alongside the public benchmark snapshots rather than as standalone "JAX vs X" marketing numbers.

## Related reading

- [Public Benchmarks Specification](/docs/benchmarks/public-benchmarks) — canonical spec.
- [Differentiable Layer blog post](/blog/differentiable-layer-nextstat-pytorch) — technical deep-dive.
- [ML Overview](/docs/ml-overview) — terminology bridge and comparison table.
- [ML Training Tutorial](/docs/ml-training) — end-to-end SignificanceLoss pipeline.
- [Validation Report Artifacts](/docs/references/validation-report) — validation pack for published snapshots.
