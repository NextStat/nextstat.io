<!--
  Blog draft (technical / trust-building).
  Canonical benchmark spec lives in: docs/benchmarks/public-benchmarks.md
-->

# Trust Offensive: Public Benchmarks

**Last updated:** 2026-02-08  
**Status:** Blog draft (technical)

Benchmarks are easy to get wrong — even when nobody is trying to cheat.

- Different inputs
- Different settings
- Different warmup behavior
- Different compilation modes
- Different “reference correctness” assumptions

In scientific software, that creates a trust gap: users can’t tell whether a performance claim is meaningful, stable, or reproducible.

So we’re doing a **trust offensive**: we will publish public benchmark snapshots that are designed like experiments — with protocols, pinned environments, correctness gates, and artifacts that others can rerun.

If you prefer the spec (what we measure, how we publish, what gets pinned), start here: `docs/benchmarks/public-benchmarks.md`.

---

## 1) What we’re benchmarking (and what we’re not)

We benchmark **end-to-end workflows** that real users run, not only micro-kernels:

- HEP / HistFactory: NLL evaluation, gradients, MLE fits, profile scans, toy ensembles
- Pharma: PK/NLME likelihood + fitting loops
- Bayesian: ESS/sec under well-defined inference settings
- ML infra: compile latency vs execution throughput where it dominates pipeline cost (e.g., JAX)

We will still keep microbenchmarks (Criterion) — but we treat them as *regression detectors*, not as the headline.

Non-goals:

- one-off “hero numbers”
- unpublished harness scripts
- performance without correctness gates

---

## 2) The hard part: making benchmarks trustworthy

If you’ve ever tried to reproduce someone else’s benchmark, you know the failure modes:

### A. The benchmark is “fast” because it’s not doing the same thing

For binned likelihood pipelines, a benchmark is meaningless if the implementation is not numerically consistent with a reference.

Our rule: before timing, the harness must validate correctness (within an explicit tolerance) for the exact inputs being benchmarked.

Example: our pyhf-vs-NextStat harness checks NLL agreement before it prints timings (`tests/benchmark_pyhf_vs_nextstat.py`).

### B. The benchmark is “fast” because it’s warmed up differently

JIT compilation, caching, GPU kernel loading, memory allocators, and Python import cost can dominate naive measurements.

Our rule: every benchmark must specify:

- warmup policy
- steady-state measurement window
- what’s included/excluded (compile time vs execution)

### C. The benchmark is not reproducible because the environment isn’t pinned

Scientific compute is extremely sensitive to:

- compiler versions and flags
- BLAS backends
- GPU drivers / runtimes
- Python version and dependency constraints

Our rule: every published snapshot includes an environment manifest (toolchains + lockfiles + hardware metadata).

### D. The benchmark is “fast” because it reports only one convenient statistic

Single numbers hide variance and can be gamed by noise.

Our rule: publish raw per-test measurements and the aggregation policy (e.g., median, min-of-N) explicitly.

---

## 3) What we will publish for each snapshot

Each public benchmark snapshot will include:

- raw results (per test, per repeat)
- summary tables
- a baseline manifest (code SHA, env versions, dataset hashes, run config)
- correctness/parity report used as gating

This is the difference between “trust me” and “rerun me”.

---

## 4) Why this matters: performance as a scientific claim

In research, we don’t accept “it worked on my machine” for results. We publish methods, assumptions, and checks.

Performance should be treated the same way — especially when performance changes what analyses are feasible:

- toy ensembles become practical at 10³–10⁵ scale
- profile scans become interactive
- ML training can optimize inference metrics directly rather than surrogates

If a benchmark can’t be reproduced, it’s not evidence. It’s an anecdote.

---

## 5) What’s next (the suites)

We’re organizing the benchmark program into suites:

- **HEP suite**: pyhf + (optionally) ROOT/RooFit harness with NLL parity gates
- **Pharma suite**: PK/NLME plan + datasets
- **Bayesian suite**: ESS/sec comparisons vs Stan + PyMC with fully specified inference settings
- **ML suite**: JAX compile vs execution comparisons (where compilation dominates)

Each suite will have:

1. a documentation page (how to run, what is measured, what gets gated),
2. public snapshot artifacts, and
3. a blog post interpreting the results and tradeoffs.

---

## 6) The ask: rerun it

Public benchmarks only work if other people rerun them.

When the first snapshot is published, the most valuable contribution you can make is:

- rerun the harness on your hardware,
- publish your manifest + results,
- and tell us what diverges (numbers, settings, correctness gates).

That is how “fast” becomes “trusted”.

