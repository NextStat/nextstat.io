---
title: "Trust Offensive: Why We Publish Reproducible Benchmarks for Scientific Software"
slug: trust-offensive-public-benchmarks
description: "NextStat's public benchmark program treats performance as a scientific claim — with protocols, pinned environments, correctness gates, and artifacts anyone can rerun. Learn why reproducible benchmarks matter for HEP, pharma, and Bayesian inference."
date: 2026-02-08
author: NextStat Team
status: shipped
keywords:
  - reproducible benchmarks
  - scientific software performance
  - statistical computing benchmarks
  - pyhf comparison
  - HistFactory benchmark
  - NLME benchmark
  - NUTS ESS per second
  - benchmark reproducibility
  - open source validation
  - NextStat benchmarks
category: trust
---

# Trust Offensive: Public Benchmarks

Benchmarks are easy to get wrong — even when nobody is trying to cheat.

- Different inputs
- Different settings
- Different warmup behavior
- Different compilation modes
- Different “reference correctness” assumptions

In scientific software, that creates a trust gap: users can’t tell whether a performance claim is meaningful, stable, or reproducible.

So we’re doing a **trust offensive**: we will publish public benchmark snapshots that are designed like experiments — with protocols, pinned environments, correctness gates, and artifacts that others can rerun.

If you prefer the spec (what we measure, how we publish, what gets pinned), start here: [Public Benchmarks Specification](/docs/benchmarks/public-benchmarks).

---

## Series (recommended reading order)

1. **Trust Offensive: Public Benchmarks** (this post) — the why + the trust model.
2. [The End of the Scripting Era](/blog/end-of-scripting-era-benchmarks) — why “rerunnable evidence” changes how scientific software is built.
3. [Benchmark Snapshots as Products](/blog/benchmark-snapshots-ci-artifacts) — CI artifacts, manifests, and baselines.
4. [Third-Party Replication: Signed Reports](/blog/third-party-replication-signed-report) — external reruns as the strongest trust signal.
5. [Building a Trustworthy HEP Benchmark Harness](/blog/hep-benchmark-harness) — methodology for HistFactory benchmarking.
6. [Numerical Accuracy](/blog/numerical-accuracy) — ROOT vs pyhf vs NextStat, with reproducible evidence.
7. [Differentiable HistFactory in PyTorch](/blog/differentiable-layer-nextstat-pytorch) — training NNs directly on $Z_0$.
8. [Bayesian Benchmarks: ESS/sec](/blog/bayesian-benchmarks-ess-per-sec) — how we make sampler comparisons meaningful.
9. [Pharma Benchmarks: PK/NLME](/blog/pharma-benchmarks-pk-nlme) — protocols for regulated-grade benchmarks.
10. [JAX Compile vs Execution](/blog/jax-compile-vs-execution) — the latency benchmark that matters in short loops.

## Companion docs (canonical runbooks)

Blog posts explain *why* and *what it means*. Docs explain *how to rerun it*.

Start here:

- [Public Benchmarks Specification](/docs/benchmarks/public-benchmarks) — what we measure, what we publish, what gets pinned.
- [First Public Benchmark Snapshot (Playbook)](/docs/benchmarks/first-public-snapshot) — the step-by-step “do this first” runbook.
- [Publishing Benchmarks](/docs/benchmarks/publishing) — CI artifacts, baseline manifests, DOI packaging, replication integration.
- [Third-Party Replication Runbook](/docs/benchmarks/replication) — external rerun + signed report template.

Suite runbooks:

- [HEP suite](/docs/benchmarks/suites/hep)
- [Pharma suite](/docs/benchmarks/suites/pharma)
- [Bayesian suite](/docs/benchmarks/suites/bayesian)
- [ML suite](/docs/benchmarks/suites/ml)

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

Example: our pyhf-vs-NextStat harness checks NLL agreement before it prints timings (`tests/benchmark_pyhf_vs_nextstat.py`). The [validation report system](/docs/references/validation-report) formalizes this: every published snapshot includes a `validation_report.json` with dataset SHA-256 hashes, model specs, and per-suite pass/fail gates.

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

## 5) The suites

The benchmark program is organized into vertical-specific suites:

- **[HEP suite](/docs/benchmarks/suites/hep)**: pyhf + ROOT/RooFit harness with NLL parity gates, GPU batch toy benchmarks (CUDA + Metal)
- **[Pharma suite](/docs/benchmarks/suites/pharma)**: PK/NLME likelihood + fitting loops with analytic reference baselines
- **[Bayesian suite](/docs/benchmarks/suites/bayesian)**: ESS/sec comparisons vs Stan + PyMC with fully specified inference settings, SBC calibration
- **[ML suite](/docs/benchmarks/suites/ml)**: compile vs execution latency, differentiable pipeline throughput
- **[Time Series suite](/docs/benchmarks/suites/timeseries)**: Kalman filter/smoother throughput, EM convergence cost
- **[Econometrics suite](/docs/benchmarks/suites/econometrics)**: Panel FE, DiD, IV/2SLS scaling with cluster count

Each suite has:

1. a runbook page (how to run, what is measured, what gets gated),
2. public snapshot artifacts (JSON + optional PDF via `nextstat validation-report`), and
3. a blog post interpreting the results and tradeoffs.

---

## 6) The ask: rerun it

Public benchmarks only work if other people rerun them.

When the first snapshot is published, the most valuable contribution you can make is:

- rerun the harness on your hardware,
- publish your manifest + results,
- and tell us what diverges (numbers, settings, correctness gates).

That is how "fast" becomes "trusted".

---

## Related reading

- [Public Benchmarks Specification](/docs/benchmarks/public-benchmarks) — the canonical spec for protocols, artifacts, and suite structure.
- [The End of the Scripting Era](/blog/end-of-scripting-era-benchmarks) — why reproducible benchmarking changes how we build scientific software.
- [Third-Party Replication: Signed Reports](/blog/third-party-replication-signed-report) — how external reruns + signed reports close the trust gap.
- [Publishing Benchmarks (CI, Artifacts, DOI)](/docs/benchmarks/publishing) — CI automation, DOI minting, and baseline management.
- [Validation Report Artifacts](/docs/references/validation-report) — the `validation_report.json` + PDF system that gates every published snapshot.
