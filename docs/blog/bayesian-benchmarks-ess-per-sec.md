---
title: "Bayesian Benchmarks That Mean Something: ESS/sec vs Wall-Time"
slug: bayesian-benchmarks-ess-per-sec
description: "How NextStat benchmarks Bayesian inference rigorously: ESS/sec methodology, stable protocols, diagnostics settings, and publishable artifacts for comparisons vs Stan and PyMC."
date: 2026-02-08
author: NextStat Team
status: draft
keywords:
  - Bayesian benchmarks
  - ESS per second
  - NUTS benchmarks
  - Stan comparison
  - PyMC comparison
  - reproducible benchmarks
  - inference diagnostics
  - NextStat
category: bayesian
---

<!--
  Blog draft (technical).
  Suite runbook: docs/benchmarks/suites/bayesian.md
-->

# Bayesian Benchmarks That Mean Something: ESS/sec vs Wall-Time

Bayesian benchmarks are notorious because they collapse a multi-dimensional object (a posterior + a sampling algorithm) into a single number.

That can be useful — if you control the experiment.

This post explains how we benchmark Bayesian inference in NextStat using **ESS/sec** in a way that is reproducible and comparable across frameworks (Stan, PyMC), and what we publish so outsiders can rerun the claim.

Runbook/spec:

- [Bayesian Benchmark Suite](/docs/benchmarks/suites/bayesian)

---

## Abstract

We treat Bayesian performance numbers as scientific claims. That means:

- we define the *posterior* (model + priors + parameterization),
- we define the *inference protocol* (warmup, adaptation, mass matrix policy, stopping rules),
- we report **sampling efficiency** (ESS/sec, bulk + tail) *and* health (divergences, treedepth saturation, $\hat{R}$, E‑BFMI),
- and we publish artifacts (raw timings, manifests, and validation packs) so an outsider can rerun.

---

## 1) Why ESS/sec is the right first metric (and when it isn’t)

For gradient-based samplers, wall-time per iteration is not the whole story. What matters is:

- how quickly the sampler explores the posterior, and
- whether the samples have low autocorrelation.

Effective Sample Size (ESS) is a standard summary of that.

So ESS/sec captures both:

- computational cost, and
- sampling efficiency.

But ESS/sec only makes sense when:

- the **model** is the same,
- the **parameterization** is the same (centered vs non-centered can dominate),
- and diagnostics are computed consistently.

---

## 2) What we publish: ESS/sec + “health” metrics

ESS/sec alone can be gamed by pathological runs (e.g., a chain that “moves” fast but is invalid).

So we publish, at minimum:

- bulk ESS/sec and tail ESS/sec (per parameter group),
- divergence rate,
- max treedepth saturation rate,
- max $\hat{R}$ across parameters,
- minimum ESS (bulk + tail) across parameters,
- minimum E‑BFMI across chains.

Those are not “extra diagnostics”; they are part of what it means for a run to count.

---

## 3) The benchmarking protocol (what must be pinned)

To avoid benchmark theater, we pin:

- model definition + priors
- parameterization (explicitly)
- warmup length and sampling length
- adaptation settings:
  - target acceptance
  - step-size adaptation policy
  - mass matrix policy (diag vs dense) + update schedule
- RNG seeding policy (and determinism policy)

If any of these differ, we don’t call it a comparison — we call it a different experiment.

---

## 4) Correctness gates: performance numbers must be “allowed to exist”

We separate **posterior correctness** from **performance** and publish both.

### 4.1 NUTS quality smoke suite (fast)

The Apex2 “NUTS quality” runner exists specifically to catch catastrophic regressions in:

- posterior transform plumbing (bounded/unbounded parameters),
- HMC/NUTS stability (finite energies, low divergence/treedepth saturation),
- diagnostics plumbing ($\hat{R}$/ESS/E‑BFMI present and finite).

With `--deterministic` it produces a deterministic JSON artifact:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_nuts_quality_report.py \
  --deterministic \
  --out tmp/apex2_nuts_quality_report.json
```

The suite includes small, interpretable cases (e.g., a Gaussian mean, a strong-prior posterior sanity, and a Neal’s funnel stress case).

### 4.2 Simulation-Based Calibration (SBC) (slow, stronger evidence)

SBC validates the *posterior* more directly: for synthetic datasets generated from the prior, posterior ranks should be uniform.

The Apex2 SBC runner is intentionally “nightly/slow”:

```bash
NS_RUN_SLOW=1 NS_SBC_RUNS=20 NS_SBC_WARMUP=200 NS_SBC_SAMPLES=200 \
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_sbc_report.py \
    --deterministic \
    --out tmp/apex2_sbc_report.json
```

This is not a performance benchmark — it’s a correctness gate that makes performance comparisons meaningful.

---

## 5) What we publish (artifacts)

For each benchmark snapshot:

- the baseline manifest (versions, hardware, settings)
- raw timing measurements
- ESS metrics (bulk + tail) per parameter group
- key diagnostics (e.g., divergences, tree depth, step size)

If a run has pathologies (divergences, failure to adapt), we publish that as a result, not as a footnote.

Publishing spec: [Publishing Benchmarks](/docs/benchmarks/publishing).

---

## 6) Known pitfalls (we will document them explicitly)

### A) Parameterization dominates

The same model in centered vs non-centered form can differ by orders of magnitude in ESS/sec.

So we treat parameterization as part of the benchmark input, not as “implementation detail”.

### B) Different defaults are different experiments

Stan, PyMC, and custom implementations have different adaptation defaults.

Benchmarks must either align these settings or publish them and interpret differences as part of the outcome.

### C) ESS computation must be consistent

ESS depends on the method and version of the diagnostics tool. We will publish the exact computation policy and version.

---

## 7) How to run a local microbenchmark (today)

For a Rust-only microbenchmark of NUTS wall-time under a pinned `NutsConfig`, we ship a Criterion bench:

```bash
cargo bench -p ns-inference --bench nuts_benchmark
```

This covers a small Normal mean model and a tiny HistFactory model fixture; it is useful for regression detection, not for cross-framework “wins”.

---

## 8) What you should take away

When we publish Bayesian benchmark numbers, the intent is that you can answer:

- “what exact posterior was sampled?”
- “under what exact settings?”
- “how healthy were the diagnostics?”

and rerun the same harness on your hardware.

That’s how a Bayesian benchmark becomes evidence rather than a chart.
