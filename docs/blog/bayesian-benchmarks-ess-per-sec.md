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

This post explains how we plan to benchmark Bayesian inference in NextStat using **ESS/sec** in a way that is reproducible and comparable across frameworks (Stan, PyMC), and what we will publish so that outsiders can rerun the claim.

Runbook/spec:

- [Bayesian Benchmark Suite](/docs/benchmarks/suites/bayesian)

---

## 1. Why ESS/sec is the right first metric (and when it isn’t)

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

## 2. The benchmarking protocol (what must be pinned)

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

## 3. What we will publish (artifacts)

For each benchmark snapshot:

- the baseline manifest (versions, hardware, settings)
- raw timing measurements
- ESS metrics (bulk + tail) per parameter group
- key diagnostics (e.g., divergences, tree depth, step size)

If a run has pathologies (divergences, failure to adapt), we publish that as a result, not as a footnote.

Publishing spec: [Publishing Benchmarks](/docs/benchmarks/publishing).

---

## 4. Known pitfalls (we will document them explicitly)

### A) Parameterization dominates

The same model in centered vs non-centered form can differ by orders of magnitude in ESS/sec.

So we treat parameterization as part of the benchmark input, not as “implementation detail”.

### B) Different defaults are different experiments

Stan, PyMC, and custom implementations have different adaptation defaults.

Benchmarks must either align these settings or publish them and interpret differences as part of the outcome.

### C) ESS computation must be consistent

ESS depends on the method and version of the diagnostics tool. We will publish the exact computation policy and version.

---

## 5. What you should take away

When we publish Bayesian benchmark numbers, the intent is that you can answer:

- “what exact posterior was sampled?”
- “under what exact settings?”
- “how healthy were the diagnostics?”

and rerun the same harness on your hardware.

That’s how a Bayesian benchmark becomes evidence rather than a chart.
