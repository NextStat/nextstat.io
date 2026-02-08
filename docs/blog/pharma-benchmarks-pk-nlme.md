---
title: "Pharma Benchmarks: PK and NLME Without Benchmark Theater"
slug: pharma-benchmarks-pk-nlme
description: "A rigorous plan for benchmarking PK/NLME workflows in NextStat: objective definitions, stopping rules, scaling protocols, correctness gates, and publishable artifacts."
date: 2026-02-08
author: NextStat Team
status: draft
keywords:
  - pharmacometrics benchmarks
  - PK model benchmark
  - NLME benchmark
  - population PK
  - reproducible benchmarks
  - regulated validation
  - NextStat
category: pharma
---

<!--
  Blog draft (technical).
  Suite runbook: docs/benchmarks/suites/pharma.md
  Related tutorials:
    - docs/tutorials/phase-13-pk.md
    - docs/tutorials/phase-13-nlme.md
-->

# Pharma Benchmarks: PK and NLME Without Benchmark Theater

Pharmacometrics benchmarks are deceptively easy to do wrong.

Two fitters can both return “a result” while measuring fundamentally different things:

- different objectives (MAP vs marginal likelihood vs FOCE-style approximations),
- different stopping rules and tolerances,
- different parameterizations (log vs linear),
- different handling of censoring / LLOQ.

This post defines what we will benchmark in NextStat’s PK/NLME baselines and how we plan to make the numbers reproducible and meaningful.

Runbook/spec:

- [Pharma Benchmark Suite](/docs/benchmarks/suites/pharma)

---

## 1. What we benchmark

We benchmark workflows users actually run:

1. **NLL and gradient evaluation** (time/call)
2. **Fit wall-time under an explicit protocol**
3. **Scaling laws** with subject count and observations per subject

We publish both micro-level metrics (time per eval) and end-to-end metrics (fit time), because they answer different questions:

- time/eval tells you the ceiling,
- fit time tells you the user experience.

---

## 2. The core pitfall: “time to convergence” is not a well-defined metric

Optimization time depends on:

- stopping criteria (tolerance on projected gradient, ΔNLL stability, max-iter),
- bound constraints and how they’re handled,
- line search and step-size policies,
- parameterization and scaling.

So any benchmark that reports “fit time” without a protocol is not evidence.

Our rule: publish at least one of:

- **fixed-iteration protocols** (e.g., N steps of a declared optimizer), and/or
- **convergence protocols** with explicit tolerances + caps,

and always publish:

- objective/gradient evaluation counts (when measurable),
- final objective value (NLL at optimum),
- and a recovery/sanity check.

---

## 3. Baseline models in NextStat (what “PK/NLME” means here)

NextStat’s baseline models are intentionally minimal and explicit:

- Individual PK: 1-compartment oral model with first-order absorption  
  Tutorial: [Phase 13 PK baseline](/docs/tutorials/phase-13-pk)

- NLME baseline: population parameters + independent log-normal random effects (diagonal Omega), joint MAP fit  
  Tutorial: [Phase 13 NLME baseline](/docs/tutorials/phase-13-nlme)

This matters because “NLME” can mean many different approximations in production tools; benchmarks must compare like with like.

---

## 4. Dataset plan: synthetic first, open datasets when licensing permits

We will benchmark on two tiers:

1. **Synthetic** datasets generated from the model (for recovery + scaling).
2. **Open** datasets where redistribution and deterministic preprocessing are possible.

Every published run must include:

- dataset ID + hash,
- generation parameters (for synthetic),
- preprocessing protocol (for real data),
- and the exact model configuration (including LLOQ policy).

---

## 5. Metrics we will publish

At minimum:

- NLL time/call (and gradient time/call if applicable)
- fit wall-time under the declared protocol
- scaling curves:
  - subjects → runtime
  - observations/subject → runtime
  - random-effects dimension → runtime

And for sanity:

- recovery error on synthetic data (not a “speed” metric, but a trust gate)

---

## 6. Why this belongs in the public benchmark program

PK/NLME is exactly the kind of domain where:

- a “fast result” can be wrong or incomparable,
- and reproducibility is non-negotiable.

So we treat benchmarks as artifacts:

- pinned environments,
- correctness gates,
- raw result publishing,
- and external reruns when possible.

Public benchmark contract: [Public Benchmarks Specification](/docs/benchmarks/public-benchmarks).
