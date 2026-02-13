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

**Trust Offensive series:** [Index](/blog/trust-offensive) · **Prev:** [Bayesian Benchmarks: ESS/sec](/blog/bayesian-benchmarks-ess-per-sec) · **Next:** [JAX Compile vs Execution](/blog/jax-compile-vs-execution)

Pharmacometrics benchmarks are deceptively easy to do wrong.

Two fitters can both return “a result” while measuring fundamentally different things:

- different objectives (MAP vs marginal likelihood vs FOCE-style approximations),
- different stopping rules and tolerances,
- different parameterizations (log vs linear),
- different handling of censoring / LLOQ.

This post defines what we benchmark in NextStat’s PK/NLME baselines and how we make the numbers reproducible, comparable, and auditable.

Runbook/spec:

- [Public Benchmarks specification (protocol + artifacts)](/docs/public-benchmarks)
- Suite runbook (repo path): `docs/benchmarks/suites/pharma.md`

---

## Abstract

We treat performance as evidence, not a screenshot. For PK/NLME that means:

- defining the objective precisely (likelihood, constraints, censoring policy),
- defining the fit protocol (stopping rule, bounds, initialization),
- publishing correctness gates (analytic checks + recovery on synthetic data),
- publishing raw measurements + manifests so outsiders can rerun.

The goal is not “a fast fit”. The goal is **a reproducible experiment**.

---

## 1) Threat model: how pharma benchmarks lie

Pharma benchmarks go wrong for the same reason HEP benchmarks go wrong: small modeling differences can produce plausible-looking outputs with incomparable compute cost.

Common failure modes:

- **Objective mismatch**: MAP vs marginal likelihood vs FOCE/Laplace approximations.
- **Solver mismatch**: different ODE solvers, tolerances, or step controls (the “same” model is not the same computation).
- **Parameterization mismatch**: log-space vs linear, constrained vs unconstrained transforms.
- **Censoring policy mismatch**: LLOQ, censored likelihood vs imputation vs drop rules.
- **Convergence mismatch**: “converged” means different tolerances, different line search rules, different max-iter caps.
- **Dataset handling mismatch**: preprocessing drift, unit conventions, time grids.

If we can’t align these, we don’t call it a benchmark comparison — we call it two different experiments.

---

## 2) What we benchmark

We benchmark workflows users actually run:

1. **NLL and gradient evaluation** (time/call)
2. **Fit wall-time under an explicit protocol**
3. **Scaling laws** with subject count and observations per subject

We publish both micro-level metrics (time per eval) and end-to-end metrics (fit time), because they answer different questions:

- time/eval tells you the ceiling,
- fit time tells you the user experience.

---

## 3) Correctness gates: analytic checks + recovery before timing

We publish performance numbers only alongside correctness evidence.

### 3.1 Apex2 pharma reference suite (shipped)

The Apex2 pharma reference runner produces deterministic, machine-readable evidence for:

- PK analytic correctness (closed-form 1-compartment oral dosing vs `predict()`),
- PK fit recovery on deterministic synthetic data,
- NLME smoke sanity (finite NLL/grad + fit improves NLL on synthetic multi-subject data).

Run:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_pharma_reference_report.py \
  --deterministic \
  --out tmp/apex2_pharma_reference_report.json
```

This report is included in the Apex2 master report and therefore in the validation pack produced by `make validation-pack`.

---

## 4) The core pitfall: “time to convergence” is not a well-defined metric

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

## 5) Baseline models in NextStat (what “PK/NLME” means here)

NextStat’s baseline models are intentionally minimal and explicit:

- Individual PK: 1-compartment oral model with first-order absorption  
  Tutorial (repo path): `docs/tutorials/phase-13-pk.md`

- NLME baseline: population parameters + independent log-normal random effects (diagonal Omega), joint MAP fit  
  Tutorial (repo path): `docs/tutorials/phase-13-nlme.md`

This matters because “NLME” can mean many different approximations in production tools; benchmarks must compare like with like.

---

## 6) Dataset plan: synthetic first, open datasets when licensing permits

We will benchmark on two tiers:

1. **Synthetic** datasets generated from the model (for recovery + scaling).
2. **Open** datasets where redistribution and deterministic preprocessing are possible.

Every published run must include:

- dataset ID + hash,
- generation parameters (for synthetic),
- preprocessing protocol (for real data),
- and the exact model configuration (including LLOQ policy).

---

## 7) Seed harness (public benchmarks skeleton)

For public snapshots we ship a minimal rerunnable seed harness under:

- `benchmarks/nextstat-public-benchmarks/suites/pharma/`

It starts with portable synthetic dataset generation so outsiders can rerun without licensing friction.

Single-case run:

```bash
python benchmarks/nextstat-public-benchmarks/suites/pharma/run.py \
  --deterministic \
  --out benchmarks/nextstat-public-benchmarks/out/pharma_pk_1c_oral.json
```

Suite runner (multiple generated cases, writes per-case JSON + a suite index):

```bash
python benchmarks/nextstat-public-benchmarks/suites/pharma/suite.py \
  --deterministic \
  --out-dir benchmarks/nextstat-public-benchmarks/out/pharma
```

Each generated dataset carries a stable dataset ID (encoding key parameters) and a SHA‑256 hash of the dataset spec.

This seed is NextStat-only today. Baseline templates (e.g. nlmixr2, Torsten) are tracked separately and will be incorporated only once their environments are reproducibly pinned.

Published artifact contracts (Pharma suite):

- per-case results: `nextstat.pharma_benchmark_result.v1`
- suite index: `nextstat.pharma_benchmark_suite_result.v1`

Canonical snapshot publishing contract + artifact inventory: [Public Benchmarks](/docs/public-benchmarks). Validation pack artifact: [Validation Report](/docs/validation-report).

---

## 8) Metrics we will publish

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

## 9) Why this belongs in the public benchmark program

PK/NLME is exactly the kind of domain where:

- a “fast result” can be wrong or incomparable,
- and reproducibility is non-negotiable.

So we treat benchmarks as artifacts:

- pinned environments,
- correctness gates,
- raw result publishing,
- and external reruns when possible.

Public benchmark contract: [Public Benchmarks](/docs/public-benchmarks).
