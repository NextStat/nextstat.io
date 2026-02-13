---
title: "Building a Trustworthy HEP Benchmark Harness (pyhf + ROOT/RooFit)"
slug: hep-benchmark-harness
description: "How NextStat benchmarks HistFactory inference engines without benchmark theater: correctness gates, optimizer convergence checks, warm-start policies, pinned environments, and auditable artifacts."
date: 2026-02-08
author: NextStat Team
status: draft
keywords:
  - HistFactory benchmark
  - pyhf comparison
  - ROOT RooFit benchmark
  - profile likelihood scan
  - correctness gates
  - reproducible benchmarks
  - NextStat
category: hep
---

<!--
  Blog draft (technical).
  Canonical suite runbook: docs/benchmarks/suites/hep.md
  Canonical public benchmark spec: docs/benchmarks/public-benchmarks.md
-->

# Building a Trustworthy HEP Benchmark Harness (pyhf + ROOT/RooFit)

**Trust Offensive series:** [Index](/blog/trust-offensive) · **Prev:** [Third-Party Replication: Signed Reports](/blog/third-party-replication) · **Next:** [Numerical Accuracy](/blog/numerical-accuracy)

Benchmarks are easy to get wrong even when nobody is trying to cheat.

In HEP, the most dangerous failure mode is simple:

> you can “win” by benchmarking the wrong inference.

If two implementations disagree on the likelihood model (interpolation codes, constraints, masking, parameterization), then timing comparisons are meaningless. You are measuring different computations.

This post explains how we build a HEP benchmark harness that treats performance like a scientific claim: **correctness gates first**, explicit protocols, pinned environments, and artifacts that other people can rerun.

If you want the runbook/spec rather than the narrative, start with:

- `docs/benchmarks/suites/hep.md` (suite definition, in-repo)
- [/docs/public-benchmarks](/docs/public-benchmarks) (global benchmark program contract)

---

## Abstract

We benchmark workflow-level HistFactory inference tasks (NLL, gradients, fits, scans, toys) across:

- **pyhf** as the primary reference implementation, and
- (selectively) **ROOT/RooFit/RooStats** when the workflow can be automated reproducibly.

The harness is designed so that speed numbers are only emitted after correctness gates pass, and so that each benchmark run produces auditable artifacts (raw distributions, manifests, validation packs).

---

## 1) Threat model: how HEP benchmarks lie

The point of the harness is not to produce impressive numbers. It is to prevent these common failure modes.

### 1.1 Model mismatch (not the same likelihood)

Two implementations can disagree “silently” if you don’t pin model conventions:

- interpolation codes (`code4` vs alternatives),
- constraints and priors,
- masking conventions when $n_i=0$ or $\nu_i\to 0$,
- parameter naming/order (especially per-bin modifiers),
- one-sided vs two-sided conventions for test statistics.

**If the model is not identical, timing comparisons do not mean anything.**

### 1.2 Optimizer mismatch (“faster” because it stopped early)

For fits and scans, you can appear faster by:

- using a looser tolerance,
- hitting bounds and calling it convergence,
- returning a suboptimal point that still “looks okay” in a table.

So the harness treats **convergence metadata** and **cross-evaluation** as first-class outputs, not optional logs.

### 1.3 Warm-start mismatch (especially for profile scans)

A profile scan benchmark is mostly a benchmark of your warm-start policy:

- cold-start each $\mu$ point from a fixed init, vs
- warm-start from the previous scan point (the standard analyst workflow).

If you don’t publish the policy, you don’t have a benchmark.

### 1.4 Environment drift (“works on my machine” performance)

Benchmarks move when any of these move:

- compiler/toolchain versions,
- BLAS/GPU stack,
- Python dependency versions,
- CPU/GPU model and throttling behavior.

So every publishable run captures an environment manifest and dataset hashes.

### 1.5 Reporting bias (single-number theater)

Single numbers hide variance and measurement choices. A trustworthy run publishes:

- raw per-repeat timings,
- an explicit aggregation policy,
- the inputs and settings used.

---

## 2) What we benchmark (workflow-first)

We benchmark end-to-end HEP workflows that dominate wall-time:

1. **NLL evaluation** at a fixed parameter point (core building block)
2. **Gradients** (required for optimizers and HMC/NUTS-style methods)
3. **MLE fits** (unconditional and conditional)
4. **Profile likelihood scans** (warm-start policy must be explicit)
5. **Toy ensembles** (toys/sec, scaling with parameter count)

Microbenchmarks still matter, but they are not the headline. The headline is what an analyst actually runs.

---

## 3) Correctness gates: fail fast before timing

Every suite run must include correctness gating *before* it prints timings.

For example, our current pyhf-vs-NextStat harness (in-repo):

- loads a pyhf workspace,
- builds the pyhf model with explicit interpolation settings (`code4`, `code4p`),
- maps parameters by **name** into NextStat’s ordering,
- evaluates NLL in both implementations, and
- **fails fast** if the NLLs disagree beyond tolerance.

Only after that does it print performance numbers.

Reference script: `tests/benchmark_pyhf_vs_nextstat.py`.

Concrete gate (today): for each case we require `abs(NLL_nextstat − NLL_pyhf) ≤ 1e-8` **or** `rel_diff ≤ 1e-12` before we record timings.

Why this matters:

- It prevents “fast because wrong” regressions from being published.
- It turns disagreement into a bug report with context, not an argument.

For public snapshots, the minimal rerunnable harness lives in the public benchmarks repo skeleton:

- `benchmarks/nextstat-public-benchmarks/suites/hep/run.py` (single case)
- `benchmarks/nextstat-public-benchmarks/suites/hep/suite.py` (multi-case + scaling)

These scripts produce machine-readable results under `manifests/schema/` and include:

- parity deltas with explicit tolerances,
- raw timing distributions (not just an aggregate),
- dataset SHA-256 hashes.

---

## 4) Avoiding apples-to-oranges: mapping and pinned conventions

### 4.1 Parameter mapping by name

Different implementations may order parameters differently (especially with per-bin modifiers like ShapeSys gammas).

Our harness maps parameter vectors by **parameter name**, not by index, before comparing NLL or timing fits.

### 4.2 Explicit interpolation codes

HistFactory has multiple interpolation conventions. Benchmarks must pin:

- NormSys interpolation code (e.g., `code4`)
- HistoSys interpolation code (e.g., `code4p`)

Otherwise you’re not benchmarking the same statistical model.

---

## 5) Timing protocol: publish raw distributions, not only a single number

Our policy is:

- warm up to avoid one-time allocation dominating timings,
- measure a stable window (calibrated `timeit` loop count),
- publish raw per-repeat timings, and
- state the aggregation policy explicitly (e.g., `min` or `median`).

We also distinguish:

- “end-to-end from Python” timings (what a user experiences), vs
- “core-only” timings (Criterion benches / Rust-only) when needed.

---

## 6) Fit benchmarking: convergence is part of the result

Fit benchmarks are meaningless if they don’t publish *how* the fit ended.

So our harness records convergence metadata from NextStat fits:

- `success` / `converged`,
- `n_iter` and `n_evaluations`,
- `termination_reason`,
- best-fit NLL and POI hat estimates (when available).

### 6.1 Cross-evaluation: separating model mismatch from optimizer mismatch

When two optimizers land at different parameter vectors, there are two possibilities:

1. **Model mismatch**: they are minimizing different functions.
2. **Optimizer mismatch**: they are minimizing the same function but converged differently.

To separate these, the harness evaluates each implementation’s NLL at the other’s best-fit point:

- $\mathrm{NLL}_\text{NextStat}(\hat\theta_\text{pyhf})$ and
- $\mathrm{NLL}_\text{pyhf}(\hat\theta_\text{NextStat})$,

with parameter vectors mapped by **name**.

If cross-evaluation deltas are large, it’s usually a model/settings mismatch (not “optimizer luck”).

This is implemented in `benchmarks/nextstat-public-benchmarks/suites/hep/run.py` under the `fit.parity` block.

---

## 7) Profile scans: cold-start vs warm-start is the whole story

Profile scans are a classic place where naive benchmarks lie.

Two implementations can both be “correct” and still measure different workflows if:

- one cold-starts every scan point from an initial parameter vector, while
- the other warm-starts from the previous scan point.

Warm-start is not a trick; it is the standard analyst workflow because it:

- reduces optimizer iterations, and
- improves reliability at tail points (where local minima are more likely).

So the harness must publish:

- the POI grid,
- the warm-start policy,
- bounds and tolerances,
- and any clipping conventions (one-sided $q_\mu$/$q_0$).

---

## 8) ROOT/RooFit comparisons: what we will (and won’t) claim

ROOT/RooFit/RooStats is often treated as “the baseline”.

In practice, it mixes multiple moving parts:

- model construction and interpolation specifics (`hist2workspace`)
- optimizer behavior (Minuit2, strategy settings, boundary handling)
- fit status codes and failure handling

So for ROOT comparisons, we will:

- publish failure modes and fit-status rates, not only averages,
- publish cross-evaluation checks (evaluate NLL from implementation A at params from B),
- keep “did it converge?” as a first-class metric.

The current in-repo validation harness is `tests/validate_root_profile_scan.py`. It:

- stages a HistFactory export into a clean run directory,
- runs `hist2workspace`,
- runs a ROOT macro to generate a profile scan artifact,
- runs the same scan in NextStat, and
- (optionally) evaluates NextStat’s NLL at ROOT’s fitted parameter points to diagnose mismatches.

We already have a rigorous numerical comparison write-up as a separate blog draft:

- [Numerical Accuracy](/blog/numerical-accuracy)

Benchmarks and numerical accuracy are two sides of the same trust problem.

---

## 9) From harness runs to public snapshots

Performance results should include:

- raw per-repeat measurements (so variance is visible),
- aggregation policy (median, min-of-N, etc.),
- environment manifest (toolchains, deps, hardware),
- and correctness reports used as gates.

For evidence-grade publication, snapshots also include a unified validation pack and a machine-readable inventory:

- `validation_report.json` (schema `validation_report_v1`)
- `snapshot_index.json` (schema `nextstat.snapshot_index.v1`)

This is the difference between a screenshot and an artifact.

Canonical specification (protocol + artifacts): [/docs/public-benchmarks](/docs/public-benchmarks).

Also see: [Benchmark Snapshots as Products](/blog/benchmark-snapshots-ci-artifacts).

---

## 10) How to rerun (today)

End-to-end pyhf vs NextStat sanity + timing (includes a correctness gate):

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/benchmark_pyhf_vs_nextstat.py --fit
```

Minimal rerunnable harness (public benchmarks skeleton):

```bash
python benchmarks/nextstat-public-benchmarks/suites/hep/run.py --deterministic --out benchmarks/nextstat-public-benchmarks/out/hep_simple_nll.json
```

ROOT profile-scan validation (requires ROOT + `hist2workspace`):

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/validate_root_profile_scan.py \
  --pyhf-json tests/fixtures/simple_workspace.json \
  --measurement GaussExample \
  --start 0.0 --stop 5.0 --points 51
```

## Closing: rerun me, don’t trust me

Our end state is not “we have good numbers”.

Our end state is:

- you can rerun the harness on your machine,
- see the same correctness gates,
- and compare results with full context.

That’s how performance becomes evidence.
