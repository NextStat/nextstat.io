<!--
  Blog draft (technical).
  Canonical suite runbook: docs/benchmarks/suites/hep.md
  Canonical public benchmark spec: docs/benchmarks/public-benchmarks.md
-->

# Building a Trustworthy HEP Benchmark Harness (pyhf + ROOT/RooFit)

**Last updated:** 2026-02-08  
**Status:** Blog draft (technical)

HEP benchmarking has a unique failure mode:

> you can “win” by benchmarking the wrong inference.

If two implementations disagree on the likelihood model (interpolation codes, constraints, masking, parameterization), then timing comparisons are meaningless. The benchmark becomes a measurement of *different* computations.

This post explains how we’re building a benchmark harness that treats performance like a scientific claim: with correctness gates, pinned environments, and artifacts that other people can rerun.

If you want the runbook/spec rather than the narrative, start with:

- `docs/benchmarks/suites/hep.md` (suite definition)
- `docs/benchmarks/public-benchmarks.md` (global benchmark program contract)

---

## 1. What we benchmark (workflow-first)

We benchmark end-to-end HEP workflows that dominate wall-time:

1. **NLL evaluation** at a fixed parameter point (core building block)
2. **Gradients** (required for optimizers and HMC/NUTS-style methods)
3. **MLE fits** (unconditional and conditional)
4. **Profile likelihood scans** (warm-start policy must be explicit)
5. **Toy ensembles** (toys/sec, scaling with parameter count)

Microbenchmarks still matter, but they are not the headline. The headline is what an analyst actually runs.

---

## 2. The benchmark “sanity law”: correctness gates before timing

Every suite run must include correctness gating *before* it prints timings.

For example, our current pyhf-vs-NextStat harness:

- loads a pyhf workspace,
- builds the pyhf model with explicit interpolation settings (`code4`, `code4p`),
- maps parameters by **name** into NextStat’s ordering,
- evaluates NLL in both implementations, and
- **fails fast** if the NLLs disagree beyond tolerance.

Only after that does it print performance numbers.

Reference implementation (today, in-repo): `tests/benchmark_pyhf_vs_nextstat.py`.

Why this matters:

- It prevents “fast because wrong” regressions from being published.
- It turns disagreement into a bug report with context, not an argument.

---

## 3. Avoiding apples-to-oranges: parameter mapping and model settings

### 3.1 Parameter mapping by name

Different implementations may order parameters differently (especially with per-bin modifiers like ShapeSys gammas).

Our harness maps parameter vectors by **parameter name**, not by index, before comparing NLL or timing fits.

### 3.2 Explicit interpolation codes

HistFactory has multiple interpolation conventions. Benchmarks must pin:

- NormSys interpolation code (e.g., `code4`)
- HistoSys interpolation code (e.g., `code4p`)

Otherwise you’re not benchmarking the same statistical model.

---

## 4. Benchmarking profile scans: cold-start vs warm-start is the whole story

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
- and any clipping conventions (one-sided qμ/q0).

---

## 5. ROOT/RooFit comparisons: what we will (and won’t) claim

ROOT/RooFit/RooStats is often treated as “the baseline”.

In practice, it mixes multiple moving parts:

- model construction and interpolation specifics (`hist2workspace`)
- optimizer behavior (Minuit2, strategy settings, boundary handling)
- fit status codes and failure handling

So for ROOT comparisons, we will:

- publish failure modes and fit-status rates, not only averages,
- publish cross-evaluation checks (evaluate NLL from implementation A at params from B),
- keep “did it converge?” as a first-class metric.

We already have a rigorous numerical comparison write-up as a separate blog draft:

- `docs/blog/numerical-accuracy.md`

Benchmarks and numerical accuracy are two sides of the same trust problem.

---

## 6. Reporting: publish raw distributions, not only a single number

Performance results should include:

- raw per-repeat measurements (so variance is visible),
- aggregation policy (median, min-of-N, etc.),
- environment manifest (toolchains, deps, hardware),
- and correctness reports used as gates.

This is the difference between a screenshot and an artifact.

Publishing spec: `docs/benchmarks/publishing.md`.

---

## 7. The punchline: rerun me, don’t trust me

Our end state is not “we have good numbers”.

Our end state is:

- you can rerun the harness on your machine,
- see the same correctness gates,
- and compare results with full context.

That’s how performance becomes evidence.

