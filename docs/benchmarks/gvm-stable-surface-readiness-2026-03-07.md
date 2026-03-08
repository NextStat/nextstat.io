# GVM Stable-Surface Readiness Memo

**Date**: 2026-03-07  
**Status**: Release-hardening memo  
**Scope**: scalar HEP measurement combinations built on the GVM engine

## Purpose

This memo answers a narrower question than the benchmark and robustness
snapshots:

- not "does the implementation exist?"
- not "is it fast on representative fixtures?"
- but "what part of the current GVM surface is strong enough to treat as a
  stable product surface, and what still needs hardening before broader
  promotion?"

The answer today is:

- **algorithmically** the engine is close to paper-complete
- **operationally** the engine now has strong evidence
- **product-wise** the foundational subset has now been promoted to `stable`,
  while the broader reporting/parity pyramid remains `research-grade`

## Current evidence base

The current release-hardening decision should be read together with:

- [GVM Benchmark Snapshot](/docs/benchmarks/gvm-measurement-combine-snapshot-2026-03-07.md)
- [GVM NumericalPaper Robustness Snapshot](/docs/benchmarks/gvm-numerical-paper-robustness-snapshot-2026-03-07.md)
- [GVM Stable-Surface Support Policy](/docs/benchmarks/gvm-stable-surface-support-policy-2026-03-07.md)
- [GVM Stable-First Promotion Decision](/docs/benchmarks/gvm-stable-first-decision-2026-03-07.md)
- [GVM Stable-First Support Matrix](/docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07.md)
- [GVM Stable-First Release Notes](/docs/benchmarks/gvm-stable-first-release-notes-2026-03-07.md)
- [GVM Stable-First Release Candidate: v0.10.0](/docs/benchmarks/gvm-stable-first-release-candidate-v0.10.0-2026-03-08.md)
- [GVM Stable-First Release PR Checklist](/docs/benchmarks/gvm-stable-first-release-pr-checklist-2026-03-07.md)
- [GVM Stable-First Launch Checklist](/docs/benchmarks/gvm-stable-first-launch-checklist-2026-03-07.md)
- [HEP GVM Measurement Combinations](/docs/tutorials/hep-gvm-measurement-combinations.md)
- [Research-grade Measurement Combinations RFC](/docs/rfcs/research-grade-measurement-combinations.md)

The evidence already committed in the repository includes:

- paper-faithful `numerical-paper` reference inference in original correlated
  `theta_s^i` space
- `analytic-perturbative` O(ε²) path with Eq. `(29)/(60)` validity gate
- `auto` fallback from perturbative to paper-faithful numerical inference
- profile-likelihood confidence intervals
- Lawley/Bartlett diagnostics
- deterministic calibration / study / scenario / campaign surfaces
- solver parity artifacts
- published performance snapshots on Apple M5 and AMD EPYC 7502P
- multi-start robustness artifacts on mixed literature + synthetic families
  through `128x96`

## What is already strong

The following claims are now evidence-backed rather than aspirational:

1. The engine is not a prototype.
   It has a numerical reference path, an analytic fast path, a deterministic
   fallback contract, and a large fixture/test layer.

2. The direct solver contract is coherent.
   `requested_solver` and `effective_solver` are surfaced in diagnostics, and
   nested study/campaign reports preserve the same contract.

3. `NumericalPaper` has a real trust story.
   The repository now contains committed multi-start robustness artifacts for
   mixed literature-backed and synthetic low-`epsilon` families through
   `128x96`.

4. Published performance evidence exists.
   Single-thread and Rayon thread-scaling snapshots are committed for desktop
   and server hardware, including campaign workloads.

5. The practical workflow is already usable.
   The tutorial, CLI reference, and Python reference document the end-to-end
   path from one fit to calibration campaigns and cached parity post-processing.

## Stable subset

The strongest stable designation is **not** the whole GVM stack.
The deliberately promoted stable subset is:

| Surface | Status | Why |
| --- | --- | --- |
| `combine-measurements-build-spec` | stable | spreadsheet-friendly ingress now closes the main adoption gap for the stable path; the same command also supports a manifest wrapper for the shortest first-run path |
| `combine-measurements` | stable | core fit contract, diagnostics, solver fallback already evidence-backed |
| `combine-measurements-calibrate` | stable | deterministic toy-calibration contract is exercised heavily |
| `combine-measurements-calibrate-study` | stable | repeated-seed stability is evidence-backed and intentionally included in first wave |
| `combine-measurements-scenario-study` | research-grade | important workflow, but larger JSON schema and more derived reporting |
| `combine-measurements-calibration-campaign` | research-grade | useful advanced workflow, but heavier artifact semantics |
| Cached post-processing / parity / portfolio layers | keep research-grade for now | high utility, but much broader surface than the core inference contract |

The same split should apply to Python:

- stable core:
  `nextstat.hep.build_measurement_combination_spec(...)`,
  `nextstat.hep.build_measurement_combination_spec_from_manifest(...)`,
  `nextstat.hep.combine_measurements(...)`,
  `nextstat.hep.calibrate_measurements(...)`,
  `nextstat.hep.calibrate_measurements_study(...)`
- keep research-grade:
  solver-parity, campaign-family, portfolio, and cached cross-artifact
  reporting helpers

## Supported operating envelope backed by current evidence

Today the repository contains direct evidence for:

- published top-mass structure around `15x25`
- synthetic medium/large tiers:
  `32x24`, `64x48`, `96x64`, `128x96`
- low-`epsilon` multi-start trust evidence on `NumericalPaper`
- single-thread and multi-thread performance snapshots on:
  - Apple M5
  - AMD EPYC 7502P

This is enough to state a **supported evidence envelope**, but not yet enough to
promise arbitrary size-free scalability.

The honest contract today is:

- supported and evidenced:
  literature-scale combinations and synthetic stress tiers through `128x96`
- operationally practical:
  direct fits, calibration, and campaign workloads within the published
  snapshot ranges
- not yet evidenced enough for stable marketing claims:
  arbitrary larger tiers beyond `128x96`, or general GVM-style workflows
  outside scalar measurement combinations

## What still blocks broader stable promotion

The remaining blockers are mostly release-hardening, not new mathematics.

1. **Operational budgets**
   Performance and robustness snapshots exist, but we still do not have a short
   release acceptance table that says what runtime and drift budgets are
   required for future wider promotion.

2. **Surface reduction**
   The larger reporting pyramid is useful, but promoting all of it as stable in
   one shot would enlarge the compatibility burden for little product value.

## Recommendation

The correct next step is **not** another blind optimizer micro-tuning pass.
The correct next step is a release decision:

1. Keep the current stable core narrow:
   direct fit + calibration + repeated-seed study.
2. Keep the wider parity/reporting pyramid marked `research-grade`.
3. Use the support policy and robustness snapshots as the gate for any second
   promotion wave.

In short:

- **Research-grade engine**: already strong
- **Stable product surface**: now established for the first wave
- **Main remaining work**: broader promotion criteria and operational budgets,
  not more core solver invention

## Release checklist for stable promotion

The first stable wave is now promoted. Any future promotion from
`research-grade` to `stable` should satisfy all of the following:

- stable subset is named explicitly in CLI and Python docs
- support envelope is stated explicitly
- benchmark snapshot and robustness snapshot are both up to date
- direct fit + calibration + selected study surfaces are green in Rust, CLI, and
  Python
- no unresolved drift between `requested_solver` and `effective_solver`
- release notes state which advanced reporting/parity surfaces remain
  research-grade

## Bottom line

The current state is best described as:

> **A strong GVM engine with a stable foundational subset and published evidence,
> while the broader reporting/parity stack intentionally remains
> research-grade.**
