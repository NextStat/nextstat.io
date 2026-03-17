# GVM Stable-First Support Matrix

**Date**: 2026-03-07  
**Status**: Executed support matrix  
**Scope**: scalar HEP measurement combinations on the GVM engine

## Purpose

This document is the short operational matrix for the promoted GVM subset.

It answers one narrow question:

- what is `stable` now
- what remains `research-grade`
- what solver and evidence contract applies to each class

## Support classes

| Class | Meaning |
| --- | --- |
| `stable` | public compatibility promise for the named stable-first subset |
| `research-grade` | versioned and tested, but still evolving without stable-surface promise |

## Stable-first CLI matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `nextstat combine-measurements-build-spec` | `stable` | tabular ingress helper for the canonical JSON spec; also supports `--manifest` as the shortest stable-first wrapper |
| `nextstat combine-measurements` | `stable` | foundational scalar GVM fit |
| `nextstat combine-measurements-calibrate` | `stable` | toy-calibration companion |
| `nextstat combine-measurements-calibrate-study` | `stable` | repeated-seed calibration/stability companion |
| `make gvm-stable-first-example` | `stable` | one-command golden path for the committed example bundle; emits spec, fit, calibration, and study artifacts |
| `nextstat combine-measurements-scenario-study` | `stable` | scenario-study layer |
| `nextstat combine-measurements-calibration-campaign` | `stable` | calibration-campaign orchestration |
| solver-parity commands | `stable` | solver-parity evidence/reporting |
| summarize / brief / family / matrix / portfolio commands | `stable` | post-processing/reporting layer |

## Stable-first Python matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `nextstat.hep.build_measurement_combination_spec(...)` | `stable` | tabular ingress helper for the canonical JSON spec |
| `nextstat.hep.build_measurement_combination_spec_from_manifest(...)` | `stable` | manifest wrapper around the same stable tabular bundle |
| `nextstat.hep.combine_measurements(...)` | `stable` | foundational scalar GVM fit |
| `nextstat.hep.calibrate_measurements(...)` | `stable` | toy-calibration companion |
| `nextstat.hep.calibrate_measurements_study(...)` | `stable` | repeated-seed stability companion |
| `nextstat.hep.study_measurement_combination_scenarios(...)` | `stable` | scenario-study layer |
| `nextstat.hep.calibrate_measurement_combination_scenarios(...)` | `stable` | calibration-campaign orchestration |
| solver-parity wrappers | `stable` | solver-parity evidence/reporting |
| digest / brief / family / matrix / portfolio builders | `stable` | post-processing/reporting layer |

## Solver contract for the stable subset

The stable-first subset supports:

- `solver="auto"` as the default
- explicit `solver="numerical-paper"`
- explicit `solver="analytic-perturbative"`
- explicit `solver="numerical"`

For stable-first outputs:

- `requested_solver` is surfaced in diagnostics
- `effective_solver` is surfaced in diagnostics
- `auto` may resolve to `analytic-perturbative` or `numerical-paper`
  depending on validity/fallback

## Evidence-backed envelope

The promoted stable-first subset is backed by repository evidence for:

- literature-backed combinations around `15x25`
- synthetic stress tiers:
  - `32x24`
  - `64x48`
  - `96x64`
  - `128x96`
- single-thread and Rayon scaling snapshots on:
  - Apple M5
  - AMD EPYC 7502P
- `NumericalPaper` multi-start robustness on mixed literature + synthetic
  families

## Verification lane

The stable-first subset is enforced by a dedicated repo gate:

- script:
  [scripts/gvm/stable_first_gate.sh](/scripts/gvm/stable_first_gate.sh)
- workflow:
  [gvm-stable-first.yml](/.github/workflows/gvm-stable-first.yml)
- make target:
  `make gvm-stable-first-gate`

The gate verifies:

- Rust core measurement-combine suite
- CLI measurement-combine suite
- Python measurement-combine suite
- formatting
- presence of the required benchmark / robustness / policy artifacts
- presence of the stable-first example bundle and external-validation kit docs
- presence of the external-validator outreach pack and invite template
- presence of the external-validation tracker template

## Companion documents

- [GVM Benchmark Snapshot](/docs/benchmarks/gvm-measurement-combine-snapshot-2026-03-07.md)
- [GVM NumericalPaper Robustness Snapshot](/docs/benchmarks/gvm-numerical-paper-robustness-snapshot-2026-03-07.md)
- [GVM Stable-Surface Readiness Memo](/docs/benchmarks/gvm-stable-surface-readiness-2026-03-07.md)
- [GVM Stable-Surface Support Policy](/docs/benchmarks/gvm-stable-surface-support-policy-2026-03-07.md)
- [GVM Stable-First Promotion Decision](/docs/benchmarks/gvm-stable-first-decision-2026-03-07.md)
- [GVM Stable-First Release Candidate: v0.10.0](/docs/benchmarks/gvm-stable-first-release-candidate-v0.10.0-2026-03-08.md)
- [GVM Stable-First Release PR Checklist](/docs/benchmarks/gvm-stable-first-release-pr-checklist-2026-03-07.md)
- [GVM Stable-First Launch Checklist](/docs/benchmarks/gvm-stable-first-launch-checklist-2026-03-07.md)

## Bottom line

The stable product promise is now intentionally narrow:

- the full GVM surface pyramid is now `stable`: foundational fit/calibration/study,
  scenario study, calibration campaign, solver-parity, and reporting layers
- future stable expansion should update this matrix explicitly, not implicitly
