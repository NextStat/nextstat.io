# GVM Stable-First Release Notes

**Date**: 2026-03-07  
**Status**: Stable-first release note  
**Scope**: scalar HEP measurement combinations on the GVM engine

## Summary

The foundational GVM measurement-combination path is now promoted to `stable`.

This promotion applies to the core fit/calibration workflow only.
Advanced scenario, campaign, parity, and cached reporting layers remain
`research-grade`.

## Promoted to stable

### CLI

- `nextstat combine-measurements-build-spec`
- `nextstat combine-measurements`
- `nextstat combine-measurements-calibrate`
- `nextstat combine-measurements-calibrate-study`
- `make gvm-stable-first-example`

### Python

- `nextstat.hep.build_measurement_combination_spec(...)`
- `nextstat.hep.build_measurement_combination_spec_from_manifest(...)`
- `nextstat.hep.combine_measurements(...)`
- `nextstat.hep.calibrate_measurements(...)`
- `nextstat.hep.calibrate_measurements_study(...)`

## What remains research-grade

- `nextstat combine-measurements-scenario-study`
- `nextstat combine-measurements-calibration-campaign`
- solver-parity commands and wrappers
- cached summarize / brief / family / matrix / portfolio layers

## Contract details

The stable-first subset now has an explicit runtime stability contract:

- direct fit results return `stability: "stable"`
- calibration reports return `stability: "stable"`
- calibration study reports return `stability: "stable"`
- wider scenario/campaign/reporting artifacts continue to return
  `stability: "research-grade"`

Diagnostics for the stable subset also surface:

- `requested_solver`
- `effective_solver`

so `auto` fallback behavior is observable rather than implicit.

## Evidence behind the promotion

This promotion is backed by:

- [GVM Benchmark Snapshot](/docs/benchmarks/gvm-measurement-combine-snapshot-2026-03-07.md)
- [GVM NumericalPaper Robustness Snapshot](/docs/benchmarks/gvm-numerical-paper-robustness-snapshot-2026-03-07.md)
- [GVM Stable-Surface Readiness Memo](/docs/benchmarks/gvm-stable-surface-readiness-2026-03-07.md)
- [GVM Stable-Surface Support Policy](/docs/benchmarks/gvm-stable-surface-support-policy-2026-03-07.md)
- [GVM Stable-First Support Matrix](/docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07.md)

## Operating envelope

The current stable-first evidence envelope covers:

- literature-backed combinations around `15x25`
- synthetic stress tiers through `128x96`
- single-thread and Rayon scaling snapshots on Apple M5 and AMD EPYC 7502P

This is a supported evidence envelope, not an unrestricted scale claim.

## Verification gate

The promoted subset is now protected by a dedicated repo gate:

- script:
  [scripts/gvm/stable_first_gate.sh](/scripts/gvm/stable_first_gate.sh)
- workflow:
  [gvm-stable-first.yml](/.github/workflows/gvm-stable-first.yml)

Local run:

```bash
make gvm-stable-first-gate
```

For first external rollout, maintainers should use:

- `docs/guides/gvm-external-validation-kit.md`
- `docs/guides/gvm-external-validator-outreach-pack.md`
- `docs/guides/gvm-external-validation-tracker-template.md`
- `docs/examples/gvm-stable-first/external-validator-invite-template.md`
- `docs/examples/gvm-stable-first/external-validation-report-template.md`

For release execution and launch sequencing, use:

- `docs/benchmarks/gvm-stable-first-release-candidate-v0.10.0-2026-03-08.md`
- `docs/benchmarks/gvm-stable-first-release-pr-checklist-2026-03-07.md`
- `docs/benchmarks/gvm-stable-first-launch-checklist-2026-03-07.md`

## Upgrade notes

For users of the promoted subset:

- no new input schema is required
- a stable tabular ingress helper is available if the source of truth is CSV/TSV
- a stable manifest wrapper is available when users want one short bundle file instead of repeating long table arguments
- no CLI rename is required
- no Python namespace move is required
- the main visible contract change is the intentional `stable` designation on
  the promoted result/report payloads

For users of scenario/campaign/parity/reporting layers:

- nothing was removed
- those surfaces remain available
- they intentionally continue to be versioned as `research-grade`

## Bottom line

This release does not claim that the entire GVM stack is stable.
It claims something narrower and stronger:

- the foundational scalar measurement-combination path is now stable
- the wider research/reporting pyramid is still available, but remains
  research-grade
