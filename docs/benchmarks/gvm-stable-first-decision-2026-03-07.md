# GVM Stable-First Promotion Decision

**Date**: 2026-03-07  
**Status**: Executed product decision  
**Scope**: first stable wave for scalar GVM measurement combinations

## Decision

The GVM surface has now been promoted beyond blanket `research-grade`
designation for a deliberately small first wave.

The first stable wave includes only the foundational inference surfaces:

### CLI

- `nextstat combine-measurements-build-spec`
- `nextstat combine-measurements`
- `nextstat combine-measurements-calibrate`
- `nextstat combine-measurements-calibrate-study`

### Python

- `nextstat.hep.build_measurement_combination_spec(...)`
- `nextstat.hep.build_measurement_combination_spec_from_manifest(...)`
- `nextstat.hep.combine_measurements(...)`
- `nextstat.hep.calibrate_measurements(...)`
- `nextstat.hep.calibrate_measurements_study(...)`

Everything else remains `research-grade` for now.

## Why this split

This is the highest-value, lowest-risk stable slice because:

1. it captures the core user value:
   fit, calibration, and repeated-seed stability
2. it keeps the public compatibility burden small
3. it avoids promoting the large reporting/parity pyramid in one shot
4. it matches the strongest existing evidence and test density

## What stays research-grade in the first wave

These surfaces remain explicitly research-grade after the first stable
promotion:

### CLI

- `nextstat combine-measurements-scenario-study`
- `nextstat combine-measurements-calibration-campaign`
- all solver-parity commands
- all cached post-processing / summarize / brief / family / matrix / portfolio
  commands

### Python

- `nextstat.hep.study_measurement_combination_scenarios(...)`
- `nextstat.hep.calibrate_measurement_combination_scenarios(...)`
- all solver-parity wrappers
- all digest / brief / family / matrix / portfolio builders and renderers

## Why the extended layers should wait

The wider reporting/parity pyramid is useful, but it has a much larger schema
and compatibility surface than the foundational inference contract.

Promoting all of it at once would:

- enlarge the stable API burden
- create more long-term maintenance cost
- add less product value than stabilizing the core fit/calibration path first

## Promotion criteria for the first wave

The first stable wave was promoted because all of the following were true:

- the support policy is accepted and published
- the readiness memo remains accurate
- benchmark and robustness snapshots are current
- the foundational subset is explicitly named in CLI and Python docs
- release notes state that broader reporting/parity surfaces remain
  research-grade

## Promotion criteria for a second wave

A later second wave may promote selected advanced surfaces, but only after:

- schema stability has been demonstrated over time
- there is clear user demand for stable guarantees on those layers
- the release burden of maintaining those artifacts as stable is justified

The most likely second-wave candidates are:

- `scenario-study`
- `calibration-campaign`

The least urgent candidates are:

- parity artifacts
- cached report builders
- brief/family/matrix/portfolio layers

## Bottom line

The adopted stable-first strategy is:

- stabilize the core inference path first
- keep advanced evidence/reporting layers research-grade
- expand only after the smaller stable contract proves durable

The operational form of that decision now lives in:

- [GVM Stable-First Support Matrix](/docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07.md)
- [GVM Stable-First Release Notes](/docs/benchmarks/gvm-stable-first-release-notes-2026-03-07.md)
