# GVM Stable-First Release Candidate: v0.10.0

**Date**: 2026-03-08  
**Status**: release candidate memo  
**Scope**: stable-first scalar GVM promotion

## Recommended target version

`v0.10.0`

## Why `v0.10.0`

The current released version is `v0.9.9`, and the stable-first GVM milestone is
larger than a patch-level change:

- it promotes a new user-facing stable subset
- it adds stable CLI and Python ingress for tabular measurement-combination data
- it adds a committed golden path, adoption route, and release-hardening gate
- it changes the public product narrative from purely research-grade GVM to a
  stable-first core plus research-grade advanced layers

This is a narrow and deliberate minor-version milestone, not a blanket
"everything GVM is stable" release.

## Proposed release scope

Promote only the stable-first scalar GVM subset.

### Stable in `v0.10.0`

- `nextstat combine-measurements-build-spec`
- `nextstat combine-measurements`
- `nextstat combine-measurements-calibrate`
- `nextstat combine-measurements-calibrate-study`
- `nextstat.hep.build_measurement_combination_spec(...)`
- `nextstat.hep.build_measurement_combination_spec_from_manifest(...)`
- `nextstat.hep.combine_measurements(...)`
- `nextstat.hep.calibrate_measurements(...)`
- `nextstat.hep.calibrate_measurements_study(...)`
- `make gvm-stable-first-example`

### Still research-grade in `v0.10.0`

- `scenario-study`
- `calibration-campaign`
- solver-parity surfaces
- cached summarize / brief / family / matrix / portfolio / reporting layers

## Curated changelog subset for the release PR

Use the stable-first GVM items below as the release-facing subset from
`CHANGELOG.md`.

### Added

- **GVM stable-first verification lane** — dedicated gate script, workflow, and
  make target protect the promoted scalar GVM subset across Rust core, CLI,
  Python, formatting, and required evidence documents.
- **GVM stable-first support matrix and release notes** — the promoted subset
  now has explicit support, release, and operational documentation.
- **GVM stable-first tabular + manifest ingress** — users can build canonical
  measurement-combination specs directly from CSV/TSV tables or from a short
  manifest wrapper instead of hand-writing JSON.
- **GVM stable-first golden path runner** — one-command end-to-end example
  execution through the committed bundle.
- **Adoption playbook Route D for stable-first GVM** — external-validation smoke
  path and pinned fixtures for scalar measurement combinations.
- **GVM stable-first release execution bundle** — release PR and launch
  checklists for the promoted subset.
- **GVM stable-first release candidate memo** — pins the recommended milestone
  version, stable-only scope, and PR-ready summary for the first stable GVM
  wave.

### Changed

- **GVM stable-first surface formalized** — the foundational
  measurement-combination fit/calibration/study path is now promoted as stable,
  while scenario/campaign/parity/reporting layers remain research-grade.

## Proposed release PR title

`Release v0.10.0`

## Proposed release PR summary

### Summary

`v0.10.0` promotes the stable-first scalar GVM surface in NextStat.

The release stabilizes the foundational measurement-combination workflow for
HEP users:

- tabular/manifest ingress
- direct combination fit
- toy calibration
- calibration study
- committed golden path and adoption route

Advanced scenario, campaign, solver-parity, and reporting-pyramid layers remain
available but intentionally stay research-grade.

### Stable in this release

- `nextstat combine-measurements-build-spec`
- `nextstat combine-measurements`
- `nextstat combine-measurements-calibrate`
- `nextstat combine-measurements-calibrate-study`
- matching `nextstat.hep` wrappers
- `make gvm-stable-first-example`

### Research-grade layers kept out of stable scope

- `scenario-study`
- `calibration-campaign`
- solver-parity commands and wrappers
- cached post-processing/reporting layers

### Evidence behind the promotion

- benchmark snapshot:
  [gvm-measurement-combine-snapshot-2026-03-07.md](/docs/benchmarks/gvm-measurement-combine-snapshot-2026-03-07.md)
- robustness snapshot:
  [gvm-numerical-paper-robustness-snapshot-2026-03-07.md](/docs/benchmarks/gvm-numerical-paper-robustness-snapshot-2026-03-07.md)
- readiness memo:
  [gvm-stable-surface-readiness-2026-03-07.md](/docs/benchmarks/gvm-stable-surface-readiness-2026-03-07.md)
- support policy:
  [gvm-stable-surface-support-policy-2026-03-07.md](/docs/benchmarks/gvm-stable-surface-support-policy-2026-03-07.md)
- support matrix:
  [gvm-stable-first-support-matrix-2026-03-07.md](/docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07.md)
- stable-first gate:
  [stable_first_gate.sh](/scripts/gvm/stable_first_gate.sh)

### Golden path

```bash
make gvm-stable-first-example
make gvm-stable-first-gate
```

## Release-prep commands

Use this memo only after the current dirty worktree is resolved and the release
branch is intentionally prepared.

```bash
git describe --tags --abbrev=0
git status --short
make gvm-stable-first-gate
```

Then:

```bash
git checkout -b codex/release-v0.10.0
```

At that point, perform the actual version bump from `0.9.9` to `0.10.0` in:

- `Cargo.toml`
- `bindings/ns-py/pyproject.toml`
- `bindings/ns-cli-py/pyproject.toml`

and update `CHANGELOG.md` from `Unreleased` into the final `0.10.0` release
section.

## Exit condition

This release candidate is ready to convert into a real release PR only when:

- the stable-first gate is green
- the worktree is intentionally cleaned/prepared for release
- the `v0.10.0` version bump is applied consistently
- the PR uses the stable-only scope defined above
