# GVM Stable-First Release PR Checklist

**Date**: 2026-03-07  
**Status**: release hardening checklist  
**Scope**: stable-first scalar GVM promotion and release PR preparation

## Purpose

This checklist is the maintainer-side `PR-ready` gate for the stable-first GVM
subset.

Use it after implementation and docs are complete, but before cutting or
shipping a release.

## Release PR scope

This checklist applies only to the promoted stable-first subset:

- `nextstat combine-measurements-build-spec`
- `nextstat combine-measurements`
- `nextstat combine-measurements-calibrate`
- `nextstat combine-measurements-calibrate-study`
- matching `nextstat.hep.*` stable-first wrappers
- `make gvm-stable-first-example`

It does not promote:

- scenario-study
- calibration-campaign
- solver-parity
- cached brief/family/matrix/portfolio/reporting layers

Those remain research-grade.

## Pre-PR checklist

### Contract

- [ ] stable-first subset is named explicitly in:
  - `docs/references/cli.md`
  - `docs/references/python-api.md`
- [ ] stable-first vs research-grade boundary is called out explicitly
- [ ] `requested_solver` / `effective_solver` are documented for the promoted subset

### Evidence

- [ ] benchmark snapshot is present and current:
  - `docs/benchmarks/gvm-measurement-combine-snapshot-2026-03-07.md`
- [ ] robustness snapshot is present and current:
  - `docs/benchmarks/gvm-numerical-paper-robustness-snapshot-2026-03-07.md`
- [ ] readiness memo is present:
  - `docs/benchmarks/gvm-stable-surface-readiness-2026-03-07.md`
- [ ] support policy is present:
  - `docs/benchmarks/gvm-stable-surface-support-policy-2026-03-07.md`
- [ ] stable-first decision is present:
  - `docs/benchmarks/gvm-stable-first-decision-2026-03-07.md`
- [ ] support matrix is present:
  - `docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07.md`
- [ ] release notes are present:
  - `docs/benchmarks/gvm-stable-first-release-notes-2026-03-07.md`
- [ ] release candidate memo is present:
  - `docs/benchmarks/gvm-stable-first-release-candidate-v0.10.0-2026-03-08.md`

### Adoption path

- [ ] committed example bundle exists:
  - `docs/examples/gvm-stable-first/`
- [ ] one-command runner exists:
  - `scripts/gvm/run_stable_first_example.sh`
- [ ] user quickstart points at the committed bundle:
  - `docs/quickstarts/hep-gvm-stable-first.md`
- [ ] Route D exists in the adoption playbook:
  - `docs/guides/README.md`
  - `docs/guides/fixtures/route_d/`

### External validation pack

- [ ] maintainer external validation kit exists:
  - `docs/guides/gvm-external-validation-kit.md`
- [ ] outreach pack exists:
  - `docs/guides/gvm-external-validator-outreach-pack.md`
- [ ] tracker template exists:
  - `docs/guides/gvm-external-validation-tracker-template.md`
- [ ] invite template exists:
  - `docs/examples/gvm-stable-first/external-validator-invite-template.md`
- [ ] validator report template exists:
  - `docs/examples/gvm-stable-first/external-validation-report-template.md`

### Verification

- [ ] `cargo fmt --all --check`
- [ ] stable-first gate passes:

```bash
make gvm-stable-first-gate
```

### Release PR payload

- [ ] PR summary names the promoted stable-first subset explicitly
- [ ] PR summary names the remaining research-grade layers explicitly
- [ ] changelog includes the stable-first milestone
- [ ] target release version is pinned explicitly as `v0.10.0`
- [ ] docs links are updated from top-level indexes

## Recommended PR summary structure

Use a short structure:

1. what was promoted to stable
2. what remains research-grade
3. what evidence backs the promotion
4. how to run the golden path

## Exit condition

The release PR is ready only when every checkbox above is green and no open
`P0` adoption blocker remains from the external validation tracker.
