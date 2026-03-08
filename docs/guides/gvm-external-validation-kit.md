---
title: "GVM External Validation Kit"
status: stable-first
---

# GVM External Validation Kit

Use this document when you want an external physicist or analysis contact to
validate the stable-first scalar GVM path on a real machine with either:

- the committed example bundle in this repo
- one small real measurement-combination case from their own analysis

This is a maintainer-facing handoff kit. It is not the first document a user
should read. The user-facing entry point stays:

- `docs/quickstarts/hep-gvm-stable-first.md`

Track the wave itself in:

- `docs/guides/gvm-external-validation-tracker-template.md`

## Goal

Collect one short, reproducible signal from an external user that answers four
questions:

1. can they run the stable-first golden path without maintainer intervention
2. can they map one small real table-based case into the canonical spec
3. do the solver diagnostics and outputs read clearly enough for first use
4. do they hit any packaging, docs, or terminology friction

## What to send

Send exactly these links:

- stable-first quickstart:
  `docs/quickstarts/hep-gvm-stable-first.md`
- committed example bundle:
  `docs/examples/gvm-stable-first/README.md`
- report template:
  `docs/examples/gvm-stable-first/external-validation-report-template.md`
- maintainer tracker:
  `docs/guides/gvm-external-validation-tracker-template.md`
- optional wider tutorial:
  `docs/tutorials/hep-gvm-measurement-combinations.md`

Do not send the full benchmark bundle or the research-grade campaign docs on
first contact. That only adds noise.

## Requested tasks for the external user

Ask them to do two passes.

### Pass 1: committed golden path

Have them run the committed stable-first example:

```bash
make gvm-stable-first-example
```

Expected outputs:

- `tmp/gvm-stable-first-example/spec.json`
- `tmp/gvm-stable-first-example/result.json`
- `tmp/gvm-stable-first-example/calibration.json`
- `tmp/gvm-stable-first-example/calibration_study.json`

### Pass 2: one small real case

Ask them to try one small real scalar measurement combination from their own
analysis.

Recommended envelope for first external validation:

- up to roughly `10` measurements
- a manageable number of systematic sources
- table-native source of truth preferred
- avoid mixing this first pass with full campaign/reporting layers

Ask them to stay on the stable-first path:

1. prepare table inputs
2. build one canonical spec
3. run fit
4. run calibrate
5. run calibrate-study

Suggested commands:

```bash
nextstat combine-measurements-build-spec \
  --manifest /path/to/manifest.yaml \
  --output /tmp/user-spec.json

nextstat combine-measurements \
  --input /tmp/user-spec.json \
  --output /tmp/user-result.json \
  --solver auto \
  --threads 1

nextstat combine-measurements-calibrate \
  --input /tmp/user-spec.json \
  --output /tmp/user-calibration.json \
  --solver auto \
  --n-toys 32 \
  --seed 42 \
  --threads 1

nextstat combine-measurements-calibrate-study \
  --input /tmp/user-spec.json \
  --output /tmp/user-study.json \
  --solver auto \
  --n-toys 32 \
  --seeds 42,43 \
  --threads 1
```

## What to ask them to return

Ask them to fill out:

- `docs/examples/gvm-stable-first/external-validation-report-template.md`

Minimum requested payload:

- platform and environment
- exact commands run
- whether Pass 1 worked without edits
- whether Pass 2 worked without maintainer help
- the generated JSON outputs or a summary of key fields
- any confusing terminology, schema friction, or unexpected diagnostics

Important fields to capture from result payloads:

- `mu_hat`
- `confidence_interval`
- `goodness_of_fit`
- `diagnostics.requested_solver`
- `diagnostics.effective_solver`
- runtime notes if clearly visible

## What counts as success

Treat the external validation as successful if:

- the committed example runs end-to-end
- one small real case can be mapped to the stable-first path
- no undocumented blocker appears
- returned confusion is localized to wording or ergonomics, not hidden solver behavior

## What counts as a blocker

Treat these as release blockers for the stable-first subset:

- the committed example does not run on a clean user machine
- the table-to-spec path is unclear enough that a maintainer has to translate the data manually
- `auto` fallback behavior is confusing despite `requested_solver` and `effective_solver`
- user cannot tell which outputs belong to the stable subset and which belong to research-grade layers

## Scope discipline

For first-contact external validation, do not ask users to validate:

- scenario-study
- calibration-campaign
- solver-parity
- brief/family/matrix/portfolio/reporting layers

Those remain research-grade and should not dilute the stable-first signal.

## Companion documents

- `docs/quickstarts/hep-gvm-stable-first.md`
- `docs/examples/gvm-stable-first/README.md`
- `docs/guides/gvm-external-validation-tracker-template.md`
- `docs/examples/gvm-stable-first/external-validation-report-template.md`
- `docs/guides/README.md`
