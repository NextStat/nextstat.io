---
title: "GVM External Validation Tracker Template"
status: stable-first
---

# GVM External Validation Tracker Template

Use this template as the maintainer-side tracking sheet for the first external
stable-first GVM validation wave.

Recommended use:

- one tracker per release wave
- one row per external validator
- update it as soon as feedback arrives
- keep links to the returned report and any follow-up issue/PR

This template is the operational companion to:

- `docs/guides/gvm-external-validation-kit.md`
- `docs/guides/gvm-external-validator-outreach-pack.md`
- `docs/examples/gvm-stable-first/external-validator-invite-template.md`
- `docs/examples/gvm-stable-first/external-validation-report-template.md`

## Wave metadata

- Wave label:
- Maintainer owner:
- Start date:
- Target stable-first release:
- Notes:

## Validator table

| Validator | Team / experiment | Contacted | Pass 1 example | Pass 2 real case | Main issue class | Report link | Follow-up owner | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `validator_01` | | | `pending` | `pending` | | | | `pending` |
| `validator_02` | | | `pending` | `pending` | | | | `pending` |
| `validator_03` | | | `pending` | `pending` | | | | `pending` |

Suggested status values:

- `pending`
- `in-progress`
- `passed`
- `passed-with-friction`
- `blocked`
- `closed`

Suggested issue classes:

- `none`
- `docs`
- `packaging`
- `input-format`
- `diagnostics`
- `runtime`
- `release-blocker`

## Findings log

Record only actionable findings here. Do not duplicate the full user report.

| Validator | Severity | Finding | Owner | Resolution target | Linked issue/PR | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `validator_01` | | | | | | `open` |

Suggested severity values:

- `P0`
- `P1`
- `P2`

## Pass criteria

Treat the wave as healthy if:

- at least `2` validators complete Pass 1 without maintainer intervention
- at least `2` validators complete Pass 2 on a small real case
- no `P0 release-blocker` remains open
- repeated confusion is localized and documented

## Release decision summary

- Ready to proceed with stable-first release wave: `yes / no`
- Remaining blockers:
- Repeated adoption friction:
- Docs updates required:
- Packaging updates required:
- Solver/runtime clarifications required:

## Maintainer notes

- Anything that should feed back into:
  - `docs/quickstarts/hep-gvm-stable-first.md`
  - `docs/guides/gvm-external-validation-kit.md`
  - `docs/examples/gvm-stable-first/external-validator-invite-template.md`
  - `docs/examples/gvm-stable-first/external-validation-report-template.md`
