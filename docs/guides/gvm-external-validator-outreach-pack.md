---
title: "GVM External Validator Outreach Pack"
status: stable-first
---

# GVM External Validator Outreach Pack

This is the maintainer-facing checklist for the first `2-3` external physics
users who should validate the stable-first scalar GVM path.

Use it together with:

- `docs/guides/gvm-external-validation-kit.md`
- `docs/guides/gvm-external-validation-tracker-template.md`
- `docs/examples/gvm-stable-first/external-validator-invite-template.md`
- `docs/examples/gvm-stable-first/external-validation-report-template.md`

## Objective

Get one short, structured signal from a real external user that answers:

- can they run the committed stable-first example
- can they map one small real case into the stable-first path
- do they understand the outputs and solver diagnostics
- where exactly does first-contact friction still exist

## Who to pick

For the first wave, prefer users who:

- already work with reduced scalar measurements
- can share one small non-sensitive case
- are comfortable running CLI tools
- are not evaluating the whole research-grade reporting pyramid

Avoid starting with:

- full campaign workflows
- HistFactory-first users who do not care about scalar combinations
- very large stress cases

## What to send

Send exactly this bundle:

1. `docs/examples/gvm-stable-first/external-validator-invite-template.md`
2. `docs/quickstarts/hep-gvm-stable-first.md`
3. `docs/examples/gvm-stable-first/README.md`
4. `docs/examples/gvm-stable-first/external-validation-report-template.md`
5. `docs/guides/gvm-external-validation-tracker-template.md`

Optional:

6. `docs/tutorials/hep-gvm-measurement-combinations.md`

Do not send benchmark snapshots or research-grade campaign docs in the first
message unless the user explicitly asks for them.

## Ask for two passes

### Pass 1

Run the committed example:

```bash
make gvm-stable-first-example
```

### Pass 2

Run one small real case on the stable-first path:

```bash
nextstat combine-measurements-build-spec ...
nextstat combine-measurements ...
nextstat combine-measurements-calibrate ...
nextstat combine-measurements-calibrate-study ...
```

## What to ask back

Request:

- completed validation report
- generated output JSONs if they can share them
- plain-text blockers if they cannot share the files

Minimum fields to capture:

- platform / environment
- whether Pass 1 worked without edits
- whether Pass 2 worked without maintainer help
- `diagnostics.requested_solver`
- `diagnostics.effective_solver`
- any confusing wording or packaging friction

## Triage rubric

Treat as `P0 release blockers`:

- committed example does not run
- stable-first path is ambiguous
- output diagnostics are misleading
- user cannot tell stable-first from research-grade surfaces

Treat as `P1 adoption issues`:

- too much manual table cleanup
- command naming friction
- unclear manifest field meanings
- small packaging/environment rough edges with obvious workarounds

Treat as `P2 docs polish`:

- wording
- missing screenshots/examples
- better summary language

## Output discipline

For each external validator, save:

- validator name/team
- date
- pass/fail for Pass 1 and Pass 2
- one short maintainer summary
- linked report or pasted report body

Track those centrally in:

- `docs/guides/gvm-external-validation-tracker-template.md`

## Scope discipline

This outreach pack is for the stable-first subset only.

Do not widen the ask to:

- scenario-study
- calibration-campaign
- solver-parity
- brief/family/matrix/portfolio layers

Those remain research-grade and should not be mixed into the first external
validation wave.
