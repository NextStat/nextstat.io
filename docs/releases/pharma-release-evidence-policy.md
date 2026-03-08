---
title: "Pharma Release Evidence Policy"
status: shipped
---

# Pharma Release Evidence Policy

This document defines the stable prerelease and release policy for Pharma validation
artifacts, especially surfaces that include FOCE and SAEM estimation.

The goal is to keep release validation scientifically meaningful without pretending
that every stochastic PK/NLME artifact is a cross-platform exact snapshot.

## Packaging policy

Prerelease and release-candidate validation must use local build artifacts only.

Required rules:

- do not depend on PyPI for `nextstat` or `nextstat-cli`
- do not rewrite `pyproject.toml` during prerelease validation
- do not validate the release package through `PYTHONPATH=bindings/ns-py/python`
  when the compiled `nextstat._core` extension is required

Canonical prerelease Python install path:

1. build local `nextstat-cli` wheels/binaries
2. build local `nextstat` wheels
3. install from a local wheelhouse with `--no-index`

This ensures the prerelease gate validates the real installed package surface,
not a source-shadowed wrapper.

## Canonical pharma release evidence

`pharma_validation.json` used by the release candidate is treated as
**canonical Linux release evidence**.

That means:

- it is generated on the canonical Linux release platform
- it is reused by downstream deterministic packaging steps
- it is not required to be bit-identical across other platforms

This matches the release-candidate workflow, where the publishable validation-pack
and M15 path reuse the already rendered `pharma_validation.json` instead of
rerunning FOCE/SAEM in later deterministic rerenders.

## Cross-platform SAEM policy

SAEM is stochastic and sensitive to BLAS/LAPACK/optimizer path differences.

This section defines the cross-platform SAEM compatibility contract for release
preparation.

Because of that, the following are **not** release-grade cross-platform exact
snapshot fields:

- `eta`
- `omega`
- `omega_matrix`
- `correlation`
- other latent path-dependent SAEM internals

These fields may still be inspected, archived, and included in canonical Linux
evidence, but they are not suitable as strict macOS-vs-Linux snapshot parity gates.

## What cross-platform compatibility means

Cross-platform prerelease validation for SAEM must use semantic acceptance
criteria, not raw snapshot equality.

Allowed acceptance dimensions include:

- run completes successfully
- convergence state is acceptable
- objective value is finite
- `theta` remains inside scientific acceptance bands
- `omega` remains finite, positive, and inside platform acceptance bands
- downstream publishable artifacts remain schema-valid and structurally complete

Not allowed as cross-platform release criteria:

- bit-identical SAEM JSON
- exact `omega` equality across platforms
- exact `eta` equality across platforms

## Same-platform reproducibility

Same-seed same-platform determinism remains a valid regression property.

That is a narrower contract:

- good for backend/platform reproducibility
- not a substitute for cross-platform release semantics

## Validation-pack policy

The validation-pack wrapper supports two pharma modes:

1. render canonical pharma evidence
2. reuse pre-seeded canonical pharma evidence with `--skip-pharma-validation`

The second path exists specifically so publishable deterministic rerenders do not
pretend to be a second independent stochastic SAEM execution.

For release preparation, this means:

- first render the canonical Linux `pharma_validation.json`
- then reuse it for deterministic bundle assembly and M15 packaging

## Release-candidate policy

The release candidate workflow must preserve these invariants:

- local-artifact-only Python install path
- canonical Linux `pharma_validation.json`
- deterministic rerender paths that reuse that canonical pharma evidence
- no hidden second SAEM rerun in publishable bundle rendering

## What this policy forbids

- mutating package metadata just to make prerelease installs work
- treating source-tree Python imports as equivalent to installed package validation
- using exact SAEM latent-output parity as a cross-platform release gate

## Related references

- [release-runbook.md](docs/releases/release-runbook.md)
- [benchmark-artifact-policy.md](docs/releases/benchmark-artifact-policy.md)
- [validation-report.md](docs/references/validation-report.md)
