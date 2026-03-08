# Simplified Likelihood Exporter Stable Source-Semantics Boundary

**Date**: 2026-03-09  
**Status**: published stable boundary for the promoted narrow exporter subset  
**Scope**: `nextstat simplify workspace` within the accepted narrow stable claim

## Purpose

This note closes one specific ambiguity in the exporter hardening track:

- not "is every exporter-compatible path stable?"
- but "what exact source semantics are covered by the promoted narrow stable subset?"

The current answer is intentionally narrow.

## Published stable boundary

The promoted stable exporter claim must stay within all of the following
boundaries:

- source workspace input stays `pyhf`-only
- the exporter claim stays single-POI only
- the stable path only promises
  `constraint_covariance_source = "source_model_constraints"`
- that promise only covers Gaussian-constrained source nuisances on the source
  side
- the emitted reduced artifact still carries
  `metadata.source_format = "derived_from_workspace"`
- the emitted artifact remains a reduced-coordinate model, not a source-level
  nuisance identity replica
- ranking and impact views on derived reduced artifacts remain reduced-coordinate
  diagnostics, not a source-level systematic breakdown

## Explicitly outside that stable claim

The following paths stay outside the promoted stable exporter claim unless a later
contract says otherwise:

- `constraint_covariance_source = "aligned_fit_covariance"`
- non-Gaussian or unconstrained source nuisances on the
  `source_model_constraints` path
- partial per-channel bin selection
- multi-POI export
- source-level nuisance identity preservation through reduction
- source-level ranking/impact semantics on derived reduced artifacts

These must remain explicit rejects or `research-grade`-only paths rather than
silent degradation.

## Machine-readable contract

The canonical JSON contract for this boundary is:

- schema:
  `docs/schemas/benchmarks/simplified_likelihood_exporter_stable_source_semantics_boundary_v0.schema.json`
- example:
  `docs/specs/benchmarks/simplified_likelihood_exporter_stable_source_semantics_boundary_v0.example.json`
- generator:
  `scripts/benchmarks/build_simplified_likelihood_exporter_stable_source_semantics_boundary.py`

The committed accepted artifact is:

- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_source_semantics_boundary.json`

## Alignment surface

This boundary is now the wording source of truth for:

- [Simplified Likelihood Exporter Acceptance](/docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md)
- [Simplified Likelihood Exporter Runtime Gate](/docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md)
- [Simplified Likelihood Stable-Surface Support Matrix](/docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md)
- [Simplified Likelihood Stable-Surface Release Notes](/docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md)
- [Simplified Likelihood Artifacts](/docs/references/simplified-likelihood-artifacts.md)
- [CLI Reference](/docs/references/cli.md)
- [Python API](/docs/references/python-api.md)
- [Server API](/docs/references/server-api.md)
- [Rust API](/docs/references/rust-api.md)

## Why this closes the blocker

The old blocker was not about missing math evidence. That part was already green
on `nextstat-bench`.

The old blocker was that public wording still treated exporter source semantics
as a loose research-grade blob. That was too vague for a stable claim.

This note closes that gap by making the stable exporter claim:

- narrow
- explicit
- versioned
- machine-readable

## Boundary reminder

This note is the stable source-boundary definition, not an automatic
promotion mechanism.

Today:

- the narrow subset is `stable`
- `automatic_stable_promotion = false`
- everything outside this boundary remains `research-grade`
