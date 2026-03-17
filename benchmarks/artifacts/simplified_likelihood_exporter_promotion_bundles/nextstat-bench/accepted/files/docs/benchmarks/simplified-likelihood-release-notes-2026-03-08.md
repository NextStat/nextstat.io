# Simplified Likelihood Stable-Surface Release Notes

**Date**: 2026-03-08  
**Status**: Stable-surface release note  
**Scope**: simplified-likelihood reinterpretation surface for HEP

## Summary

The simplified-likelihood consume/audit path and the narrow exporter subset are
now promoted to `stable`.

This promotion is intentionally narrow. It covers the versioned input/audit
contracts plus the direct reinterpretation path for audit, fit, asymptotic
CLs, upper-limit, scan, and the narrow `nextstat simplify workspace` exporter
path. Discovery-style outputs, ranking, toy CLs, and broader export semantics
remain a `research-grade fallback`.

## Promoted to stable

### Contracts

- `nextstat_simplified_likelihood_v0`
- `nextstat_simplified_likelihood_audit_v0`
- `nextstat_apex2_simplified_likelihood_report_v0`

Published but still research-grade contract:

- `nextstat_simplified_likelihood_export_benchmark_snapshot_report_v0`

Promoted narrow exporter contracts:

- `nextstat_simplified_likelihood_derive_v0`
- `nextstat_simplified_likelihood_export_report_v0`
- `nextstat_simplified_likelihood_exporter_stable_promotion_decision_v0`

### CLI

- `nextstat config schema --name simplified_likelihood_v0`
- `nextstat config schema --name simplified_likelihood_audit_v0`
- `nextstat audit`
- `nextstat fit`
- `nextstat hypotest`
- `nextstat upper-limit`
- `nextstat scan`
- `nextstat config schema --name simplified_likelihood_derive_v0`
- `nextstat config schema --name simplified_likelihood_export_report_v0`
- `nextstat simplify workspace`

### Python and tool surface

- `nextstat.workspace_audit(...)`
- `nextstat.tools.execute_tool("nextstat_workspace_audit", ...)`
- `nextstat.tools.execute_tool("nextstat_fit", ...)`
- `nextstat.tools.execute_tool("nextstat_hypotest", ...)`
- `nextstat.tools.execute_tool("nextstat_upper_limit", ...)`
- `nextstat.tools.execute_tool("nextstat_scan", ...)`

### Server-safe tool surface

- `nextstat_workspace_audit`
- `nextstat_fit`
- `nextstat_hypotest`
- `nextstat_upper_limit`
- `nextstat_scan`

## What remains research-grade

- `nextstat significance` / `nextstat_discovery_asymptotic` on
  simplified-likelihood inputs
- `nextstat hypotest-toys` / `nextstat_hypotest_toys` on
  simplified-likelihood inputs
- ranking / impact surfaces on simplified-likelihood inputs
- advanced export/publication flows beyond the promoted narrow exporter subset
- `aligned_fit_covariance`, non-Gaussian or unconstrained source nuisances, and broader derived-from-workspace variations
- covariance-only ranking/source-semantics guarantees
- source-level nuisance mapping for `derived_from_workspace` reduced artifacts
- real-world external validation beyond the current synthetic Apex2 matrix

## Contract details

The stable subset now has an explicit March 8, 2026 support-class boundary:

- audit, fit, asymptotic CLs, upper-limit, and scan are the promoted
  simplified-likelihood subset
- the derive/export request contract is now part of the promoted narrow stable subset, and the matching runtime path is `nextstat simplify workspace`
- the same runtime emits a machine-readable `nextstat_simplified_likelihood_export_report_v0` artifact through `nextstat simplify workspace --report ...`; that report is now part of the promoted narrow stable subset
- the current runtime boundary is explicit: source workspaces are `pyhf`-only on this path, partial per-channel bin selections are rejected, and derive/export now records `reduction.constraint_covariance_source` explicitly
- the published stable exporter boundary is explicit and narrow: `pyhf` source only, single-POI only, and `constraint_covariance_source="source_model_constraints"` for Gaussian-constrained nuisance sources
- `constraint_covariance_source="aligned_fit_covariance"` remains available as a research-grade compatibility fallback for sources that do not expose Gaussian nuisance constraints
- discovery, ranking, and toy CLs remain callable, but only as
  compatibility-tested `research-grade` surfaces for simplified-likelihood
  inputs
- when ranking is used on simplified-likelihood inputs, impacts apply to reduced
  nuisance coordinates from the compiled model, not source-level systematics;
  covariance-form and `derived_from_workspace` artifacts do not preserve
  original nuisance identities
- HS3 remains an explicit reject on the simplified-likelihood audit path

Tool guidance, server guidance, CLI docs, Python docs, and release-hardening
docs now use the same stable-vs-research-grade wording.

Release hardening now also includes a machine-readable promotion evidence bundle
plus a paired verification report, so the release workflow can ship the
bench-host JSON evidence instead of relying only on the benchmark snapshot note.
That evidence is no longer only a `tmp/` handoff: the accepted frozen bundle is
now persisted under
`benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/`
with a committed `snapshot_index.json`.
The research-grade exporter follow-up evidence is also no longer only a `tmp/`
handoff: the current `nextstat-bench` exporter benchmark artifact now lives at
`benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/`
with its own machine-readable `export_benchmark_snapshot_report.json` and
`snapshot_index.json`.
That exporter path now also has an explicit acceptance note and
runtime gate for stable-surface governance:
`docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md` and
`docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md`.
That same exporter path now also has an accepted promotion-readiness bundle and
runbook under
`benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/`
and `docs/benchmarks/simplified-likelihood-exporter-promotion-runbook-2026-03-09.md`.
That same exporter path now also has a formal stable-review checklist and
machine-readable `stable_review_assessment.json` under
`benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/`
as the historical evidence basis for the promoted narrow stable subset.
That same exporter path now also has a machine-readable
`stable_source_semantics_boundary.json` and a matching public note that publish
the narrow stable source claim for the promoted runtime subset.
That same exporter path now also has a machine-readable stable-candidate blocker
matrix under
`benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_blocker_matrix.json`,
so the delta-to-stable process is explicit instead of buried in prose.
The same accepted path now also carries a validator-facing
`stable_candidate_review_packet.json`, so maintainers have one committed review
input instead of stitching evidence together manually.
The same accepted path now also carries `stable_evidence_policy.json`, so the
release-facing `9 public / 11 total` admission policy and maintenance cadence
are machine-readable instead of living only in prose.
The same accepted path now also carries `stable_evidence_freshness_report.json`,
so the `45-day` freshness window and any freshness breach are machine-readable
release workflow inputs instead of manual checklist interpretation.
As of March 9, 2026, that accepted path also carries
`stable_promotion_decision.json`, and the accepted blocker matrix now reports
`open_blocker_count = 0` for the promoted narrow exporter subset.
The release-facing evidence set now also publishes
`export_public_validation_report.json` as the stable evidence surface for the
curated public exporter matrix without widening the runtime support claim.
The release workflow now consumes `stable_evidence_freshness_report.json`
alongside `stable_evidence_policy.json` and `stable_promotion_decision.json`
for the stable exporter subset.
That explicit decision is what keeps the stable exporter subset narrow while
preserving the wider research-grade fallback.

## Evidence behind the promotion

This promotion is backed by:

- [Simplified Likelihood Stable-Surface Acceptance](/docs/benchmarks/simplified-likelihood-stable-surface-acceptance-2026-03-08.md)
- [Simplified Likelihood Stable-Surface Support Matrix](/docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md)
- [Simplified Likelihood Benchmark Snapshot: 2026-03-08](/docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08.md)
- [Simplified Likelihood Artifacts](/docs/references/simplified-likelihood-artifacts.md)

## Verification gate

The promoted subset is protected by a dedicated repo gate:

- script:
  [simplified_likelihood_stable_surface_gate.sh](/scripts/benchmarks/simplified_likelihood_stable_surface_gate.sh)
- workflow:
  [simplified-likelihood-stable-surface.yml](/.github/workflows/simplified-likelihood-stable-surface.yml)

Local run:

```bash
make simplified-likelihood-stable-surface-gate
```

## Upgrade notes

For users on the promoted subset:

- no schema rename is required
- no CLI rename is required
- no server route rename is required
- the main visible change is that the promoted consume/audit path is now
  documented and governed as `stable`

For users relying on discovery, ranking, toy CLs, or exporter behavior outside
the narrow stable boundary:

- nothing was removed
- those surfaces remain available
- they intentionally continue to be versioned as `research-grade` for
  simplified-likelihood workflows

## Bottom line

This release does not claim that every simplified-likelihood workflow is
stable. It claims something narrower and stronger:

- the consume/audit reinterpretation path is now stable
- the narrow exporter path is now stable under the published source boundary
- the broader reduced-model analysis/export surface remains research-grade
