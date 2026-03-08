# Simplified Likelihood Stable-Surface Release Notes

**Date**: 2026-03-08  
**Status**: Stable-surface release note  
**Scope**: simplified-likelihood reinterpretation surface for HEP

## Summary

The simplified-likelihood consume/audit path is now promoted to `stable`.

This promotion is intentionally narrow. It covers the versioned input/audit
contracts plus the direct reinterpretation path for audit, fit, asymptotic
CLs, upper-limit, and scan. Discovery-style outputs, ranking, toy CLs, and
derive/export workflows remain `research-grade`.

## Promoted to stable

### Contracts

- `nextstat_simplified_likelihood_v0`
- `nextstat_simplified_likelihood_audit_v0`
- `nextstat_apex2_simplified_likelihood_report_v0`

Published but still research-grade contract:

- `nextstat_simplified_likelihood_derive_v0`

### CLI

- `nextstat config schema --name simplified_likelihood_v0`
- `nextstat config schema --name simplified_likelihood_audit_v0`
- `nextstat audit`
- `nextstat fit`
- `nextstat hypotest`
- `nextstat upper-limit`
- `nextstat scan`

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
- `nextstat simplify workspace` and export/publication flows from full workspaces into reduced models
- `nextstat config schema --name simplified_likelihood_derive_v0` and the corresponding derived-from-workspace examples
- covariance-only ranking/source-semantics guarantees
- source-level nuisance mapping for `derived_from_workspace` reduced artifacts
- real-world external validation beyond the current synthetic Apex2 matrix

## Contract details

The stable subset now has an explicit March 8, 2026 support-class boundary:

- audit, fit, asymptotic CLs, upper-limit, and scan are the promoted
  simplified-likelihood subset
- the derive/export request contract is now published and versioned, and the matching runtime path is `nextstat simplify workspace`; that path remains research-grade until it is separately promoted
- the current runtime boundary is explicit: source workspaces are `pyhf`-only on this path and partial per-channel bin selections are rejected
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

For users relying on discovery, ranking, toy CLs, or derive/export:

- nothing was removed
- those surfaces remain available
- they intentionally continue to be versioned as `research-grade` for
  simplified-likelihood workflows

## Bottom line

This release does not claim that every simplified-likelihood workflow is
stable. It claims something narrower and stronger:

- the consume/audit reinterpretation path is now stable
- the broader reduced-model analysis/export surface remains research-grade
