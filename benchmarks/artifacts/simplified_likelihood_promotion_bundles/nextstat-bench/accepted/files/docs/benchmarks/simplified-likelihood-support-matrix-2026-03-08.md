# Simplified Likelihood Stable-Surface Support Matrix

**Date**: 2026-03-08  
**Status**: Executed support matrix  
**Scope**: simplified-likelihood reinterpretation surface for HEP

## Purpose

This document is the short operational matrix for the promoted
simplified-likelihood subset.

It answers one narrow question:

- what is `stable` now
- what remains `research-grade`
- which CLI, Python, and server surfaces are covered by the March 8, 2026
  promotion

## Support classes

| Class | Meaning |
| --- | --- |
| `stable` | public compatibility promise for the named simplified-likelihood subset |
| `research-grade` | versioned and tested, but still evolving without stable-surface promise |

## Stable CLI matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `nextstat config schema --name simplified_likelihood_v0` | `stable` | published input contract |
| `nextstat config schema --name simplified_likelihood_audit_v0` | `stable` | published audit contract |
| `nextstat config schema --name simplified_likelihood_derive_v0` | `research-grade` | published derive/export planning contract; runtime exporter is not in the promoted stable subset |
| `nextstat audit` | `stable` | pyhf + simplified-likelihood audit path; HS3 remains an explicit reject |
| `nextstat fit` | `stable` | direct consume path for simplified-likelihood JSON |
| `nextstat hypotest` | `stable` | asymptotic CLs consume path for simplified-likelihood JSON |
| `nextstat upper-limit` | `stable` | promoted reinterpretation benchmark path |
| `nextstat scan` | `stable` | promoted reduced-model profile-scan path |
| `nextstat simplify workspace` | `research-grade` | runtime derive/export path from fitted pyhf workspaces into `derived_from_workspace` artifacts; explicit boundary: pyhf-only source and no partial-bin selection |
| `nextstat significance` | `research-grade` | simplified-likelihood input is compatibility-tested, but discovery-style output is outside the promoted stable subset |
| `nextstat hypotest-toys` | `research-grade` | simplified-likelihood input is compatibility-tested, but toy CLs is outside the promoted stable subset |
| ranking / impact surfaces | `research-grade` | simplified-likelihood input is compatibility-tested, but ranking acts on reduced nuisance coordinates rather than source-level systematics; covariance-form and `derived_from_workspace` source semantics remain outside the promoted stable subset |
| export / publication commands | `research-grade` | derive/export schema, examples, and runtime export are available, but they remain outside the promoted stable subset |

## Stable Python matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `nextstat.workspace_audit(...)` | `stable` | published `nextstat_simplified_likelihood_audit_v0` artifact for simplified-likelihood inputs |
| `nextstat.tools.execute_tool("nextstat_workspace_audit", ...)` | `stable` | local and server tool surface |
| `nextstat.tools.execute_tool("nextstat_fit", ...)` | `stable` | local and server tool surface |
| `nextstat.tools.execute_tool("nextstat_hypotest", ...)` | `stable` | local and server tool surface |
| `nextstat.tools.execute_tool("nextstat_upper_limit", ...)` | `stable` | local and server tool surface |
| `nextstat.tools.execute_tool("nextstat_scan", ...)` | `stable` | local and server tool surface |
| `nextstat.tools.execute_tool("nextstat_discovery_asymptotic", ...)` | `research-grade` | simplified-likelihood input is compatibility-tested, but discovery-style output is outside the promoted stable subset |
| `nextstat.tools.execute_tool("nextstat_hypotest_toys", ...)` | `research-grade` | simplified-likelihood input is compatibility-tested, but toy CLs is outside the promoted stable subset |
| `nextstat.tools.execute_tool("nextstat_ranking", ...)` | `research-grade` | simplified-likelihood input is compatibility-tested, but ranking acts on reduced nuisance coordinates rather than source-level systematics |

## Stable server matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `nextstat_workspace_audit` | `stable` | remote server-safe audit path for simplified-likelihood JSON |
| `nextstat_fit` | `stable` | remote server-safe fit path for simplified-likelihood JSON |
| `nextstat_hypotest` | `stable` | remote server-safe asymptotic CLs path |
| `nextstat_upper_limit` | `stable` | remote server-safe reinterpretation benchmark path |
| `nextstat_scan` | `stable` | remote server-safe reduced-model profile scan |
| `nextstat_discovery_asymptotic` | `research-grade` | simplified-likelihood input is compatibility-tested, but discovery-style output is outside the promoted stable subset |
| `nextstat_hypotest_toys` | `research-grade` | simplified-likelihood input is compatibility-tested, but toy CLs is outside the promoted stable subset |
| `nextstat_ranking` | `research-grade` | simplified-likelihood input is compatibility-tested, but ranking acts on reduced nuisance coordinates rather than source-level systematics |

## Boundaries that remain research-grade

- derive/export workflows from full workspaces beyond the current research-grade `nextstat simplify workspace` path
- covariance-only ranking and source-semantics guarantees
- source-level nuisance mapping for `derived_from_workspace` reduced artifacts
- arbitrary non-Gaussian reduced nuisance models
- multi-POI reduced likelihood support
- bench-host external validation beyond the current synthetic Apex2 promotion
  matrix; curated public-style consume fixtures now participate in an Apex2
  runtime matrix, but they are still not promotion-grade benchmark evidence on
  `nextstat-bench`

## Evidence and gate

The promoted stable subset is backed by:

- [Simplified Likelihood Stable-Surface Acceptance](/docs/benchmarks/simplified-likelihood-stable-surface-acceptance-2026-03-08.md)
- [Simplified Likelihood Benchmark Snapshot: 2026-03-08](/docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08.md)
- [Simplified Likelihood Stable-Surface Release Notes](/docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md)
- [Simplified Likelihood Artifacts](/docs/references/simplified-likelihood-artifacts.md)

Companion Apex2 evidence now also exists for curated public-style consume
fixtures through the optional `--include-public-fixtures` matrix in
`tests/apex2_simplified_likelihood_report.py`. That matrix is runtime evidence
for public-style basis/covariance/derived examples, not a replacement for the
paired synthetic speedup artifact or the `nextstat-bench` promotion gate.

Operational gate:

- script:
  [simplified_likelihood_stable_surface_gate.sh](/scripts/benchmarks/simplified_likelihood_stable_surface_gate.sh)
- workflow:
  [simplified-likelihood-stable-surface.yml](/.github/workflows/simplified-likelihood-stable-surface.yml)
- make target:
  `make simplified-likelihood-stable-surface-gate`

## Bottom line

The stable product promise is intentionally narrow:

- audit, fit, asymptotic CLs, upper-limit, and scan are `stable`
- discovery, ranking, and toy CLs remain available but stay
  `research-grade` for simplified-likelihood inputs
- export/publication and broader reduced-model semantics remain outside the
  current promoted subset
