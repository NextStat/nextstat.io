# Simplified Likelihood Stable-Surface Acceptance

**Date**: 2026-03-08  
**Status**: Release-hardening acceptance policy  
**Scope**: simplified-likelihood reinterpretation surface for HEP

## Purpose

This document defines the acceptance criteria for treating the current
simplified-likelihood surface as a stable product surface rather than a
research-only prototype.

It answers one narrow question:

- what must be true before we say the current simplified-likelihood surface is
  acceptable for stable reinterpretation workflows?

## Support classes

This policy uses two support classes:

| Class | Meaning |
| --- | --- |
| `stable` | versioned contract with explicit acceptance gates and benchmark evidence |
| `research-grade` | versioned and tested, but not yet covered by the stable-surface acceptance promise |

## Stable subset in scope

The current stable subset is intentionally narrow:

- `nextstat_simplified_likelihood_v0` input contract
- `nextstat_simplified_likelihood_audit_v0` audit contract
- direct consumption through `fit`, `hypotest`, `upper-limit`, and `scan`
- CLI, Python, and server audit dispatch for simplified-likelihood inputs
- covariance-form factorization with explicit diagnostics
- Apex2 verification artifact:
  `nextstat_apex2_simplified_likelihood_report_v0`

Out of scope for the current stable acceptance promise:

- publication/export workflows from full workspaces
- ranking/source-semantics guarantees for covariance-only public releases
- arbitrary non-Gaussian reduced nuisance models
- multi-POI reduced likelihood support

## Acceptance criteria

All items below must be true for acceptance.

### 1. Contract

- input schema is versioned and published
- audit schema is versioned and published
- Apex2 report schema is versioned and published
- CLI/Python/server behavior is explicit for accepted and rejected formats
- HS3 remains an explicit reject on the audit path

### 2. Verification matrix

- Rust translation/integration coverage for simplified-likelihood is green
- CLI smoke coverage for simplified-likelihood is green
- Python smoke coverage for schemas/examples/report runner is green
- the optional Apex2 public-style fixture matrix is green when changes touch the curated public fixture program or external-validation surface
- no unresolved contract drift between docs/examples and emitted artifacts

### 3. Fidelity gates

These are hard acceptance gates for reduced-vs-full comparisons:

- `abs(mu_hat_sl - mu_hat_full) / sigma_mu_full <= 0.05`
- `max_abs(q_mu_sl - q_mu_full) <= 0.1`
- `upper_limit_sl / upper_limit_full` stays in `[0.95, 1.05]`

### 4. Reduction and size gates

- `reduced_nuisance_count / full_nuisance_count <= 0.25`
- `simplified_json_bytes / full_workspace_json_bytes <= 0.35`

### 5. Performance gates

Two performance levels are required:

- CI/release gate:
  `min_end_to_end_upper_limit_speedup >= 3x`
- bench-host promotion gate:
  `min_end_to_end_upper_limit_speedup >= 10x` on `nextstat-bench`

The bench-host gate is the promotion threshold for making the
"~10x reinterpretation speedup" claim on the stable product surface.

## Operational gate surface

The stable subset is enforced through these operational entry points:

- local/CI gate:
  `make simplified-likelihood-stable-surface-gate`
- dedicated workflow:
  `.github/workflows/simplified-likelihood-stable-surface.yml`
- release PR checklist:
  `docs/benchmarks/simplified-likelihood-release-pr-checklist-2026-03-08.md`
- bench-host promotion runbook:
  `docs/benchmarks/simplified-likelihood-promotion-runbook-2026-03-08.md`

### 6. Evidence artifacts

- a current Apex2 JSON report is archived
- a current `nextstat-bench` benchmark snapshot is published
- acceptance claims are linked to committed artifacts, not terminal-only output

## Acceptance decision rule

The surface is acceptable as `stable` only if:

1. all contract artifacts exist and are documented
2. all verification layers are green
3. all fidelity gates pass
4. all reduction/size gates pass
5. the CI performance gate passes
6. the bench-host promotion gate passes for the currently published benchmark note

If any one of these fails, the affected surface remains `research-grade`.

## Current decision for March 8, 2026

Based on the current repository evidence, the acceptance criteria above are
met for the current simplified-likelihood consume/audit surface.

Supporting evidence:

- [Simplified Likelihood Artifacts](/docs/references/simplified-likelihood-artifacts.md)
- [Simplified Likelihood Stable-Surface Support Matrix](/docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md)
- [Simplified Likelihood Stable-Surface Release Notes](/docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md)
- [Simplified Likelihood Benchmark Snapshot: 2026-03-08](/docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08.md)
- [Simplified Likelihood RFC](/docs/rfcs/simplified-likelihoods-reinterpretation.md)

## What acceptance does not imply

Acceptance does not mean:

- arbitrary scale-free speedup claims
- automatic acceptance of export/publication workflows
- exact preservation of full nuisance semantics for covariance-only releases
- stable support for surfaces not listed in the scoped stable subset
