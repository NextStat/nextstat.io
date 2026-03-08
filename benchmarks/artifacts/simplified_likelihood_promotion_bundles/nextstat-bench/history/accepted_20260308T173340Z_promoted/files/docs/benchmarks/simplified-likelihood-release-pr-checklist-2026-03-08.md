# Simplified Likelihood Release PR Checklist

**Date**: 2026-03-08  
**Status**: release hardening checklist  
**Scope**: stable simplified-likelihood reinterpretation surface

## Purpose

This checklist is the maintainer-side `PR-ready` gate for the currently accepted
simplified-likelihood stable subset.

Use it after implementation and docs are complete, but before cutting or
shipping a release that claims simplified-likelihood as a stable product
surface.

## Release PR scope

This checklist applies only to the promoted stable subset:

- `nextstat_simplified_likelihood_v0`
- `nextstat_simplified_likelihood_audit_v0`
- `nextstat_apex2_simplified_likelihood_report_v0`
- simplified-likelihood input consumption through `fit`, `hypotest`, `upper-limit`, and `scan`
- CLI, Python, and server audit dispatch for simplified-likelihood inputs
- covariance-form factorization with explicit diagnostics

It does not promote:

- derive/export workflows from full workspaces
- covariance-only ranking/source-semantics guarantees
- arbitrary non-Gaussian reduced nuisance models
- multi-POI reduced likelihood support
- real-world external fixture validation beyond the current synthetic Apex2 matrix

Those remain `research-grade` until separately promoted.

## Pre-PR checklist

### Contract

- [ ] input, audit, and Apex2 report schemas are versioned and published
- [ ] CLI/Python/server accepted-vs-rejected behavior is explicit
- [ ] HS3 remains an explicit reject on the audit path
- [ ] stable vs `research-grade` boundary is stated explicitly in public docs

### Evidence

- [ ] acceptance policy is present:
  - `docs/benchmarks/simplified-likelihood-stable-surface-acceptance-2026-03-08.md`
- [ ] support matrix is present:
  - `docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md`
- [ ] release notes are present:
  - `docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md`
- [ ] current benchmark snapshot is present:
  - `docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08.md`
- [ ] release PR checklist is present:
  - `docs/benchmarks/simplified-likelihood-release-pr-checklist-2026-03-08.md`
- [ ] bench-host promotion runbook is present:
  - `docs/benchmarks/simplified-likelihood-promotion-runbook-2026-03-08.md`
- [ ] artifact/reference page is current:
  - `docs/references/simplified-likelihood-artifacts.md`

### Verification

- [ ] dedicated stable-surface gate workflow exists:
  - `.github/workflows/simplified-likelihood-stable-surface.yml`
- [ ] release workflow requires the stable-surface gate:
  - `.github/workflows/release.yml`
- [ ] local one-command gate exists:
  - `make simplified-likelihood-stable-surface-gate`
- [ ] stable-surface gate passes:

```bash
make simplified-likelihood-stable-surface-gate
```

### Promotion evidence

- [ ] current `nextstat-bench` artifact exists and is linked from the snapshot note
- [ ] current `nextstat-bench` artifact passes the promotion threshold:
  - `min_end_to_end_upper_limit_speedup >= 10x`
- [ ] no acceptance claim relies on terminal-only output without an archived JSON artifact

### Messaging

- [ ] PR summary names the promoted stable subset explicitly
- [ ] PR summary names the remaining `research-grade` layers explicitly
- [ ] no blanket claim is made about all simplified-likelihood workflows being stable
- [ ] `~10x` speedup wording is used only with the `nextstat-bench` artifact in scope

## Recommended PR summary structure

Use a short structure:

1. what is stable now
2. what remains `research-grade`
3. what evidence backs the stable claim
4. how to rerun the stable-surface gate

## Exit condition

The release PR is ready only when every checkbox above is green and the
published stable claim stays within the accepted subset defined by the March 8,
2026 acceptance policy.
