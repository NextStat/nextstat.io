<!--
Audit Report
Generated: 2026-02-05 18:20:20 +0100
Git Commit: unknown
Scope: docs/plans + repo infra (versions/CI/security/release) + bias/coverage policy
Invoked: /audit plans-infra-bias
-->

# Project Audit Report

Date: 2026-02-05  
Project: NextStat (nextstat.io)  
Auditor: Codex  
Scope: `docs/plans` + CI/security/release automation + bias/coverage policy  
Git Commit: unknown

---

## Executive Summary

- Critical: 0
- Major: 0
- Minor: 2
- Total: 2

Overall Health Score: 97/100  
Top risks: long-term, "toys validation" quality depends on the correctness of the `fit(..., data=...)` path and scheduled runs; short-term, keep dependency drift under control via Dependabot and `scripts/versions_audit.py`.

---

## Critical Issues

None found in the audited scope.

## Major Issues

None found in the audited scope.

## Minor Issues

### [Compliance] Python license metadata is partially UNKNOWN in `THIRD_PARTY_LICENSES`
- File: `THIRD_PARTY_LICENSES:122`
- Evidence: some packages (e.g. `numpy`, `pytest`, `maturin`) do not populate `info.license` and/or license classifiers on PyPI, so the report shows `UNKNOWN`.
- Impact: the license report remains useful, but does not always provide a confident answer for Python dependencies.
- Fix: in Phase 3/release process, add a Python license report generated from the lockfile / built wheel environment (e.g. `pip-licenses`) and/or manual clarifications for critical deps.
- Confidence: medium

### [Testing] Bias/coverage gates are defined, but require nightly automation
- File: `docs/plans/standards.md:118`
- Evidence: bias/pull/coverage policy and tasks exist, but the toys-validation workflow is only described as an optional task (Phase 3).
- Impact: without regular runs it is easy to miss statistical drift vs pyhf during optimizations.
- Fix: add `.github/workflows/toys-validation.yml` (nightly/manual) as part of Phase 3 acceptance criteria and save JSON artifacts.
- Confidence: high

---

## Feature Checklist

### Plans + Infrastructure
- [x] Versions/pins confirmed by a snapshot script
- [x] Bias policy (bias/pull/coverage) locked + tasks/gates added
- [x] Security automation: CodeQL + gitleaks + dep audit workflow
- [x] Release automation: wheels + CLI artifacts in GitHub Release
- [x] Third-party licenses report + generator script

---

## Unverified Areas

- Real end-to-end release (wheels install/run on fresh machines) was not verified.
- CodeQL results depend on enabling GitHub Security features at the repo level.

## Files Reviewed

- `docs/plans/*`
- `.github/workflows/*`
- `scripts/versions_audit.py`
- `scripts/generate_third_party_licenses.sh`
- `crates/ns-translate/src/pyhf/model.rs`

## Recommendations

1. Priority 1: add a nightly toys-validation workflow (Phase 3) + artifacts.
2. Priority 2: improve Python license reporting (lockfile-based) before the first public release.
3. Priority 3: keep `docs/plans/versions.md` synchronized with real CI (via `scripts/versions_audit.py` in review).
