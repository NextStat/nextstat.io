<!--
Audit Report
Generated: 2026-02-05 16:55:45
Git Commit: unknown
Scope: docs/plans + toolchain/deps/CI
Invoked: /audit repo-plans-versions
-->

# Project Audit Report

Date: 2026-02-05  
Project: NextStat (nextstat.io)  
Auditor: Codex  
Scope: `docs/plans` + repo toolchain/deps/CI  
Git Commit: unknown

---

## Executive Summary

- Critical: 0
- Major: 3
- Minor: 4
- Total: 7

Overall Health Score: 88/100  
Top risks: missing security automation (CodeQL/secret scanning) plus missing release pipeline, which makes it hard to publish artifacts safely and reproducibly.

---

## Critical Issues

None found in the audited scope after syncing toolchain/deps and making CI lint/test passes locally.

---

## Major Issues

### [Security] No CodeQL / secret scanning workflows in the repository
- File: `.github/workflows/`
- Evidence: only `rust-tests.yml` and `python-tests.yml` exist; `codeql.yml` / secret scan workflow is missing.
- Impact: no automated SAST/secret scanning in PRs/`main`; higher risk of missing vulnerabilities or accidentally committing secrets.
- Fix:
  - Add `codeql.yml` (init + analysis for Rust/Python).
  - Add a secret scanning job (e.g. gitleaks action) or enable GitHub Advanced Security/secret scanning at the repo level.
- Confidence: high

### [Release] No release pipeline for wheels / binaries / crates
- File: `.github/workflows/`
- Evidence: missing `release.yml` / tag-based pipeline (build wheels for Linux/macOS/Windows, publish GitHub Release).
- Impact: releases will be manual and non-reproducible; higher risk of breaking distributions, especially Python bindings.
- Fix:
  - Add `release.yml` (workflow_dispatch + tag push), build wheels via maturin and publish as Release artifacts.
  - Lock a policy: what exactly is published as OSS vs Pro (see `docs/legal/open-core-boundaries.md`).
- Confidence: high

### [Compliance] Missing `THIRD_PARTY_LICENSES` and a third-party license report generator
- File: `NOTICE`
- Evidence: `NOTICE` implies a third-party license list exists, but `THIRD_PARTY_LICENSES` and a generator are missing.
- Impact: weaker legal/OSS hygiene; can block enterprise adoption.
- Fix:
  - Add generation via `cargo-about`/`cargo-deny` (Rust) and a Python equivalent (if runtime deps appear).
  - Make it part of the release checklist.
- Confidence: medium

---

## Minor Issues

### [Governance] RFC template placeholder
- File: `GOVERNANCE.md:143`
- Evidence: `RFC-XXX: Title`.
- Impact: cosmetic/unclear for external contributors.
- Fix: replace with a real RFC example or move to a standalone template in `docs/`/`.github/`.
- Confidence: high

### [Docs] Need a single "source of truth" for CI/versions
- File: `docs/plans/versions.md`
- Evidence: a baseline exists, but it is important to keep plans and real files (`Cargo.toml`, workflows) in sync.
- Impact: drift risk (plans become stale faster than code).
- Fix: keep `docs/plans/versions.md` as navigation + use `scripts/versions_audit.py` in review.
- Confidence: high

### [Python] Mypy/ruff config is minimal
- File: `bindings/ns-py/pyproject.toml`
- Evidence: missing `tool.ruff`/`tool.mypy` settings.
- Impact: as Python surface area grows, quality will depend on defaults.
- Fix: add a minimal config (line-length, target-version, basic lint set) when Python modules beyond stubs appear.
- Confidence: medium

### [Tooling] rust-toolchain pinned; local install instructions required
- File: `rust-toolchain.toml`
- Evidence: toolchain pinned to `1.93.0`.
- Impact: new contributors without rustup / the pinned toolchain will hit errors.
- Fix: add a short block in `README.md`/`CONTRIBUTING.md`: "Install Rust via rustup; toolchain auto-pins".
- Confidence: medium

---

## Feature Checklist

### Plans + Versioning
- [x] Version baseline exists (`docs/plans/versions.md`)
- [x] Real repo toolchain pinned (`rust-toolchain.toml`)
- [x] CI pins modernized (`actions/*@v6`, `codecov@v5`, rust-cache)
- [x] Clippy passes with `-D warnings` locally
- [ ] Security automation (CodeQL/secret scanning)
- [ ] Release pipeline (wheels/binaries)
- [ ] Third-party licenses report

---

## Unverified Areas

- Real wheel publishing (Linux/Windows) was not verified locally.
- `LICENSE-COMMERCIAL` correctness/completeness was not reviewed by counsel (draft quality).

---

## Files Reviewed

- `docs/plans/*`
- `Cargo.toml`, `rust-toolchain.toml`, `rustfmt.toml`
- `.github/workflows/rust-tests.yml`, `.github/workflows/python-tests.yml`
- `bindings/ns-py/*`
- `crates/ns-translate/src/pyhf/*` (for CI/clippy pass)

---

## Recommendations

1. Priority 1: add `codeql.yml` + secret scanning workflow.
2. Priority 2: add `release.yml` for building wheels (maturin) and release artifacts.
3. Priority 3: implement `THIRD_PARTY_LICENSES` generation and include it in the release checklist.
