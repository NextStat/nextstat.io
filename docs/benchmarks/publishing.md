---
title: "Publishing Benchmarks (CI, Artifacts, DOI, Replication)"
status: draft
last_updated: 2026-02-08
---

# Publishing Benchmarks (CI, Artifacts, DOI, Replication)

This page defines how benchmark snapshots are published so that “fast” becomes “trusted”.

It covers the backlog items:

- Automate publishing: CI artifacts + baseline manifests + README snippets
- Publish benchmark snapshots with DOI + `CITATION.cff`
- Third-party replication: external rerun + signed report

## 1) Snapshot anatomy (what must exist)

Each published snapshot must include:

- **Raw results** (per test, per repeat)
- **Aggregated summaries** (tables + plots if applicable)
- **Baseline manifest**:
  - NextStat commit SHA
  - toolchains: Rust, Python, compilers
  - dependency locks: `Cargo.lock`, Python lock
  - hardware: CPU, RAM, GPU, driver/runtime versions
  - benchmark configuration (flags, suite selection, warmup policy)
- **Correctness gates report** (e.g., parity deltas used to validate the run)

## 2) CI publishing workflow

Publishing should be automated in CI with the following properties:

- **Immutable**: a snapshot ID maps to a fixed set of artifacts.
- **Re-runnable**: the same harness can be executed locally.
- **Auditable**: manifests and raw results are preserved, not only human summaries.

Best practices:

- store raw results as CI artifacts
- store a small “index” JSON for discovery (suite → snapshot ID → artifact URLs)
- keep baseline comparisons explicit (no silent moving baselines)

## 3) Baselines and regression detection

Benchmarks serve two roles:

1. **Public evidence** (published snapshots)
2. **Regression detection** (CI checks)

For regression detection:

- use stable quick suites (reduced runtime)
- compare against a pinned baseline snapshot
- fail only on meaningful deltas (avoid flakiness)

## 4) DOI + citation

When a snapshot is intended to be cited (papers, technical reports, blog claims), we publish it with:

- a DOI (e.g., Zenodo)
- a `CITATION.cff` describing how to cite the benchmark dataset/snapshot

The DOI should point to:

- raw benchmark outputs
- manifests
- the exact harness version

## 5) Third-party replication (external rerun + signed report)

We treat third-party replication as a first-class trust primitive.

### What “counts” as a replication

At minimum:

- same suite + same dataset IDs
- same harness version (or documented diffs)
- published manifest and raw results

### Signed report format (suggested)

The replication report should include:

- snapshot ID being replicated
- rerun environment manifest
- summary deltas (time distributions, not only one number)
- correctness/parity deltas (must still pass)
- signature (GPG, Sigstore, or equivalent)

### What we do with replications

- link replications from the canonical snapshot index
- prefer “rerun me” evidence over “trust us” claims in blog posts

