---
title: "Public Benchmarks Repo Skeleton"
status: shipped
last_updated: 2026-02-08
---

# Public Benchmarks Repo Skeleton (Pinned Environment)

This document specifies the intended structure of the **public benchmarks** repository.

Goal: make benchmarks **rerunnable by outsiders** with minimal friction and minimal ambiguity.

Canonical benchmark program overview: `docs/benchmarks/public-benchmarks.md`.

Seed implementation (in this repo): `benchmarks/nextstat-public-benchmarks/`.

## Principles

- **One snapshot = one immutable artifact set**
- **Pinned envs** (toolchains + dependencies) are part of the repo, not an “optional” note
- **Correctness gates before timing** are mandatory
- **Raw results are published**, not only summaries

## Repository layout (proposed)

```
nextstat-public-benchmarks/
  README.md
  LICENSE
  CITATION.cff

  manifests/
    schema/
      baseline_manifest_v1.schema.json
      snapshot_index_v1.schema.json
      replication_report_v1.schema.json
      benchmark_result_v1.schema.json
      benchmark_suite_result_v1.schema.json
      pharma_benchmark_result_v1.schema.json
      pharma_benchmark_suite_result_v1.schema.json
      hep_root_baseline_result_v1.schema.json
      pharma_baseline_result_v1.schema.json
    snapshots/
      <snapshot_id>/
        baseline_manifest.json
        snapshot_index.json
        README_snippet.md
        hep/
          hep_suite.json
          *.json
        pharma/
          pharma_suite.json
          *.json
        replication/
          replication_report.json  # optional (external rerun)

  suites/
    hep/
      README.md
      datasets/
      run.py
      report.py
    pharma/
      README.md
      datasets/
      run.py
      report.py
    bayesian/
      README.md
      datasets/
      run.py
      report.py
    ml/
      README.md
      run.py
      report.py

  env/
    docker/
      cpu.Dockerfile
      cuda.Dockerfile
      README.md
    python/
      pyproject.toml
      uv.lock   # or requirements.txt + hash lock
    rust/
      rust-toolchain.toml
      Cargo.lock

  ci/
    publish.yml
    verify.yml
```

Notes:

- The exact locking tool for Python (`uv`, `pip-tools`, Poetry) is a choice, but the outcome must be: “install produces the same deps”.
- For GPU benchmarks, publishing must include the runtime versions (CUDA toolkit/driver or Metal/macOS version) in the manifest.

## Snapshot ID and manifest format

Each CI run that publishes numbers creates:

- a snapshot ID (opaque, immutable)
- a baseline manifest JSON (schema-validated)
- raw results per suite

In addition, each snapshot includes a small `snapshot_index.json` containing file hashes for the full
artifact set, which makes snapshots discoverable and supports third-party replication.

The snapshot manifest should include:

- NextStat commit SHA and repo URL
- harness repo commit SHA
- environment versions (Rust/Python, OS, CPU/GPU)
- dataset IDs + hashes
- suite configuration (flags, warmup policy, repeats, aggregation policy)

## Correctness gates (required)

Every suite must emit a machine-readable correctness report and fail CI if gates fail.

Examples:

- HEP: NLL parity vs pyhf for representative workspaces and parameter points
- Bayesian: posterior sanity checks + diagnostics validity + seed policy
- ML: shape/dtype checks + deterministic cache policy for cold-start runs

## What belongs in this repo vs NextStat

**In NextStat (`nextstat.io` repo):**

- canonical methodology docs (what we measure, how to read it)
- suite definitions and threat-model docs (non-executable reference)
- blog posts interpreting results

**In the public benchmarks repo:**

- the runnable harness
- pinned environments
- dataset acquisition scripts (or pinned dataset artifacts when legally possible)
- snapshot manifests + raw results

## Why not put everything in one repo?

Keeping the harness in a separate repo is a practical trust decision:

- the harness should be auditable and runnable without building the entire product repo
- published artifacts should remain stable even as product development continues
