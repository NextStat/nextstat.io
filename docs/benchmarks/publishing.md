---
title: "Publishing Benchmarks (CI, Artifacts, DOI, Replication)"
description: "How NextStat benchmark snapshots are published: CI automation, immutable artifacts, baseline manifests, DOI minting, validation report integration, and third-party replication with signed reports."
status: draft
last_updated: 2026-02-08
keywords:
  - benchmark publishing
  - CI benchmark artifacts
  - DOI benchmarks
  - Zenodo scientific software
  - benchmark replication
  - signed benchmark report
  - validation report
  - NextStat
---

# Publishing Benchmarks (CI, Artifacts, DOI, Replication)

This page defines how benchmark snapshots are published so that “fast” becomes “trusted”.

It covers the backlog items:

- Automate publishing: CI artifacts + baseline manifests + README snippets
- Publish benchmark snapshots with DOI + `CITATION.cff`
- Third-party replication: external rerun + signed report

If you want a single “do this first” checklist, use: [First Public Benchmark Snapshot (Playbook)](/docs/benchmarks/first-public-snapshot).

For the shipped unbinned cross-framework benchmark tables (NextStat/RooFit/zfit/MoreFit):
- [Unbinned Likelihood Benchmark Suite](/docs/benchmarks/unbinned-benchmark-suite)

For exact rerun commands and output JSON contract in one place:
- [Unbinned Benchmark Reproducibility](/docs/benchmarks/unbinned-reproducibility)

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
- **Validation report** (`validation_report.json` + optional `validation_report.pdf`) produced by [`nextstat validation-report`](/docs/validation-report), containing dataset SHA-256 fingerprint, model spec, environment, and per-suite pass/fail matrix (plus a signable `validation_pack_manifest.json` + signature files when used)

The single-command entrypoint for generating a complete validation pack is:

```bash
make validation-pack
```

This produces `apex2_master_report.json` + `validation_report.json` (+ optional PDF) + `validation_pack_manifest.json` in `tmp/validation_pack/`.

## 2) CI publishing workflow

Publishing should be automated in CI with the following properties:

- **Immutable**: a snapshot ID maps to a fixed set of artifacts.
- **Re-runnable**: the same harness can be executed locally.
- **Auditable**: manifests and raw results are preserved, not only human summaries.

Best practices:

- store raw results as CI artifacts
- store a small “index” JSON for discovery (suite → snapshot ID → artifact URLs and/or SHA-256 hashes)
- keep baseline comparisons explicit (no silent moving baselines)

In this repo, CI uses a minimal snapshot index format (`snapshot_index.json`) with a stable schema:
`docs/schemas/benchmarks/snapshot_index_v1.schema.json`.

Unbinned benchmark contract is also CI-gated:
- workflow: `.github/workflows/python-tests.yml`
- job: `validation-pack`
- artifact: `artifacts/unbinned_bench_smoke.json`
- schema: `docs/schemas/benchmarks/unbinned_run_suite_result_v1.schema.json`
- baseline drift gate: `scripts/benchmarks/compare_unbinned_bench.py`
  (`gauss_exp` smoke, fails if `nextstat._wall_ms` regresses by >2.5x)
- drift summary artifact: `artifacts/unbinned_bench_drift_summary.json`

Unbinned toy-fit parity is CI-gated as a backend matrix:
- workflow: `.github/workflows/unbinned-toy-parity.yml`
- backends: `cpu`, `cuda`, `metal` (GPU jobs run on push, or PRs labeled `ci:gpu`)
- per-backend artifacts: `unbinned-toy-parity-<backend>` (JSON files emitted by parity tests)
- aggregated matrix diff-report artifact: `unbinned-toy-parity-matrix-report`
  (`artifacts/unbinned_toy_parity_matrix_report.json`)
- schemas:
  - `docs/schemas/benchmarks/unbinned_toy_parity_report_v1.schema.json`
  - `docs/schemas/benchmarks/unbinned_toy_parity_matrix_report_v1.schema.json`

### Public benchmarks repo template: pin the exact NextStat wheel

In the standalone public benchmarks harness (seed: `benchmarks/nextstat-public-benchmarks/`), CI templates install a
**pinned** NextStat wheel by URL + SHA-256, and then generate a snapshot directory containing `baseline_manifest.json`
and `snapshot_index.json`.

Important: the wheel must match the CI runner OS/arch and Python version (e.g., `ubuntu-latest` + Python `3.13`).

For DOI/public snapshots, the publisher can include the wheel file itself inside the snapshot directory as
`nextstat_wheel.whl` (in addition to recording the wheel hash in the baseline manifest).

Build a wheel from a NextStat checkout:

```bash
cd /path/to/nextstat.io/bindings/ns-py
maturin build --release
ls target/wheels/nextstat-*.whl
```

Compute the wheel SHA-256:

```bash
# macOS
shasum -a 256 target/wheels/nextstat-*.whl

# Linux
sha256sum target/wheels/nextstat-*.whl
```

Then configure the public benchmarks repo CI templates:

- `ci/verify.yml`: set repo variables `NEXTSTAT_WHEEL_URL` + `NEXTSTAT_WHEEL_SHA256`
- `ci/publish.yml`: provide `nextstat_wheel_url` + `nextstat_wheel_sha256` as workflow inputs

If you don’t have a hosted wheel URL yet, the template `ci/publish.yml` can also build the wheel from source
when `nextstat_wheel_url` is empty (requires a pinned `nextstat_ref`).

### First public snapshot (checklist)

1. Decide the **suite label** (e.g. `public-benchmarks`) and an immutable **snapshot id** (e.g. `snapshot-2026-02-08`).
2. Choose the **runner hardware** (GitHub-hosted, self-hosted, or a lab machine) and record CPU/GPU details in the manifest.
3. Produce a **pinned NextStat wheel** for that OS/arch + Python version and record its SHA-256.
4. Run the publisher (`scripts/publish_snapshot.py`) to create:
   - suite outputs (raw JSON)
   - `baseline_manifest.json` (+ `nextstat.wheel_sha256`)
   - `snapshot_index.json` (hash inventory)
   - optionally `nextstat_wheel.whl` embedded in the snapshot (DOI-friendly)
5. Upload the snapshot directory as CI artifacts and/or package it for DOI publishing.

### Packaging a DOI-grade snapshot (Zenodo template)

Once you have a complete snapshot directory (for example, downloaded from CI artifacts or created locally by the seed harness),
package it into a publishable archive:

```bash
cd benchmarks/nextstat-public-benchmarks
python3 zenodo/package_snapshot.py \
  --snapshot-dir manifests/snapshots/<snapshot_id> \
  --out-dir tmp/zenodo_out/<snapshot_id>
```

This produces:

- a `.tar.gz` archive of the snapshot directory
- a SHA-256 digest file for the archive
- a derived `zenodo_deposition.json` (metadata you can paste into a Zenodo deposition)

Then upload the archive to Zenodo (manual step) and record the DOI in:

- the snapshot README or index page,
- the blog post where you cite the result, and
- `CITATION.cff` in the standalone benchmarks repo (recommended).

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

The `validation_report.json` includes SHA-256 hashes for both the workspace and the Apex2 master report. Signing this JSON with GPG or Sigstore creates a complete chain: *data hash → validation result → signer identity*.

### What we do with replications

- link replications from the canonical snapshot index
- prefer "rerun me" evidence over "trust us" claims in blog posts

## Related reading

- [Benchmark Snapshots as Products](/blog/benchmark-snapshots-ci-artifacts) — why snapshots are artifact sets (not screenshots).
- [Public Benchmarks Specification](/docs/public-benchmarks) — canonical spec for suite structure and protocols.
- [Unbinned Likelihood Benchmark Suite](/docs/benchmarks/unbinned-benchmark-suite) — published cross-framework tables.
- [Unbinned Benchmark Reproducibility](/docs/benchmarks/unbinned-reproducibility) — commands + JSON schema contract.
- [Validation Report Artifacts](/docs/validation-report) — the `validation_report.json` + PDF system.
- [Third-Party Replication Runbook](/docs/benchmarks/replication) — step-by-step rerun + signed report template.
- [Third-Party Replication: Signed Reports](/blog/third-party-replication-signed-report) — blog post on the replication protocol.
