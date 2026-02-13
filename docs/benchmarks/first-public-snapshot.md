---
title: "First Public Benchmark Snapshot (Playbook)"
description: "Step-by-step playbook to publish the first NextStat public benchmark snapshot: standalone harness repo, pinned build reference (wheel or source ref), CI artifact snapshot, and DOI packaging template."
status: draft
last_updated: 2026-02-09
keywords:
  - public benchmarks
  - benchmark snapshot
  - CI artifacts
  - baseline manifest
  - Zenodo DOI
  - reproducible benchmarks
  - NextStat
---

# First Public Benchmark Snapshot (Playbook)

This playbook is the practical “day 1” path for publishing the **first** public benchmark snapshot.

Canonical specs (read later, reference always):

- [Public Benchmarks Specification](/docs/public-benchmarks)
- [Publishing Benchmarks (CI, Artifacts, DOI, Replication)](/docs/benchmarks/publishing)
- [Public Benchmarks Repo Skeleton](/docs/benchmarks/repo-skeleton)

## Outcome (what you will have at the end)

- A standalone repo directory (`nextstat-public-benchmarks`) containing the runnable harness + pinned env scaffolding.
- A CI run that uploads an immutable snapshot artifact directory:
  - `baseline_manifest.json`
  - `snapshot_index.json`
  - `nextstat_wheel.whl` (embedded; self-contained snapshot)
  - suite outputs (e.g. `hep/hep_suite.json`, per-case JSON)
  - `README_snippet*.md`
- (Optional) A packaged archive ready for Zenodo upload (DOI-grade snapshot).

## Step 0 — choose the snapshot id (immutable)

Pick a snapshot id that you will never reuse, e.g.:

- `snapshot-2026-02-09`
- `snapshot-2026-02-09-ubuntu-latest-cpu`

Treat the snapshot id as an immutable “dataset version”.

## Step 1 — create the standalone benchmarks repo (bootstrap)

From this monorepo, export the seed harness into a clean directory:

```bash
python3 benchmarks/nextstat-public-benchmarks/scripts/export_seed_repo.py \
  --out /path/to/nextstat-public-benchmarks \
  --with-github-workflows
```

The exporter intentionally excludes local/host artifacts (virtualenvs, caches, snapshot outputs, and macOS `._*` AppleDouble files) so the standalone repo starts clean.

Then initialize git:

```bash
cd /path/to/nextstat-public-benchmarks
git init
git add -A
git commit -m "Initial public benchmarks harness"
```

## Step 2 — wire CI (GitHub Actions template)

The seed ships workflow templates:

- `.github/workflows/publish.yml` — manual “publish snapshot” workflow (`workflow_dispatch`)
- `.github/workflows/verify.yml` — verification workflow for PR/push (optional)

### Decide how to pin the measured NextStat build

You must publish *what exactly was measured*. The CI template supports two modes.

**Mode A (preferred when you have a hosted wheel):** `nextstat_wheel_url + nextstat_wheel_sha256`

- Pros: fully pinned binary; no source build variance
- Cons: you need to host wheels for the runner OS/arch + Python version

**Mode B (recommended for the first Linux snapshot):** build from source with `nextstat_ref`

- Pros: no wheel hosting required; still pinned to an exact git ref
- Cons: build toolchain becomes part of the environment; publish the ref + commit SHA in the manifest

The template enforces: either (A) `nextstat_wheel_url` + `nextstat_wheel_sha256`, or (B) `nextstat_ref`.

Default repo for build-from-source mode is `nextstat/nextstat.io` with `nextstat_py_subdir = bindings/ns-py`.

## Step 3 — run the publish workflow once (first snapshot)

Trigger `.github/workflows/publish.yml` (“Publish Benchmark Snapshot (Template)”).

Provide:

- `snapshot_id`: your immutable snapshot id
- `run_hep`, `run_pharma`, `run_bayesian`, `run_ml`: pick suites for the first snapshot

Then choose one of:

- `nextstat_wheel_url` + `nextstat_wheel_sha256`, or
- `nextstat_ref` (plus optional `nextstat_repo`, `nextstat_py_subdir`)

What CI does:

1. installs pinned harness deps (`env/python/requirements.txt`)
2. installs NextStat (downloaded wheel or built wheel)
3. runs `scripts/publish_snapshot.py --deterministic --nextstat-wheel tmp/nextstat.whl ...`
4. uploads `manifests/snapshots/<snapshot_id>/` as a CI artifact

## Step 4 — package for DOI (Zenodo template)

After CI completes, download the artifact directory `manifests/snapshots/<snapshot_id>/`.

Package it using the seed Zenodo helper:

```bash
python3 zenodo/package_snapshot.py \
  --snapshot-dir manifests/snapshots/<snapshot_id> \
  --out-dir tmp/zenodo_out/<snapshot_id>
```

This generates:

- a `.tar.gz` archive of the snapshot directory
- SHA-256 for the archive
- `zenodo_deposition.json` metadata stub

Upload to Zenodo (manual) and record the DOI in:

- the snapshot announcement blog post (when you cite results)
- `CITATION.cff` in the standalone benchmarks repo (recommended)

## Optional — local dry run (no GitHub Actions)

If you want to validate the harness end-to-end before wiring CI, you can generate a local snapshot directory:

```bash
python3 scripts/publish_snapshot.py \
  --snapshot-id snapshot-local-smoke \
  --deterministic \
  --suite public-benchmarks \
  --fit --fit-repeat 3
```

To pin the measured build, pass a wheel path:

```bash
python3 scripts/publish_snapshot.py \
  --snapshot-id snapshot-local-smoke \
  --deterministic \
  --nextstat-wheel /path/to/nextstat-*.whl
```

Then package it:

```bash
python3 zenodo/package_snapshot.py \
  --snapshot-dir manifests/snapshots/snapshot-local-smoke \
  --out-dir tmp/zenodo_out/snapshot-local-smoke
```

## Step 5 — prepare the replication ask (public)

Replication is the trust multiplier. For the first snapshot announcement, include:

- the snapshot id
- a link to the artifact/DOI
- a short “how to rerun” section (one command + pinned build ref)
- what to publish back (their `snapshot_index.json` + replication report)

Runbook: [Third-Party Replication Runbook](/docs/benchmarks/replication).
