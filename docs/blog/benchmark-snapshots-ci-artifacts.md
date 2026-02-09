---
title: "Benchmark Snapshots as Products: CI Artifacts, Manifests, and Baselines"
slug: benchmark-snapshots-ci-artifacts
description: "How NextStat publishes benchmark snapshots as rerunnable artifact sets: CI automation, baseline manifests, correctness gates, and validation-report-backed evidence."
date: 2026-02-08
author: NextStat Team
status: draft
keywords:
  - benchmark snapshots
  - CI benchmark artifacts
  - baseline manifest
  - reproducible benchmarks
  - benchmark publishing
  - validation report
  - NextStat
category: trust
---

<!--
  Blog draft (technical / trust-building).
  Canonical publishing spec: docs/benchmarks/publishing.md
-->

# Benchmark Snapshots as Products: CI Artifacts, Manifests, and Baselines

**Trust Offensive series:** [Index](/blog/trust-offensive-public-benchmarks) · **Prev:** [The End of the Scripting Era](/blog/end-of-scripting-era-benchmarks) · **Next:** [Third-Party Replication: Signed Reports](/blog/third-party-replication-signed-report)

Benchmarks are not just measurements — they are *claims*.

If a claim is not rerunnable, it is not evidence. It’s a screenshot.

This post explains the publishing layer of our public benchmark program: how we turn “we ran it once” into a **benchmark snapshot** that others can rerun, audit, and (eventually) cite.

Canonical spec: [Publishing Benchmarks](/docs/benchmarks/publishing).

If you want a single “do this first” runbook (first snapshot end-to-end), use: [First Public Benchmark Snapshot (Playbook)](/docs/benchmarks/first-public-snapshot).

---

## Abstract

We treat a benchmark snapshot as a **product artifact set**, not a blog table:

- raw per-test measurements (distributions, not only medians),
- correctness gates (proof that the computation matches what we claim),
- pinned environments (so “install” means “same deps”),
- manifests + hashes (so “downloaded” means “unchanged”), and
- an index format for discovery and replication.

This is boring on purpose. The goal is to make the performance claim **auditable**.

---

## 1) Definitions: what a “snapshot” is (and isn’t)

A snapshot is an immutable set of files produced by a benchmark harness run.

In our language:

- **Harness**: the code that runs benchmark workflows and writes outputs.
- **Snapshot ID**: an opaque identifier (e.g. `snapshot-2026-02-08`) that maps to *exactly one* artifact set.
- **Raw results**: per-test, per-repeat timings and correctness deltas.
- **Manifest**: machine-readable “what was run on what” metadata.
- **Correctness gate**: an explicit check that fails fast if results are inconsistent with the reference.
- **Deterministic mode**: best-effort stable JSON/PDF output to support hashing and diffing.

What a snapshot is **not**:

- a single “best run” number,
- a chart without the raw samples,
- a benchmark that doesn’t prove it computed the same model as the reference.

---

## 2) Snapshot anatomy: the minimum publishable artifact set

At minimum, each snapshot includes:

1. **Raw benchmark outputs** (per test, per repeat).
2. **Aggregated summaries** (tables/plots derived from raw outputs).
3. **Baseline manifest** (code version, environment, hardware, settings, dataset hashes).
4. **Pinned NextStat build reference** (recommended): either a `nextstat.wheel_sha256` in the manifest, or the wheel file itself (`nextstat_wheel.whl`) embedded in the snapshot for DOI publishing.
5. **Correctness gates report** (parity/sanity checks used to validate the run).
6. **Validation pack** (a unified, signed/“signable” bundle for audit and replication):
   - `validation_report.json` (schema `validation_report_v1`)
   - optional `validation_report.pdf`
   - `validation_pack_manifest.json` (SHA-256 + sizes for core files)

The single-command entrypoint for generating a complete validation pack is:

```bash
make validation-pack
```

This produces `apex2_master_report.json` + `validation_report.json` (+ optional PDF) + `validation_pack_manifest.json` in `tmp/validation_pack/`. See: [Validation Report Artifacts](/docs/references/validation-report).

---

## 3) Determinism: why hashing is a feature, not a nicety

In benchmark publishing, two properties matter:

1. **Immutability**: a snapshot ID maps to a fixed artifact set.
2. **Verifiability**: outsiders can confirm they got the same bytes you published.

That is why we invest in deterministic artifact generation:

- omit timestamps where possible (e.g., `generated_at: null` in `validation_report.json`),
- stable JSON key ordering,
- stable ordering for “set-like” arrays,
- fixture-driven PDF rendering for audit packs.

In CI we treat determinism as an invariant for the validation pack: re-rendering the same inputs must produce **bit-identical** JSON/PDF and an identical `validation_pack_manifest.json` (same SHA-256 hashes across reruns).

---

## 4) Why CI is the right publisher (and what CI does *not* solve)

Local benchmarks are useful, but they are not publication-grade evidence because:

- environment drift is invisible,
- cache state is inconsistent,
- operator steps are undocumented,
- results often aren’t indexed or preserved.

CI is a better publisher because:

- the harness is automated and versioned,
- snapshots are consistent and indexed,
- artifacts are attached immutably,
- reruns are “same script, same inputs”.

What CI does **not** solve:

- **hardware representativeness** (CI hardware may not match yours),
- **cross-platform variance** (macOS vs Linux, different BLAS, different GPU stacks).

That’s why a snapshot is not “the truth” — it’s **a reproducible experiment**.

---

## 5) Baselines: avoid “moving targets”

Baselines serve two roles:

1. regression detection (did we break performance?),
2. trend analysis (how does performance evolve?).

Baselines become meaningless if they move silently.

So we treat baselines as explicit, versioned references:

- “compare against snapshot X”,
- not “compare against whatever ran last week”.

In the public benchmarks repo skeleton, baseline manifests are versioned JSON documents with a schema. For example (abridged):

```json
{
  "schema_version": "nextstat.baseline_manifest.v1",
  "snapshot_id": "snapshot-2026-02-08",
  "deterministic": true,
  "harness": { "repo": "nextstat-public-benchmarks", "git_commit": "…" },
  "nextstat": { "version": "0.1.0", "wheel_sha256": "…" },
  "environment": { "python": "3.13.1", "platform": "Linux-6.8…" },
  "datasets": [{ "id": "hep/simple_workspace.json", "sha256": "…" }],
  "results": [{ "suite": "hep", "path": "out/hep_simple_nll.json", "sha256": "…" }]
}
```

The important detail is not the exact fields — it’s that the baseline is a **named, hashed reference**.

---

## 6) Indexing: make snapshots discoverable (and diffable)

An artifact set that cannot be discovered is not “public”.

We use a minimal “snapshot index” JSON format (schema `nextstat.snapshot_index.v1`) to link:

- suite name,
- git SHA/ref,
- workflow metadata,
- artifact paths + SHA-256 hashes.

This index is also the anchor for third-party replication: if you can’t identify *exactly* what was published, you can’t replicate it.

---

## 7) How an outsider verifies a snapshot (recipe)

A snapshot should be verifiable without trust in our blog post.

A minimal verification loop looks like:

1. Download the published artifact set (raw results + manifests).
2. Verify hashes (from `snapshot_index.json` or `validation_pack_manifest.json`).
3. Validate JSON schemas (e.g. `validation_report_v1`).
4. Rerun the harness on your machine (same suite + same dataset IDs).
5. Compare your rerun artifact hashes and/or semantic deltas to the original.

The benchmark program is designed so “verification” is mostly file operations, not interpretation.

---

## 8) DOI + `CITATION.cff`: when benchmarks become citable evidence

When a snapshot is stable enough to cite in a paper or technical report:

- we publish it with a DOI (e.g. Zenodo),
- we include `CITATION.cff` (machine-readable citation metadata),
- and we point the DOI to the full artifact set, not a screenshot table.

That’s not paperwork. It’s the difference between:

- a marketing claim, and
- a citable dataset.

---

## 9) Replication and signed reports: the strongest trust signal

The strongest trust signal is an independent rerun.

That’s why our replication protocol is a first-class part of the program:

- a third party reruns the harness,
- produces a replica snapshot index,
- computes a comparison,
- and publishes a signed replication report.

See:

- [Third-Party Replication Runbook](/docs/benchmarks/replication)
- [Third-Party Replication: Signed Reports](/blog/third-party-replication-signed-report)

---

## Closing: trust scales when artifacts scale

Public benchmarks are not a one-time launch. They are an ongoing program:

- new versions,
- new hardware,
- new suites,
- replications.

The only way to keep that trustworthy is to make the **artifacts** first-class — so every performance claim remains rerunnable evidence, not a screenshot.
