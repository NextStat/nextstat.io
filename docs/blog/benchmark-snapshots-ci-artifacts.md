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

If a benchmark result is not reproducible, it’s not a benchmark.

It’s a screenshot.

This post explains the publishing layer of our public benchmark program — the part that turns “we ran it once” into “anyone can rerun it”.

Canonical spec: [Publishing Benchmarks](/docs/benchmarks/publishing).

---

## 1. What we publish (a snapshot is an artifact set)

Each benchmark snapshot includes:

- raw per-test measurements
- aggregated summaries (tables/plots)
- a baseline manifest (versions, hardware, settings, dataset hashes)
- correctness gate reports (parity/sanity checks)

This is boring on purpose.

The point is not to impress — it’s to make the claim auditable.

---

## 2. Why CI is the right publisher

Local benchmarks are useful, but they’re not publication-grade evidence because:

- environment drift is invisible
- cache state is inconsistent
- operator steps are undocumented

CI runs are better because:

- the harness is automated
- snapshots are consistent and indexed
- artifacts can be attached immutably

---

## 3. Baselines: avoid “moving targets”

Baselines are essential for regression detection and trend analysis.

But baselines become meaningless if they move silently.

So we treat baselines as versioned, explicit references:

- “compare against snapshot X”
- not “compare against whatever ran last week”

---

## 4. DOI and citation: when benchmarks become citable evidence

When a snapshot is stable enough to cite:

- we publish it with a DOI (e.g., Zenodo),
- and include a machine-readable `CITATION.cff`.

That’s not paperwork — it’s the difference between:

- a blog claim, and
- a citable dataset.

---

## 5. The meta-point: trust scales when artifacts scale

Public benchmarks are not a one-time launch.

They are an ongoing program:

- new versions
- new hardware
- new suites
- external replications

The only way to keep that trustworthy is to make the *artifacts* first-class.
