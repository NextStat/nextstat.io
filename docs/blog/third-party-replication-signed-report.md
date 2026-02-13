---
title: "Third-Party Replication: How External Reruns and Signed Reports Close the Benchmark Trust Gap"
slug: third-party-replication
description: "The strongest trust signal for software benchmarks is an independent rerun. Learn how NextStat's replication protocol works — same harness, published manifests, signed reports, and verifiable artifacts."
date: 2026-02-08
author: NextStat Team
status: draft
keywords:
  - benchmark replication
  - third-party validation
  - signed benchmark report
  - reproducible performance
  - scientific software trust
  - benchmark verification
  - GPG signed report
  - Sigstore verification
  - NextStat validation
category: trust
---

# Third-Party Replication: External Reruns + Signed Reports

**Trust Offensive series:** [Index](/blog/trust-offensive) · **Prev:** [Benchmark Snapshots as Products](/blog/benchmark-snapshots-ci-artifacts) · **Next:** [Building a Trustworthy HEP Benchmark Harness](/blog/hep-benchmark-harness)

If you’ve ever been burned by an “impressive benchmark”, you already know the problem:

benchmarks are not just measurements — they are *claims*.

And the only robust way to evaluate a claim is to replicate it.

That’s why our public benchmark program treats **third-party replication** as a first-class feature, not a nice-to-have.

Canonical specification for the benchmark program (including replication as a trust signal): [/docs/public-benchmarks](/docs/public-benchmarks).

---

## Abstract

The strongest trust signal for a benchmark is not “more charts”, “more machines”, or “more blog posts”. It is an **independent rerun**.

In NextStat we operationalize replication as a publishable artifact set with explicit contracts:

- a rerun **snapshot index** (hashed artifact inventory),
- a machine-readable **replication report** comparing original vs rerun,
- the rerun **validation pack** (including a `validation_report.json`), and
- optional **signatures** for integrity and attribution.

If you can’t map a claim to a concrete artifact set, you don’t have evidence — you have a story.

---

## 1) Why replication is different from “more benchmarks”

We can publish more machines, more suites, more graphs — and still fail the trust test.

Replication is qualitatively different because it adds:

- independent hardware
- independent operator errors (the realistic ones)
- independent scrutiny of the harness and assumptions

If a benchmark can’t survive an external rerun, it shouldn’t be used as evidence.

---

## 2) What we mean by “replication”

A replication is not “I ran something similar”.

At minimum it means:

- same suite definition
- same dataset IDs
- same harness version (or documented diffs)
- the same correctness gates still pass
- raw outputs and a baseline manifest are published

The goal is that disagreements become diagnosable:

- environment differences (compiler, BLAS, GPU driver)
- dataset drift
- harness changes
- or a bug

---

## 3) The replication artifact set (what gets published)

Replication only works if outsiders can identify *exactly* what was compared.

We therefore publish two small, machine-readable “index” documents alongside the raw results:

1. `snapshot_index.json` (schema `nextstat.snapshot_index.v1`) — the list of artifact paths + sizes + SHA-256 hashes.
2. `replication_report.json` (schema `nextstat.replication_report.v1`) — a structured comparison (overlap count + mismatches).

Additionally, each validation pack includes:

- `validation_report.json` (schema `validation_report_v1`) — dataset fingerprint + environment + suite pass/fail summary
- `validation_pack_manifest.json` — SHA-256 + sizes for the core pack files (used for signing and replication)

Schema examples live in-repo:

- `docs/specs/snapshot_index_v1.example.json`
- `docs/specs/replication_report_v1.example.json`
- `docs/specs/validation_report_v1.example.json`

---

## 4) Why signed reports

If replication reports matter, they must be attributable and tamper-resistant.

A signed report is a lightweight way to guarantee:

- who produced the report
- what snapshot it refers to
- that the published artifact hasn't been modified

The `validation_report.json` produced by [`nextstat validation-report`](/docs/validation-report) already includes SHA-256 hashes for both the workspace and the Apex2 master report. Adding a GPG or Sigstore signature to that JSON creates a complete chain: *data hash → validation result → signer identity*.

We don't need bureaucracy. We need integrity.

---

## 5) Step-by-step: a minimal replication loop

The minimal loop is:

1. Download the original snapshot artifacts (`snapshot_index.json`, `validation_pack_manifest.json`, `validation_report.json`).
2. Verify original signatures (if provided).
3. Rerun the suite to produce your own validation pack (`make validation-pack`).
4. Write your rerun `snapshot_index.json`.
5. Generate a `replication_report.json` comparing original vs rerun.
6. Sign and publish your replication bundle.

This is intentionally designed to be mostly file operations, not “trust our interpretation”.

---

## 6) What we will do with replications

Replications should not disappear in a comment thread.

We plan to:

- link replications directly from the snapshot index
- use replicated numbers in public claims
- prefer “rerun me” evidence over “trust us” language

---

## 7) The ask

If you care about reproducible scientific computing, the most valuable contribution is:

- rerun a published snapshot on your hardware
- publish your manifest + raw results
- sign the report

That’s how performance claims become community knowledge.

---

## Related reading

- [Trust Offensive: Public Benchmarks](/blog/trust-offensive)
- [The End of the Scripting Era](/blog/end-of-scripting-era)
- [Benchmark Snapshots as Products](/blog/benchmark-snapshots-ci-artifacts)
- [Public Benchmarks](/docs/public-benchmarks)
- [Validation Report Artifacts](/docs/validation-report)
