---
title: "Third-Party Replication: How External Reruns and Signed Reports Close the Benchmark Trust Gap"
slug: third-party-replication-signed-report
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

If you’ve ever been burned by an “impressive benchmark”, you already know the problem:

benchmarks are not just measurements — they are *claims*.

And the only robust way to evaluate a claim is to replicate it.

That’s why our public benchmark program treats **third-party replication** as a first-class feature, not a nice-to-have.

The canonical replication protocol is documented here: [Publishing Benchmarks (CI, Artifacts, DOI, Replication)](/docs/benchmarks/publishing).

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

## 3) Why signed reports

If replication reports matter, they must be attributable and tamper-resistant.

A signed report is a lightweight way to guarantee:

- who produced the report
- what snapshot it refers to
- that the published artifact hasn't been modified

The `validation_report.json` produced by [`nextstat validation-report`](/docs/references/validation-report) already includes SHA-256 hashes for both the workspace and the Apex2 master report. Adding a GPG or Sigstore signature to that JSON creates a complete chain: *data hash → validation result → signer identity*.

We don't need bureaucracy. We need integrity.

---

## 4) What we will do with replications

Replications should not disappear in a comment thread.

We plan to:

- link replications directly from the snapshot index
- use replicated numbers in public claims
- prefer “rerun me” evidence over “trust us” language

---

## 5) The ask

If you care about reproducible scientific computing, the most valuable contribution is:

- rerun a published snapshot on your hardware
- publish your manifest + raw results
- sign the report

That’s how performance claims become community knowledge.

---

## Related reading

- [Trust Offensive: Public Benchmarks](/blog/trust-offensive-public-benchmarks) — why we publish reproducible benchmarks.
- [The End of the Scripting Era](/blog/end-of-scripting-era-benchmarks) — how benchmarking shifts from scripts to experiments.
- [Publishing Benchmarks (CI, Artifacts, DOI)](/docs/benchmarks/publishing) — CI automation, DOI minting, and baseline management.
- [Validation Report Artifacts](/docs/references/validation-report) — the `validation_report.json` + PDF system.
