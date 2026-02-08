<!--
  Blog draft (trust-building).
  Canonical replication spec lives in: docs/benchmarks/publishing.md
-->

# Third-Party Replication: External Reruns + Signed Reports

**Last updated:** 2026-02-08  
**Status:** Blog draft (technical)

If you’ve ever been burned by an “impressive benchmark”, you already know the problem:

benchmarks are not just measurements — they are *claims*.

And the only robust way to evaluate a claim is to replicate it.

That’s why our public benchmark program treats **third-party replication** as a first-class feature, not a nice-to-have.

The canonical replication protocol is documented here: `docs/benchmarks/publishing.md`.

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
- that the published artifact hasn’t been modified

We don’t need bureaucracy. We need integrity.

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

