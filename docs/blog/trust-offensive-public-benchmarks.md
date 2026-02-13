---
title: "Trust Offensive: Why We Publish Reproducible Benchmarks for Scientific Software"
slug: trust-offensive
description: "NextStat's public benchmark program treats performance as a scientific claim — with protocols, pinned environments, correctness gates, and artifacts anyone can rerun. Learn why reproducible benchmarks matter for HEP, pharma, and Bayesian inference."
date: 2026-02-08
author: NextStat Team
status: shipped
keywords:
  - reproducible benchmarks
  - scientific software performance
  - statistical computing benchmarks
  - pyhf comparison
  - HistFactory benchmark
  - NLME benchmark
  - NUTS ESS per second
  - benchmark reproducibility
  - open source validation
  - NextStat benchmarks
category: trust
---

# Trust Offensive: Public Benchmarks

## Abstract

Performance claims in scientific software are scientific claims: they should be reproducible from artifacts alone, not from narratives.

The core failure mode is not malice; it is ambiguity:

- input drift,
- warmup and cache drift,
- environment drift,
- and “correctness” drift (timing a different computation under the same name).

NextStat’s **Trust Offensive** treats benchmarks as experiments. Each published benchmark snapshot is defined by:

- a protocol (task + metrics + warmup/repeat policy),
- a pinned environment,
- correctness gates,
- and a minimal set of auditable artifacts with explicit schemas.

Canonical specification (protocol + artifacts): [/docs/public-benchmarks](/docs/public-benchmarks).

---

## Series (recommended reading order)

1. **Trust Offensive: Public Benchmarks** (this post)
2. [The End of the Scripting Era](/blog/end-of-scripting-era)
3. [Benchmark Snapshots as Products](/blog/benchmark-snapshots-ci-artifacts)
4. [Third-Party Replication: Signed Reports](/blog/third-party-replication)
5. [Building a Trustworthy HEP Benchmark Harness](/blog/hep-benchmark-harness)
6. [Numerical Accuracy](/blog/numerical-accuracy)
7. [Differentiable HistFactory in PyTorch](/blog/differentiable-layer)
8. [Bayesian Benchmarks: ESS/sec](/blog/bayesian-benchmarks-ess-per-sec)
9. [Pharma Benchmarks: PK/NLME](/blog/pharma-benchmarks-pk-nlme)
10. [JAX Compile vs Execution](/blog/jax-compile-vs-execution)
11. [Unbinned Event-Level Analysis](/blog/unbinned-event-level-analysis)
12. [Compiler-Symbolic vs Hybrid-Neural GPU Fits](/blog/compiler-vs-hybrid-gpu-fits)

---

## 1. Definitions (what the artifacts mean)

We use the following terms with strict meanings.

- **Benchmark snapshot**: a uniquely identified run (typically CI) with immutable artifacts.
- **Protocol**: task definition + metrics + warmup and repeat policy.
- **Correctness gate**: a pass/fail test that must pass before timing is accepted.
- **Baseline manifest**: a machine-readable description of code SHA, environment versions, dataset hashes, and run configuration.
- **Validation report**: a unified JSON+PDF pack summarizing correctness evidence and environment metadata.
- **Replication report**: a structured comparison between an original snapshot and an external rerun.

These are not prose concepts; they correspond to concrete artifacts and schemas.

---

## 2. Scope (what we benchmark)

We benchmark **end-to-end workflows** users actually run (not just micro-kernels):

- **HEP / HistFactory**: NLL evaluation, gradients, MLE fits, profile scans, toy ensembles.
- **Pharma**: PK/NLME likelihood evaluation + fitting loops.
- **Bayesian**: ESS/sec and wall-time under explicitly defined inference settings.
- **ML infra**: compile latency vs steady-state throughput where it dominates total cost.
- **Time series**: Kalman throughput and EM convergence cost.
- **Econometrics**: panel FE / DiD / IV scaling and estimator-level tradeoffs.

Microbenchmarks (Criterion) still exist, but they are treated as regression detectors rather than headline claims.

**Non-goals:** hero runs on one cherry-picked machine, unpublished harness scripts, performance numbers without correctness gates.

---

## 3. Trust model (what must be verifiable)

For every published snapshot you should be able to answer, from artifacts alone:

1. **What was measured?** (task + metric definitions)
2. **On what inputs?** (dataset identity + SHA-256)
3. **From what code?** (NextStat git SHA + lockfiles)
4. **Under what environment?** (OS/CPU/GPU + toolchains + key dependencies)
5. **Does it still match reference?** (parity checks before timing)
6. **How stable is the number?** (repeat distributions + aggregation policy)

This is the minimal bar for turning “fast” into evidence.

---

## 4. Reproducibility contract (protocol invariants)

### A. Correctness gating before timing

Timing is invalid if the computation is not the same computation.

Rule: before recording performance numbers, the harness must validate that outputs are sane and, when applicable, agree with a reference implementation within a stated tolerance.

The formal output is a validation pack produced by `nextstat validation-report`.

Canonical reference: [/docs/validation-report](/docs/validation-report).

### B. Environment pinning

Published snapshots must include enough information to reconstruct the build and runtime environment:

- Rust toolchain (`rust-toolchain.toml`) and dependency lock (`Cargo.lock`).
- Python version and dependency resolution (when Python harnesses are involved).
- GPU runtime details when used (CUDA/driver or Metal).

### C. Warmup and caches

Warmup policy is part of the protocol. Each benchmark must state:

- what counts as warmup,
- what is excluded (e.g. compilation vs execution),
- and what cache state is assumed.

### D. Reporting distributions, not a single number

Single numbers hide variance.

Rule: publish raw per-test measurements and state the aggregation policy explicitly (median, min-of-N, etc.).

---

## 5. What we publish (artifacts and schemas)

Every snapshot is a directory of immutable artifacts. The minimal set is:

### 5.1 Validation report pack (`validation_report_v1`)

The unified validation pack is generated via:

```bash
nextstat validation-report \
  --apex2 tmp/apex2_master_report.json \
  --workspace workspace.json \
  --out validation_report.json \
  --pdf validation_report.pdf \
  --deterministic
```

It produces:

- `validation_report.json` (schema version `validation_report_v1`)
- `validation_report.pdf` (optional)
- and a manifest with SHA-256 hashes (for signing and replication)

### 5.2 Snapshot index (`nextstat.snapshot_index.v1`)

Each published snapshot includes a machine-readable index describing:

- `snapshot_id`
- git SHA/ref
- CI metadata
- a list of artifacts with `bytes` and `sha256`

Schema example: `docs/specs/snapshot_index_v1.example.json`.

### 5.3 Replication report (`nextstat.replication_report.v1`)

External reruns close the trust gap by comparing digests of the same artifact set.
The comparison is recorded as a replication report with explicit mismatches.

Schema example: `docs/specs/replication_report_v1.example.json`.

---

## 6. The ask: rerun it

Public benchmarks only work if other people rerun them.

If you rerun a snapshot on your hardware, publish:

- your `snapshot_index.json` (with artifact hashes),
- your `validation_report.json`,
- and (optionally) a `replication_report.json` comparing your rerun to the published snapshot.

That is how benchmark numbers become durable.
