# NextStat Implementation Plans

These documents are executable plans for building NextStat: a high-performance statistical inference toolkit with pyhf JSON compatibility.

## Core Architecture Principle

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLEAN ARCHITECTURE                      │
│                                                                 │
│  Business logic (model + inference) must not depend on backends  │
│                                                                 │
│  GPU/CUDA/Metal are accelerators, not the core of the system     │
│                                                                 │
│  Most scientific cluster workloads are CPU-first                 │
└─────────────────────────────────────────────────────────────────┘
```

Priorities:

1. Correctness: results match pyhf within the Phase 1 contract
2. Clean architecture: trait-based abstraction, dependency inversion
3. CPU performance: Rayon, SIMD, and batch workflows
4. GPU acceleration: optional, as a performance layer

## Quick Navigation

| Document | Description | Priority |
|---|---|---|
| [Master Plan](./2026-02-05-nextstat-implementation-plan.md) | Cross-phase plan and structure | - |
| [Project Standards](./standards.md) | Numerical contract, determinism, precision policy | - |
| [Version Baseline](./versions.md) | Toolchain and dependency versions | - |
| [Phase 0: Infrastructure](./phase-0-infrastructure.md) | Repo, CI/CD, licensing | P0 |
| [Phase 1: MVP-alpha](./phase-1-mvp-alpha.md) | Core engine, pyhf compatibility | P0 |
| [Phase 2A: CPU Parallelism](./phase-2a-cpu-parallelism.md) | Rayon, SIMD, cluster workflows | P0 |
| [Phase 2B: Autodiff and Optimizers](./phase-2b-autodiff.md) | AD, gradients/Hessian, LBFGS(-B) | P0 |
| [Phase 2C: GPU Backends](./phase-2c-gpu-backends.md) | Metal, CUDA (optional) | P1 |
| [Phase 3: Production Ready](./phase-3-production.md) | Releases, docs, viz, validation | P1 |
| [Phase 4: Enterprise and SaaS](./phase-4-enterprise.md) | Audit, compliance, scale, hub, dashboard | P2 |

## Bias and Bayesian Notes

- Bias / coverage policy: `docs/plans/standards.md` and toy regression tests (`tests/python/test_bias_pulls.py`).
- Bayesian roadmap (NUTS/HMC): `docs/plans/phase-3-production.md` (and referenced standards).

## Timeline Overview

```
2026
├── Feb-Mar (Weeks 1-4)     Phase 0: Infrastructure
├── Apr-Jun (Weeks 5-12)    Phase 1: MVP-alpha Core Engine
├── Jul-Sep (Weeks 13-20)   Phase 2A: CPU Parallelism + Phase 2B: Autodiff
└── Oct-Dec (Weeks 21-28)   Phase 2C: GPU Backends (optional)

2027
├── Jan-Jun (Months 9-15)   Phase 3: Production Ready
└── Jul-Dec (Months 15-24)  Phase 4: Enterprise and SaaS
```

## Key Milestones

| Milestone | Target | Deliverables |
|---|---:|---|
| M1: First working fit | Month 3 | `nextstat fit` produces a correct mu_hat |
| M2: pyhf parity | Month 4 | Validation suite passes |
| M3: CPU speedup | Month 6 | CPU-parallel + batched toy fits |
| M4: GPU acceleration (optional) | Month 9 | Metal/CUDA backend with relaxed parity |
| M5: White paper | Month 9 | Publishable white paper draft |
| M6: First enterprise customer | Month 18 | Paid customer |
| M7: SaaS launch | Month 24 | Cloud service live |

## Architecture Summary

### Dependency Inversion

```
┌─────────────────────────────────────────────────────────────────┐
│                        HIGH-LEVEL LOGIC                          │
│                   (backend-agnostic inference)                    │
│                                                                 │
│  ns-inference: MLE, NUTS, Profile Likelihood                      │
│  - depends on abstractions only                                   │
│                                                                 │
│  ns-core: types, traits, errors                                   │
│                                                                 │
├────────────────────────────┬────────────────────────────────────┤
│                            │ implemented by                      │
│  ns-translate: pyhf model  │  ns-compute: kernels                │
│  ns-ad: autodiff           │  (optional) GPU backends            │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Architecture

| Problem | Resolution |
|---|---|
| Many users/clusters have no GPU | CPU path works everywhere |
| Reproducibility | Deterministic CPU mode is the contract baseline |
| Testability | Unit and parity tests run on CPU |
| New accelerators | Add implementations without rewriting inference logic |

## Tech Stack (Current Direction)

| Component | Technology | Rationale |
|---|---|---|
| Core | Rust 2024 | Performance + safety |
| Python | PyO3 + maturin | Ecosystem integration |
| CPU perf | Rayon + SIMD | Cluster-friendly, portable |
| GPU | CUDA + Metal | Cross-platform acceleration (optional) |
| CI/CD | GitHub Actions | Standard, reproducible |

## Acceptance Criteria (Global)

Numerical accuracy (Phase 1 contract; see `standards.md` for canonical tolerances):

- NLL parity vs pyhf for fixtures (deterministic mode)
- MLE bestfit and uncertainties within fixture tolerances

Quality:

- Strong test coverage for new code
- No clippy warnings
- Documentation for public APIs

## Working With These Plans

### For Developers

```bash
git checkout -b phase-0/infrastructure

# Work task-by-task using TDD:
# 1. read the task
# 2. write a failing test
# 3. implement the minimal fix
# 4. verify tests pass
# 5. commit
```

### For AI Agents

Guidelines:

- Work sequentially (task-by-task) and close checklists per epic.
- Do not change numerical tolerances casually; `docs/plans/standards.md` is the source of truth.
- If a plan references a missing file, create it (stub first, then fill) and fix the link.

