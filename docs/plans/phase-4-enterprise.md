# Phase 4: Enterprise and SaaS (P2 / Pro)

Execution note (humans and AI agents): work sequentially. Canonical definitions, tolerances, and determinism are defined in `docs/plans/standards.md`.

Important: most of Phase 4 is NextStat Pro (commercial). Implementation may live in a private repo, but public interfaces and contracts must stay aligned with the OSS core.

Goal: add enterprise value around the core engine: audit/compliance, scaling, model registry, operational UI, and (optionally) a SaaS offering.

Duration: Months 15-24

Dependencies: Phase 3 (stable releases, docs, validation), and legal/governance decisions (`docs/legal/open-core-boundaries.md`).

Suggested tech stack: Rust core, Postgres (audit/registry), OpenTelemetry, Kubernetes, Ray/HTCondor adapters, Next.js dashboard, SSO (OIDC/SAML).

## Contents

- [Acceptance Criteria (Phase)](#acceptance-criteria-phase)
- [Sprint 4.1: NS-Audit (21 CFR Part 11)](#sprint-41-ns-audit-21-cfr-part-11-weeks-61-68)
- [Sprint 4.2: NS-Compliance packs](#sprint-42-ns-compliance-packs-weeks-69-72)
- [Sprint 4.3: NS-Scale (distributed fits)](#sprint-43-ns-scale-distributed-fits-weeks-73-80)
- [Sprint 4.4: NS-Hub (model registry)](#sprint-44-ns-hub-model-registry-weeks-81-88)
- [Sprint 4.5: NS-Dashboard](#sprint-45-ns-dashboard-weeks-89-96)
- [Phase Exit Criteria](#phase-exit-criteria)

## Acceptance Criteria (Phase)

| Metric | Target | Measurement |
|---|---|---|
| Audit trail | immutable + queryable | verified event log + export |
| Signatures | e-sign + approvals | signed run artifacts |
| Registry | versioned models/runs | CRUD + RBAC + retention |
| Scale | distributed fits | 10k fits/day (pilot) |
| Dashboard | ops UI | run status + artifacts + alerts |

## Sprint 4.1: NS-Audit (21 CFR Part 11) (Weeks 61-68)

### Epic 4.1.1: Audit trail primitives (Pro)

#### Task 4.1.1.1: Audit event schema + append-only storage

Priority: P0  
Effort: 12-18 hours  
Dependencies: Phase 3 outputs (stable FitResult + metadata)

Deliverables:

- Canonical audit event schema (JSON)
- Append-only store interface (Postgres table or object-store log)
- Export (CSV/JSON) + retention policy

Notes:

- 21 CFR Part 11 requirements must be confirmed with compliance experts.

#### Task 4.1.1.2: Run artifact signing (hash + signature)

Priority: P0  
Effort: 8-12 hours  
Dependencies: Task 4.1.1.1

Acceptance criteria:

- Each run has an immutable `run_id` and a content hash
- Signatures are stored next to artifacts and are verifiable

## Sprint 4.2: NS-Compliance packs (Weeks 69-72)

### Epic 4.2.1: Domain report generators (Pro)

#### Task 4.2.1.1: Compliance report skeletons

Priority: P1  
Effort: 10-16 hours  
Dependencies: Task 4.1.1.1

Acceptance criteria:

- Report templates (e.g., regulatory formats) are generated from audit logs + run artifacts
- Template versioning is supported

## Sprint 4.3: NS-Scale (distributed fits) (Weeks 73-80)

### Epic 4.3.1: Execution orchestration (Pro)

#### Task 4.3.1.1: Job submission abstraction (local/SLURM/K8s)

Priority: P0  
Effort: 16-24 hours  
Dependencies: Phase 2A job arrays + deterministic RNG

Acceptance criteria:

- Unified submit API (e.g., `nextstat submit ...`)
- Deterministic partitioning + resumability
- Artifacts are collected back into the registry/audit trail

#### Task 4.3.1.2: Ray adapter (optional)

Priority: P2  
Effort: 12-18 hours  
Dependencies: Task 4.3.1.1

## Sprint 4.4: NS-Hub (model registry) (Weeks 81-88)

### Epic 4.4.1: Registry + RBAC (Pro)

#### Task 4.4.1.1: Model + dataset registry schema

Priority: P0  
Effort: 12-18 hours  
Dependencies: Phase 3 release discipline

Acceptance criteria:

- Model versions are immutable (content-addressed or semver)
- Runs reference exact model/dataset versions
- Basic RBAC roles exist (admin/editor/viewer)

## Sprint 4.5: NS-Dashboard (Weeks 89-96)

### Epic 4.5.1: Ops UI (Pro)

#### Task 4.5.1.1: Dashboard MVP (runs + artifacts + alerts)

Priority: P1  
Effort: 24-40 hours  
Dependencies: registry + audit primitives

Acceptance criteria:

- Runs list + status + filters
- Artifact browser (plots/logs/results)
- Alerting hooks (webhook/email) for failed runs and regressions

## Phase Exit Criteria

Phase 4 is complete when:

1. Audit trail + artifact signing works end-to-end (P0)
2. Distributed submission + results collection supports a pilot workload (P0)
3. Registry stores versioned models/runs with RBAC (P0)
4. Dashboard MVP is available to pilot users (P1)

