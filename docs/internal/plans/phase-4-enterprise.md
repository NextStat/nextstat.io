# Фаза IV: Enterprise & SaaS (P2 / Pro)

> **Execution note (humans + AI agents):** Выполнять задачи последовательно. Каноничные определения/допуски/детерминизм: `docs/plans/standards.md`.
>
> **Важно:** большая часть Phase 4 — это **NextStat Pro** (commercial). Реализация может вестись в private repo, но публичные интерфейсы/контракты должны быть согласованы с OSS ядром.

**Goal:** Добавить enterprise-ценность вокруг ядра: аудит/комплаенс, масштабирование, модельный реестр, UI/операционная оболочка и (опционально) SaaS.

**Duration:** Месяцы 15-24

**Dependencies:** Phase 3 (stable releases + docs + validation), legal/governance decisions (`docs/legal/open-core-boundaries.md`).

**Tech Stack (предпочтительно):** Rust core, Postgres (audit/registry), OpenTelemetry, Kubernetes, Ray/HTCondor adapters, Next.js dashboard, SSO (OIDC/SAML).

---

## Содержание

- [Acceptance Criteria (Фаза)](#acceptance-criteria-фаза)
- [Sprint 4.1: NS-Audit (21 CFR Part 11)](#sprint-41-ns-audit-21-cfr-part-11-недели-61-68)
- [Sprint 4.2: NS-Compliance packs](#sprint-42-ns-compliance-packs-недели-69-72)
- [Sprint 4.3: NS-Scale (distributed fits)](#sprint-43-ns-scale-distributed-fits-недели-73-80)
- [Sprint 4.4: NS-Hub (model registry)](#sprint-44-ns-hub-model-registry-недели-81-88)
- [Sprint 4.5: NS-Dashboard](#sprint-45-ns-dashboard-недели-89-96)
- [Критерии завершения фазы](#критерии-завершения-фазы)

---

## Acceptance Criteria (Фаза)

| Метрика | Target | Measurement |
|---------|--------|-------------|
| Audit trail | immutable + queryable | verified event log + export |
| Signatures | e-sign + approvals | signed run artifacts |
| Registry | versioned models/runs | CRUD + RBAC + retention |
| Scale | distributed fits | 10k fits / day (pilot) |
| Dashboard | ops UI | run status + artifacts + alerts |

---

## Sprint 4.1: NS-Audit (21 CFR Part 11) (Недели 61-68)

### Epic 4.1.1: Audit trail primitives (Pro)

#### Task 4.1.1.1: Audit event schema + append-only storage

**Priority:** P0  
**Effort:** 12-18 часов  
**Dependencies:** Phase 3 outputs (stable FitResult + metadata)

**Deliverables:**
- [ ] Canonical audit event schema (JSON)
- [ ] Append-only store interface (Postgres table or object store log)
- [ ] Export (CSV/JSON) + retention policy

**Notes (non-legal):**
- Реализация и требования 21 CFR Part 11 должны быть подтверждены специалистом по compliance.

#### Task 4.1.1.2: Run artifact signing (hash + signature)

**Priority:** P0  
**Effort:** 8-12 часов  
**Dependencies:** Task 4.1.1.1

**Acceptance Criteria:**
- [ ] Каждый run имеет immutable `run_id` и content-hash
- [ ] Подпись хранится рядом с артефактами и проверяется

---

## Sprint 4.2: NS-Compliance packs (Недели 69-72)

### Epic 4.2.1: Domain report generators (Pro)

#### Task 4.2.1.1: Compliance report skeletons

**Priority:** P1  
**Effort:** 10-16 часов  
**Dependencies:** Task 4.1.1.1

**Acceptance Criteria:**
- [ ] Report templates (Basel/IFRS / CFR) генерируются из audit log + run artifacts
- [ ] Версионирование шаблонов

---

## Sprint 4.3: NS-Scale (distributed fits) (Недели 73-80)

### Epic 4.3.1: Execution orchestration (Pro)

#### Task 4.3.1.1: Job submission abstraction (local/SLURM/K8s)

**Priority:** P0  
**Effort:** 16-24 часа  
**Dependencies:** Phase 2A job arrays + deterministic RNG

**Acceptance Criteria:**
- [ ] Unified “submit” API: `nextstat submit ...`
- [ ] Deterministic job partitioning + resumability
- [ ] Artifacts collected back into registry/audit

#### Task 4.3.1.2: Ray adapter (optional)

**Priority:** P2  
**Effort:** 12-18 часов  
**Dependencies:** Task 4.3.1.1

---

## Sprint 4.4: NS-Hub (model registry) (Недели 81-88)

### Epic 4.4.1: Registry + RBAC (Pro)

#### Task 4.4.1.1: Model + dataset registry schema

**Priority:** P0  
**Effort:** 12-18 часов  
**Dependencies:** Phase 3 release discipline

**Acceptance Criteria:**
- [ ] Model versions immutable (content-addressed or semver)
- [ ] Runs reference exact model/dataset versions
- [ ] Basic RBAC roles (admin/editor/viewer)

---

## Sprint 4.5: NS-Dashboard (Недели 89-96)

### Epic 4.5.1: Ops UI (Pro)

#### Task 4.5.1.1: Dashboard MVP (runs + artifacts + alerts)

**Priority:** P1  
**Effort:** 24-40 часов  
**Dependencies:** registry + audit primitives

**Acceptance Criteria:**
- [ ] Runs list + status + filters
- [ ] Artifacts browser (plots/logs/results)
- [ ] Alerting hooks (webhook/email) for failed runs/regressions

---

## Критерии завершения фазы

Phase 4 завершена когда:

1. [ ] Audit trail + artifact signing работают end-to-end (P0)
2. [ ] Distributed submission + results collection поддерживает пилотный workload (P0)
3. [ ] Registry хранит версии моделей/ранов с RBAC (P0)
4. [ ] Dashboard MVP доступен для pilot пользователей (P1)

