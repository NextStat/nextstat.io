# Фаза VII: Hierarchical / Multilevel Models (P0/P1)

> **Execution note (humans + AI agents):** Выполнять задачи последовательно. Эта фаза делает NextStat полезным для соцнаук/фармы/ML, где multilevel модели — ежедневная практика. Основной риск: стабильность NUTS и корректность параметризаций.

**Goal:** добавить иерархические модели (random intercept/slope), частичное пулингование, коррелированные эффекты, и безопасный sampling (NUTS) с проверками (SBC, diagnostics).

**Duration:** 8–14 недель

**Dependencies:** Phase V (generic inference) + Phase VI (GLM foundation желательно).

---

## Текущий статус (2026-02-06)

BMCP (эпик Phase 7: `1cd82a1a-6085-4a03-82bb-c44e3e83a340`):

- [ ] P7 7.1.1 Group-indexed effects (random intercept foundation) (`0cb69721-0422-4b30-9084-bfd4856872ab`)
- [ ] P7 7.1.2 GLM + random effects examples (logistic + linear) (`6e465141-04cf-422c-97d5-58fa6a02c856`)
- [ ] P7 7.2.1 Non-centered parameterization toolkit (`7f7d3110-8f3c-4eec-a394-daa527d6c168`)
- [ ] P7 7.3.1 Correlated random effects (LKJ + Cholesky) (`cfe3f482-6e48-4343-b42d-a8ce16466cfa`)
- [ ] P7 7.4.1 SBC for hierarchical models (Gaussian + Bernoulli) (`1191d9a5-95f2-47a5-b7b9-aaaa661dd682`)
- [ ] P7 7.4.2 Posterior predictive checks (PPC) + tutorials (`823ef0cb-a244-43f4-a82c-324d17dc65cc`)

Что уже готово и снимает блокировки (Phase 5/6):

- [x] Model builder (composition) MVP (`d9e63572-e62c-46c7-a66b-21a9c3192e53`)
- [x] Python data adapters (`X/y/group_idx`) (`66a46cde-3bc2-4070-b50b-4a3f749c78de`)
- [x] Generic NUTS multichain API + Python surface (`63d22c34-c406-497a-9f1a-8732abd2419e`)
- [x] Bijectors + Jacobians (`11a7ad77-2eaf-4511-929a-84f157f89748`)
- [x] Base distributions logpdf (`38174d2f-9c34-416c-a858-1b23c3c5fad3`)
- [x] GLM surfaces (Phase 6) для примеров: linear/logistic (`dfc8f82e-3b07-48ef-9a6c-427991cc5969`, `99082536-02a4-410b-a27b-94319702e5cf`)

## Acceptance Criteria (Фаза)

| Метрика | Target | Measurement |
|---------|--------|-------------|
| Random effects | есть и стабильно | varying intercept + slope на synthetic datasets |
| Parameterizations | non-centered support | non-centered и centered дают сопоставимые постериоры (SBC/diagnostics) |
| Correlations | LKJ/Cholesky | коррелированные эффекты корректны и тестируются |
| Sampling quality | минимальные гарантии | R-hat/ESS/divergences thresholds на эталонных моделях |

---

## Содержание

- [Sprint 7.1: Random intercept / slope](#sprint-71-random-intercept--slope-недели-1-3)
- [Sprint 7.2: Non-centered parameterization toolkit](#sprint-72-non-centered-parameterization-toolkit-недели-4-5)
- [Sprint 7.3: Correlated random effects (LKJ)](#sprint-73-correlated-random-effects-lkj-недели-6-8)
- [Sprint 7.4: Validation (SBC + PPC) и docs](#sprint-74-validation-sbc--ppc-и-docs-недели-9-12)
- [Критерии завершения фазы](#критерии-завершения-фазы)

---

## Sprint 7.1: Random intercept / slope (Недели 1-3)

### Epic 7.1.1: Group-indexed effects

**Acceptance Criteria:**
- [ ] модель принимает `group_idx` и параметры группы
- [ ] partial pooling через Normal prior на эффекты

### Epic 7.1.2: GLM + random effects

**Acceptance Criteria:**
- [ ] logistic regression + random intercept (классический пример соцнаук)
- [ ] linear regression + random intercept/slope (классический пример mixed models)

---

## Sprint 7.2: Non-centered parameterization toolkit (Недели 4-5)

### Epic 7.2.1: Reparameterization library

**Goal:** ввести стандартный способ описывать `theta = mu + sigma * z`, где `z ~ Normal(0,1)` и корректно учитывать jacobian (если требуется).

**Acceptance Criteria:**
- [ ] единый helper в model builder
- [ ] тесты: centered vs non-centered на synthetic (качество mixing)

---

## Sprint 7.3: Correlated random effects (LKJ) (Недели 6-8)

### Epic 7.3.1: LKJ prior + Cholesky

**Acceptance Criteria:**
- [ ] LKJ(eta) для correlation matrix
- [ ] Cholesky parametrization для стабильности
- [ ] тесты: positive-definite гарантии + finite-diff sanity

---

## Sprint 7.4: Validation (SBC + PPC) и docs (Недели 9-12)

### Epic 7.4.1: SBC для иерархических моделей

**Acceptance Criteria:**
- [ ] SBC на random intercept модели (Gaussian и Bernoulli)
- [ ] thresholds: divergences < 1%, R-hat < 1.01

### Epic 7.4.2: Posterior predictive checks (PPC)

**Acceptance Criteria:**
- [ ] PPC utilities в Python (генерация реплик, сравнение summary stats)
- [ ] минимум 2 примера ноутбука/туториала

---

## Критерии завершения фазы

Phase VII завершена когда:

1. [ ] random intercept/slope модели доступны в Python API
2. [ ] non-centered parameterization доступна и рекомендована
3. [ ] correlated random effects (LKJ) работают и тестируются
4. [ ] SBC + PPC базовые проверки проходят на эталонных задачах
