# Фаза VI: Regression & GLM Pack (P0)

> **Execution note (humans + AI agents):** Выполнять задачи последовательно. Эта фаза опирается на Phase V (generic модели + `ns-prob`) и должна дать пользователям “первый успех” в общей статистике/ML: регрессии, предсказания, регуляризация, метрики качества.

**Goal:** предоставить production-usable regression/GLM слой в Python и Rust: линейная/логистическая/пуассоновская регрессии, NegBin, регуляризация, доверительные интервалы и предиктивные распределения.

**Duration:** 6–10 недель

**Dependencies:** Phase V.

---

## Текущий статус (2026-02-06)

BMCP (эпик Phase 6: `30fb7478-9489-4e79-810b-4b07ff4efa63`):

- [x] P6 6.1.1.1 Linear regression model + API (OLS baseline) (`dfc8f82e-3b07-48ef-9a6c-427991cc5969`)
- [x] P6 6.1.1.2 Linear regression tests + golden (statsmodels) (`6ff3aa08-32b4-4edf-bb9d-a374a3f74d45`)
- [x] P6 6.2.1 Logistic regression GLM + parity vs statsmodels (`99082536-02a4-410b-a27b-94319702e5cf`)
- [ ] P6 6.2.2 Robustness: separability + numeric stability (`b8792fe3-bd45-4290-b247-01d518e1eb57`)
- [x] P6 6.3.1 Poisson regression + offset/exposure (`15c58561-9b4f-40ce-b585-05a7114aa68f`)
- [x] P6 6.3.2 Negative binomial regression (mean/dispersion) (`b93a4d44-aa42-4573-b969-09f45adf5d36`)
- [x] P6 6.4.1 Regularization via MAP (ridge + optional lasso) (`65a576bf-b737-4c68-8366-cb9ee3f6e5eb`)
- [x] P6 6.4.2 CV utilities + metrics (Python) (`981a3f86-1629-4c43-93f2-2b0269d4cb4c`)
- [x] P6 Benchmarks: end-to-end fit/predict baselines (`d38990fe-07fe-43c8-b2a0-f2cb2557b421`)

Реализация (основные места в коде):
- Rust модели: `crates/ns-inference/src/regression.rs`
- Python surfaces: `bindings/ns-py/python/nextstat/glm/linear.py`, `bindings/ns-py/python/nextstat/glm/logistic.py`, `bindings/ns-py/python/nextstat/glm/poisson.py`
- Golden fixtures: `tests/fixtures/regression/*.json`, генератор: `tests/generate_golden_regression.py`
- Criterion бенчи (nll/grad): `crates/ns-inference/benches/regression_benchmark.rs`

## Acceptance Criteria (Фаза)

| Метрика | Target | Measurement |
|---------|--------|-------------|
| GLM API | стабильный Python surface | `nextstat.glm.*` + типы/доки |
| MLE parity | совпадает с эталоном | сравнение с statsmodels на golden datasets |
| Regularization | работает и тестируется | ridge/lasso: sanity + сравнение с sklearn где применимо |
| Predictive | корректные предсказания | posterior predictive (Bayes) + point predictions (MLE/MAP) |
| Benchmarks | измеримо | end-to-end `fit` и `predict` на synthetic datasets |

---

## Содержание

- [Sprint 6.1: Linear regression (Gaussian)](#sprint-61-linear-regression-gaussian-недели-1-2)
- [Sprint 6.2: Logistic regression (Bernoulli)](#sprint-62-logistic-regression-bernoulli-недели-3-4)
- [Sprint 6.3: Count models (Poisson/NegBin)](#sprint-63-count-models-poissonnegbin-недели-5-6)
- [Sprint 6.4: Regularization + model selection](#sprint-64-regularization--model-selection-недели-7-8)
- [Критерии завершения фазы](#критерии-завершения-фазы)

---

## Sprint 6.1: Linear regression (Gaussian) (Недели 1-2)

### Epic 6.1.1: Core model + closed-form baseline

#### Task 6.1.1.1: Реализовать `LinearRegressionModel`

**Priority:** P0  
**Effort:** 8–14 часов  
**Files (примерно):**
- Create: `crates/ns-models/src/glm/linear.rs` (или отдельный crate `ns-models`)
- Create: `bindings/ns-py/python/nextstat/glm.py` (или `nextstat/glm/linear.py`)

**Acceptance Criteria:**
- [x] `logpdf` и `grad` корректны
- [x] closed-form OLS доступен как reference path (для тестов и sanity)

#### Task 6.1.1.2: Tests + golden

**Priority:** P0  
**Acceptance Criteria:**
- [x] сравнение OLS closed-form vs MLE solver (одинаковый результат)
- [x] сравнение стандартных ошибок vs эталон (табличные значения)

---

## Sprint 6.2: Logistic regression (Bernoulli) (Недели 3-4)

### Epic 6.2.1: Logistic GLM

**Acceptance Criteria:**
- [x] MLE сходится на golden datasets
- [x] коэффициенты и SE совпадают со statsmodels в пределах tolerances
- [x] есть predict_proba и predict

### Epic 6.2.2: Robustness

**Acceptance Criteria:**
- [ ] обработка separability (предупреждение/регуляризация)
- [ ] numeric stability: log-sum-exp, clipping

---

## Sprint 6.3: Count models (Poisson/NegBin) (Недели 5-6)

### Epic 6.3.1: Poisson regression

**Acceptance Criteria:**
- [x] parity vs statsmodels (MLE coefficients)
- [x] exposure/offset поддержка (важно для эпидемиологии/соцнаук)

### Epic 6.3.2: Negative Binomial

**Acceptance Criteria:**
- [x] параметризация (mean/dispersion) зафиксирована и документирована
- [x] устойчивое фиттинг поведение на synthetic overdispersed datasets

---

## Sprint 6.4: Regularization + model selection (Недели 7-8)

### Epic 6.4.1: Ridge/Lasso (MAP)

**Goal:** трактуем регуляризацию как prior -> MAP fit.

**Acceptance Criteria:**
- [x] L2 prior (Normal) как MAP работает
- [ ] L1 prior (Laplace) — сначала как опция (можно без second-order)

### Epic 6.4.2: CV и метрики (Python)

**Acceptance Criteria:**
- [x] k-fold CV utilities (минимальные)
- [x] метрики: RMSE, log-loss, deviance

---

## Критерии завершения фазы

Phase VI завершена когда:

1. [x] Linear/Logistic/Poisson/NegBin regression доступны в Python API
2. [x] parity/golden tests против statsmodels покрывают минимум 3 модели
3. [x] регуляризация (ridge) и базовые метрики работают
4. [x] бенчмарки фиксируют baseline `fit`/`predict`

**Status: Phase VI COMPLETE** (все acceptance criteria выполнены, baseline записан через `tests/record_baseline.py`)
