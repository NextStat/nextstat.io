# Фаза VIII: Time Series & State Space (P1)

> **Execution note (humans + AI agents):** Выполнять задачи последовательно. Для временных рядов важны: корректность likelihood, стабильность фильтрации/сглаживания, прогнозирование, и скорость.

**Goal:** добавить time-series модели, которые реально используются в статистике/соцнауках/фарме: AR(1)/ARMA, local level/trend (state space), базовый Kalman filter + MLE/MAP, и posterior predictive forecasting.

**Duration:** 10–16 недель

**Dependencies:** Phase V (generic core), желательно Phase VII (иерархия, если хотим hierarchical time series).

---

## Текущий статус (2026-02-06)

BMCP (эпик Phase 8: `264e96f5-0389-42f3-9d66-a0fb589322e5`):

- [x] P8 8.1.1 Kalman filter/smoother + log-likelihood (`c633d279-75fb-40c6-b7a1-58eb2364a452`)
- [x] P8 8.1.2 Parameterization/transforms for variances and bounds (`34ec5e33-3aa2-45ef-a0e3-9b99fa42fc16`)
- [x] P8 8.2.1 AR(1) via state-space + parity vs statsmodels (`2b0c356f-4689-4ce3-87d5-fbf828197fd9`)
- [x] P8 8.2.2 ARMA(1,1) baseline (optional) (`ae5df9f4-426b-43f2-a175-6b8f45b883c6`)
- [x] P8 8.3.1 Forecast API + prediction intervals (+PPC) (`4e07c134-fdf5-4560-847d-3f8e434a5428`)
- [x] P8 8.4.1 Missing data policy for gaps (likelihood + tests) (`c0cf8ef8-6cce-4028-8793-e4c920337ec5`)
- [x] P8 8.4.2 Seasonality component (Fourier/seasonal states) (`9dc044e0-bfb1-4cb4-8887-a66aba38ddfd`)

Реализация (основные места в коде):
- Rust: `crates/ns-inference/src/timeseries/kalman.rs`

## Acceptance Criteria (Фаза)

| Метрика | Target | Measurement |
|---------|--------|-------------|
| Kalman baseline | корректность | сравнение log-likelihood vs statsmodels state-space на golden series |
| Forecasting | работает | out-of-sample forecasts + prediction intervals |
| MLE/MAP | устойчивость | конвергенция на synthetic series, reasonable params |
| Performance | измеримо | бенчмарки фильтрации/fit по длине ряда |

---

## Содержание

- [Sprint 8.1: Gaussian state-space core](#sprint-81-gaussian-state-space-core-недели-1-4)
- [Sprint 8.2: AR(1)/ARMA как state-space](#sprint-82-ar1arma-как-state-space-недели-5-7)
- [Sprint 8.3: Forecasting + PPC](#sprint-83-forecasting--ppc-недели-8-10)
- [Sprint 8.4: Расширения (seasonality, missing data)](#sprint-84-расширения-seasonality-missing-data-недели-11-14)
- [Критерии завершения фазы](#критерии-завершения-фазы)

---

## Sprint 8.1: Gaussian state-space core (Недели 1-4)

### Epic 8.1.1: Kalman filter / smoother

**Acceptance Criteria:**
- [x] implement filter + log-likelihood accumulation
- [x] numerical stability (Joseph form / square-root optional later)
- [x] unit tests on tiny known sequences

### Epic 8.1.2: Parameterization and bounds

**Acceptance Criteria:**
- [x] positive variances handled via transforms (exp/softplus)
- [x] deterministic results with fixed seeds

---

## Sprint 8.2: AR(1)/ARMA как state-space (Недели 5-7)

### Epic 8.2.1: AR(1) model

**Acceptance Criteria:**
- [x] AR(1) available as state-space model (builder + EM fit for 1D `F[0,0]` optional)
- [x] parity vs statsmodels Kalman filter on golden fixtures

### Epic 8.2.2: ARMA(p,q) (optional baseline)

**Acceptance Criteria:**
- [x] start with ARMA(1,1) minimal
- [x] docs state limitations clearly

---

## Sprint 8.3: Forecasting + PPC (Недели 8-10)

### Epic 8.3.1: Forecast API

**Acceptance Criteria:**
- [x] forecast mean + intervals (analytic for Gaussian)
- [ ] posterior predictive forecasting when sampling is used

---

## Sprint 8.4: Расширения (seasonality, missing data) (Недели 11-14)

### Epic 8.4.1: Missing data policy

**Acceptance Criteria:**
- [x] missing observations supported in likelihood (skip/update)
- [x] tests for gaps

### Epic 8.4.2: Seasonality (P2)

**Acceptance Criteria:**
- [x] simple seasonal component (Fourier or seasonal states)

---

## Критерии завершения фазы

Phase VIII завершена когда:

1. [x] Gaussian state-space core (Kalman) работает и тестируется
2. [x] AR(1) fit + forecasting доступны в Python API
3. [x] golden tests против statsmodels проходят
4. [x] бенчмарки для фильтрации/fit фиксируют baseline
