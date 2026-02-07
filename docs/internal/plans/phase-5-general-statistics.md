# Фаза V: General Statistics Core (P0)

> **Execution note (humans + AI agents):** Выполнять задачи последовательно. Мы расширяем NextStat за пределы HEP/HistFactory, не ломая существующие HEP workflow. Публичные документы на английском, планы остаются внутренними.

**Goal:** сделать NextStat универсальным инструментом для статистики и probabilistic modeling (регрессии, иерархические модели, временные ряды) с частыми (MLE/MAP) и байесовскими (NUTS) методами, плюс воспроизводимость, валидация и удобный Python API.

**Duration:** 8–12 недель (после стабилизации Phase 3).

**Dependencies:** Phase 1–3 (pyhf ingestion + MLE + NUTS + diagnostics), Phase 2B (AD/градиенты), стабильные Python bindings.

**Tech Stack:** Rust, nalgebra/ndarray, rand/rand_distr, (опционально) serde для сериализации моделей; Python: numpy/pandas, arviz/xarray как потребитель формата draws.

---

## 0) Принципы и границы (важно зафиксировать)

1. **HEP baseline остаётся.** `HistFactoryModel` и pyhf parity не должны деградировать.
2. **Один общий inference engine.** Новые модели подключаются через общий trait/интерфейс, а не через копипасты `fit_*`.
3. **Детерминизм и воспроизводимость по умолчанию.** Для sampling и toys: seed обязателен и влияет на всё.
4. **Никаких "магических" исправлений статистики.** Любые bias/coverage "улучшения" только как явный режим и с отчётом.
5. **Фокус на минимальном ядре.** Мы не пытаемся "переписать Stan". Мы закрываем 80/20: GLM + multilevel + базовый state-space.

Decision points (не блокируют план, но влияют на приоритеты):
- Что важнее в MVP для новых доменов: **frequentist (GLM/MAP/Laplace)** или **Bayesian (NUTS + PPC)**?
- Разрешаем ли зависимости в Python API уровня `pandas`/`arviz` (скорее да), но держим ядро (Rust) независимым.

---

## 1) Что должно появиться как продукт

- **Универсальный интерфейс модели**: параметры (имена/границы), `logpdf` и `grad` (если доступен), симуляция данных (опционально), предиктивные функции.
- **Библиотека распределений и трансформаций** (общая): Normal, Student-t, Bernoulli/Binomial, Poisson, NegBin, Gamma/LogNormal, Beta/Dirichlet и стандартные bijectors.
- **Единый inference слой** для любых моделей:
  - MLE/MAP (bounded)
  - NUTS/HMC sampling (уже есть, но сейчас завязан на HistFactory)
  - диагностики (R-hat/ESS, divergences)
- **Единый формат результатов** (Rust + Python):
  - FitResult (frequentist)
  - SamplerResult (Bayesian) + ArviZ-friendly output
- **Validation harness**:
  - golden tests для базовых моделей vs референсы
  - Simulation-Based Calibration (SBC) для NUTS на эталонных задачах

---

## 2) Acceptance Criteria (Фаза)

| Метрика | Target | Measurement |
|---------|--------|-------------|
| Generic model interface | есть и используется | MLE+NUTS принимают generic model, HistFactory — один из impl |
| Regression baseline | сходится с эталоном | linear/logistic/poisson regression: сравнение с statsmodels/Stan на golden datasets |
| Hierarchical toy | корректность sampling | SBC pass на 1–2 простых иерархических моделях (R-hat, divergences) |
| Reproducibility | deterministic seeds | одинаковый seed -> одинаковые draws/fit при фикс. потоках |
| Perf hygiene | есть бенчи | criterion benches для logpdf/grad и end-to-end fit/sampling |

---

## 3) Содержание

- [Sprint 5.1: Core traits + generic inference](#sprint-51-core-traits--generic-inference-недели-1-2)
- [Sprint 5.2: `ns-prob` (распределения + bijectors)](#sprint-52-ns-prob-распределения--bijectors-недели-3-4)
- [Sprint 5.3: Model builder (composition)](#sprint-53-model-builder-composition-недели-5-7)
- [Sprint 5.4: Validation harness (golden + SBC)](#sprint-54-validation-harness-golden--sbc-недели-8-10)
- [Критерии завершения фазы](#критерии-завершения-фазы)

---

## Sprint 5.1: Core traits + generic inference (Недели 1-2)

### Epic 5.1.1: Универсальный интерфейс модели (Rust)

**Goal:** расширить `ns-core` так, чтобы inference не зависел от `HistFactoryModel`.

#### Task 5.1.1.1: Новый trait для log-density модели

**Priority:** P0  
**Effort:** 6–10 часов  
**Files:**
- Modify: `crates/ns-core/src/traits.rs`
- Modify: `crates/ns-inference/src/mle.rs`
- Modify: `crates/ns-inference/src/posterior.rs`

**Acceptance Criteria:**
- [ ] Есть trait типа `LogDensityModel` (имя можно выбрать), который включает:
  - параметры (имена/границы)
  - `logpdf(theta)` или `nll(theta)`
  - `grad(theta)` (обязателен для NUTS; для MLE допускается fallback на finite-diff как временный режим)
- [ ] `HistFactoryModel` адаптирован (через impl или thin adapter).
- [ ] `MaximumLikelihoodEstimator` и `Posterior` используют trait, а не конкретный тип.

**Implementation sketch (псевдо-API):**

```rust
pub trait LogDensityModel: Send + Sync {
    fn dim(&self) -> usize;
    fn parameter_names(&self) -> Vec<String>;
    fn bounds(&self) -> Vec<(f64, f64)>;
    fn logpdf(&self, theta: &[f64]) -> ns_core::Result<f64>;
    fn grad_logpdf(&self, theta: &[f64]) -> ns_core::Result<Vec<f64>>;
}
```

#### Task 5.1.1.2: Вынести bijectors/границы в общий слой

**Priority:** P0  
**Effort:** 4–8 часов  
**Files:**
- Modify: `crates/ns-inference/src/transforms.rs` (или вынести в `ns-core`)

**Acceptance Criteria:**
- [ ] Transform слой не зависит от HistFactory и может применяться к любым bounds.
- [ ] Тесты: roundtrip + grad log|J| для bijectors.

### Epic 5.1.2: Generic sampling API (Rust + Python)

#### Task 5.1.2.1: Generalize `sample_nuts_multichain` на trait

**Priority:** P0  
**Effort:** 6–12 часов  
**Files:**
- Modify: `crates/ns-inference/src/chain.rs`
- Modify: `crates/ns-inference/src/nuts.rs`
- Modify: `bindings/ns-py/src/lib.rs`

**Acceptance Criteria:**
- [ ] `sample(model, ...)` работает для HistFactory через адаптер.
- [ ] API не зависит от HEP-специфики (observed data setter как optional extension).

---

## Sprint 5.2: `ns-prob` (распределения + bijectors) (Недели 3-4)

### Epic 5.2.1: Базовые распределения

**Goal:** обеспечить общий набор распределений для regression/hierarchical/time-series.

#### Task 5.2.1.1: Добавить crate `ns-prob`

**Priority:** P0  
**Effort:** 4–8 часов  
**Files:**
- Create: `crates/ns-prob/Cargo.toml`
- Create: `crates/ns-prob/src/lib.rs`

**Acceptance Criteria:**
- [ ] `ns-prob` компилируется и имеет базовые `logpdf` функции (scalar + vector).

#### Task 5.2.1.2: Реализовать logpdf для набора распределений

**Priority:** P0  
**Effort:** 12–20 часов  
**Acceptance Criteria:**
- [ ] Normal, Student-t
- [ ] Bernoulli/Binomial (logit parametrization helper)
- [ ] Poisson, Negative Binomial
- [ ] Gamma/LogNormal (минимум для positive reals)
- [ ] Unit tests vs known values (табличные) + property tests (symmetry, monotonicity where applicable)

### Epic 5.2.2: Bijectors и Jacobians (общие)

**Goal:** единая библиотека трансформаций для bounded параметров + jacobians.

**Acceptance Criteria:**
- [ ] Identity, Exp, Softplus, Sigmoid, Affine+Sigmoid for (a,b)
- [ ] log|J| и grad(log|J|) корректны (finite diff tests)

---

## Sprint 5.3: Model builder (composition) (Недели 5-7)

### Epic 5.3.1: Композиция log-density

**Goal:** дать возможность собирать модели из блоков без ручного написания `logpdf` каждый раз.

Подход (минимальный):
- builder хранит параметры + их transforms
- user-defined deterministic nodes (линейный предиктор)
- likelihood блоки (Normal/Bernoulli/Poisson) + priors

**Acceptance Criteria:**
- [ ] Можно собрать linear regression как модель из блоков
- [ ] Можно собрать logistic regression
- [ ] Можно добавить random intercept (подготовка к Phase 7)

### Epic 5.3.2: Data adapters (Python-first)

**Goal:** минимальная поддержка design matrix и индексации групп.

**Acceptance Criteria:**
- [ ] Python API принимает `y`, `X`, `group_idx` и строит model object
- [ ] Сериализация/репродукция: модель + данные можно сохранить/восстановить (минимум для тестов)

---

## Sprint 5.4: Validation harness (golden + SBC) (Недели 8-10)

### Epic 5.4.1: Golden problems (frequentist)

**Goal:** зафиксировать эталонные результаты на маленьких задачах.

**Acceptance Criteria:**
- [ ] linear regression: сравнение с closed-form решением (наша математика)
- [ ] logistic regression: сравнение с statsmodels (Python tests)
- [ ] poisson regression: сравнение с statsmodels (Python tests)

### Epic 5.4.2: SBC для NUTS (bayesian)

**Goal:** доверие к sampling корректности.

**Acceptance Criteria:**
- [ ] SBC на 1D/2D Normal и на простой hierarchical intercept модели
- [ ] thresholds: R-hat < 1.01, divergences < 1%, reasonable ESS/sec baseline

---

## Критерии завершения фазы

Phase V завершена когда:

1. [ ] Есть универсальный trait для моделей и generic inference (MLE + NUTS)
2. [ ] `ns-prob` содержит базовые распределения и bijectors
3. [ ] Regression golden tests проходят, SBC базовые тесты проходят
4. [ ] Python API покрывает минимум: regression + sampling + diagnostics + PPC skeleton

