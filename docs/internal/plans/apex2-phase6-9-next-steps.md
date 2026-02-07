# Apex2: План работ Phase 6–9 (по состоянию на 2026-02-06)

Цель этого документа: зафиксировать, **что уже готово**, какие задачи **не заблокированы** и какую **последовательность** (Planning → Exploration → Execution → Verification) держим дальше, чтобы NextStat был реально применим для:

- статистики и ML (регрессии, иерархические модели)
- фармы и соцнаук (survival/longitudinal/ordinal, reproducible reporting)
- временных рядов (state-space, Kalman, прогнозирование)

Документ внутренний: живет в `docs/plans/` и не должен попадать в публичные артефакты.

---

## Что уже готово (ключевые блоки)

### Phase 5: General Statistics Core

- [x] Universal `LogDensityModel` + подготовка (`PreparedModelRef`) (P5 5.1.1.1)
- [x] NUTS multichain API + Python surface (P5 5.1.2.1)
- [x] `ns-prob` + distributions + bijectors (P5 5.2.*)
- [x] Model builder (composition) MVP (P5 5.3.1)
- [x] Python data adapters (`X/y/group_idx`) (P5 5.3.2)
- [x] Golden regression fixtures + generator (P5 5.4.1)

### Phase 6: Regression & GLM Pack

- [x] Linear regression API + tests + golden SE (P6 6.1.1.*)
- [x] Logistic regression parity vs statsmodels (P6 6.2.1)
- [x] Poisson regression + offset/exposure (P6 6.3.1)
- [x] Negative binomial regression baseline (P6 6.3.2)
- [x] Regularization via MAP (ridge, intercept penalty policy) (P6 6.4.1)
- [x] CV utilities + basic metrics (P6 6.4.2)
- [x] P6 Benchmarks: end-to-end fit/predict baselines + baseline recorder (P6 Benchmarks)

### Phase 7: Hierarchical Models

- [x] Random intercepts (centered + non-centered) (P7 7.1.1)
- [x] Random slopes (P7 7.2.1)
- [x] Correlated random effects (LKJ + Cholesky) (P7 7.3.1)
- [x] PPC: posterior predictive checks (P7 PPC)

### Phase 8: Time Series

- [x] Kalman filter/smoother + log-likelihood (P8 8.1.1)
- [x] EM parameter estimation + partial missing data (P8 8.1.2)
- [x] AR(1) parameter transforms (softplus/sigmoid bijectors) (P8 8.2.1)
- [x] Forecast API + confidence bands (P8 8.3.1)
- [x] Simulate from state-space model (P8 8.4.1)

### Public Docs: inventory

- [x] Инвентаризация public vs internal (P0 Docs inventory)

---

## Что можно начинать прямо сейчас (не заблокировано зависимостями)

Phase 6:
- P6 6.2.2 Robustness: separability + numeric stability

Phase 8:
- P8 8.2.2 AR(1) parity vs statsmodels (прогон + golden fixture)
- P8 8.5.1 ARMA / structural time series (higher-order state-space)

Phase 9:
- P9 9.1.1 Survival analysis foundations (Cox PH)

Docs:
- P0 Translate public docs to English
- P0 White Paper v1 (full draft)

---

## Рекомендуемая последовательность (Apex2)

### ~~Шаг 1. Phase 6: Benchmarks (P6)~~ ✅ DONE

Выполнено: end-to-end fit/predict бенчи для всех GLM семейств (linear/logistic/poisson/negbin), Criterion бенчи, baseline recorder (`tests/record_baseline.py`) с полным environment fingerprint, Apex2 P6 runner с comparison.

### ~~Шаг 2. Phase 8: transforms → missing → AR(1) → forecast~~ ✅ DONE

Выполнено: SoftplusBijector + SigmoidBijector, AR(1) parameter bounds, partial missing data в EM, forecast API + confidence bands, simulate from state-space model.

### ~~Шаг 3. Phase 7: random effects → non-centered → LKJ → SBC/PPC~~ ✅ DONE

Выполнено: random intercepts (centered + NCP), random slopes, correlated effects (LKJ + Cholesky), PPC для всех GLM семейств включая Poisson.

### Шаг 4. Docs: перевод + white paper

**Planning:** что считаем публичным, как структурируем white paper (problem → method → validation → perf → use-cases).
**Execution:** перевести public docs, затем собрать WP v1 в едином стиле, затем pipeline (PDF/versioning).

### Шаг 5. Phase 9: Pharma / Social Sciences

**Planning:** какие модели (Cox PH, ordinal, longitudinal), parity targets (R survival, statsmodels).
**Execution:** начать с Cox PH как наиболее востребованного, затем ordinal logistic.

---

## Гейт на каждую задачу (Definition of Done)

- Unit/integration tests (Rust/Python) на корректность + крайние случаи
- Минимальный bench или измерение, если задача про performance
- Документация/докстринги для публичного Python API
- Determinism: фиксируем seed, не полагаемся на случайность без явного RNG

