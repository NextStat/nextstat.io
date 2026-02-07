# Фаза IX: Pharma + Social Sciences Packs (P1/P2)

> **Execution note:** это набор "доменных пакетов" поверх общего ядра (Phase V–VIII). Здесь важнее UX, корректность крайних случаев (цензура/пропуски), и reproducible reporting, чем микросекундная оптимизация.

**Goal:** сделать NextStat практически применимым в фарме и социальных науках: survival/цензура, продольные данные (mixed models), ordinal/категориальные модели, missing data, и базовые causal workflows.

**Duration:** 12–24 недели (можно дробить на подпакеты)

**Dependencies:** Phase V + (желательно) Phase VII–VIII.

---

## Текущий статус (2026-02-06)

BMCP (эпик Phase 9: `0fad00ab-336c-4d20-ba54-ef00da9a3a27`):

- [x] P9 9.A.1 Parametric survival (censoring/truncation baseline) (`8a44040a-d256-4541-9e35-0dead4e5c4c3`) — 43bfe9e
- [x] P9 9.A.2 Cox PH (partial likelihood + ties policy) (`8948b1fd-b17b-4e45-85fe-3e505dc3d47e`) — 293f0d8
- [x] P9 9.B.1 Longitudinal mixed models (LMM baseline: MAP/Laplace) (`fb036100-3d2a-4a16-9e5e-9f802178fba7`) — 2967719, b1d9946, 81a398a
- [x] P9 9.C.1 Ordinal logistic/probit (ordered outcomes) (`0c826930-b889-4620-a633-7f9bd106b592`)
- [x] P9 9.C.2 Missing data policy + measurement error (baseline) (`86566033-ab21-407f-87b0-2666c4adf208`)
- [x] P9 9.C.3 Causal helpers (propensity score + diagnostics) (`2bb21137-4955-4e4e-86b4-e729a51f4be3`)
- [x] P9 9.R.1 Reproducible reporting artifacts (data hash + model spec) (`6cf126cb-94f8-42da-8941-62e2bcea8435`)
- [x] P9 9.R.2 Compliance/audit trail hooks (open-core boundary aware) (`7bd89c56-9772-47d5-81c5-dcb3050271b7`)

Примечание по зависимостям:

- 9.B.1 (LMM) зависит от Phase 7 foundation (group-indexed effects + non-centered).
- 9.A.* и 9.C.* желательно начинать после стабилизации Phase 6/7/8 API, но дизайн API можно делать параллельно.

## Pack A: Survival analysis (фарм, медицина, соцнауки)

### Epic 9.A.1: Parametric survival (baseline) -- DONE

**Scope:** Exponential, Weibull, LogNormal AFT.

**Acceptance Criteria:**
- [x] right-censoring via `(times, events)` interface
- [x] MLE/MAP корректны на synthetic datasets (Rust finite-diff grad tests)
- [x] Python surface: `nextstat.survival.{exponential,weibull,lognormal_aft}()`
- [x] Contract tests: `test_survival_contract.py`
- [ ] left-truncation (deferred)
- [ ] сравнение с lifelines (Python) на golden datasets (deferred)

### Epic 9.A.2: Cox PH (P1) -- DONE

**Acceptance Criteria:**
- [x] partial likelihood + ties policy (Efron/Breslow) — `CoxPhModel`
- [x] Python surface: `nextstat.survival.cox_ph(times, events, x, ties=...)`
- [x] Rust finite-diff gradient tests
- [ ] parity vs lifelines/statsmodels survival (deferred)

---

## Pack B: Longitudinal / mixed-effects (фарм: repeated measures)

### Epic 9.B.1: Linear mixed model (LMM) baseline -- DONE

**Acceptance Criteria:**
- [x] random intercept/slope (marginal likelihood with analytic integration)
- [x] fit via ML marginal likelihood + NUTS optional (`fit()` / `sample()` integration)
- [x] Laplace approximation utilities (`ns-inference::laplace`)
- [x] Heuristic parameter initialization (intercept=mean(y), slopes, group-wise RE estimates)
- [x] Golden parity via precomputed fixtures (`tests/fixtures/lmm/`)
- [x] Parameter recovery smoke tests (`test_lmm_marginal_smoke.py`)
- [x] Contract tests (`test_lmm_contract.py`)
- [x] Golden parity tests (`test_lmm_golden_parity.py`)
- [ ] External parity vs lme4/pymer4/Stan (deferred: requires offline reference computation)

---

## Pack C: Social science staples

### Epic 9.C.1: Ordinal logistic/probit

**Acceptance Criteria:**
- [ ] ordered outcomes + cutpoints parametrization
- [ ] validation vs stan/pymc goldens

### Epic 9.C.2: Measurement error + missing data (MAR baseline)

**Acceptance Criteria:**
- [ ] explicit missing-data policy (drop/impute/model)
- [ ] simple imputation workflows documented (не как "магия")

### Epic 9.C.3: Causal inference helpers (P2)

**Acceptance Criteria:**
- [ ] propensity score + diagnostics helpers (Python)
- [ ] clear limitations and docs (не pretending it's “ground truth”)

---

## Reporting & Compliance (для фармы)

Если реально целимся в фарму (GxP, 21 CFR Part 11), понадобятся:
- audit trail (что запускалось, с какими seed/versions)
- immutable artifacts (model spec + data hash + results)
- validation packs (предопределённые тесты/отчёты)

Это может быть OSS “baseline” + Pro расширения (см. open-core boundary).
