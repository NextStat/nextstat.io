# NextStat Project Standards (Source of Truth)

Этот документ фиксирует **единые стандарты** для всех фаз/планов: определения (`twice_nll`), допуски, precision policy и детерминизм.  
Любые расхождения между фазовыми планами должны разрешаться **в пользу этого документа**.

---

## 0) Source of Truth

- `docs/plans/README.md` — навигация + таймлайн + высокоуровневые цели.
- `docs/plans/2026-02-05-nextstat-implementation-plan.md` — master plan (сквозная структура).
- `docs/plans/phase-*.md` — исполняемые планы по фазам (epics/tasks/код/команды).
- `docs/references/*` — внешние вводные/ресёрч (если используется в планах, должно быть доступно по ссылке).

---

## 1) Термины и математика (канонично)

### 1.1 `logpdf`, `nll`, `twice_nll`

- `logpdf(θ)` — логарифм правдоподобия (и constraint terms) **как в `pyhf.Model.logpdf`**.
- `nll(θ) = -logpdf(θ)`.
- `twice_nll(θ) = 2 * nll(θ) = -2 * logpdf(θ)`.

> Важно: `twice_nll` используется для parity с pyhf и для отчётности. В hot-path оптимизации допускается использовать вариант без констант, **но** контракт parity должен явно учитывать это (см. раздел 3).

### 1.2 Poisson term (binned)

Для одного бина:

- `log P(n | λ) = n * ln(λ) - λ - ln Γ(n + 1)`
- `poisson_nll(n, λ) = λ - n * ln(λ) + ln Γ(n + 1)`
- `poisson_twice_nll = 2 * poisson_nll`

Где `Γ` — гамма-функция (обобщает факториал: `Γ(n+1) = n!` для целых `n`).

**Контракт:** для Phase 1 (детерминизм + f64) `twice_nll` должен совпадать с `float(model.logpdf(pars, data)[0] * -2)` из pyhf (numpy backend).

### 1.3 Gaussian constraint term

Для constraint на параметр `x ~ Normal(μ, σ)`:

- `log N(x | μ, σ) = -0.5 * z^2 - ln(σ) - 0.5 * ln(2π)`, где `z = (x-μ)/σ`
- `gaussian_nll = 0.5 * z^2 + ln(σ) + 0.5 * ln(2π)`
- `gaussian_twice_nll = z^2 + 2 ln(σ) + ln(2π)`

> Если какая-то часть констант намеренно опускается в вычислениях (performance), это должно быть оформлено как отдельный режим/функция и тестироваться как `Δtwice_nll`, а не как абсолютное значение.

---

## 2) Precision Policy

### 2.1 Типы

- CPU reference/validation path: **`f64` везде**, стабильные редукции.
- GPU path: допускается **mixed precision** (например `f32` compute + `f64` accumulation), но parity-тесты выполняются через CPU reference path.

### 2.2 Нормы точности

- **Deterministic CPU parity (Phase 1 contract):**
  - `twice_nll`: `rtol=1e-6`, `atol=1e-8` (строго только в deterministic режиме)
  - `bestfit` (Phase 1 fixtures, сравнение **по именам параметров**): `atol=2e-4`
  - `uncertainties` (Phase 1 fixtures, сравнение **по именам параметров**): `atol=5e-4`
- **Performance modes (Rayon / GPU):**
  - допускаются более мягкие допуски (фиксируются отдельно в соответствующих фазах),
  - сравнение `twice_nll` предпочтительно делать как `Δtwice_nll` относительно bestfit.

> Все допуски должны быть вынесены в одно место (константы) в тестах, а планы должны ссылаться на эти константы (не дублировать числа в нескольких файлах).

---

## 3) Детерминизм (обязателен для parity)

### 3.1 Требование

Parity с pyhf должен выполняться в режиме:

- `backend=cpu`
- `threads=1` **или** детерминированный parallel-reduce
- фиксированный порядок суммирования
- фиксированные seed-ы RNG

### 3.2 Rayon и редукции

Если используется Rayon:

- должен быть режим `deterministic=true`, который:
  - либо принудительно выполняет вычисления последовательно,
  - либо использует фиксированный порядок редукции (pairwise/tree reduce с детерминированным разбиением на чанки).

---

## 4) Testing & Coverage

### 4.1 Тестовая матрица

- Unit tests (Rust): `cargo test --all`
- Property tests: ограничивать runtime и фиксировать seed при флейках
- Python validation against pyhf: deterministic CPU backend

### 4.2 Coverage (кроссплатформенно)

Предпочтительный инструмент для Rust coverage: `cargo llvm-cov` (вместо tarpaulin как дефолта).

---

## 5) Стандарты “исполняемых планов”

- Код-блоки в планах должны быть **компилируемыми/валидными** (или помечены как псевдокод).
- Формулировка “Write failing test” используется только для реального test harness.  
  Если речь про “команда должна упасть, потому что файла нет” — писать “Sanity check (expected to fail)”.
- Любые TODO/placeholder в планах должны быть:
  - либо вынесены в отдельный backlog-раздел,
  - либо иметь чёткий критерий “когда закрываем” и ссылку на задачу/эпик.

### 5.1 Принятые проектные решения (Phase 1–2)

Эти решения **осознанно допускают** временные отклонения от “идеальной” архитектуры ради скорости и простоты:

1) **`ns-inference` зависит от `ns-translate::HistFactoryModel` напрямую.**  
   Причина: сейчас есть только один тип модели (HistFactory). Абстрактный trait будет введён, когда появится второй модельный тип.

2) **`FitResult.n_evaluations` = `n_iter` (argmin limitation).**  
   Точное `n_fev` требует счётчика вызовов cost/grad. Решение: оставить как есть до Phase 3 polish, затем либо переименовать поле, либо добавить `n_fev`.

---

## 6) Bias / Coverage (Frequentist QA)

В планах NextStat базовый режим — frequentist (MLE, profile likelihood, asymptotics).  
Это **не означает**, что все оценки будут “unbiased” для любых малых статистик/границ параметров.

### 6.1 Термины (канонично)

- **Bias (смещение)**: `bias(θ̂) = E[θ̂] - θ_true`.
- **Pull** (для параметров, где есть σ̂): `pull = (θ̂ - θ_true) / σ̂`.
  - Ожидаемое поведение в “хорошем” режиме: `mean(pull) ≈ 0`, `std(pull) ≈ 1`.
- **Coverage**: доля toy-экспериментов, где интервал (например `θ̂ ± 1σ̂`) покрывает `θ_true`.

### 6.2 Политика (BIAS решения)

1) **Phase 1 contract = parity с pyhf**, а не “магическое устранение bias”.
   - Если в pyhf есть систематическое смещение/undercoverage на конкретных моделях (границы, нелинейности, малые счёты) — NextStat не должен “чинить” это молча.

2) **Мы обязаны измерять bias/coverage на toy-ансамблях** и контролировать, что NextStat **не вводит дополнительного смещения** относительно pyhf.

3) **Bias-correction (если появится)**:
   - только как **явная опция** (off-by-default),
   - только после toy-валидации,
   - с документированным эффектом на coverage и edge cases (границы, constraints).

### 6.3 Quality gates (по фазам)

- **Phase 1 (smoke, P1):** toy pull/coverage regression vs pyhf на `tests/fixtures/*`.
  - Сравниваем `mean/std(pull)` и coverage для POI (обычно `mu`) между NextStat и pyhf.
  - Фиксируем seed, чтобы не было flake.
  - Тест **не должен** быть тяжёлым (ориентир: `N_TOYS=200`).
  - Suggested tolerances (difference vs pyhf):
    - `|Δmean(pull_mu)| <= 0.05`
    - `|Δstd(pull_mu)| <= 0.05`
    - `|Δcoverage_1sigma(mu)| <= 0.03`
  - Source of truth task: `docs/plans/phase-1-mvp-alpha.md` → Task 1.5.1.2

- **Phase 3 (certification, P0/P1):** расширенный bias/coverage отчёт (nightly/manual).
  - Для fixture-pack “small/medium/large” прогоняем `N_TOYS >= 5000` (offline или nightly workflow).
  - Артефакт: JSON report + графики pull/coverage (публикуем как build artifact).
  - Source of truth task: `docs/plans/phase-3-production.md` → Epic 3.5.2

---

## 7) Bayesian / Posterior contract (P1 optional)

NextStat позиционируется как “Freq + Bayesian”, но **байесовский слой не должен ломать frequentist parity**.

### 7.1 Термины (канонично)

- **Likelihood**: `L(data | θ)`; `log L` может включать auxiliary/constraint terms (как в HEP HistFactory).
- **Prior**: `p(θ)`; задаётся явно (в первую очередь для POI и unconstrained NP).
- **Posterior**: `p(θ | data) ∝ L(data | θ) * p(θ)`.
  - `log posterior = log L + log prior + const`.

### 7.2 Политика (решения)

1) **Constraints ≠ “extra prior” по умолчанию.**
   - Контракт Phase 1/2: `model.logpdf`/`model.nll` уже включает constraint terms (parity с pyhf).
   - В Bayesian режиме мы добавляем **только дополнительный prior**, чтобы не было double-count.

2) **Phase 3 goal = HMC/NUTS в Rust core (sampling “внутри” `ns-inference`).**
   - Минимум: корректный `posterior_logpdf(params)` + `posterior_grad(params)` (через Phase 2B AD).
   - Sampling в Python допускается только как **кросс‑проверка/диагностика**, но не как основной runtime.

3) **MAP sanity contract:**
   - При flat prior (или отсутствующем prior) MAP должен совпадать с MLE в пределах frequentist допусков.

### 7.3 Алгоритмические решения (MVP)

Чтобы NUTS/HMC был корректным и устойчивым на bounded параметрах:

- **Unconstrained parameterization (обязательно):**
  - Сэмплируем `z ∈ R^n`, а `θ = transform(z)` с учётом bounds.
  - В `log posterior` добавляем `log|det J|` (якобиан трансформации).
  - Базовые bijectors:
    - `(−∞, ∞)` → identity
    - `(0, ∞)` → `exp(z)`
    - `(a, ∞)` → `a + exp(z)`
    - `(a, b)` → `a + (b-a) * sigmoid(z)`

- **Integrator:** leapfrog (symplectic) + сохранение “energy error” в diagnostics.
- **NUTS variant:** Stan-style multinomial NUTS + stop criterion “no U-turn”.
- **Warmup/adaptation (Phase 3 baseline):**
  - Dual averaging для step size (`target_accept` по умолчанию 0.8).
  - Mass matrix: diagonal (Welford/online variance), windowed adaptation (как в Stan, упрощённая версия).
- **Determinism:** фиксированные seed-ы + независимые цепочки (seed = base + chain_id).

### 7.3 Quality gates (по фазам)

- **Phase 3 (P1, optional):**
  - Есть единый `Posterior` API (Rust + Python surface), который:
    - не double-count constraints,
    - принимает seed/chain id для воспроизводимости,
    - запускает NUTS/HMC в Rust core и возвращает draws + diagnostics.
  - Минимальные “golden” тесты для sampler’а:
    - 1D Normal: mean≈0, var≈1
    - MVN (dim=4): mean/cov в пределах статистической погрешности
    - “Banana” / funnel (stress): нет exploding energy, фиксируем divergence rate и treedepth
  - Минимальные метрики качества (ориентиры):
    - `R-hat < 1.01` на toy задачах
    - divergence rate < 1% (или явный отчёт почему нет)
    - приемлемая ESS на параметр (фиксируем baseline per‑sec для регрессий)
