# Фаза III: Production Ready (P1)

> **Execution note (humans + AI agents):** Выполнять задачи последовательно. Каноничные определения/допуски/детерминизм: `docs/plans/standards.md`.

**Goal:** Довести NextStat до “production-ready” состояния: стабильные API, релизы, документация, визуализации, расширенная валидация и готовность к пилотам.

**Duration:** Месяцы 9-15

**Dependencies:** Фазы 0–2 (особенно 2A CPU performance и 2B AD/градиенты).

**Tech Stack:** Rust, PyO3/maturin, tracing, Plotly/Matplotlib (Python), Sphinx/MkDocs, GitHub Actions releases.

---

## Содержание

- [Acceptance Criteria (Фаза)](#acceptance-criteria-фаза)
- [Sprint 3.1: Frequentist limits (P0)](#sprint-31-frequentist-limits-p0-недели-37-40)
- [Sprint 3.2: Bayesian interface (P1, optional)](#sprint-32-bayesian-interface-p1-optional-недели-41-44)
- [Sprint 3.3: Production hardening](#sprint-33-production-hardening-недели-45-48)
- [Sprint 3.4: Visualization](#sprint-34-visualization-недели-49-52)
- [Sprint 3.5: Documentation](#sprint-35-documentation-недели-53-60)
- [Sprint 3.6: Validation Suite v1](#sprint-36-validation-suite-v1-недели-61-68)
- [Критерии завершения фазы](#критерии-завершения-фазы)

---

## Acceptance Criteria (Фаза)

| Метрика | Target | Measurement |
|---------|--------|-------------|
| Release discipline | semver + changelog | tag `v0.x.y` + `CHANGELOG.md` |
| Wheels | Linux/macOS wheels | `pip install nextstat` без компиляции |
| Deterministic parity | стабильно в CI | `pytest tests/python/test_pyhf_validation.py` |
| Bias / coverage regression | совпадает с pyhf на toys | `NS_RUN_SLOW=1 pytest -m slow tests/python/test_bias_pulls.py` + report artifacts |
| Limits (CLs) | совпадает с pyhf (asymptotics) | `pytest -k hypotest tests/python` (golden points) |
| Profile likelihood | q(mu) совпадает с pyhf | `pytest -k profile tests/python` (golden points) |
| Bayesian sampling (NUTS/HMC) | Rust core sampler + diagnostics | `cargo test -p ns-inference` (sampler golden tests) |
| Viz outputs | pulls/ranking/corr | golden image/json tests |
| Docs | “hello fit → ranking” | docs build green + tutorials |

> Bayesian sampling реализуем как **NUTS/HMC в Rust core** (`ns-inference`). Он остаётся P1/optional для закрытия Phase 3, но контракт и тесты фиксируем заранее (см. `docs/plans/standards.md` → раздел 7).

---

## Sprint 3.1: Frequentist limits (P0) (Недели 37-40)

Цель спринта: закрыть каноничный HEP workflow для **upper limits**:

- profile likelihood scan (фиксируем POI, профилируем nuisance)
- асимптотический **CLs** (pyhf-compatible, `test_stat="qtilde"`)
- удобные surfaces (CLI/Python) + parity tests

### Epic 3.1.1: Profile likelihood scan (q_mu / qtilde_mu)

#### Task 3.1.1.1: Fast minimization path (skip Hessian)

**Priority:** P0  
**Goal:** уметь делать много фит-минимизаций быстро (scan/limits), не вычисляя covariance/Hessian.

**Acceptance Criteria:**
- [ ] Есть API/внутренний helper: минимизация NLL без Hessian
- [ ] Не ломаем `FitResult` и существующий `nextstat.fit(...)`
- [ ] Детеминизм: одинаковый вход даёт одинаковый минимум

#### Task 3.1.1.2: Scan API + q(mu)

**Acceptance Criteria:**
- [ ] Unconditional fit: `(mu_hat, nll_hat)`
- [ ] Conditional fit: `nll(mu)` при зафиксированном POI
- [ ] Возвращаем структуру scan результата: `mu, q, nll, converged, n_iter`
- [ ] Golden parity: q(mu) совпадает с pyhf на `simple_workspace.json` в нескольких точках

### Epic 3.1.2: Asymptotic CLs limits (pyhf hypotest compatibility)

#### Task 3.1.2.1: Hypotest (asymptotics, qtilde)

**Acceptance Criteria:**
- [ ] Реализован `hypotest(mu_test)` с `test_stat="qtilde"`, `calctype="asymptotics"`
- [ ] Возвращаем `CLs`, `CLs+b`, `CLb` и внутренние `qtilde_obs`/`qtilde_A`
- [ ] Golden parity: значения совпадают с pyhf для `simple_workspace.json` (mu=0,0.5,1,2)

#### Task 3.1.2.2: Upper limit finder (CLs=alpha)

**Acceptance Criteria:**
- [ ] `upper_limit(alpha=0.05)` (bracket + root find) для фиксированной модели/данных
- [ ] Устойчиво на разумных workspace (simple/complex fixtures)

### Epic 3.1.3: Surfaces + docs

#### Task 3.1.3.1: CLI + Python surface

**Acceptance Criteria:**
- [ ] CLI: `nextstat hypotest ...` и `nextstat upper-limit ...` с JSON output
- [ ] Python: `nextstat.infer.hypotest(...)`, `nextstat.infer.upper_limit(...)`, `nextstat.infer.scan(...)`

---

## Sprint 3.2: Bayesian interface (P1, optional) (Недели 41-44)

### Epic 3.2.1: Posterior contract + transforms

**Цель:** Зафиксировать интерфейс для posterior logpdf/grad без обязательства “сразу писать NUTS”.

> Source of truth: `docs/plans/standards.md` → раздел 7 (Bayesian / Posterior contract).

#### Task 3.2.1.1: `Posterior` API (logpdf + grad) (required for NUTS)

**Priority:** P1  
**Effort:** 6-10 часов  
**Dependencies:** Phase 2B (аналитические градиенты доступны для CPU reference path)

**Files:**
- Create: `crates/ns-inference/src/posterior.rs`
- Create: `crates/ns-inference/src/transforms.rs` (bijectors + log|J| + grad chain-rule)
- Modify: `crates/ns-core/src/traits.rs` (добавить trait для logpdf/grad или адаптер поверх `ComputeBackend`)

**Acceptance Criteria:**
- [ ] Posterior определяет prior-термы для POI/NP (минимум: flat для POI, Normal для constrained NP)
- [ ] `posterior_logpdf` = `model.logpdf + prior_terms` (канонично)
- [ ] `posterior_grad` использует `Model::gradient()` если доступен
- [ ] При flat prior MAP совпадает с MLE (sanity contract из `docs/plans/standards.md`)
- [ ] Bounded параметры корректно переводятся в `z ∈ R^n` (см. standards §7.3), `log|J|` учитывается

**Implementation sketch (pseudocode; trait names may differ):**

```rust
// crates/ns-inference/src/posterior.rs
use ns_core::types::Float;
use ns_core::Model;

pub struct Posterior<'a, M: Model> {
    model: &'a M,
}

impl<'a, M: Model> Posterior<'a, M> {
    pub fn new(model: &'a M) -> Self {
        Self { model }
    }

    pub fn logpdf(&self, x: &[Float]) -> ns_core::Result<Float> {
        let lp = self.model.logpdf(x)?;
        Ok(lp + self.prior_logpdf(x))
    }

    pub fn grad(&self, x: &[Float]) -> ns_core::Result<Vec<Float>> {
        // Phase 2B provides `Model::gradient()`.
        let mut g = self.model.gradient(x)?;
        let pg = self.prior_grad(x);
        for (gi, pgi) in g.iter_mut().zip(pg.iter()) {
            *gi += *pgi;
        }
        Ok(g)
    }

    fn prior_logpdf(&self, _x: &[Float]) -> Float {
        // Minimal default: flat priors for POI, no extra priors for NP (constraints are already in model.logpdf).
        0.0
    }

    fn prior_grad(&self, x: &[Float]) -> Vec<Float> {
        vec![0.0; x.len()]
    }
}
```

#### Task 3.2.1.2: Transform correctness tests (golden)

**Priority:** P1  
**Effort:** 4-8 часов  
**Dependencies:** Task 3.2.1.1

**Files:**
- Create: `crates/ns-inference/src/transforms_tests.rs`

**Acceptance Criteria:**
- [ ] `inv(transform(z)) ≈ z` (rtol 1e-10) для всех bijectors
- [ ] `grad_z(log|J|)` совпадает с finite-diff (rtol 1e-7)
- [ ] Градиент chain-rule: `∂/∂z log p(θ(z))` совпадает с finite-diff на простых функциях

---

### Epic 3.2.2: HMC kernel (leapfrog)

#### Task 3.2.2.1: Leapfrog integrator + accept/reject (static HMC)

**Priority:** P1  
**Effort:** 10-16 часов  
**Dependencies:** Epic 3.2.1 (posterior logpdf+grad in unconstrained space)

**Files:**
- Create: `crates/ns-inference/src/hmc.rs`
- Create: `crates/ns-inference/src/hmc_tests.rs`

**Acceptance Criteria:**
- [ ] Для 1D Normal static-HMC даёт mean≈0, var≈1 (N=5k, burn-in=1k)
- [ ] Детерминизм при фиксированном seed
- [ ] Диагностика: acceptance rate и energy error доступны в результатах

---

### Epic 3.2.3: NUTS + adaptation (core)

#### Task 3.2.3.1: Dual averaging step-size adaptation

**Priority:** P1  
**Effort:** 6-10 часов  
**Dependencies:** Epic 3.2.2

**Files:**
- Create: `crates/ns-inference/src/adapt.rs`

**Acceptance Criteria:**
- [ ] `target_accept` достигается в warmup (±0.05) на MVN toy
- [ ] Step-size фиксируется после warmup и воспроизводим

#### Task 3.2.3.2: NUTS tree building (Stan-style multinomial)

**Priority:** P1  
**Effort:** 14-24 часа  
**Dependencies:** Task 3.2.3.1

**Files:**
- Create: `crates/ns-inference/src/nuts.rs`
- Create: `crates/ns-inference/src/nuts_tests.rs`

**Acceptance Criteria:**
- [ ] MVN(dim=4): mean/cov в пределах статистической погрешности vs analytic
- [ ] Есть `max_treedepth` guard + diagnostics (treedepth saturations)
- [ ] Divergences считаются и репортятся

#### Task 3.2.3.3: Diagonal mass matrix adaptation (warmup windows)

**Priority:** P1  
**Effort:** 8-14 часов  
**Dependencies:** Task 3.2.3.2

**Acceptance Criteria:**
- [ ] Для MVN с несбалансированными масштабами существенно растёт ESS/sec vs unit mass
- [ ] Масса фиксируется после warmup; в draws сохраняется финальная diag mass

---

### Epic 3.2.4: Multi-chain runner + diagnostics

#### Task 3.2.4.1: Multi-chain + R-hat/ESS (baseline)

**Priority:** P1  
**Effort:** 8-12 часов  
**Dependencies:** Epic 3.2.3

**Files:**
- Create: `crates/ns-inference/src/diagnostics.rs`
- Create: `crates/ns-inference/src/chain.rs` (draws + stats struct)

**Acceptance Criteria:**
- [ ] 4-chain runner (Rayon optional) детерминирован по seed-ам
- [ ] R-hat/ESS считаются для test distributions; `R-hat < 1.01` на простых задачах

#### Task 3.2.4.2: Python surface for draws (optional)

**Priority:** P2  
**Effort:** 4-8 часов  
**Dependencies:** Task 3.2.4.1

**Goal:** экспорт draws в numpy/arrow-friendly формат + анализ в ArviZ.

---

## Sprint 3.3: Production hardening (Недели 45-48)

### Epic 3.3.1: Release + observability

#### Task 3.3.1.1: Release pipeline (maturin + GitHub Actions)

**Priority:** P0  
**Effort:** 6-10 часов  
**Dependencies:** Phase 1.4 (Python bindings) + Phase 0 CI

**Files:**
- Create: `.github/workflows/release.yml`
- Create: `CHANGELOG.md`
- Modify: `pyproject.toml` (versioning policy)

**Acceptance Criteria:**
- [ ] `release.yml` строит wheels для Linux/macOS и прикрепляет к GitHub Release
- [ ] Manual release (workflow_dispatch) + tag-based release
- [ ] `CHANGELOG.md` обновляется перед релизом

#### Task 3.3.1.2: Structured logging (`tracing`) end-to-end

**Priority:** P0  
**Effort:** 4-6 часов  
**Dependencies:** Phase 1 CLI

**Files:**
- Modify: `crates/ns-cli/src/main.rs`
- Modify: `crates/ns-core/src/error.rs` (error contexts)

**Acceptance Criteria:**
- [ ] CLI имеет `--log-level` и печатает structured logs
- [ ] Ошибки включают контекст (какой файл, какой backend, какой POI)

---

## Sprint 3.4: Visualization (Недели 49-52)

### Epic 3.4.1: Plots for HEP workflows

#### Task 3.4.1.1: Pull plot + ranking plot (Python)

**Priority:** P0  
**Effort:** 8-12 часов  
**Dependencies:** Phase 1.5 validation outputs + Phase 2C ranking optimization

**Files:**
- Create: `bindings/ns-py/python/nextstat/viz/pulls.py`
- Create: `bindings/ns-py/python/nextstat/viz/ranking.py`
- Test: `tests/python/test_viz_outputs.py`

**Acceptance Criteria:**
- [ ] Функции `plot_pulls(result)` и `plot_ranking(entries)` возвращают Plotly figure
- [ ] Golden tests: JSON snapshot (no pixel-flakes)

#### Task 3.4.1.2: Correlation matrix plot

**Priority:** P1  
**Effort:** 6-8 часов  
**Dependencies:** correlation extraction available (Phase 2B/3.3)

---

## Sprint 3.5: Documentation (Недели 53-60)

### Epic 3.5.1: Docs site + tutorials

#### Task 3.5.1.1: Build docs (Sphinx/MkDocs) + API reference

**Priority:** P0  
**Effort:** 6-10 часов  
**Dependencies:** stable public API

**Acceptance Criteria:**
- [ ] Docs build in CI
- [ ] Tutorial: `load → fit → ranking → scan`

---

## Sprint 3.6: Validation Suite v1 (Недели 61-68)

### Epic 3.6.1: Expand fixtures + regression tests

#### Task 3.6.1.1: Fixture pack “small/medium/large”

**Priority:** P0  
**Effort:** 8-12 часов  
**Dependencies:** Phase 1 parity suite

**Acceptance Criteria:**
- [ ] >= 10 workspaces covering modifier combinations
- [ ] Stored references: bestfit/uncertainties/Δtwice_nll
- [ ] CI matrix runs deterministic parity

---

### Epic 3.6.2: Bias & Coverage certification (toys)

Цель: контролировать **смещение и покрытие** на toy-ансамблях и гарантировать, что NextStat не расходится с pyhf не только на единичных workspace, но и статистически.

> Важно: это **slow** нагрузка. В PR-матрицу не добавляем. Запуск: nightly/manual workflow + публикация отчётов как artifacts.

#### Task 3.6.2.1: Pull/bias regression vs pyhf (toy ensembles)

**Priority:** P1  
**Effort:** 6-10 часов  
**Dependencies:** Phase 1 parity suite + стабильный `nextstat.fit(..., data=...)` API

**Files:**
- Use/extend: `tests/python/test_bias_pulls.py` (Phase 1 adds a smoke version; opt-in via `NS_RUN_SLOW=1`)
- Create: `reports/bias_coverage/README.md`
- (optional) Create: `.github/workflows/toys-validation.yml` (nightly/manual)

**Acceptance Criteria:**
- [ ] Для fixture-pack (минимум: `simple`, `complex`) собран pull для `mu` и ключевых constrained NP.
- [ ] Сводные метрики NextStat vs pyhf в пределах допусков (см. `docs/plans/standards.md` → раздел 6).
- [ ] Генерируется JSON report (mean/std/coverage) и сохраняется как artifact.

#### Task 3.6.2.2: Coverage checks for 1σ / 2σ intervals (baseline)

**Priority:** P1  
**Effort:** 4-8 часов  
**Dependencies:** Task 3.6.2.1

**Notes / decisions:**
- Baseline интервал в Phase 3: `mu_hat ± 1σ_hat` и `mu_hat ± 2σ_hat` (Hessian-based) — быстро и стабильно.
- Позже (Phase 3/4) добавить profile-likelihood intervals и coverage для них (медленнее, но каноничнее).

**Acceptance Criteria:**
- [ ] Coverage(68%) и Coverage(95%) сравниваются с pyhf на toy-ансамбле.
- [ ] Разница coverage NextStat vs pyhf <= 1–3% (tunable; зависит от `N_TOYS`).

---

## Критерии завершения фазы

Phase 3 завершена когда:

1. [ ] Release pipeline выпускает wheels и changelog (P0)
2. [ ] Документация и tutorials проходят сборку в CI (P0)
3. [ ] Pulls/ranking/correlation визуализации доступны из Python (P0/P1)
4. [ ] Validation suite v1 стабильна (детерминизм + регрессии) (P0)
5. [ ] (P1) Bias/coverage regression vs pyhf задокументирован и воспроизводим (toys report)
