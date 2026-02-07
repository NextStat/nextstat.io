<!--
Plan
Generated: 2026-02-07 19:34:34
Git Commit: 181db78d7897ee14f21028cfa4591a2b1b8fb740
Scope: ML-facing features (PyTorch layer, ranking, metrics, gym)
Method: TDD (tests first)
-->

# NextStat → ML-инженеры: план реализации (TDD)

Статус (2026-02-07):
- M0–M4: done
- M5 (Gymnasium env, optional): done

Цель: превратить NextStat из “инференс-движка для HEP” в **компонент ML стека**, который:
1) встраивается в evaluation pipeline (sweeps, experiment tracking),
2) даёт быстрый sensitivity/debugging (ranking по систематикам),
3) (флагман) может выступать как **дифференцируемый слой** для обучения на физической метрике,
4) (опционально) предоставляет быстрый RL “gym” для design-of-experiments.

## Принципы (что защищаем тестами)

- **Parity / корректность:** значения метрик и артефактов должны совпадать с CPU эталоном (и где релевантно — с pyhf контрактом).
- **Детерминизм:** одинаковые входы/seed → одинаковые выходы (особенно для ranking/toys).
- **Fail-fast на невалидных данных:** лучше явная ошибка/флаг degraded, чем “тихая деградация”.
- **Performance by default:** быстрые API не должны делать лишнюю работу (например, Hessian на каждый refit).

## Милестоуны (в порядке выгод/риска)

### M0 — Пререквизиты качества (обязательное)
Почему: эти пункты напрямую влияют на доверие ML-пайплайнов и на то, что можно обещать в питче.

**TDD чек-лист:**
- Добавить тесты/контракты, которые ловят:
  - отсутствие ZSTD/XZ декомпрессии ROOT,
  - “uncertainties=0” в CUDA warm-start fit,
  - “placeholder ln_hi/ln_lo”/degraded NormSys на GPU,
  - time budget для ranking (регрессия по времени).

**Реализация (коротко):**
- Закрыть задачи из эпика аудита (см. BMCP):
  - ROOT `ZS/XZ` поддержка или честное ограничение,
  - ranking: minimum-only + warm-start + tape reuse + constrained-only default,
  - CUDA warm-start: либо честный “min-only” API, либо вычисление uncertainties,
  - GPU NormSys: валидация/явный degraded флаг.

**Definition of Done (DoD):**
- `cargo test --workspace --all-features` и `pytest -m "not slow"` проходят.

---

### M1 — “Model Card” / Experiment Tracking (быстрый win)
Фокус: сделать NextStat evaluator-ом для sweeps (W&B/MLflow) без тяжёлых зависимостей.

**API (целевое):**
- CLI: единый флаг `--json-metrics <path|->` для команд `fit/hypotest/upper_limits/profile_scan`.
- Стандартизованный JSON (v0):
  - `metric`: `mu_hat`, `mu_up`, `cls`, `q_mu`, …
  - `uncertainty`: `mu_sigma`, `cov_available`, …
  - `timing`: `fit_time_s`, …
  - `status`: `converged`, `warnings[]`, `degraded_flags[]`
  - `provenance`: версия, commit, threads, eval_mode, device.

**Тесты (сначала):**
- Python: snapshot/contract тест на схему и обязательные поля для каждого subcommand (`tests/python/test_metrics_contract.py`).
- Rust: unit-тест сериализации структуры метрик + “stable ordering” (не зависит от HashMap порядка).

**Реализация:**
- Вынести общую структуру метрик в `ns-core` или `ns-cli` (лучше в `ns-core` как `serde`-тип).
- В CLI добавить общий writer: stdout или файл.
- В Python: helper `nextstat.report.metrics_to_wandb_row(...)` (без зависимости от wandb: возвращает dict).

**DoD:**
- Можно запускать sweep, где NextStat пишет одну строку/JSON на модель и это легко логируется.

---

### M2 — Fast Sensitivity / Ranking v1 (для ML-debugging)
Фокус: “какая систематика доминирует” должно быть быстро и интерпретируемо.

**API (целевое):**
- Python: `nextstat.viz.ranking_artifact(model, top_n=...)` уже существует — довести контракт.
- Rust: `MaximumLikelihoodEstimator::ranking_fast(...)` (или переработать текущий `ranking`).

**Тесты (сначала):**
- Контракт: ranking возвращает только constrained nuisances по умолчанию; POI исключён; сортировка стабильна.
- Производительность: тест “не вычисляет Hessian N раз” (через counters `n_fev/n_gev` или тайм-бюджет на фикстуре).
- Детерминизм: одинаковый seed/threads → одинаковый порядок и числа в parity mode.

**Реализация:**
- Использовать минимум-only fit для up/down точек, warm-start, bounds-clamp, tape reuse per thread.
- Явно определять σ: брать constraint_width; для shapesys/staterror — корректно из tau/unc.

**DoD:**
- Ranking на “сотни параметров” не выглядит как minutes-hours.

---

### M3 — Differentiable Layer (PyTorch) — MVP без “магии”
Фокус: получить реальный autograd signal в training loop **по входам от сети**, даже если сначала на упрощённой метрике.

#### MVP-метрика (реалистичная ступень)
Сначала делаем дифференцируемую цель без внутреннего оптимизатора:
- **Asimov-style differentiable significance** на фиксированных nuisance (или с простым профилированием без implicit diff).
- Вход: ожидаемые yields в main bins (например, сигнал из сети + фиксированный фон).
- Выход: scalar loss и grad по входным yields.

**Тесты (сначала):**
- Rust: `loss_and_grad(y)` совпадает с finite-diff по `y[i]` на маленькой фикстуре.
- Python: контракт `_core.autograd_*` возвращает `(loss, grad)` правильных размеров и без NaN.
- Если torch доступен: `torch.autograd.gradcheck` на double.

**Реализация:**
- Добавить в Rust публичный entrypoint для “loss+grad w.r.t. yields”, используя `ns-ad::tape::Tape`.
- В Python добавить `nextstat.torch`:
  - `class NextStatLoss(torch.autograd.Function)` вызывает `_core.loss_and_grad(...)`.
  - `nn.Module`-обёртка для удобства.

#### Контракт данных (важно для корректности)
В рантайме мы хотим дифференцировать не “по всему workspace”, а **по вектору номиналов** одного sample (main bins),
который приходит из ML (soft-binning / differentiable cuts).

Чтобы градиент был корректным без пересборки модели, мы ограничиваемся sample, у которого модификаторы
**не зависят от номинала через шаблоны/aux-terms**. На практике сейчас поддерживаем:
- `NormFactor`, `NormSys`, `Lumi`, `ShapeFactor` (чисто мультипликативные; у `ShapeFactor` нет constraint/aux по умолчанию).

Fail-fast (пока не поддерживаем дифференцирование по nominal):
- `HistoSys` (дельты/интерполяция завязаны на nominal и hi/lo),
- `ShapeSys` / `StatError` (aux/τ зависят от nominal; нужна цепочка через τ и стабильность).

**DoD:**
- В репо есть минимальный пример training loop (скрипт), который backprop-ит через NextStat.

---

### M4 — Differentiable Layer (PyTorch) — профилирование + implicit gradients (флагман)
Фокус: “train on discovery significance / limits with systematics” честно, без backprop через итерации оптимизатора.

**Идея:**
Пусть оптимизация по nuisance решает условия стационарности `∂NLL/∂θ = 0`.
Тогда градиент целевой функции по входам (yields) получаем через **implicit differentiation** с использованием Hessian/блочной структуры.

**Тесты (сначала):**
- Сравнение implicit-grad vs finite-diff на маленькой модели (2–5 параметров, 2–4 бина).
- Набор edge-case тестов: bounds, near-zero rates, nonconvergence → понятные ошибки/flags.
- Стабильность: одинаковый seed → одинаковые грады.

**Реализация (по шагам):**
1) Определить, что именно дифференцируем: `mu_up(alpha)` / `q_mu` / `Z` (зафиксировать контракт).
2) Реализовать вычисление нужных Jacobian/Hessian блоков:
   - `H_{θθ}`, `H_{θy}`, и т.п. (использовать существующие grad/Hessian пути; по возможности — reuse).
3) Решать линейную систему (Cholesky/LDLT + damping) для implicit grads.
4) Экспортировать в PyTorch как stable API.

**DoD:**
- Демонстрация: toy классификатор тренируется на NextStat-метрике и метрика улучшается на eval.

---

### M5 — Gymnasium RL Environment (опционально)
Фокус: показать “миллионы шагов за минуты” и дать playground для DOE/RL.

**Тесты (сначала):**
- Gymnasium API contract (reset/step/seed, shapes, termination rules).
- Determinism: фиксированный seed → воспроизводимый эпизод.
- Perf smoke: N шагов < budget на фикстуре.

**Реализация:**
- `nextstat.gym` модуль с минимальным environment:
  - State: текущие параметры (bin edges, cuts, toggles каналов),
  - Action: дискретные/непрерывные изменения,
  - Reward: выбранная NextStat метрика (CLS/Z).
- Пример baseline агента (random / hill-climb).

---

## Что НЕ делаем в первой версии (cut-lines)

- JAX/TF слой до стабилизации PyTorch API и implicit-grad ядра.
- Полная W&B/MLflow интеграция как обязательная зависимость (только адаптеры/примеры).
- RL env без стабильного evaluator/metrics schema.

## Команды проверки (локально/CI)

- Rust: `cargo test --workspace --all-features`
- Python: `pytest -q -m "not slow" tests/python`
- Parity: `pytest -v tests/python/test_pyhf_validation.py tests/python/test_expected_data_parity.py`
