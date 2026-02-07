# Plan: pyhf NumPy как спецификация + Parity/Fast режимы

> **Дата:** 2026-02-07
> **Статус:** ✅ ЗАВЕРШЁН (все 11 задач, 4 фазы)
> **Цель:** ≤1e-5 parity с pyhf NumPy backend + максимальная скорость
> **Результат:** per-bin parity 1e-12, NLL parity 1e-10, Kahan overhead <5%, 45+ новых тестов

---

## 1. Анализ текущего состояния

### Что уже есть ✅

| Компонент | Статус | Детали |
|-----------|--------|--------|
| PreparedModel с кешами | ✅ Готово | `ln_factorials`, `obs_mask`, `constraint_const`, `n_zero_obs` |
| SIMD Poisson NLL | ✅ Готово | `wide::f64x4` с scalar `ln()` (bit-exact) |
| Accelerate backend | ✅ Готово | `vvlog()` + `vDSP`, feature-gated |
| Golden tests vs pyhf | ✅ Готово | 7 workspace'ов, 10+ точек каждый, 94MB goldens |
| Tolerance contract | ✅ Готово | `_tolerances.py`: NLL 1e-6 rel / 1e-8 abs |
| `--threads 1` = deterministic | ✅ Частично | Отключает Accelerate, но нет Kahan |
| Batch toy fitting (Rust) | ✅ Готово | `map_init()` с tape reuse |
| Optimizer diagnostic | ✅ Готово | Cross-eval подтвердил 1e-13 parity на objective |
| CUDA batch backend | ✅ Готово | GPU lockstep L-BFGS-B |

### Чего не хватает ❌

| Компонент | Важность | Описание |
|-----------|----------|----------|
| Kahan summation | Высокая | Naive `+=` в SIMD и scalar; при 10K+ бинах ошибка растёт |
| Explicit Parity Mode API | Высокая | `--threads 1` — единственный контроль; нет флага `--parity` |
| Gradient parity vs pyhf | Средняя | Phase 1 не включает; нужно для Phase 2 |
| Deterministic Rayon reduce | Средняя | `par_iter().collect()` не гарантирует порядок редукции |
| expected_data golden per-bin | Средняя | Тестируется через pytest.approx, нет per-bin report |
| Python batch toy API | Средняя | Rust есть, Python binding нет |
| Tolerance tier document | Низкая | Tolerance уровни в коде, нет формальной иерархии |
| CI golden gate | Низкая | Golden tests запускаются вручную |

---

## 2. Архитектура Parity/Fast режимов

### 2.1. Parity Mode (детерминированный, для валидации)

```
Цель: bit-exact результаты между запусками, максимальная близость к pyhf

Характеристики:
- threads = 1 (no Rayon parallelism)
- Accelerate = disabled (scalar SIMD only)
- Summation = Kahan compensated
- Float order = fixed (no work-stealing)
- FMA = disabled (если возможно через compiler flags)
```

### 2.2. Fast Mode (по умолчанию, для продакшена)

```
Цель: максимальная скорость при допустимых расхождениях

Характеристики:
- threads = auto (Rayon work-stealing)
- Accelerate = enabled (if available)
- Summation = naive (или Kahan по выбору)
- Float order = non-deterministic across runs
- Допуск: ≤1e-12 от Parity Mode (FP noise)
```

### 2.3. API Design

**Rust:**
```rust
pub enum EvalMode {
    Parity,  // Deterministic, Kahan, single-thread
    Fast,    // Rayon + SIMD/Accelerate, naive sum
}

impl PreparedModel {
    pub fn nll_with_mode(&self, params: &[f64], mode: EvalMode) -> f64;
}
```

**Python:**
```python
model.nll(params, mode="parity")   # Deterministic
model.nll(params, mode="fast")     # Default
model.nll(params)                  # = "fast"
```

**CLI:**
```bash
nextstat fit workspace.json --parity     # Parity mode
nextstat fit workspace.json              # Fast mode (default)
nextstat fit workspace.json --threads 4  # Fast, explicit threads
```

---

## 3. План доработок

### Phase A: Numerical Precision (Kahan + Parity Mode)

#### A1. Kahan summation в ns-compute
- **Файлы:** `ns-compute/src/simd.rs`, `ns-compute/src/accelerate.rs`
- **Задача:** Добавить `poisson_nll_simd_kahan()` и `poisson_nll_scalar_kahan()`
- **Паттерн:** Kahan compensated sum:
  ```rust
  let mut sum = 0.0;
  let mut comp = 0.0;  // compensation
  for term in terms {
      let y = term - comp;
      let t = sum + y;
      comp = (t - sum) - y;
      sum = t;
  }
  ```
- **Для SIMD:** Kahan на `f64x4` (4 компенсаторов), финальный horizontal Kahan

#### A2. EvalMode enum + dispatch
- **Файлы:** `ns-compute/src/lib.rs`, `ns-translate/src/pyhf/model.rs`
- **Задача:** `EvalMode` enum, `nll_with_mode()` dispatch, `--parity` CLI flag
- **Контракт:** Parity mode = Kahan + scalar + threads=1
- **Обратная совместимость:** `nll()` = `nll_with_mode(Fast)`

#### A3. Deterministic reduction для Fast mode
- **Файлы:** `ns-inference/src/batch.rs`
- **Задача:** Фиксированный порядок chunk reduction при Rayon
- **Паттерн:** `par_chunks(chunk_size).map(|chunk| sum_kahan(chunk)).reduce(|| 0.0, add)`
- **Результат:** Fast mode даёт одинаковый результат между запусками (при одинаковых threads)

### Phase B: Расширение контракта parity

#### B1. Gradient parity test vs pyhf
- **Файлы:** `tests/python/test_pyhf_validation.py`, `_tolerances.py`
- **Задача:** Добавить `test_gradient_parity_vs_pyhf()`
- **Метод:** pyhf finite-diff gradient vs NextStat AD gradient
- **Tolerance:** `GRADIENT_ATOL = 1e-6` (AD точнее FD, но FD шумит)
- **Покрытие:** simple + complex workspace, init + random points

#### B2. Per-bin expected_data golden report
- **Файлы:** `tests/python/test_expected_data_parity.py`
- **Задача:** Per-bin max|Δ| report + worst-bin identification
- **Output:** JSON с per-bin deltas для каждого workspace
- **Tolerance:** `EXPECTED_DATA_PER_BIN_ATOL = 1e-12` (чистая арифметика)

#### B3. Tolerance tier document
- **Файл:** `docs/pyhf-parity-contract.md`
- **Содержание:**

| Уровень | Величина | Parity Tol | Fast Tol | Обоснование |
|---------|----------|-----------|----------|-------------|
| expected_data per-bin | Bin-level | 1e-12 | 1e-10 | Чистая арифметика |
| expected_data L2 | Vector | 1e-10 | 1e-8 | Accumulation noise |
| NLL value | Scalar | 1e-10 | 1e-6 | Sum over bins |
| Gradient | Vector | 1e-8 | 1e-6 | AD vs FD |
| Fit params | Vector | 1e-5 | 2e-4 | Optimizer surface |
| Uncertainties | Vector | 1e-4 | 5e-4 | Hessian sensitivity |

### Phase C: Python API + Integration

#### C1. Python batch toy fitting
- **Файлы:** `bindings/ns-py/src/lib.rs`
- **Задача:** Expose `fit_toys_batch()` + `fit_toys_batch_gpu()`
- **API:**
  ```python
  results = nextstat.fit_toys_batch(model, n_toys=1000, seed=42)
  results_gpu = nextstat.fit_toys_batch(model, n_toys=1000, seed=42, device="cuda")
  ```

#### C2. Python parity mode
- **Файлы:** `bindings/ns-py/src/lib.rs`
- **Задача:** `mode="parity"` parameter в `nll()`, `grad_nll()`, `fit()`
- **Контракт:** Python parity = Rust parity (bit-exact)

#### C3. CI golden gate
- **Файл:** `.github/workflows/pyhf-parity.yml`
- **Задача:** GitHub Action на каждый PR:
  1. Build ns-py wheel
  2. Run `test_pyhf_model_zoo_goldens.py` (no pyhf dependency)
  3. Run `test_pyhf_validation.py` (requires pyhf)
  4. Gate: fail PR если tolerance exceeded

### Phase D: Performance Validation

#### D1. Benchmark: Parity vs Fast throughput
- **Файл:** `tests/python/benchmark_parity_vs_fast.py`
- **Задача:** Измерить overhead Kahan summation в Parity mode
- **Ожидание:** <5% overhead (Kahan — 4 FLOP вместо 1)

#### D2. Validate Fast mode tolerance vs Parity
- **Файл:** `tests/python/test_fast_vs_parity_tolerance.py`
- **Задача:** Для всех workspace'ов: `|NLL_fast - NLL_parity| < 1e-12`
- **Покрытие:** simple, complex, tHu, tttt, postFit_PTV

---

## 4. Приоритизация

```
Phase A (Numerical Precision):  КРИТИЧНО — фундамент для всего остального
  A1 Kahan summation           [P0] — 2-3 часа
  A2 EvalMode + dispatch       [P0] — 2-3 часа
  A3 Deterministic reduction   [P1] — 1-2 часа

Phase B (Contract Extension):   ВАЖНО — укрепление гарантий
  B1 Gradient parity test      [P1] — 1-2 часа
  B2 Per-bin expected golden   [P1] — 1 час
  B3 Tolerance tier doc        [P2] — 30 мин

Phase C (API + Integration):    ПОЛЕЗНО — developer experience
  C1 Python batch toy API      [P1] — 2 часа
  C2 Python parity mode        [P2] — 1 час
  C3 CI golden gate            [P2] — 1-2 часа

Phase D (Performance):          ОПЦИОНАЛЬНО — confidence
  D1 Benchmark parity vs fast  [P2] — 1 час
  D2 Validate fast vs parity   [P2] — 1 час
```

---

## 5. Риски и mitigation

| Риск | Вероятность | Mitigation |
|------|-------------|------------|
| Kahan overhead >5% | Низкая | Kahan только в Parity mode; Fast = naive |
| SIMD Kahan сложный | Средняя | Начать со scalar Kahan, потом SIMD |
| pyhf тоже имеет FP noise | Низкая | Golden tests фиксируют конкретные числа |
| Rayon deterministic reduce медленнее | Низкая | `par_chunks` — фиксированная стоимость |
| `--parity` flag breaking changes | Низкая | Additive API, default = Fast |

---

## 6. Definition of Done

- [ ] `nextstat fit --parity workspace.json` даёт bit-exact результаты между запусками
- [ ] `NLL_parity` ≤ 1e-10 от pyhf NumPy на всех fixtures
- [ ] `NLL_fast` ≤ 1e-6 от pyhf NumPy (существующий контракт)
- [ ] `|NLL_fast - NLL_parity|` ≤ 1e-12
- [ ] Gradient parity test проходит на simple + complex workspaces
- [ ] CI gate блокирует PR при нарушении tolerance
- [ ] Python API: `nll(params, mode="parity")` работает
- [ ] Overhead Parity mode < 5% vs Fast mode
