---
title: "Optimizer Convergence & Best-NLL Philosophy"
status: stable
---

# Optimizer Convergence & Best-NLL Philosophy

## TL;DR

NextStat использует L-BFGS-B и по умолчанию стремится к **наилучшему минимуму NLL** (best-NLL).
Расхождения с pyhf в bestfit параметрах на больших моделях (>100 params) —
**ожидаемое и задокументированное** поведение, а не баг.

---

## 1. Позиция: Best-NLL по умолчанию

### Принцип

NextStat **не ограничивает** оптимизатор ради совпадения с конкретным инструментом.
Если L-BFGS-B находит более глубокий минимум, чем SLSQP pyhf — это корректный результат.

### Обоснование

1. **Objective parity подтверждена**: NLL вычисления NextStat и pyhf совпадают на уровне 1e-9 — 1e-13
   при одинаковых параметрах (cross-eval). Модели идентичны.

2. **Различия — в оптимизаторе, не в модели**: pyhf SLSQP останавливается с ||grad|| = 1.4–4.6
   на больших моделях, не достигая истинного минимума. L-BFGS-B с quasi-Newton Hessian approximation
   продолжает спуск.

3. **Кросс-валидация**: pyhf(init=NS_hat) достигает того же NLL, что NS → минимум доступен обоим.
   NS(init=pyhf_hat) уходит ещё ниже → NS робастнее к начальной точке.

4. **Multi-start анализ**: SLSQP multi-start (20 starts) на больших моделях converges
   в <25% случаев; лучший результат всё равно хуже NS.

### Масштаб расхождений

| Модель | Параметры | ΔNLL (NS - pyhf) | Причина |
|--------|-----------|-------------------|---------|
| simple_workspace | 2 | 0.0 | Оба converge |
| complex_workspace | 9 | 0.0 | Оба converge |
| tchannel | 184 | -0.01 to -0.08 | pyhf SLSQP premature stop |
| tHu | ~200 | -0.08 | pyhf SLSQP premature stop |
| tttt | 249 | -0.01 | pyhf SLSQP premature stop |

> Отрицательный ΔNLL означает: NextStat находит **лучший** (более низкий) минимум.

---

## 2. Уровни parity

### Level 1: Objective Parity (P0, обязательный)

NLL(params) совпадает между NextStat и pyhf при одинаковых params.

- Tolerance: rtol=1e-6, atol=1e-8 (см. `docs/plans/standards.md` §2.2)
- Проверяется: golden tests на fixture workspaces
- Статус: **выполнен** для всех 7 fixture workspaces

### Level 2: Fit Parity (P1, conditional)

Bestfit параметры совпадают в пределах допусков.

- Tolerance: atol=2e-4 на параметры, atol=5e-4 на uncertainties
- **Ожидаемое поведение**: полное совпадение на малых моделях (<50 params),
  расхождения на больших моделях из-за разных оптимизаторов
- **Не является дефектом**, если NS NLL <= pyhf NLL

### Level 3: Optimizer Compat (не реализован, не планируется)

Специальная деградация оптимизатора для совпадения с SLSQP. **Отвергнуто** —
это искусственное ограничение без научной ценности.

---

## 3. Как верифицировать

### Для пользователей

```python
import nextstat, json

ws = json.load(open("workspace.json"))
model = nextstat.from_pyhf(json.dumps(ws))
result = nextstat.fit(model)

# NLL в минимуме — чем ниже, тем лучше
print(f"NLL: {result.nll}")
print(f"||grad||: ...")  # доступно через диагностику
```

### Для разработчиков (parity check)

```bash
# Objective parity (должен пройти всегда)
make pyhf-audit-nll

# Fit parity (может расходиться на больших моделях — ожидаемо)
make pyhf-audit-fit
```

### Диагностический скрипт

```bash
# Cross-eval: проверяет что NS и pyhf вычисляют одинаковый NLL в точках друг друга
python tests/diagnose_optimizer.py workspace.json
```

---

## 4. Warm-start для pyhf-совместимости

Если **конкретный** use case требует совпадения с pyhf (например, воспроизведение
опубликованного результата), можно использовать warm-start от точки pyhf:

```python
import pyhf, nextstat, json

# 1. Fit в pyhf
ws = json.load(open("workspace.json"))
model = pyhf.Workspace(ws).model()
pyhf_pars, _ = pyhf.infer.mle.fit(model.config.suggested_init(), model, return_uncertainties=True)

# 2. Warm-start в NextStat от точки pyhf
ns_model = nextstat.from_pyhf(json.dumps(ws))
result = nextstat.fit(ns_model, init_pars=pyhf_pars.tolist())
# result.nll <= pyhf NLL (гарантировано)
```

Это **не** режим совместимости — NS всё равно может уйти ниже.
Но если SLSQP действительно нашёл минимум, NS подтвердит это.

---

## 5. L-BFGS-B vs SLSQP: технические детали

| Аспект | L-BFGS-B (NextStat) | SLSQP (pyhf/scipy) |
|--------|---------------------|---------------------|
| Hessian | Quasi-Newton (m=10 history) | Rank-1 update |
| Bounds | Native box constraints | Native box constraints |
| Constraints | Через reparameterization | Native linear/nonlinear |
| Convergence | ||proj_grad|| < ftol | ||grad|| threshold (tunable) |
| Scaling | O(m*n) per iteration | O(n^2) per iteration |
| Большие модели (>100 params) | Robust | Часто premature stop |

L-BFGS-B с m=10 хранит 10 пар (s, y) для аппроксимации Hessian —
это даёт лучшую curvature information при N>100, чем rank-1 update SLSQP.

---

## References

- Optimizer diagnostic report: `audit/2026-02-07_pyhf-optimizer-diagnostic.md`
- Precision standards: `docs/plans/standards.md`
- Parity modes: `docs/plans/2026-02-07_pyhf-spec-parity-plan.md`
- Diagnostic scripts: `tests/diagnose_optimizer.py`, `tests/repeat_mle_fits.py`
