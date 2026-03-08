---
title: "HEP GVM Stable-First Быстрый старт"
status: stable-first
---

# HEP GVM Stable-First Быстрый старт

Цель: примерно за 5 минут получить реальный результат комбинации скалярных
измерений на committed example bundle, не собирая руками ни JSON spec, ни
длинную CLI-команду с таблицами.

Это самый короткий поддерживаемый путь для stable-first GVM subset:

- `nextstat combine-measurements-build-spec`
- `nextstat combine-measurements`
- `nextstat combine-measurements-calibrate`
- `nextstat combine-measurements-calibrate-study`

Исходный bundle лежит здесь:

- [docs/examples/gvm-stable-first](/Users/andresvlc/WebDev/nextstat.io/docs/examples/gvm-stable-first)

One-command golden path:

```bash
make gvm-stable-first-example
```

Эта команда записывает spec, fit, calibration и calibration-study результаты
в `tmp/gvm-stable-first-example/`.

## 1. Собрать канонический spec из manifest bundle

```bash
nextstat combine-measurements-build-spec \
  --manifest docs/examples/gvm-stable-first/manifest.yaml \
  --output /tmp/gvm-spec.json
```

Если нужен более низкоуровневый контроль, тот же command по-прежнему принимает
прямые флаги `--measurements`, `--stat-covariance`, `--systematics`,
`--correlations`.

## 2. Запустить stable fit

```bash
nextstat combine-measurements \
  --input /tmp/gvm-spec.json \
  --output /tmp/gvm-result.json \
  --solver auto \
  --threads 1
```

Смотреть:

- `mu_hat`
- `confidence_interval`
- `goodness_of_fit`
- `diagnostics.requested_solver`
- `diagnostics.effective_solver`

## 3. Запустить детерминистическую toy calibration

```bash
nextstat combine-measurements-calibrate \
  --input /tmp/gvm-spec.json \
  --output /tmp/gvm-calibration.json \
  --solver auto \
  --n-toys 32 \
  --seed 42 \
  --threads 1
```

## 4. Запустить repeated-seed stability

```bash
nextstat combine-measurements-calibrate-study \
  --input /tmp/gvm-spec.json \
  --output /tmp/gvm-study.json \
  --solver auto \
  --n-toys 32 \
  --seeds 42,43 \
  --threads 1
```

## 5. Эквивалентный Python путь

```python
from nextstat import hep

spec = hep.build_measurement_combination_spec(
    # Прямые пути к таблицам всё ещё поддерживаются.
    "docs/examples/gvm-stable-first/measurements.csv",
    "docs/examples/gvm-stable-first/stat_covariance.csv",
    poi="mu",
    systematics_table="docs/examples/gvm-stable-first/systematics.csv",
    correlations_table="docs/examples/gvm-stable-first/correlations.csv",
)

manifest_spec = hep.build_measurement_combination_spec_from_manifest(
    "docs/examples/gvm-stable-first/manifest.yaml"
)

result = hep.combine_measurements(spec, solver="auto")
calibration = hep.calibrate_measurements(spec, solver="auto", n_toys=32, seed=42)
study = hep.calibrate_measurements_study(spec, solver="auto", n_toys=32, seeds=[42, 43])
```

## Stable scope

Stable-first subset:

- `build-spec --manifest`
- `build-spec`
- `combine`
- `calibrate`
- `calibrate-study`

По-прежнему research-grade:

- `scenario-study`
- `calibration-campaign`
- solver parity
- cached reporting / brief / family / portfolio layers

Для полного контекста:

- [HEP GVM-комбинации измерений](/Users/andresvlc/WebDev/nextstat.io/docs/ru/tutorials/hep-gvm-measurement-combinations.md)
- [GVM Stable-First Support Matrix](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07.md)
