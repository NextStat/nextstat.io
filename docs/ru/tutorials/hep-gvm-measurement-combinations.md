---
title: "HEP GVM-комбинации измерений"
status: stable-first
---

# HEP GVM-комбинации измерений

Это практическое русскоязычное руководство по stable-first движку NextStat для
комбинации скалярных измерений с коррелированными систематиками и
`error_on_error`, где более широкие advanced/reporting layers остаются
research-grade.

Stable-first статус теперь распространяется на базовый inference subset:

- `nextstat combine-measurements`
- `nextstat combine-measurements-calibrate`
- `nextstat combine-measurements-calibrate-study`
- `nextstat.hep.combine_measurements(...)`
- `nextstat.hep.calibrate_measurements(...)`
- `nextstat.hep.calibrate_measurements_study(...)`

Scenario/campaign/parity и более высокие reporting layers ниже по тексту
остаются research-grade.

Используйте этот workflow, когда у вас уже есть:

- набор редуцированных измерений
- статистическая ковариационная матрица
- систематические источники с per-source корреляционными матрицами

Не используйте этот путь для HistFactory или pyhf workspace fits. Для них
остаются `nextstat fit`, `nextstat hypotest` и остальные workspace-команды.

## Рекомендуемый stable-first путь для входных данных

Если ваш source of truth — таблица или spreadsheet, не начинайте с ручной
сборки JSON spec. Правильный stable-first golden path такой:

1. подготовить CSV/TSV таблицы
2. один раз собрать канонический JSON spec
3. запускать fit, calibration и repeated-seed study уже на сгенерированном spec

Stable-first entry points для табличного входа:

- CLI: `nextstat combine-measurements-build-spec`
- Python: `nextstat.hep.build_measurement_combination_spec(...)`
- Python manifest wrapper: `nextstat.hep.build_measurement_combination_spec_from_manifest(...)`

Канонический runnable bundle в этом repo:

- `docs/examples/gvm-stable-first/`

Самый короткий stable-first path внутри этого bundle:

- `docs/examples/gvm-stable-first/manifest.yaml`
- `make gvm-stable-first-example`

Если вы проводите первую внешнюю валидацию с физиком или analysis contact, не
собирайте handoff вручную. Используйте готовый maintainer kit:

- `docs/guides/gvm-external-validation-kit.md`
- `docs/guides/gvm-external-validator-outreach-pack.md`
- `docs/guides/gvm-external-validation-tracker-template.md`
- `docs/examples/gvm-stable-first/external-validator-invite-template.md`
- `docs/examples/gvm-stable-first/external-validation-report-template.md`

Так сохраняется канонический JSON contract на runtime-слое, но пользовательский
вход становится ближе к тому, как реальные HEP combinations обычно живут в
таблицах.

## Ссылка на статью

L. Canonero and G. Cowan, "Combination of measurements and the BLUE method
generalized by allowing for errors in the error assignments," *Eur. Phys. J. C*
**85**, 156 (2025).

NextStat — **единственная известная публичная реализация** GVM-правдоподобия.
Ни один другой HEP-пакет (pyhf, ROOT/RooFit и др.) его не предоставляет.

## Что дает GVM workflow

- likelihood-based комбинацию скалярных измерений
- поддержку `error_on_error` через Gamma Variance Model
- три независимых пути решения для перекрёстной валидации
- поправку Lawley/Bartlett высшего порядка O(ε²) с быстрым (Woodbury) и референсным путём
- доверительные интервалы профильного правдоподобия с экспоненциальным брэкетингом и бисекцией
- детерминистическую toy-калибровку
- scenario studies по нескольким `error_on_error`
- repeated-seed calibration campaigns
- 11-уровневую пирамиду калибровки от одного фита до стабильности портфеля
- solver-parity артефакты между:
  - `numerical-paper`
  - `analytic-perturbative`

## Дефолтный solver contract

По умолчанию используется `auto` и для stable-first core, и для более широких
research-grade extensions.

`auto` означает:

1. сначала пробуем perturbative paper path
2. если не проходит validity gate по Eq. `(29)/(60)`, автоматически падаем в
   paper-faithful numerical path в original correlated `theta_s^i` basis

Этот контракт одинаков для:

- `nextstat combine-measurements`
- `nextstat combine-measurements-calibrate`
- `nextstat combine-measurements-calibrate-study`
- `nextstat combine-measurements-scenario-study`
- `nextstat combine-measurements-calibration-campaign`
- `nextstat.hep.*`

Для reproducible parity используйте `--threads 1`.

## Минимальный входной spec

Канонический input artifact: `nextstat_measurement_combination_v0`.

Пример:

```json
{
  "schema_version": "nextstat_measurement_combination_v0",
  "poi": "mu",
  "measurements": [
    { "name": "atlas_ljets", "value": 172.40 },
    { "name": "cms_ljets", "value": 172.62 }
  ],
  "stat_covariance": [
    [0.04, 0.00],
    [0.00, 0.05]
  ],
  "systematics": [
    {
      "name": "b-JES",
      "magnitudes": [0.30, 0.28],
      "corr": [
        [1.0, 0.8],
        [0.8, 1.0]
      ],
      "error_on_error": 0.10,
      "aux_mean": 0.0
    },
    {
      "name": "hadronization",
      "magnitudes": [0.20, 0.18],
      "corr": [
        [1.0, 1.0],
        [1.0, 1.0]
      ],
      "error_on_error": 0.00,
      "aux_mean": 0.0
    }
  ]
}
```

Смысл полей:

- `measurements[].value`: скалярные observed values
- `stat_covariance`: статистическая ковариация между измерениями
- `systematics[].magnitudes`: assigned systematic magnitude для каждого измерения
- `systematics[].corr`: per-source correlation matrix между измерениями
- `systematics[].error_on_error`: относительная неопределенность размера систематики
- `systematics[].aux_mean`: optional auxiliary offset, обычно `0.0`

Важно:

- v1 использует normalized convention из RFC
- raw `v_s` во входной схеме нет
- `magnitudes` задаются сразу как assigned systematic errors

## Stable-first табличный bundle

Если вы стартуете не из JSON, а из таблиц, минимальный bundle такой:

- `measurements.csv`
  - колонки: `name,value`
- `stat_covariance.csv`
  - именованная квадратная матрица с row/column measurement names
- optional `systematics.csv`
  - колонки: `systematic,measurement,magnitude,error_on_error,aux_mean`
- optional `correlations.csv`
  - колонки: `systematic,row_measurement,col_measurement,corr`

Если `correlations.csv` отсутствует, для каждого systematic source по
умолчанию используется identity correlation.

Пример CLI:

```bash
nextstat combine-measurements-build-spec \
  --manifest docs/examples/gvm-stable-first/manifest.yaml \
  --output spec.json
```

Прямые флаги с таблицами по-прежнему поддерживаются, если manifest wrapper не нужен.

Пример Python:

```python
from nextstat import hep

spec = hep.build_measurement_combination_spec(
    "docs/examples/gvm-stable-first/measurements.csv",
    "docs/examples/gvm-stable-first/stat_covariance.csv",
    poi="mu",
    systematics_table="docs/examples/gvm-stable-first/systematics.csv",
    correlations_table="docs/examples/gvm-stable-first/correlations.csv",
)

manifest_spec = hep.build_measurement_combination_spec_from_manifest(
    "docs/examples/gvm-stable-first/manifest.yaml"
)
```

## Минимальный scenario config

Scenario studies и calibration campaigns используют второй JSON artifact:

```json
{
  "schema_version": "nextstat_measurement_combination_scenarios_v0",
  "scenarios": [
    {
      "name": "bjes_0p1",
      "error_on_error": [
        { "systematic": "b-JES", "value": 0.10 }
      ]
    },
    {
      "name": "bjes_0p3",
      "error_on_error": [
        { "systematic": "b-JES", "value": 0.30 }
      ]
    },
    {
      "name": "theory_core_0p2",
      "error_on_error": [
        { "systematic": "hadronization", "value": 0.20 }
      ]
    }
  ]
}
```

## CLI workflow

### 1. Сборка spec из таблиц

```bash
nextstat combine-measurements-build-spec \
  --manifest docs/examples/gvm-stable-first/manifest.yaml \
  --output spec.json
```

### 2. Комбинация одного spec

```bash
nextstat combine-measurements \
  --input spec.json \
  --output result.json \
  --ci-level 0.68 \
  --solver auto \
  --threads 1
```

Что смотреть в `result.json`:

- `mu_hat`
- `confidence_interval`
- `goodness_of_fit`
- `optimizer.method`
- `diagnostics.requested_solver`
- `diagnostics.effective_solver`
- `diagnostics.perturbative_validity`
- `diagnostics.bartlett`

Типичные solver outcomes:

- `analytic_perturbative_order_eps2`
- `numerical_profile_gvm_original_theta`
- `closed_form_blue`, когда все `error_on_error == 0`

Если вы запрашиваете `--solver auto`, смотрите на
`diagnostics.requested_solver` и `diagnostics.effective_solver`, чтобы понять,
остался ли runtime на perturbative fast path или откатился на paper-faithful
numerical path.

### 3. Toy calibration

```bash
nextstat combine-measurements-calibrate \
  --input spec.json \
  --output calibration.json \
  --ci-level 0.68 \
  --solver auto \
  --n-toys 256 \
  --seed 42 \
  --threads 1
```

Ключевые поля:

- `reference`
- `summary.mean_q`
- `summary.mean_q_star`
- `summary.mean_sigma`
- `summary.mean_sigma_star`
- `summary.mean_sigma_star_to_sigma_ratio`
- `summary.bartlett_improves_mean_q`

### 4. Repeated-seed calibration study

```bash
nextstat combine-measurements-calibrate-study \
  --input spec.json \
  --output calibration_study.json \
  --ci-level 0.68 \
  --solver auto \
  --n-toys 256 \
  --seeds 42,43,44 \
  --threads 1
```

### 5. Scenario study

```bash
nextstat combine-measurements-scenario-study \
  --input spec.json \
  --scenarios scenarios.json \
  --output scenario_study.json \
  --ci-level 0.68 \
  --solver auto \
  --threads 1
```

### 6. Calibration campaign

```bash
nextstat combine-measurements-calibration-campaign \
  --input spec.json \
  --scenarios scenarios.json \
  --output campaign.json \
  --ci-level 0.68 \
  --solver auto \
  --n-toys 128 \
  --seeds 42,43,44 \
  --threads 1
```

### 6. Summary / Markdown review

```bash
nextstat combine-measurements-calibration-campaign-summarize \
  --input campaign.json \
  --output campaign_summary.json

nextstat combine-measurements-calibration-campaign-summarize \
  --input campaign.json \
  --format markdown \
  --output campaign_summary.md
```

### 7. Solver parity

Scenario-study parity:

```bash
nextstat combine-measurements-solver-parity-scenario-study \
  --input spec.json \
  --scenarios scenarios.json \
  --output parity_scenarios.json \
  --lhs-solver numerical-paper \
  --rhs-solver analytic-perturbative \
  --threads 1
```

Calibration-campaign parity:

```bash
nextstat combine-measurements-solver-parity-calibration-campaign \
  --input spec.json \
  --scenarios scenarios.json \
  --output parity_campaign.json \
  --lhs-solver numerical-paper \
  --rhs-solver analytic-perturbative \
  --n-toys 128 \
  --seeds 42,43,44 \
  --threads 1
```

## Python workflow

### 1. Комбинация одного spec

```python
import json
import nextstat

spec = json.load(open("spec.json"))
result = nextstat.hep.combine_measurements(
    spec,
    ci_level=0.68,
    solver="auto",
)

print(result["mu_hat"])
print(result["optimizer"]["method"])
print(result["diagnostics"]["bartlett"]["supported"])
```

### 2. Toy calibration

```python
report = nextstat.hep.calibrate_measurements(
    spec,
    ci_level=0.68,
    solver="auto",
    n_toys=256,
    seed=42,
)
```

### 3. Scenario study и campaign

```python
scenarios = json.load(open("scenarios.json"))

scenario_report = nextstat.hep.study_measurement_combination_scenarios(
    spec,
    scenarios,
    ci_level=0.68,
    solver="auto",
)

campaign = nextstat.hep.calibrate_measurement_combination_scenarios(
    spec,
    scenarios,
    ci_level=0.68,
    solver="auto",
    n_toys=128,
    seeds=[42, 43, 44],
)
```

### 4. Summary / render

```python
summary = nextstat.hep.summarize_measurement_combination_calibration_campaign(campaign)
markdown = nextstat.hep.render_measurement_combination_calibration_campaign_summary(summary)
```

### 5. Solver parity

```python
parity = nextstat.hep.compare_measurement_combination_scenario_study_solvers(
    spec,
    scenarios,
    ci_level=0.68,
    lhs_solver="numerical-paper",
    rhs_solver="analytic-perturbative",
)
```

## Как выбирать solver

Используйте:

- `auto`
  - дефолт
  - лучший выбор для большинства случаев
  - сначала perturbative path, затем safe fallback
- `numerical-paper`
  - paper-faithful reference path в original correlated `theta_s^i` basis
  - лучший выбор для parity studies и paper comparisons
- `analytic-perturbative`
  - прямое использование Eq. `(21)-(28)` / Appendix B approximation
  - invalid cases не fallback’ятся, а отклоняются
- `numerical`
  - compatibility path для reduced-basis numerical solver

## Рекомендуемый порядок работы

1. `combine-measurements --solver auto --threads 1`
2. смотрим `diagnostics.perturbative_validity`
3. при сомнениях сравниваем с `numerical-paper`
4. запускаем `...-calibrate`
5. затем `...-calibrate-study`
6. затем `...-scenario-study`
7. `...-calibration-campaign` используем только когда нужен единый artifact по
   scenarios и seeds одновременно

## Правила воспроизводимости

Для research-grade parity:

- используйте `--threads 1`
- всегда фиксируйте `--seed` или `--seeds`
- держите committed JSON artifacts под версионным контролем
- используйте cached post-processing commands, когда не нужно заново гонять fits
  и toys

Для published evidence используйте не только parity/performance snapshots, но и
committed robustness snapshots:

- [GVM Benchmark Snapshot](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-measurement-combine-snapshot-2026-03-07.md)
- [GVM NumericalPaper Robustness Snapshot](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-numerical-paper-robustness-snapshot-2026-03-07.md)
- [GVM Stable-Surface Readiness Memo](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-stable-surface-readiness-2026-03-07.md)
- [GVM Stable-Surface Support Policy](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-stable-surface-support-policy-2026-03-07.md)
- [GVM Stable-First Promotion Decision](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-stable-first-decision-2026-03-07.md)
- [GVM Stable-First Support Matrix](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07.md)
- [GVM Stable-First Release Notes](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-stable-first-release-notes-2026-03-07.md)

## Troubleshooting

### `analytic perturbative path is outside the Eq. (29)/(60) validity radius`

Это значит, что perturbative approximation для данного случая не считается
надежной.

Что делать:

- использовать `solver="auto"` или `--solver auto`
- либо явно переключиться на `numerical-paper`

### Non-PSD source correlation matrices

Опубликованные HEP inputs могут содержать raw per-source matrices, которые не
являются строго PSD. NextStat сохраняет raw inputs в BLUE covariance construction
и применяет minimal regularization только там, где нужна инверсия.

Смотрите:

- `diagnostics.corr_regularization_deltas`

## Три независимых солвера

NextStat предоставляет три пути решения для перекрёстной валидации:

| Солвер | Метод | Когда использовать |
|--------|-------|--------------------|
| `analytic-perturbative` | Пертурбативное разложение Eq. (21)-(28) / Appendix B до O(ε²) | Быстрый путь, малые-умеренные error_on_error |
| `numerical-paper` | Численное профилирование в original correlated θ basis | Референсный путь, parity studies, большие error_on_error |
| `numerical` | Reduced-basis с QR-декомпозицией | Совместимость, быстрейший численный |
| `auto` (по умолчанию) | Пертурбативный → fallback на `numerical-paper` вне радиуса валидности | Лучший выбор для большинства задач |

Когда все `error_on_error` равны нулю, GVM вырождается в стандартный BLUE, и
результат вычисляется аналитически — без итераций.

## Поправка Бартлетта-Лоули

Статистика отношения профильных правдоподобий GVM может отклоняться от χ² при
нетривиальных error_on_error. NextStat вычисляет поправку O(ε²):

- **Быстрый (Woodbury)** — формула Шермана-Моррисона-Вудбери, сложность O(N·K)
- **Референсный** — прямое плотноматричное вычисление для перекрёстной валидации

Скорректированная статистика `q*` и доверительный интервал `σ*` доступны в
`diagnostics.bartlett` каждого результата.

## 11-уровневая пирамида калибровки

| Ур. | Артефакт | Что добавляет |
|-----|----------|---------------|
| 1 | Fit | Одна комбинация: μ̂, CI, GoF, Bartlett |
| 2 | Calibrate | Toy-калибровка (один seed): σ*/σ |
| 3 | Calibrate Study | Стабильность калибровки по нескольким seed |
| 4 | Scenario Study | Sweep по error_on_error |
| 5 | Campaign | Сценарии × seed-ы в одном артефакте |
| 6 | Digest / Summary | Компактные метрики кампании |
| 7 | Brief | Агрегация по нескольким кампаниям |
| 8 | Family Report | Сравнение по нескольким brief |
| 9 | Family Matrix | Попарная матрица доминирования |
| 10 | Portfolio | Портфельный вид по нескольким матрицам |
| 11 | Portfolio Stability | Стабильность портфельных выводов по сеткам seed |

Каждый артефакт — версионированный JSON с `schema_version`, каждый уровень может
генерировать Markdown. 17 CLI-команд и 28 функций Python API покрывают все уровни.

## Производительность

Criterion-бенчмарки (однопоточно, release):

| Фикстура | Солвер | Время |
|-----------|--------|-------|
| Top-mass из статьи (15 × 22) | auto | 101 мкс |
| Синтетический (32 × 24) | analytic-perturbative | 13.4 мс |
| Синтетический (32 × 24) | numerical-paper | 44.6 мс |
| Синтетический (64 × 48) | analytic-perturbative | 124.7 мс |

## Тестовое покрытие

- **122 юнит-теста на Rust** — все пути солверов, граничные случаи, уровни калибровки
- **37 golden-file фикстур** (JSON + Markdown) для регрессионного тестирования
- **Интеграционные CLI-тесты** для всех 17 команд
- **Тесты Python API** через `test_hep_module_api.py`
- Фикстура top-mass из Table 1 Canonero & Cowan

## Связанные материалы

- Англоязычный tutorial: `docs/tutorials/hep-gvm-measurement-combinations.md`
- Справочник CLI: `docs/references/cli.md`
- Справочник Python API: `docs/references/python-api.md`
- RFC: `docs/rfcs/research-grade-measurement-combinations.md`
- Stable-surface memo: `docs/benchmarks/gvm-stable-surface-readiness-2026-03-07.md`
- Stable-surface policy: `docs/benchmarks/gvm-stable-surface-support-policy-2026-03-07.md`
- Stable-first decision: `docs/benchmarks/gvm-stable-first-decision-2026-03-07.md`
- Stable-first support matrix: `docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07.md`
- Stable-first release notes: `docs/benchmarks/gvm-stable-first-release-notes-2026-03-07.md`
