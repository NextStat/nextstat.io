---
title: "ROOT/TRExFitter Parity (HistFactory) — NextStat Validation"
status: draft
---

# ROOT/TRExFitter Parity (HistFactory) — NextStat Validation

Цель: прогнать одни и те же HistFactory модели через **ROOT/HistFactory** (эталон в HEP‑экосистеме) и через **NextStat**, сравнить расхождения и померить скорость.

На практике TRExFitter обычно является *генератором* HistFactory XML + ROOT histograms и/или RooWorkspace, а математика “движка” живёт в ROOT/RooFit/RooStats. Поэтому минимальный “эталонный” контур — это ROOT `hist2workspace` + RooFit профилирование.

## Prerequisites

1) ROOT (с HistFactory/RooStats) доступен в PATH:

```bash
command -v root
command -v hist2workspace
```

2) Python bindings NextStat собраны/установлены:

```bash
cd bindings/ns-py
maturin develop --release
```

3) Для конвертации HistFactory XML ↔ pyhf JSON нужен `uproot`:

```bash
pip install -e "bindings/ns-py[validation]"
```

## Apex2 workflow (Planning → Exploration → Execution → Verification)

Ниже самый воспроизводимый путь, который удобно запускать на кластере (где есть ROOT и TRExFitter).

### Planning (окружение и зависимости)

Минимально нужно:
- `root` + `hist2workspace` в `PATH`
- Python 3 + зависимости для валидации (`pyhf`, `uproot`, и python bindings NextStat)

Рекомендуемая проверка prereqs (быстро, без прогонов):

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_master_report.py --root-prereq-only
```

Если в кластере нет `.venv`, используй любой эквивалентный Python (conda/venv/модуль), но важно:
- `PYTHONPATH=bindings/ns-py/python`
- `pip install -e "bindings/ns-py[validation]"` выполнен в этом env

### Exploration (найти тестовые модели)

Тестовые модели для ROOT/TRExFitter в этом контуре это HistFactory экспорты с `combination.xml`.

Если у тебя есть директория с экспортами TRExFitter (или любыми HistFactory export-ами), можно:

1) Сгенерировать cases JSON (наиболее контролируемо, удобно для CI/архива):

```bash
./.venv/bin/python tests/generate_apex2_root_cases.py \
  --search-dir /abs/path/to/trex/output \
  --out tmp/apex2_root_cases.json \
  --include-fixtures \
  --absolute-paths
```

`name` каждого кейса генерится как относительный путь папки экспорта (от `--search-dir`), чтобы избежать коллизий (в больших наборах часто повторяются одинаковые имена подпапок).

2) Либо не генерировать вручную, а дать директорию прямо мастер-раннеру (см. Execution).

### Execution (прогоны)

#### Вариант A: один master-report (pyhf + ROOT-suite)

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_master_report.py \
  --root-search-dir /abs/path/to/trex/output \
  --root-include-fixtures \
  --root-cases-absolute-paths
```

Артефакт:
- `tmp/apex2_master_report.json`

Внутри будет:
- `pyhf.status` (`ok`/`fail`)
- `root.status` (`ok`/`fail`/`skipped`)
- ссылки на `tmp/apex2_pyhf_report.json` и `tmp/apex2_root_suite_report.json`

#### Вариант B: отдельно ROOT-suite (если нужно фокусно)

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_root_suite_report.py \
  --cases tmp/apex2_root_cases.json \
  --keep-going \
  --out tmp/apex2_root_suite_report.json
```

### Verification (интерпретация и “почему”)

1) Первичный “зеленый/красный” сигнал:
- `pyhf.status == ok` значит NLL/expected_data совпадают с эталоном `pyhf`
- `root.status == ok` значит q(mu) профиль совпал с ROOT в заданных допусках
- `root.status == skipped` значит не было prereqs (например, нет `hist2workspace` или `uproot`)

2) Если ROOT-suite дал `fail`, в `tmp/apex2_root_suite_report.json` для каждого кейса есть:
- `run_dir` (папка с артефактами одного прогона)
- `summary_path`
- `diff.max_abs_dq_mu` и `diff.d_mu_hat`

3) Для разбора расхождений по конкретному `run_dir` (без ROOT) используй:

```bash
./.venv/bin/python tests/explain_root_vs_nextstat_profile_diff.py \
  --run-dir /abs/path/to/tmp/root_parity_suite/<case>/run_<timestamp>
```

## 1) Проверка профилирования q(mu) vs ROOT

### Вариант A: стартуем от pyhf JSON (fixtures)

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/validate_root_profile_scan.py \
  --pyhf-json tests/fixtures/simple_workspace.json \
  --measurement GaussExample \
  --start 0.0 --stop 5.0 --points 51
```

Скрипт:
- экспортирует workspace в HistFactory XML + `data.root` через `pyhf.writexml`
- строит RooWorkspace через `hist2workspace`
- делает free fit и fixed‑POI fits в ROOT → q(mu)
- делает `nextstat.infer.profile_scan` на той же сетке mu
- печатает JSON summary и пишет артефакты в `tmp/root_parity/...`

### Вариант B: стартуем от HistFactory Combination XML (например, экспорт TRExFitter)

Если у тебя есть `combination.xml`, который ссылается на XML каналов и ROOT histograms (часто `data.root`), можно прогнать так:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/validate_root_profile_scan.py \
  --histfactory-xml /abs/path/to/combination.xml \
  --start 0.0 --stop 5.0 --points 51
```

Опция `--rootdir` нужна только если в XML относительные пути должны резолвиться не от папки с `combination.xml`.

## Apex2 runners (рекомендуется)

Для воспроизводимости и отчетов используем Apex2-скрипты:

1) Сгенерировать `cases` файл из директории с TRExFitter/HistFactory экспортами:

```bash
./.venv/bin/python tests/generate_apex2_root_cases.py \
  --search-dir /abs/path/to/trex/output \
  --out tmp/root_cases.json \
  --include-fixtures
```

2) Прогнать suite-отчет (агрегирует результаты нескольких моделей):

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_root_suite_report.py \
  --cases tmp/root_cases.json \
  --keep-going \
  --out tmp/apex2_root_suite_report.json
```

3) Если есть расхождения, объяснить их по артефактам одного run_dir (без ROOT):

```bash
./.venv/bin/python tests/explain_root_vs_nextstat_profile_diff.py \
  --run-dir tmp/root_parity_suite/<case>/run_<timestamp>
```

## 2) Что считать “совпадением”

Ожидаемые источники отличий ROOT vs pyhf/NextStat:
- разные минимизаторы/стратегии и критерии остановки
- разные дефолтные ограничения/параметризации (особенно на границах)
- нюансы включения константных членов в NLL (offsets/normalization)

Рекомендуемая метрика на первом проходе:
- `mu_hat` (best fit POI)
- `max_abs_dq_mu` по сетке mu (q(mu) разница)
- стабильность статуса минимизации (ROOT status codes)

## 3) Performance / profiling

`tests/validate_root_profile_scan.py` печатает wall‑time для:
- `hist2workspace` (построение RooWorkspace)
- ROOT profile scan
- NextStat profile scan

Для честной профилировки “движка” обычно отдельно сравнивают:
- время *построения модели* (парсинг/инициализация)
- время *одного NLL eval*
- время *одного fit* и *скана из N фиксированных fit’ов*

Следующий шаг — добавить отдельный бенч “NLL eval в ROOT vs NextStat” и “fit time”, но сначала важно зафиксировать паритет математики на q(mu).
