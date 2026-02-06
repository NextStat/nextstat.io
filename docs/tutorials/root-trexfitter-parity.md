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
