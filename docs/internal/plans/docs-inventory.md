# Инвентаризация документации (public vs internal)

Цель: зафиксировать **что считаем публичной документацией**, что является **внутренними планами/мемо**, и что именно переводим/стандартизируем на английский.

Правило:
- `docs/plans/**` это **внутренние планы**: язык **русский**, директория **в gitignore**, в публичный репозиторий не попадает.
- публичные доки: язык **английский** (EN). Если сейчас есть RU фрагменты, они попадают в backlog на перевод.

## Срез (по состоянию на 2026-02-06)

### Публичные документы (должны быть EN)

| Путь | Тип | Язык сейчас | Действие | Примечание |
|---|---|---|---|---|
| `README.md` | основной README | EN | review | сверить актуальность с фазами P5-P8 (GLM, time series) |
| `CHANGELOG.md` | changelog | EN | keep | ок |
| `CONTRIBUTING.md` | contribution guide | EN | keep | ок |
| `docs/benchmarks.md` | benchmark guide | EN | updated | Criterion + Apex2 Python benches + baseline recorder |
| `docs/WHITEPAPER.md` | white paper (draft) | EN | continue | расширять: use-cases beyond HEP, validation, perf baselines |
| `docs/papers/model-specification.md` | math spec | EN | continue | важный публичный референс для `ns-translate` |
| `docs/papers/nextstat-software.md` | software paper | EN | continue | позиционирование + архитектура |
| `docs/papers/reproducibility.md` | validation paper | EN | continue | описывает parity/валидацию; сверить с Apex2 suite |
| `docs/tutorials/phase-3.1-frequentist.md` | tutorial | EN | keep | ок |
| `docs/tutorials/phase-7-hierarchical.md` | tutorial | EN | keep | random effects + LKJ |
| `docs/tutorials/phase-7-ppc.md` | tutorial | EN | keep | posterior predictive checks |
| `docs/tutorials/phase-8-timeseries.md` | tutorial | EN | keep | Kalman/EM/forecast |
| `docs/tutorials/root-trexfitter-parity.md` | tutorial | EN (обновлён) | updated | Apex2 methodology + baseline recorder + cluster templates |
| `docs/tutorials/README.md` | tutorial index | EN | keep | индекс всех tutorial-страниц |
| `docs/tutorials/phase-13-pk.md` | tutorial | EN | keep | 1-compartment oral PK baseline |
| `docs/tutorials/phase-13-nlme.md` | tutorial | EN | keep | NLME baseline; python bindings exposed |
| `docs/legal/open-core-boundaries.md` | legal draft | EN | internal review | по смыслу скорее внутренний/юридический, но уже EN |
| `docs/references/python-api.md` | API reference | EN | keep | публичный Python surface (`nextstat`) |
| `docs/references/rust-api.md` | API reference | EN | keep | Rust crates: `ns-core`, `ns-inference`, ... |
| `docs/references/cli.md` | CLI reference | EN | keep | `nextstat` CLI commands + contracts |

### Внутренние документы (не переводим, остаются RU, gitignored)

| Путь | Тип | Язык | Примечание |
|---|---|---|---|
| `docs/plans/*` | планы по фазам + стандарты | RU | все файлы под `docs/plans/` должны оставаться gitignored |

### Внутренние мемо/референсы (не публичные)

| Путь | Тип | Язык сейчас | Действие | Примечание |
|---|---|---|---|---|
| `docs/references/NextStat_Business_Strategy_Analysis.md` | strategy memo | EN | keep internal | не публичный документ |
| `docs/references/TRExFitter_vs_NextStat_Analysis.md` | technical memo | EN | keep internal | не публичный документ |

## Backlog (следующие шаги по docs epic)

1) Проверить что во всех публичных docs нет RU фрагментов (поиск по `[А-Яа-я]` вне `docs/plans/**`).
2) White paper v1:
   - добавить секции про general statistics (GLM, builder) и Phase 8 time-series (Kalman/AR(1)/EM/forecast)
   - добавить секцию про Phase 7 hierarchical models (random effects, LKJ, PPC)
   - добавить ссылки на Apex2 отчеты/скрипты как reproducibility pipeline
   - упомянуть `tests/record_baseline.py` как infrastructure для воспроизводимых baseline сравнений
