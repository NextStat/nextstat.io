# Документация NextStat

Этот репозиторий использует Markdown-документацию. Начните здесь и перейдите в раздел, соответствующий вашей задаче.

## Дорожная карта

- Дорожная карта, вехи и известные ограничения: `docs/ROADMAP.md`

## Начните здесь

- Указатель руководств (RU): `docs/ru/tutorials/README.md`
- Руководство по GVM-комбинациям измерений (stable-first core, advanced layers вынесены отдельно): `docs/ru/tutorials/hep-gvm-measurement-combinations.md`
- Быстрый старт (результат за 10 минут): `docs/ru/quickstarts/README.md`
- HEP GVM stable-first быстрый старт (5 минут до первого результата на committed example bundle; включает `make gvm-stable-first-example`): `docs/ru/quickstarts/hep-gvm-stable-first.md`
- Справочник Python API: `docs/references/python-api.md`
- Пакетирование Python (wheels/extras): `docs/references/python-packaging.md`
- Ввод-вывод Arrow / Parquet (таблицы гистограмм): `docs/references/arrow-parquet-io.md`
- Справочник CLI: `docs/references/cli.md`
- Справочник Rust API: `docs/references/rust-api.md`
- Терминология и стиль: `docs/references/terminology.md`
- Глоссарий (определения терминов по доменам): `docs/ru/references/glossary.md`

## Демонстрации

- Демо Physics Assistant (ROOT → сканирование аномалий → p-значения + графики): `docs/demos/physics-assistant.md`

## Бенчмарки и артефакты доверия

- Хаб бенчмарков: `docs/benchmarks.md`
- GVM benchmark snapshot (Apple M5 + AMD EPYC): `docs/benchmarks/gvm-measurement-combine-snapshot-2026-03-07.md`
- GVM NumericalPaper robustness snapshot (mixed literature + synthetic tiers): `docs/benchmarks/gvm-numerical-paper-robustness-snapshot-2026-03-07.md`
- GVM stable-surface readiness memo: `docs/benchmarks/gvm-stable-surface-readiness-2026-03-07.md`
- GVM stable-surface support policy: `docs/benchmarks/gvm-stable-surface-support-policy-2026-03-07.md`
- GVM stable-first promotion decision: `docs/benchmarks/gvm-stable-first-decision-2026-03-07.md`
- GVM stable-first support matrix: `docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07.md`
- GVM stable-first release notes: `docs/benchmarks/gvm-stable-first-release-notes-2026-03-07.md`
- GVM stable-first release candidate (`v0.10.0`): `docs/benchmarks/gvm-stable-first-release-candidate-v0.10.0-2026-03-08.md`
- Публичные наборы бенчмарков (seed-репо): `benchmarks/nextstat-public-benchmarks/`
- Валидационный отчёт (контракт JSON/PDF): `docs/references/validation-report.md`

## Инструменты и сервер (интеграция с LLM/агентами)

- Контракт Tool API: `docs/references/tool-api.md`
- Server API (`/v1/tools/execute` и т.д.): `docs/references/server-api.md`
- Артефакты графиков (JSON): `docs/references/plot-artifacts.md`

## Нейронная оценка плотности

- Руководство по нейронным PDF (FlowPdf, DcrSurrogate, обучение, ONNX): `docs/neural-density-estimation.md`
- Дифференцируемый слой HistFactory (PyTorch): `docs/differentiable-layer.md`

## Привязки R

- Справочник R-пакета (экспериментальный): `docs/references/r-bindings.md`

## Arrow / Parquet

- Схема Parquet для биннированных гистограмм (v2, с модификаторами): `docs/references/binned-parquet-schema.md`
- Схема Parquet для небиннированных событий (v1): `docs/references/unbinned-parquet-schema.md`

## Поддержка GPU

- Контракт GPU-паритета и матрица бэкендов: `docs/gpu-contract.md`

## Персоны

Навигационные страницы, связывающие концепции NextStat с рабочими процессами за пределами физики частиц.

- Специалисты по данным: `docs/ru/personas/data-scientists.md`
- Количественные аналитики: `docs/ru/personas/quants.md`
- Биологи / фармакометрики: `docs/ru/personas/biologists.md`
