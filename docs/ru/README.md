# Документация NextStat

Этот репозиторий использует Markdown-документацию. Начните здесь и перейдите в раздел, соответствующий вашей задаче.

## Дорожная карта

- Дорожная карта, вехи и известные ограничения: `docs/ROADMAP.md`

## Начните здесь

- Указатель руководств (сквозные рабочие процессы): `docs/tutorials/README.md`
- Быстрый старт (результат за 10 минут): `docs/ru/quickstarts/README.md`
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
