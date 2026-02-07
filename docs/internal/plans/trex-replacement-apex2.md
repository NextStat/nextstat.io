---
title: "TRExFitter Replacement Plan (Apex2)"
status: active
created: 2026-02-06
---

# TRExFitter Replacement Plan (Apex2)

Цель: **полная** замена TRExFitter (включая удобный конфиг/оркестрацию), допускаем свой визуальный стиль, но с **идентичными числами** (в рамках явного parity contract) и **publication‑ready** векторным экспортом **PDF/SVG**.

Methodology: **Apex2** — Planning → Exploration → Execution → Verification.

## 0) Non-negotiables

1) **Numbers-first artifacts** (canonical JSON)
   - Engine produces numeric artifacts only (bin edges, per-sample yields, bands, pulls, corr, etc).
   - Renderers (PDF/SVG) consume artifacts and **never recompute** numbers.
2) **Determinism mode** (parity reference path)
   - Default parity path: `threads=1`, stable ordering, stable reductions.
   - All parity tests run only in this mode.
3) **Golden numeric tests (no pixel flakes)**
   - We test JSON artifacts, not images.
   - Baseline refresh is manual/nightly only.

## 0.1) Что уже умеем (важно для плана)

Сейчас в репозитории уже есть большая часть "фундамента" для M0/M1/M2:

- **ROOT чтение (ns-root)** — полный нативный стек, zero ROOT C++ dependency:
  - TH1D/TH1F: bin edges + contents + optional `sumw2` (`crates/ns-root/src/file.rs`, `crates/ns-root/src/histogram.rs`).
  - **TTree/TBranch** binary reader: mmap I/O, ROOT class reference system, TObjArray dispatch, TLeaf type detection. 7 leaf types (f32/f64/i32/i64/i16/i8/bool). (`crates/ns-root/src/objects/ttree.rs`, `crates/ns-root/src/tree.rs`).
  - **Basket decompression** (zlib/LZ4) with **rayon-parallel** columnar extraction (`crates/ns-root/src/basket.rs`, `crates/ns-root/src/branch_reader.rs`).
- **Expression engine** для селекций/весов/переменных — рекурсивный спуск, арифметика, сравнения, `&&`/`||`, функции (`abs`, `sqrt`, `log`, `exp`, `pow`, `min`, `max`) (`crates/ns-root/src/expr.rs`).
- **Expression engine** для селекций/весов/переменных — компиляция AST в **bytecode (stack VM)** и быстрый **bulk eval** по колонкам; арифметика, сравнения, `&&`/`||`, функции (`abs`, `sqrt`, `log`, `exp`, `pow`, `min`, `max`) (`crates/ns-root/src/expr.rs`).
  - **Histogram filler**: один проход по колонкам, селекция/веса, variable bins, `sumw2` (`crates/ns-root/src/filler.rs`).
  - **Бенчмарк**: ~8.5x быстрее uproot+numpy на full pipeline (file open → TTree parse → read branches → selection → histogram fill). TTree metadata parse — 28x быстрее.
- **Ntuple → Workspace (ns-translate)** — полный аналог TRExFitter NTUP mode:
  - `NtupleWorkspaceBuilder`: fluent Rust API, ROOT ntuples → HistFactory `Workspace` (`crates/ns-translate/src/ntuple/`).
  - Все типы модификаторов: NormFactor, NormSys, WeightSys (up/down weight expressions), TreeSys (up/down ROOT файлы), StatError (auto `sqrt(sumw2)`).
  - Кэширование ROOT файлов, Asimov data, тот же `Workspace` struct что и из pyhf JSON.
- **HistFactory XML → pyhf Workspace (ns-translate)**:
  - Парсер `combination.xml` + чтение ROOT гистограмм и сборка pyhf‑совместимого `Workspace` (`crates/ns-translate/src/histfactory/builder.rs`).
  - Фикстуры под HistFactory уже есть (`tests/fixtures/histfactory/*`).
- **Apex2 ROOT/TRExFitter parity harness уже существует**:
  - Инструменты для генерации case‑паков, запусков на кластере/агрегации/базлайнов: см. `docs/tutorials/root-trexfitter-parity.md` и `tests/*.py` (например, `tests/generate_apex2_root_cases.py`, `tests/record_baseline.py`).

Следствие: **все три входных пути реализованы** (pyhf JSON, HistFactory XML, ROOT ntuples). План теперь фокусируется на: (a) продуктовой CLI‑оркестрации, (b) численных артефактах отчётов, (c) удобном конфиге, (d) parity/базлайнах.

### 0.2) Статус среза (2026-02-07)

Уже сделано (numbers-first; рендер опционален):
- `nextstat import histfactory --xml ...` → pyhf `workspace.json` (fixtures‑driven).
- `nextstat viz distributions` → `trex_report_distributions_v0` (per-sample prefit/postfit + ratio, edges из ROOT).
- `nextstat viz pulls` → `trex_report_pulls_v0`.
- `nextstat viz corr` → `trex_report_corr_v0`.
- `nextstat report`:
  - по умолчанию **делает MLE fit** (если `--fit` не задан) и пишет `fit.json` в `--out-dir`
  - пишет `distributions.json`, `yields.json` (+ `yields.csv`, `yields.tex`), `pulls.json`, `corr.json` (если есть covariance), `uncertainty.json` (если не `--skip-uncertainty`)
- **Golden numeric tests** для report artifacts (без pixel diffs):
  - тест: `crates/ns-cli/tests/cli_report_golden.rs`
  - goldens: `tests/fixtures/trex_report_goldens/histfactory_v0/*.json`
- JSON schemas v0 под report artifacts: `docs/schemas/trex/report_*_v0.schema.json` (+ smoke tests).

Следующий “replaceability jump”: **TREx baseline parity pack** (реальные кейсы) + **analysis spec** (удобная конфигурация) + “one command” оркестратор.

## 1) Milestones (replaceability ladder)

### M0 — TREx export dir → Fit + Report (no TRExFitter required at runtime)

Input: HistFactory export directory (e.g., TREx output) containing `combination.xml` + ROOT hist files.  
Output: run bundle + numeric artifacts + PDF/SVG report pack.

**Definition of Done (M0):**
- `nextstat import histfactory ...` works on fixtures and produces deterministic output. ✅
- `nextstat report ...` produces:
  - numeric artifacts (schema-validated, stable ordering)
  - PDF (vector) + per-plot SVG (vector)
- Golden numeric tests pass on the fixture pack.
- Parity vs pyhf in deterministic mode for fit + expected data surfaces.

### M1 — HIST mode (binned ROOT histograms)

Input: binned histograms (ROOT) + analysis spec.  
Output: same as M0.

### M2 — NTUP mode (ROOT ntuples)

Input: ntuples + analysis spec (or imported TREx `.config`).  
Output: hist pack → workspace → fit → artifacts → PDF/SVG.

## 2) Workstreams mapped to BMCP

BMCP project: `nextstat`.

### 2.1 Parity contract + baselines

Epic: `f4ead082-aa4a-49f0-8468-6201df649039`  
Why first: without a contract + comparator, “identical numbers” is not enforceable.

### 2.2 HistFactory ingest

Epic: `d23fc61d-51a3-4e65-ae1e-3c400419ec4d`  
Why early: fastest path to M0 replaceability (TREx export dirs are the common interchange format).

### 2.3 Report artifacts + PDF/SVG renderer

Epic: `218fdf7c-21b7-4807-b59e-7a9a8e81348a`  
Why early: “publication-ready output” is the tangible replacement surface.

### 2.4 Analysis spec + orchestrator

Epic: `e7e59d19-b238-4b22-a734-24e491c70634`  
Why: replaces TREx configs with IDE-friendly YAML + schema, and provides “one command”.

### 2.5 NTUP/HIST pipeline + preprocessing

Epics:
- `d071c179-32ee-4947-9035-66a8c47581ce` (NTUP/HIST pipeline)
- `f6a460e9-b61b-42fe-bfd8-10d3e0270f9d` (preprocessing: smoothing/symmetrize/prune)

## 2.6) TDD “матрица” (что тестируем первым)

Мы делаем TDD не “для галочки”, а чтобы гарантировать: **идентичные числа** и **стабильные артефакты**.

1) **Schema/ordering tests (RED→GREEN)**
   - JSON schemas для артефактов отчёта (v0) + smoke‑тесты на обязательные поля/типы.
   - “Stable ordering” тесты: каналы/семплы/параметры сортируются детерминированно.
2) **Parity tests (numbers‑only)**
   - Fit parity: `twice_nll`, bestfit, uncertainties (в deterministic mode).
   - Expected data parity: main+aux ordering/values.
   - Baseline comparator: имя‑ориентированное сравнение с понятным diff‑репортом.
3) **Report artifact golden tests (no images)**
   - На fixture‑паке фиксируем golden JSON артефакты (prefit/postfit stack+ratio, pulls, corr).
   - Pixel/image сравнения не делаем.
4) **Renderer conformance tests**
   - Рендерер обязан быть pure consumer: сравниваем input artifact hash ↔ embedded metadata (в run bundle).
   - Проверка, что выходы — векторные (PDF/SVG) и без silent rasterization (минимальная эвристика/метаданные).

## 3) Apex2 execution template (per task)

Every task must explicitly produce artifacts in each phase:

### Planning
- Scope: what is in/out.
- Inputs/outputs: file formats + stable ordering rules.
- Contract: what numbers are compared and with what tolerances.
- Test plan: exact tests to write first (RED).

### Exploration
- Inspect existing code paths and fixtures.
- Identify ordering/determinism risks.
- Decide minimal viable design, with explicit tradeoffs.

### Execution
- Implement smallest slice that makes tests pass (GREEN).
- Keep logic close to the contract; avoid “helpful” hidden behavior.

### Verification
- Run the new tests locally.
- Add regression guard (golden numeric or stable schema checks).
- Document CLI usage + failure modes.

## 4) Next step (we start here)

We now move to the “real parity” slice for M0:

1) **Add TREx parity fixture pack** (minimal → realistic) and record baselines (external env).
2) Wire `tests/python/_trex_baseline_compare.py` into a strict-but-actionable CI gate (numbers-only, deterministic ordering, clear diffs).
3) Start `analysis.yaml` (NextStat-native config) that covers the M0 “one command” path:
   HistFactory export dir → workspace → fit → report artifacts → render.

См. также:
- ROOT/TRExFitter parity harness (Apex2): `docs/tutorials/root-trexfitter-parity.md`
- Высокоуровневые заметки: `docs/references/TRExFitter_vs_NextStat_Analysis.md`
- Численные стандарты/толерансы: `docs/plans/standards.md`, `tests/python/_tolerances.py`
