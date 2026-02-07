# TRExFitter Full Replacement — Apex2 Plan (v2)

Created: 2026-02-07  
Updated: 2026-02-07  
Status: active

Цель: **полная замена TRExFitter** с:
- **идентичными числами** (fit/profile/expected/yields) на фиксированных surface-контрактах,
- удобным конфигурированием (YAML + JSON Schema + IDE),
- экспортом **publication-ready** **PDF/SVG** (вектор), при этом **числа — источник истины** (JSON artifacts).

Методология: **Apex2: Planning → Exploration → Execution → Verification**.

## 0) Что уже есть (по коду)

- ROOT-native чтение (TH1 + TTree) и детерминированное заполнение гистограмм: `crates/ns-root`.
- HistFactory XML + ROOT hists → pyhf workspace: `nextstat import histfactory`.
- TREx config importer:
  - `ReadFrom: NTUP` + region/sample overrides (selection/weight composition): `crates/ns-translate/src/trex/mod.rs`.
  - `ReadFrom: HIST` (HistFactory export dir) + TREx-like фильтры (region/channel selection + sample masking).
- Reports/render:
  - numbers-first artifacts + рендер в PDF/SVG (Matplotlib renderer).
- Determinism: `--threads 1` + автоматическое отключение Accelerate в deterministic mode.

## 1) Каноничные данные для тестов

### 1.1 pyhf validation fixtures (в репозитории)
- `tests/fixtures/pyhf_xmlimport/` (OverallSys + StatError + NormFactor).
- `tests/fixtures/pyhf_coupled_histosys/` (shared/coupled HistoSys NP).
- `tests/fixtures/pyhf_multichannel/` (signal+control, ShapeSys).

### 1.2 HEPData bundles (воспроизводимые из интернета)
- `tests/hepdata/manifest.json` + `tests/hepdata/fetch_workspaces.py` → `tests/hepdata/workspaces/...`.

## 2) Что реально осталось до “TREx replacement” (BMCP)

### 2.1 Внешние baselines на реальных TREx export dirs (блокер)
- Epic: `TREx Replacement: Parity Contracts + Baselines` (`f4ead082-aa4a-49f0-8468-6201df649039`)
  - Task (DEFERRED): `23711a70-dfb5-48e3-a96c-27abaa1f8fdc`
    - Нужны 1–3 реальных TREx export dir (каждый: `combination.xml` + ROOT hists).
    - Пользователь собирает cases JSON; мы прогоняем ROOT-suite в окружении с ROOT/TREx и пишем baseline через `tests/record_baseline.py` (см. `README.md`).

### 2.2 Expr-compat для NTUP (реальные TREx configs)
- Epic: `TREx Replacement: Expression Compatibility (TTreeFormula subset + vector branches)` (`53b3d3fc-ef1f-42b2-b067-b4df90c1044e`)
  - ROOT/TMath spellings (`TMath::Abs`, `fabs`, `Power`, …) — закрываем без ручных rewrite.
  - По корпусу выражений добавляем минимальные missing math funcs.
  - При необходимости: vector branches + `jet_pt[0]` (материализация scalar virtual columns).
  - Coverage report в импортере.

## 3) Apex2 execution loop (TDD)

### Planning
- Зафиксировать surfaces и tolerances (q(μ), μ̂, NLL, yields).
- Определить baseline artifacts (numbers-first) и правила determinism.
- Правило безопасной работы: не использовать destructive git действия; артефакты — только в новых `tmp/...` dirs.

### Exploration
- Собрать 1–3 реалистичных TREx export dirs и прогнать ROOT-suite.
- Собрать корпус `Selection`/`Weight`/`Variable` из реальных `.config` и составить support matrix.

### Execution
- Вносить изменения маленькими инкрементами:
  1) failing test
  2) минимальный фикс
  3) локальная проверка (unit/integration)
  4) обновление baseline только когда контракт стабилен

### Verification
- ROOT-suite parity на реалистичных export dirs проходит по фиксированным surface-контрактам.
- 90%+ выражений из корпуса компилируются/исполняются без ручного rewrite; для остального есть coverage report со span + подсказкой.

