# TRExFitter Full Replacement — Apex2 Plan (v2)

Created: 2026-02-07  
Updated: 2026-02-08  
Status: active

Цель: **полная замена TRExFitter** с:
- **идентичными числами** (fit/profile/expected/yields) на фиксированных surface-контрактах,
- удобным конфигурированием (YAML + JSON Schema + IDE),
- экспортом **publication-ready** **PDF/SVG** (вектор), при этом **числа — источник истины** (JSON artifacts).

Методология: **Apex2: Planning → Exploration → Execution → Verification**.

## 0) Что уже есть (по коду)

- ROOT-native чтение (TH1 + TTree) и детерминированное заполнение гистограмм: `crates/ns-root`.
- NTUP expressions: ROOT/TMath алиасы + static/dynamic indexing (`jet_pt[0]`, `jet_pt[idx]`) end-to-end.
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

### 2.1 ROOT-suite parity closure на реалистичных export dirs (блокер)
- Epic: `ROOT-suite Parity Closure: Profile Scan q(mu) matches ROOT` (`c2f00478-1927-4a5a-94b0-6fc66d4e21f8`)
  - Текущее состояние (2026-02-08): профайл-скан на реалистичном fixture `tests/fixtures/tttt-prod_workspace.json` проходит gate:
    - grid: μ in [0, 3], 11 points
    - `max_abs_dq_mu ≈ 1.4e-2` (atol=2e-2)
    - `mu_hat_delta ≈ -2.7e-3` (atol=5e-2)
  - Диагностика: `tests/validate_root_profile_scan.py --dump-root-params` теперь сравнивает fitted params ROOT vs NextStat (в NextStat порядке)
    и пишет `ns_vs_root_params_*` в `summary.json`.

### 2.2 Внешние baselines на реальных TREx export dirs (DEFERRED, вернемся позже)
- Epic: `External TREx Export Dirs: Collect + Record Baselines` (`fe04caac-ab97-4567-b0cc-77d11681072c`)
  - Task (DEFERRED): `24475739-0c6e-4572-a342-a9dc369463f6`
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
