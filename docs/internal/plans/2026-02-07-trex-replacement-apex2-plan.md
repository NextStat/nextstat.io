---
title: "TRExFitter Full Replacement — Apex2 Plan (v2)"
status: active
created: 2026-02-07
updated: 2026-02-07
---

# TRExFitter Full Replacement — Apex2 Plan (v2)

Цель: **полная замена TRExFitter** с:
- **идентичными числами** на ключевых поверхностях (fit/profile/expected/yields),
- удобным конфигурированием (YAML + schema + IDE),
- экспортом **publication-ready** **PDF/SVG** (вектор), при этом **числа — источник истины** (JSON artifacts).

Методология: **Apex2: Planning → Exploration → Execution → Verification**.

## 0) Что уже есть в NextStat (факты по коду)

### ROOT / HistFactory / TREx replacement pipeline
- ROOT-native чтение (TH1 + TTree) и заполнение гистограмм: `crates/ns-root` + `docs/references/rust-api.md`.
- Импорт HistFactory XML + ROOT hists → pyhf workspace: `nextstat import histfactory` (CLI) / `ns_translate::histfactory::*`.
- Экспорт pyhf workspace → HistFactory XML + ROOT hists: `nextstat export histfactory`.
- Однокомандный оркестратор: `nextstat run --config <analysis.yml>` (analysis spec v0).
- TREx config importer (subset, но уже практичный):
  - `ReadFrom: NTUP` (ntuple pipeline) + region/sample overrides (selection/weight composition) — `crates/ns-translate/src/trex/mod.rs`.
  - `ReadFrom: HIST` (HistFactory export dir) — импорт `combination.xml` + **TREx-like фильтры** (region/channel selection + sample masking), включая basedir семантику для ROOT файлов.

### Выражения/selection/weights
- Rust expression engine (bulk eval + ternary + indexing + spans): `crates/ns-root/src/expr.rs` (`ns_root::CompiledExpr`).
- Python “vectorized expression evaluator” для контрактов/баслайнов: `bindings/ns-py/python/nextstat/analysis/expr_eval.py` + `tests/python/test_expr_eval.py`.
- Region/sample overrides (selection/weight/variable) + правила композиции — реализовано (BMCP task `cd263fa4-aa37-4349-ab28-d16818f03c78`, код в `ns-translate`/pipeline).

### Determinism / parity mode
- `--threads 1` делает детерминированный ран (в т.ч. авто-отключение Accelerate): `crates/ns-cli/src/main.rs`, `crates/ns-compute/src/lib.rs`, `docs/references/cli.md`.

### Baselines / contracts
- TREx replacement parity contract: `docs/references/trex_replacement_parity_contract.md`.
- Baseline schema + compare helpers: `docs/schemas/trex/*`, `tests/_trex_baseline_compare.py`, `tests/compare_trex_analysis_spec_with_latest_baseline.py`.
- ROOT-suite harness (hist2workspace + RooFit scan + NextStat scan): `tests/validate_root_profile_scan.py`, `tests/apex2_root_suite_report.py`, `docs/tutorials/root-trexfitter-parity.md`.

## 1) Каноничные данные для тестов (то, чему “можно доверять”)

### 1.1 pyhf validation (в репозитории)
Три “каноничных” HistFactory-фикстуры (XML + ROOT):
- `tests/fixtures/pyhf_xmlimport/` (OverallSys + StatError + NormFactor) — *simple*.
- `tests/fixtures/pyhf_coupled_histosys/` (shared/coupled HistoSys NP) — *medium+*.
- `tests/fixtures/pyhf_multichannel/` (signal+control, ShapeSys) — *medium*.

Их удобно использовать как:
- тест “сборки workspace из HistFactory XML”,
- тест “selection/weight” (если добавим ntuple слой),
- тест ROOT-suite parity (NextStat vs RooFit).

### 1.2 HEPData likelihood bundles (интернет-источник, воспроизводимый)
HEPData DOI bundles → pyhf JSON workspaces (патчи) для “реалистичных” моделей. Механика:
`tests/hepdata/manifest.json` + `tests/hepdata/fetch_workspaces.py` → `tests/hepdata/workspaces/...`.

## 2) Что осталось для “полного паритета TRExFitter” (BMCP + код, актуально)

### 2.1 BMCP (живые хвосты)
- Epic `TREx Replacement: Parity Contracts + Baselines` (`f4ead082-aa4a-49f0-8468-6201df649039`)
  - **Осталось:** записать ROOT-suite baselines из 1–3 реалистичных TREx export dirs (task `23711a70-dfb5-48e3-a96c-27abaa1f8fdc`) — **deferred/backlog** (нужны export dirs).
- Epic `TREx Replacement: Analysis Spec (YAML) + CLI Orchestrator` (`e7e59d19-b238-4b22-a734-24e491c70634`) — **DONE**.
- Epic `TREx Replacement: Expression Compatibility (TTreeFormula subset + vector branches)` (`53b3d3fc-ef1f-42b2-b067-b4df90c1044e`)
  - функции/алиасы ROOT/TMath (`TMath::Abs`, `fabs`, `Power`, …),
  - дополнительные math-функции по корпусу выражений,
  - (если реально нужно) vector branches + `jet_pt[0]`,
  - coverage report для импортера.

### 2.2 Технический хвост (не только BMCP)
Критично (для “replacement” как продукта, а не демо):
- **Внешняя валидация на “реальных” TREx export dirs**: это единственный способ гарантировать, что HIST-mode импорт + masks совпадают с ожиданиями TREx на практике.
- **Expression-compat для NTUP**: большинство реальных TREx `.config` используют ROOT-диалект (TMath::*, fabs, и т.д.) и часто завязаны на vector branches.

## 3) Apex2 — План работ (TDD)

### 3.1 Planning (что фиксируем заранее)
1) **Определяем canonical triad cases**:
   - `pyhf_xmlimport` (simple),
   - `pyhf_coupled_histosys` (coupled NP),
   - `pyhf_multichannel` (ShapeSys).
2) **Определяем surfaces для паритета** (минимум):
   - expected_data (main+aux),
   - bestfit + twice_nll,
   - profile-scan q(mu) на фиксированной сетке mu.
3) **Определяем артефакты**:
   - `trex_baseline_v0` (numbers-first) для analysis-spec gate,
   - apex2 ROOT-suite report (numbers-first).
4) **Определяем правила “без разрушений”**:
   - никаких `git restore/reset/clean`,
   - никакого удаления артефактов; только новые timestamped `tmp/…`.

### 3.2 Exploration (собираем факты до кода)
1) Собираем 1–3 “реалистичных” TREx export dirs (см. BMCP task `23711a70…`).
2) Прогоняем ROOT-suite на них (ROOT env) и фиксируем “source of truth” surfaces (q(mu), mu_hat, scans, yields).
3) Собираем корпус выражений из реальных TREx configs (Selection/Weight/Variable) и составляем support matrix (Epic `53b3d3fc…` → task `0c0af226…`).

### 3.3 Execution (исправляем по одному контракту)
TDD-подход:
1) Expr-compat инкременты (по одному, с unit tests):
   - алиасы ROOT/TMath (токенайзер/парсер),
   - затем минимальный набор новых функций (log10/sin/cos/atan2) по корпусу,
   - затем (если реально нужно) vector-branch indexing `name[idx]` с интеграционными тестами на маленьком ROOT fixture.
2) HIST-mode verification:
   - сравнить channel/sample masking на реалистичных export dirs с тем, что делает TREx.

### 3.4 Verification (что считаем “готово”)
1) Реалистичные TREx export dirs:
   - ROOT-suite report проходит (NextStat vs ROOT) на фиксированной сетке mu,
   - baseline recorder (external env) пишет артефакты без ручных правок.
2) Expression corpus:
   - 90%+ выражений компилируются и исполняются без ручного rewrite,
   - для оставшихся есть coverage report с точным span + предложением rewrite.
