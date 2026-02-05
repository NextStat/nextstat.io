# TRExFitter vs NextStat — Technical Analysis (Draft)

> Status: **DRAFT** (high-level lessons learned; validate with real users).  
> Goal: понять, какие “полевые” требования и UX ожидания нужно закрыть, чтобы реально заменить TRExFitter/pyhf workflows.

---

## 1) What TRExFitter is (в контексте HEP workflows)

TRExFitter исторически используется как “workhorse” для binned likelihood fits:
- управление большим количеством каналов/сэмплов/систематик,
- генерация/агрегация статистических моделей,
- профили, ranking, pulls, correlation matrices,
- отчётность/плейтинг под ATLAS/CMS стиль.

Важное наблюдение: ценность TRExFitter — не только “fit”, а **весь пайплайн вокруг**.

---

## 2) Where NextStat should win (позиционирование)

### 2.1 Engine correctness + portability

- “Source of truth” математики фиксируется и валидируется против pyhf (`twice_nll`).
- Rust core + Python API → переносимость, меньшая зависимость от ROOT.

### 2.2 Performance where it matters

TRExFitter workflows часто упираются в:
- большое число nuisance parameters,
- ranking (много повторных fits),
- scans/limits.

NextStat должен выиграть через:
- AD (градиенты/HVP),
- CPU parallelism + batching,
- reproducible job-array execution на кластере.

---

## 3) Must-have feature parity (заменяемость)

Чтобы “заменить” TRExFitter на практике, минимум нужно:

1) **Workspace ingestion**
   - pyhf JSON (P0)
   - HistFactory XML import (P1) или конвертер через pyhf

2) **Fit & diagnostics**
   - MLE fit + uncertainties
   - pulls + constraints summary
   - correlation matrix

3) **Workflow primitives**
   - Asimov dataset
   - profile likelihood scan
   - ranking/impact (NP impact on POI)

4) **Systematics preprocessing (часто недооценено)**
   - smoothing (shape systematics)
   - pruning (малозначимые NP)
   - symmetrisation

> Эти пункты отражены в roadmap (Phase 2D/2E в master plan). Для заменяемости важно держать их в видимой части плана, даже если реализация позже.

---

## 4) Lessons learned (что копировать, что избегать)

### 4.1 Copy (сохранить UX ожидания)
- Конфиг “одной кнопкой” для типичных анализов (presets).
- Отчёты: pulls/ranking/corr матрицы в стандартном стиле.
- Ясные лог-сообщения “что было зафиксировано/профилировано”.

### 4.2 Avoid (снизить сложность и фрагментацию)
- “ROOT-first” как обязательная зависимость ядра.
- Слишком тесная связка preprocessing ↔ inference ↔ plotting (монолит).
- Невоспроизводимые результаты из-за параллельных редукций без deterministic режима.

---

## 5) Compatibility strategy (практичная миграция)

Реалистичный путь миграции:

1) **pyhf parity** на ограниченном наборе моделей (fixtures).
2) **“TREx-like outputs”**: pulls/ranking/corr матрицы в знакомом формате.
3) **Import pipeline**: HistFactory XML → NextStat model (или через pyhf).
4) **Preprocessing**: дать пользователю те же рычаги (smoothing/pruning), но в виде модульного пайплайна.

---

## 6) What to validate with users (список интервью)

1) Какие отчёты/графики они реально смотрят каждый день?
2) Какие preprocessing шаги обязательны (и в каком порядке)?
3) Какая максимальная модель (bins/channels/NP) считается “нормой”?
4) Где болит время: ranking, scans, limits, toys?
5) Какие требования к reproducibility/cluster execution?

