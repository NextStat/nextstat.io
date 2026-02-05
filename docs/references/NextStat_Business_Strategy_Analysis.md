# NextStat — Business Strategy Analysis (Draft)

> Status: **DRAFT** (strategy memo, not a commitment).  
> Audience: founders/maintainers, early design decisions, planning inputs.  
> Goal: зафиксировать **почему** план устроен именно так (P0 correctness + CPU-first), и какие бизнес-риски это закрывает.

---

## 1) Executive Summary

NextStat выигрывает не “GPU-ускорением”, а **контрактом корректности + удобной архитектурой**:

- **Parity-first:** результаты совпадают с pyhf/TRExFitter в детерминированном CPU-режиме.
- **CPU-first:** 80% реальных запусков в науке — это кластеры/батч с CPU.
- **Open-core:** OSS ядро закрывает научные нужды, Pro закрывает compliance/аудит/enterprise.

Рекомендация по фокусу на первые 6–9 месяцев:
1) pyhf parity + удобный CLI/Python API,  
2) CPU parallelism + AD (градиенты/Гессиан),  
3) только потом — GPU как ускоритель.

---

## 2) Target Segments (первичный приоритет)

### 2.1 HEP / Scientific computing (Primary wedge)

**Почему:** понятный рынок ранних адоптеров, сильный pain (время fits, сложность toolchain), наличие “золотого” референса (pyhf) для валидации.

**Value props:**
- Быстрее fits (CPU параллельность + AD).
- Детальная валидация (parity suite).
- Лёгкая установка (pip + wheels) и “без ROOT” по умолчанию.

### 2.2 Finance / Model Risk (Secondary)

**Почему:** платёжеспособность + требования к reproducibility/аудиту.

**Value props (Pro):**
- Audit trail, model registry, governance.
- Reproducible runs, signed artifacts, approvals.

### 2.3 Med / 21 CFR Part 11 (Later)

**Почему:** высокая стоимость compliance и продаж; не запускать это без уже работающего ядра.

---

## 3) Monetization (Open-core)

### 3.1 OSS (AGPL)

Цель OSS: стать **best-in-class engine** и стандартом интерфейса.

В OSS остаются:
- core likelihood engine + inference (fit/scan/ranking базово),
- парсеры/трансляторы (pyhf JSON, HistFactory XML import),
- CLI/Python API,
- CPU performance path (Rayon/SIMD) и deterministic reference.

### 3.2 Pro (Commercial)

Pro продаёт:
- **Audit & compliance**: 21 CFR Part 11 audit trail, e-signatures, validation pack.
- **Scale**: distributed execution orchestration (K8s/Ray) + job management UX.
- **Hub/Dashboard**: model registry, RBAC, collaboration.
- Support/SLA + consulting.

> Важно: границы OSS/Pro должны быть описаны в `docs/legal/open-core-boundaries.md` (и подтверждены юристом).

---

## 4) Packaging & Pricing (рабочие гипотезы)

- OSS: бесплатно, но AGPL (даёт “коммерческий рычаг”).
- Pro: годовая подписка на команду/организацию + поддержка.
- Enterprise: отдельный контракт (SLA, on-prem, compliance bundle).

**Value metric (кандидаты):**
- число “projects/models” в registry,
- число “runs”/“fit-hours” в orchestration,
- количество пользователей/Seats (наименее точно).

---

## 5) Timeline Guidance (привязка к планам)

### Months 0–4 (Phase 0–1)

Цель: “First working fit” + parity на небольших моделях.

Критично:
- determinism + numeric contract (`docs/plans/standards.md`),
- валидаторы и fixtures,
- минимальный CLI/Python API.

### Months 4–9 (Phase 2)

Цель: сделать продукт **полезным на реальных моделях** (50–200 NP).

Критично:
- AD (градиенты/HVP) → стабильные uncertainties/ranking,
- CPU parallelism + batching,
- GPU — опционально (в первую очередь для dev laptops и специальных workload).

### Months 9–15 (Phase 3)

Цель: production readiness (документация, визуализации, валидация, стабильные релизы).

---

## 6) Key Risks & Mitigations

### 6.1 “Parity drift” (математика расходится с pyhf)
- Митигация: единый `standards.md` + golden tests + deterministic mode.

### 6.2 Over-engineering (GPU раньше времени)
- Митигация: GPU как Phase 2C (optional), не блокирует core.

### 6.3 Legal ambiguity (AGPL + Commercial)
- Митигация: open-core boundaries doc + counsel review + contribution policy (DCO/CLA) до внешних контрибов.

### 6.4 Adoption friction (сложная установка)
- Митигация: wheels, минимальные зависимости, быстрый “hello fit”, готовые примеры.

---

## 7) Decisions Needed (чтобы планы исполнялись)

1) Contribution policy: DCO vs CLA (и почему).
2) Repo layout: single repo + private pro modules vs split repos.
3) Release policy: какие артефакты публикуются под AGPL и в каком виде.
4) Minimum supported platforms: Linux x86_64 + macOS arm64 (на старте).

