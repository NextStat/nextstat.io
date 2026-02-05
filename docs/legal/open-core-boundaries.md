# Open-core boundaries (NextStat) — Draft

> **DRAFT (requires counsel review).** Не является юридической консультацией.  
> Цель: зафиксировать рабочие границы OSS/Pro, чтобы планы (Phase 0+) были исполнимы и не “ломали” лицензионную модель.

---

## 1) Principle

- **OSS (AGPL)**: всё, что нужно для корректного статистического inference engine, воспроизводимости и базовых workflow (fit/scan/ranking).
- **Pro (Commercial)**: enterprise value вокруг engine — audit/compliance, orchestration, collaboration, governance, UI.

---

## 2) Proposed module split (initial)

### 2.1 OSS (AGPL)

- `ns-core` — типы, модельные интерфейсы, общие примитивы
- `ns-compute` — NLL/expected data kernels (CPU reference + performance modes)
- `ns-inference` — minimizers, fits, scans, ranking (без enterprise orchestration)
- `ns-translate` — ingestion/conversion (pyhf JSON, HistFactory XML import)
- `ns-cli` — CLI
- `ns-py` — Python bindings (PyO3)

### 2.2 Pro (Commercial)

- `ns-audit` — audit trail, e-signatures, validation packs
- `ns-compliance` — domain-specific reporting (Basel/IFRS, 21 CFR)
- `ns-scale` — distributed execution/orchestration primitives
- `ns-hub` — model registry, versioning, governance
- `ns-dashboard` — UI/monitoring

> Примечание: “GPU backends” могут быть OSS или Pro в зависимости от стратегии. Если GPU — differentiator/enterprise-only, это должно быть явно отражено в планах и в license strategy (с юристом).

---

## 3) Repository layout decision (must decide early)

Решение нужно **до первого внешнего контрибьютора/клиента**:

Option A (simplest for OSS):  
- Public repo: OSS crates (AGPL)  
- Separate private repo: Pro crates (Commercial)

Option B (single monorepo):  
- OSS crates public, Pro crates private submodule/monorepo-split tooling

**Recommendation (default):** Option A.  
Причины: меньше риска смешения лицензий и случайной публикации proprietary кода.

---

## 4) Contributions policy (must decide early)

Варианты:
- **DCO** (легче для OSS): “Signed-off-by” в коммитах.
- **CLA** (иногда предпочтительнее для open-core): отдельный документ + подписание.

**Recommendation (default):** начать с DCO для OSS, CLA — только если counsel настоятельно рекомендует.

---

## 5) Trademark/branding (baseline policy)

Черновая политика (уточнить с юристом):
- Разрешить **descriptive use**: “compatible with NextStat”.
- Запретить использование “NextStat” как названия продукта/форка без разрешения.
- Лого/товарные знаки — только по отдельной политике/лицензии.

---

## 6) Release policy (artifact boundaries)

Зафиксировать:
- какие бинарники/колёса публикуются под AGPL,
- какие сборки доступны только Pro,
- какие “telemetry / update checks” допустимы (по умолчанию opt-in).

