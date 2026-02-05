<!--
Audit Report
Generated: 2026-02-05 16:55:45
Git Commit: unknown
Scope: docs/plans + toolchain/deps/CI
Invoked: /audit repo-plans-versions
-->

# Project Audit Report

Date: 2026-02-05  
Project: NextStat (nextstat.io)  
Auditor: Codex  
Scope: `docs/plans` + repo toolchain/deps/CI  
Git Commit: unknown

---

## Executive Summary

- Critical: 0
- Major: 3
- Minor: 4
- Total: 7

Overall Health Score: 88/100  
Top risks: отсутствие security automation (CodeQL/secret scanning) + отсутствие release pipeline → сложно безопасно и предсказуемо публиковать артефакты.

---

## Critical Issues

None found in the audited scope after syncing toolchain/deps and making CI lint/test passes locally.

---

## Major Issues

### [Security] Нет CodeQL / secret scanning workflows в репозитории
- File: `.github/workflows/`
- Evidence: присутствуют только `rust-tests.yml` и `python-tests.yml`; отсутствуют `codeql.yml`/secret scan workflow.
- Impact: нет автоматического SAST/secret scanning в PR/`main`; выше риск пропустить уязвимость или случайно закоммитить секрет.
- Fix:
  - Добавить `codeql.yml` (инициализация + анализ для Rust/Python).
  - Добавить secret scanning job (например gitleaks action) или включить GitHub Advanced Security/secret scanning на уровне репо.
- Confidence: high

### [Release] Нет release pipeline для wheels / binaries / crates
- File: `.github/workflows/`
- Evidence: отсутствует `release.yml`/tag-based pipeline (сборка wheels для Linux/macOS/Windows, публикация GitHub Release).
- Impact: публикации будут ручными и нерепродуцируемыми; выше риск “сломать” дистрибутивы, особенно для Python bindings.
- Fix:
  - Добавить `release.yml` (workflow_dispatch + tag push), сборка wheels через maturin и публикация в Release artifacts.
  - Зафиксировать policy: что именно публикуем как OSS vs Pro (см. `docs/legal/open-core-boundaries.md`).
- Confidence: high

### [Compliance] Нет `THIRD_PARTY_LICENSES` и генерации third‑party license report
- File: `NOTICE`
- Evidence: `NOTICE` предполагает наличие списка third-party лицензий, но в репо нет `THIRD_PARTY_LICENSES` и генератора.
- Impact: юридическая/OSS‑гигиена слабее; может блокировать enterprise adoption.
- Fix:
  - Добавить генерацию через `cargo-about`/`cargo-deny` (Rust) и аналог для Python (если появятся runtime deps).
  - Зафиксировать как часть релизного чеклиста.
- Confidence: medium

---

## Minor Issues

### [Governance] RFC template placeholder
- File: `GOVERNANCE.md:143`
- Evidence: `RFC-XXX: Title`.
- Impact: косметика/неясность для внешних контрибьюторов.
- Fix: заменить на реальный пример RFC или вынести в отдельный шаблон в `docs/`/`.github/`.
- Confidence: high

### [Docs] Нужен единый “source of truth” для CI/versions
- File: `docs/plans/versions.md`
- Evidence: baseline уже есть, но важно удерживать синхронизацию планов и реальных файлов (`Cargo.toml`, workflows).
- Impact: риск дрейфа (планы устаревают быстрее кода).
- Fix: удерживать `docs/plans/versions.md` как навигацию + использовать `scripts/versions_audit.py` в ревью.
- Confidence: high

### [Python] Mypy/ruff config пока минимальный
- File: `bindings/ns-py/pyproject.toml`
- Evidence: отсутствуют настройки `tool.ruff`/`tool.mypy`.
- Impact: по мере роста Python surface area качество будет зависеть от дефолтов.
- Fix: добавить минимальный config (line-length, target-version, basic lint set) когда появятся python модули beyond stubs.
- Confidence: medium

### [Tooling] rust-toolchain pinned → требуются инструкции для локальной установки
- File: `rust-toolchain.toml`
- Evidence: toolchain pinned на `1.93.0`.
- Impact: новые контрибьюторы без rustup/без нужного toolchain получат ошибки.
- Fix: в `README.md`/`CONTRIBUTING.md` добавить короткий блок “Install Rust via rustup; toolchain auto-pins”.
- Confidence: medium

---

## Feature Checklist

### Plans + Versioning
- [x] Version baseline exists (`docs/plans/versions.md`)
- [x] Real repo toolchain pinned (`rust-toolchain.toml`)
- [x] CI pins modernized (`actions/*@v6`, `codecov@v5`, rust-cache)
- [x] Clippy passes with `-D warnings` locally
- [ ] Security automation (CodeQL/secret scanning)
- [ ] Release pipeline (wheels/binaries)
- [ ] Third-party licenses report

---

## Unverified Areas

- Реальная публикация wheels (Linux/Windows) не проверялась локально.
- Не проверялась корректность и полнота `LICENSE-COMMERCIAL` с юристом (draft quality).

---

## Files Reviewed

- `docs/plans/*`
- `Cargo.toml`, `rust-toolchain.toml`, `rustfmt.toml`
- `.github/workflows/rust-tests.yml`, `.github/workflows/python-tests.yml`
- `bindings/ns-py/*`
- `crates/ns-translate/src/pyhf/*` (для CI/clippy pass)

---

## Recommendations

1. Priority 1: добавить `codeql.yml` + secret scanning workflow.
2. Priority 2: добавить `release.yml` для сборки wheels (maturin) и release artifacts.
3. Priority 3: внедрить генерацию `THIRD_PARTY_LICENSES` и включить в релизный чеклист.
