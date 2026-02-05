<!--
Audit Report
Generated: 2026-02-05 18:20:20 +0100
Git Commit: unknown
Scope: docs/plans + repo infra (versions/CI/security/release) + bias/coverage policy
Invoked: /audit plans-infra-bias
-->

# Project Audit Report

Date: 2026-02-05  
Project: NextStat (nextstat.io)  
Auditor: Codex  
Scope: `docs/plans` + CI/security/release automation + bias/coverage policy  
Git Commit: unknown

---

## Executive Summary

- Critical: 0
- Major: 0
- Minor: 2
- Total: 2

Overall Health Score: 97/100  
Top risks: долгосрочно — качество “toys validation” зависит от реализации `fit(..., data=...)` path и ночных прогонов; краткосрочно — нужно следить за drift зависимостей через Dependabot/`scripts/versions_audit.py`.

---

## Critical Issues

None found in the audited scope.

## Major Issues

None found in the audited scope.

## Minor Issues

### [Compliance] Python license metadata частично UNKNOWN в `THIRD_PARTY_LICENSES`
- File: `THIRD_PARTY_LICENSES:122`
- Evidence: ряд пакетов (например `numpy`, `pytest`, `maturin`) не заполняют `info.license` и/или license classifiers на PyPI → отчёт показывает `UNKNOWN`.
- Impact: license-report остаётся полезным, но не всегда даёт уверенный ответ для Python deps.
- Fix: в Phase 3/релизный процесс добавить генерацию python license report из lockfile/собранного wheel environment (например `pip-licenses`) и/или ручные clarifications для critical deps.
- Confidence: medium

### [Testing] Bias/coverage gates описаны, но требуют nightly automation
- File: `docs/plans/standards.md:118`
- Evidence: bias/pull/coverage policy и tasks добавлены, но workflow для toys validation пока только как опциональная задача (Phase 3).
- Impact: без регулярных прогонов легко пропустить статистическое расхождение vs pyhf при оптимизациях.
- Fix: добавить `.github/workflows/toys-validation.yml` (nightly/manual) как часть Phase 3 acceptance criteria и сохранить JSON artifacts.
- Confidence: high

---

## Feature Checklist

### Plans + Infrastructure
- [x] Версии/пины подтверждены snapshot-скриптом
- [x] BIAS policy (bias/pull/coverage) зафиксирована + добавлены tasks/gates
- [x] Security automation: CodeQL + gitleaks + dep audit workflow
- [x] Release automation: wheels + CLI artifacts в GitHub Release
- [x] Third-party licenses report + generator script

---

## Unverified Areas

- Реальный end-to-end release (wheels install/run on fresh machines) не проверялся.
- CodeQL результаты зависят от включения GitHub Security features на уровне репозитория.

## Files Reviewed

- `docs/plans/*`
- `.github/workflows/*`
- `scripts/versions_audit.py`
- `scripts/generate_third_party_licenses.sh`
- `crates/ns-translate/src/pyhf/model.rs`

## Recommendations

1. Priority 1: добавить nightly toys-validation workflow (Phase 3) + artifacts.
2. Priority 2: улучшить Python license reporting (lockfile-based) перед первым публичным релизом.
3. Priority 3: держать `docs/plans/versions.md` синхронизированным с реальным CI (через `scripts/versions_audit.py` в ревью).
