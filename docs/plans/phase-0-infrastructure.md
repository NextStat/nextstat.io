# Фаза 0: Подготовка инфраструктуры

> **Execution note (humans + AI agents):** Выполнять задачи последовательно. Каноничные определения/допуски/детерминизм: `docs/plans/standards.md`.

**Goal:** Настроить репозиторий, CI/CD, инструменты разработки и юридическую базу проекта.

**Duration:** Недели 1-4

**Architecture:** Rust monorepo с workspace, Python bindings через PyO3/maturin.

**Tech Stack:** Rust 2024 edition, Python 3.11+, GitHub Actions, pre-commit.

---

## Содержание

- [Sprint 0.1: Юридическая подготовка](#sprint-01-юридическая-подготовка-неделя-1)
- [Sprint 0.2: Структура репозитория](#sprint-02-структура-репозитория-неделя-2)
- [Sprint 0.3: Development Environment](#sprint-03-development-environment-недели-3-4)
- [Критерии завершения фазы](#критерии-завершения-фазы)

---

## Sprint 0.1: Юридическая подготовка (Неделя 1)

### Epic 0.1.1: Лицензирование и IP

**Цель:** Защитить интеллектуальную собственность и установить правовую базу для Open Core модели.

---

#### Task 0.1.1.1: Выбор и настройка лицензии

**Priority:** P0 (критический)
**Estimated effort:** 2 часа
**Dependencies:** Нет

**Files:**
- Create: `LICENSE` (AGPL-3.0)
- Create: `LICENSE-COMMERCIAL`
- Create: `NOTICE`

**Acceptance Criteria:**
- [ ] AGPL-3.0 текст в корне репозитория
- [ ] Заголовок коммерческой лицензии
- [ ] NOTICE файл с attribution

**Step 1: Создать AGPL-3.0 лицензию**

```bash
curl -o LICENSE https://www.gnu.org/licenses/agpl-3.0.txt
```

**Step 2: Verify license downloaded**

```bash
head -5 LICENSE
# Expected: GNU AFFERO GENERAL PUBLIC LICENSE
```

**Step 3: Draft коммерческой лицензии (DRAFT — требуется ревью юриста)**

```markdown
# LICENSE-COMMERCIAL

NextStat Commercial License
===========================

Copyright (c) 2026 NextStat Inc. All rights reserved.

This software is available under a commercial license for organizations
that do not wish to comply with the AGPL-3.0 license terms.

For licensing inquiries, contact: licensing@nextstat.io

The open source version of this software is available under AGPL-3.0.
See LICENSE file for details.

---

Commercial License Benefits:
- No AGPL copyleft requirements
- Priority support and SLA
- Access to NS-Audit and NS-Compliance modules
- On-premise deployment assistance

Pricing is available upon request.
```

> Примечание: это **черновик заголовка**, а не юридически корректный текст лицензии.  
> До публичного релиза требуется консультация юриста и документ “open-core boundaries”
> (какой код OSS vs Pro, где хранится proprietary, как принимаются контрибы).

**Step 4: Создать NOTICE файл**

```text
NextStat
Copyright 2026 NextStat Inc.

This product includes software developed at NextStat Inc.

Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
unless a commercial license has been obtained.

Third-party dependencies and their licenses are listed in THIRD_PARTY_LICENSES.
```

**Step 5: Commit**

```bash
git add LICENSE LICENSE-COMMERCIAL NOTICE
git commit -m "chore(legal): add AGPL-3.0 and commercial licensing"
```

---

#### Task 0.1.1.2: Contribution policy (DCO vs CLA)

**Priority:** P0 (критический)
**Estimated effort:** 1-2 часа (draft) + counsel review при необходимости
**Dependencies:** Task 0.1.1.1

**Goal:** Выбрать и зафиксировать политику приёма контрибьютов для open-core проекта.

**Files:**
- Create (если выбран DCO): `DCO.md`
- Create (если выбран CLA): `CLA.md`

**Acceptance Criteria:**
- [ ] Решение **DCO или CLA** принято и отражено в `docs/legal/open-core-boundaries.md`
- [ ] В репозитории есть соответствующий документ (`DCO.md` или `CLA.md`)
- [ ] `CONTRIBUTING.md` ссылается на выбранную политику (см. Task 0.2.2.1)

**Step 1: Выбрать политику**

- **DCO (recommended default):** низкое трение для OSS, требует `Signed-off-by` в коммитах.
- **CLA:** иногда предпочтительнее для open-core, но требует процесса подписания (бот/DocuSign) и повышает friction.

**Step 2A (DCO): Добавить DCO.md**

```markdown
# Developer Certificate of Origin
# Version 1.1
#
# https://developercertificate.org/
#
# By making a contribution to this project, I certify that:
#
# (a) The contribution was created in whole or in part by me and I have the
#     right to submit it under the open source license indicated in the file; or
#
# (b) The contribution is based upon previous work that, to the best of my
#     knowledge, is covered under an appropriate open source license and I have
#     the right under that license to submit that work with modifications,
#     whether created in whole or in part by me, under the same open source
#     license (unless I am permitted to submit under a different license), as
#     indicated in the file; or
#
# (c) The contribution was provided directly to me by some other person who
#     certified (a), (b) or (c) and I have not modified it.
#
# (d) I understand and agree that this project and the contribution are public
#     and that a record of the contribution (including all personal information
#     I submit with it, including my sign-off) is maintained indefinitely and
#     may be redistributed consistent with this project or the open source
#     license(s) involved.
```

**Step 2B (CLA): Добавить CLA.md (optional)**

Если выбран CLA:
- добавить `CLA.md` (draft, counsel-reviewed),
- добавить CLA-бот/процесс подписания,
- обновить `CONTRIBUTING.md` и PR templates.

**Step 3: Commit**

```bash
# If DCO chosen:
git add DCO.md docs/legal/open-core-boundaries.md
git commit -m "chore(legal): adopt DCO contribution policy"

# If CLA chosen:
git add CLA.md docs/legal/open-core-boundaries.md
git commit -m "chore(legal): adopt CLA contribution policy"
```

---

#### Task 0.1.1.3: Open-core boundaries (OSS vs Pro) + repo structure

**Priority:** P0  
**Estimated effort:** 2-4 часа (без учёта юр.консультации)  
**Dependencies:** Task 0.1.1.1

**Deliverable:** Однозначный документ, который “прибивает гвоздями” границы open-core.

**Files:**
- Create: `docs/legal/open-core-boundaries.md`

**Acceptance Criteria:**
- [ ] Список OSS модулей и Pro модулей (по crates/пакетам)
- [ ] Решение: один repo vs split repos (public OSS + private Pro)
- [ ] Политика contributions: CLA vs DCO (и что принимается в OSS)
- [ ] Политика trademark/branding (что можно использовать в forks)
- [ ] Политика релизов: какие артефакты публикуются под AGPL

**Step 1: Create skeleton doc**

```markdown
# Open-core boundaries (NextStat) — Draft

> **DRAFT (requires counsel review).** Не является юридической консультацией.

## Principle

- OSS (AGPL): inference engine + reproducibility + core workflows
- Pro (Commercial): audit/compliance, orchestration, collaboration, UI

## OSS (AGPL)
- ns-core
- ns-compute
- ns-inference
- ns-translate
- ns-cli
- ns-py

## Pro (Commercial)
- ns-audit
- ns-compliance
- ns-scale
- ns-hub
- ns-dashboard

## Repository layout decision
- **Decision (Phase 0):** выбрать layout до первого внешнего контрибьютора/клиента.
  - Option A (default): public OSS repo + отдельный private Pro repo
  - Option B: monorepo + private submodule/split tooling

## Contributions
- **Decision:** DCO vs CLA (default: DCO для OSS; CLA — только если counsel рекомендует)
- See: `CONTRIBUTING.md` (+ `CLA.md` если выбран CLA)
- Security: vulnerabilities reported per `SECURITY.md`

## Trademark/branding
- Baseline: разрешить descriptive use (“compatible with NextStat”), ограничить использование лого/названия в derived products (draft; counsel review)
```

**Step 2: Commit**

```bash
git add docs/legal/open-core-boundaries.md
git commit -m "docs(legal): define open-core boundaries"
```

---

#### Task 0.1.1.4: Регистрация товарного знака

**Priority:** P1 (важный)
**Estimated effort:** 4 часа (исследование) + внешняя работа
**Dependencies:** Нет

**Deliverables:**
- [ ] Проверка доступности "NextStat" в USPTO TESS
- [ ] Проверка доступности в EUIPO eSearch
- [ ] Решение о подаче заявки

**Step 1: Проверить USPTO**

```
1. Открыть https://tmsearch.uspto.gov/
2. Выбрать "Basic Word Mark Search"
3. Искать: NEXTSTAT
4. Записать результаты
```

**Step 2: Проверить EUIPO**

```
1. Открыть https://euipo.europa.eu/eSearch/
2. Искать: NEXTSTAT
3. Записать результаты
```

**Step 3: Документировать решение**

Создать файл `docs/legal/trademark-search-results.md` с результатами поиска.

**Note:** Подача заявки на товарный знак — отдельный процесс, требующий юриста.

---

## Sprint 0.2: Структура репозитория (Неделя 2)

### Epic 0.2.1: Monorepo Setup

**Цель:** Создать структуру Rust workspace с Python bindings.

---

#### Task 0.2.1.1: Инициализация Rust workspace

**Priority:** P0 (критический)
**Estimated effort:** 2 часа
**Dependencies:** Git initialized

**Files:**
- Create: `Cargo.toml`
- Create: `rust-toolchain.toml`
- Create: `.cargo/config.toml`

**Acceptance Criteria:**
- [ ] `cargo build` проходит без ошибок
- [ ] Workspace содержит все планируемые crates
- [ ] Rust toolchain зафиксирован

**Step 1: Написать failing test (cargo check)**

```bash
cargo check
# Expected: FAIL - no Cargo.toml
```

**Step 2: Создать workspace Cargo.toml**

```toml
# Cargo.toml
[workspace]
resolver = "2"
members = [
    "crates/ns-core",
    "crates/ns-compute",
    "crates/ns-ad",
    "crates/ns-inference",
    "crates/ns-translate",
    "crates/ns-viz",
    "crates/ns-cli",
    "bindings/ns-py",
]

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "AGPL-3.0-or-later OR LicenseRef-Commercial"
repository = "https://github.com/nextstat/nextstat"
authors = ["NextStat Contributors"]
rust-version = "1.93"

[workspace.dependencies]
# Core numerics
ndarray = { version = "0.17", features = ["serde"] }
num-traits = "0.2"
approx = "0.5"
statrs = "0.18"
rayon = "1.11"

# Linear algebra / optimization
nalgebra = "0.34"
argmin = "0.11"
argmin-math = { version = "0.5", features = ["ndarray_latest"] }

# Error handling
thiserror = "2.0"
anyhow = "1.0"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde_yaml_ng = "0.10"

# Async runtime
tokio = { version = "1.0", features = ["full"] }

# CLI
clap = { version = "4.5", features = ["derive"] }

# Python bindings
pyo3 = { version = "0.28", features = ["extension-module"] }
numpy = "0.27"

# Testing
proptest = "1.10"
criterion = { version = "0.8", features = ["html_reports"] }

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

[profile.release]
lto = true
codegen-units = 1
panic = "abort"

[profile.bench]
lto = true
codegen-units = 1
```

**Step 3: Создать rust-toolchain.toml**

```toml
# rust-toolchain.toml
[toolchain]
channel = "1.93.0"
components = ["rustfmt", "clippy", "rust-analyzer"]
targets = ["x86_64-unknown-linux-gnu", "aarch64-apple-darwin"]
```

**Step 4: Создать cargo config**

```toml
# .cargo/config.toml
[build]
# Use mold linker on Linux for faster linking
# Uncomment if mold is installed:
# rustflags = ["-C", "link-arg=-fuse-ld=mold"]

[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=native"]

[target.aarch64-apple-darwin]
rustflags = ["-C", "target-cpu=native"]

[alias]
t = "test"
c = "check"
b = "build"
r = "run"
```

**Step 5: Создать минимальный ns-core crate**

```bash
mkdir -p crates/ns-core/src
```

```toml
# crates/ns-core/Cargo.toml
[package]
name = "ns-core"
version.workspace = true
edition.workspace = true
license.workspace = true
description = "NextStat core types and traits"

[dependencies]
ndarray.workspace = true
num-traits.workspace = true
thiserror.workspace = true
serde.workspace = true
serde_json.workspace = true

[dev-dependencies]
approx.workspace = true
proptest.workspace = true
```

```rust
// crates/ns-core/src/lib.rs
//! NextStat Core - fundamental types and traits
//!
//! This crate provides the core abstractions for statistical models.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod error;
pub mod model;
pub mod parameter;
pub mod types;

pub use error::{Error, Result};
pub use model::Model;
pub use parameter::Parameter;
```

**Step 6: Создать остальные placeholder crates**

```bash
for crate in ns-compute ns-ad ns-inference ns-translate ns-viz ns-cli; do
    mkdir -p crates/$crate/src
    cat > crates/$crate/Cargo.toml << EOF
[package]
name = "$crate"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
ns-core = { path = "../ns-core" }
EOF
    echo "//! $crate placeholder" > crates/$crate/src/lib.rs
done
```

**Step 7: Verify build**

```bash
cargo check
# Expected: PASS
```

**Step 8: Commit**

```bash
git add Cargo.toml rust-toolchain.toml .cargo/ crates/
git commit -m "chore: initialize Rust workspace with crate structure"
```

---

#### Task 0.2.1.2: Python bindings project setup (`bindings/ns-py`)

**Priority:** P0 (критический)  
**Estimated effort:** 1 час  
**Dependencies:** Task 0.2.1.1

**Files:**
- Create: `bindings/ns-py/pyproject.toml`
- Create: `bindings/ns-py/Cargo.toml`
- Create: `bindings/ns-py/src/lib.rs`
- Create: `bindings/ns-py/python/nextstat/__init__.py`
- Create: `bindings/ns-py/python/nextstat/py.typed`
- Create: `bindings/ns-py/python/nextstat/_core.pyi`

**Acceptance Criteria:**
- [ ] `cd bindings/ns-py && maturin develop --release` работает
- [ ] `python -c "import nextstat; print(nextstat.__version__)"` не падает
- [ ] Type stubs (`bindings/ns-py/python/nextstat/*.pyi`) включены в пакет

**Step 1: Sanity check (expected FAIL)**

```bash
python -c "import nextstat"
# Expected: FAIL - module not built/installed yet
```

**Step 2: Создать `bindings/ns-py/pyproject.toml`**

```toml
# bindings/ns-py/pyproject.toml
[build-system]
requires = ["maturin>=1.11,<2.0"]
build-backend = "maturin"

[project]
name = "nextstat"
version = "0.1.0"
description = "High-performance statistical fitting for High Energy Physics"
readme = "../../README.md"
requires-python = ">=3.11"
license = { text = "AGPL-3.0-or-later OR LicenseRef-Commercial" }
authors = [{ name = "NextStat Contributors" }]
keywords = ["statistics", "physics", "hep", "fitting", "likelihood"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Physics",
    "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Rust",
]

[project.optional-dependencies]
dev = [
    "mypy>=1.19",
    "pytest>=9.0",
    "pytest-cov>=7.0",
    "ruff>=0.15",
]
validation = [
    "numpy>=2.0",
    "pyhf>=0.7.6",
]

[project.urls]
Homepage = "https://nextstat.io"
Repository = "https://github.com/nextstat/nextstat"
Documentation = "https://docs.nextstat.io"

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "python"
module-name = "nextstat._core"
```

**Step 3: Создать Python package wrapper + type stubs**

```bash
mkdir -p bindings/ns-py/python/nextstat
```

```python
# bindings/ns-py/python/nextstat/__init__.py
try:
    from ._core import __version__  # type: ignore
except ImportError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
```

```python
# bindings/ns-py/python/nextstat/py.typed
# Marker file for PEP 561
```

```python
# bindings/ns-py/python/nextstat/_core.pyi
__version__: str
```

**Step 4: Verify build + import**

```bash
python -m pip install --upgrade pip
python -m pip install "maturin>=1.11,<2.0"

cd bindings/ns-py
maturin develop --release
python -c "import nextstat; print(nextstat.__version__)"
```

**Step 5: Commit**

```bash
git add bindings/ns-py/
git commit -m "chore(bindings): add ns-py maturin project skeleton"
```

---

#### Task 0.2.1.3: CI/CD Pipeline

**Priority:** P0 (критический)
**Estimated effort:** 2 часа
**Dependencies:** Tasks 0.2.1.1, 0.2.1.2

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/release.yml`
- Create: `.github/dependabot.yml`

**Acceptance Criteria:**
- [ ] CI runs on every PR
- [ ] Rust tests, clippy, fmt checked
- [ ] Python tests run
- [ ] Release workflow готов (disabled)

**Step 1: Создать CI workflow**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1
  RUSTFLAGS: "-D warnings"

jobs:
  # Rust checks
  rust-check:
    name: Rust Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6

      - name: Install Rust
        uses: dtolnay/rust-toolchain@v1
        with:
          components: rustfmt, clippy

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Check formatting
        run: cargo fmt --all -- --check

      - name: Clippy
        run: cargo clippy --all-targets --all-features

      - name: Build
        run: cargo build --all-features

  # Rust tests
  rust-test:
    name: Rust Test
    runs-on: ${{ matrix.os }}
    needs: rust-check
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v6

      - name: Install Rust
        uses: dtolnay/rust-toolchain@v1

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Run tests
        run: cargo test --all-features

  # Python tests
  python-test:
    name: Python Test
    runs-on: ubuntu-latest
    needs: rust-check
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v6

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v6
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install Rust
        uses: dtolnay/rust-toolchain@v1

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Install maturin
        run: pip install maturin

      - name: Build Python extension
        run: maturin develop

      - name: Install test dependencies
        run: pip install -e ".[dev,validation]"

      - name: Run Python tests
        run: pytest tests/python -v

      - name: Lint with ruff
        run: ruff check bindings/ns-py/python tests/

  # Benchmarks (only on main)
  bench:
    name: Benchmarks
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    needs: [rust-test, python-test]
    steps:
      - uses: actions/checkout@v6

      - name: Install Rust
        uses: dtolnay/rust-toolchain@v1

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Run benchmarks
        run: cargo bench --no-run

      # Optional: integrate benchmark tracking (e.g. bencher.dev) once benchmarks stabilize
```

**Step 2: Создать Dependabot config**

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "cargo"
    directory: "/"
    schedule:
      interval: "weekly"
    groups:
      rust-dependencies:
        patterns:
          - "*"

  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

**Step 3: Commit**

```bash
git add .github/
git commit -m "ci: add GitHub Actions workflows"
```

---

### Epic 0.2.2: Governance & Security

**Цель:** Подготовить репозиторий к внешним пользователям/контрибьюторам: политики, шаблоны, security scanning, licence compliance.

---

#### Task 0.2.2.1: Repo governance files (CONTRIBUTING/SECURITY/etc.)

**Priority:** P0  
**Estimated effort:** 2-3 часа  
**Dependencies:** Task 0.2.1.3

**Files:**
- Create: `CONTRIBUTING.md`
- Create: `CODE_OF_CONDUCT.md`
- Create: `SECURITY.md`
- Create: `.github/PULL_REQUEST_TEMPLATE.md`
- Create: `.github/CODEOWNERS`
- Create: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Create: `.github/ISSUE_TEMPLATE/feature_request.yml`

**Acceptance Criteria:**
- [ ] У репозитория есть базовые public policies
- [ ] Issues/PR стандартизированы шаблонами
- [ ] Есть понятный security disclosure process

**Step 1: Add minimal templates**

```markdown
# CONTRIBUTING.md

## Development

- Run Rust tests: `cargo test --all`
- Run Python tests: `pytest -v`

## Contribution policy

- Follow the policy in `DCO.md` (default) or `CLA.md` (if adopted).
- If using DCO: all commits must include `Signed-off-by:`.

## PR checklist

- [ ] Tests green
- [ ] `cargo fmt` and `cargo clippy` clean
- [ ] Public API/doc changes documented
```

```markdown
# SECURITY.md

## Reporting a Vulnerability

Please report security issues privately.

- Email: security@nextstat.io
- Expected response time: 72 hours

Do not open public issues for vulnerabilities.
```

```markdown
# CODE_OF_CONDUCT.md

This project follows the Contributor Covenant Code of Conduct.
```

```text
# .github/CODEOWNERS
* @andresvlc
```

**Step 2: Commit**

```bash
git add CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md .github/
git commit -m "docs: add repo governance and templates"
```

---

#### Task 0.2.2.2: Security scanning in CI (CodeQL + deps audit)

**Priority:** P0  
**Estimated effort:** 2-3 часа  
**Dependencies:** Task 0.2.1.3

**Files:**
- Create: `.github/workflows/codeql.yml`
- Create: `.github/workflows/secret-scan.yml`
- Create: `.deny.toml`
- Modify: `.github/workflows/ci.yml` (add `cargo audit` + `pip-audit`)

**Acceptance Criteria:**
- [ ] CodeQL runs on PRs + main
- [ ] Secret scan runs on PRs + main
- [ ] Dependency audits run in CI
- [ ] License/advisory policy enforced (`cargo deny`)
- [ ] Findings are visible in GitHub Security tab

**Step 1: Add CodeQL**

```yaml
# .github/workflows/codeql.yml
name: CodeQL

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: "0 6 * * 1"

jobs:
  analyze:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      actions: read
      contents: read
    steps:
      - uses: actions/checkout@v6
      - uses: github/codeql-action/init@v4
        with:
          languages: python
      - uses: github/codeql-action/analyze@v4
```

**Step 1.1: Add secret scanning (gitleaks)**

```yaml
# .github/workflows/secret-scan.yml
name: Secret Scan (gitleaks)

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  gitleaks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 0
      - uses: gitleaks/gitleaks-action@v2
```

**Step 2: Add dependency audit steps (example)**

```yaml
# .github/workflows/ci.yml (add to an existing job)
- uses: taiki-e/install-action@v2
  with:
    tool: cargo-audit

- name: Rust dependency audit
  run: cargo audit

- uses: taiki-e/install-action@v2
  with:
    tool: cargo-deny

- name: Rust license/advisory policy (cargo-deny)
  run: cargo deny check

- name: Python dependency audit
  run: pip install pip-audit && pip-audit
```

**Step 2.1: Add minimal `cargo-deny` policy**

```toml
# .deny.toml (minimal; refine as project grows)
[advisories]
ignore = []

[licenses]
allow = ["Apache-2.0", "MIT", "BSD-3-Clause", "ISC", "Zlib"]
confidence-threshold = 0.8
```

**Step 3: Commit**

```bash
git add .github/workflows/codeql.yml .github/workflows/secret-scan.yml .github/workflows/ci.yml .deny.toml
git commit -m "ci(security): add CodeQL and dependency audits"
```

---

#### Task 0.2.2.3: Generate `THIRD_PARTY_LICENSES`

**Priority:** P1  
**Estimated effort:** 2-4 часа  
**Dependencies:** Task 0.2.2.1

**Files:**
- Create: `THIRD_PARTY_LICENSES`
- Create: `scripts/generate_third_party_licenses.sh`

**Acceptance Criteria:**
- [ ] `THIRD_PARTY_LICENSES` существует и обновляемый скриптом
- [ ] `NOTICE` ссылается на реальный файл

**Step 1: Add generator script**

```bash
# scripts/generate_third_party_licenses.sh
set -euo pipefail

OUT="THIRD_PARTY_LICENSES"

echo "# Third-party licenses" > "$OUT"
echo "" >> "$OUT"
echo "Generated (UTC): $(date -u '+%Y-%m-%d %H:%M:%S')" >> "$OUT"
echo "" >> "$OUT"

echo "## Rust (Cargo)" >> "$OUT"
cargo metadata --format-version 1 | python3 - <<'PY' >> "$OUT"
import json, sys
data = json.load(sys.stdin)
rows = []
for p in data.get("packages", []):
    name = p.get("name")
    ver = p.get("version")
    lic = p.get("license") or "UNKNOWN"
    rows.append((name, ver, lic))
for name, ver, lic in sorted(set(rows)):
    print(f"- {name} {ver} — {lic}")
PY

echo "" >> "$OUT"
echo "## Python (pip)" >> "$OUT"
python3 - <<'PY' >> "$OUT"
import importlib.metadata as md

rows = []
for dist in md.distributions():
    meta = dist.metadata
    name = meta.get("Name") or dist.name
    version = meta.get("Version") or dist.version
    license_ = meta.get("License") or "UNKNOWN"
    rows.append((name, version, license_))

for name, version, license_ in sorted(set(rows)):
    print(f"- {name} {version} — {license_}")
PY
```

**Step 2: Commit**

```bash
git add THIRD_PARTY_LICENSES scripts/generate_third_party_licenses.sh
git commit -m "chore(legal): add third-party license report generator"
```

---

## Sprint 0.3: Development Environment (Недели 3-4)

### Epic 0.3.1: Developer Tools

---

#### Task 0.3.1.1: Pre-commit hooks

**Priority:** P1 (важный)
**Estimated effort:** 1 час
**Dependencies:** Sprint 0.2

**Files:**
- Create: `.pre-commit-config.yaml`
- Create: `rustfmt.toml`

**Step 1: Создать pre-commit config**

```yaml
# .pre-commit-config.yaml
repos:
  # Rust hooks
  - repo: local
    hooks:
      - id: cargo-fmt
        name: cargo fmt
        entry: cargo fmt --all --
        language: system
        types: [rust]
        pass_filenames: false

      - id: cargo-clippy
        name: cargo clippy
        entry: cargo clippy --workspace --all-targets --all-features -- -D warnings
        language: system
        types: [rust]
        pass_filenames: false
        stages: [pre-push]

      - id: cargo-test
        name: cargo test
        entry: cargo test --workspace --all-features
        language: system
        types: [rust]
        pass_filenames: false
        stages: [pre-push]

  # Python hooks
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  # General hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: detect-private-key

  # Commit message
  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v4.3.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]
        args: [feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert]
```

**Step 2: Создать rustfmt config**

```toml
# rustfmt.toml
edition = "2024"
max_width = 100
use_small_heuristics = "Max"
imports_granularity = "Module"
group_imports = "StdExternalCrate"
reorder_imports = true
reorder_modules = true
remove_nested_parens = true
use_field_init_shorthand = true
use_try_shorthand = true
format_code_in_doc_comments = true
format_macro_matchers = true
format_strings = false
wrap_comments = true
comment_width = 80
normalize_comments = true
normalize_doc_attributes = true
```

**Step 3: Установить pre-commit**

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push
```

**Step 4: Commit**

```bash
git add .pre-commit-config.yaml rustfmt.toml
git commit -m "chore: add pre-commit hooks"
```

---

#### Task 0.3.1.2: Test fixtures setup

**Priority:** P0 (критический)
**Estimated effort:** 2 часа
**Dependencies:** None

**Files:**
- Create: `tests/fixtures/simple_workspace.json`
- Create: `tests/fixtures/complex_workspace.json`
- Create: `tests/fixtures/README.md`

**Step 1: Создать простой workspace**

```json
{
  "channels": [
    {
      "name": "singlechannel",
      "samples": [
        {
          "name": "signal",
          "data": [5.0, 10.0],
          "modifiers": [
            {"name": "mu", "type": "normfactor", "data": null}
          ]
        },
        {
          "name": "background",
          "data": [50.0, 60.0],
          "modifiers": [
            {
              "name": "bkg_uncert",
              "type": "staterror",
              "data": [5.0, 6.0]
            }
          ]
        }
      ]
    }
  ],
  "observations": [
    {"name": "singlechannel", "data": [55.0, 70.0]}
  ],
  "measurements": [
    {
      "name": "Measurement",
      "config": {
        "poi": "mu",
        "parameters": []
      }
    }
  ],
  "version": "1.0.0"
}
```

**Step 2: Создать complex workspace**

```json
{
  "channels": [
    {
      "name": "SR",
      "samples": [
        {
          "name": "signal",
          "data": [12.0, 15.0, 8.0],
          "modifiers": [
            {"name": "mu", "type": "normfactor", "data": null},
            {"name": "lumi", "type": "lumi", "data": null},
            {
              "name": "signal_theory",
              "type": "normsys",
              "data": {"hi": 1.1, "lo": 0.9}
            }
          ]
        },
        {
          "name": "background1",
          "data": [100.0, 120.0, 80.0],
          "modifiers": [
            {"name": "lumi", "type": "lumi", "data": null},
            {
              "name": "bkg1_norm",
              "type": "normsys",
              "data": {"hi": 1.05, "lo": 0.95}
            },
            {
              "name": "bkg1_shape",
              "type": "histosys",
              "data": {
                "hi_data": [105.0, 125.0, 82.0],
                "lo_data": [95.0, 115.0, 78.0]
              }
            }
          ]
        },
        {
          "name": "background2",
          "data": [30.0, 25.0, 35.0],
          "modifiers": [
            {
              "name": "bkg2_stat",
              "type": "staterror",
              "data": [5.0, 4.0, 6.0]
            }
          ]
        }
      ]
    },
    {
      "name": "CR",
      "samples": [
        {
          "name": "background1",
          "data": [500.0, 600.0],
          "modifiers": [
            {"name": "lumi", "type": "lumi", "data": null},
            {
              "name": "bkg1_norm",
              "type": "normsys",
              "data": {"hi": 1.05, "lo": 0.95}
            }
          ]
        }
      ]
    }
  ],
  "observations": [
    {"name": "SR", "data": [142.0, 160.0, 123.0]},
    {"name": "CR", "data": [510.0, 590.0]}
  ],
  "measurements": [
    {
      "name": "SignalStrength",
      "config": {
        "poi": "mu",
        "parameters": [
          {
            "name": "lumi",
            "auxdata": [1.0],
            "sigmas": [0.02],
            "bounds": [[0.9, 1.1]],
            "inits": [1.0]
          }
        ]
      }
    }
  ],
  "version": "1.0.0"
}
```

**Step 3: Создать README для fixtures**

```markdown
# Test Fixtures

This directory contains JSON workspace files for testing.

## Files

### simple_workspace.json

Minimal workspace with:
- 1 channel, 2 bins
- 1 signal sample with normfactor (POI)
- 1 background sample with staterror

Use for basic parsing and NLL tests.

### complex_workspace.json

More realistic workspace with:
- 2 channels (SR + CR)
- Multiple samples with various modifiers
- normsys, histosys, staterror, lumi
- Correlated systematics across channels

Use for integration tests and validation.

## Reference Values

Reference values are computed with pyhf. To regenerate:

```python
import pyhf
import json

ws = pyhf.Workspace(json.load(open("simple_workspace.json")))
model = ws.model()
data = ws.data(model)

bestfit = pyhf.infer.mle.fit(data, model)
twice_nll = float(pyhf.infer.mle.twice_nll(bestfit, data, model).item())

print("bestfit:", bestfit.tolist())
print("twice_nll:", twice_nll)

# Note: pyhf 0.7.x doesn't return uncertainties directly.
# For a numerical Hessian reference, see `tests/python/test_pyhf_validation.py`.
```

## Adding New Fixtures

When adding fixtures:
1. Ensure they are valid pyhf workspaces
2. Add reference values to this README
3. Add corresponding tests
```

**Step 4: Commit**

```bash
git add tests/fixtures/
git commit -m "test: add pyhf workspace fixtures"
```

---

#### Task 0.3.1.3: Benchmark infrastructure

**Priority:** P1 (важный)
**Estimated effort:** 1 час
**Dependencies:** Task 0.2.1.1

**Files:**
- Create: `benches/README.md`
- Create: `crates/ns-compute/benches/nll_bench.rs`
- Modify: `crates/ns-compute/Cargo.toml`

**Step 1: Создать benchmark structure**

```markdown
# benches/README.md

# Benchmarks

Run benchmarks with:

```bash
cargo bench -p ns-compute
```

Results are saved to `target/criterion/`.

## Benchmarks

- `nll_bench`: NLL computation performance
- `fit_bench`: MLE fitting performance (planned: Phase 1.3)
- `gradient_bench`: AD gradient computation (planned: Phase 2B)

## Tracking

We track performance over time. Regression > 10% triggers CI failure.
```

**Step 2: Создать placeholder benchmark**

```rust
// crates/ns-compute/benches/nll_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use statrs::function::gamma::ln_gamma;

fn nll_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("nll");

    // Sizes to benchmark
    for n_bins in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("poisson_nll", n_bins),
            n_bins,
            |b, &n| {
                let observed: Vec<f64> = (0..n).map(|i| (i % 100) as f64).collect();
                let expected: Vec<f64> = (0..n).map(|i| (i % 100 + 1) as f64).collect();
                // Observed-only constants (do not depend on λ): Σ ln Γ(n+1)
                let const_terms: f64 = observed.iter().map(|&n| ln_gamma(n + 1.0)).sum();

                b.iter(|| {
                    // Benchmark the variable part: Σ (λ - n ln λ)
                    let var_terms: f64 = observed
                        .iter()
                        .zip(expected.iter())
                        .map(|(&n, &lam)| if lam > 0.0 { lam - n * lam.ln() } else { f64::INFINITY })
                        .sum();
                    let nll = const_terms + var_terms;
                    black_box(nll)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, nll_benchmark);
criterion_main!(benches);
```

**Step 3: Добавить bench в Cargo.toml**

```toml
# crates/ns-compute/Cargo.toml

[dependencies]
ns-core = { path = "../ns-core" }
statrs.workspace = true

[dev-dependencies]
criterion.workspace = true

[[bench]]
name = "nll_bench"
harness = false
```

**Step 4: Commit**

```bash
git add benches/ crates/ns-compute/Cargo.toml crates/ns-compute/benches/
git commit -m "bench: add criterion benchmark infrastructure"
```

---

## Критерии завершения фазы

### Checklist

- [ ] **Юридическое:**
  - [ ] AGPL-3.0 лицензия в репозитории
  - [ ] `docs/legal/open-core-boundaries.md` готов (draft → counsel review)
  - [ ] Политика contributions определена (DCO или CLA) + документация (`DCO.md`/`CLA.md`)
  - [ ] Trademark search выполнен

- [ ] **Репозиторий:**
  - [ ] Cargo workspace настроен
  - [ ] Python package настроен
  - [ ] Все crate placeholders созданы
  - [ ] `cargo build` проходит
  - [ ] `pip install -e .` работает

- [ ] **CI/CD:**
  - [ ] GitHub Actions CI настроен
  - [ ] Rust tests запускаются
  - [ ] Python tests запускаются
  - [ ] Dependabot настроен

- [ ] **Dev Tools:**
  - [ ] Pre-commit hooks установлены
  - [ ] Rustfmt настроен
  - [ ] Ruff настроен
  - [ ] Test fixtures созданы
  - [ ] Benchmark infrastructure готова

### Exit Criteria

Фаза считается завершённой когда:

1. `cargo test` проходит (пустые тесты OK)
2. `pytest` проходит (пустые тесты OK)
3. CI pipeline зелёный
4. Pre-commit hooks работают локально
5. Все документы в `docs/` актуальны

---

*Следующая фаза: [Phase 1: MVP-α Core Engine](./phase-1-mvp-alpha.md)*
