# NextStat: Полный План Реализации

> **Execution note (humans + AI agents):** План рассчитан на последовательное выполнение task-by-task (TDD там, где уместно). Каноничные определения/допуски/детерминизм: `docs/plans/standards.md`.

**Goal:** Создать высокопроизводительный статистический фреймворк для profile likelihood fits с поддержкой HEP, финансов и медицины.

**Architecture:** Rust-ядро с AD (автоматическое дифференцирование), Python API через PyO3, GPU-ускорение через CUDA/Metal. Open Core модель (AGPL + Commercial).

**Tech Stack:** Rust, PyO3, Python, JAX/PyTorch backends, CUDA, Protocol Buffers, YAML configs.

---

## Содержание

1. [Обзор проекта](#1-обзор-проекта)
2. [Входные данные и критерии приемки](#2-входные-данные-и-критерии-приемки)
3. [Фаза 0: Подготовка инфраструктуры](#фаза-0-подготовка-инфраструктуры-недели-1-4)
4. [Фаза I: MVP-α Core Engine](#фаза-i-mvp-α-core-engine-месяцы-2-4)
5. [Фаза II: MVP-β Performance (CPU + AD + optional GPU)](#фаза-ii-mvp-β-performance-cpu--ad--optional-gpu-месяцы-4-9)
6. [Фаза III: Production Ready](#фаза-iii-production-ready-месяцы-9-15)
7. [Фаза IV: Enterprise и SaaS](#фаза-iv-enterprise-и-saas-месяцы-15-24)
8. [Приложения](#приложения)

---

## 1. Обзор проекта

### 1.1 Цели NextStat

NextStat — статистический фреймворк нового поколения для:
- **HEP (физика высоких энергий):** замена TRExFitter + pyhf
- **Финансы:** Model Risk Management, Basel III/IFRS 9
- **Медицина:** 21 CFR Part 11 compliant analysis

### 1.2 Конкурентное позиционирование

```
                    Binned only              Unbinned support
                    ┌─────────────────┬─────────────────────┐
  Frequentist only  │  TRExFitter     │      RooFit         │
                    │  pyhf           │                     │
                    ├─────────────────┼─────────────────────┤
  Freq + Bayesian   │                 │   ★ NextStat ★     │
                    │                 │   Stan (partial)    │
                    └─────────────────┴─────────────────────┘
```

### 1.3 Архитектура системы

```
┌─────────────────────────────────────────────────────────────────┐
│              OPEN SOURCE (AGPL v3)                              │
│                                                                 │
│  NS-Compute    — Rust AD engine, SIMD, GPU                      │
│  NS-Inference  — MLE, NUTS, Profile Likelihood                  │
│  NS-Translate  — pyhf JSON, HistFactory XML                     │
│  NS-Core-Viz   — Pull plots, correlation matrix                 │
│  CLI           — nextstat fit / validate / bench                │
│  Python API    — nextstat-py                                    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────────┐
│            COMMERCIAL (NextStat Pro)                            │
│                                                                 │
│  NS-Audit      — 21 CFR Part 11 compliant                       │
│  NS-Compliance — Basel III/IFRS 9 report gen                    │
│  NS-Scale      — Distributed fitting (Ray/K8s)                  │
│  NS-Hub        — Model registry, versioning                     │
│  NS-Dashboard  — Real-time monitoring, what-if UI               │
│  Support       — SLA, priority fixes, consulting                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Входные данные и критерии приемки

### 2.1 Референсные документы

| Документ | Описание | Использование |
|----------|----------|---------------|
| [`docs/references/NextStat_Business_Strategy_Analysis.md`](../references/NextStat_Business_Strategy_Analysis.md) | Стратегия монетизации, тайминг, риски | Бизнес-требования, приоритизация |
| [`docs/references/TRExFitter_vs_NextStat_Analysis.md`](../references/TRExFitter_vs_NextStat_Analysis.md) | Архитектурный анализ TRExFitter | Технические требования, lessons learned |
| [`docs/legal/open-core-boundaries.md`](../legal/open-core-boundaries.md) | Границы OSS/Pro, contribution policy | Юридическая исполнимость open-core |
| pyhf JSON Schema | Формат HistFactory workspace | Совместимость |
| HistFactory XML | Legacy формат ROOT | Импорт существующих моделей |

### 2.2 Глобальные критерии приемки (Definition of Done)

Каждая задача считается завершённой если:

1. **Код:**
   - [ ] Все тесты проходят (`cargo test`, `pytest`)
   - [ ] Покрытие тестами ≥ 80% для нового кода
   - [ ] Нет warnings от `cargo clippy` и `ruff`
   - [ ] Документация API обновлена

2. **TDD:**
   - [ ] Тест написан ДО реализации
   - [ ] Тест сначала красный, потом зелёный
   - [ ] Минимальный код для прохождения теста

3. **Численная корректность:**
   - [ ] Результаты совпадают с референсом (pyhf/ROOT) до 6 знаков
   - [ ] Property-based тесты проходят
   - [ ] Benchmark зафиксирован
   - [ ] Bias/pull/coverage не ухудшается относительно pyhf там, где применимо (policy: `docs/plans/standards.md`, раздел 6)

4. **Git:**
   - [ ] Atomic commits (одна логическая единица)
   - [ ] Conventional commits формат
   - [ ] PR review пройден

### 2.3 Численные референсы

| Метрика | Референс | Источник | Допуск |
|---------|----------|----------|--------|
| MLE μ̂ | pyhf result | pyhf v0.7.6 | ±1e-6 |
| σ(μ̂) | pyhf Hessian | pyhf v0.7.6 | ±1e-5 |
| NLL value | pyhf twice_nll | pyhf v0.7.6 | ±1e-8 |
| Pull(mean/std) for μ | pyhf toys | pyhf v0.7.6 | Δmean≤0.05, Δstd≤0.05 |
| Gradient | JAX autodiff | JAX | ±1e-7 |
| Profile likelihood | pyhf scan | pyhf v0.7.6 | ±1e-5 |

### 2.4 Performance Targets

| Операция | Baseline (pyhf) | Target (NextStat) | Фаза |
|----------|-----------------|-------------------|------|
| Simple fit (10 NP) | 50ms | <10ms | MVP-α |
| Complex fit (100 NP) | 2s | <200ms | MVP-β |
| Ranking plot (100 NP) | 6 min | <30s | MVP-β |
| 1000 toy fits | 30 min | <1 min (GPU) | Phase III |

---

## Фаза 0: Подготовка инфраструктуры (Недели 1-4)

### Цели фазы
- Настроить репозиторий и CI/CD
- Выбрать и настроить инструменты
- Создать скелет проекта

### Sprint 0.1: Юридическая и организационная подготовка (Неделя 1)

#### Epic 0.1.1: Юридическая структура

**Task 0.1.1.1: Выбор лицензии**

**Files:**
- Create: `LICENSE`
- Create: `LICENSE-COMMERCIAL`
- Create: `CLA.md`

**Step 1: Создать AGPL-3.0 лицензию**

```bash
# Скачать официальный текст AGPL-3.0
curl -o LICENSE https://www.gnu.org/licenses/agpl-3.0.txt
```

**Step 2: Создать заголовок для коммерческой лицензии**

```markdown
# NextStat Commercial License

Copyright (c) 2026 NextStat Inc.

This software is licensed under a commercial license.
Contact sales@nextstat.io for licensing terms.

The open source version is available under AGPL-3.0.
```

**Step 3: Создать CLA (Contributor License Agreement)**

```markdown
# NextStat Contributor License Agreement

By contributing to NextStat, you agree that:

1. You grant NextStat Inc. a perpetual, worldwide, non-exclusive,
   royalty-free license to use your contributions.

2. You have the right to submit the contribution.

3. Your contribution may be relicensed under commercial terms.
```

**Step 4: Commit**

```bash
git add LICENSE LICENSE-COMMERCIAL CLA.md
git commit -m "chore: add AGPL-3.0 and commercial licensing"
```

---

**Task 0.1.1.2: Регистрация товарного знака**

**Deliverable:** Заявка на регистрацию "NextStat" в USPTO/EUIPO

**Checklist:**
- [ ] Проверить доступность через TESS/eSearch
- [ ] Подготовить описание товаров/услуг (класс 9, 42)
- [ ] Подать заявку или нанять патентного поверенного

---

### Sprint 0.2: Репозиторий и инструменты (Неделя 2)

#### Epic 0.2.1: Структура репозитория

**Task 0.2.1.1: Создать monorepo структуру**

**Files:**
- Create: `Cargo.toml` (workspace)
- Create: `pyproject.toml`
- Create: `.github/workflows/ci.yml`

**Step 1: Инициализировать Rust workspace**

```toml
# Cargo.toml (root)
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
	rust-version = "1.93"
	license = "AGPL-3.0-or-later OR LicenseRef-Commercial"
	repository = "https://github.com/nextstat/nextstat"

[workspace.dependencies]
# Core
ndarray = "0.17"
num-traits = "0.2"
thiserror = "2.0"
anyhow = "1.0"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde_yaml_ng = "0.10"

# Async
tokio = { version = "1.0", features = ["full"] }

# CLI
clap = { version = "4.5", features = ["derive"] }

# Python bindings
pyo3 = { version = "0.28", features = ["extension-module"] }
numpy = "0.27"

# Testing
approx = "0.5"
proptest = "1.10"
criterion = "0.8"

# Numerics / performance
statrs = "0.18"
rayon = "1.11"

# Linear algebra / optimization
nalgebra = "0.34"
argmin = "0.11"
argmin-math = { version = "0.5", features = ["ndarray_latest"] }

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

**Step 2: Создать Python project**

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.11,<2.0"]
build-backend = "maturin"

[project]
name = "nextstat"
version = "0.1.0"
description = "High-performance statistical fitting framework"
readme = "README.md"
license = { text = "AGPL-3.0-or-later OR LicenseRef-Commercial" }
requires-python = ">=3.11"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU Affero General Public License v3",
    "Programming Language :: Python :: 3",
    "Programming Language :: Rust",
    "Topic :: Scientific/Engineering :: Physics",
]
dependencies = [
    "numpy>=2.0",
    "pyhf>=0.7.6",  # for validation
]

[project.optional-dependencies]
dev = [
    "pytest>=9.0",
    "pytest-cov>=7.0",
    "ruff>=0.15",
    "mypy>=1.19",
]
jax = ["jax>=0.4", "jaxlib>=0.4"]
torch = ["torch>=2.0"]

[project.scripts]
nextstat = "nextstat.cli:main"

[tool.maturin]
features = ["pyo3/extension-module"]
module-name = "nextstat._core"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=nextstat --cov-report=term-missing"
```

**Step 3: Создать CI pipeline**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  rust-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: dtolnay/rust-toolchain@v1
        with:
          components: clippy, rustfmt
      - uses: Swatinem/rust-cache@v2
      - name: Check formatting
        run: cargo fmt --all -- --check
      - name: Clippy
        run: cargo clippy --all-targets --all-features -- -D warnings
      - name: Test
        run: cargo test --all-features

  python-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.11"
      - uses: dtolnay/rust-toolchain@v1
      - name: Install dependencies
        run: |
          pip install maturin
          pip install -e ".[dev]"
      - name: Lint
        run: ruff check .
      - name: Test
        run: pytest

  bench:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v6
      - uses: dtolnay/rust-toolchain@v1
      - name: Run benchmarks
        run: cargo bench --no-run
```

**Step 4: Commit**

```bash
git add Cargo.toml pyproject.toml .github/
git commit -m "chore: setup monorepo with Rust workspace and Python bindings"
```

---

**Task 0.2.1.2: Создать структуру директорий**

**Files:**
- Create: `crates/ns-core/Cargo.toml`
- Create: `crates/ns-core/src/lib.rs`
- Create: множество директорий

**Step 1: Создать все crates**

```bash
mkdir -p crates/{ns-core,ns-compute,ns-inference,ns-translate,ns-viz,ns-cli,ns-py}/src
mkdir -p tests/{rust,python}
mkdir -p benches
mkdir -p docs/{api,guides,plans}
mkdir -p examples/{hep,finance,medical}
```

**Step 2: Создать ns-core Cargo.toml**

```toml
# crates/ns-core/Cargo.toml
[package]
name = "ns-core"
version.workspace = true
edition.workspace = true
license.workspace = true

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

**Step 3: Создать минимальный lib.rs**

```rust
// crates/ns-core/src/lib.rs
//! NextStat Core - fundamental types and traits
//!
//! This crate provides the core abstractions for statistical models:
//! - `Model` trait for defining statistical models
//! - `Parameter` and `Modifier` types
//! - Error types and results

pub mod error;
pub mod model;
pub mod parameter;
pub mod types;

pub use error::{Error, Result};
pub use model::Model;
pub use parameter::Parameter;
```

**Step 4: Создать placeholder modules**

```rust
// crates/ns-core/src/error.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Model specification error: {0}")]
    ModelSpec(String),

    #[error("Numerical error: {0}")]
    Numerical(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
```

```rust
// crates/ns-core/src/types.rs
//! Fundamental numeric types

/// Floating point precision for all calculations
pub type Float = f64;

/// Index type for bins, parameters, etc.
pub type Index = usize;

/// Hash for audit trail
pub type Hash = [u8; 32];
```

```rust
// crates/ns-core/src/parameter.rs
use serde::{Deserialize, Serialize};
use crate::types::Float;

/// A parameter in the statistical model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    /// Unique name
    pub name: String,
    /// Initial value
    pub init: Float,
    /// Lower bound (None = unbounded)
    pub lower: Option<Float>,
    /// Upper bound (None = unbounded)
    pub upper: Option<Float>,
    /// Is this parameter fixed?
    pub fixed: bool,
}

impl Parameter {
    pub fn new(name: impl Into<String>, init: Float) -> Self {
        Self {
            name: name.into(),
            init,
            lower: None,
            upper: None,
            fixed: false,
        }
    }

    pub fn bounded(mut self, lower: Float, upper: Float) -> Self {
        self.lower = Some(lower);
        self.upper = Some(upper);
        self
    }

    pub fn fixed(mut self) -> Self {
        self.fixed = true;
        self
    }
}
```

```rust
// crates/ns-core/src/model.rs
use crate::{parameter::Parameter, types::Float, Result};

/// Trait for statistical models that can be fitted
pub trait Model {
    /// Return the negative log-likelihood at given parameter values
    fn nll(&self, params: &[Float]) -> Result<Float>;

    /// Return gradient of NLL (if available)
    fn gradient(&self, params: &[Float]) -> Result<Vec<Float>> {
        let _ = params;
        Err(crate::Error::Numerical("Gradient not implemented".into()))
    }

    /// Return list of parameters
    fn parameters(&self) -> &[Parameter];

    /// Number of parameters
    fn n_params(&self) -> usize {
        self.parameters().len()
    }
}
```

**Step 5: Verify compilation**

```bash
cd crates/ns-core && cargo build
```

**Step 6: Commit**

```bash
git add crates/ tests/ benches/ docs/ examples/
git commit -m "chore: create crate structure and core types"
```

---

### Sprint 0.3: Development Environment (Неделя 3-4)

#### Epic 0.3.1: Настройка инструментов разработки

**Task 0.3.1.1: Pre-commit hooks**

**Files:**
- Create: `.pre-commit-config.yaml`
- Create: `rustfmt.toml`
- Create: `.clippy.toml`

**Step 1: Создать pre-commit config**

```yaml
# .pre-commit-config.yaml
repos:
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
        entry: cargo clippy --all-targets --all-features -- -D warnings
        language: system
        types: [rust]
        pass_filenames: false

      - id: cargo-test
        name: cargo test
        entry: cargo test
        language: system
        types: [rust]
        pass_filenames: false

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
```

**Step 2: Создать rustfmt config**

```toml
# rustfmt.toml
edition = "2024"
max_width = 100
use_small_heuristics = "Max"
imports_granularity = "Module"
group_imports = "StdExternalCrate"
```

**Step 3: Commit**

```bash
git add .pre-commit-config.yaml rustfmt.toml
git commit -m "chore: add pre-commit hooks and formatting config"
```

---

**Task 0.3.1.2: Настройка бенчмарков**

**Files:**
- Create: `crates/ns-compute/benches/nll_benchmark.rs`
- Modify: `crates/ns-compute/Cargo.toml`
- Modify: `Cargo.toml` (`[workspace.dependencies]`)

**Step 1: Добавить criterion в workspace**

```toml
# Add to Cargo.toml [workspace.dependencies]
criterion = { version = "0.8", features = ["html_reports"] }
statrs = "0.18"
```

**Step 2: Создать benchmark scaffold**

```rust
// crates/ns-compute/benches/nll_benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use statrs::function::gamma::ln_gamma;

#[inline]
fn poisson_nll_var_terms(observed: &[f64], expected: &[f64]) -> f64 {
    observed
        .iter()
        .zip(expected.iter())
        .map(|(&n, &lam)| if lam > 0.0 { lam - n * lam.ln() } else { f64::INFINITY })
        .sum()
}

fn nll_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("nll");

    for n_bins in [10usize, 100, 1000].iter().copied() {
        let observed: Vec<f64> = (0..n_bins).map(|i| (i % 100) as f64).collect();
        let expected: Vec<f64> = (0..n_bins).map(|i| (i % 100 + 1) as f64).collect();
        // Observed-only constants (do not depend on λ): Σ ln Γ(n+1)
        let const_terms: f64 = observed.iter().map(|&n| ln_gamma(n + 1.0)).sum();

        group.bench_with_input(
            BenchmarkId::new("poisson_nll", n_bins),
            &n_bins,
            |b, _| {
                b.iter(|| {
                    // Benchmark the variable part: Σ (λ - n ln λ)
                    let var_terms = poisson_nll_var_terms(&observed, &expected);
                    black_box(const_terms + var_terms)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, nll_benchmark);
criterion_main!(benches);
```

**Step 2.1: Wire Criterion bench**

```toml
# crates/ns-compute/Cargo.toml
[dev-dependencies]
criterion.workspace = true

[[bench]]
name = "nll_benchmark"
harness = false
```

**Step 3: Commit**

```bash
git add crates/ns-compute/benches/ crates/ns-compute/Cargo.toml Cargo.toml
git commit -m "chore: add criterion benchmarks scaffold"
```

---

**Task 0.3.1.3: Валидационный тест-сьют против pyhf**

**Files:**
- Create: `tests/python/conftest.py`
- Create: `tests/python/test_pyhf_validation.py`
- Create: `tests/fixtures/simple_workspace.json`

**Step 1: Создать pyhf fixture**

```json
// tests/fixtures/simple_workspace.json
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

**Step 2: Создать validation test**

```python
# tests/python/conftest.py
import pytest
import json
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures"

@pytest.fixture
def simple_workspace():
    with open(FIXTURES / "simple_workspace.json") as f:
        return json.load(f)

@pytest.fixture
def pyhf_result(simple_workspace):
    """Reference result from pyhf"""
    import pyhf

    workspace = pyhf.Workspace(simple_workspace)
    model = workspace.model()
    data = workspace.data(model)

    bestfit = pyhf.infer.mle.fit(data, model)
    twice_nll = float(pyhf.infer.mle.twice_nll(bestfit, data, model).item())

    return {
        "bestfit": bestfit.tolist(),
        "twice_nll": twice_nll,
    }
```

```python
# tests/python/test_pyhf_validation.py
"""
Validation tests: NextStat results must match pyhf within tolerance.

These tests define the numerical contract NextStat must satisfy.
"""
import pytest
import numpy as np

# Tolerance for numerical comparison
RTOL = 1e-6  # relative tolerance
ATOL = 1e-8  # absolute tolerance


class TestPyhfValidation:
    """Tests that validate NextStat against pyhf reference."""

    def test_simple_fit_bestfit(self, simple_workspace, pyhf_result):
        """MLE best-fit values must match pyhf."""
        import json
        import nextstat

        model = nextstat.from_pyhf(json.dumps(simple_workspace))
        result = nextstat.fit(model)

        np.testing.assert_allclose(
            result.bestfit,
            pyhf_result["bestfit"],
            rtol=RTOL,
            atol=ATOL,
            err_msg="Best-fit values differ from pyhf"
        )

    def test_simple_fit_uncertainties(self, simple_workspace, pyhf_result):
        """MLE uncertainties must match pyhf."""
        pytest.skip("NextStat not yet implemented")

        import nextstat

        model = nextstat.from_pyhf(simple_workspace)
        result = nextstat.fit(model)

        np.testing.assert_allclose(
            result.uncertainties,
            pyhf_result["uncertainties"],
            rtol=1e-5,  # slightly looser for Hessian
            atol=1e-7,
            err_msg="Uncertainties differ from pyhf"
        )

    def test_simple_fit_nll(self, simple_workspace, pyhf_result):
        """NLL at best-fit must match pyhf."""
        pytest.skip("NextStat not yet implemented")

        import nextstat

        model = nextstat.from_pyhf(simple_workspace)
        result = nextstat.fit(model)

        np.testing.assert_allclose(
            result.twice_nll,
            pyhf_result["twice_nll"],
            rtol=RTOL,
            atol=ATOL,
            err_msg="NLL values differ from pyhf"
        )
```

**Step 3: Commit**

```bash
git add tests/
git commit -m "test: add pyhf validation test suite (skipped until implementation)"
```

---

## Фаза I: MVP-α Core Engine (Месяцы 2-4)

### Цели фазы
- Читать pyhf JSON workspace
- Вычислять NLL для HistFactory модели
- Выполнять MLE fit
- Результат совпадает с pyhf до 6 знаков

### Sprint 1.1: HistFactory Model Parser (Недели 5-6)

#### Epic 1.1.1: pyhf JSON Schema Support

**Task 1.1.1.1: Определить типы для pyhf workspace**

**Files:**
- Create: `crates/ns-translate/Cargo.toml`
- Create: `crates/ns-translate/src/lib.rs`
- Create: `crates/ns-translate/src/pyhf/mod.rs`
- Create: `crates/ns-translate/src/pyhf/schema.rs`
- Test: `crates/ns-translate/src/pyhf/tests.rs`

**Step 1: Написать failing test**

```rust
// crates/ns-translate/src/pyhf/tests.rs
#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_WORKSPACE: &str = r#"
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
                {"name": "bkg_uncert", "type": "staterror", "data": [5.0, 6.0]}
              ]
            }
          ]
        }
      ],
      "observations": [{"name": "singlechannel", "data": [55.0, 70.0]}],
      "measurements": [
        {"name": "Measurement", "config": {"poi": "mu", "parameters": []}}
      ],
      "version": "1.0.0"
    }
    "#;

    #[test]
    fn test_parse_simple_workspace() {
        let workspace: Workspace = serde_json::from_str(SIMPLE_WORKSPACE)
            .expect("Failed to parse workspace");

        assert_eq!(workspace.channels.len(), 1);
        assert_eq!(workspace.channels[0].name, "singlechannel");
        assert_eq!(workspace.channels[0].samples.len(), 2);
    }

    #[test]
    fn test_parse_modifiers() {
        let workspace: Workspace = serde_json::from_str(SIMPLE_WORKSPACE).unwrap();

        let signal = &workspace.channels[0].samples[0];
        assert_eq!(signal.modifiers.len(), 1);

        match &signal.modifiers[0] {
            Modifier::NormFactor { name, .. } => {
                assert_eq!(name, "mu");
            }
            _ => panic!("Expected NormFactor modifier"),
        }
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p ns-translate test_parse_simple_workspace
# Expected: FAIL - module not found
```

**Step 3: Implement schema types**

```rust
// crates/ns-translate/src/pyhf/schema.rs
use serde::{Deserialize, Serialize};

/// pyhf JSON Workspace (HistFactory specification)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workspace {
    pub channels: Vec<Channel>,
    pub observations: Vec<Observation>,
    pub measurements: Vec<Measurement>,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Channel {
    pub name: String,
    pub samples: Vec<Sample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    pub name: String,
    pub data: Vec<f64>,
    pub modifiers: Vec<Modifier>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub name: String,
    pub data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    pub name: String,
    pub config: MeasurementConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementConfig {
    pub poi: String,
    pub parameters: Vec<ParameterConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterConfig {
    pub name: String,
    #[serde(default)]
    pub fixed: bool,
    #[serde(default)]
    pub inits: Option<Vec<f64>>,
    #[serde(default)]
    pub bounds: Option<Vec<[f64; 2]>>,
}

/// Modifier types in HistFactory
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Modifier {
    /// Normalization factor (POI or free parameter)
    #[serde(rename = "normfactor")]
    NormFactor {
        name: String,
        data: Option<serde_json::Value>,
    },

    /// Statistical error (Barlow-Beeston)
    #[serde(rename = "staterror")]
    StatError {
        name: String,
        data: Vec<f64>,
    },

    /// Histogram systematic
    #[serde(rename = "histosys")]
    HistoSys {
        name: String,
        data: HistoSysData,
    },

    /// Normalization systematic
    #[serde(rename = "normsys")]
    NormSys {
        name: String,
        data: NormSysData,
    },

    /// Shape systematic
    #[serde(rename = "shapesys")]
    ShapeSys {
        name: String,
        data: Vec<f64>,
    },

    /// Luminosity
    #[serde(rename = "lumi")]
    Lumi {
        name: String,
        data: Option<serde_json::Value>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoSysData {
    pub hi_data: Vec<f64>,
    pub lo_data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormSysData {
    pub hi: f64,
    pub lo: f64,
}
```

**Step 4: Create module structure**

```rust
// crates/ns-translate/src/pyhf/mod.rs
mod schema;
#[cfg(test)]
mod tests;

pub use schema::*;
```

```rust
// crates/ns-translate/src/lib.rs
//! NextStat Translate - format converters
//!
//! Supports:
//! - pyhf JSON workspace
//! - HistFactory XML (planned)
//! - ROOT TH1/TTree via uproot (planned)

pub mod pyhf;

pub use pyhf::Workspace;
```

```toml
# crates/ns-translate/Cargo.toml
[package]
name = "ns-translate"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
ns-core = { path = "../ns-core" }
serde.workspace = true
serde_json.workspace = true

[dev-dependencies]
approx.workspace = true
```

**Step 5: Run test to verify it passes**

```bash
cargo test -p ns-translate
# Expected: PASS
```

**Step 6: Commit**

```bash
git add crates/ns-translate/
git commit -m "feat(translate): add pyhf JSON workspace parser"
```

---

**Task 1.1.1.2: Конвертер pyhf → Internal Model**

**Files:**
- Create: `crates/ns-translate/src/pyhf/convert.rs`
- Modify: `crates/ns-core/src/model.rs`
- Test: inline

**Step 1: Написать failing test**

```rust
// crates/ns-translate/src/pyhf/convert.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_workspace_to_model() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let workspace: Workspace = serde_json::from_str(json).unwrap();

        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        // Should have: mu, 2 staterror gammas
        assert_eq!(model.n_params(), 3);
        assert_eq!(model.poi_index(), Some(0));
    }

    #[test]
    fn test_model_expected_data() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let workspace: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        // At mu=1, gammas=1: expected = signal + background
        let params = vec![1.0, 1.0, 1.0]; // mu, gamma_0, gamma_1
        let expected = model.expected_data(&params);

        // signal=[5,10], background=[50,60] → expected=[55,70]
        assert!((expected[0] - 55.0).abs() < 1e-10);
        assert!((expected[1] - 70.0).abs() < 1e-10);
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p ns-translate test_convert_workspace
# Expected: FAIL - HistFactoryModel not defined
```

**Step 3: Implement HistFactoryModel**

```rust
// crates/ns-translate/src/pyhf/convert.rs
use ns_core::{Error, Result, Parameter, types::Float};
use super::schema::*;

/// HistFactory model converted from pyhf workspace
#[derive(Debug, Clone)]
pub struct HistFactoryModel {
    /// Model parameters
    parameters: Vec<Parameter>,
    /// Index of POI in parameters
    poi_idx: Option<usize>,
    /// Channel data
    channels: Vec<ChannelModel>,
    /// Observed data (concatenated across channels)
    observed: Vec<Float>,
}

#[derive(Debug, Clone)]
struct ChannelModel {
    name: String,
    n_bins: usize,
    samples: Vec<SampleModel>,
}

#[derive(Debug, Clone)]
struct SampleModel {
    name: String,
    /// Nominal histogram
    nominal: Vec<Float>,
    /// Modifiers affecting this sample
    modifiers: Vec<ModifierModel>,
}

#[derive(Debug, Clone)]
enum ModifierModel {
    NormFactor { param_idx: usize },
    StatError { param_indices: Vec<usize>, sigmas: Vec<Float> },
    NormSys { param_idx: usize, hi_factor: Float, lo_factor: Float },
    HistoSys { param_idx: usize, hi_data: Vec<Float>, lo_data: Vec<Float> },
}

impl HistFactoryModel {
    pub fn from_workspace(ws: &Workspace) -> Result<Self> {
        let mut parameters = Vec::new();
        let mut channels = Vec::new();
        let mut observed = Vec::new();
        let mut poi_idx = None;

        // Collect POI and measurement config
        let measurement = ws.measurements.first()
            .ok_or_else(|| Error::ModelSpec("No measurement defined".into()))?;
        let poi_name = &measurement.config.poi;

        // First pass: collect all unique parameter names
        let mut param_map = std::collections::HashMap::new();

        for channel in &ws.channels {
            for sample in &channel.samples {
                for modifier in &sample.modifiers {
                    let name = match modifier {
                        Modifier::NormFactor { name, .. }
                        | Modifier::NormSys { name, .. }
                        | Modifier::HistoSys { name, .. } => name.clone(),
                        Modifier::StatError { name, data } => {
                            // StatError creates one gamma per bin
                            for (i, _) in data.iter().enumerate() {
                                let gamma_name = format!("{}[{}]", name, i);
                                if !param_map.contains_key(&gamma_name) {
                                    let idx = parameters.len();
                                    param_map.insert(gamma_name.clone(), idx);
                                    parameters.push(
                                        Parameter::new(&gamma_name, 1.0).bounded(0.0, 10.0)
                                    );
                                }
                            }
                            continue;
                        }
                        _ => {
                            return Err(Error::ModelSpec(
                                "Unsupported modifier type in Phase 1 converter".into(),
                            ));
                        }
                    };

                    if !param_map.contains_key(&name) {
                        let idx = parameters.len();
                        param_map.insert(name.clone(), idx);

                        let param = if &name == poi_name {
                            poi_idx = Some(idx);
                            Parameter::new(&name, 1.0).bounded(0.0, 10.0)
                        } else {
                            Parameter::new(&name, 1.0).bounded(0.0, 10.0)
                        };
                        parameters.push(param);
                    }
                }
            }
        }

        // Second pass: build channel models
        for channel in &ws.channels {
            let n_bins = channel.samples.first()
                .map(|s| s.data.len())
                .unwrap_or(0);

            let mut samples = Vec::new();

            for sample in &channel.samples {
                let mut modifiers = Vec::new();

                for modifier in &sample.modifiers {
                    match modifier {
                        Modifier::NormFactor { name, .. } => {
                            let idx = *param_map.get(name).unwrap();
                            modifiers.push(ModifierModel::NormFactor { param_idx: idx });
                        }
                        Modifier::StatError { name, data } => {
                            let indices: Vec<_> = (0..data.len())
                                .map(|i| *param_map.get(&format!("{}[{}]", name, i)).unwrap())
                                .collect();
                            modifiers.push(ModifierModel::StatError {
                                param_indices: indices,
                                sigmas: data.clone(),
                            });
                        }
                        Modifier::NormSys { name, data } => {
                            let idx = *param_map.get(name).unwrap();
                            modifiers.push(ModifierModel::NormSys {
                                param_idx: idx,
                                hi_factor: data.hi,
                                lo_factor: data.lo,
                            });
                        }
                        Modifier::HistoSys { name, data } => {
                            let idx = *param_map.get(name).unwrap();
                            modifiers.push(ModifierModel::HistoSys {
                                param_idx: idx,
                                hi_data: data.hi_data.clone(),
                                lo_data: data.lo_data.clone(),
                            });
                        }
                        _ => {
                            return Err(Error::ModelSpec(
                                "Unsupported modifier type in Phase 1 converter".into(),
                            ));
                        }
                    }
                }

                samples.push(SampleModel {
                    name: sample.name.clone(),
                    nominal: sample.data.clone(),
                    modifiers,
                });
            }

            channels.push(ChannelModel {
                name: channel.name.clone(),
                n_bins,
                samples,
            });
        }

        // Collect observations
        for obs in &ws.observations {
            observed.extend(obs.data.iter().cloned());
        }

        Ok(Self {
            parameters,
            poi_idx,
            channels,
            observed,
        })
    }

    pub fn n_params(&self) -> usize {
        self.parameters.len()
    }

    pub fn poi_index(&self) -> Option<usize> {
        self.poi_idx
    }

    /// Compute expected data at given parameter values
    pub fn expected_data(&self, params: &[Float]) -> Vec<Float> {
        let mut expected = Vec::new();

        for channel in &self.channels {
            let mut channel_exp = vec![0.0; channel.n_bins];

            for sample in &channel.samples {
                let mut sample_exp = sample.nominal.clone();

                for modifier in &sample.modifiers {
                    match modifier {
                        ModifierModel::NormFactor { param_idx } => {
                            let factor = params[*param_idx];
                            for v in &mut sample_exp {
                                *v *= factor;
                            }
                        }
                        ModifierModel::StatError { param_indices, .. } => {
                            for (i, &idx) in param_indices.iter().enumerate() {
                                sample_exp[i] *= params[idx];
                            }
                        }
                    }
                }

                for (i, v) in sample_exp.iter().enumerate() {
                    channel_exp[i] += v;
                }
            }

            expected.extend(channel_exp);
        }

        expected
    }

    pub fn observed_data(&self) -> &[Float] {
        &self.observed
    }

    pub fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }
}
```

**Step 4: Run test to verify it passes**

```bash
cargo test -p ns-translate
# Expected: PASS
```

**Step 5: Commit**

```bash
git add crates/ns-translate/
git commit -m "feat(translate): convert pyhf workspace to internal model"
```

---

### Sprint 1.2: Likelihood Computation (Недели 7-8)

#### Epic 1.2.1: NLL Implementation

**Task 1.2.1.1: Poisson + Gaussian constraint NLL**

**Files:**
- Create: `crates/ns-compute/Cargo.toml`
- Create: `crates/ns-compute/src/lib.rs`
- Create: `crates/ns-compute/src/nll.rs`
- Test: inline

**Step 1: Написать failing test**

```rust
// crates/ns-compute/src/nll.rs
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_poisson_nll_simple() {
        // observed=10, expected=10 → NLL = 10 - 10*ln(10) + ln(10!)
        // = 10 - 10*2.302585 + 15.104412 ≈ 2.078562
        let nll = poisson_nll(10.0, 10.0);
        assert_relative_eq!(nll, 2.078562, epsilon = 1e-4);
    }

    #[test]
    fn test_poisson_nll_vectorized() {
        let observed = vec![10.0, 20.0];
        let expected = vec![10.0, 20.0];

        let nll = poisson_nll_vec(&observed, &expected);

        // Sum of individual Poisson NLLs
        let expected_nll = poisson_nll(10.0, 10.0) + poisson_nll(20.0, 20.0);
        assert_relative_eq!(nll, expected_nll, epsilon = 1e-10);
    }

    #[test]
    fn test_gaussian_constraint() {
        // Normal(x | μ, σ): NLL = 0.5*z^2 + ln(σ) + 0.5*ln(2π)
        let base = (0.1f64).ln() + 0.5 * (2.0 * std::f64::consts::PI).ln();

        // gamma=1.0, sigma=0.1 → z=0
        let nll = gaussian_constraint_nll(1.0, 1.0, 0.1);
        assert_relative_eq!(nll, base, epsilon = 1e-12);

        // gamma=1.1 → z=1 → base + 0.5
        let nll = gaussian_constraint_nll(1.1, 1.0, 0.1);
        assert_relative_eq!(nll, base + 0.5, epsilon = 1e-12);
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p ns-compute test_poisson_nll
# Expected: FAIL - not implemented
```

**Step 3: Implement NLL functions**

```rust
// crates/ns-compute/src/nll.rs
//! Negative log-likelihood computations

use ns_core::types::Float;

/// Poisson negative log-likelihood (single bin)
///
/// Computes: `λ - n * ln(λ) + ln Γ(n+1)`
///
/// This matches the canonical definition in `docs/plans/standards.md` and the
/// `pyhf` Poisson `log_prob` (via `lgamma(n+1)`).
#[inline]
pub fn poisson_nll(observed: Float, expected: Float) -> Float {
    if expected <= 0.0 {
        return Float::INFINITY;
    }
    expected - observed * expected.ln() + statrs::function::gamma::ln_gamma(observed + 1.0)
}

/// Poisson NLL for multiple bins
pub fn poisson_nll_vec(observed: &[Float], expected: &[Float]) -> Float {
    observed.iter()
        .zip(expected.iter())
        .map(|(&n, &lam)| poisson_nll(n, lam))
        .sum()
}

/// Gaussian constraint NLL
/// For parameter x with nominal value x0 and uncertainty sigma:
/// NLL = 0.5*z^2 + ln(σ) + 0.5*ln(2π)
#[inline]
pub fn gaussian_constraint_nll(value: Float, nominal: Float, sigma: Float) -> Float {
    if sigma <= 0.0 {
        return Float::INFINITY;
    }
    let z = (value - nominal) / sigma;
    0.5 * z * z + sigma.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// Full NLL with Poisson data term and Gaussian constraints
pub fn full_nll(
    observed: &[Float],
    expected: &[Float],
    constraint_values: &[(Float, Float, Float)], // (value, nominal, sigma)
) -> Float {
    let poisson = poisson_nll_vec(observed, expected);
    let constraints: Float = constraint_values.iter()
        .map(|&(v, nom, sig)| gaussian_constraint_nll(v, nom, sig))
        .sum();
    poisson + constraints
}
```

```rust
// crates/ns-compute/src/lib.rs
//! NextStat Compute - numerical computations

pub mod nll;

pub use nll::{poisson_nll, poisson_nll_vec, gaussian_constraint_nll, full_nll};
```

```toml
# crates/ns-compute/Cargo.toml
[package]
name = "ns-compute"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
ns-core = { path = "../ns-core" }
statrs = "0.18"

[dev-dependencies]
approx.workspace = true
proptest.workspace = true
```

**Step 4: Run test to verify it passes**

```bash
cargo test -p ns-compute
# Expected: PASS
```

**Step 5: Commit**

```bash
git add crates/ns-compute/
git commit -m "feat(compute): implement Poisson and Gaussian constraint NLL"
```

---

**Task 1.2.1.2: HistFactory NLL (полная модель)**

**Files:**
- Modify: `crates/ns-translate/src/pyhf/convert.rs`
- Test: `tests/rust/test_histfactory_nll.rs`

**Step 1: Написать failing test против pyhf reference**

```rust
// tests/rust/test_histfactory_nll.rs
use ns_translate::pyhf::{Workspace, HistFactoryModel};
use ns_compute::full_nll;
use approx::assert_relative_eq;

/// Reference values computed with pyhf
/// ```python
/// import pyhf
/// import json
/// ws = pyhf.Workspace(json.load(open("simple_workspace.json")))
/// model = ws.model()
/// data = ws.data(model)
/// pars = [1.0, 1.0, 1.0]  # mu, gamma_0, gamma_1
/// print(-model.logpdf(pars, data))  # twice_nll / 2
/// ```
const PYHF_REFERENCE_NLL: f64 = 5.434296; // Example - replace with actual pyhf output

#[test]
fn test_histfactory_nll_matches_pyhf() {
    let json = include_str!("../fixtures/simple_workspace.json");
    let workspace: Workspace = serde_json::from_str(json).unwrap();
    let model = HistFactoryModel::from_workspace(&workspace).unwrap();

    let params = vec![1.0, 1.0, 1.0]; // mu, gamma_0, gamma_1
    let nll = model.nll(&params);

    assert_relative_eq!(nll, PYHF_REFERENCE_NLL, epsilon = 1e-5);
}
```

**Step 2: Implement Model trait for HistFactoryModel**

```rust
// Add to crates/ns-translate/src/pyhf/convert.rs

impl ns_core::Model for HistFactoryModel {
    fn nll(&self, params: &[Float]) -> ns_core::Result<Float> {
        let expected = self.expected_data(params);
        let observed = self.observed_data();

        // Poisson term
        let mut nll = ns_compute::poisson_nll_vec(observed, &expected);

        // Gaussian constraints for staterror parameters
        for channel in &self.channels {
            for sample in &channel.samples {
                for modifier in &sample.modifiers {
                    if let ModifierModel::StatError { param_indices, sigmas } = modifier {
                        for (i, (&idx, &sigma)) in param_indices.iter().zip(sigmas.iter()).enumerate() {
                            // Relative uncertainty on the nominal
                            let rel_uncert = sigma / sample.nominal[i];
                            nll += ns_compute::gaussian_constraint_nll(params[idx], 1.0, rel_uncert);
                        }
                    }
                }
            }
        }

        Ok(nll)
    }

    fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }
}
```

**Step 3: Run test and compare with actual pyhf**

```bash
# First, get reference value from pyhf
python3 -c "
import pyhf
import json

ws = pyhf.Workspace(json.load(open('tests/fixtures/simple_workspace.json')))
model = ws.model()
data = ws.data(model)
pars = [1.0] + [1.0] * (model.config.npars - 1)
nll = -model.logpdf(pars, data)[0]
print(f'PYHF_REFERENCE_NLL: {nll}')
"
```

**Step 4: Update test with correct reference and verify**

```bash
cargo test test_histfactory_nll_matches_pyhf
# Expected: PASS (after updating reference value)
```

**Step 5: Commit**

```bash
git add crates/ns-translate/ tests/
git commit -m "feat(translate): implement HistFactory NLL matching pyhf"
```

---

### Sprint 1.3: MLE Optimizer (Недели 9-10)

#### Epic 1.3.1: L-BFGS-B Minimizer

**Task 1.3.1.1: Implement gradient-free wrapper**

**Files:**
- Create: `crates/ns-inference/Cargo.toml`
- Create: `crates/ns-inference/src/lib.rs`
- Create: `crates/ns-inference/src/minimize.rs`
- Test: inline

**Step 1: Написать failing test**

```rust
// crates/ns-inference/src/minimize.rs
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_minimize_rosenbrock() {
        // Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        // Minimum at (1, 1) with f(1,1) = 0
        let rosenbrock = |x: &[f64]| -> f64 {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };

        let result = minimize(
            rosenbrock,
            &[0.0, 0.0],  // initial
            &[(-5.0, 5.0), (-5.0, 5.0)],  // bounds
            MinimizeOptions::default(),
        ).unwrap();

        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-4);
        assert_relative_eq!(result.fun, 0.0, epsilon = 1e-8);
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p ns-inference test_minimize_rosenbrock
# Expected: FAIL - not implemented
```

**Step 3: Implement minimizer (using argmin crate)**

```toml
# crates/ns-inference/Cargo.toml
[package]
name = "ns-inference"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
ns-core = { path = "../ns-core" }
ns-compute = { path = "../ns-compute" }
argmin = "0.11"
argmin-math = { version = "0.5", features = ["ndarray_latest"] }
ndarray.workspace = true

[dev-dependencies]
approx.workspace = true
```

```rust
// crates/ns-inference/src/minimize.rs
//! Numerical minimization

use argmin::core::{CostFunction, Executor, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::{Array1, ArrayView1};
use ns_core::types::Float;

/// Result of minimization
#[derive(Debug, Clone)]
pub struct MinimizeResult {
    /// Optimal parameters
    pub x: Vec<Float>,
    /// Function value at optimum
    pub fun: Float,
    /// Number of function evaluations
    pub nfev: usize,
    /// Number of iterations
    pub nit: usize,
    /// Success flag
    pub success: bool,
    /// Message
    pub message: String,
}

/// Options for minimizer
#[derive(Debug, Clone)]
pub struct MinimizeOptions {
    pub maxiter: usize,
    pub gtol: Float,
    pub ftol: Float,
}

impl Default for MinimizeOptions {
    fn default() -> Self {
        Self {
            maxiter: 1000,
            gtol: 1e-8,
            ftol: 1e-12,
        }
    }
}

struct ObjectiveFunction<F> {
    func: F,
    bounds: Vec<(Float, Float)>,
    nfev: std::cell::Cell<usize>,
}

impl<F> CostFunction for ObjectiveFunction<F>
where
    F: Fn(&[Float]) -> Float,
{
    type Param = Array1<Float>;
    type Output = Float;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        self.nfev.set(self.nfev.get() + 1);

        // Project parameters into bounds
        let bounded: Vec<Float> = p.iter()
            .zip(self.bounds.iter())
            .map(|(&x, &(lo, hi))| x.clamp(lo, hi))
            .collect();

        Ok((self.func)(&bounded))
    }
}

/// Minimize a function using L-BFGS with numerical gradients
pub fn minimize<F>(
    func: F,
    x0: &[Float],
    bounds: &[(Float, Float)],
    options: MinimizeOptions,
) -> ns_core::Result<MinimizeResult>
where
    F: Fn(&[Float]) -> Float,
{
    let problem = ObjectiveFunction {
        func,
        bounds: bounds.to_vec(),
        nfev: std::cell::Cell::new(0),
    };

    let init_param = Array1::from_vec(x0.to_vec());

    // Use finite differences for gradient
    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 7)
        .with_tolerance_grad(options.gtol)?
        .with_tolerance_cost(options.ftol)?;

    // Phase 2B: switch to analytical gradients via `Model::gradient()` (AD).

    let result = Executor::new(problem, solver)
        .configure(|state| state.param(init_param).max_iters(options.maxiter as u64))
        .run();

    match result {
        Ok(res) => {
            let state = res.state();
            Ok(MinimizeResult {
                x: state.best_param.as_ref()
                    .map(|p| p.to_vec())
                    .unwrap_or_else(|| x0.to_vec()),
                fun: state.best_cost,
                nfev: state.func_counts.cost as usize,
                nit: state.iter as usize,
                success: state.terminated(),
                message: format!("{:?}", state.termination_reason),
            })
        }
        Err(e) => Err(ns_core::Error::Numerical(e.to_string())),
    }
}
```

**Step 4: Alternative simpler implementation (Nelder-Mead for initial version)**

```rust
// crates/ns-inference/src/minimize.rs (alternative)
//! Numerical minimization - Simple Nelder-Mead implementation

use ns_core::types::Float;

/// Minimize using Nelder-Mead simplex method
pub fn minimize<F>(
    func: F,
    x0: &[Float],
    bounds: &[(Float, Float)],
    options: MinimizeOptions,
) -> ns_core::Result<MinimizeResult>
where
    F: Fn(&[Float]) -> Float,
{
    let n = x0.len();
    let mut nfev = 0;

    // Create initial simplex
    let mut simplex: Vec<Vec<Float>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());

    for i in 0..n {
        let mut vertex = x0.to_vec();
        vertex[i] += 0.05 * (bounds[i].1 - bounds[i].0).abs().max(0.00025);
        simplex.push(vertex);
    }

    // Evaluate all vertices
    let bounded_eval = |x: &[Float]| -> Float {
        let bounded: Vec<Float> = x.iter()
            .zip(bounds.iter())
            .map(|(&v, &(lo, hi))| v.clamp(lo, hi))
            .collect();
        func(&bounded)
    };

    let mut values: Vec<Float> = simplex.iter()
        .map(|v| {
            nfev += 1;
            bounded_eval(v)
        })
        .collect();

    // Nelder-Mead parameters
    let alpha = 1.0;  // reflection
    let gamma = 2.0;  // expansion
    let rho = 0.5;    // contraction
    let sigma = 0.5;  // shrink

    for iter in 0..options.maxiter {
        // Sort by function values
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

        let best_idx = indices[0];
        let worst_idx = indices[n];
        let second_worst_idx = indices[n - 1];

        // Check convergence
        let spread = values[worst_idx] - values[best_idx];
        if spread < options.ftol {
            return Ok(MinimizeResult {
                x: simplex[best_idx].clone(),
                fun: values[best_idx],
                nfev,
                nit: iter,
                success: true,
                message: "Converged".to_string(),
            });
        }

        // Compute centroid (excluding worst)
        let mut centroid = vec![0.0; n];
        for &i in &indices[..n] {
            for j in 0..n {
                centroid[j] += simplex[i][j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as Float;
        }

        // Reflection
        let reflected: Vec<Float> = centroid.iter()
            .zip(simplex[worst_idx].iter())
            .map(|(&c, &w)| c + alpha * (c - w))
            .collect();
        nfev += 1;
        let f_reflected = bounded_eval(&reflected);

        if f_reflected < values[second_worst_idx] && f_reflected >= values[best_idx] {
            simplex[worst_idx] = reflected;
            values[worst_idx] = f_reflected;
            continue;
        }

        // Expansion
        if f_reflected < values[best_idx] {
            let expanded: Vec<Float> = centroid.iter()
                .zip(reflected.iter())
                .map(|(&c, &r)| c + gamma * (r - c))
                .collect();
            nfev += 1;
            let f_expanded = bounded_eval(&expanded);

            if f_expanded < f_reflected {
                simplex[worst_idx] = expanded;
                values[worst_idx] = f_expanded;
            } else {
                simplex[worst_idx] = reflected;
                values[worst_idx] = f_reflected;
            }
            continue;
        }

        // Contraction
        let contracted: Vec<Float> = centroid.iter()
            .zip(simplex[worst_idx].iter())
            .map(|(&c, &w)| c + rho * (w - c))
            .collect();
        nfev += 1;
        let f_contracted = bounded_eval(&contracted);

        if f_contracted < values[worst_idx] {
            simplex[worst_idx] = contracted;
            values[worst_idx] = f_contracted;
            continue;
        }

        // Shrink
        for i in 1..=n {
            for j in 0..n {
                simplex[i][j] = simplex[best_idx][j] + sigma * (simplex[i][j] - simplex[best_idx][j]);
            }
            nfev += 1;
            values[i] = bounded_eval(&simplex[i]);
        }
    }

    // Find best after max iterations
    let best_idx = values.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    Ok(MinimizeResult {
        x: simplex[best_idx].clone(),
        fun: values[best_idx],
        nfev,
        nit: options.maxiter,
        success: false,
        message: "Max iterations reached".to_string(),
    })
}
```

**Step 5: Run test to verify it passes**

```bash
cargo test -p ns-inference test_minimize_rosenbrock
# Expected: PASS
```

**Step 6: Commit**

```bash
git add crates/ns-inference/
git commit -m "feat(inference): implement Nelder-Mead minimizer"
```

---

**Task 1.3.1.2: MLE fit function**

**Files:**
- Create: `crates/ns-inference/src/mle.rs`
- Test: inline

**Step 1: Написать failing test**

```rust
// crates/ns-inference/src/mle.rs
#[cfg(test)]
mod tests {
    use super::*;
    use ns_translate::pyhf::{Workspace, HistFactoryModel};
    use approx::assert_relative_eq;

    #[test]
    fn test_mle_fit_simple_workspace() {
        let json = include_str!("../../../tests/fixtures/simple_workspace.json");
        let workspace: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let result = mle_fit(&model, FitOptions::default()).unwrap();

        // Compare with pyhf reference
        // pyhf.infer.mle.fit returns: [mu, gamma_0, gamma_1]
        // Reference values from pyhf (update with actual values)
        assert!(result.success);
        assert_relative_eq!(result.bestfit[0], 1.0, epsilon = 0.1); // mu ~ 1
    }
}
```

**Step 2: Implement MLE fit**

```rust
// crates/ns-inference/src/mle.rs
use ns_core::{Model, Result, types::Float};
use crate::minimize::{minimize, MinimizeOptions, MinimizeResult};

/// Options for MLE fit
#[derive(Debug, Clone)]
pub struct FitOptions {
    pub minimize_options: MinimizeOptions,
    pub compute_uncertainties: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            minimize_options: MinimizeOptions::default(),
            compute_uncertainties: true,
        }
    }
}

/// Result of MLE fit
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Best-fit parameter values
    pub bestfit: Vec<Float>,
    /// Parameter uncertainties (from Hessian)
    pub uncertainties: Vec<Float>,
    /// Twice negative log-likelihood at minimum
    pub twice_nll: Float,
    /// Correlation matrix
    pub correlation: Option<Vec<Vec<Float>>>,
    /// Minimization details
    pub minimize_result: MinimizeResult,
    /// Success flag
    pub success: bool,
}

/// Perform Maximum Likelihood Estimation
pub fn mle_fit<M: Model>(model: &M, options: FitOptions) -> Result<FitResult> {
    let params = model.parameters();
    let n = params.len();

    // Initial values
    let x0: Vec<Float> = params.iter().map(|p| p.init).collect();

    // Bounds
    let bounds: Vec<(Float, Float)> = params.iter()
        .map(|p| {
            let lo = p.lower.unwrap_or(-10.0);
            let hi = p.upper.unwrap_or(10.0);
            (lo, hi)
        })
        .collect();

    // Minimize NLL
    let obj = |x: &[Float]| -> Float {
        model.nll(x).unwrap_or(Float::INFINITY)
    };

    let min_result = minimize(obj, &x0, &bounds, options.minimize_options)?;

    // Compute uncertainties via numerical Hessian
    let uncertainties = if options.compute_uncertainties {
        compute_uncertainties(model, &min_result.x)?
    } else {
        vec![Float::NAN; n]
    };

    Ok(FitResult {
        bestfit: min_result.x.clone(),
        uncertainties,
        twice_nll: 2.0 * min_result.fun,
        // Phase 3 adds correlation matrix extraction/plots.
        correlation: None,
        minimize_result: min_result.clone(),
        success: min_result.success,
    })
}

/// Compute uncertainties from numerical Hessian
fn compute_uncertainties<M: Model>(model: &M, x: &[Float]) -> Result<Vec<Float>> {
    let n = x.len();
    let h = 1e-5; // step size for finite differences

    let f0 = model.nll(x)?;
    let mut hessian = vec![vec![0.0; n]; n];

    // Compute diagonal elements
    for i in 0..n {
        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        x_plus[i] += h;
        x_minus[i] -= h;

        let f_plus = model.nll(&x_plus)?;
        let f_minus = model.nll(&x_minus)?;

        hessian[i][i] = (f_plus - 2.0 * f0 + f_minus) / (h * h);
    }

    // Uncertainties are sqrt of diagonal of inverse Hessian
    // For now, just use diagonal approximation
    let uncertainties: Vec<Float> = hessian.iter()
        .map(|row| {
            let diag = row.iter().enumerate().find(|(i, _)| true).map(|(i, &v)| v).unwrap_or(1.0);
            if diag > 0.0 {
                (1.0 / diag).sqrt()
            } else {
                Float::NAN
            }
        })
        .collect();

    Ok(uncertainties)
}
```

**Step 3: Run test**

```bash
cargo test -p ns-inference test_mle_fit
# Expected: PASS (with appropriate tolerance)
```

**Step 4: Commit**

```bash
git add crates/ns-inference/
git commit -m "feat(inference): implement MLE fit with uncertainties"
```

---

### Sprint 1.4: CLI и Python API (Недели 11-12)

#### Epic 1.4.1: Command Line Interface

**Task 1.4.1.1: Базовая CLI**

**Files:**
- Create: `crates/ns-cli/Cargo.toml`
- Create: `crates/ns-cli/src/main.rs`
- Test: integration test

**Step 1: Написать expected CLI interface**

```bash
# Target CLI usage:
nextstat fit --input workspace.json --output results.json
nextstat validate --input workspace.json --reference pyhf
nextstat bench --input workspace.json --iterations 100
```

**Step 2: Implement CLI**

```toml
# crates/ns-cli/Cargo.toml
[package]
name = "ns-cli"
version.workspace = true
edition.workspace = true
license.workspace = true

[[bin]]
name = "nextstat"
path = "src/main.rs"

[dependencies]
ns-core = { path = "../ns-core" }
ns-translate = { path = "../ns-translate" }
ns-inference = { path = "../ns-inference" }
clap = { version = "4.5", features = ["derive"] }
serde_json.workspace = true
anyhow.workspace = true
```

```rust
// crates/ns-cli/src/main.rs
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use anyhow::Result;

#[derive(Parser)]
#[command(name = "nextstat")]
#[command(about = "High-performance statistical fitting framework")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Perform MLE fit
    Fit {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for results
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Print verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Validate against reference implementation
    Validate {
        /// Input workspace
        #[arg(short, long)]
        input: PathBuf,

        /// Reference implementation (pyhf)
        #[arg(short, long, default_value = "pyhf")]
        reference: String,
    },

    /// Run benchmarks
    Bench {
        /// Input workspace
        #[arg(short, long)]
        input: PathBuf,

        /// Number of iterations
        #[arg(short = 'n', long, default_value = "100")]
        iterations: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Fit { input, output, verbose } => {
            cmd_fit(&input, output.as_deref(), verbose)
        }
        Commands::Validate { input, reference } => {
            cmd_validate(&input, &reference)
        }
        Commands::Bench { input, iterations } => {
            cmd_bench(&input, iterations)
        }
    }
}

fn cmd_fit(input: &PathBuf, output: Option<&PathBuf>, verbose: bool) -> Result<()> {
    use ns_translate::pyhf::{Workspace, HistFactoryModel};
    use ns_inference::mle::{mle_fit, FitOptions};
    use ns_core::Model;

    // Read workspace
    let json = std::fs::read_to_string(input)?;
    let workspace: Workspace = serde_json::from_str(&json)?;
    let model = HistFactoryModel::from_workspace(&workspace)?;

    if verbose {
        eprintln!("Model has {} parameters", model.n_params());
    }

    // Fit
    let result = mle_fit(&model, FitOptions::default())?;

    // Output
    let output_json = serde_json::json!({
        "bestfit": result.bestfit,
        "uncertainties": result.uncertainties,
        "twice_nll": result.twice_nll,
        "success": result.success,
        "nfev": result.minimize_result.nfev,
        "nit": result.minimize_result.nit,
    });

    if let Some(path) = output {
        std::fs::write(path, serde_json::to_string_pretty(&output_json)?)?;
        eprintln!("Results written to {:?}", path);
    } else {
        println!("{}", serde_json::to_string_pretty(&output_json)?);
    }

    Ok(())
}

fn cmd_validate(input: &PathBuf, reference: &str) -> Result<()> {
    // Phase 1 validation is implemented via the Python parity suite (pytest + pyhf).
    // A standalone `nextstat validate` command can be added later to avoid bundling pyhf runtime.
    eprintln!(
        "Validation (reference={}): run `pytest tests/python/test_pyhf_validation.py` for parity checks.",
        reference
    );
    Ok(())
}

fn cmd_bench(input: &PathBuf, iterations: usize) -> Result<()> {
    use ns_translate::pyhf::{Workspace, HistFactoryModel};
    use ns_inference::mle::{mle_fit, FitOptions};
    use std::time::Instant;

    let json = std::fs::read_to_string(input)?;
    let workspace: Workspace = serde_json::from_str(&json)?;
    let model = HistFactoryModel::from_workspace(&workspace)?;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = mle_fit(&model, FitOptions::default())?;
    }
    let elapsed = start.elapsed();

    println!("Iterations: {}", iterations);
    println!("Total time: {:?}", elapsed);
    println!("Per fit: {:?}", elapsed / iterations as u32);

    Ok(())
}
```

**Step 3: Build and test**

```bash
cargo build -p ns-cli
./target/debug/nextstat fit --input tests/fixtures/simple_workspace.json
```

**Step 4: Commit**

```bash
git add crates/ns-cli/
git commit -m "feat(cli): add nextstat CLI with fit/validate/bench commands"
```

---

#### Epic 1.4.2: Python Bindings

**Task 1.4.2.1: PyO3 bindings**

**Files:**
- Create: `bindings/ns-py/Cargo.toml`
- Create: `bindings/ns-py/src/lib.rs`
- Create: `bindings/ns-py/python/nextstat/__init__.py`
- Test: `tests/python/test_nextstat.py`

**Step 1: Implement Python module**

```toml
# bindings/ns-py/Cargo.toml
[package]
name = "ns-py"
version.workspace = true
edition.workspace = true
license.workspace = true

[lib]
name = "_core"
crate-type = ["cdylib"]

[dependencies]
ns-core = { path = "../../crates/ns-core" }
ns-translate = { path = "../../crates/ns-translate" }
ns-inference = { path = "../../crates/ns-inference" }
pyo3.workspace = true
serde_json.workspace = true
numpy.workspace = true
```

```rust
// bindings/ns-py/src/lib.rs
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1};

/// Fit result returned to Python
#[pyclass]
#[derive(Clone)]
struct PyFitResult {
    #[pyo3(get)]
    bestfit: Vec<f64>,
    #[pyo3(get)]
    uncertainties: Vec<f64>,
    #[pyo3(get)]
    twice_nll: f64,
    #[pyo3(get)]
    success: bool,
}

/// Load a pyhf workspace and return a model handle
#[pyfunction]
fn from_pyhf(workspace_json: &str) -> PyResult<PyModel> {
    let workspace: ns_translate::pyhf::Workspace = serde_json::from_str(workspace_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

    let model = ns_translate::pyhf::HistFactoryModel::from_workspace(&workspace)
        .map_err(|e| PyValueError::new_err(format!("Invalid workspace: {}", e)))?;

    Ok(PyModel { inner: model })
}

/// Model wrapper for Python
#[pyclass]
struct PyModel {
    inner: ns_translate::pyhf::HistFactoryModel,
}

#[pymethods]
impl PyModel {
    /// Number of parameters
    fn n_params(&self) -> usize {
        use ns_core::Model;
        self.inner.n_params()
    }

    /// Compute NLL at given parameters
    fn nll(&self, params: PyReadonlyArray1<f64>) -> PyResult<f64> {
        use ns_core::Model;
        let params = params.as_slice()?;
        self.inner.nll(params)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))
    }
}

/// Perform MLE fit
#[pyfunction]
fn fit(model: &PyModel) -> PyResult<PyFitResult> {
    use ns_inference::mle::{mle_fit, FitOptions};

    let result = mle_fit(&model.inner, FitOptions::default())
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

    Ok(PyFitResult {
        bestfit: result.bestfit,
        uncertainties: result.uncertainties,
        twice_nll: result.twice_nll,
        success: result.success,
    })
}

/// NextStat Python module
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(from_pyhf, m)?)?;
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_class::<PyModel>()?;
    m.add_class::<PyFitResult>()?;
    Ok(())
}
```

```python
# bindings/ns-py/python/nextstat/__init__.py
"""NextStat - High-performance statistical fitting framework"""

from nextstat._core import from_pyhf, fit, PyModel, PyFitResult
import json
from pathlib import Path
from typing import Union

__version__ = "0.1.0"
__all__ = ["from_pyhf", "fit", "load_workspace"]


def load_workspace(path: Union[str, Path]) -> PyModel:
    """Load a pyhf workspace from file."""
    with open(path) as f:
        return from_pyhf(f.read())
```

**Step 2: Test Python bindings**

```python
# tests/python/test_nextstat.py
import pytest
import json
import numpy as np
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures"


class TestNextStatPython:
    def test_load_workspace(self):
        import nextstat

        model = nextstat.load_workspace(FIXTURES / "simple_workspace.json")
        assert model.n_params() == 3  # mu + 2 gammas

    def test_compute_nll(self):
        import nextstat

        model = nextstat.load_workspace(FIXTURES / "simple_workspace.json")
        params = np.array([1.0, 1.0, 1.0])
        nll = model.nll(params)

        assert isinstance(nll, float)
        assert nll > 0

    def test_fit(self):
        import nextstat

        model = nextstat.load_workspace(FIXTURES / "simple_workspace.json")
        result = nextstat.fit(model)

        assert result.success
        assert len(result.bestfit) == 3
        assert len(result.uncertainties) == 3
```

**Step 3: Build and test**

```bash
cd bindings/ns-py
maturin develop --release
pytest tests/python/test_nextstat.py
```

**Step 4: Commit**

```bash
git add bindings/ns-py/ tests/python/
git commit -m "feat(python): add PyO3 bindings for nextstat"
```

---

## Остальные фазы (краткое описание)

Ниже — high-level roadmap оставшихся фаз. Детальные подпланы для Phase 2A/2B/2C вынесены в отдельные документы (source of truth — они).

---

## Фаза II: MVP-β Performance (CPU + AD + optional GPU) (Месяцы 4-9)

Эта фаза разбита на подпланы (source of truth — отдельные документы):

- **Phase 2A (P0): CPU Parallelism** → `docs/plans/phase-2a-cpu-parallelism.md`
- **Phase 2B (P0): Autodiff & Optimizers** → `docs/plans/phase-2b-autodiff.md`
- **Phase 2C (P1, optional): GPU Backends (Metal/CUDA)** → `docs/plans/phase-2c-gpu-backends.md`

### Sprint 2B.*: Automatic Differentiation (Недели 13-20)
- **Epic 2B.1:** Forward-mode AD (минимум для градиентов малых моделей)
- **Epic 2B.2:** Reverse-mode AD (основной путь: градиенты + Jacobians)
- **Epic 2B.3:** Hessian / LM / uncertainty extraction
- **Epic 2B.4:** Gradient-based optimizer (L-BFGS(-B), trust-region)
- **Optional:** Интеграция/сверка с JAX backend (golden reference)

### Sprint 2A.*: CPU Parallelism (Недели 13-20)
- **Epic 2A.1:** Rayon work-stealing (batched NLL/expected)
- **Epic 2A.2:** SIMD hot loops (std::simd)
- **Epic 2A.3:** Memory-efficient batching (streaming JSONL, bounded RAM)
- **Epic 2A.4:** Cluster job arrays (SLURM/HTCondor), reproducible RNG

### Sprint 2C.*: GPU Acceleration (Недели 21-28, optional)
- **Epic 2C.1:** Backend abstraction + CPU parity as baseline
- **Epic 2C.2:** Metal backend (macOS)
- **Epic 2C.3:** CUDA backend (Linux/NVIDIA)
- **Epic 2C.4:** Batched operations (toys/scan), mixed precision policy

### Sprint 2D.*: Preprocessing Pipeline (Недели 29-36)
- **Epic 2.5.1:** Smoothing систематик
- **Epic 2.5.2:** Pruning систематик
- **Epic 2.5.3:** Symmetrisation

### Sprint 2E.*: Advanced Inference (Недели 37-44)
- **Epic 2.7.1:** Profile likelihood scans
- **Epic 2.7.2:** Ranking plots (NP impact)
- **Epic 2.8.1:** CLs limits
- **Epic 2.8.2:** Significance calculation

---

## Фаза III: Production Ready (Месяцы 9-15)

> Детальный подплан (source of truth): `docs/plans/phase-3-production.md`.

### Sprint 3.1: Bayesian interface (P1, optional)
- **Epic 3.1.1:** Posterior contract + transforms (unconstrained z, log|J|; no double-count constraints)
- **Epic 3.1.2:** HMC kernel (leapfrog + accept/reject)
- **Epic 3.1.3:** NUTS + warmup/adaptation (dual averaging + diag mass matrix)
- **Epic 3.1.4:** Diagnostics (divergences, treedepth, R-hat, ESS) + multi-chain runner

### Sprint 3.2: Production hardening
- **Epic 3.2.1:** Release pipeline + observability

### Sprint 3.3-3.4: Visualization
- **Epic 3.3.1:** Pull plots
- **Epic 3.3.2:** Correlation matrix plots
- **Epic 3.3.3:** Interactive plots (Plotly)
- **Epic 3.4.1:** ATLAS/CMS style templates

### Sprint 3.5-3.6: Documentation & Validation
- **Epic 3.5.1:** API documentation (rustdoc + sphinx)
- **Epic 3.5.2:** User guides
- **Epic 3.6.1:** Comprehensive validation suite
- **Epic 3.6.2:** White Paper (arXiv)

---

## Фаза IV: Enterprise и SaaS (Месяцы 15-24)

> Детальный подплан (source of truth): `docs/plans/phase-4-enterprise.md`.

### Sprint 4.1-4.2: NS-Audit (21 CFR Part 11)
- **Epic 4.1.1:** Audit trail infrastructure
- **Epic 4.1.2:** Electronic signatures
- **Epic 4.2.1:** IQ/OQ/PQ validation framework

### Sprint 4.3-4.4: NS-Scale
- **Epic 4.3.1:** Ray integration для distributed fits
- **Epic 4.3.2:** Kubernetes operator
- **Epic 4.4.1:** Cloud CLI (nextstat cloud submit)

### Sprint 4.5-4.6: NS-Hub
- **Epic 4.5.1:** Model registry
- **Epic 4.5.2:** Versioning и collaboration
- **Epic 4.6.1:** Dashboard UI

---

## Приложения

### A. Глоссарий

| Термин | Определение |
|--------|-------------|
| NLL | Negative Log-Likelihood |
| POI | Parameter of Interest |
| NP | Nuisance Parameter |
| AD | Automatic Differentiation |
| MLE | Maximum Likelihood Estimation |
| CLs | Modified frequentist confidence level |
| Asimov | Asimov dataset (expected data) |
| HistFactory | Statistical model specification |

### B. Референсные команды

```bash
# Сборка
cargo build --release
maturin develop --release

# Тесты
cargo test --all-features
pytest -v

# Бенчмарки
cargo bench

# Линтинг
cargo clippy --all-targets
ruff check .

# Валидация против pyhf
python -c "
import pyhf
import nextstat
import json

ws = pyhf.Workspace(json.load(open('workspace.json')))
pyhf_model = ws.model()
pyhf_data = ws.data(pyhf_model)
pyhf_result = pyhf.infer.mle.fit(pyhf_data, pyhf_model)

ns_model = nextstat.load_workspace('workspace.json')
ns_result = nextstat.fit(ns_model)

import numpy as np
np.testing.assert_allclose(ns_result.bestfit, pyhf_result, rtol=1e-5)
print('VALIDATION PASSED')
"
```

### C. Контакты и ресурсы

- **GitHub:** https://github.com/nextstat/nextstat
- **Documentation:** https://docs.nextstat.io
- **pyhf reference:** https://pyhf.readthedocs.io
- **TRExFitter:** https://trexfitter-docs.web.cern.ch

---

*План создан: 2026-02-05*
*Версия: 1.0.0*
