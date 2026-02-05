# NextStat

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.93%2B-orange.svg)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)

**NextStat** — высокопроизводительный статистический фреймворк для анализа данных в физике высоких энергий (HEP), разработанный на Rust с Python bindings.

## Возможности

- 🚀 **Высокая производительность** — оптимизированные вычисления на CPU (Rayon, SIMD) и опционально GPU (Metal, CUDA)
- 🔬 **pyhf совместимость** — полная поддержка формата pyhf JSON
- 🎯 **Численная точность (Phase 1 fixtures)** — NLL parity vs pyhf до `~1e-8`, MLE bestfit/uncertainties в пределах допусков из `docs/plans/standards.md`
- 🏗️ **Чистая архитектура** — trait-based backend abstraction (CPU/Metal/CUDA)
- 📊 **Статистические методы** — MLE, MCMC (NUTS), profile likelihood
- 🐍 **Python интеграция** — нативные bindings через PyO3

## Быстрый старт

### Установка

#### Из crates.io (Rust)

```bash
cargo add ns-core ns-inference ns-compute
```

#### Из PyPI (Python)

```bash
pip install nextstat
```

#### Сборка из исходников

```bash
# Клонировать репозиторий
git clone https://github.com/nextstat/nextstat.git
cd nextstat

# Собрать Rust workspace
cargo build --release

# Собрать Python bindings
cd bindings/ns-py
maturin develop --release
```

### Использование

#### Rust API

```rust
use ns_inference::mle::MaximumLikelihoodEstimator;
use ns_translate::pyhf::{HistFactoryModel, Workspace};

let json = std::fs::read_to_string("workspace.json")?;
let workspace: Workspace = serde_json::from_str(&json)?;
let model = HistFactoryModel::from_workspace(&workspace)?;

let mle = MaximumLikelihoodEstimator::new();
let result = mle.fit(&model)?;

println!("Best-fit params: {:?}", result.parameters);
println!("NLL at minimum: {}", result.nll);
```

#### Python API

```python
import json

import nextstat

workspace = json.loads(open("workspace.json").read())
model = nextstat.from_pyhf(json.dumps(workspace))
result = nextstat.fit(model)

poi_idx = model.poi_index()
print("POI index:", poi_idx)
print("Best-fit POI:", result.bestfit[poi_idx])
print("Uncertainty:", result.uncertainties[poi_idx])
```

#### CLI

```bash
# Fit a model
nextstat fit --input workspace.json

# Version info
nextstat version
```

## Архитектура

NextStat построен на принципах чистой архитектуры с инверсией зависимостей:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ВЫСОКОУРОВНЕВАЯ ЛОГИКА                       │
│  ns-inference (MLE, NUTS, Profile Likelihood)                   │
│  - Использует trait ComputeBackend                              │
│  - НЕ зависит от CPU/GPU/Metal/CUDA                             │
└─────────────────────────┬───────────────────────────────────────┘
                          │ зависит от абстракции
┌─────────────────────────┴───────────────────────────────────────┐
│             trait ComputeBackend (ns-core)                      │
│  - nll(&self, params) -> f64                                    │
│  - gradient(&self, params) -> Vec<f64>                          │
│  - hessian(&self, params) -> Vec<Vec<f64>>                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │ реализован в
┌─────────────────────────┴───────────────────────────────────────┐
│                НИЗКОУРОВНЕВЫЕ РЕАЛИЗАЦИИ                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │ CpuBackend│  │MetalBackend│ │CudaBackend│                     │
│  │ (Rayon)  │  │ (feature) │  │ (feature)│                      │
│  │ P0       │  │ P1        │  │ P1       │                      │
│  └──────────┘  └──────────┘  └──────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

**Приоритеты:**
1. **CPU Parallelism** (P0) — Rayon, SIMD, cluster support — работает везде
2. **GPU Acceleration** (P1) — Metal/CUDA — опциональные ускорители

## Структура проекта

```
nextstat/
├── crates/              # Rust crates (OSS)
│   ├── ns-core/        # Core types, traits, error handling
│   ├── ns-compute/     # Compute backends (CPU/Metal/CUDA)
│   ├── ns-ad/          # Autodiff + optimizers (Phase 2B)
│   ├── ns-inference/   # Statistical inference (MLE, NUTS, etc.)
│   ├── ns-translate/   # Format translators (pyhf, ROOT, XML)
│   ├── ns-viz/         # Visualization utilities
│   └── ns-cli/         # Command-line interface
├── bindings/
│   └── ns-py/          # Python bindings (PyO3 + maturin)
├── docs/               # Documentation
└── tests/              # Integration tests
```

## Разработка

### Требования

- Rust 1.93+ (edition 2024)
- Python 3.11+ (для bindings)
- maturin (для Python bindings)

### Сборка и тестирование

```bash
# Собрать все crates
cargo build --workspace

# Запустить тесты (включая feature-gated backends)
cargo test --workspace --all-features

# Запустить opt-in "медленные" Rust тесты (toy fits и т.п.)
cargo test -p ns-inference -- --ignored

# Проверить форматирование
cargo fmt --check

# Запустить clippy
cargo clippy --workspace -- -D warnings

# Собрать документацию
cargo doc --workspace --no-deps --open
```

### Python тесты

```bash
# В CI wheel собирается через maturin и ставится в venv.
# Локально удобнее всего повторить это через:
cd bindings/ns-py
maturin develop --release
cd ../..

# Быстрые Python тесты (parity + API contracts)
pytest -q -m "not slow" tests/python

# Медленные toy regression тесты (опционально)
NS_RUN_SLOW=1 NS_TOYS=200 NS_SEED=0 pytest -q -m slow tests/python
```

### Бенчмарки

```bash
cargo bench --workspace

# Основные таргеты:
# - ns-translate: parse/build + model eval/scaling
cargo bench -p ns-translate --bench model_benchmark
# - ns-translate: NLL/expected_data на fixtures
cargo bench -p ns-translate --bench nll_benchmark
# - ns-compute: SIMD vs scalar + векторные kernels
cargo bench -p ns-compute --bench simd_benchmark
# - ns-inference: MLE + gradients + toys
cargo bench -p ns-inference --bench mle_benchmark
# - ns-inference: hypotest / upper limit
cargo bench -p ns-inference --bench hypotest_benchmark
# - ns-ad: tape + dual forward-mode
cargo bench -p ns-ad --bench ad_benchmark
# - ns-core: FitResult (correlation) kernels
cargo bench -p ns-core --bench core_benchmark
```

## Статус разработки

**Текущая фаза:** Phase 1 — MVP-α Core Engine ✅

- [x] pyhf JSON парсер + модификаторы (fixtures parity)
- [x] NLL (Poisson + constraints) + Barlow-Beeston auxiliary data
- [x] MLE (L-BFGS-B) + uncertainties (Hessian)
- [x] CLI (`nextstat fit`) + Python bindings (PyO3/maturin)
- [x] Parity suite vs pyhf: `pytest -m "not slow" tests/python`

**Следующие фазы:**
- Phase 1: MVP-α Core Engine (Q2 2026)
- Phase 2: CPU Parallelism + Autodiff (Q3 2026)
- Phase 3: Production Ready (Q1-Q2 2027)
- Phase 4: Enterprise & SaaS (Q3-Q4 2027)

Подробности в [docs/plans/README.md](docs/plans/README.md)

## Вклад в проект

Мы приветствуем contributions! Пожалуйста, прочитайте [CONTRIBUTING.md](CONTRIBUTING.md) для получения инструкций.

**Важно:** Все коммиты должны быть подписаны DCO (Developer Certificate of Origin). Используйте `git commit -s` для автоматической подписи.

## Лицензия

NextStat использует **dual licensing model**:

- **Open Source:** [AGPL-3.0](LICENSE) для некоммерческого использования и open source проектов
- **Commercial:** Проприетарная лицензия для коммерческих организаций, не желающих соблюдать условия AGPL

Для получения коммерческой лицензии обратитесь по адресу: licensing@nextstat.io

Подробности в [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL)

## Связь

- **Документация:** https://docs.nextstat.io
- **GitHub:** https://github.com/nextstat/nextstat
- **Сайт:** https://nextstat.io
- **Email:** info@nextstat.io

## Благодарности

NextStat вдохновлен следующими проектами:

- [pyhf](https://github.com/scikit-hep/pyhf) — референсная реализация HistFactory в Python
- [TRExFitter](https://gitlab.cern.ch/oarakji/TRExFitter) — ATLAS фиттер на C++
- [RooFit](https://root.cern/manual/roofit/) — ROOT фреймворк

---

*NextStat — быстрее, чище, лучше.*
