# NextStat Implementation Plans

> Детальный план разработки NextStat — высокопроизводительного статистического фреймворка.

## Ключевой принцип архитектуры

```
┌─────────────────────────────────────────────────────────────────┐
│                    ЧИСТАЯ АРХИТЕКТУРА                          │
│                                                                │
│   Бизнес-логика (Model, Inference) НЕ ЗАВИСИТ от backend       │
│                                                                │
│   GPU/CUDA/Metal — это УСКОРИТЕЛИ, не ядро системы             │
│                                                                │
│   80%+ вычислений на научных кластерах = CPU                   │
└─────────────────────────────────────────────────────────────────┘
```

**Приоритеты:**
1. **Корректность** — результаты совпадают с pyhf до 6 знаков
2. **Чистая архитектура** — trait-based abstraction, dependency inversion
3. **CPU Parallelism** — Rayon, SIMD, cluster job arrays (ОБЯЗАТЕЛЬНО)
4. **GPU Acceleration** — Metal, CUDA (ОПЦИОНАЛЬНО, как ускоритель)

## Quick Navigation

| Документ | Описание | Приоритет |
|----------|----------|-----------|
| [Master Plan](./2026-02-05-nextstat-implementation-plan.md) | Полный план всех фаз | — |
| [Project Standards](./standards.md) | Численный контракт, determinism, precision policy | — |
| [Version Baseline](./versions.md) | Актуальные версии toolchain/deps (validated) | — |
| [Phase 0: Infrastructure](./phase-0-infrastructure.md) | Репозиторий, CI/CD, лицензирование | P0 |
| [Phase 1: MVP-α](./phase-1-mvp-alpha.md) | Core engine, pyhf совместимость | P0 |
| [Phase 2A: CPU Parallelism](./phase-2a-cpu-parallelism.md) | Rayon, SIMD, кластеры | P0 |
| [Phase 2B: Autodiff & Optimizers](./phase-2b-autodiff.md) | AD, градиенты/Гессиан, L-BFGS(-B) | P0 |
| [Phase 2C: GPU Backends](./phase-2c-gpu-backends.md) | Metal, CUDA (опционально) | P1 |
| [Metal Batch GPU Plan](./metal-batch-gpu.md) | Metal f32 batch fitting — **COMPLETE** | P1 |
| [Phase 3: Production Ready](./phase-3-production.md) | Releases, docs, viz, validation | P1 |
| [Phase 4: Enterprise & SaaS](./phase-4-enterprise.md) | Audit, compliance, scale, hub, dashboard | P2 |
| [Phase 5: General Statistics Core](./phase-5-general-statistics.md) | Universal model API + distributions + validation harness | P0 |
| [Phase 6: Regression & GLM Pack](./phase-6-regression-glm.md) | Linear/Logistic/Poisson/NegBin + regularization | P0 |
| [Phase 7: Hierarchical Models](./phase-7-hierarchical-models.md) | Multilevel + LKJ + SBC/PPC | P0/P1 |
| [Phase 8: Time Series](./phase-8-time-series.md) | State-space + Kalman + forecasting | P1 |
| [Phase 9: Pharma + Social Sciences](./phase-9-pharma-social-sciences.md) | Survival + longitudinal + ordinal/missing/causal | P1/P2 |

## Bias & Bayesian (где искать)

- **BIAS решения / coverage policy:** `docs/plans/standards.md` → §6 + toy regression tests (`tests/python/test_bias_pulls.py`).
- **Bayesian (NUTS/HMC в Rust core):** `docs/plans/standards.md` → §7 + `docs/plans/phase-3-production.md` → Sprint 3.2.

## Timeline Overview

```
2026
├── Feb-Mar (Weeks 1-4)     Phase 0: Infrastructure
├── Apr-Jun (Weeks 5-12)    Phase 1: MVP-α Core Engine
├── Jul-Sep (Weeks 13-20)   Phase 2A: CPU Parallelism + Phase 2B: Autodiff
└── Oct-Dec (Weeks 21-28)   Phase 2C: GPU Backends (optional)
│
2027
├── Jan-Jun (Months 9-15)   Phase 3: Production Ready
└── Jul-Dec (Months 15-24)  Phase 4: Enterprise & SaaS

2028+
├── Phase 5: General Statistics Core (универсальный modeling слой)
├── Phase 6: Regression & GLM Pack (первый “wow” для широкой статистики/ML)
├── Phase 7: Hierarchical / Multilevel Models (соцнауки/фарма/ML)
├── Phase 8: Time Series & State Space (прогнозирование, longitudinal signals)
└── Phase 9: Pharma + Social Sciences Packs (survival, censored data, ordinal, missing, causal)
```

## Key Milestones

| Milestone | Target Date | Deliverables |
|-----------|-------------|--------------|
| M1: First Working Fit | Month 3 | `nextstat fit` produces correct μ̂ |
| M2: pyhf Parity | Month 4 | All validation tests pass |
| M3: 10x Faster (CPU) | Month 6 | CPU-parallel + batched toy fits |
| M4: GPU Acceleration (optional) | Month 9 | Metal/CUDA backend with relaxed parity |
| M5: White Paper | Month 9 | arXiv publication |
| M6: First Enterprise | Month 18 | Paid customer |
| M7: SaaS Launch | Month 24 | Cloud service live |

## Architecture Summary

### Принцип Dependency Inversion

```
┌─────────────────────────────────────────────────────────────────┐
│                    ВЫСОКОУРОВНЕВАЯ ЛОГИКА                       │
│            (не зависит от деталей реализации)                   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │   ns-inference: MLE, NUTS, Profile Likelihood            │   │
│  │   - Использует trait ComputeBackend                      │   │
│  │   - НЕ знает о CPU/GPU/Metal/CUDA                        │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │ depends on abstraction            │
│  ┌─────────────────────────┴───────────────────────────────┐   │
│  │   trait ComputeBackend {                                 │   │
│  │       fn nll(&self, ...) -> Result<f64>;                 │   │
│  │       fn gradient(&self, ...) -> Result<Vec<f64>>;       │   │
│  │       fn fit_batch(&self, ...) -> Result<Vec<Fit>>;      │   │
│  │   }                                                      │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │ implemented by                    │
├────────────────────────────┼───────────────────────────────────┤
│                    НИЗКОУРОВНЕВЫЕ ДЕТАЛИ                        │
│              (реализации, взаимозаменяемые)                     │
│                            │                                   │
│  ┌─────────────┬───────────┴───────────┬─────────────┐         │
│  │ CpuBackend  │    MetalBackend       │ CudaBackend │         │
│  │             │    (macOS only)       │ (NVIDIA)    │         │
│  │ - Rayon     │    - Metal shaders    │ - CUDA      │         │
│  │ - SIMD      │    - Accelerate       │ - cuBLAS    │         │
│  │ - ВСЕГДА    │    - ОПЦИОНАЛЬНО      │ - ОПЦИОНАЛЬНО│        │
│  └─────────────┴───────────────────────┴─────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Слои системы

```
┌─────────────────────────────────────────────────────────────────┐
│                    NextStat Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  ns-cli     │  │  ns-py      │  │  ns-viz     │  User Layer │
│  │  (CLI)      │  │  (Python)   │  │  (Plots)    │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│  ┌──────┴────────────────┴────────────────┴──────┐             │
│  │                  ns-inference                  │  Inference  │
│  │         MLE, NUTS, Profile Likelihood          │  Layer      │
│  │    (ЧИСТАЯ ЛОГИКА, backend-agnostic)           │             │
│  └──────────────────────┬────────────────────────┘             │
│                         │                                       │
│  ┌──────────────────────┴────────────────────────┐             │
│  │              ns-compute (trait-based)          │  Compute    │
│  │                                                │  Layer      │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐        │             │
│  │  │   CPU   │  │  Metal  │  │  CUDA   │        │             │
│  │  │ (Rayon) │  │(feature)│  │(feature)│        │             │
│  │  └─────────┘  └─────────┘  └─────────┘        │             │
│  └──────────────────────┬────────────────────────┘             │
│                         │                                       │
│  ┌──────────────────────┴────────────────────────┐             │
│  │                   ns-core                      │  Core       │
│  │         Types, Traits, Error handling          │  Layer      │
│  └───────────────────────────────────────────────┘             │
│                                                                 │
│  ┌───────────────────────────────────────────────┐             │
│  │                  ns-translate                  │  I/O        │
│  │       pyhf JSON, HistFactory XML, ROOT         │  Layer      │
│  └───────────────────────────────────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Почему такая архитектура

| Проблема | Решение |
|----------|---------|
| Ученые на Mac без GPU | CpuBackend работает ВЕЗДЕ |
| Кластеры без GPU (lxplus, NAF) | CpuBackend + Rayon + job arrays |
| Воспроизводимость | Детерминизм не зависит от backend |
| Тестирование | Можно тестировать с CpuBackend |
| Новые ускорители (TPU, etc.) | Добавить новый impl ComputeBackend |

## Tech Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Core | Rust 2024 | Performance, safety, WASM target |
| AD | Custom + JAX | Numerical precision, GPU support |
| Python | PyO3 + maturin | Ecosystem integration |
| GPU | CUDA + Metal | Cross-platform acceleration |
| Config | YAML + JSON Schema | Validation, IDE support |
| CI/CD | GitHub Actions | Industry standard |

## Acceptance Criteria

Каждая фаза имеет свои критерии приемки. Глобальные требования:

### Численная точность
- MLE μ̂: ±1e-6 относительно pyhf
- Uncertainties: ±1e-5 относительно pyhf
- NLL: ±1e-8 относительно pyhf

### Performance
- Simple fit (10 NP): < 10ms
- Complex fit (100 NP): < 200ms
- Ranking plot (100 NP): < 30s

### Quality
- Test coverage: ≥ 80%
- No clippy warnings
- Documentation for public API

## Working with Plans

### Для разработчиков

```bash
# Начать работу над фазой
cd /path/to/nextstat
git checkout -b phase-0/infrastructure

# Выполнять задачи по TDD
# 1. Прочитать задачу
# 2. Написать failing test
# 3. Реализовать минимальный код
# 4. Проверить что тест проходит
# 5. Commit

# После завершения epic
git push -u origin phase-0/infrastructure
# Создать PR
```

### Для AI агентов

Рекомендации:
- Выполнять задачи **последовательно** (task-by-task), закрывая чек-листы в конце каждого epic.
- Не менять математику/допуски “на глаз” — source of truth: `docs/plans/standards.md`.
- Если план ссылается на файл/документ, которого нет — **создать** его (stub → заполнить) и поправить ссылки.

Каждый план содержит пошаговые инструкции с:
- Точными путями к файлам
- Полным кодом (не "добавить валидацию")
- Командами для запуска
- Ожидаемыми результатами

## Reference Documents

Планы основаны на следующих документах:

1. **[NextStat_Business_Strategy_Analysis.md](../references/NextStat_Business_Strategy_Analysis.md)**
   - Стратегия монетизации
   - Timeline рекомендации
   - Risk mitigation

2. **[TRExFitter_vs_NextStat_Analysis.md](../references/TRExFitter_vs_NextStat_Analysis.md)**
   - Архитектурные уроки
   - Что копировать vs избегать
   - Preprocessing requirements

3. **[open-core-boundaries.md](../legal/open-core-boundaries.md)**
   - Границы OSS/Pro
   - Contribution policy (DCO/CLA)
   - Repo layout decision

## Contact

- Project Lead: @andresvlc
- Technical Lead: @andresvlc
- GitHub: https://github.com/nextstat/nextstat

---

*Last updated: 2026-02-07*
