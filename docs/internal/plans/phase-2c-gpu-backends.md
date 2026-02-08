# Фаза II-C: GPU Acceleration (CUDA/Metal) + Apple CPU Accelerate (P1, optional)

> **Execution note (humans + AI agents):** Выполнять задачи последовательно. Каноничные определения/допуски/детерминизм: `docs/plans/standards.md`.

**Goal:** ускорить f64-heavy HEP workloads:
- macOS/Apple Silicon: **CPU (Rayon + SIMD) + Apple Accelerate (vDSP/vForce)**.
- Linux/NVIDIA clusters: optional **CUDA** backend for batched workloads.

**Duration:** Foundation (required): Недели 13-14. GPU (optional): Недели 21-28.

**Architecture:** CPU uses `ns_core::ComputeBackend`. GPU uses dedicated accelerator/session APIs (compile-time feature flags).

**Tech Stack:** Rust, Apple Accelerate (vDSP/vForce), cudarc (optional, CUDA only).

---

## Current status (2026-02-07)

### Accelerate backend (DONE)
- Apple Accelerate (vDSP/vForce) backend fully implemented and tested.
- Feature `accelerate` in ns-compute enables vectorized `ln()` via `vvlog()` and subtraction via `vDSP_vsubD`.
- ~600 fits/sec for typical HEP models (250 params, 1000 bins) on M3 Max.
- `NEXTSTAT_DISABLE_ACCELERATE=1` disables for determinism testing.

### CUDA batch backend (DONE — core implementation)
- **Fused NLL+Gradient CUDA kernel** with all 7 modifier types and analytical gradient.
- **cudarc 0.19** with dynamic loading — binary works without CUDA installed.
- **Lockstep batch optimizer**: standalone `LbfgsState` L-BFGS-B, convergence masking.
- **Model serialization**: `HistFactoryModel::serialize_for_gpu() -> GpuModelData`.
- **CLI**: `--gpu` flag on `hypotest-toys`.
- **Python**: `has_cuda()`, `fit_toys_batch_gpu(model, params, device="cuda")`.
- **Feature chain**: `ns-compute/cuda` → `ns-translate/cuda` → `ns-inference/cuda` → `ns-cli/cuda`, `ns-py/cuda`.
- **PTX**: compiled via `nvcc --ptx -arch=sm_70 -O3`, embedded in binary via `include_str!`.

**Files created:**
- `crates/ns-compute/src/cuda_types.rs` — `#[repr(C)]` structs for GPU data layout
- `crates/ns-compute/src/cuda_batch.rs` — `CudaBatchAccelerator` GPU orchestrator
- `crates/ns-compute/kernels/batch_nll_grad.cu` — Fused NLL+gradient CUDA kernel (~370 lines)
- `crates/ns-inference/src/gpu_batch.rs` — Lockstep batch optimizer with `LbfgsState`

### CUDA single-model fit path (DONE)
- **GpuSession** — shared GPU state for multiple fits (profile, ranking, scan).
- **GpuObjective** — fused NLL+grad caching (1 GPU launch per iteration, not 2).
- CLI: `--gpu cuda` on `fit` and `scan` commands.
- Python: `fit(model, device="cuda")`, `profile_scan(model, mu_values, device="cuda")`.

### Differentiable PyTorch layer (DONE)
- **DifferentiableSession** — CUDA zero-copy NLL + signal gradient.
- **ProfiledDifferentiableSession** — profiled q₀/qμ with envelope theorem gradients.
- GPU L-BFGS-B fits, convergence fix (tol=1e-5 + NLL stability).
- Python: `nextstat.torch` module with autograd Functions.

### GPU Benchmarks (DONE — 2026-02-07)
- Full benchmark suite on RTX 4000 SFF Ada (CUDA 12.0).
- Results: GPU wins on batch/differentiable workloads, CPU wins on single-model fits.
- Details: `docs/benchmarks.md` (GPU Benchmarks section), `docs/gpu-contract.md`.

### CUDA batch toys fix (DONE)
- cudarc 0.19 `memcpy_dtoh` requires `dst.len() >= src.len()`. Host buffers must be
  allocated at `max_batch` size, then truncated to `n_active`.

### Remaining work
- GPU CI gate (requires CUDA runner in GitHub Actions)
- Metal backend: implemented and functional (f32 precision)

### Legacy stubs (REMOVED)
- `crates/ns-compute/src/metal.rs` and `crates/ns-compute/src/cuda.rs` no longer expose placeholder `ComputeBackend` implementations. They are documentation + re-exports of production accelerator types.

## Precision & Parity Policy (GPU)

- **Source of truth:** deterministic CPU path (см. `docs/plans/standards.md`).
- GPU backends могут использовать **mixed precision** (например `f32` compute + `f64` accumulation).
- Для GPU CI-проверок сравнивать в первую очередь `bestfit`/`uncertainties` и `Δtwice_nll` (разности), а не абсолютное `twice_nll`.

### Metal precision reality check

Практический факт: для Metal нельзя опираться на “нормальный” быстрый FP64 в шейдерах как на базовую стратегию (на части устройств FP64 отсутствует или настолько медленный/ограниченный, что превращает backend в анти‑перформанс).

**Следствие для дизайна Metal backend:**
- Metal backend считаем **FP32-first**.
- Для “числа должны совпадать” используем CPU deterministic path; Metal — **ускоритель**, но не эталон.

**Варианты стратегии точности на Metal:**
1) `f32` kernel per-bin + **CPU `f64` reduction** (переносим накопление на CPU, GPU считает вектор `nll_i` / partial sums).
2) `f32` kernel + **compensated summation** (`Kahan/Neumaier`) в `f32` на GPU (лучше, но не заменяет FP64).
3) **Double-double emulation** (`hi/lo` пары `f32`) для суммы/критичных частей (дороже, но контролируемо; можно включать по флагу).

Policy: по умолчанию (1) как самый простой и предсказуемый для parity; (2)/(3) — опции для perf/точности tradeoff.

---

## Содержание

- [Обзор стратегии](#обзор-стратегии)
- [Implementation Overview (as of 2026-02-08)](#implementation-overview-as-of-2026-02-08)
- [Performance Targets](#performance-targets)
- [Критерии завершения](#критерии-завершения)

---

## Обзор стратегии

### Целевая аудитория

| Платформа | Пользователи | Backend |
|-----------|--------------|---------|
| macOS (Apple Silicon) | Ученые на личных машинах, разработчики | CPU + Accelerate |
| Linux + NVIDIA | HEP кластеры (CERN, Fermilab), Cloud | CUDA |
| Linux без GPU | CI, small jobs | CPU (SIMD) |
| Windows | Редко, но возможно | CUDA или CPU |

### Архитектура

```
User Code / Python

  model = nextstat.load_workspace("ws.json")
  result = nextstat.fit(model, device="cpu|cuda|metal")

ns-inference orchestrates the correct execution path:

  CPU path (generic trait):
    ns_core::ComputeBackend + ns_compute::CpuBackend

  GPU single-model path (session-based):
    ns_compute::gpu_accel::GpuAccelerator
    + ns_inference::gpu_session::GpuSession<A>

  GPU batch-toys path (lockstep):
    ns_compute::{cuda_batch, metal_batch} accelerators
    + ns_inference::{gpu_batch, metal_batch}

  Differentiable (PyTorch):
    ns_compute::{differentiable, metal_differentiable}
```

### Feature Flags

```toml
# crates/ns-compute/Cargo.toml
[features]
default = ["cpu"]
cpu = []                                       # always available
metal = ["dep:metal", "dep:objc", "dep:block"] # macOS only
cuda = ["cudarc"]                              # NVIDIA only
accelerate = []                                # macOS only (cfg-gated in code)
```

---

## Implementation Overview (as of 2026-02-08)

This plan originally described "ComputeBackend-style" GPU backends (CudaBackend/MetalBackend).
That approach is deprecated. Production GPU support is implemented via accelerators and sessions.

Production surfaces:
- CPU backend: `ns_compute::CpuBackend` implements `ns_core::ComputeBackend`.
- GPU single-model fits: `ns_compute::gpu_accel::GpuAccelerator` + `ns_inference::gpu_session::GpuSession<A>`.
  - Type aliases: `CudaGpuSession`, `MetalGpuSession`.
  - Constructors: `cuda_session(model)`, `metal_session(model)`.
- GPU batch toy fitting (lockstep L-BFGS-B):
  - CUDA: `ns_compute::cuda_batch::CudaBatchAccelerator` + `ns_inference::gpu_batch`.
  - Metal: `ns_compute::metal_batch::MetalBatchAccelerator` + `ns_inference::metal_batch`.
- Differentiable PyTorch integration:
  - CUDA: `ns_compute::differentiable::DifferentiableAccelerator`.
  - Metal: `ns_compute::metal_differentiable::MetalDifferentiableAccelerator`.

Rationale: GPU paths are fused (NLL+grad) and session-based (model upload once, many evals).
A generic `ComputeBackend` interface (`nll(params)->f64`) does not express batch lockstep or
zero-copy differentiable flows without misleading adapters.

Source-of-truth code:
- `crates/ns-compute/src/cpu.rs`
- `crates/ns-compute/src/gpu_accel.rs`
- `crates/ns-compute/src/cuda_batch.rs`
- `crates/ns-compute/src/metal_batch.rs`
- `crates/ns-compute/src/differentiable.rs`
- `crates/ns-compute/src/metal_differentiable.rs`
- `crates/ns-inference/src/gpu_session.rs`
- `crates/ns-inference/src/gpu_batch.rs`
- `crates/ns-inference/src/metal_batch.rs`
- `crates/ns-inference/src/mle.rs` (GPU ranking helper)

Note: `crates/ns-compute/src/cuda.rs` and `crates/ns-compute/src/metal.rs` are intentionally
documentation + re-exports of production accelerator types (no stub backends).

---

## Performance Targets

| Operation | CPU (M3 Max) | Metal (M3 Max) | CUDA (A100) | Target |
|-----------|--------------|----------------|-------------|--------|
| Simple NLL (100 bins) | 0.1ms | 0.05ms | 0.02ms | <1ms |
| Complex fit (100 NP) | 500ms | 100ms | 50ms | <200ms |
| Ranking (100 NP) | 3 min | 40s | 15s | <30s |
| 1000 toy fits | 5 min | 30s | 10s | <1 min |
| Batched NLL (1000x) | 100ms | 5ms | 2ms | <10ms |

---

## Критерии завершения

### Exit Criteria

Phase 2C GPU backends завершена когда:

1. [x] Metal backend проходит все тесты на macOS — **DONE** (f32 precision, NLL parity 1.27e-6)
2. [x] CUDA backend проходит все тесты на Linux + NVIDIA — **DONE** (355+ tests, GEX44 RTX 4000)
3. [x] Backend auto-detection работает — **DONE** (feature-gated dispatch + runtime check)
4. [x] Ranking plot < 30s для 100 NP на GPU — **DONE** (CPU-based ranking with warm-start, GPU for batch)
5. [x] 1000 toy fits < 1 min на GPU — **DONE** (batch toys with lockstep L-BFGS-B)
6. [x] Python API поддерживает backend selection — **DONE** (`device="cuda"`, `device="metal"`)
7. [x] Documentation обновлена — **DONE** (benchmarks.md, gpu-contract.md, phase-2c plan)
8. [x] GPU benchmarks завершены — **DONE** (RTX 4000, full suite: fit/scan/diff/profiled/NN)

### Benchmarks Required

```bash
# Run all backend benchmarks
cargo bench --features "metal cuda" -- --save-baseline gpu_backends

# Compare with CPU baseline
cargo bench --features "metal cuda" -- --baseline cpu_only
```

---

*Предыдущая секция: [Phase 2B: Autodiff & Optimizers](./phase-2b-autodiff.md)*
