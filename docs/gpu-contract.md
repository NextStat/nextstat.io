# GPU Parity Contract

GPU-accelerated paths must produce results within specified tolerance of the CPU reference.

## Tolerance Tiers

| Tier | Metric | CUDA (f64) | Metal (f32) |
|------|--------|-----------|-------------|
| 1 | NLL at same params | atol=1e-8, rtol=1e-6 | atol=1e-3 |
| 2 | Gradient per-element | atol=1e-5 | atol=1e-2 |
| 3 | Best-fit params | atol=2e-4 | atol=1e-2 |
| 4 | Fit NLL | atol=1e-6 | atol=1e-3 |

## Interpolation

GPU HistFactory likelihood kernels currently support only smooth interpolation:

- NormSys: Code4
- HistoSys: Code4p

This matches the NextStat default for pyhf JSON inputs. If you select strict pyhf defaults
(Code1/Code0) via `--interp-defaults pyhf`, GPU commands will fail-fast.

Unbinned helper kernels may support additional interpolation modes:

- Per-event WeightSys morphing (`kernels/unbinned_weight_sys.cu`): Code0, Code4p

## CUDA f64

- Same precision as CPU (f64 throughout)
- Tolerance arises from: atomic reduction order, GPU floating-point rounding
- All 7 modifier types supported: NormFactor, ShapeSys, ShapeFactor, NormSys, HistoSys, StatError, Lumi

## Metal f32

- All computation in f32 (Apple Silicon has no hardware f64)
- f64↔f32 conversion at API boundary
- L-BFGS-B convergence tolerance relaxed to 1e-3 (vs 1e-6 for CUDA)
- Auxiliary Poisson `lgamma` precomputed on CPU (Metal has no lgamma)

## Differentiable Layer

Both CUDA and Metal support the differentiable NLL layer:

| Feature | CUDA | Metal |
|---------|------|-------|
| Signal upload | Zero-copy via raw pointer | CPU → GPU (f64→f32) |
| Gradient return | Zero-copy or Vec<f64> | Vec<f64> (f32→f64) |
| Profiled q₀/qμ | GPU L-BFGS-B + envelope theorem | Same algorithm, f32 precision |
| Multi-channel signal | Supported | Supported |
| PyTorch integration | Direct (same CUDA context) | Via CPU tensor bridge |

## Batch Toy Fitting

Both CUDA and Metal support GPU-accelerated batch toy fitting for CLs hypothesis testing:

| Entry Point | Description |
|-------------|-------------|
| `fit_toys_batch_gpu` / `fit_toys_batch_metal` | High-level: generate toys from model params |
| `fit_toys_from_data_gpu` / `fit_toys_from_data_metal` | Low-level: custom expected data, init, bounds |
| `hypotest_qtilde_toys_gpu(device="cuda"\|"metal")` | Full CLs workflow: Phase A (CPU baseline) + Phase B (GPU ensemble) |

Architecture: Phase A performs 3 baseline CPU fits (free, conditional at μ_test, conditional at μ=0), then Phase B dispatches to the appropriate GPU backend for batch toy ensemble generation.

CLI: `nextstat hypotest-toys --gpu cuda` or `nextstat hypotest-toys --gpu metal` (requires building with the corresponding feature).

## Ranking (Nuisance Impacts)

`/v1/ranking` uses a hybrid strategy on both CUDA and Metal:

- Nominal fit: CPU f64 (needs Hessian for pull/constraint).
- Per-nuisance refits (±1σ): GPU-accelerated session (CUDA f64 or Metal f32), warm-started and bounds-clamped.

Contract:
- `pull`, `constraint`: CPU f64 nominal fit; treat as the reference (should match CPU ranking).
- `delta_mu_up/down`: Metal refits are f32; expected agreement with CPU ranking is approximate.
  - For meaningful impacts (`max(|delta_mu|) > 0.01`): expect ~1e-3 relative error; use `atol ~ 1e-2` for regression tests.
  - Sorting: top-10 impacts should be stable; tail can permute when impacts are near numerical noise.

## Performance (RTX 4000 SFF Ada, CUDA 12.0)

| Operation | CPU | CUDA | Winner |
|-----------|-----|------|--------|
| MLE fit (8 params) | 2.3 ms | 136.3 ms | CPU 59x |
| MLE fit (184 params) | 520.8 ms | 1,272.0 ms | CPU 2.4x |
| Profile scan (184p, 21pt) | 8.4 s | 7.9 s | **GPU 1.07x** |
| Batch toys (184p, 1000 toys) | 771.4 s | 119.9 s | **GPU 6.4x** |
| Batch toys (8p, 1000 toys) | 40 ms | 1,838 ms | CPU 46x |

### Metal (Apple M5, f32)

| Operation | CPU | Metal | Winner |
|-----------|-----|-------|--------|
| Batch toys (184p, 1000 toys) | 359.1 s | 56.8 s | **GPU 6.3x** |
| Batch toys (8p, 1000 toys) | 132 ms | 2,378 ms | CPU 18x |
| Diff NLL + grad (8 params) | — | 0.12 ms | GPU-only |
| Diff NLL + grad (184 params) | — | 3.66 ms | GPU-only |
| Profiled q₀ (8 params) | — | 3.0 ms | GPU-only |
| NN training loop | — | 2.4 ms/step | GPU-only |

**Recommendation**: Use GPU for batch toy fitting on large models (6.4x at 184 params),
differentiable training, and large-model scans. Use CPU for single-model fits and small models.

## Unbinned Flow NLL Reduction

A dedicated CUDA kernel (`flow_nll_reduce`) handles NLL reduction from externally-computed
log-prob values (flow PDFs evaluated via ONNX Runtime). This separates PDF evaluation
from likelihood reduction, enabling mixed parametric+flow models.

### Contract

| Metric | Tolerance (CUDA f64) |
|--------|---------------------|
| NLL at same log-prob input | atol=1e-8 |
| NLL with host-upload vs device-resident | exact (same kernel) |

### Supported features

- Multi-process logsumexp reduction (signal + background)
- Gaussian nuisance constraints
- Host-upload path (`nll`): log-prob evaluated on CPU, uploaded per iteration
- Device-resident path (`nll_device`): log-prob from ONNX CUDA EP stays on GPU (zero-copy)
- **f32 device-pointer path (`nll_device_ptr_f32`)**: accepts a raw CUDA device pointer to `float` log-probs produced by ONNX Runtime CUDA EP with I/O Binding. Eliminates the f64 host-upload entirely — the f32 tensor stays on the GPU from ONNX EP output through NLL reduction. Up to **57× faster** than the f64 host-upload path at typical event counts (~1K). Python binding: `GpuFlowSession.nll_device_ptr_f32(ptr, params)`. Tolerance: NLL agrees with the f64 path to ~1e-3 (f32 precision).
- Batch variant (`flow_batch_nll_reduce`): 1 CUDA block per toy dataset

### Validation

```bash
# GPU flow session tests (requires NVIDIA GPU)
cargo test -p ns-inference --features cuda --test gpu_flow_session_test

# CPU flow integration tests (no GPU needed, requires ONNX fixtures)
cargo test -p ns-unbinned --features neural --test flow_integration
```

## TensorRT Execution Provider for Neural PDFs

When built with `--features neural-tensorrt`, `FlowPdf` attempts TensorRT EP first
with automatic fallback to CUDA EP for unsupported ops.

### Configuration (`FlowGpuConfig`)

| Field | Default | Description |
|-------|---------|-------------|
| `fp16` | `true` | FP16 inference (2× throughput on Tensor Cores) |
| `engine_cache_path` | `~/.cache/nextstat/tensorrt/` | Compiled TRT engine cache (skips recompilation) |
| `profile_min_batch` | `1` | TRT optimization profile: minimum batch |
| `profile_opt_batch` | `1024` | TRT optimization profile: optimal batch |
| `profile_max_batch` | `65536` | TRT optimization profile: maximum batch |

### Runtime introspection

```rust
let flow = FlowPdf::from_manifest(&path, &[])?;
match flow.gpu_ep_kind() {
    Some(FlowGpuEpKind::TensorRtEp) => println!("TensorRT active"),
    Some(FlowGpuEpKind::CudaEp)     => println!("CUDA EP fallback"),
    None                              => println!("CPU only"),
}
```

### Engine cache behaviour

- **First run**: TRT compiles the ONNX model → engine file saved to cache dir (10–60 s).
- **Subsequent runs**: engine loaded from cache (<1 s).
- Cache is keyed by model hash + TRT version + GPU architecture.
- Set `engine_cache_path: None` to disable caching.

### Feature flags

```toml
# Cargo.toml
[features]
neural-tensorrt = ["neural-cuda", "ort/tensorrt"]
```

`neural-tensorrt` implies `neural-cuda`. CUDA EP is always available as fallback.

## Direct-to-GPU Parquet Pipeline

End-to-end path from Parquet file to GPU-resident event buffers, bypassing intermediate
Arrow/DataFrame materialization. Designed for unbinned likelihood fits with large event
datasets.

### Data flow

```
Parquet file
  → mmap (memmap2, zero-copy)
  → row group predicate pushdown (min/max statistics)
  → parallel row group decode (rayon)
  → Arrow Float64Array → SoA f64 buffer
  → CUDA: clone_htod → CudaSlice<f64>      (f64 precision, zero conversion)
  → Metal: f64→f32 + new_buffer_with_data → MTLBuffer  (f32 precision)
```

### Modules

| Module | Crate | Function |
|--------|-------|----------|
| `arrow::parquet` | `ns-translate` | `read_parquet_events_soa()` → `ParquetEventData` |
| `cuda_parquet` | `ns-compute` | `upload_events_to_cuda()`, `upload_events_with_bounds_to_cuda()` |
| `metal_parquet` | `ns-compute` | `upload_events_to_metal()`, `upload_events_with_bounds_to_metal()` |

### Performance (MacBook M5, 100k events, 4 observables)

Parquet read benchmarks (Criterion, `benches/parquet_read.rs`):

| Operation | Time | Notes |
|-----------|------|-------|
| mmap + decode (all cols, 100k) | 557 µs | Single-threaded, 4 columns |
| mmap + decode (2/4 cols projected) | 362 µs | Column projection saves ~35% |
| mmap + decode (1 col projected) | 259–314 µs | Column projection saves ~50% |
| Predicate pushdown (no filter) | 813 µs | Baseline, 10 row groups |
| Predicate pushdown (25% pass) | 728 µs | ~10% faster via RG pruning |
| Predicate pushdown (10% pass) | 159 µs | **5× faster** — 9/10 RGs pruned |
| Parallel decode (all RGs) | 3.16 ms | rayon overhead on small data |
| Sequential decode (all RGs) | 943 µs | Sequential faster for <1M events |
| Parallel + filtered (10% pass) | 1.11 ms | Parallel helps when RGs pass |
| Sequential + filtered (10% pass) | 303 µs | Sequential wins for selective filters |

Metal GPU upload (RTX → MTLBuffer, f64→f32 conversion):

| Operation | Time | Notes |
|-----------|------|-------|
| SoA upload (6 elements, roundtrip) | <0.15 ms | Dominated by Metal device init |

CUDA GPU upload (Hetzner RTX 4000 SFF Ada):

| Operation | Time | Notes |
|-----------|------|-------|
| SoA upload (6 elements, roundtrip) | <0.15 ms | Dominated by CUDA context init |

**Recommendation**: For production datasets (>100k events), predicate pushdown provides
the largest speedup (up to 5×). Column projection adds ~35% on top. Parallel decode
benefits large files (>1M events, many row groups). The GPU upload itself is negligible
compared to Parquet decode.

## Known Issues (Fixed)

- **Batch toys memcpy_dtoh panic**: cudarc 0.19 requires `dst.len() >= src.len()` for
  device-to-host copies. When toys converge and `n_active < max_batch`, the host buffer
  was too small. Fix: allocate host buffers at `max_batch` size and truncate.
- **ProfiledDifferentiableSession convergence**: L-BFGS-B tolerance 1e-6 too tight for
  projected gradient near parameter bounds. Fix: tolerance 1e-5 + NLL stability criterion.

## Validation

### Rust integration tests

```bash
# Single-model fit + gradient parity
cargo test -p ns-inference --features cuda -- --nocapture

# Metal batch tests
cargo test -p ns-inference --features metal -- --nocapture
```

Rust-side GPU regression tests live alongside the GPU session implementation in `crates/ns-inference/src/gpu_session.rs`.

### Python parity tests

```bash
pytest tests/python/test_gpu_parity.py -v
```

### Tolerance source of truth

`tests/python/_tolerances.py` — all GPU/Metal tolerances defined here.
`crates/ns-inference/src/gpu_session.rs` — Rust-side CUDA regression tests (skip automatically if CUDA not available).
