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

GPU kernels currently support only smooth interpolation:

- NormSys: Code4
- HistoSys: Code4p

This matches the NextStat default for pyhf JSON inputs. If you select strict pyhf defaults
(Code1/Code0) via `--interp-defaults pyhf`, GPU commands will fail-fast.

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

CLI: `nextstat hypotest-toys --gpu cuda` or `nextstat hypotest-toys --gpu metal`.

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
