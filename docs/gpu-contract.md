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

GPU kernels always use Code4 (NormSys) and Code4p (HistoSys) interpolation, matching the CPU default.

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

## Validation

### Rust integration tests

```bash
cargo test -p ns-inference --features cuda -- --nocapture
```

Tests in `crates/ns-inference/src/gpu_single.rs`:
- `test_gpu_nll_matches_cpu`
- `test_gpu_grad_matches_cpu`
- `test_gpu_fit_matches_cpu`
- `test_gpu_session_reuse`
- `test_gpu_complex_workspace`
- `test_gpu_nll_and_grad_at_multiple_points`

### Python parity tests

```bash
pytest tests/python/test_gpu_parity.py -v
```

### Tolerance source of truth

`tests/python/_tolerances.py` — all GPU/Metal tolerances defined here.
`crates/ns-inference/src/gpu_single.rs` — Rust-side constants for integration tests.
