# Metal GPU — Full Implementation Plan

**Status**: Batch fitting COMPLETE. Single-model fit + profile scan deferred to backlog.
**Created**: 2026-02-07
**Updated**: 2026-02-07 (batch backend implemented and validated)
**Prerequisite**: GPU-Accelerated Single-Model Fit Path (✅ DONE)

## Context

NextStat has a working CUDA GPU backend covering **three acceleration paths**:
1. **Batch toy fitting** — lockstep L-BFGS-B, 10K toys in parallel (`gpu_batch.rs`)
2. **Single-model fit** — fused NLL+gradient with caching, 1 kernel/iteration (`gpu_single.rs`)
3. **Profile likelihood scan** — shared GpuSession, warm-start across mu values (`profile_likelihood.rs::scan_gpu`)

Apple Silicon has no hardware f64 in Metal, but f32 is acceptable for CLs (toy ranking, not absolute precision) and single-model fits (tolerance relaxed to 1e-4). This plan implements a real Metal backend mirroring all three CUDA paths.

**f32 Precision Assessment (PoC results on tHu, 184 params, 2026-02-07):**

| Metric | Value | Verdict |
|--------|-------|---------|
| NLL rel error at init | 3.4e-7 | OK (6-7 digits) |
| NLL rel error perturbed | 4.6e-6 | OK for ranking |
| FD gradient max rel error | 4.0x | UNUSABLE (FD amplifies f32 noise) |
| FD gradient median rel error | 1.0 (100%) | UNUSABLE |
| **Analytical gradient max rel error** | **3.2e-4** | **SUFFICIENT** |
| **Analytical gradient median rel error** | **4.5e-7** | **EXCELLENT** |
| **Analytical gradient P99 rel error** | **7.6e-5** | **SUFFICIENT** |
| **Sign mismatches** | **0 / 184** | **PERFECT** |
| **Perturbed max rel error** | **3.7e-5** | **SUFFICIENT** |

**Conclusion (validated)**: f32 analytical gradients are accurate enough for L-BFGS-B on 184-param models. The worst-case component has 0.03% error, median is sub-ppm, and there are zero sign disagreements. Metal GPU implementation is viable for both batch toy fitting AND single-model fits.

Validated via `Dual32` forward-mode AD: `test_f32_analytical_gradient_poc` in `ns-translate/src/pyhf/tests.rs`.

---

## Architecture Decision: Device-Agnostic GPU Abstraction

The CUDA single-model fit path introduces `GpuSession` + `GpuObjective` with 6 methods in `mle.rs` and 1 in `profile_likelihood.rs`. Rather than duplicating all of this for Metal, we extract a trait:

```rust
/// Trait for GPU accelerators (CUDA f64, Metal f32).
pub trait GpuAccelerator {
    fn single_nll_grad(&mut self, params: &[f64]) -> Result<(f64, Vec<f64>)>;
    fn upload_observed_single(&mut self, obs: &[f64], lnf: &[f64], mask: &[f64]) -> Result<()>;
    fn n_params(&self) -> usize;
    fn n_main_bins(&self) -> usize;
}
```

Then `GpuSession<A: GpuAccelerator>` is generic. Both `CudaBatchAccelerator` and `MetalBatchAccelerator` implement the trait. Result: `mle.rs` GPU methods work for both backends with zero duplication.

---

## Files Overview

### New files (7)
| File | Description |
|------|-------------|
| `crates/ns-compute/src/metal_types.rs` | f32 `#[repr(C)]` structs (always available, no feature gate) |
| `crates/ns-compute/src/metal_batch.rs` | `MetalBatchAccelerator` — buffer mgmt, kernel dispatch |
| `crates/ns-compute/src/gpu_accel.rs` | `GpuAccelerator` trait (always available) |
| `crates/ns-compute/kernels/batch_nll_grad.metal` | MSL kernel (port from .cu, all float) |
| `crates/ns-inference/src/metal_batch.rs` | `fit_toys_batch_metal()` — lockstep entry point |
| `crates/ns-inference/src/lbfgs.rs` | Extracted `LbfgsState` (shared by CUDA + Metal) |
| `crates/ns-inference/src/gpu_session.rs` | Generic `GpuSession<A>` (extracted from `gpu_single.rs`) |

### Modified files (13)
| File | Change |
|------|--------|
| `Cargo.toml` (workspace) | Add `metal = "0.33"` dependency |
| `crates/ns-compute/Cargo.toml` | `metal = ["dep:metal"]` feature, macOS-only dep |
| `crates/ns-compute/src/lib.rs` | Add `metal_types`, `metal_batch`, `gpu_accel` modules |
| `crates/ns-compute/src/cuda_batch.rs` | `impl GpuAccelerator for CudaBatchAccelerator` |
| `crates/ns-compute/build.rs` | Optional .metallib offline compilation |
| `crates/ns-translate/Cargo.toml` | `metal = ["ns-compute/metal"]` |
| `crates/ns-inference/Cargo.toml` | `metal = ["ns-translate/metal", "ns-compute/metal"]` |
| `crates/ns-inference/src/lib.rs` | Add `lbfgs`, `metal_batch`, `gpu_session` modules |
| `crates/ns-inference/src/gpu_single.rs` | Refactor: use `GpuSession<CudaBatchAccelerator>` from `gpu_session.rs` |
| `crates/ns-inference/src/mle.rs` | GPU methods generic over accelerator + Metal variants |
| `crates/ns-inference/src/profile_likelihood.rs` | `scan_gpu` generic + `scan_metal` |
| `crates/ns-inference/src/batch.rs` | Add `is_metal_batch_available()` |
| `crates/ns-inference/src/gpu_batch.rs` | Use `crate::lbfgs::LbfgsState` instead of inline |
| `crates/ns-cli/Cargo.toml` + `main.rs` | `--gpu cuda\|metal` on fit, scan, hypotest-toys |
| `bindings/ns-py/Cargo.toml` + `lib.rs` | `has_metal()`, `device="metal"` on all GPU functions |

---

## Step-by-step

### Step 1: Dependencies & Feature Chain (~10 min)

**Workspace Cargo.toml**: add `metal = { version = "0.33" }`

**ns-compute/Cargo.toml**:
```toml
[target.'cfg(target_os = "macos")'.dependencies]
metal = { workspace = true, optional = true }

[features]
metal = ["dep:metal"]   # was empty []
```

**Feature propagation** (same pattern as `cuda`):
- ns-translate: `metal = ["ns-compute/metal"]`
- ns-inference: `metal = ["ns-translate/metal", "ns-compute/metal"]`
- ns-cli: `metal = ["ns-inference/metal", "ns-translate/metal", "ns-compute/metal"]`
- ns-py: same chain

**Verify**: `cargo check -p ns-compute --features metal` compiles.

### Step 2: Metal Types (f32 repr(C)) (~45 min)

**New file**: `crates/ns-compute/src/metal_types.rs` (always available, no feature gate — like `cuda_types.rs`)

Key types mirroring `cuda_types.rs` but f32:
- `MetalSampleInfo` — same as `GpuSampleInfo` (all u32, no change)
- `MetalModifierDesc` — same as `GpuModifierDesc` (all u32/u8, no change)
- `MetalAuxPoissonEntry` — f32 tau, observed_aux, **plus `lgamma_obs: f32`** (precomputed, Metal has no lgamma)
- `MetalGaussConstraintEntry` — f32 center, inv_width
- `MetalModelData` — all `Vec<f32>` for compute data

**Conversion**: `impl MetalModelData { fn from_gpu_data(data: &GpuModelData) -> Self }` — casts all f64→f32, precomputes lgamma.

Reuse `GpuModifierType` enum from `cuda_types` (no duplication).

**Register in lib.rs**: `pub mod metal_types;` (unconditional, like `cuda_types`).

### Step 3: Extract LbfgsState (~30 min)

**New file**: `crates/ns-inference/src/lbfgs.rs`

Move `LbfgsState` (lines 31-256 from `gpu_batch.rs`) to shared module. Add public accessors: `parameters()`, `fval()`, `converged()`, `iter()`, `n_fev()`, `n_gev()`.

**Update `gpu_batch.rs`**: replace inline struct with `use crate::lbfgs::LbfgsState;`.

**Update `lib.rs`**: add `pub(crate) mod lbfgs;`

**Verify**: `cargo test -p ns-inference --features cuda` still passes (CUDA path unchanged).

### Step 4: MSL Kernel (~3-4 hours)

**New file**: `crates/ns-compute/kernels/batch_nll_grad.metal`

Line-by-line port from `batch_nll_grad.cu` (563 lines) with translations:

| CUDA | Metal (MSL) |
|------|------------|
| `__global__ void` | `kernel void` |
| `blockIdx.x` | `uint toy_idx [[threadgroup_position_in_grid]]` |
| `threadIdx.x` | `uint tid [[thread_position_in_threadgroup]]` |
| `blockDim.x` | `uint block_size [[threads_per_threadgroup]]` |
| `extern __shared__ double[]` | `threadgroup float *shared [[threadgroup(0)]]` |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `atomicAdd(&ptr, val)` | `atomic_fetch_add_explicit((device atomic_float*)&ptr, val, memory_order_relaxed)` |
| `double` | `float` everywhere |
| `lgamma(obs+1)` | `aux.lgamma_obs` (precomputed) |
| `log()`, `exp()`, `fabs()` | same names in `metal_stdlib` |

**Two kernels**: `batch_nll_grad` (NLL + gradient) and `batch_nll_only` (NLL only).

**Buffer bindings**: 20 `[[buffer(N)]]` slots (14 data buffers + 6 scalar constants via `set_bytes`).

**Shared memory**: `threadgroup float *shared [[threadgroup(0)]]` — size set at dispatch via `set_threadgroup_memory_length(0, (n_params + block_size) * 4)`.

**Gradient atomics**: Use `device atomic_float*` for `g_grad_out`. Requires Apple GPU family 7+ (M1 and later). Gradient buffer zeroed before dispatch.

**Verify**: `xcrun -sdk macosx metal -c kernels/batch_nll_grad.metal -o /dev/null`

### Step 5: GpuAccelerator Trait + MetalBatchAccelerator (~3-4 hours)

**New file**: `crates/ns-compute/src/gpu_accel.rs` (always available, no feature gate)

```rust
/// Device-agnostic GPU accelerator trait.
///
/// Implemented by CudaBatchAccelerator (f64) and MetalBatchAccelerator (f32).
/// Enables generic GpuSession<A: GpuAccelerator> for single-model fit path.
pub trait GpuAccelerator {
    fn single_nll_grad(&mut self, params: &[f64]) -> ns_core::Result<(f64, Vec<f64>)>;
    fn single_nll(&mut self, params: &[f64]) -> ns_core::Result<f64>;
    fn upload_observed_single(&mut self, obs: &[f64], lnf: &[f64], mask: &[f64]) -> ns_core::Result<()>;
    fn n_params(&self) -> usize;
    fn n_main_bins(&self) -> usize;
    fn max_batch(&self) -> usize;

    // Batch methods (for toy fitting)
    fn upload_observed(&mut self, obs: &[f64], lnf: &[f64], mask: &[f64], n_toys: usize) -> ns_core::Result<()>;
    fn batch_nll_grad(&mut self, params: &[f64], n_active: usize) -> ns_core::Result<(Vec<f64>, Vec<f64>)>;
    fn batch_nll(&mut self, params: &[f64], n_active: usize) -> ns_core::Result<Vec<f64>>;
}
```

**Register in lib.rs**: `pub mod gpu_accel;`

**Update `cuda_batch.rs`**: `impl GpuAccelerator for CudaBatchAccelerator` — delegates to existing methods.

---

**New file**: `crates/ns-compute/src/metal_batch.rs` (feature-gated `#[cfg(feature = "metal")]`)

Mirrors `CudaBatchAccelerator` API + implements `GpuAccelerator`:
- `is_available() -> bool` — `Device::system_default().is_some()`
- `from_metal_data(data, max_batch)` — compile MSL at runtime, upload static buffers
- `upload_observed(obs, lnf, mask, n_toys)` — f64→f32 conversion, memcpy to shared buffer
- `batch_nll_grad(params, n_active) -> (Vec<f64>, Vec<f64>)` — f64→f32 in, dispatch, f32→f64 out
- `batch_nll(params, n_active) -> Vec<f64>` — NLL-only variant
- `single_nll_grad()`, `single_nll()`, `upload_observed_single()` — convenience for n=1

**Key implementation details**:
- `MTLResourceOptions::StorageModeShared` for unified memory (zero-copy on Apple Silicon)
- MSL compiled via `device.new_library_with_source(MSL_SRC, &CompileOptions::new())`
- Scalar args via `encoder.set_bytes(index, size, ptr)` (not separate buffers)
- Threadgroup memory via `encoder.set_threadgroup_memory_length(0, bytes)`
- Read results via `buffer.contents()` pointer cast (shared memory = CPU-accessible)
- f64↔f32 conversion at API boundary (all internal compute in f32)

**Register in lib.rs**: `#[cfg(feature = "metal")] pub mod metal_batch;`

### Step 6: Generic GpuSession + Metal Single-Model Fit (~1-2 hours)

**New file**: `crates/ns-inference/src/gpu_session.rs`

Extract `GpuSession` from `gpu_single.rs` and make it generic:

```rust
pub struct GpuSession<A: GpuAccelerator> {
    accel: RefCell<A>,
    n_params: usize,
    n_main_bins: usize,
}

impl<A: GpuAccelerator> GpuSession<A> {
    pub fn from_accelerator(accel: A) -> Self { ... }
    pub fn upload_observed(&self, model: &HistFactoryModel) -> Result<()> { ... }
    pub fn nll_grad(&self, params: &[f64]) -> Result<(f64, Vec<f64>)> { ... }
    pub fn fit_minimum(...) -> Result<OptimizationResult> { ... }
    pub fn fit_minimum_from(...) -> Result<OptimizationResult> { ... }
    pub fn fit_minimum_from_with_bounds(...) -> Result<OptimizationResult> { ... }
}
```

**`GpuObjective<A>`**: same fused NLL+gradient caching pattern, generic over accelerator.

**Update `gpu_single.rs`**: replace inline `GpuSession` with:
```rust
pub type GpuSession = crate::gpu_session::GpuSession<CudaBatchAccelerator>;

pub fn new_cuda_session(model: &HistFactoryModel) -> Result<GpuSession> {
    let gpu_data = model.serialize_for_gpu()?;
    let mut accel = CudaBatchAccelerator::from_gpu_data(&gpu_data, 1)?;
    // upload observed...
    Ok(crate::gpu_session::GpuSession::from_accelerator(accel))
}
```

**Add Metal session factory** (in `gpu_session.rs` or `metal_single.rs`):
```rust
#[cfg(feature = "metal")]
pub fn new_metal_session(model: &HistFactoryModel) -> Result<GpuSession<MetalBatchAccelerator>> {
    let gpu_data = model.serialize_for_gpu()?;
    let metal_data = MetalModelData::from_gpu_data(&gpu_data);
    let mut accel = MetalBatchAccelerator::from_metal_data(&metal_data, 1)?;
    // upload observed (f64→f32 at boundary)...
    Ok(GpuSession::from_accelerator(accel))
}
```

**Update `mle.rs`**: GPU methods become device-parameterized:
- `fit_minimum_gpu()` stays (CUDA), add `fit_minimum_metal()` that calls same generic path
- Or better: internal `fit_minimum_gpu_session<A>(&self, session: &GpuSession<A>)` shared by both

**Update `profile_likelihood.rs`**: `scan_gpu` becomes generic, add `scan_metal` entry point.

**Register**: in `ns-inference/lib.rs`.

### Step 7: Metal Batch Fitting Entry Point (~1-2 hours)

**New file**: `crates/ns-inference/src/metal_batch.rs`

Mirrors `gpu_batch.rs::fit_toys_batch_gpu()`:
1. Serialize model → `GpuModelData` → `MetalModelData` (f64→f32)
2. Create `MetalBatchAccelerator`
3. Generate toys on CPU (Rayon), precompute ln_facts/obs_mask
4. Upload observed data (single transfer)
5. Lockstep loop: compact active params → GPU batch_nll_grad → L-BFGS-B step
6. Collect `FitResult`s

**Key difference from CUDA**: `config.tol = config.tol.max(1e-4)` — relaxed for f32.

**Register**: `#[cfg(feature = "metal")] pub mod metal_batch;` in `ns-inference/lib.rs`.

**batch.rs**: Add `is_metal_batch_available()` with cfg gates (same pattern as `is_cuda_batch_available`).

### Step 8: CLI Integration (~1 hour)

Change `--gpu` from `bool` to `Option<String>` on **all three commands**:

```
# fit command
nextstat fit --gpu cuda     # NVIDIA GPU (f64)
nextstat fit --gpu metal    # Apple Silicon GPU (f32)

# scan command
nextstat scan --gpu cuda
nextstat scan --gpu metal

# hypotest-toys command
nextstat hypotest-toys --gpu cuda
nextstat hypotest-toys --gpu metal
```

Dispatch by device string with `#[cfg(feature = "...")]` guards. For `fit` and `scan`, create the appropriate session type and call generic GPU methods. For `hypotest-toys`, call `fit_toys_batch_gpu()` or `fit_toys_batch_metal()`.

### Step 9: Python Bindings (~1 hour)

- Add `has_metal() -> bool` function
- Update `fit_toys_batch_gpu()`: accept `device="metal"` alongside `device="cuda"`
- Update `fit(..., device=)`: accept `device="metal"` (currently only `"cuda"`)
- Update `profile_scan(..., device=)`: accept `device="metal"`
- Register `has_metal` in module init

All Python GPU functions gain Metal support via `device="metal"` parameter.

### Step 10: Tests & Validation (~3 hours)

**Unit tests** (in `metal_batch.rs`):
- `test_metal_available` — returns true on macOS
- `test_metal_accelerator_creation` — minimal model, verify no crash

**Integration tests — Batch** (in ns-inference `metal_batch.rs`):
- `test_metal_batch_smoke` — fit 2 toys on simple_workspace, verify finite NLL + convergence
- `test_metal_batch_matches_cpu` — same seed, verify NLL diff < 1e-3, param diff < 1e-2

**Integration tests — Single fit** (in ns-inference `gpu_session.rs`):
- `test_metal_single_nll_grad` — verify NLL matches CPU within f32 tolerance
- `test_metal_single_fit` — fit simple_workspace, verify convergence + params close to CPU
- `test_metal_profile_scan` — 5-point scan, verify PLR shape

**Generic GpuSession tests**:
- `test_gpu_session_caching` — verify GpuObjective cache hit on same params
- `test_gpu_session_cuda_still_works` — regression test, CUDA path unchanged

**Python tests**:
- `test_has_metal()` — returns bool
- `test_metal_batch_fit()` — 2 toys, verify convergence
- `test_metal_single_fit()` — `fit(device="metal")`, verify convergence
- `test_metal_profile_scan()` — `profile_scan(device="metal")`, verify shape
- `test_metal_matches_cpu()` — f32 vs f64 parity within tolerance

### Step 11: Documentation (~30 min)

- Update README GPU section with Metal option
- Update `docs/references/rust-api.md` and `docs/references/python-api.md`
- Update CHANGELOG.md

---

## Verification

```bash
# 1. Feature chain compiles
cargo check --features metal -p ns-compute
cargo check --features metal -p ns-translate
cargo check --features metal -p ns-inference
cargo check --features metal -p ns-cli

# 2. MSL compiles
xcrun -sdk macosx metal -c crates/ns-compute/kernels/batch_nll_grad.metal

# 3. Tests pass
cargo test --features metal -p ns-compute -p ns-inference

# 4. Single-model fit smoke test
cargo run --release --features metal -p ns-cli -- fit \
  --input tests/fixtures/simple_workspace.json --gpu metal

# 5. Profile scan smoke test
cargo run --release --features metal -p ns-cli -- scan \
  --input tests/fixtures/simple_workspace.json --start 0 --stop 5 --points 11 --gpu metal

# 6. Batch toy smoke test
cargo run --release --features metal -p ns-cli -- hypotest-toys \
  --input tests/fixtures/simple_workspace.json --mu 1.0 --n-toys 100 --gpu metal

# 7. Python
cd bindings/ns-py && maturin develop --release --features metal
python -c "import nextstat; print('Metal:', nextstat.has_metal())"
python -c "import nextstat; r = nextstat.fit('tests/fixtures/simple_workspace.json', device='metal'); print('NLL:', r.nll)"
```

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| `atomic_float` not available on older GPUs | Require Apple GPU family 7+ (M1+), document requirement |
| f32 precision insufficient for large models | Validate on tHu (184 params) and tttt (249 params); fallback: Kahan summation in kernel |
| `metal` crate API differences from examples | Use `metal` 0.33 (stable API), test on macOS 15+ |
| MSL compiler rejects kernel | Pre-validate with `xcrun metal -c` before Rust integration |
| Linter strips dead code | Add all new functions + callers atomically |
| GpuAccelerator trait refactor breaks CUDA | Run existing CUDA tests after trait extraction (Step 5) |
| GpuSession generification ripple effects | Type alias `type CudaGpuSession = GpuSession<CudaBatchAccelerator>` preserves existing API |
| NormSys non-positive factors (hi≤0 or lo≤0) | `serialize_for_gpu()` returns `Err(Validation)` — GPU polynomial kernel cannot represent CPU piecewise-linear fallback. No real workspace affected (0/8+ fixtures). |

---

## CUDA→Metal Translation Reference

### Struct mapping
| CUDA (f64) | Metal (f32) | Notes |
|------------|-------------|-------|
| `GpuSampleInfo` | `MetalSampleInfo` | Same (all u32) |
| `GpuModifierDesc` | `MetalModifierDesc` | Same (all u32/u8) |
| `GpuAuxPoissonEntry` | `MetalAuxPoissonEntry` | f32 + lgamma precomputed |
| `GpuGaussConstraintEntry` | `MetalGaussConstraintEntry` | f32 center/inv_width |
| `GpuModelData` | `MetalModelData` | All Vec<f32> |
| `CudaBatchAccelerator` | `MetalBatchAccelerator` | Same API, f64↔f32 at boundary |

### API mapping
| cudarc | metal crate |
|--------|-------------|
| `CudaContext::new(0)` | `Device::system_default()` |
| `ctx.default_stream()` | `device.new_command_queue()` |
| `stream.clone_htod()` | `device.new_buffer_with_data()` |
| `stream.alloc_zeros()` | `device.new_buffer(size, opts)` |
| `Ptx::from_src()` | `device.new_library_with_source()` |
| `module.load_function()` | `library.get_function()` |
| `stream.launch_builder()` | `cmd_buf.new_compute_command_encoder()` |
| `builder.arg(&buffer)` | `encoder.set_buffer(idx, buf, 0)` |
| `builder.launch(config)` | `encoder.dispatch_threadgroups(grid, tg)` |
| `stream.memcpy_dtoh()` | `buffer.contents()` (unified memory) |
| `stream.synchronize()` | `cmd_buf.wait_until_completed()` |
