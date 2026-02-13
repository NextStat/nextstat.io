---
title: "Neural Density Estimation for Unbinned Fits"
description: "End-to-end guide: train normalizing flows in Python, export to ONNX, use as PDFs in NextStat unbinned likelihood fits. Covers FlowPdf, DcrSurrogate, and the FAIR-HUC training protocol."
status: stable
last_updated: 2026-02-10
keywords:
  - normalizing flow
  - neural spline flow
  - ONNX
  - unbinned likelihood
  - DCR surrogate
  - template morphing
  - FAIR-HUC
  - NextStat
---

# Neural Density Estimation for Unbinned Fits

NextStat supports **ONNX-backed normalizing flows** as PDFs in unbinned likelihood fits. This enables:

- **Learned signal/background shapes** that capture complex multi-modal distributions
- **Smooth systematic morphing** via conditional flows (replacing binned HistoSys interpolation)
- **Drop-in replacement** — `FlowPdf` and `DcrSurrogate` implement the same `UnbinnedPdf` trait as `GaussianPdf`, `CrystalBallPdf`, etc.

## Overview

```
┌────────────────────────────────────────────────────────┐
│                    Training (Python)                    │
│                                                        │
│  Data ──→ zuko NSF ──→ torch.onnx.export ──→ .onnx    │
│                                                        │
│  Output: flow_manifest.json + log_prob.onnx            │
│          (+ optional sample.onnx)                      │
└────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│                   Inference (Rust)                      │
│                                                        │
│  FlowPdf::from_manifest("flow_manifest.json", &[])    │
│       │                                                │
│       ├─ log_prob_batch(events, params, out)            │
│       ├─ log_prob_grad_batch(events, params, logp, grad)│
│       └─ sample(params, n, support, rng)               │
│                                                        │
│  Used inside UnbinnedModel for MLE / profile scan      │
└────────────────────────────────────────────────────────┘
```

## Prerequisites

```bash
# Rust: enable neural feature
cargo build -p ns-unbinned --features neural

# Python: training dependencies
pip install torch zuko onnx onnxruntime numpy scipy
```

## Quick Start: Unconditional Flow

### 1. Train and export

```bash
python scripts/neural/train_flow.py \
    --data training_data.npy \
    --observables mass \
    --support 5000 6000 \
    --output-dir models/signal_flow \
    --epochs 500
```

This produces:
- `models/signal_flow/flow_manifest.json`
- `models/signal_flow/log_prob.onnx`
- `models/signal_flow/sample.onnx`

### 2. Validate

```bash
python scripts/neural/validate_flow.py \
    --manifest models/signal_flow/flow_manifest.json \
    --data training_data.npy
```

Checks:
- **Normalization**: ∫p(x)dx ≈ 1.0 (Gauss-Legendre quadrature)
- **PIT uniformity**: Kolmogorov-Smirnov test on probability integral transform
- **Closure**: train/test NLL gap within tolerance

### 3. Use in a spec

```yaml
channels:
  - name: SR
    observables:
      - name: mass
        bounds: [5000.0, 6000.0]
    processes:
      - name: signal
        pdf:
          type: flow
          manifest: models/signal_flow/flow_manifest.json
        yield:
          type: scaled
          base_yield: 100.0
          scale: mu
```

### 4. Use in Rust directly

```rust
use ns_unbinned::{FlowPdf, UnbinnedPdf};
use ns_unbinned::event_store::{EventStore, ObservableSpec};

let flow = FlowPdf::from_manifest(
    "models/signal_flow/flow_manifest.json".as_ref(),
    &[],  // no context params (unconditional)
)?;

let obs = ObservableSpec::branch("mass", (5000.0, 6000.0));
let events = EventStore::from_columns(
    vec![obs],
    vec![("mass".into(), observed_masses)],
    None,
)?;

let mut logp = vec![0.0; events.n_events()];
flow.log_prob_batch(&events, &[], &mut logp)?;

let nll: f64 = logp.iter().map(|lp| -lp).sum();
```

## Conditional Flows (Systematic Parameters)

For PDFs that depend on nuisance parameters `p(x | α₁, α₂, ...)`:

### Training

```bash
python scripts/neural/train_flow.py \
    --data training_data.npy \
    --observables mass \
    --support 5000 6000 \
    --context jes_alpha jer_alpha \
    --context-data context_params.npy \
    --output-dir models/conditional_flow
```

### Spec

```yaml
- name: background
  pdf:
    type: conditional_flow
    manifest: models/conditional_flow/flow_manifest.json
    context_params: [jes_alpha, jer_alpha]
  yield:
    type: parameter
    name: nu_bkg
```

The `context_params` list maps model parameter names to flow context vector positions. During fitting, their current values are automatically passed to the flow.

### Gradients

`log_prob_grad_batch` computes `∂ log p(x|α) / ∂α`. Two strategies are supported:

1. **Analytical Jacobian** (preferred): When the manifest includes a `models.log_prob_grad` ONNX model, a single forward pass produces both `log p` and `∂ log p / ∂ context`. This is orders of magnitude faster than finite differences for models with many nuisance parameters.
2. **Finite differences** (fallback): Central differences with ε=1e-4. Requires `2 × n_context` extra forward passes.

## DCR Surrogate (Neural Template Morphing)

The **DCR (Direct Classifier Ratio) surrogate** replaces binned `MorphingHistogramPdf` with a conditional flow trained on HistFactory templates. Benefits:

- No binning artifacts — smooth, continuous morphing
- Works in multi-D without the curse of dimensionality
- Follows the [FAIR-HUC protocol](references/glossary.md) for surrogate validation

### Training

From a NextStat/pyhf workspace:

```bash
python scripts/neural/train_dcr.py \
    --workspace workspace.json \
    --channel SR \
    --process background \
    --output-dir models/bkg_dcr \
    --n-alpha-points 200 \
    --n-samples-per-point 5000
```

From explicit templates:

```bash
python scripts/neural/train_dcr.py \
    --templates templates.json \
    --output-dir models/bkg_dcr
```

### Template JSON format

```json
{
  "observable": "mass",
  "bin_edges": [5000, 5100, 5200, 5300, 5400, 5500],
  "nominal": [100.0, 95.0, 80.0, 60.0, 40.0],
  "systematics": [
    {
      "name": "jes_alpha",
      "up": [102.0, 94.0, 82.0, 61.0, 39.0],
      "down": [98.0, 96.0, 78.0, 59.0, 41.0],
      "interp_code": "code0"
    }
  ]
}
```

### Spec

```yaml
- name: background
  pdf:
    type: dcr_surrogate
    manifest: models/bkg_dcr/flow_manifest.json
    systematics: [jes_alpha, jer_alpha]
  yield:
    type: parameter
    name: nu_bkg
```

### Validation

The training script automatically validates the DCR surrogate against the original templates:
- **Mean |NLL diff|** — average log-likelihood difference at random α points
- **Normalization deviation** — ∫p(x|α)dx at various α values

These metrics are written to `flow_manifest.json` under the `validation` key.

## Flow Manifest Schema

Every flow is described by a `flow_manifest.json`:

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Must be `"nextstat_flow_v0"` |
| `flow_type` | string | Architecture identifier (`"nsf"`, `"dcr_nsf"`, etc.) |
| `features` | int | Number of observables |
| `context_features` | int | Number of context parameters (0 = unconditional) |
| `observable_names` | string[] | Observable names matching the spec |
| `context_names` | string[] | Context parameter names |
| `support` | float[][] | Per-observable `[lo, hi]` bounds |
| `base_distribution` | string | `"standard_normal"` |
| `models.log_prob` | string | Path to log_prob ONNX model (relative to manifest) |
| `models.sample` | string? | Path to sample ONNX model (optional) |
| `models.log_prob_grad` | string? | Path to analytical Jacobian model (optional) |
| `training` | object | Training metadata (library, epochs, loss, etc.) |
| `validation` | object? | Validation metrics (normalization, PIT, closure) |

## ONNX Model Contract

### log_prob model

- **Input**: `input` — `[batch, features + context_features]` float32
  - First `features` columns: observable values
  - Remaining columns: context parameter values (broadcast across batch)
- **Output**: `log_prob` — `[batch]` float32

### sample model (optional)

- **Input**: `input` — `[batch, features + context_features]` float32
  - First `features` columns: standard normal noise `z ~ N(0,1)`
  - Remaining columns: context parameter values
- **Output**: `samples` — `[batch, features]` float32

## Feature Gates

| Cargo feature | What it enables |
|--------------|-----------------|
| `neural` | `ort` crate + `download-binaries` + `tls-native` for ONNX Runtime |
| `neural-cuda` | CUDA Execution Provider (NVIDIA GPUs) |
| `neural-tensorrt` | TensorRT Execution Provider (optimized NVIDIA inference) |

Without `neural`, the `Flow`, `ConditionalFlow`, and `DcrSurrogate` spec variants produce a clear compile-time error message directing users to enable the feature.

## GPU Flow Evaluation

Flow PDFs can leverage GPU acceleration for NLL reduction. Two paths are supported:

### Path 1: CPU Flow + GPU NLL Reduction (Phase 3 MVP)

The flow evaluates `log p(x|θ)` on CPU via ONNX Runtime's default (CPU) execution provider.
The resulting log-prob array is uploaded to GPU where a dedicated CUDA kernel performs the
extended unbinned likelihood reduction:

```
NLL = ν_tot − Σ_events log(Σ_procs ν_p · p_p(x_i)) + constraints
```

GPU reduction is handled by `GpuFlowSession`; the caller provides the per-process `logp_flat` buffer:

```rust
use ns_inference::gpu_flow_session::{GpuFlowSession, GpuFlowSessionConfig, FlowProcessDesc};

let config = GpuFlowSessionConfig {
    processes: vec![FlowProcessDesc {
        process_index: 0,
        base_yield: 100.0,
        yield_param_idx: Some(0),  // mu parameter
        yield_is_scaled: true,     // yield = 100 * mu
        context_param_indices: vec![],
    }],
    n_events: events.n_events(),
    n_params: 1,
    n_context: 0,
    gauss_constraints: vec![],
    constraint_const: 0.0,
};

let mut session = GpuFlowSession::new(config)?;

// Evaluate flow on CPU (caller), reduce on GPU
let logp = evaluate_flow_logprob(&flow, &events, &params);
let nll = session.nll(&logp, &params)?;
```

### Path 2: CUDA EP + I/O Binding (Full GPU)

When built with `--features neural-cuda`, ONNX Runtime uses the CUDA Execution Provider.
The flow forward pass runs on GPU, and the output log-prob tensor stays device-resident.
The NLL reduction kernel reads directly from GPU memory — zero host↔device copies for the
log-prob buffer.

### Path 3: TensorRT EP (Optimized GPU)

When built with `--features neural-tensorrt`, ONNX Runtime uses the **TensorRT Execution
Provider** which compiles the ONNX graph into an optimized TensorRT engine. This provides
significantly faster inference than the CUDA EP for fixed-architecture models.

Key configuration via `FlowGpuConfig`:
- **Engine caching**: Set `engine_cache_path` to avoid recompilation on subsequent runs
- **FP16 inference**: Enabled by default for ~2× throughput
- **Dynamic batch profiles**: `profile_min_batch` / `profile_opt_batch` / `profile_max_batch`

```rust
use ns_unbinned::FlowPdf;
use ns_unbinned::pdf::flow::FlowGpuConfig;

let gpu_config = FlowGpuConfig {
    fp16: true,
    engine_cache_path: Some("/tmp/trt_cache".into()),
    profile_opt_batch: 50_000,
    ..Default::default()
};

let flow = FlowPdf::from_manifest_with_config(
    "flow_manifest.json".as_ref(),
    &[],
    &gpu_config,
)?;
```

The EP selection follows a **fallback chain**: TensorRT → CUDA → CPU.
Use `flow.gpu_ep_kind()` to check which EP was loaded at runtime.

### Gradient

The `GpuFlowSession` supports two gradient modes:

1. **Analytical** (`nll_grad_analytical`): Uses the analytical Jacobian ONNX model +
   a fused CUDA kernel (`flow_nll_grad_reduce`) to compute NLL and gradient intermediates
   in a single GPU launch. The full gradient is assembled on the CPU from the kernel outputs.
   Requires only **1 forward pass** per process.

2. **Finite differences** (`nll_grad`): Central differences (ε=1e-4).
   Each gradient evaluation requires `2·n_params + 1` NLL calls.

### When to use GPU

| Scenario | Recommended path |
|----------|-----------------|
| N < 10,000 events, few params | CPU only (flow + NLL) |
| N > 50,000 events, multi-process model | CPU flow + GPU NLL (Path 1) |
| N > 100,000 events, NVIDIA GPU available | CUDA EP + GPU NLL (Path 2) |
| N > 100,000 events, TensorRT installed | TensorRT EP + GPU NLL (Path 3) |
| Batch toy generation (1000+ toys) | GPU NLL with batch kernel |

## Batch Toy Fitting (Flow PDFs on CUDA)

For hypothesis tests and coverage studies, NextStat supports **batch toy fitting** where
thousands of toy datasets are fit simultaneously on GPU. Each toy gets one CUDA block;
the optimizer runs in lockstep across all toys.

### Single GPU

```rust
use ns_inference::unbinned_gpu_batch::fit_flow_toys_batch_cuda;
use ns_compute::cuda_flow_batch::{FlowBatchNllConfig, FlowBatchProcessDesc};

let config = FlowBatchNllConfig {
    total_events: toy_offsets[n_toys] as usize,
    n_toys,
    toy_offsets: toy_offsets.clone(),
    processes: vec![FlowBatchProcessDesc {
        base_yield: 100.0,
        yield_param_idx: Some(0),
        yield_is_scaled: true,
    }],
    n_params: 1,
    gauss_constraints: vec![],
    constraint_const: 0.0,
};

// logp_flat: [n_procs × total_events] — pre-computed log-prob for all toy events
let (results, _accel) = fit_flow_toys_batch_cuda(
    &config, &logp_flat, &[1.0], &[(0.0, 10.0)], None,
)?;
```

### Multi-GPU

```rust
use ns_inference::unbinned_gpu_batch::{shard_flow_toys, fit_flow_toys_batch_multi_gpu};

let n_gpus = 4;
let shards = shard_flow_toys(&config, &logp_flat, n_gpus);

let (configs, logps, dev_ids): (Vec<_>, Vec<_>, Vec<_>) =
    shards.into_iter().map(|(c, l, d)| (c, l, d)).multiunzip();

let results = fit_flow_toys_batch_multi_gpu(
    configs, logps, &dev_ids, &[1.0], &[(0.0, 10.0)], None,
)?;
```

Toys are sharded evenly across devices. Each GPU runs its shard independently via
`fit_lockstep`. Results are collected in original toy order.

## Architecture Notes

- **Interior mutability**: `FlowPdf` wraps `ort::Session` in `Mutex` because `ort` 2.x `run()` requires `&mut self`, while `UnbinnedPdf` trait methods take `&self`.
- **No `download-binaries` in library**: The `ns-unbinned` *library* crate delegates ONNX Runtime provisioning to the end-user binary. The `neural` feature in `ns-unbinned/Cargo.toml` enables `download-binaries` for convenience, but downstream crates can override this.
- **Normalization**: Well-trained flows produce ∫p≈1 by construction (change-of-variables formula). `FlowPdf` verifies via `QuadratureGrid::auto()` which selects the optimal strategy per dimensionality: Gauss-Legendre (1-3D), low-order tensor product N16/N8 (4-5D), Halton quasi-Monte Carlo (6D+). Results are cached in `NormalizationCache` (parameter-keyed, 6-digit precision) to avoid recomputation. A correction is applied if the deviation exceeds 0.1%.
- **Analytical gradients**: When the `log_prob_grad` ONNX model is available, `FlowPdf` computes gradients analytically via a single forward pass. The fused CUDA kernel `flow_nll_grad_reduce` computes NLL + gradient intermediates in one launch. Fallback: central finite differences (ε=1e-4).
- **GPU EP fallback chain**: `FlowPdf` tries TensorRT EP → CUDA EP → CPU, selecting the best available at runtime. The `FlowGpuEpKind` enum tracks which EP is active.

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/neural/train_flow.py` | Train unconditional/conditional NSF, export ONNX |
| `scripts/neural/train_dcr.py` | Train DCR surrogate from HistFactory templates |
| `scripts/neural/validate_flow.py` | Normalization, PIT, closure validation |
| `scripts/neural/generate_test_fixtures.py` | Generate ONNX test fixtures for CI |
| `scripts/neural/generate_dcr_test_fixtures.py` | Generate DCR test fixtures for CI |

## Testing and CI Fixtures

This repo commits small ONNX fixtures for **deterministic CI** under:

- `tests/fixtures/flow_test` (unconditional `FlowPdf` sanity + end-to-end unbinned MLE/toys)
- `tests/fixtures/dcr_test` (1D Gaussian DCR surrogate with shift δ=0.5·α)

Regenerate fixtures (only needed if the fixture format changes):

```bash
python scripts/neural/generate_test_fixtures.py
python scripts/neural/generate_dcr_test_fixtures.py
```

Run the integration tests locally:

```bash
cargo test -p ns-unbinned --features neural --test flow_integration
cargo test -p ns-unbinned --features neural --test dcr_integration
```
