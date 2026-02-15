# Unbinned Spec Reference (`nextstat_unbinned_spec_v0`)

Declarative YAML/JSON configuration for event-level (unbinned) extended likelihood models.

**JSON Schema:** [`docs/schemas/unbinned/unbinned_spec_v0.schema.json`](../schemas/unbinned/unbinned_spec_v0.schema.json)

**CLI usage:**

```bash
nextstat unbinned-fit --config model.yaml [--gpu cuda|metal]
nextstat unbinned-scan --config model.yaml --start 0 --stop 5 --points 21
nextstat unbinned-fit-toys --config model.yaml --n-toys 1000 --seed 42
nextstat unbinned-hypotest --config model.yaml --mu 1.0
nextstat unbinned-hypotest-toys --config model.yaml --mu 1.0 --n-toys 1000 --seed 42
nextstat unbinned-ranking --config model.yaml
```

---

## Top-Level Structure

```yaml
$schema: https://nextstat.io/schemas/unbinned/unbinned_spec_v0.schema.json  # optional, IDE support
schema_version: nextstat_unbinned_spec_v0
model:
  poi: mu                    # optional parameter-of-interest name
  parameters: [...]
channels: [...]
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `$schema` | string | no | JSON Schema URI for IDE validation |
| `schema_version` | string | **yes** | Must be `nextstat_unbinned_spec_v0` |
| `model` | object | **yes** | Model-level settings (parameters, POI) |
| `channels` | array | **yes** | One or more analysis channels |

---

## `model`

```yaml
model:
  poi: mu
  parameters:
    - name: mu
      init: 1.0
      bounds: [0.0, 10.0]
    - name: alpha_jes
      init: 0.0
      bounds: [-5.0, 5.0]
      constraint:
        type: gaussian
        mean: 0.0
        sigma: 1.0
```

### `model.parameters[]`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **yes** | Unique parameter name |
| `init` | number | **yes** | Initial value for the optimizer |
| `bounds` | `[lo, hi]` | **yes** | Optimizer bounds |
| `constraint` | object | no | Nuisance prior (see below) |

### Constraint types

| Type | Fields | Math |
|------|--------|------|
| `gaussian` | `mean`, `sigma` | $-\ln \mathcal{N}(\theta \mid \mu, \sigma)$ added to NLL |

---

## `channels[]`

Each channel defines: observed data (ROOT TTree or Parquet), observables, and processes.

```yaml
channels:
  - name: SR
    include_in_fit: true    # default: true
    data:
      file: data.root
      tree: nominal
      selection: "pt > 25 && abs(eta) < 2.5"   # optional
      weight: weight_mc                          # optional
    observables:
      - name: mass
        expr: dimuon_mass    # optional, defaults to name
        bounds: [60.0, 120.0]
    processes: [...]
```

### `data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | path | **yes** | Data source path (ROOT or Parquet). Absolute or relative to the spec file. |
| `channel` | string | Parquet only | Optional channel selector for multi-channel Parquet files (filters rows to `_channel == channel`). |
| `tree` | string | ROOT only | TTree name (required for ROOT sources; omitted for Parquet). |
| `selection` | string | ROOT only | Boolean cut expression (ns-root expression engine). |
| `weight` | string | ROOT only | Per-event weight expression. Applied as a **frequency weight** `w_i` multiplying each event’s `-log L` contribution. Must be finite and `>= 0` (and not all-zero). For Parquet sources, embed weights in the Parquet file (e.g. `nextstat convert --weight ...`). |

#### Observed weights policy

NextStat enforces a **non-negative frequency weight** contract for observed data:

| Requirement | Enforced at |
|---|---|
| `w_i` must be finite | `EventStore`, CPU NLL, CUDA, Metal |
| `w_i >= 0` | `EventStore`, CPU NLL, CUDA, Metal |
| `sum(w_i) > 0` | `EventStore`, CPU NLL, CUDA, Metal |

**Rationale.** The weighted extended NLL is `NLL = Σ_j ν_j − Σ_i w_i log f(x_i; θ)`. When all `w_i = 1` this reduces to the standard extended MLE. Non-negative frequency weights arise naturally from detector efficiency corrections, luminosity reweighting, or MC truth-matching. Negative weights (e.g. from subtraction-based background estimation) break the probabilistic interpretation of the likelihood and can cause the optimizer to diverge; they are therefore **rejected at ingest time**.

**When to use weights:**
- MC-to-data reweighting (pileup, trigger efficiency).
- Luminosity scaling across run periods.
- Pre-computed selection efficiency corrections.

**When NOT to use weights:**
- Background subtraction (use sPlot or signal+background model instead).
- Signed MC generator weights (aggregate to positive per-event weights before ingestion, or model the sign separately).

**Effective sample size.** When weights vary significantly, the effective sample size `N_eff = (Σ w_i)² / Σ w_i²` can be much smaller than `N`. The CLI reports `N_eff` in the fit summary when weights are present. Large `max(w_i) / min(w_i > 0)` ratios (> 100) trigger a diagnostic warning.

### `observables[]`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **yes** | Observable name (used by PDFs) |
| `expr` | string | no | TTree expression. Defaults to `name` if omitted |
| `bounds` | `[lo, hi]` | **yes** | Support region Ω for normalization |

---

## `processes[]`

Each process has a PDF (shape) and a yield (expected event count).

```yaml
processes:
  - name: signal
    pdf:
      type: gaussian
      observable: mass
      params: [mu_sig, sigma_sig]
    yield:
      type: scaled
      base_yield: 100.0
      scale: mu
```

---

## PDF Types

### Parametric PDFs

| Type | `params` | Description |
|------|----------|-------------|
| `gaussian` | `[mu, sigma]` | Gaussian with analytic normalization |
| `exponential` | `[lambda]` | Exponential on bounded support |
| `argus` | `[c, p]` | ARGUS background shape (cutoff taken from observable upper bound). CPU-only (no `--gpu`). |
| `crystal_ball` | `[mu, sigma, alpha, n]` | Crystal Ball (single tail) |
| `double_crystal_ball` | `[mu, sigma, alpha_l, n_l, alpha_r, n_r]` | Double Crystal Ball |
| `chebyshev` | `[c_1, ..., c_order]` | Chebyshev polynomial (order = len). GPU: order ≤ 16 |
| `voigtian` | `[mu, sigma, gamma]` | Voigtian (Gaussian ⊗ Breit-Wigner), normalized on observable bounds. CPU-only (no `--gpu`). |

All parametric PDFs are auto-normalized over the observable bounds.

### Non-parametric PDFs (inline)

```yaml
pdf:
  type: histogram
  observable: mass
  bin_edges: [60, 70, 80, 90, 100, 110, 120]
  bin_content: [10, 50, 200, 180, 40, 15]
  pseudo_count: 0.01   # optional
```

```yaml
pdf:
  type: kde
  observable: mass
  bandwidth: 2.0
  centers: [65, 75, 85, 95, 105, 115]
  weights: [0.1, 0.5, 2.0, 1.8, 0.4, 0.15]   # optional
```

```yaml
pdf:
  type: spline
  observable: mass
  knots_x: [60, 70, 80, 90, 100, 110, 120]
  knots_y: [0.1, 0.5, 2.0, 1.8, 0.4, 0.15]
```

### PDF Composition (CPU-only)

`product` builds a multi-observable PDF by multiplying independent 1D component PDFs:

`p(x, y, ...) = p₁(x) · p₂(y) · ...`

Notes:
- Components must use **disjoint** observables.
- CPU-only (no `--gpu`).
- Current limitation: components must be **inline** PDFs (no `*_from_tree` and no neural PDFs).

```yaml
pdf:
  type: product
  components:
    - type: gaussian
      observable: x
      params: [x_mu, x_sigma]
    - type: exponential
      observable: y
      params: [y_lambda]
```

### Non-parametric PDFs (from ROOT TTree)

```yaml
pdf:
  type: histogram_from_tree
  observable: mass
  bin_edges: [60, 70, 80, 90, 100, 110, 120]
  pseudo_count: 0.01
  source:
    file: mc_bkg.root
    tree: nominal
    selection: "is_bkg == 1"
    weight: weight_mc
  max_events: 100000   # optional cap
  weight_systematics: [...]     # optional (see below)
  horizontal_systematics: [...]  # optional (see below)
```

```yaml
pdf:
  type: kde_from_tree
  observable: mass
  bandwidth: 2.0
  source:
    file: mc_bkg.root
    tree: nominal
  weight_systematics: [...]
  horizontal_systematics: [...]
```

GPU note:
- `histogram_from_tree` with `--gpu` is currently supported only as a pre-materialized histogram shape:
  - `horizontal_systematics` must be empty.
  - `weight_systematics` may be used only with `apply_to_shape: false` (yield-only).
- `kde_from_tree` remains CPU-only.

### Neural PDFs (feature `neural`)

Requires `--features neural` at build time. Uses ONNX Runtime via the `ort` crate.

```yaml
# Unconditional normalizing flow
pdf:
  type: flow
  manifest: models/signal_flow/flow_manifest.json

# Conditional flow (context = nuisance parameters)
pdf:
  type: conditional_flow
  manifest: models/signal_flow/flow_manifest.json
  context_params: [alpha_syst1, alpha_syst2]

# DCR surrogate (drop-in replacement for morphing histogram)
pdf:
  type: dcr_surrogate
  manifest: models/bkg_dcr/flow_manifest.json
  systematics: [jes_alpha, jer_alpha]
```

| Type | Fields | Description |
|------|--------|-------------|
| `flow` | `manifest` | Unconditional ONNX normalizing flow |
| `conditional_flow` | `manifest`, `context_params` | Conditional flow (nuisance → context vector) |
| `dcr_surrogate` | `manifest`, `systematics` | Neural Direct Classifier Ratio surrogate |

The `manifest` path points to a `flow_manifest.json` file following the `nextstat_flow_v0` schema.

---

## Per-event Systematics (shape-level)

These live inside `histogram_from_tree` and `kde_from_tree` PDF specs. They modify the **shape** (and optionally the yield) of the PDF by reading per-event up/down weight ratios or observable shifts from the source TTree.

### `weight_systematics[]`

Per-event weight reweighting. At `α = ±1`, the up/down weight ratios morph the template.

```yaml
weight_systematics:
  - param: alpha_jes
    up: "weight_jes_up / weight_mc"
    down: "weight_jes_down / weight_mc"
    interp: code4p          # default: code0
    apply_to_shape: true    # default: true
    apply_to_yield: true    # default: true
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `param` | string | **yes** | — | Nuisance parameter name α |
| `up` | string | **yes** | — | Expression for `w_up / w_nom` per event |
| `down` | string | **yes** | — | Expression for `w_down / w_nom` per event |
| `interp` | `code0` \| `code4p` | no | `code0` | HistFactory-style interpolation code |
| `apply_to_shape` | bool | no | `true` | Morph PDF shape via template weights |
| `apply_to_yield` | bool | no | `true` | Apply yield rate modifier from total weights |

When `apply_to_shape: true`:
- **histogram_from_tree** → `MorphingHistogramPdf` (per-bin mass morphing)
- **kde_from_tree** → `MorphingKdePdf` (per-kernel weight morphing)

When `apply_to_yield: true`: a `RateModifier::WeightSys` is automatically generated from the total up/down weight sums.

### `horizontal_systematics[]`

Observable-shift systematics: at `α = ±1`, the observable is evaluated using a different expression.

```yaml
horizontal_systematics:
  - param: alpha_jes
    up: "jet_pt * 1.03"
    down: "jet_pt * 0.97"
    interp: code4p
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `param` | string | **yes** | — | Nuisance parameter name α |
| `up` | string | **yes** | — | Observable expression at α = +1 |
| `down` | string | **yes** | — | Observable expression at α = -1 |
| `interp` | `code0` \| `code4p` | no | `code0` | HistFactory-style interpolation code |

For `histogram_from_tree`: shifts the histogram bin filling.
For `kde_from_tree`: shifts the KDE kernel centers (`HorizontalMorphingKdePdf`).

---

## Yield Expressions

The expected event count ν for each process.

### `fixed`

```yaml
yield:
  type: fixed
  value: 500.0
  modifiers: [...]   # optional
```

### `parameter`

Yield is a free parameter (directly optimized).

```yaml
yield:
  type: parameter
  name: n_bkg
  modifiers: [...]
```

### `scaled`

Signal-strength style: `ν = base_yield × μ`.

```yaml
yield:
  type: scaled
  base_yield: 100.0
  scale: mu
  modifiers: [...]
```

---

## Yield Rate Modifiers

Optional multiplicative modifiers on the yield, specified in `yield.modifiers[]`.

### `normsys`

HistFactory-like normalization systematic: `ν → ν × f(α)` where `f` uses piecewise exponential interpolation (`hi^α` when α > 0, `lo^{-α}` when α < 0).

```yaml
modifiers:
  - type: normsys
    param: alpha_lumi
    lo: 0.98
    hi: 1.02
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `param` | string | **yes** | Nuisance parameter name |
| `lo` | number | **yes** | Yield factor at α = -1 (must be > 0) |
| `hi` | number | **yes** | Yield factor at α = +1 (must be > 0) |

### `weightsys`

HistFactory-like template interpolation on a scalar yield factor: `ν → ν × f(α)` where `f` interpolates between `(lo, 1, hi)` using code0 (piecewise linear) or code4p (smooth polynomial).

```yaml
modifiers:
  - type: weightsys
    param: alpha_jes
    lo: 0.92
    hi: 1.08
    interp_code: code4p   # optional, default: code0
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `param` | string | **yes** | — | Nuisance parameter name |
| `lo` | number | **yes** | — | Yield factor at α = -1 (must be > 0) |
| `hi` | number | **yes** | — | Yield factor at α = +1 (must be > 0) |
| `interp_code` | `code0` \| `code4p` | no | `code0` | Interpolation code |

---

## GPU Acceleration

The `--gpu cuda|metal` flag offloads NLL + gradient computation to GPU. A **conservative subset** of the spec is supported:

| Feature | GPU support |
|---------|-------------|
| Data sources | ROOT and Parquet (selection/weight expressions: ROOT only) |
| Observables | 1D per channel |
| PDFs | `gaussian`, `exponential`, `crystal_ball`, `double_crystal_ball`, `chebyshev` (order ≤ 16), `histogram` |
| Yields | `fixed`, `parameter`, `scaled` |
| Yield modifiers | `normsys`, `weightsys` |
| Constraints | `gaussian` |
| Multi-channel | yes |
| Per-event weights | yes (frequency weights) |
| Non-parametric PDFs | `histogram` only (e.g. `kde`/`spline` are CPU-only) |
| Neural PDFs | GPU flow NLL reduction via `cuda_flow_nll` (separate path) |

Unsupported specs produce an explicit error directing users to the CPU path.

### GPU toy sampling

```bash
nextstat unbinned-fit-toys --config model.yaml --n-toys 1000 --gpu cuda --gpu-sample-toys
nextstat unbinned-fit-toys --config model.yaml --n-toys 1000 --gpu metal --gpu-sample-toys
nextstat unbinned-fit-toys --config model.yaml --n-toys 1000 --gpu cuda --gpu-devices 0,1
nextstat unbinned-fit-toys --config model.yaml --n-toys 1000 --gpu cuda --gpu-shards 4
nextstat unbinned-fit-toys --config model.yaml --n-toys 1000 --gpu cuda --gpu-sample-toys --gpu-shards 4
```

`--gpu-sample-toys` samples toys on the GPU (Gaussian, Exponential, CrystalBall, DoubleCrystalBall, Chebyshev, Histogram; 1D channels only).

- `--gpu auto` (policy-gated) selects CPU vs CUDA based on model topology and estimated events/toy and logs the decision reason. Use `--gpu cuda|metal` for explicit control.

- **CUDA:** device-resident toy→fit pipeline keeps `obs_flat` on the GPU between sampling and batch fitting (eliminates the D2H+H2D round-trip).
- **CUDA multi-GPU (host toys):** `--gpu-devices 0,1,...` shards CPU-sampled toys across selected devices and merges toy results in original order.
- **CUDA sharded host-toy pipeline:** `--gpu-shards N` (without `--gpu-sample-toys`) shards host-sampled toys into `N` logical shards (`pipeline = cuda_host_sharded`), assigned to `--gpu-devices` round-robin.
- **CUDA sharded device-resident pipeline:** `--gpu-sample-toys --gpu-shards N` splits toys into `N` logical shards; shards are assigned to `--gpu-devices` round-robin. This also enables single-GPU emulation of multi-shard orchestration.
- **Metal:** device-resident toy→fit pipeline keeps `obs_flat` as a Metal shared `Buffer` (f32) between sampling and batch fitting, eliminating the f32→f64→f32 conversion round-trip. Uses `sample_toys_1d_device()` → `from_unbinned_static_and_toys_device()` when `--gpu-sample-toys` is active.

---

## Full Example

```yaml
$schema: https://nextstat.io/schemas/unbinned/unbinned_spec_v0.schema.json
schema_version: nextstat_unbinned_spec_v0

model:
  poi: mu
  parameters:
    - name: mu
      init: 1.0
      bounds: [0.0, 10.0]
    - name: mu_sig
      init: 91.2
      bounds: [88.0, 94.0]
    - name: sigma_sig
      init: 2.5
      bounds: [0.5, 5.0]
    - name: lambda_bkg
      init: -0.02
      bounds: [-0.1, -0.001]
    - name: alpha_lumi
      init: 0.0
      bounds: [-5.0, 5.0]
      constraint: { type: gaussian, mean: 0.0, sigma: 1.0 }
    - name: alpha_jes
      init: 0.0
      bounds: [-5.0, 5.0]
      constraint: { type: gaussian, mean: 0.0, sigma: 1.0 }

channels:
  - name: SR
    data:
      file: data.root
      tree: nominal
      selection: "pass_sr == 1"
    observables:
      - name: mass
        bounds: [60.0, 120.0]
    processes:
      - name: signal
        pdf:
          type: gaussian
          observable: mass
          params: [mu_sig, sigma_sig]
        yield:
          type: scaled
          base_yield: 50.0
          scale: mu
          modifiers:
            - type: normsys
              param: alpha_lumi
              lo: 0.98
              hi: 1.02
            - type: weightsys
              param: alpha_jes
              lo: 0.95
              hi: 1.05
              interp_code: code4p
      - name: background
        pdf:
          type: exponential
          observable: mass
          params: [lambda_bkg]
        yield:
          type: fixed
          value: 500.0
          modifiers:
            - type: normsys
              param: alpha_lumi
              lo: 0.98
              hi: 1.02
```
