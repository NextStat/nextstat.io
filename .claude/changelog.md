# Changelog Writing Rules

## Structure

```
## [Unreleased]
### Added
#### Section Name        ← functional area, NOT "Phase N"
- Feature description    ← user-facing, 1-2 lines max
### Fixed
- Bug description
---
## [X.Y.Z] — YYYY-MM-DD
```

## Section Names (Unreleased)

Use these exact section names. Do NOT invent new ones without reason:

| Section | What goes here |
|---|---|
| **GPU Acceleration** | CUDA, Metal, Accelerate, batch toys, tape optimization |
| **Differentiable Analysis (PyTorch)** | DifferentiableSession, torch module, zero-copy kernels |
| **Gymnasium RL Environment** | nextstat.gym, RL/DOE environments |
| **Deterministic Validation** | EvalMode, parity tests, tolerance contracts |
| **Native ROOT I/O** | TTree reader, expression engine, histogram filler |
| **Ntuple-to-Workspace Pipeline** | NtupleWorkspaceBuilder, ROOT→Workspace flow |
| **TRExFitter Interop** | Config import, HIST mode, Analysis Spec, build-hists |
| **Systematics Preprocessing** | Smoothing, pruning, preprocess CLI, caching |
| **HistFactory Enhancements** | Interpolation codes, patchset support, model features |
| **Report System** | `nextstat report`, artifacts, rendering, blinding |
| **Survival Analysis** | Parametric, Cox PH, Schoenfeld, survival CLI |
| **Linear Mixed Models** | LMM, Laplace approximation |
| **Ordinal Models** | Ordered logit/probit |
| **Econometrics & Causal Inference** | Panel FE, DiD, IV/2SLS, AIPW, propensity |
| **Pharmacometrics** | ODE, PK/NLME |
| **Applied Statistics API** | Formula, from_formula, robust SE, sklearn adapters |
| **WASM Playground** | Browser-based inference |
| **Visualization** | Plot functions, viz subcommands |
| **CLI & Infrastructure** | Logging, CI, validation, release pipeline |

## Rules

### DO
- **One line per feature** — what the user gets, not how it's implemented internally
- **Start with the user-facing API** — CLI flag, Python function, or behavior change
- **Use em-dash (—)** to separate the feature name from its description
- **Bold** the key noun in list items when there are multiple items at same level
- **Include CLI/Python examples** inline: `--gpu cuda`, `device="metal"`
- **Keep Fixed section flat** — one line per bug, no sub-sections

### DO NOT
- "Phase N" — internal dev milestone numbers mean nothing to users
- Internal module paths (`ns-compute::cuda_batch`, `ns-inference::gpu_single`)
- Rust implementation details (`#[inline]`, `RefCell<GpuCache>`, feature chains)
- File paths inside the codebase (`crates/ns-compute/kernels/batch_nll_grad.cu`)
- Build system internals (`build.rs`, `cargo:rustc-link-lib`)
- Test counts ("5 new unit tests") — tests are expected, not a feature
- Benchmark tables — those go in README or docs, not changelog
- Sub-sub-sections (H4) — keep it flat: H2 (version) → H3 (area) → list

### Line Format Examples

Good:
```
- **CUDA (NVIDIA, f64)** — fused NLL+gradient kernel covering all 7 modifier types.
  `nextstat.fit(model, device="cuda")`, `--gpu cuda` CLI flag.
- `nextstat report` — generates distributions, pulls, correlations, yields, and uncertainty ranking.
- `--blind` flag masks observed data for unblinded regions.
```

Bad:
```
- `CudaBatchAccelerator` in `ns-compute::cuda_batch` manages GPU buffers   ← internal module path
- PTX build system: `build.rs` compiles via `nvcc --ptx -arch=sm_70`       ← build system noise
- 5 new unit tests: scalar parity, zero-delta, Code0/Code4p agreement      ← test inventory
- `par_iter().map_init()` pattern: one Tape per Rayon worker thread         ← implementation detail
```

## When to Add Entries

- **After merging a feature branch** — not after every commit
- **After fixing a user-visible bug** — not internal refactors
- **Before cutting a release** — review [Unreleased] and move to `[X.Y.Z]`

## Release Process

1. Rename `[Unreleased]` to `[X.Y.Z] — YYYY-MM-DD`
2. Add a new empty `[Unreleased]` section at top
3. Update the website `Changelog.tsx` to match (same sections, same order)
4. Commit both repos, push, `railway up --detach`

## Syncing with Website

The website `Changelog.tsx` at `nextstat.io_web/ref/app/src/pages/docs/Changelog.tsx`
mirrors CHANGELOG.md 1:1. Same sections, same order. Use these component patterns:

```tsx
<H3>Section Name</H3>
<ul className={UL}>
  <li>Feature description</li>
</ul>
```

No `H4` sub-sections. Keep it flat.
