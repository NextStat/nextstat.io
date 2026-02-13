---
title: "CLI Reference (nextstat)"
status: stable
---

# CLI Reference (nextstat)

The `nextstat` CLI is implemented in `crates/ns-cli` and focuses on:
- deterministic parity mode (`--threads 1`)
- JSON in / JSON out contracts for reproducible workflows

## Global flags

- `--interp-defaults {root|pyhf}` (pyhf JSON only)
  - `root` (default): NormSys=Code4, HistoSys=Code4p (smooth, TREx/ROOT-style)
  - `pyhf`: NormSys=Code1, HistoSys=Code0 (strict pyhf defaults)
  - Note: GPU backends currently support only `root` interpolation defaults.
    HS3 inputs always use ROOT defaults (Code1/Code0) and are not GPU-accelerated yet.
- `--log-level {error|warn|info|debug|trace}` — log verbosity (default: `warn`)
- `--bundle <DIR>` — optional output directory for structured run artifacts (fit JSON, logs, diagnostics)

## Commands (high level)

For the HistFactory configuration format, see `docs/references/analysis-config.md`.
For event-level fits, use the `unbinned_spec_v0` schema (`nextstat config schema --name unbinned_spec_v0`).

HEP / HistFactory (pyhf JSON and HS3 JSON auto-detected):
- `nextstat validate --config analysis.yaml`
- `nextstat config schema [--name analysis_spec_v0]`
- `nextstat import histfactory --xml combination.xml --output workspace.json`
- `nextstat import trex-config --config trex.txt --output workspace.json [--analysis-yaml analysis.yaml] [--coverage-json coverage.json] [--expr-coverage-json expr_coverage.json]`
- `nextstat import patchset --workspace BkgOnly.json --patchset patchset.json [--patch-name ...]`
- `nextstat export histfactory --input workspace.json --out-dir export/ [--prefix meas] [--overwrite] [--python]`
- `nextstat build-hists --config trex.config --out-dir out/ [--base-dir ...] [--coverage-json coverage.json] [--expr-coverage-json expr_coverage.json]`
- `nextstat trex import-config --config trex.config --out analysis.yaml [--report analysis.mapping.json]`
- `nextstat audit --input workspace.json [--format text|json] [--output audit.json]`
- `nextstat fit --input workspace.json [--gpu cuda|metal] [--fit-regions SR1,SR2] [--validation-regions VR1] [--asimov] [--parity] [--json-metrics metrics.json] [--threads 1]`
- `nextstat hypotest --input workspace.json --mu 1.0 [--expected-set] [--json-metrics metrics.json] [--threads 1]`
- `nextstat hypotest-toys --input workspace.json --mu 1.0 [--n-toys 1000 --seed 42] [--expected-set] [--threads 0] [--gpu cuda|metal] [--json-metrics metrics.json]`
- `nextstat significance --input workspace.json [--json-metrics metrics.json] [--threads 1]`
- `nextstat goodness-of-fit --input workspace.json [--json-metrics metrics.json] [--threads 1]`
- `nextstat upper-limit --input workspace.json [--expected] [--scan-start ... --scan-stop ... --scan-points ...] [--json-metrics metrics.json] [--threads 1]`
- `nextstat scan --input workspace.json --start 0 --stop 5 --points 21 [--gpu cuda|metal] [--json-metrics metrics.json] [--threads 1]`
- `nextstat combine <ws1.json> <ws2.json> [ws3.json ...] [--output combined.json] [--prefix-channels]`
- `nextstat viz profile --input workspace.json --start 0 --stop 5 --points 21 [--output profile.json] [--threads 1]`
- `nextstat viz cls --input workspace.json [--alpha 0.05] [--scan-start 0 --scan-stop 5 --scan-points 201] [--output cls.json] [--threads 1]`
- `nextstat viz ranking --input workspace.json [--output ranking.json] [--threads 1]`
- `nextstat viz pulls --input workspace.json --fit fit.json [--output pulls.json] [--threads 1]`
- `nextstat viz corr --input workspace.json --fit fit.json [--include-covariance] [--output corr.json] [--threads 1]`
- `nextstat viz distributions --input workspace.json --histfactory-xml combination.xml [--fit fit.json] [--output distributions.json] [--threads 1]`
- `nextstat viz gammas --input workspace.json --fit fit.json [--output gammas.json] [--threads 1]`
- `nextstat viz separation --input workspace.json [--signal-samples signal] [--histfactory-xml combination.xml] [--output separation.json] [--threads 1]`
- `nextstat viz summary fit1.json fit2.json [--labels "Analysis A,Analysis B"] [--output summary.json]`
- `nextstat viz pie --input workspace.json [--fit fit.json] [--output pie.json] [--threads 1]`
- `nextstat mass-scan --workspaces-dir mass_workspaces/ [--alpha 0.05] [--scan-start 0 --scan-stop 5 --scan-points 41] [--labels m100,m200] [--json-metrics metrics.json] [--threads 1]`
- `nextstat preprocess smooth --input workspace.json [--output smoothed.json] [--max-variation 0.0]`
- `nextstat preprocess prune --input workspace.json [--output pruned.json] [--threshold 0.005]`
- `nextstat report --input workspace.json --histfactory-xml combination.xml --out-dir report/ [--fit fit.json] [--render] [--deterministic] [--blind-regions SR1,SR2] [--include-covariance] [--uncertainty-grouping prefix_1] [--skip-uncertainty] [--overwrite] [--pdf report.pdf] [--svg-dir svg/] [--python python3]`
- `nextstat validation-report --apex2 master_report.json --workspace workspace.json --out validation_report.json [--pdf validation_report.pdf] [--deterministic]`
- `nextstat version`

HEP / Unbinned (event-level) (Phase 1, experimental):
- `nextstat convert --input data.root --tree MyTree --output events.parquet --observable mass:100:180 [--selection "..."] [--weight "..."] [--max-events 1000000]`
  - Writes the unbinned Parquet schema v1 (`docs/references/unbinned-parquet-schema.md`) and embeds observable bounds in Parquet metadata.
- `nextstat unbinned-fit --config unbinned.json [--threads 0] [--gpu cuda|metal] [--opt-max-iter N] [--opt-tol X] [--opt-m M] [--opt-smooth-bounds]`
  - Optimizer overrides tune L-BFGS-B for this fit invocation only.
  - `--opt-smooth-bounds` enables smooth parameter transforms (often reduces iterations on constrained fits).
  - Supported PDFs (v0): `gaussian`, `crystal_ball`, `double_crystal_ball`, `exponential`, `chebyshev`, `histogram`, `histogram_from_tree`, `kde`, `kde_from_tree`
  - Supported yield modifiers (v0): `normsys` (Code1: `hi^alpha` / `lo^{-alpha}`), `weightsys` (code0/code4p interpolation)
  - Tree-driven non-parametrics (`histogram_from_tree`, `kde_from_tree`) support `weight_systematics` (per-event weight ratios) and `horizontal_systematics` (up/down observable expressions).
  - `--gpu` is currently supported only for a conservative subset (multi-channel supported; each channel must be 1D):
    - Observables: `n_obs=1` per channel
    - PDFs:
      - `gaussian`, `exponential`, `crystal_ball`, `double_crystal_ball`, `chebyshev` (Chebyshev order `<= 16`)
      - `histogram`
      - `histogram_from_tree` **only** when used as a pre-materialized histogram shape (no shape morphing): `horizontal_systematics` must be empty and `weight_systematics` may be used only with `apply_to_shape: false` (yield-only).
	    - Yields: fixed / parameter / scaled
	    - Yield modifiers: `normsys`, `weightsys` (ROOT interpolation defaults only)
	    - Gaussian constraints: supported
	    - Observed-data weights: supported (finite and `>= 0`)
	  - Observed-data weights can be provided via:
	    - ROOT: `channels[].data.weight: "<expr>"` (e.g. `weight_mc`)
	    - Parquet: use `nextstat convert --weight "<expr>" ...` to embed a `weights` column
	    Unsupported specs error out (CPU path remains available without `--gpu`).
HEP / Hybrid (binned + unbinned) (Phase 4):
- `nextstat hybrid-fit --binned workspace.json --unbinned unbinned.yaml [--output result.json] [--json-metrics metrics.json] [--threads 1]`
  - Combines a binned (pyhf/HS3 JSON) and unbinned (YAML/JSON spec) model into a single `HybridLikelihood` with shared parameters matched by name. Shared parameters get intersected bounds.
  - POI resolution: prefers binned model's POI, falls back to unbinned spec's `model.poi`.
  - Output JSON includes `hybrid: true`, `n_shared`, `n_binned_params`, `n_unbinned_params`, bestfit, uncertainties, covariance.

- `nextstat unbinned-scan --config unbinned.json --start 0 --stop 5 --points 21 [--threads 0] [--gpu cuda|metal]`
  - Requires `model.poi` in the spec (POI is scanned).
- `nextstat unbinned-fit-toys --config unbinned.json --n-toys 100 --seed 42 [--gen init|mle] [--set name=value ...] [--threads 0] [--gpu cuda|metal] [--gpu-devices 0,1,...] [--gpu-sample-toys] [--gpu-native] [--gpu-shards N] [--shard INDEX/TOTAL]`
  - Requires `model.poi` in the spec (POI is summarized across toys).
  - Output includes per-toy convergence flags (`results.converged[]`) and pull summaries (from converged toys only).
  - CPU toy fitting uses warm-start (MLE θ̂), retry with jitter (up to 3 attempts), smooth bounds escalation on last retry, and Rayon parallel iteration. Hessian is skipped by default — only computed when pull guardrails are active (`--max-abs-poi-pull-mean` / `--poi-pull-std-range`).
  - **Hessian skip policy**: Hessian adds ~40-50% compute per toy. For HEP CLs, only q̃(μ) = 2·ΔNLL is needed (no Hessian). For pharma, pulls are optional diagnostics. Hessian is auto-enabled by the CLI when pull guardrails are specified; otherwise it is skipped for throughput.
  - Error metrics are split: `results.n_validation_error` (PDF/spec constraint violations), `results.n_computation_error` (numeric failures), `results.n_nonconverged` (optimizer did not converge). `n_error = n_validation_error + n_computation_error`.
  - **CPU farm mode** (`--shard INDEX/TOTAL`): runs only a deterministic slice of the toy range.
    Each shard `k` of `M` total runs toys `[start_k .. start_k + count_k)` with `seed + start_k` as the base seed.
    This enables linear scale-out across cluster nodes. Combine results with `unbinned-merge-toys`.
    Example (4-node farm, 10000 toys):
    ```bash
    nextstat unbinned-fit-toys --config spec.json --n-toys 10000 --shard 0/4 -o shard0.json
    nextstat unbinned-fit-toys --config spec.json --n-toys 10000 --shard 1/4 -o shard1.json
    nextstat unbinned-fit-toys --config spec.json --n-toys 10000 --shard 2/4 -o shard2.json
    nextstat unbinned-fit-toys --config spec.json --n-toys 10000 --shard 3/4 -o shard3.json
    nextstat unbinned-merge-toys shard0.json shard1.json shard2.json shard3.json -o merged.json
    ```
- `nextstat unbinned-merge-toys <shard1.json> <shard2.json> [<shard3.json> ...] [-o merged.json]`
  - Merges shard outputs from `unbinned-fit-toys --shard` into a single result.
  - Validates consistent full `gen` config across shards (point, params, overrides, seed, n_toys).
  - Merged output includes `overall_convergence`, `fittable_convergence`, per-shard detail, and all per-toy arrays concatenated in shard order.
- In `--gpu` mode, multi-channel specs are supported (each included channel must be 1D; total NLL is the sum across channels).
- `--gpu-devices` selects CUDA devices for host-toy sharding (example: `--gpu cuda --gpu-devices 0,1`). If omitted, defaults to device `0`.
- `--gpu-sample-toys` enables experimental GPU toy sampling (default remains CPU toy sampling). GPU toy sampling supports Gaussian/Exponential/CrystalBall/DoubleCrystalBall/Chebyshev/Histogram PDFs (per included channel; yields and yield modifiers are supported).
- `--gpu-native` (CUDA only, experimental) enables persistent on-device L-BFGS (`pipeline = "cuda_gpu_native"` or `"cuda_gpu_native_sharded"`). This flag is explicit opt-in and is never auto-enabled.
  - `--gpu-shards N` (CUDA only) enables logical toy sharding (round-robin over `--gpu-devices`):
    - with `--gpu-sample-toys`: sharded device-resident path (`pipeline = "cuda_device_sharded"`),
    - without `--gpu-sample-toys`: sharded host-toy path (`pipeline = "cuda_host_sharded"`).
    - Practical single-GPU validation commands (no 2+ GPU stand required):
      - `nextstat unbinned-fit-toys --config unbinned.json --n-toys 2000 --gpu cuda --gpu-sample-toys --gpu-devices 0 --gpu-shards 4 --json-metrics metrics.json`
      - `nextstat unbinned-fit-toys --config unbinned.json --n-toys 2000 --gpu cuda --gpu-devices 0 --gpu-shards 4 --json-metrics metrics.json`
      - Expect `timing.breakdown.toys.device_shard_plan = [0,0,0,0]`.
  - `histogram_from_tree` is also supported under `--gpu-sample-toys` when it falls into the GPU subset (materialized histogram shape, i.e. no shape morphing: no `horizontal_systematics`, and `weight_systematics` only with `apply_to_shape: false`).
  - Integration coverage includes this `histogram_from_tree` subset on both CUDA and Metal.
    - CUDA: `cuda_device` (single shard) / `cuda_device_sharded` (multi-shard) pipelines keep `obs_flat` device-resident for batch fits (eliminates D2H+H2D round-trip).
    - Metal: `metal_gpu_sample` pipeline runs sampling on Metal; the batch fitter still uploads host `obs_flat` to the GPU (no device-resident toy→fit path yet).
  - If `--json-metrics <path>` is provided, `timing.breakdown.toys` includes a coarse timing breakdown:
    - `pipeline`: `cpu` (CPU fit) | `host` (CPU sample + single-GPU batch fit) | `cuda_host_sharded` (CPU sample + sharded CUDA host-toy batch fit) | `cuda_host_multi_gpu` (CPU sample + multi-GPU batch fit) | `cuda_device` (GPU sample + single-shard GPU batch fit) | `cuda_device_sharded` (GPU sample + sharded GPU batch fit) | `cuda_gpu_native` (persistent on-device L-BFGS) | `cuda_gpu_native_sharded` (sharded persistent on-device L-BFGS) | `metal_gpu_sample` (Metal sample + GPU batch fit)
    - `device_ids`: CUDA device ids used for sharding (only for CUDA runs)
    - `device_shard_plan`: CUDA shard→device mapping when `--gpu-shards` is active (only for CUDA runs)
    - `warm_start`, `sample_s`, `batch_build_s`, `batch_fit_s`, `poi_sigma_s`
  - CI guardrails (optional): `--require-all-converged`, `--max-abs-poi-pull-mean <x>`, `--poi-pull-std-range <low> <high>`
- `nextstat unbinned-ranking --config unbinned.json [--threads 0]`
  - Requires `model.poi` in the spec (impacts are computed on POI).
- `nextstat unbinned-hypotest --config unbinned.json --mu 1.0 [--threads 0] [--gpu cuda|metal]`
  - Computes `q_mu` (and `q0` if `mu=0` is within bounds).
- `nextstat unbinned-hypotest-toys --config unbinned.json --mu 1.0 [--n-toys 1000 --seed 42] [--expected-set] [--threads 0] [--gpu cuda|metal] [--gpu-devices 0,1,...] [--gpu-sample-toys] [--gpu-shards N]`
- Toy-based CLs `hypotest` (qtilde) for unbinned models.
- In `--gpu` mode: per-toy fits are accelerated via GPU batch/lockstep fitting (conservative GPU subset applies). Multi-channel specs are supported (each included channel must be 1D).
- `--gpu-devices` selects CUDA devices for host-toy sharding (default `0` if omitted).
- Toys are sampled on CPU by default; `--gpu-sample-toys` enables experimental GPU toy sampling (Gaussian/Exponential/CrystalBall/DoubleCrystalBall/Chebyshev/Histogram PDFs, per included channel).
  - `--gpu-shards N` (CUDA only) enables logical toy sharding (round-robin over `--gpu-devices`):
    - with `--gpu-sample-toys`: sharded device-resident path (`pipeline = "cuda_device_sharded"`),
    - without `--gpu-sample-toys`: sharded host-toy path (`pipeline = "cuda_host_sharded"`).
    - Practical single-GPU validation commands:
      - `nextstat unbinned-hypotest-toys --config unbinned.json --mu 1.0 --n-toys 2000 --gpu cuda --gpu-sample-toys --gpu-devices 0 --gpu-shards 4 --json-metrics metrics.json`
      - `nextstat unbinned-hypotest-toys --config unbinned.json --mu 1.0 --n-toys 2000 --gpu cuda --gpu-devices 0 --gpu-shards 4 --json-metrics metrics.json`
      - Expect `timing.breakdown.toys.device_shard_plan = [0,0,0,0]`.
  - `histogram_from_tree` is also supported under `--gpu-sample-toys` when it falls into the GPU subset (materialized histogram shape, i.e. no shape morphing: no `horizontal_systematics`, and `weight_systematics` only with `apply_to_shape: false`).
  - Integration coverage includes this `histogram_from_tree` subset on both CUDA and Metal.
  - In `--gpu cuda` mode, `--json-metrics <path>` includes `timing.breakdown` with:
    - `obs_fits_s`: observed-data baseline fits time (free + fixed generation points)
    - `toys`: per-ensemble timings for `b` and `sb` (sampling + batch build + free/fixed fits)

Time series (Phase 8):
- `nextstat timeseries kalman-filter --input kalman_1d.json`
- `nextstat timeseries kalman-smooth --input kalman_1d.json`
- `nextstat timeseries kalman-em --input kalman_1d.json ...`
- `nextstat timeseries kalman-fit --input kalman_1d.json ...`
- `nextstat timeseries kalman-viz --input kalman_1d.json [--max-iter 50] [--level 0.95] [--forecast-steps ...]`
- `nextstat timeseries kalman-forecast --input kalman_1d.json ...`
- `nextstat timeseries kalman-simulate --input kalman_1d.json ...`
- `nextstat timeseries garch11-fit --input returns.json`
- `nextstat timeseries sv-logchi2-fit --input returns.json`

Survival analysis (Phase 9):
- `nextstat survival cox-ph-fit --input cox.json [--ties efron|breslow] [--no-robust] [--no-cluster-correction] [--no-baseline]`
- `nextstat survival km --input km.json [--conf-level 0.95] [--output km.json]`
- `nextstat survival log-rank-test --input lr.json [--output lr_result.json]`

Churn / Subscription (Phase 7):
- `nextstat churn generate-data [--n-customers 2000] [--n-cohorts 6] [--max-time 24] [--seed 42]`
- `nextstat churn retention --input churn.json [--conf-level 0.95]`
- `nextstat churn risk-model --input churn.json`
- `nextstat churn bootstrap-hr --input churn.json [--n-bootstrap 1000] [--seed 42]`
- `nextstat churn uplift --input churn.json [--horizon 12]`
- `nextstat churn ingest --input raw.json [--observation-end 24]`
- `nextstat churn cohort-matrix --input churn.json --periods 1,3,6,12,24`
- `nextstat churn compare --input churn.json [--correction bh|bonferroni] [--alpha 0.05]`
- `nextstat churn diagnostics --input churn.json [--trim 0.01] [--covariate-names name1,name2] [--out-dir artifacts/] [--output diag.json]`
- `nextstat churn uplift-survival --input churn.json [--horizon 12] [--eval-horizons 3,6,12,24]`

## GPU acceleration

The `--gpu <device>` flag enables GPU acceleration, where `<device>` is one of:

- **`cuda`** — NVIDIA GPU (f64 precision). Requires `cuda` feature and an NVIDIA GPU at runtime.
- **`metal`** — Apple Silicon GPU (f32 precision). Requires `metal` feature and Apple Silicon (M1+) at runtime.

Without `--gpu`, the standard CPU (SIMD/Rayon + Accelerate) path is used. If the requested GPU is not available at runtime, the command exits with an error.

### Build with GPU support

```bash
# CUDA (NVIDIA)
cargo build -p ns-cli --features cuda --release

# Metal (Apple Silicon)
cargo build -p ns-cli --features metal --release

# Both
cargo build -p ns-cli --features "cuda,metal" --release
```

### Supported commands

**`fit --gpu cuda|metal`** — Single-model MLE fit on GPU. Uses `GpuSession` for fused NLL+gradient (1 kernel launch per L-BFGS iteration). Hessian computed via finite differences of GPU gradient at the end.

```bash
nextstat fit --input workspace.json --gpu cuda
nextstat fit --input workspace.json --gpu metal
```

**`scan --gpu cuda|metal`** — GPU-accelerated profile likelihood scan. A single `GpuSession` is shared across all scan points with warm-start between mu values.

```bash
nextstat scan --input workspace.json --start 0 --stop 5 --points 21 --gpu cuda
nextstat scan --input workspace.json --start 0 --stop 5 --points 21 --gpu metal
```

**`hypotest-toys --gpu cuda|metal`** — Batch toy fitting on GPU. The lockstep GPU batch optimizer computes NLL + analytical gradient for all toys in a single kernel launch per iteration. Both CUDA (f64) and Metal (f32) backends are supported.

```bash
# CUDA (NVIDIA, f64)
nextstat hypotest-toys --input workspace.json --mu 1.0 --n-toys 10000 --gpu cuda

# Metal (Apple Silicon, f32 — tolerance relaxed to 1e-3)
nextstat hypotest-toys --input workspace.json --mu 1.0 --n-toys 10000 --gpu metal
```

### GPU troubleshooting

**"error: CUDA not available"** — The binary was not built with `--features cuda`, or no NVIDIA GPU/driver is detected at runtime. Verify: `nvidia-smi` must show a GPU. Rebuild with `cargo build -p ns-cli --features cuda --release`.

**"error: Metal not available"** — The binary was not built with `--features metal`, or the machine is not Apple Silicon (M1+). Metal requires macOS 13+ and Apple GPU family 7+.

**Metal f32 precision** — Metal uses f32 (Apple Silicon has no hardware f64). Convergence tolerance is automatically clamped to `max(tol, 1e-3)`. Expect NLL relative differences of ~1e-6 vs CPU f64. This is sufficient for toy-based hypothesis testing but not for strict parity validation — use CPU with `--parity` for that.

**GPU interpolation defaults** — GPU backends require `--interp-defaults root` (Code4/Code4p polynomial interpolation). If your model uses Code1 (exponential) NormSys with non-positive hi/lo factors, the GPU kernel cannot represent the CPU piecewise-linear fallback and will error. Use CPU path for such models.

**Out-of-memory on large models** — Each toy in batch mode uses shared memory proportional to `n_params + n_bins`. For very large models (>500 params), reduce `--n-toys` or use CPU path. CUDA: check `nvidia-smi` for VRAM usage. Metal: unified memory is shared with the system.

**Slow first run** — Metal compiles MSL kernels at runtime on first invocation (~1-2s). Subsequent runs reuse the compiled pipeline. CUDA uses pre-compiled PTX (no first-run overhead).

## Determinism and Parity Mode

NextStat separates "specification correctness" (pyhf parity) from "speed" via two evaluation modes:

### `--parity` flag (recommended)

```bash
nextstat fit --input workspace.json --parity
```

When `--parity` is active:
1. **EvalMode::Parity** is set process-wide
2. **Kahan compensated summation** replaces naive `+=` in Poisson NLL
3. **Apple Accelerate** is automatically disabled
4. **Thread count forced to 1** (sequential Rayon)
5. Results are **bit-exact reproducible** across runs

### `--threads 1` (legacy)

```bash
nextstat fit --input workspace.json --threads 1
```

Same as `--parity` but without Kahan summation. Use `--parity` instead for full determinism.

Interpolation note:
- Use `--interp-defaults {root|pyhf}` to choose interpolation defaults for pyhf JSON inputs.
- GPU backends currently require `--interp-defaults root` (Code4/Code4p).
- HS3 inputs use ROOT HistFactory defaults (Code1 for NormSys, Code0 for HistoSys) and are CPU-only for now.

### Environment variable

```bash
NEXTSTAT_DISABLE_ACCELERATE=1 nextstat fit --input workspace.json
```

Disables Apple Accelerate only, without forcing single-thread or Kahan.

### Tolerance contract

| Tier | Metric | Parity Tolerance | Fast Tolerance |
|------|--------|-----------------|----------------|
| 1 | Per-bin expected data | 1e-12 | 1e-10 |
| 2 | Expected data vector | 1e-8 | 1e-8 |
| 3 | NLL value | 1e-8 atol | 1e-6 rtol |
| 4 | Gradient (AD vs FD) | 1e-6 atol | 1e-6 atol |
| 5 | Best-fit params | 2e-4 | 2e-4 |
| 6 | Uncertainties | 5e-4 | 5e-4 |
| 7 | Toy ensemble | 0.03–0.05 | 0.03–0.05 |

Full tolerance values: `tests/python/_tolerances.py`.

### Python API equivalent

```python
import nextstat
nextstat.set_eval_mode("parity")  # or "fast" (default)
print(nextstat.get_eval_mode())
```

## Upper limit: bisection vs scan mode

`upper-limit` supports two modes:

1. Bisection (root finding): default.
2. Scan mode: provide `--scan-start`, `--scan-stop`, `--scan-points` to compute limits from a dense CLs curve.

Scan mode is useful for:
- storing a full curve for plotting
- avoiding repeated root-finding (including expected-set curves)

## Discovery significance

`nextstat significance` tests the background-only hypothesis (mu=0) and reports the observed discovery p-value (p0) and significance (Z = inverse_normal(1-p0) ~ sqrt(q0)). Comparable to TRExFitter `GetSignificance`.

```bash
nextstat significance --input workspace.json
nextstat significance --input workspace.json --json-metrics metrics.json --threads 4
```

Flags:
- `--input, -i` — input workspace (pyhf/HS3 JSON)
- `--output, -o` — output file (JSON). Defaults to stdout
- `--json-metrics` — standardized metrics JSON (schema `nextstat_metrics_v0`)
- `--threads` — thread count (default: 1)

## Goodness-of-fit test

`nextstat goodness-of-fit` fits the model, then computes the Poisson deviance chi-squared between the best-fit expected yields and observed data. Reports chi-squared, ndof, and p-value. Comparable to TRExFitter saturated-model GoF.

```bash
nextstat goodness-of-fit --input workspace.json
nextstat goodness-of-fit --input workspace.json --json-metrics metrics.json
```

Flags:
- `--input, -i` — input workspace (pyhf/HS3 JSON)
- `--output, -o` — output file (JSON). Defaults to stdout
- `--json-metrics` — standardized metrics JSON (schema `nextstat_metrics_v0`)
- `--threads` — thread count (default: 1)

## Combine workspaces

`nextstat combine` merges multiple pyhf JSON workspaces into a single workspace. Channels, observations, and measurement parameter configs are merged. Systematics with the same name are automatically shared (correlated). Comparable to TRExFitter `MultiFit` combination and `pyhf combine`.

```bash
nextstat combine ws1.json ws2.json --output combined.json
nextstat combine ws1.json ws2.json ws3.json --prefix-channels
```

Flags:
- `inputs` — positional, at least 2 input workspace files (pyhf JSON)
- `--output, -o` — output file (pyhf JSON). Defaults to stdout
- `--prefix-channels` — prefix channel names with workspace index to avoid conflicts (e.g. "SR" becomes "ws0_SR", "ws1_SR"). If omitted, channel names must be unique across all inputs.

## Mass scan

`nextstat mass-scan` runs asymptotic CLs upper limits across multiple workspaces (one per mass/signal hypothesis) and outputs observed and expected (+-1sigma/+-2sigma) limits — the data for an ATLAS/CMS-style exclusion plot (mu_up vs mass).

```bash
nextstat mass-scan --workspaces-dir mass_workspaces/
nextstat mass-scan --workspaces-dir mass_workspaces/ --alpha 0.05 --scan-start 0 --scan-stop 10 --scan-points 101 --labels "100,200,300"
```

Flags:
- `--workspaces-dir` — directory containing workspace JSON files (one per mass point). Files are sorted lexicographically; use zero-padded names (e.g. `mass_100.json`, `mass_200.json`)
- `--alpha` — target CLs level (default: 0.05)
- `--scan-start` — CLs scan start mu (default: 0.0)
- `--scan-stop` — CLs scan stop mu (default: 5.0)
- `--scan-points` — CLs scan points per mass point (default: 41)
- `--labels` — optional labels for each mass point (comma-separated). If omitted, filenames (without extension) are used
- `--output, -o` — output file (JSON). Defaults to stdout
- `--json-metrics` — standardized metrics JSON (schema `nextstat_metrics_v0`)
- `--threads` — thread count (default: 1)

## Workspace preprocessing

Native Rust preprocessing passes (no Python dependency). Both commands read a pyhf JSON workspace and write a modified workspace.

### `preprocess smooth`

Smooth histosys templates (353QH,twice — ROOT TH1::Smooth equivalent). Applies running-median smoothing to HistoSys up/down deltas (variation - nominal), preserving the nominal shape.

```bash
nextstat preprocess smooth --input workspace.json --output smoothed.json
nextstat preprocess smooth --input workspace.json --max-variation 0.5
```

Flags:
- `--input, -i` — input workspace (pyhf JSON)
- `--output, -o` — output workspace (pyhf JSON). Defaults to stdout
- `--max-variation` — max relative variation cap (default: 0.0 = disabled). Bins where |delta/nominal| > cap are clamped

### `preprocess prune`

Prune negligible systematics from a workspace. Removes HistoSys/NormSys modifiers whose max relative effect is below a threshold.

```bash
nextstat preprocess prune --input workspace.json --output pruned.json
nextstat preprocess prune --input workspace.json --threshold 0.01
```

Flags:
- `--input, -i` — input workspace (pyhf JSON)
- `--output, -o` — output workspace (pyhf JSON). Defaults to stdout
- `--threshold` — pruning threshold: modifiers with max |delta/nominal| < threshold are removed (default: 0.005)

## Experiment tracking (`--json-metrics`)

Most statistical commands support `--json-metrics <path>` to write a standardized metrics JSON (schema `nextstat_metrics_v0`) for experiment tracking and CI pipelines. Pass `-` to write to stdout.

Supported commands: `fit`, `hypotest`, `hypotest-toys`, `significance`, `goodness-of-fit`, `upper-limit`, `scan`, `mass-scan`, `unbinned-fit`, `hybrid-fit`.

## JSON contracts

The CLI outputs pretty JSON to stdout by default, or to `--output`.

### Survival (Phase 9): Cox PH (`survival cox-ph-fit`)

Input JSON:

```json
{
  "times": [2.0, 1.0, 1.0, 0.5, 0.5, 0.2],
  "events": [true, true, false, true, false, false],
  "x": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0], [0.0, -1.0], [0.5, 0.5]],
  "groups": [1, 1, 2, 2, 3, 3]
}
```

Command:

```bash
nextstat survival cox-ph-fit --input cox.json --ties efron
```

Notes:
- `groups` is optional. If present, robust SE are **cluster-robust** by default (`robust_kind="cluster"`).
- Robust SE are enabled by default; disable with `--no-robust`.
- Cluster small-sample correction is enabled by default; disable with `--no-cluster-correction`.
- Baseline cumulative hazard output is enabled by default; disable with `--no-baseline`.

Output JSON keys (subset):
- `coef`: fitted beta coefficients (no intercept)
- `se`, `cov`: observed-information SE/covariance
- `robust_se`, `robust_cov`, `robust_kind`, `robust_meta`: sandwich SE/covariance (HC0 or cluster)
- `baseline_times`, `baseline_cumhaz`: baseline cumulative hazard estimate at event times

`nextstat validate --config ...` validates either:
- legacy `nextstat run` config (`run.yaml`/`run.json`), or
- analysis spec v0 (`schema_version: trex_analysis_spec_v0`)

It prints a small JSON summary (`config_type: run_config_legacy|analysis_spec_v0`) and exits non-zero on validation errors.

For analysis spec v0:
- `nextstat validate --config ...` fail-fast checks `gates.baseline_compare` manifest presence when the gate is enabled.
- `nextstat run --config ...` executes `gates.baseline_compare` and fails the run if the baseline gate fails.

`nextstat report` writes multiple artifacts into `--out-dir` (currently: `distributions.json`, `pulls.json`, `corr.json`, `yields.json`, `uncertainty.json`, plus `yields.csv` and `yields.tex`). `uncertainty.json` is ranking-based and can be skipped via `--skip-uncertainty`. When `--render` is enabled it calls `python -m nextstat.report render ...` to produce a multi-page PDF and per-plot SVGs (requires `matplotlib`, see `nextstat[viz]` extra).

If `--fit` is omitted, `nextstat report` runs an MLE fit itself and writes `fit.json` into `--out-dir` before producing the report artifacts.

For time series input formats, see:
- `docs/tutorials/phase-8-timeseries.md`
- `docs/tutorials/phase-8-volatility.md`

For the frequentist (CLs) workflow, see:
- `docs/tutorials/phase-3.1-frequentist.md`

## Input format auto-detection (pyhf vs HS3)

All commands that accept `--input workspace.json` automatically detect the JSON format:

- **pyhf JSON** — standard HistFactory workspace with `"channels"` + `"measurements"` at top level.
- **HS3 JSON** — HEP Statistics Serialization Standard (v0.2) with `"distributions"` + `"metadata"` (containing `"hs3_version"`), as produced by ROOT 6.37+.

Detection is instant (prefix scan of the first ~2 KB). No `--format` flag is needed.

```bash
# pyhf workspace (auto-detected)
nextstat fit --input workspace.json

# HS3 workspace from ROOT (auto-detected)
nextstat fit --input workspace-postFit_PTV.json

# Both produce the same HistFactoryModel internally
```

When HS3 is detected, the CLI uses ROOT HistFactory default interpolation codes (Code1 for NormSys, Code0 for HistoSys) and selects the first analysis and `"default_values"` parameter point set.

## Import notes

### HistFactory XML import semantics

`nextstat import histfactory` parses `combination.xml` + channel XMLs and reads ROOT histograms to produce a pyhf-style `workspace.json`.
It follows HistFactory conventions used by pyhf's `readxml` validation fixtures:

- `ShapeSys` histograms (`<ShapeSys HistoName="...">`) are treated as **relative** per-bin uncertainties and converted to absolute
  `sigma_abs = rel * nominal`.
- `StatError` follows channel `<StatErrorConfig ConstraintType=...>`:
  - `ConstraintType="Poisson"` => preserves `staterror` (per-channel, name `staterror_<channel>`) and attaches per-bin `Gamma` constraint
    metadata (non-standard extension) to `measurements[].config.parameters[]` entries named `staterror_<channel>[i]`.
  - `ConstraintType="Gaussian"` => preserves `staterror` (per-channel, name `staterror_<channel>`) with Gaussian penalty (pyhf-style).
- `StatError` histograms (`<StatError Activate="True" HistoName="...">`) are treated as **relative** per-bin uncertainties and converted to
  absolute `sigma_abs = rel * nominal`.
- ROOT/HistFactory defaults when `<StatErrorConfig>` is omitted: `ConstraintType="Poisson"` and `RelErrorThreshold=0.05` (bins with relative
  stat error below threshold are pruned, i.e. the corresponding `staterror_<channel>[i]` is fixed at 1.0).
- Samples with `NormalizeByTheory="True"` receive a `lumi` modifier named `Lumi`.
- `LumiRelErr` and `ParamSetting Const="True"` are surfaced via `measurements[].config.parameters` (`auxdata=[1]`, `sigmas=[LumiRelErr]`, `fixed=true`).
- `<NormFactor Val/Low/High>` is surfaced via `measurements[].config.parameters` as `inits` and `bounds`.
- HistFactory `<ConstraintTerm>` (ROOT extension) is preserved as a non-standard field
  `measurements[].config.parameters[].constraint` and is interpreted by NextStat when building the model:
  - `Type="LogNormal"`: applies ROOT's `alphaOfBeta` transform for `normsys` evaluation (keeps Gaussian constraint on `alpha`).
  - `Type="Gamma"`: applies a Gamma constraint term and interprets the parameter as a positive `beta` with `alpha=(beta-1)/rel`.
  - `Type="Uniform"`: removes the Gaussian penalty (flat within bounds).
  - `Type="NoConstraint"/"NoSyst"`: fixes the parameter at nominal.

`nextstat import trex-config` currently supports only a small subset of TRExFitter configs:
- `ReadFrom: NTUP` (or omitted).
- `Region:` blocks with `Variable`, `Binning`, optional `Selection`.
- `Sample:` blocks with `File`, optional `Weight`, and optional simple modifiers (`NormFactor`, `NormSys`, `StatError`).
- `Systematic:` blocks for `Type: norm|weight|tree` applied by `Samples:` and optional `Regions:`.

`ReadFrom: HIST` is supported only as a wrapper over an existing HistFactory export:
- Provide `HistoPath: /path/to/export_dir` (must contain exactly one `combination.xml` under it), or
- Provide `CombinationXml: /path/to/combination.xml` explicitly.

Partial TREx semantics in `ReadFrom: HIST`:
- If the config includes `Region:` blocks, they act as an **include-list** for channels (in config order).
- If the config includes `Sample:` blocks, they act as an **include-list** for samples. Masking rules:
  - `Sample: X` with `Regions: ...` masks `X` only in those channels.
  - `Sample: X` nested under a `Region: Y` block and **without** `File`/`Path` is treated as a region-scoped filter entry:
    it selects sample `X` only in channel `Y`. Repeating the same `Sample: X` under multiple regions is allowed.
  - If a channel has any region-scoped filter entries, they take precedence over any global sample include-list for that channel.
  Empty channels are dropped unless channels were explicitly selected via `Region:` blocks (then it is an error).
- In HIST mode, `Region:` blocks do **not** require `Variable`/`Binning`, and `Sample:` blocks do **not** require `File`
  (they can be used as pure filters).
- When `HistoPath` is provided, it is used as the **base directory** for resolving relative paths inside the HistFactory XML
  (common when `combination.xml` lives under `config/` but ROOT inputs are referenced from the export root).

Optional outputs:
- `--analysis-yaml` writes an analysis spec v0 wrapper (`inputs.mode=trex_config_txt`) to drive `nextstat run`.
- `--coverage-json` writes a best-effort report of unknown keys/attrs to help parity work against legacy configs.
- `--expr-coverage-json` writes a report of expression-bearing keys (selection/weight/variable + weight systematics) and
  whether they compile under NextStat's ROOT-dialect expression engine.

`nextstat trex import-config` is a best-effort migration helper for TRExFitter `.config` files:
- it emits an analysis spec v0 YAML using `inputs.mode=trex_config_yaml`
- it also writes a mapping report listing mapped and unmapped keys

Example config: `docs/examples/trex_config_ntup_minimal.txt`.
Example config (HIST wrapper): `docs/examples/trex_config_hist_minimal.txt`.

## Workspace audit

`nextstat audit` inspects a pyhf/HS3 workspace and reports channel/sample/modifier counts plus any unsupported features:

```bash
nextstat audit --input workspace.json
nextstat audit --input workspace.json --format json --output audit.json
```

## Export

`nextstat export histfactory` converts a pyhf workspace back to HistFactory XML + ROOT histogram files:

```bash
nextstat export histfactory --input workspace.json --out-dir export/
nextstat export histfactory --input workspace.json --out-dir export/ --prefix meas --overwrite --python
```

`--python` generates a Python driver script alongside the XML/ROOT artifacts.

## Time series visualization

`nextstat timeseries kalman-viz` produces a plot-friendly JSON artifact with smoothed states, observations, marginal normal bands, and optional forecast:

```bash
nextstat timeseries kalman-viz --input kalman_1d.json
nextstat timeseries kalman-viz --input kalman_1d.json --level 0.99 --forecast-steps 20
```

Internally runs EM → Kalman smooth → computes `±z_{α/2} × √diag(P)` bands at the requested `--level` (default 0.95).

## Volatility (Phase 12)

Input JSON for volatility commands:

```json
{ "returns": [0.01, -0.02, 0.005, 0.0] }
```

### GARCH(1,1)

```bash
nextstat timeseries garch11-fit --input returns.json
```

### Stochastic volatility (log-chi² approximation)

```bash
nextstat timeseries sv-logchi2-fit --input returns.json
```

## Version

```bash
nextstat version
```

Prints the NextStat version string and exits.

---

## `nextstat-server` (separate binary)

A self-hosted REST API for shared GPU inference. Built as a separate binary in `crates/ns-server`.

### Build

```bash
# CPU only
cargo build -p ns-server --release

# With GPU support
cargo build -p ns-server --features cuda --release
cargo build -p ns-server --features metal --release
```

### Run

```bash
nextstat-server --port 3742 --gpu cuda
nextstat-server --host 0.0.0.0 --port 3742 --gpu metal --threads 8
```

Arguments:
- `--port <PORT>` — listening port (default: 3742)
- `--host <HOST>` — bind address (default: 0.0.0.0)
- `--gpu <DEVICE>` — GPU device: `cuda` or `metal` (omit for CPU-only). If the binary was built without the corresponding feature, `nextstat-server` exits with an error.
- `--threads <N>` — Rayon thread pool size (default: 0 = auto)
- `--max-body-mb <MiB>` — maximum request body size in MiB (default: 64). Requests exceeding the limit return HTTP 413.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/fit` | MLE fit (workspace → FitResult) |
| `POST` | `/v1/ranking` | Nuisance parameter ranking |
| `POST` | `/v1/batch/fit` | Batch fit (multiple workspaces, max 100) |
| `POST` | `/v1/batch/toys` | Batch toy fitting |
| `POST` | `/v1/models` | Upload and cache a model |
| `GET` | `/v1/models` | List cached models |
| `DELETE` | `/v1/models/{id}` | Remove cached model |
| `GET` | `/v1/health` | Server health check |

All endpoints accept/return JSON. Errors return `{"error": "<message>"}` with appropriate HTTP status codes.

Notes:
- `POST /v1/ranking`: hybrid CPU+GPU. Nominal fit is CPU (f64, Hessian), per-nuisance refits use the configured GPU (CUDA f64 or Metal f32) when `gpu=true`.

### Model caching

Models are cached by SHA-256 hash of the workspace JSON. Pass `model_id` instead of `workspace` in fit/ranking requests to skip re-parsing. LRU eviction at 64 models by default.

### Python client

See `nextstat.remote` in the Python API reference (`docs/references/python-api.md`).

## Execution policy: CPU vs GPU

### Pharma / regulated environments

For pharma (FDA/EMA-regulated) and production-grade statistical analyses, **CPU-only execution** is the validated path:

- Deterministic f64 arithmetic, bit-reproducible across runs with `--threads 1`
- CPU farm mode (`--shard INDEX/TOTAL`) provides linear scale-out across cluster nodes
- Warm-start + retry + smooth bounds achieves >= 99.5% fittable convergence on validated models
- GPU paths (CUDA/Metal) are R&D-only; not part of the validated production pipeline

### HEP / exploratory R&D

GPU acceleration is useful for:
- Batch toys on large models (>= 150 parameters): GPU 6x faster than CPU
- Profile scans on large models: GPU marginally faster (1.07x at 184 params)
- Differentiable NLL for PyTorch integration (training, gradient-based optimization)

CPU wins for:
- Single-model MLE fits (GPU kernel launch overhead dominates)
- Small models (< 150 parameters) in any mode
- Batch toys on small models (< 50 parameters)

### Hessian skip policy

By default, `unbinned-fit-toys` skips Hessian computation on toys (uncertainties = 0). This saves ~40-50% compute per toy.

**Rationale**: toy-based hypothesis testing needs only q̃(μ) = 2·(NLL(μ) − NLL(μ̂)), which requires two MLE fits but no Hessian. This applies to both HEP CLs and pharma toy studies.

Hessian IS needed for:
- Parameter pulls: `(θ̂ − θ₀) / σ̂` requires σ̂ from the Hessian diagonal
- Ranking / impact plots: computed on the nominal fit, not on toys
- Wald approximation: asymptotic CLs uses Hessian for σ (one fit, not toy loop)

The CLI auto-enables Hessian when pull guardrails are active (`--max-abs-poi-pull-mean`, `--poi-pull-std-range`). Python users set `compute_hessian=True` explicitly.

## Benchmark matrix gate (pre-release)

Before each release, run the convergence benchmark matrix:

| Model | Events | Toys | Target fittable conv | Target wall-time |
|-------|--------|------|----------------------|-----------------|
| Gauss+Exp 10k | 10k | 10000 | >= 99.9% | baseline |
| CB 10k | 10k | 10000 | >= 99.5% | <= +10% vs prev |
| CB 100k | 100k | 10000 | >= 99.9% | <= +10% vs prev |

Run on CPU (`--threads 0`, 16+ cores recommended). Use `--shard` for cluster runs.

Metrics to track per model:
- `results.n_converged / (results.n_toys - results.n_validation_error)` = fittable convergence
- `results.n_validation_error` = spec/PDF errors (should be 0 after spec fix)
- `results.n_computation_error` = numeric failures
- Wall-clock time (from `--json-metrics`)
