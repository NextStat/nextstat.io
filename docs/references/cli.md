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

## Commands (high level)

For the configuration file format, see `docs/references/analysis-config.md`.

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
- `nextstat fit --input workspace.json [--gpu cuda]`
- `nextstat hypotest --input workspace.json --mu 1.0 [--expected-set]`
- `nextstat hypotest-toys --input workspace.json --mu 1.0 [--n-toys 1000 --seed 42] [--expected-set] [--threads 0] [--gpu cuda|metal]`
- `nextstat upper-limit --input workspace.json [--expected] [--scan-start ... --scan-stop ... --scan-points ...]`
- `nextstat scan --input workspace.json --start 0 --stop 5 --points 21 [--gpu cuda]`
- `nextstat viz profile --input workspace.json ...`
- `nextstat viz cls --input workspace.json ...`
- `nextstat viz ranking --input workspace.json`
- `nextstat viz pulls --input workspace.json --fit fit.json`
- `nextstat viz corr --input workspace.json --fit fit.json`
- `nextstat viz distributions --input workspace.json --histfactory-xml combination.xml [--fit fit.json]`
- `nextstat report --input workspace.json --histfactory-xml combination.xml --out-dir report/ [--fit fit.json] [--render]`
- `nextstat validation-report --apex2 master_report.json --workspace workspace.json --out validation_report.json [--pdf validation_report.pdf] [--deterministic]`
- `nextstat version`

Time series (Phase 8):
- `nextstat timeseries kalman-filter --input kalman_1d.json`
- `nextstat timeseries kalman-smooth --input kalman_1d.json`
- `nextstat timeseries kalman-em --input kalman_1d.json ...`
- `nextstat timeseries kalman-fit --input kalman_1d.json ...`
- `nextstat timeseries kalman-viz --input kalman_1d.json [--max-iter 50] [--level 0.95] [--forecast-steps ...]`
- `nextstat timeseries kalman-forecast --input kalman_1d.json ...`
- `nextstat timeseries kalman-simulate --input kalman_1d.json ...`

Survival analysis (Phase 9):
- `nextstat survival cox-ph-fit --input cox.json [--ties efron|breslow] [--no-robust] [--no-cluster-correction] [--no-baseline]`

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

Full details: `docs/references/pyhf-parity-contract.md`.

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

`nextstat report` writes multiple artifacts into `--out-dir` (currently: `distributions.json`, `pulls.json`, `corr.json`, `yields.json`, `uncertainty.json`, plus `yields.csv` and `yields.tex`). `uncertainty.json` is ranking-based and can be skipped via `--skip-uncertainty`. When `--render` is enabled it calls `python -m nextstat.report render ...` to produce a multi-page PDF and per-plot SVGs (requires `matplotlib`, see `nextstat[viz]` extra).

If `--fit` is omitted, `nextstat report` runs an MLE fit itself and writes `fit.json` into `--out-dir` before producing the report artifacts.

For time series input formats, see:
- `docs/tutorials/phase-8-timeseries.md`

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
  - `ConstraintType="Poisson"` => Barlow-Beeston `shapesys` (per-sample, name `staterror_<channel>_<sample>`).
  - `ConstraintType="Gaussian"` => `staterror` (per-channel, name `staterror_<channel>`).
- `StatError` histograms (`<StatError Activate="True" HistoName="...">`) are treated as **relative** per-bin uncertainties and converted to
  absolute `sigma_abs = rel * nominal`.
- If `<StatErrorConfig>` is omitted, NextStat matches ROOT/HistFactory default behavior by importing `staterror` and attaching per-bin
  `Gamma` constraint metadata to `measurements[].config.parameters[]` entries named `staterror_<channel>[i]` (non-standard extension).
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
