---
title: "CLI Reference (nextstat)"
status: stable
---

# CLI Reference (nextstat)

The `nextstat` CLI is implemented in `crates/ns-cli` and focuses on:
- deterministic parity mode (`--threads 1`)
- JSON in / JSON out contracts for reproducible workflows

## Commands (high level)

HEP / HistFactory:
- `nextstat validate --config analysis.yaml`
- `nextstat config schema [--name analysis_spec_v0]`
- `nextstat import histfactory --xml combination.xml --output workspace.json`
- `nextstat import trex-config --config trex.txt --output workspace.json [--analysis-yaml analysis.yaml] [--coverage-json coverage.json]`
- `nextstat import patchset --workspace BkgOnly.json --patchset patchset.json [--patch-name ...]`
- `nextstat build-hists --config trex.config --out-dir out/ [--base-dir ...] [--coverage-json coverage.json]`
- `nextstat trex import-config --config trex.config --out analysis.yaml [--report analysis.mapping.json]`
- `nextstat fit --input workspace.json [--gpu]`
- `nextstat hypotest --input workspace.json --mu 1.0 [--expected-set]`
- `nextstat hypotest-toys --input workspace.json --mu 1.0 [--n-toys 1000 --seed 42] [--expected-set] [--threads 0] [--gpu]`
- `nextstat upper-limit --input workspace.json [--expected] [--scan-start ... --scan-stop ... --scan-points ...]`
- `nextstat scan --input workspace.json --start 0 --stop 5 --points 21 [--gpu]`
- `nextstat viz profile --input workspace.json ...`
- `nextstat viz cls --input workspace.json ...`
- `nextstat viz ranking --input workspace.json`
- `nextstat viz pulls --input workspace.json --fit fit.json`
- `nextstat viz corr --input workspace.json --fit fit.json`
- `nextstat viz distributions --input workspace.json --histfactory-xml combination.xml [--fit fit.json]`
- `nextstat report --input workspace.json --histfactory-xml combination.xml --out-dir report/ [--fit fit.json] [--render]`

Time series (Phase 8):
- `nextstat timeseries kalman-filter --input kalman_1d.json`
- `nextstat timeseries kalman-smooth --input kalman_1d.json`
- `nextstat timeseries kalman-em --input kalman_1d.json ...`
- `nextstat timeseries kalman-fit --input kalman_1d.json ...`
- `nextstat timeseries kalman-forecast --input kalman_1d.json ...`
- `nextstat timeseries kalman-simulate --input kalman_1d.json ...`

## GPU acceleration

The `--gpu` flag enables CUDA GPU acceleration (requires `cuda` feature and an NVIDIA GPU at runtime). If no CUDA GPU is available at runtime, the command exits with an error. Without `--gpu`, the standard CPU (SIMD/Rayon + Accelerate) path is used.

Build with CUDA support:

```bash
cargo build -p ns-cli --features cuda --release
```

### Supported commands

**`fit --gpu`** — Single-model MLE fit on GPU. Uses `GpuSession` for fused NLL+gradient (1 kernel launch per L-BFGS iteration). Hessian computed via finite differences of GPU gradient at the end.

```bash
nextstat fit --input workspace.json --gpu
```

**`scan --gpu`** — GPU-accelerated profile likelihood scan. A single `GpuSession` is shared across all scan points with warm-start between mu values.

```bash
nextstat scan --input workspace.json --start 0 --stop 5 --points 21 --gpu
```

**`hypotest-toys --gpu`** — Batch toy fitting on GPU. The lockstep GPU batch optimizer computes NLL + analytical gradient for all toys in a single kernel launch per iteration.

```bash
nextstat hypotest-toys --input workspace.json --mu 1.0 --n-toys 10000 --gpu
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

Full details: `docs/pyhf-parity-contract.md`.

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

## Import notes

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
- If the config includes `Sample:` blocks, they act as an **include-list** for samples; per-sample `Regions:` filters
  are respected to mask samples per-channel. Empty channels are dropped unless channels were explicitly selected via
  `Region:` blocks (then it is an error).
- When `HistoPath` is provided, it is used as the **base directory** for resolving relative paths inside the HistFactory XML
  (common when `combination.xml` lives under `config/` but ROOT inputs are referenced from the export root).

Optional outputs:
- `--analysis-yaml` writes an analysis spec v0 wrapper (`inputs.mode=trex_config_txt`) to drive `nextstat run`.
- `--coverage-json` writes a best-effort report of unknown keys/attrs to help parity work against legacy configs.

`nextstat trex import-config` is a best-effort migration helper for TRExFitter `.config` files:
- it emits an analysis spec v0 YAML using `inputs.mode=trex_config_yaml`
- it also writes a mapping report listing mapped and unmapped keys

Example config: `docs/examples/trex_config_ntup_minimal.txt`.
Example config (HIST wrapper): `docs/examples/trex_config_hist_minimal.txt`.
