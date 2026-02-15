# Unbinned Cross-Framework Benchmark Suite

This directory contains a small, reproducible benchmark harness that runs the
same unbinned toy problems through:

- NextStat (`nextstat unbinned-fit`)
- RooFit (via `root -b -q`)
- zfit (optional; Python)
- MoreFit (optional; C++ runner)

The goal is to compare:

- best-fit parameter values
- parameter uncertainties (where available)
- NLL values (note: conventions differ across frameworks)
- wall-clock time

For a strict CPU apples-to-apples rerun between NextStat and MoreFit (separating
`wall` vs warm in-process fit timing), use:

- `benchmarks/unbinned/bench_cpu_symmetry.py`

## Run

```bash
./.venv/bin/python3 benchmarks/unbinned/run_suite.py --out tmp/unbinned_bench.json
```

If you want to provide an explicit NextStat CLI binary:

```bash
NS_CLI_BIN=target/release/nextstat python3 benchmarks/unbinned/run_suite.py
```

RooFit requires `root` on `PATH`. zfit/MoreFit are optional and will be skipped
if not importable.

## Symmetric CPU benchmark (NextStat vs MoreFit)

```bash
NS_CLI_BIN=target/release/nextstat \
python3 benchmarks/unbinned/bench_cpu_symmetry.py \
  --n-events 10000,100000,1000000 \
  --seeds 42,43 \
  --threads 20
```

Optional NextStat optimizer overrides (passed to both CLI and Python library paths):

- `--nextstat-opt-max-iter`
- `--nextstat-opt-tol`
- `--nextstat-opt-m`
- `--nextstat-opt-smooth-bounds`

## PF3.1 CUDA shard matrix (remote)

Use `scripts/benchmarks/pf31_remote_matrix.sh` for remote toy-fit shard matrix runs.
Defaults target a Hetzner CUDA host and auto-detect whether to run single-device
(`0`) or dual-device (`0;0,1`) cases.

Host analytical CUDA modes are now explicit and labeled in artifacts:

- `PF31_HOST_FIT_MODES=host,native` (default) runs both:
  - host lockstep (`..._host_t...`)
  - explicit native (`..._native_t...`, adds `--gpu-native`)
- summary rows include `fit_mode` (`cpu|host|native|host_sharded|device_sharded`)

```bash
PF31_HOST=88.198.23.172 \
PF31_KEY=~/.ssh/rundesk_hetzner \
PF31_BIN=/root/nextstat.io/target/release/nextstat \
PF31_SPEC=/root/nextstat.io/benchmarks/unbinned/specs/pf31_gauss_exp_10k.json \
PF31_TOYS=10000,100000 \
PF31_SHARDS=2,4,8 \
PF31_HOST_FIT_MODES=host,native \
PF31_THREADS=1 \
bash scripts/benchmarks/pf31_remote_matrix.sh
```

Threading note:
- `PF31_THREADS` controls `nextstat --threads` for all cases in this run (default `1`).

## PF3.1 publication matrix bundle (prepared runbook)

For publication-grade GPU benchmark snapshots (multi-case matrix, environment
manifest, summary gates, snapshot index), use:

- matrix: `benchmarks/unbinned/matrices/pf31_publication_v1.json`
- remote preflight gate: `scripts/benchmarks/pf31_remote_preflight.sh`
- orchestrator: `scripts/benchmarks/pf31_publication_matrix.sh`
- report builder: `scripts/benchmarks/pf31_publication_report.py`
- docs runbook: `docs/benchmarks/unbinned-publication-runbook.md`

Run remote preflight first (cheap fail-fast gate):

```bash
PF31_HOST=<gpu-host> \
PF31_PORT=<ssh-port> \
PF31_KEY=~/.ssh/<key> \
bash scripts/benchmarks/pf31_remote_preflight.sh
```

Dry-run (required before expensive stand execution):

```bash
PF31_DRY_RUN=1 \
PF31_HOST=<gpu-host> \
PF31_PORT=<ssh-port> \
PF31_KEY=~/.ssh/<key> \
bash scripts/benchmarks/pf31_publication_matrix.sh
```

Defaults for MoreFit binaries:

- `/root/morefit/morefit_gauss_exp`
- `/root/morefit/morefit_gauss_exp_agrad`
- `/root/morefit/morefit_gauss_exp_mt`
- `/root/morefit/morefit_gauss_exp_agrad_mt`

Override with:

- `--morefit-1t-num`
- `--morefit-1t-agrad`
- `--morefit-20t-num`
- `--morefit-20t-agrad`

## PF3.1 legacy artifact migration (strict schemas)

Older PF3.1 runs emitted `*.meta.json` and per-case `summary.json` without
`schema_version`. To retroactively apply strict schema validation, migrate the
run directory in-place:

```bash
python3 scripts/benchmarks/pf31_migrate_legacy_artifacts.py \
  --root benchmarks/unbinned/artifacts/<YYYY-MM-DD>/<PF31_RUN_ID>
```

## Validate JSON artifacts

Validate schema'd artifacts (recommended for new runs):

```bash
python3 benchmarks/unbinned/validate_artifacts.py --strict tmp/unbinned_bench.json
python3 benchmarks/unbinned/validate_artifacts.py benchmarks/unbinned/matrices/pf31_publication_v1.json
python3 benchmarks/unbinned/validate_artifacts.py benchmarks/unbinned/specs
```

To validate an artifacts directory that may contain legacy JSONs without `schema_version`,
omit `--strict`:

```bash
python3 benchmarks/unbinned/validate_artifacts.py benchmarks/unbinned/artifacts
```

## PF3.4 Metal matrix (local Apple Silicon)

PF3.4 introduces a dedicated local runner for Metal throughput baselines and
telemetry collection across:

- `unbinned-fit-toys` (`fit_host`, `fit_device`)
- `unbinned-hypotest-toys` (`hypotest_host`, `hypotest_device`)

Matrix and schema:

- `benchmarks/unbinned/matrices/pf34_metal_v1.json`
- `benchmarks/unbinned/schemas/pf34_metal_matrix_v1.schema.json`

Run dry-run first:

```bash
python3 scripts/benchmarks/pf34_metal_matrix.py --dry-run
```

Run real benchmark:

```bash
PF34_BIN=target/release/nextstat \
python3 scripts/benchmarks/pf34_metal_matrix.py \
  --matrix benchmarks/unbinned/matrices/pf34_metal_v1.json \
  --threads 1
```

The runner performs a fail-fast Metal preflight before the full matrix
(`preflight.json` in the artifact directory). If no Metal device is available,
the run exits early with a clear runtime reason.

Outputs are written to `benchmarks/unbinned/artifacts/<date>/pf34_metal_<stamp>/`:

- per-run: `*.meta.json`, `*.metrics.json`, `*.out.json`, `*.err`
- aggregate: `run_manifest.json`, `summary.json`, `summary.md`
