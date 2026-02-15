---
title: "Unbinned Benchmark Reproducibility"
description: "Canonical runbook for reproducing the unbinned cross-framework benchmark suite (NextStat, RooFit, zfit, MoreFit), including exact commands and output JSON schema contract."
status: published
last_updated: 2026-02-13
---

# Unbinned Benchmark Reproducibility

This page is the single source of truth for reproducing the unbinned benchmark
suite and validating the output artifact contract.

## 1) Commands

### 1.1 Local (NextStat only)

```bash
cargo build --release -p ns-cli
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip pyarrow numpy scipy
NS_CLI_BIN=target/release/nextstat \
  .venv/bin/python3 benchmarks/unbinned/run_suite.py \
  --cases gauss_exp,cb_exp,product2d \
  --n-events 100000 \
  --seed 42 \
  --out tmp/unbinned_bench.json
```

### 1.2 With RooFit + zfit

Prerequisites:
- `root` available on `PATH`
- `zfit` installed in `.venv`

```bash
source .venv/bin/activate
pip install zfit tensorflow
NS_CLI_BIN=target/release/nextstat \
  .venv/bin/python3 benchmarks/unbinned/run_suite.py \
  --cases gauss_exp,cb_exp,product2d \
  --n-events 100000 \
  --seed 42 \
  --out tmp/unbinned_bench_full.json
```

### 1.3 Hetzner / MoreFit runner

Prerequisites:
- built MoreFit runner binary:
  `benchmarks/unbinned/morefit_gauss_exp.cc`
- binary path exported via `MOREFIT_BIN`

```bash
source .venv/bin/activate
export MOREFIT_BIN=/absolute/path/to/morefit_gauss_exp
NS_CLI_BIN=target/release/nextstat \
  .venv/bin/python3 benchmarks/unbinned/run_suite.py \
  --cases gauss_exp \
  --n-events 100000 \
  --seed 42 \
  --out tmp/unbinned_bench_morefit.json
```

### 1.4 Symmetric CPU rerun (NextStat vs MoreFit)

This run separates:
- external wall time (`subprocess` wall)
- warm in-process fit time (`times_ms`, cold run excluded)
- NextStat library-level fit time (`nextstat.fit()` via Python bindings)

```bash
# One-time: make Python bindings importable in system Python
python3 -m pip install --break-system-packages -U maturin
python3 -m pip install --break-system-packages -e bindings/ns-py

NS_CLI_BIN=target/release/nextstat \
python3 benchmarks/unbinned/bench_cpu_symmetry.py \
  --n-events 10000,100000,1000000 \
  --seeds 42,43 \
  --threads 20 \
  --morefit-1t-num /root/morefit/morefit_gauss_exp \
  --morefit-1t-agrad /root/morefit/morefit_gauss_exp_agrad \
  --morefit-20t-num /root/morefit/morefit_gauss_exp_mt \
  --morefit-20t-agrad /root/morefit/morefit_gauss_exp_agrad_mt
```

Example artifact from this protocol:
- baseline: `benchmarks/unbinned/artifacts/2026-02-11/bench_symm_cpu_gex44_diag2_20260211T225229Z.json`

Tuned run (recommended for constrained unbinned fits):

```bash
NS_CLI_BIN=target/release/nextstat \
python3 benchmarks/unbinned/bench_cpu_symmetry.py \
  --n-events 10000,100000,1000000 \
  --seeds 42,43 \
  --threads 20 \
  --nextstat-opt-m 20 \
  --nextstat-opt-smooth-bounds
```

Example tuned artifact:
- `benchmarks/unbinned/artifacts/2026-02-11/bench_symm_cpu_gex44_tune2_m20smooth_20260211T225807Z.json`

### 1.5 PF3.1 CUDA sharded toy-fit matrix (remote)

This runbook executes CPU + CUDA host + CUDA device-sharded toy fits and pulls all
artifacts (`*.meta.json`, `*.metrics.json`, `*.out.json`, `*.err`) locally.

```bash
PF31_HOST=88.198.23.172 \
PF31_KEY=~/.ssh/rundesk_hetzner \
PF31_BIN=/root/nextstat.io/target/release/nextstat \
PF31_SPEC=/root/nextstat.io/benchmarks/unbinned/specs/pf31_gauss_exp_10k.json \
PF31_TOYS=10000,100000 \
PF31_SHARDS=2,4,8 \
bash scripts/benchmarks/pf31_remote_matrix.sh
```

Notes:
- `PF31_GPU_DEVICE_SETS` is optional; when unset, the script auto-detects remote GPUs and uses:
  - `0` for single-GPU hosts
  - `0;0,1` for 2+ GPU hosts
- Set `PF31_GPU_DEVICE_SETS="0;0,1"` explicitly if you need strict 2-device parity runs.
- Set `PF31_INCLUDE_HOST_SHARDED=1` to include `--gpu-shards N` without `--gpu-sample-toys`.

### 1.6 CPU farm cluster run (HEP, SSH multi-host)

Use this path when GPU is unavailable or when you want CPU farm throughput on a
university cluster. The run is split into three explicit steps.

1) Preflight topology probe:

```bash
python3 scripts/farm/preflight_cluster.py \
  --hosts-file hosts.txt \
  --ssh-user analysis \
  --ssh-key ~/.ssh/hep_cluster_ed25519 \
  --out tmp/farm/preflight.json
```

2) Distributed toy execution (`n_toys` split by physical cores):

```bash
python3 scripts/farm/run_unbinned_fit_toys_cluster.py \
  --hosts-file hosts.txt \
  --preflight-json tmp/farm/preflight.json \
  --config /abs/path/spec.json \
  --n-toys 10000 \
  --seed 42 \
  --threads physical \
  --weight-mode physical \
  --ssh-user analysis \
  --ssh-key ~/.ssh/hep_cluster_ed25519 \
  --nextstat-bin /shared/nextstat/target/release/nextstat \
  --remote-workdir /shared/nextstat \
  --local-out-dir tmp/farm/runs
```

3) Merge shard outputs into one canonical artifact:

```bash
python3 scripts/farm/merge_unbinned_toys_results.py \
  --manifest tmp/farm/runs/<run_id>/manifest.json \
  --out tmp/farm/runs/<run_id>/merged.out.json
```

Notes:
- `hosts.txt` supports per-host overrides: `weight=...`, `threads=...`, `user=...`, `port=...`.
- For deterministic planning checks, run step (2) first with `--dry-run`.
- The launcher stores full scheduling metadata (`toy_start`, shard seed, host, threads) in `manifest.json`.

### 1.6b CPU farm cluster run (HEP, SLURM/HTCondor scheduler)

Use this path when you have a scheduler and a shared filesystem and want the
same deterministic shard contract.

1) Render scheduler jobs + manifest:

```bash
python3 scripts/farm/render_unbinned_fit_toys_scheduler.py \
  --scheduler slurm \
  --config /shared/spec.json \
  --n-toys 10000 \
  --seed 42 \
  --shards 100 \
  --threads 0 \
  --nextstat-bin /shared/nextstat/target/release/nextstat \
  --out-dir /shared/runs \
  --slurm-cpus-per-task 20 \
  --slurm-time 04:00:00
```

2) Submit:

```bash
sbatch /shared/runs/<run_id>/slurm_array_job.sh
```

3) Collect results into manifest:

```bash
python3 scripts/farm/collect_unbinned_fit_toys_scheduler.py \
  --manifest /shared/runs/<run_id>/manifest.json \
  --in-place
```

4) Merge:

```bash
python3 scripts/farm/merge_unbinned_toys_results.py \
  --manifest /shared/runs/<run_id>/manifest.json \
  --out /shared/runs/<run_id>/merged.out.json
```

### 1.7 PF3.1 publication matrix (GPU/CPU, snapshot-grade bundle)

For publication-grade PF3.1 runs use the dedicated orchestrator:

- runbook: `docs/benchmarks/unbinned-publication-runbook.md`
- matrix definition: `benchmarks/unbinned/matrices/pf31_publication_v1.json`
- orchestrator: `scripts/benchmarks/pf31_publication_matrix.sh`

Dry-run first (mandatory, no GPU spend):

```bash
PF31_DRY_RUN=1 \
PF31_HOST=<gpu-host> \
PF31_PORT=<ssh-port> \
PF31_USER=root \
PF31_KEY=~/.ssh/<key> \
bash scripts/benchmarks/pf31_publication_matrix.sh
```

Real run:

```bash
PF31_DRY_RUN=0 \
PF31_HOST=<gpu-host> \
PF31_PORT=<ssh-port> \
PF31_USER=root \
PF31_KEY=~/.ssh/<key> \
PF31_GPU_DEVICE_SETS="0;0,1" \
bash scripts/benchmarks/pf31_publication_matrix.sh
```

Expected bundle outputs include:

- `publication_summary.json` + `publication_summary.md` (gates and per-case metrics)
- `snapshot_index.json` (hash inventory for audit/replication)

## 2) Output Contract

`benchmarks/unbinned/run_suite.py` writes one JSON artifact per run with this
contract:

- top-level:
  - `schema_version = "nextstat.unbinned_run_suite_result.v1"`
  - `suite = "benchmarks/unbinned/run_suite.py"`
  - `seed`, `n_events`
  - `availability` (`root`, `zfit`, `morefit`)
  - `cases[]`
- per-case:
  - `case` (`gauss_exp`, `cb_exp`, `product2d`)
  - `kind` (`gauss_exp_ext`, `cb_exp_ext`, `product2d`)
  - tool blocks: `nextstat`, `roofit`, `zfit`, `morefit`
- tool block semantics:
  - success payload (tool-specific metrics like `nll`, timing fields)
  - or `{ "skipped": true, "reason": ... }`
  - or `{ "failed": true, "reason": ... }`

Canonical JSON Schema:
- `docs/schemas/benchmarks/unbinned_run_suite_result_v1.schema.json`

Symmetric CPU artifact schema:
- `schema_version = "nextstat.unbinned_cpu_symmetry_bench.v1"`
- top-level includes `records[]` (per N×seed raw runs) and `summary` (median-of-seeds rollup)

## 3) Validation (recommended)

Use JSON Schema validation in CI or locally:

```bash
python3 - <<'PY'
import json
from pathlib import Path
import jsonschema

schema = json.loads(Path("docs/schemas/benchmarks/unbinned_run_suite_result_v1.schema.json").read_text())
data = json.loads(Path("tmp/unbinned_bench.json").read_text())
jsonschema.validate(data, schema)
print("ok: unbinned benchmark artifact matches schema")
PY
```

CI guard (already wired):
- `.github/workflows/python-tests.yml`, job `validation-pack`
- runs `benchmarks/unbinned/run_suite.py --cases gauss_exp --n-events 2000 --seed 42`
- validates output against `docs/schemas/benchmarks/unbinned_run_suite_result_v1.schema.json`
- uploads `artifacts/unbinned_bench_smoke.json` as workflow artifact
- compares against previous successful baseline artifact using:
  - `scripts/benchmarks/compare_unbinned_bench.py`
  - gate: `nextstat._wall_ms` regression ratio > `2.5x` fails CI
  - summary artifact: `artifacts/unbinned_bench_drift_summary.json`

## 4) Result Tables

Published benchmark tables and interpretation:
- `docs/benchmarks/unbinned-benchmark-suite.md`
