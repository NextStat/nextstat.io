---
title: "Unbinned GPU Publication Runbook (PF3.1)"
description: "Publication-grade protocol for large-scale unbinned GPU/CPU toy benchmarks: preflight, matrix execution, gates, and artifact packaging."
status: draft
last_updated: 2026-02-14
---

# Unbinned GPU Publication Runbook (PF3.1)

This is the execution protocol for publication-grade PF3.1 unbinned toy benchmarks.

Primary objective:

- Produce a reproducible benchmark snapshot for `CPU` vs `CUDA` (host/device/native, sharded, multi-GPU) on fixed specs with explicit correctness gates.

## 0) Recommended stand profile (publication tier)

Preferred:

- `4x H100 SXM` (80 GB each; total 320 GB VRAM)
- `>= 64 vCPU` (80 vCPU is sufficient)
- `>= 256 GB RAM` (752 GB is more than sufficient)
- CUDA driver/runtime with working `cuInit` at runtime
- isolated node (no external GPU jobs during run)

Acceptable fallback for smoke/protocol checks:

- `2x H100` or `4x L40` (L40 is not used for final headline performance numbers)

## 1) Scope and matrix

Matrix definition file:

- `benchmarks/unbinned/matrices/pf31_publication_v1.json`

Current cases:

- `gauss_exp_2m` (`2M events/toy`, `10k toys`)
- `gauss_exp_10k` (`10k events/toy`, `10k/50k toys`)
- `cb_exp_10k` (`10k toys`)
- `dcb_exp_10k` (`10k toys`)

Core publication gates (from matrix):

- all runs `rc == 0`
- convergence rate `== 100%`
- `n_error == 0`

### 1.1) PF3.3 gate matrix (CPU vs CUDA)

PF3.3 uses a smaller matrix intended for runtime policy gates (choose CPU vs CUDA).

Matrix definition file:

- `benchmarks/unbinned/matrices/pf33_gate_v1.json`

Additional specs (100k cases):

- `benchmarks/unbinned/specs/pf33_gauss_exp_100k.json`
- `benchmarks/unbinned/specs/pf33_cb_exp_100k.json`

## 2) One-command orchestrator

Use:

- `scripts/benchmarks/pf31_publication_matrix.sh`

What it does:

1. Expands matrix cases into concrete benchmark jobs.
2. Captures local/remote environment manifest (`local_git_status.txt`, `remote_env.txt`).
3. Runs `scripts/benchmarks/pf31_remote_matrix.sh` per case.
4. Builds publication summary:
   - `publication_summary.json`
   - `publication_summary.md`
5. Writes artifact hash index:
   - `snapshot_index.json`

## 3) Mandatory preflight (low-cost gate before matrix)

Run remote CUDA preflight first (quick smoke + environment capture):

```bash
PF31_HOST=<gpu-host> \
PF31_PORT=<ssh-port> \
PF31_USER=root \
PF31_KEY=~/.ssh/<key> \
PF31_MIN_GPUS=2 \
bash scripts/benchmarks/pf31_remote_preflight.sh
```

This writes a standalone preflight artifact bundle (`summary.json`, `summary.md`,
`remote_env.txt`, smoke outputs/metrics) under:

- `benchmarks/unbinned/artifacts/<YYYY-MM-DD>/pf31_preflight_<stamp>/`

Only if preflight is `PASS`, continue to dry-run and full matrix.

Always run dry-run first:

```bash
PF31_DRY_RUN=1 \
PF31_HOST=<gpu-host> \
PF31_PORT=<ssh-port> \
PF31_USER=root \
PF31_KEY=~/.ssh/<key> \
bash scripts/benchmarks/pf31_publication_matrix.sh
```

Dry-run guarantees:

- no SSH execution
- no benchmark run
- explicit command expansion in `planned_cases.tsv`

### 3.1) Latest smoke snapshot (2026-02-14)

- CUDA preflight smoke on GEX44 (`1x GPU`, explicit single-device override):
  - command policy: `PF31_MIN_GPUS=1 PF31_DEVICE_SETS="0" PF31_SMOKE_TOYS=64`
  - artifact root: `benchmarks/unbinned/artifacts/2026-02-14/pf31_preflight_20260214T123651Z/`
  - checks: `cuda1_host_smoke` and `cuda1_device_smoke` both `rc=0`
  - observed pipelines in metrics:
    - host: `cuda_gpu_native`
    - device+shards: `cuda_gpu_native_sharded`

- Metal smoke preflight in this Codex runtime environment:
  - artifact root: `benchmarks/unbinned/artifacts/2026-02-14/pf34_metal_20260214T123635Z/`
  - preflight result: `ok=false`, reason `Metal is not available at runtime (no Metal device found)`
  - implication: run Metal runtime matrix only on a host with visible Apple Metal device (outside this sandbox runtime).

### 3.2) Large-case checkpoint (2026-02-14, GEX44)

Case:

- spec: `benchmarks/unbinned/specs/pf31_gauss_exp_2m.json`
- toys: `10000`
- device set: `0` (single RTX 4000 SFF Ada 20GB)
- artifact root: `benchmarks/unbinned/artifacts/2026-02-11/pf31_matrix_20260214T124159Z/`

Observed:

- `cuda1_host_t10000`:
  - `rc=0`, `wall_time_s=1294.99` (~21.6 min)
  - `pipeline=cuda_gpu_native_sharded`
  - `device_shard_plan_len=17`
  - fit quality: `n_converged=10000`, `n_error=0`, `n_nonconverged=0`
- `cuda1_native_t10000`:
  - `rc=0`, `wall_time_s=1444.04` (~24.1 min)
  - `pipeline=cuda_gpu_native_sharded`
  - `device_shard_plan_len=17`
  - fit quality: `n_converged=10000`, `n_error=0`, `n_nonconverged=0`

Device-resident sharded failure modes on this 20GB card:

- `--gpu-sample-toys --gpu-shards 4`: validation fail, per-shard toy-offset overflow `u32`
- `--gpu-sample-toys --gpu-shards 8`: runtime fail, `CUDA_ERROR_OUT_OF_MEMORY`
- `--gpu-sample-toys --gpu-shards 17`: process terminated externally (`rc=143`, empty stderr), needs dedicated stabilization/debug task

## 4) Real execution

Run only on isolated stand (no external workload contention):

```bash
PF31_DRY_RUN=0 \
PF31_HOST=<gpu-host> \
PF31_PORT=<ssh-port> \
PF31_USER=root \
PF31_KEY=~/.ssh/<key> \
PF31_REMOTE_REPO=/root/nextstat.io \
PF31_BIN=/root/nextstat.io/target/release/nextstat \
PF31_GPU_DEVICE_SETS="0;0,1" \
PF31_THREADS=1 \
bash scripts/benchmarks/pf31_publication_matrix.sh
```

Notes:

- `PF31_GPU_DEVICE_SETS` can be omitted (auto-detect in underlying runner).
- `PF31_THREADS` controls `nextstat --threads` across the matrix (`default=1`).
- `gauss_exp_2m` case intentionally uses higher shard counts to stay within safe 32-bit toy-offset budget.

### 4.1) PF3.3 gate run (GEX44 / 1 GPU)

```bash
PF31_PUBLICATION_MATRIX=benchmarks/unbinned/matrices/pf33_gate_v1.json \
PF31_DRY_RUN=0 \
PF31_HOST=88.198.23.172 \
PF31_PORT=22 \
PF31_USER=root \
PF31_KEY=~/.ssh/rundesk_hetzner \
PF31_REMOTE_REPO=/root/nextstat.io \
PF31_BIN=/root/nextstat.io/target/release/nextstat \
PF31_GPU_DEVICE_SETS=0 \
PF31_THREADS=20 \
bash scripts/benchmarks/pf31_publication_matrix.sh
```

Strict report writer (numbers-only table format per `.claude/benchmark-protocol.md`):

```bash
python3 scripts/benchmarks/pf33_gate_report.py \
  --run-root <RUN_ROOT_FROM_pf31_publication_matrix> \
  --matrix <RUN_ROOT_FROM_pf31_publication_matrix>/matrix.json \
  --out-md docs/benchmarks/pf33_unbinned_gate_<YYYY-MM-DD>.md
```

### CUDA4 Saturation Spot-Check (optional, recommended on expensive stands)

Before/after full matrix, run one `cuda4`-only stress to verify the stand can hold sustained 4-GPU load:

- use `--gpu cuda --gpu-devices 0,1,2,3 --gpu-sample-toys --gpu-shards 16`
- collect `nvidia-smi` samples at 1s cadence during the run
- archive `gpu_util.csv` with run metrics

Important constraint:

- keep `n_toys * expected_events_per_toy <= u32::MAX` for toy-offset safety
- e.g. `gauss_exp_2m` with `100k toys` is invalid (overflows 32-bit toy offsets)

Reference artifacts (A100 4x80GB, 2026-02-14):

- `benchmarks/unbinned/artifacts/2026-02-14/pf31_cuda4_only_20260214T121023Z`
- `benchmarks/unbinned/artifacts/2026-02-14/pf31_cuda4_only_20260214T121214Z_t300k`

## 5) Output bundle contract

The run creates one bundle root:

- `benchmarks/unbinned/artifacts/<YYYY-MM-DD>/<run_id>/`

Minimal files required for publication review:

- `matrix.json`
- `run.meta`
- `remote_env.txt`
- per-case artifacts (`*.meta.json`, `*.metrics.json`, `*.out.json`, `*.err`)
- `publication_summary.json`
- `publication_summary.md`
- `snapshot_index.json`

## 6) Review checklist before publication

1. `publication_summary.json` has `overall_pass=true`.
2. No gate failures (`gate_failures=[]`).
3. Environment manifest is complete (`remote_env.txt` includes GPU model, driver, nvcc/runtime).
4. `snapshot_index.json` exists and hashes the full bundle.
5. Public docs tables are updated from this bundle (no hand-edited numbers).

## 7) Follow-up packaging (DOI/replication)

For DOI-grade publication and third-party replication process, use:

- `docs/benchmarks/publishing.md`
- `docs/benchmarks/replication.md`
- `docs/benchmarks/first-public-snapshot.md`

This runbook focuses on the PF3.1 unbinned benchmark execution layer.
