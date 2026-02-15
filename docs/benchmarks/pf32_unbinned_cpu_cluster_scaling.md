---
title: "PF3.2 CPU Farm Scaling (Unbinned HEP Toys)"
description: "Runbook + artifact contract for CPU-only unbinned-fit-toys scaling on a university cluster (SLURM/HTCondor/SSH farm)."
status: draft
last_updated: 2026-02-15
---

# PF3.2 CPU Farm Scaling (Unbinned HEP Toys)

Goal: validate CPU-only scale-out for realistic HEP loads (O(2M events/toy), O(10k toys)) and produce rerunnable artifacts.

This document intentionally focuses on reproducibility:
- deterministic shard split (`--shard INDEX/TOTAL`)
- deterministic per-shard seeds (`seed + toy_start`)
- canonical merge artifact (`merged.out.json`)

## Workload

Reference spec:
- `benchmarks/unbinned/specs/pf31_gauss_exp_2m.json` (expected `~2,000,000 events/toy` via yields)

Typical production shape:
- `n_toys=10000`
- `threads`: match allocated CPU cores per job (or `0` for auto if the scheduler enforces CPU affinity)

## Option A: SLURM scheduler (recommended)

### 1) Build NextStat binary on shared filesystem

Example:
- `/shared/nextstat/target/release/nextstat`

### 2) Render an array job + manifest

```bash
python3 scripts/farm/render_unbinned_fit_toys_scheduler.py \
  --scheduler slurm \
  --config /shared/nextstat/benchmarks/unbinned/specs/pf31_gauss_exp_2m.json \
  --n-toys 10000 \
  --seed 42 \
  --shards 200 \
  --threads 0 \
  --nextstat-bin /shared/nextstat/target/release/nextstat \
  --out-dir /shared/nextstat_runs \
  --slurm-cpus-per-task 20 \
  --slurm-time 08:00:00
```

Notes:
- `--shards` controls parallelism (number of array tasks). Choose based on cluster quota.
- `--threads 0` uses NextStat auto threads; safe only if the scheduler constrains CPUs per task.

### 3) Submit

```bash
sbatch /shared/nextstat_runs/<run_id>/slurm_array_job.sh
```

### 4) Collect results into manifest

```bash
python3 scripts/farm/collect_unbinned_fit_toys_scheduler.py \
  --manifest /shared/nextstat_runs/<run_id>/manifest.json \
  --in-place
```

### 5) Merge

```bash
python3 scripts/farm/merge_unbinned_toys_results.py \
  --manifest /shared/nextstat_runs/<run_id>/manifest.json \
  --out /shared/nextstat_runs/<run_id>/merged.out.json
```

## Option B: HTCondor scheduler

Render:

```bash
python3 scripts/farm/render_unbinned_fit_toys_scheduler.py \
  --scheduler htcondor \
  --config /shared/nextstat/benchmarks/unbinned/specs/pf31_gauss_exp_2m.json \
  --n-toys 10000 \
  --seed 42 \
  --shards 200 \
  --threads 1 \
  --nextstat-bin /shared/nextstat/target/release/nextstat \
  --out-dir /shared/nextstat_runs \
  --condor-request-cpus 1 \
  --condor-request-memory "2GB"
```

Notes:
- If one execute node is flaky or unreachable (common symptom: negotiator says “Successfully matched” but the job stays `Idle`), exclude it with `--condor-requirements`, e.g.:

```bash
  --condor-requirements '(Machine != "nextstat-gex44")'
```

Submit:

```bash
condor_submit /shared/nextstat_runs/<run_id>/condor_job.sub
```

If execute nodes do not have access to the absolute paths in `manifest.json`
(no shared filesystem between submit+execute), submit the transfer-mode file instead:

```bash
condor_submit /shared/nextstat_runs/<run_id>/condor_job_transfer.sub
```

Collect + merge are identical to the SLURM flow.

## Result Snapshot (HTCondor Pool)

Run date: 2026-02-15

- Workload: `pf31_gauss_exp_2m.json` (Gauss+Exp, ~2M events via yields), `n_toys=10000`
- Scheduler: HTCondor
- Shards: `74`
- Threads per shard: `1`
- Requirements: excluded `nextstat-gex44` (jobs were being matched to gex44 but not starting; likely claim/file-transfer/connectivity issue)

Result:
- Convergence: `10000/10000` converged, `0` errors
- Makespan (max shard elapsed): `249s`
- Throughput: `~40.2 toys/s`

Artifacts (submit node):
- run dir: `/home/actions-runner/nextstat_runs/pf32_full_2m10k_sh74_20260215T134329Z`
- merged: `/home/actions-runner/nextstat_runs/pf32_full_2m10k_sh74_20260215T134329Z/merged.out.json`

### Additional Snapshot: All Nodes (Including gex44)

Run date: 2026-02-15

- Workload: `pf31_gauss_exp_2m.json`, `n_toys=10000`
- Scheduler: HTCondor
- Shards: `74`
- Threads per shard: `1`
- Requirements: none (pool included `nextstat-bench`, `nextstat-coolify`, `nextstat-gex44`)

Result:
- Convergence: `10000/10000` converged, `0` errors
- Host distribution: `nextstat-bench=58`, `nextstat-coolify=10`, `nextstat-gex44=6`
- Makespan (max shard elapsed): `344s`
- Throughput: `~29.1 toys/s`

Artifacts:
- run dir: `/home/actions-runner/nextstat_runs/pf32_full_2m10k_sh74_allnodes_20260215T150040Z`
- merged: `/home/actions-runner/nextstat_runs/pf32_full_2m10k_sh74_allnodes_20260215T150040Z/merged.out.json`

Notes:
- This run is **not** directly comparable to the earlier `249s` snapshot: it was executed later in the day and the tail was dominated by shards running on `nextstat-bench` (likely higher contention / different background load at the time of measurement).

## Option C: SSH multi-host farm (no scheduler)

Use `scripts/farm/preflight_cluster.py` + `scripts/farm/run_unbinned_fit_toys_cluster.py` + `scripts/farm/merge_unbinned_toys_results.py`.

See:
- `scripts/farm/README.md`
- `docs/benchmarks/unbinned-reproducibility.md` (section CPU farm cluster run)

## Artifact contract

Minimal archive set:
- `<run_id>/manifest.json`
- `<run_id>/shard_*.out.json`
- `<run_id>/shard_*.metrics.json` (optional but recommended)
- `<run_id>/merged.out.json`

## Recording environment

At minimum record:
- cluster scheduler (`slurm`/`htcondor`)
- CPU model (from `lscpu | grep 'Model name'`)
- `rustc --version`
- `nextstat --version`

## Shard sweep (scaling matrix)

To test scaling, run the same workload with several shard counts (e.g. `50,100,200,400`)
and compare the makespan (max shard elapsed) and throughput.

Render all runs + sweep manifest:

```bash
python3 scripts/farm/pf32_unbinned_shard_sweep.py render \
  --scheduler slurm \
  --config /shared/nextstat/benchmarks/unbinned/specs/pf31_gauss_exp_2m.json \
  --n-toys 10000 \
  --seed 42 \
  --threads 0 \
  --nextstat-bin /shared/nextstat/target/release/nextstat \
  --out-root /shared/nextstat_runs \
  --sweep-id pf32_hep_2m10k \
  --shards-list 50,100,200,400 \
  --slurm-cpus-per-task 20 \
  --slurm-time 08:00:00
```

After jobs finish:

```bash
python3 scripts/farm/pf32_unbinned_shard_sweep.py collect --sweep-json /shared/nextstat_runs/pf32_hep_2m10k.sweep.json
python3 scripts/farm/pf32_unbinned_shard_sweep.py merge --sweep-json /shared/nextstat_runs/pf32_hep_2m10k.sweep.json
python3 scripts/farm/pf32_unbinned_shard_sweep.py summarize \
  --sweep-json /shared/nextstat_runs/pf32_hep_2m10k.sweep.json \
  --out-md /shared/nextstat_runs/pf32_hep_2m10k.summary.md
```

Optional (HTCondor only): run the sweep sequentially from the submit node (avoids contention)
and automatically run collect+merge after each run directory finishes:

```bash
python3 scripts/farm/pf32_unbinned_shard_sweep.py submit \
  --sweep-json /shared/nextstat_runs/pf32_hep_2m10k.sweep.json \
  --mode transfer \
  --poll-s 10
```

### Snapshot: HTCondor Shard Sweep (bench+coolify, transfer mode, 2GB)

Run date: 2026-02-15

- Workload: `pf31_gauss_exp_2m.json`, `n_toys=10000`
- Scheduler: HTCondor
- Threads per shard: `1`
- Submit file: `condor_job_transfer.sub` (execute nodes do not need a shared filesystem)
- Requirements: excluded `nextstat-gex44` (bench+coolify only)
- `request_memory`: `2GB` (allows higher concurrency vs `4GB` which caps `nextstat-bench` at ~30 slots)

Sweep:
- sweep json: `/home/actions-runner/nextstat_runs/pf32_2m10k_shard_sweep_20260215T152638Z_transfer2gb_nogex44.sweep.json`

| Shards | Makespan | Throughput | Converged | Errors | Run dir |
|---:|---:|---:|---:|---:|---|
| 40 | 425s | 23.529 toys/s | 10000 | 0 | `/home/actions-runner/nextstat_runs/pf32_2m10k_shard_sweep_20260215T152638Z_transfer2gb_nogex44_sh40` |
| 74 | 359s | 27.855 toys/s | 10000 | 0 | `/home/actions-runner/nextstat_runs/pf32_2m10k_shard_sweep_20260215T152638Z_transfer2gb_nogex44_sh74` |
| 80 | 339s | 29.499 toys/s | 10000 | 0 | `/home/actions-runner/nextstat_runs/pf32_2m10k_shard_sweep_20260215T152638Z_transfer2gb_nogex44_sh80` |
| 120 | 250s | 40.000 toys/s | 10000 | 0 | `/home/actions-runner/nextstat_runs/pf32_2m10k_shard_sweep_20260215T152638Z_transfer2gb_nogex44_sh120` |
| 160 | 197s | 50.761 toys/s | 10000 | 0 | `/home/actions-runner/nextstat_runs/pf32_2m10k_shard_sweep_20260215T152638Z_transfer2gb_nogex44_sh160` |

Interpretation:
- On a heterogeneous pool (bench fast, coolify slower), **oversharding** can reduce the long tail and increase throughput: `160 shards` was best in this sweep.

### Snapshot: Two-Node Matrix (bench+coolify only, higher oversharding)

Run date: 2026-02-15

- Workload: `pf31_gauss_exp_2m.json`, `n_toys=10000`
- Scheduler: HTCondor
- Threads per shard: `1`
- Submit file: `condor_job_transfer.sub`
- Requirements: `(Machine == "nextstat-bench") || (Machine == "nextstat-coolify")`
- `request_memory`: `2GB`

Results:

| Shards | Makespan | Throughput | Notes | Run dir |
|---:|---:|---:|---|---|
| 160 | 196s | 51.020 toys/s | rep1 | `/home/actions-runner/nextstat_runs/pf32_2m10k_rep1_sh160_20260215T163705Z` |
| 160 | 328s | 30.488 toys/s | rep2 (likely external contention on bench) | `/home/actions-runner/nextstat_runs/pf32_2m10k_rep2_sh160_20260215T163705Z` |
| 160 | 207s | 48.309 toys/s | rep3 | `/home/actions-runner/nextstat_runs/pf32_2m10k_rep3_sh160_20260215T163705Z` |
| 200 | 160s | 62.500 toys/s | sweep | `/home/actions-runner/nextstat_runs/pf32_2m10k_sweep_sh200_20260215T163705Z` |
| 240 | 142s | 70.423 toys/s | sweep (best in this mini-matrix) | `/home/actions-runner/nextstat_runs/pf32_2m10k_sweep_sh240_20260215T163705Z` |

Interpretation:
- For this 2-node pool (`64 + 10 = 74 CPUs`), shard counts around `~3x` total CPUs (`240`) improved throughput vs the earlier `<=160` sweep.
- Repeat variance at `shards=160` can be large if the submit/execute node has non-Condor background load; for publication-grade numbers, ensure the pool is idle (and record `uptime` / load averages alongside Condor snapshots).

### Stability: Repeats at Shards=200 and Shards=240 (Two-Node Pool)

Run date: 2026-02-15

- Requirements: `(Machine == "nextstat-bench") || (Machine == "nextstat-coolify")`
- Submit file: `condor_job_transfer.sub`
- `request_memory`: `2GB`, `threads=1`, `n_toys=10000`, `seed=42`

Aggregates (3 runs each):

| Shards | Makespan (median) | Makespan (mean ± std) | Throughput (median) | Throughput (mean ± std) | Notes |
|---:|---:|---:|---:|---:|---|
| 200 | 160s | 161.3 ± 1.9s | 62.500 toys/s | 61.992 ± 0.719 toys/s | stable |
| 240 | 142s | 150.7 ± 13.7s | 70.423 toys/s | 66.892 ± 5.720 toys/s | higher variance (one slow run) |

Interpretation:
- `shards=200` is very stable in this dataset and pool configuration.
- `shards=240` provides higher median throughput, but is somewhat more sensitive to contention/long-tail on `nextstat-bench`.
