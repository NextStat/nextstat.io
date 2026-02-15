# CPU Farm Runbook for `unbinned-fit-toys`

This folder contains lightweight SSH tooling for CPU-only toy studies on
multiple hosts (HEP cluster prep, 10k+ toys, large event counts).

Scripts:
- `preflight_cluster.py`: probe per-host CPU topology (`logical`, `physical`, HT ratio).
- `run_unbinned_fit_toys_cluster.py`: split toys across hosts and run one shard per host.
- `merge_unbinned_toys_results.py`: merge shard `out.json` files back into one output.
- `render_unbinned_fit_toys_scheduler.py`: render SLURM/HTCondor array jobs for `--shard INDEX/TOTAL`.
- `collect_unbinned_fit_toys_scheduler.py`: collect scheduler outputs back into `manifest.json` format.
- `pf32_unbinned_shard_sweep.py`: render/collect/merge/summarize a shard sweep (`--shards` list) into one markdown table.

## 1) Prepare hosts file

Example `hosts.txt`:

```text
# host [optional key=value overrides]
node01.hep.local
node02.hep.local weight=48
node03.hep.local threads=64
node04.hep.local user=analysis port=2222 weight=32 threads=32
```

Supported per-host overrides:
- `weight=<float>`: toy split weight.
- `threads=<int>`: override `--threads` policy.
- `user=<name>`: override `--ssh-user` for this host.
- `port=<int>`: override `--ssh-port` for this host.

## 2) Preflight (recommended)

```bash
python3 scripts/farm/preflight_cluster.py \
  --hosts-file hosts.txt \
  --ssh-user analysis \
  --ssh-key ~/.ssh/hep_cluster_ed25519 \
  --out tmp/farm/preflight.json
```

The JSON is used by the launcher for CPU-aware thread policy and weighting.

## 3) Run distributed toys

```bash
python3 scripts/farm/run_unbinned_fit_toys_cluster.py \
  --hosts-file hosts.txt \
  --preflight-json tmp/farm/preflight.json \
  --config /absolute/path/to/spec.json \
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

Notes:
- `--threads physical` is usually the safest default for CPU farms.
- BLAS/OMP thread env vars are pinned to 1 by default to avoid oversubscription.
- Shard seeds are deterministic: `seed_i = base_seed + toy_start_i`.
- Use `--dry-run` first to verify shard plan without launching jobs.

## 4) Merge shard outputs

```bash
python3 scripts/farm/merge_unbinned_toys_results.py \
  --manifest tmp/farm/runs/<run_id>/manifest.json \
  --out tmp/farm/runs/<run_id>/merged.out.json
```

The merged artifact keeps the same core contract as a regular
`unbinned-fit-toys` output (`results`, `summary`, `guardrails`) plus a `merge`
block with source shard metadata.

## 5) HEP-scale practical profile

For studies around `~2M` events and `~10k` toys:
- Start with `--threads physical` and `--weight-mode physical`.
- Keep one shard per host (this launcher model).
- Scale host count first, then tune per-host `threads=...` overrides.
- Always archive `manifest.json` and `merged.out.json` immediately.

## 6) Scheduler bridge (SLURM / HTCondor)

Use this path when you have a scheduler and a shared filesystem, and you want
the same deterministic shard/merge contract without SSH fan-out.

### SLURM (array job)

Render:

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

Submit:

```bash
sbatch /shared/runs/<run_id>/slurm_array_job.sh
```

Collect results into manifest:

```bash
python3 scripts/farm/collect_unbinned_fit_toys_scheduler.py \
  --manifest /shared/runs/<run_id>/manifest.json \
  --in-place
```

Merge:

```bash
python3 scripts/farm/merge_unbinned_toys_results.py \
  --manifest /shared/runs/<run_id>/manifest.json \
  --out /shared/runs/<run_id>/merged.out.json
```

### HTCondor

Render:

```bash
python3 scripts/farm/render_unbinned_fit_toys_scheduler.py \
  --scheduler htcondor \
  --config /shared/spec.json \
  --n-toys 10000 \
  --seed 42 \
  --shards 100 \
  --threads 1 \
  --nextstat-bin /shared/nextstat/target/release/nextstat \
  --out-dir /shared/runs \
  --condor-request-cpus 1 \
  --condor-request-memory "2GB" \
  --condor-requirements '(Machine != "nextstat-gex44")'
```

Submit:

```bash
condor_submit /shared/runs/<run_id>/condor_job.sub
```

If execute nodes do not have access to the absolute paths in `manifest.json`
(no shared filesystem), submit the transfer-mode file instead:

```bash
condor_submit /shared/runs/<run_id>/condor_job_transfer.sub
```

Transfer-mode notes:
- For JSON unbinned specs, the renderer will also transfer `channels[].data.file` inputs and stage them under `./specs/` so relative paths like `../artifacts/...` resolve inside the job sandbox.

Collect + merge are identical to the SLURM flow (use the same `manifest.json`).

## 7) Shard Sweep (Scaling Matrix)

For scaling studies, use:
- `scripts/farm/pf32_unbinned_shard_sweep.py` (render/collect/merge/summarize across a list of shard counts)

HTCondor-only convenience: you can run the sweep sequentially from the submit node
(avoids cross-run contention) and automatically run collect+merge after each run directory finishes:

```bash
python3 scripts/farm/pf32_unbinned_shard_sweep.py submit \
  --sweep-json /shared/runs/<sweep_id>.sweep.json \
  --mode transfer \
  --poll-s 10
```
