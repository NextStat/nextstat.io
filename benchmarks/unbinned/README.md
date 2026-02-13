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

```bash
PF31_HOST=88.198.23.172 \
PF31_KEY=~/.ssh/rundesk_hetzner \
PF31_BIN=/root/nextstat.io/target/release/nextstat \
PF31_SPEC=/root/nextstat.io/benchmarks/unbinned/specs/pf31_gauss_exp_10k.json \
PF31_TOYS=10000,100000 \
PF31_SHARDS=2,4,8 \
bash scripts/benchmarks/pf31_remote_matrix.sh
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
