# RNTuple Decode Benchmark (nextstat-bench, 2026-02-16)

This note records a direct `ns-root` vs ROOT decode comparison on the same host for a large RNTuple fixture.

## Environment

- Host: `nextstat-bench` (`64` vCPU, Ubuntu `24.04.3`)
- Date: `2026-02-16`
- ROOT runtime: snap `root-framework v6-36-04`
- Fixture: `tests/fixtures/rntuple_bench_large_primitive.root`
- Fixture content: `Events` ntuple with `pt: float` and `n: int32`, `5,000,000` entries, `20` cluster groups (`250,000` entries each), file size about `34 MB`

## Commands

`ns-root`:

```bash
cd /root/nextstat.io
NS_ROOT_RNTUPLE_PERF_CASES=/var/snap/root-framework/common/bench/rntuple_bench_large_primitive.root \
NS_ROOT_RNTUPLE_PERF_ITERS=5 \
NS_ROOT_RNTUPLE_PERF_MAX_AVG_MS=100000 \
NS_ROOT_RNTUPLE_PERF_MIN_SUITES_PER_SEC=0 \
cargo test -p ns-root --test rntuple_perf_gate rntuple_decode_perf_gate_baseline --release -- --ignored --nocapture
```

ROOT:

```bash
cd /var/snap/root-framework/common/bench
root -l -b -q 'rntuple_root_read_primitive_bench.C("/var/snap/root-framework/common/bench/rntuple_bench_large_primitive.root",5)'
```

The ROOT macro uses `ROOT::RNTupleReader::GetView<T>()` for both columns and scans all entries.

## Results

| Tool | entries | iters | avg_ms | entries_per_sec |
|---|---:|---:|---:|---:|
| `ns-root` (`rntuple_perf_gate`) | 5,000,000 | 5 | 102.628 | 48,719,758.6 |
| ROOT (`RNTupleReader` view loop) | 5,000,000 | 5 | 434.046 | 11,519,500 |

Relative on this harness:

- `ns-root` is about **4.23x faster** than ROOT (`entries_per_sec` ratio).

## Notes

- ROOT was installed via snap and run from snap-accessible path (`/var/snap/root-framework/common/bench`) because strict confinement cannot read `/root/nextstat.io` directly.
- This benchmark is for a primitive-only large fixture. Mixed nested/variable large-layout stress decode coverage is now marked `verified` in `docs/references/rntuple-compatibility-matrix.md` (`tests/fixtures/rntuple_bench_large.root`).

## Mixed-Layout Verification Addendum (2026-02-16)

To verify large mixed-layout decode support end-to-end, release perf-gate was run with case override:

```bash
NS_ROOT_RNTUPLE_PERF_CASES=/Users/andresvlc/WebDev/nextstat.io/tests/fixtures/rntuple_bench_large.root \
NS_ROOT_RNTUPLE_PERF_ITERS=5 \
NS_ROOT_RNTUPLE_PERF_MAX_AVG_MS=100000 \
NS_ROOT_RNTUPLE_PERF_MIN_SUITES_PER_SEC=0 \
cargo test -p ns-root --test rntuple_perf_gate rntuple_decode_perf_gate_baseline --release -- --ignored --nocapture
```

Observed result:

- `suite_size=1`
- `total_entries=2,000,000`
- `avg_ms=99.088`
- `entries_per_sec=20,184,010.9`

This confirms deterministic all-cluster decode for `primitive + variable-array + nested-pair` mixed layout and matches regression coverage in `crates/ns-root/tests/rntuple_discovery.rs`.

## Reproducible Harness (Repo Script)

Use the repository harness to run `ns-root` and ROOT on the same fixture and print the ratio:

```bash
make rntuple-root-vs-nsroot
```

Under the hood this uses:

- `scripts/benchmarks/run_rntuple_root_vs_nsroot.sh`
- `scripts/benchmarks/rntuple_root_read_primitive_bench.C`

## Local Re-Run Snapshot (Darwin arm64, ROOT 6.38.00)

Command:

```bash
make rntuple-root-vs-nsroot
```

Observed result (2026-02-16):

| Tool | entries | iters | avg_ms | entries_per_sec |
|---|---:|---:|---:|---:|
| `ns-root` (`rntuple_perf_gate`) | 5,000,000 | 5 | 43.836 | 114,062,130.1 |
| ROOT (`RNTupleReader` view loop) | 5,000,000 | 5 | 164.370 | 30,419,200 |

Relative on this host:

- `ns-root` is about **3.75x faster** than ROOT.

This local rerun is not directly comparable to `nextstat-bench` due to different hardware/runtime.
