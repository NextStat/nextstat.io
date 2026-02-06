# Benchmarks

NextStat uses [Criterion.rs](https://crates.io/crates/criterion) for Rust micro-benchmarks.

This doc focuses on pragmatic workflows:
- running benches locally (full vs quick)
- saving and comparing baselines
- CI workflows used for bench compilation and opt-in perf smoke

## Local Runs

Run all benches (slow):

```bash
cargo bench --workspace
```

Run a common entry point:

```bash
cargo bench -p ns-inference --bench mle_benchmark
```

Criterion writes HTML reports to `target/criterion/**/report/index.html`.

## Quick Mode

For fast iteration (less stable numbers):

```bash
cargo bench -p ns-inference --bench mle_benchmark -- --quick
```

Use `--quick` for CI smoke runs. Do not use quick mode for published numbers.

## Baselines (Criterion)

Save a baseline:

```bash
cargo bench -p ns-inference --bench mle_benchmark -- --save-baseline main
```

Compare against a baseline:

```bash
cargo bench -p ns-inference --bench mle_benchmark -- --baseline main
```

Baselines are stored under `target/criterion`.

## CI

Bench compilation and scheduled quick runs live in `.github/workflows/bench.yml`.

An opt-in, non-blocking perf smoke workflow is available as:
- `.github/workflows/perf-smoke.yml` (manual `workflow_dispatch`)

The perf smoke job is intended to catch obvious breakage (bench runtime errors) without
gating merges on absolute timing thresholds.

