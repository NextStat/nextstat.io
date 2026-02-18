# Benchmark Artifacts — NextStat v0.9.6: Zero-JIT Tax, ESS/grad, and Convergence

Blog post: `/docs/blog/nextstat-mams-zero-jit-tax.md`

## Files

| File | Description | Source |
|------|-------------|--------|
| `v100-multi-seed-matrix-canonical.json` | **Canonical** V100 parity aggregate (3 seeds, built-in BlackJAX warmup + tuned inverse mass matrix) | V100 GPU bench |
| `v100-multi-seed-matrix-builtinwarmup.json` | Mirror of canonical V100 aggregate for tooling compatibility | V100 GPU bench |
| `v100-multi-seed-matrix.json` | Legacy filename updated to canonical content | V100 GPU bench |
| `v100-multi-seed-matrix-essgradfix.json` | Legacy filename updated to canonical content | V100 GPU bench |
| `v100-parity-chart-data-canonical.csv` | Plot-ready parity table (cold/warm, ESS/s, ESS/grad, R-hat) | V100 GPU bench |
| `v100-essgrad-ratio-canonical.csv` | Plot-ready NS/BJ ratio table (ESS/grad and ESS/s) | V100 GPU bench |
| `v100_v096_builtinwarmup_3seed_20260218T224654Z/seed_42/gpu_triple_bench.json` | Raw seed 42 (std/eight/glm) | V100 GPU bench |
| `v100_v096_builtinwarmup_3seed_20260218T224654Z/seed_123/gpu_triple_bench.json` | Raw seed 123 (std/eight/glm) | V100 GPU bench |
| `v100_v096_builtinwarmup_3seed_20260218T224654Z/seed_777/gpu_triple_bench.json` | Raw seed 777 (std/eight/glm) | V100 GPU bench |
| `v100_ns_funnel_3seed_20260218T231337Z/seed_*/gpu_triple_bench.json` | Raw NS LAPS funnel addendum runs (3 seeds) | V100 GPU bench |
| `v100_bj_funnel_builtin3seed_20260218T231204Z/seed_*/gpu_triple_bench.json` | Raw BlackJAX funnel addendum runs (3 seeds) | V100 GPU bench |
| `2026-02-17-v096-refresh-v100-epyc.md` | V100 + EPYC refresh notes and methodology | Internal notes |
| `epyc-multi-seed-matrix.json` | EPYC MAMS multi-seed convergence matrix (3 seeds × 6 models) | EPYC CPU bench |
| `epyc-mams-suite.json` | Full EPYC MAMS suite output | EPYC CPU bench |
| `epyc-funnel-control-3seed.json` | EPYC funnel centered vs NCP control (3 seeds) | EPYC CPU bench |

## Run status

All JSON artifacts in this directory are from **real benchmark runs**.

Canonical source run locations:

- V100 parity (std/eight/glm, 3 seeds, built-in warmup):
  - `/tmp/a100_v096_builtin3seed_20260218T224654Z/seed_42/gpu_triple_bench.json`
  - `/tmp/a100_v096_builtin3seed_20260218T224654Z/seed_123/gpu_triple_bench.json`
  - `/tmp/a100_v096_builtin3seed_20260218T224654Z/seed_777/gpu_triple_bench.json`
- V100 funnel addendum:
  - NS LAPS: `/tmp/a100_ns_funnel_3seed_20260218T231337Z/seed_*/gpu_triple_bench.json`
  - BlackJAX: `/tmp/a100_bj_funnel_builtin3seed_20260218T231204Z/seed_*/gpu_triple_bench.json`
- EPYC matrix + suite run:
  - `/data/nextstat/.internal/docs/benchmarks/epyc_v096_real_3seed_20260218T095935Z/epyc-multi-seed-matrix.json`
  - `/data/nextstat/.internal/docs/benchmarks/epyc_v096_real_3seed_20260218T095935Z/epyc-mams-suite.json`
  - `/data/nextstat/.internal/docs/benchmarks/epyc_v096_real_3seed_20260218T095935Z/epyc-funnel-control-3seed.json`

Migration note:

- Earlier placeholders referenced missing historical paths from 2026-02-17.
- Canonical V100 artifacts were refreshed on 2026-02-18 with built-in BlackJAX warmup and tuned inverse-mass sampling path.
- Legacy benchmark script/file naming used `a100_*`; canonical naming is now `gpu_*` (the `a100_*` names are kept only for backward compatibility in historical folders).
