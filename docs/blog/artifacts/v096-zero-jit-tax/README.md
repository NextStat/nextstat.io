# Benchmark Artifacts — NextStat v0.9.6: Zero-JIT Tax, ESS/grad, and Convergence

Blog post: `/docs/blog/nextstat-mams-zero-jit-tax.md`

## Files

| File | Description | Source |
|------|-------------|--------|
| `v100-multi-seed-matrix.json` | V100 parity aggregate (3 seeds, median/mean/std) | V100 GPU bench |
| `v100_v096_real_3seed_20260218T100007Z/seed_42.json` | V100 parity raw seed 42 | V100 GPU bench |
| `v100_v096_real_3seed_20260218T100007Z/seed_123.json` | V100 parity raw seed 123 | V100 GPU bench |
| `v100_v096_real_3seed_20260218T100007Z/seed_777.json` | V100 parity raw seed 777 | V100 GPU bench |
| `v100_v096_real_3seed_20260218T100007Z/run.log` | V100 parity driver log | V100 GPU bench |
| `2026-02-17-v096-refresh-v100-epyc.md` | V100 + EPYC refresh notes and methodology | Internal notes |
| `epyc-multi-seed-matrix.json` | EPYC MAMS multi-seed convergence matrix (3 seeds × 6 models) | EPYC CPU bench |
| `epyc-mams-suite.json` | Full EPYC MAMS suite output | EPYC CPU bench |
| `epyc-funnel-control-3seed.json` | EPYC funnel centered vs NCP control (3 seeds) | EPYC CPU bench |

## Run status

All JSON artifacts in this directory are from **real benchmark runs**.

Source run locations:

- V100 parity run (3 seeds):
  - `/root/nextstat/.internal/docs/benchmarks/v100_v096_real_3seed_20260218T100007Z/seed_42/a100_triple_bench.json/a100_triple_bench.json`
  - `/root/nextstat/.internal/docs/benchmarks/v100_v096_real_3seed_20260218T100007Z/seed_123/a100_triple_bench.json/a100_triple_bench.json`
  - `/root/nextstat/.internal/docs/benchmarks/v100_v096_real_3seed_20260218T100007Z/seed_777/a100_triple_bench.json/a100_triple_bench.json`
- EPYC matrix + suite run:
  - `/data/nextstat/.internal/docs/benchmarks/epyc_v096_real_3seed_20260218T095935Z/epyc-multi-seed-matrix.json`
  - `/data/nextstat/.internal/docs/benchmarks/epyc_v096_real_3seed_20260218T095935Z/epyc-mams-suite.json`
  - `/data/nextstat/.internal/docs/benchmarks/epyc_v096_real_3seed_20260218T095935Z/epyc-funnel-control-3seed.json`

Migration note:

- Earlier placeholders referenced missing historical paths from 2026-02-17.
- Those placeholders were replaced with fresh real-run artifacts on 2026-02-18.
