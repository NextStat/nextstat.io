# Benchmark Artifacts — NextStat v0.9.6: Zero-JIT Tax, ESS/grad, and Convergence

Canonical blog post: `docs/blog/nextstat-mams-zero-jit-tax.md`

Canonical dataset status:

- Final publication dataset is the **3-seed canonical set** (`42/123/777`) used in section 3/4 of the blog post.
- Historical files are retained for auditability, but headline numbers come from canonical files listed below.

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
| `ns_vs_stan_epyc_v096_glm1000_20260221T021540Z/bench_nuts_vs_cmdstan.json` | EPYC NS NUTS vs CmdStan head-to-head, 3 seeds, GLM `n=1000,p=10` | EPYC CPU bench |
| `ns_vs_stan_epyc_v096_glm5000_20260221T021711Z/bench_nuts_vs_cmdstan.json` | EPYC NS NUTS vs CmdStan head-to-head, 3 seeds, GLM `n=5000,p=20` | EPYC CPU bench |
| `2026-02-21-epyc-ns-vs-stan-3seed-summary.md` | Consolidated 3-seed EPYC NS vs Stan summary (includes wall medians and quality) | EPYC CPU bench |

## Run status

All JSON artifacts in this directory are from **real benchmark runs**.

Canonical source run locations:

- These are original source-machine paths (V100/EPYC hosts) used for provenance.
- The portable copies used by the blog are the files in this directory.

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
- Legacy benchmark script/file naming used `a100_*`; canonical naming is now `gpu_*`.
- Any `a100_*` path names above are **legacy directory names only**; these runs were executed on **Tesla V100-PCIE-16GB**.
