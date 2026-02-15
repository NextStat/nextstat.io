# Torsten / CmdStan Baseline

This directory contains a runnable **Stan baseline** for the pharma population PK cases
(`foce_*` / `saem_*`) produced by `suites/pharma/run.py`.

Current implementation:
- engine: `cmdstanr` + CmdStan `optimize` (`lbfgs`)
- model: hierarchical 1-compartment oral PK with additive error (`pop_pk_1cpt_map.stan`)
- output schema: `nextstat.pharma_baseline_result.v1`

Run:

```bash
R_LIBS_USER=/path/to/r-lib \
CMDSTAN=/path/to/cmdstan-2.xx.x \
Rscript run.R \
  --in ../../out/pharma/cases/foce_1c_oral_nsub_100.json \
  --out ../../out/pharma/baselines/torsten_foce_100.json \
  --repeat 3 \
  --iter 500
```

Arguments:
- `--in` (required): pharma case JSON path
- `--out` (required): output baseline JSON path
- `--repeat` (optional): timed fit repetitions after one warmup run (default: `1`)
- `--iter` (optional): max optimizer iterations per run (default: `500`)
- `--seed` (optional): base seed for repeated runs (default: `12345`)

Notes:
- For this benchmark family we use an analytic Stan model (no ODE solver required).
- `baseline="torsten"` is kept for compatibility with the benchmark taxonomy.
