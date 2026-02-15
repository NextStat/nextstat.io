# Pharma Suite (Seed)

This is a **pharmacometrics** benchmark suite for NextStat:

- PK (1-compartment oral) likelihood timing
- optional MLE fit wall-time (NextStat)
- optional external baselines for population PK:
  - `nlmixr2` (FOCEI/SAEM)
  - `torsten` label (CmdStan/Stan MAP baseline runner)

Run a single case:

```bash
python run.py --deterministic --out ../../out/pharma_pk_1c_oral.json
```

Suite runner (multiple generated cases):

```bash
python suite.py --deterministic --out-dir ../../out/pharma
```

Suite runner with external baselines:

```bash
python suite.py \
  --deterministic \
  --fit --fit-repeat 7 \
  --run-baselines \
  --baselines nlmixr2,torsten \
  --baseline-repeat 5 \
  --torsten-iter 1200 \
  --baseline-r-libs-user /path/to/r-lib \
  --baseline-cmdstan /path/to/cmdstan-2.38.0 \
  --out-dir ../../out/pharma
```
