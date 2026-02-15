# Insurance Suite

This directory contains the **insurance Chain Ladder benchmark suite** for the standalone public benchmarks repo.

## Scope

- **Deterministic Chain Ladder** on classic 10x10 (Taylor-Ashe) and synthetic 20x20 triangles
- **Mack stochastic Chain Ladder** with standard errors on the 10x10 Taylor-Ashe triangle

## Schemas

- `nextstat.insurance_benchmark_result.v1` (per case)
- `nextstat.insurance_benchmark_suite_result.v1` (suite index)

## Cases

| Case | Kind | Triangle | Description |
|------|------|----------|-------------|
| `chain_ladder_10x10` | `chain_ladder` | Taylor-Ashe 10x10 | Classic deterministic CL |
| `mack_10x10` | `mack` | Taylor-Ashe 10x10 | Mack stochastic CL with SE |
| `chain_ladder_20x20` | `chain_ladder` | Synthetic 20x20 | Larger synthetic triangle |

## Run

```bash
# Full suite
python3 suites/insurance/suite.py --deterministic --out-dir out/insurance

# Single case
python3 suites/insurance/run.py --case chain_ladder_10x10 --kind chain_ladder --n 10 --out out/insurance/cl_10.json
python3 suites/insurance/run.py --case mack_10x10 --kind mack --n 10 --out out/insurance/mack_10.json
python3 suites/insurance/run.py --case chain_ladder_20x20 --kind chain_ladder --n 20 --out out/insurance/cl_20.json
```

## Parity

When `chainladder-python` is installed, parity metrics are computed automatically:

- Development factors: max abs diff, max rel diff
- Ultimates: max abs diff, max rel diff
- Total IBNR: abs diff, rel diff
- Mack SE: max abs diff, max rel diff (where available)

If `chainladder-python` is not installed, parity is reported as `status="skipped"`.

## Optional baseline

```bash
pip install chainladder
```
