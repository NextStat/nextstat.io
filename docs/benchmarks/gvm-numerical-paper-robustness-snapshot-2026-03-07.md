# GVM NumericalPaper Robustness Snapshot

**Date**: 2026-03-07  
**Status**: Published  
**Scope**: Multi-start stability of the paper-faithful `numerical-paper` solver on
mixed literature-backed and synthetic low-`epsilon` tiers

## Source of truth

This snapshot is derived directly from the committed mixed-family trust artifact:

- [measurement_combine_numerical_paper_multistart_mixed_family_report.json](/tests/fixtures/measurement_combine_numerical_paper_multistart_mixed_family_report.json)
- [measurement_combine_numerical_paper_multistart_mixed_family_report.md](/tests/fixtures/measurement_combine_numerical_paper_multistart_mixed_family_report.md)

The artifact is built by the test-owned report layer in:

- [measurement_combine.rs](/crates/ns-inference/src/measurement_combine.rs)

and validated by the ignored slow gate:

- `numerical_paper_multistart_mixed_family_stays_stable_across_literature_and_synthetic_tiers`

## Coverage

The current mixed family covers:

- one literature-backed tier: `literature_topmass_bjes_0p05`
- four synthetic tiers: `32x24`, `64x48`, `96x64`, `128x96`
- deterministic low-`epsilon` stress conditions at `epsilon = 0.05`
- repeated cold-start perturbations of the original-theta numerical optimizer

## Tier summary

| Tier | Shape | Starts | `mu_tol` | `fval_tol` | `ci_tol` | Max `|Δmu|` | Max `|Δfval|` | Max `|ΔCI|` | Within tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `literature_topmass_bjes_0p05` | `15x25` | 3 | `5e-7` | `1e-8` | `5e-6` | `1.525040147e-7` | `1.955413609e-11` | `3.409814951e-6` | yes |
| `synthetic_32x24` | `32x24` | 3 | `3e-7` | `1e-8` | `1e-6` | `1.703667181e-7` | `1.627995516e-10` | `4.641265150e-11` | yes |
| `synthetic_64x48` | `64x48` | 2 | `1e-6` | `1e-7` | `5e-6` | `1.296383516e-7` | `1.346052159e-10` | `5.391314062e-10` | yes |
| `synthetic_96x64` | `96x64` | 1 | `3e-6` | `3e-7` | `1e-5` | `3.378971769e-8` | `5.093170330e-11` | `1.125783911e-10` | yes |
| `synthetic_128x96` | `128x96` | 1 | `1e-5` | `1e-6` | `3e-5` | `1.717061650e-6` | `1.600710675e-10` | `2.521289844e-10` | yes |

## Aggregate findings

- All tiers are within their committed tolerance envelopes.
- The worst `mu` drift is on `synthetic_128x96`: `1.717061650e-6`.
- The worst `fval` drift is on `synthetic_32x24`: `1.627995516e-10`.
- The worst CI drift is on the literature-backed tier `literature_topmass_bjes_0p05`: `3.409814951e-6`.
- The mixed family therefore supports a stronger claim than the synthetic-only artifact:
  `NumericalPaper` remains stable both on published top-mass structure and on synthetic stress tiers through `128x96`.

## Interpretation

This snapshot is about **non-perturbative trust**, not speed.

It answers a narrower question than the performance snapshot:

- If we restart `numerical-paper` from materially different initial points,
  does it return the same optimum and profile interval within a fixed
  research-grade envelope?

For the current committed family, the answer is **yes**.

## Reproducing

Run the slow gate explicitly:

```bash
CARGO_TARGET_DIR=/tmp/nextstat-sota-target cargo test \
  -p ns-inference \
  --lib numerical_paper_multistart_mixed_family_stays_stable_across_literature_and_synthetic_tiers \
  -- --ignored --nocapture
```

Then compare the generated report against the committed fixtures:

- [measurement_combine_numerical_paper_multistart_mixed_family_report.json](/tests/fixtures/measurement_combine_numerical_paper_multistart_mixed_family_report.json)
- [measurement_combine_numerical_paper_multistart_mixed_family_report.md](/tests/fixtures/measurement_combine_numerical_paper_multistart_mixed_family_report.md)

## Related evidence

- [GVM Benchmark Snapshot](/docs/benchmarks/gvm-measurement-combine-snapshot-2026-03-07.md)
- [HEP GVM Measurement Combinations](/docs/tutorials/hep-gvm-measurement-combinations.md)
