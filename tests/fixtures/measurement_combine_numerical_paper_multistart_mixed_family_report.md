# NumericalPaper Multi-Start Family Stability

- Solver: `numerical-paper`
- CI level: `0.68`
- Epsilon: `0.05`
- All tiers within tolerance: `yes`

| Tier | Starts | Max |mu| drift | Max fval drift | Max CI drift | Worst start | Within tolerance |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| literature_topmass_bjes_0p05 | 3 | 1.525040147e-7 | 1.955413609e-11 | 3.409814951e-6 | start_1 | yes |
| synthetic_32x24 | 3 | 1.703667181e-7 | 1.627995516e-10 | 4.641265150e-11 | start_2 | yes |
| synthetic_64x48 | 2 | 1.296383516e-7 | 1.346052159e-10 | 5.391314062e-10 | start_2 | yes |
| synthetic_96x64 | 1 | 3.378971769e-8 | 5.093170330e-11 | 1.125783911e-10 | start_1 | yes |
| synthetic_128x96 | 1 | 1.717061650e-6 | 1.600710675e-10 | 2.521289844e-10 | start_1 | yes |

## Worst Starts

| Tier | Start | mu_shift | nuisance_scale | phase | |mu| drift | fval drift | max CI drift |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| literature_topmass_bjes_0p05 | start_1 | 0.493272 | 0.200000 | 0.000000 | 1.081900791e-9 | 1.699618224e-11 | 3.409814951e-6 |
| synthetic_32x24 | start_2 | -0.113887 | 0.200000 | 1.000000 | 1.703667181e-7 | 1.627995516e-10 | 2.179945113e-11 |
| synthetic_64x48 | start_2 | -0.151205 | 0.200000 | 1.000000 | 1.296383516e-7 | 1.818989404e-11 | 1.201385658e-10 |
| synthetic_96x64 | start_1 | 0.177677 | 0.200000 | 0.000000 | 3.378971769e-8 | 5.093170330e-11 | 1.125783911e-10 |
| synthetic_128x96 | start_1 | 0.247256 | 0.200000 | 0.000000 | 1.717061650e-6 | 1.600710675e-10 | 2.521289844e-10 |
