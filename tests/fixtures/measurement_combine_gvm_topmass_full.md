Full literature-backed fixture for the 7-8 TeV ATLAS-CMS top-mass combination.

Source material:
- GVM paper: `/Users/andresvlc/Downloads/s10052-025-13884-w.pdf`
- Official supplementary tables: [CMS-TOP-22-001 / ATLAS-CONF-2023-066](https://cms-results.web.cern.ch/cms-results/public-results/publications/TOP-22-001/)

What is encoded in `measurement_combine_gvm_topmass_full.json`:
- 15 input measurements:
  - ATLAS 2011: `dil`, `lj`, `aj`
  - ATLAS 2012: `dil`, `lj`, `aj`
  - CMS 2011: `dil`, `lj`, `aj`
  - CMS 2012: `dil`, `lj`, `aj`, `t`, `J/psi`, `vtx`
- Diagonal statistical covariance from Table 4 and Table 5
- 25 systematic sources with per-measurement magnitudes from Table 4 and Table 5
- Full per-source correlation matrices from Additional Tables 1-25

Category mapping:
- Additional Table 1 -> `JES 1`
- Additional Table 2 -> `JES 2`
- Additional Table 3 -> `JES 3`
- Additional Table 4 -> `b-JES`
- Additional Table 5 -> `g-JES`
- Additional Table 6 -> `l-JES`
- Additional Table 7 -> `CMS JES 1`
- Additional Table 8 -> `JER`
- Additional Table 9 -> `Leptons`
- Additional Table 10 -> `b tagging`
- Additional Table 11 -> `pTmiss`
- Additional Table 12 -> `Pileup`
- Additional Table 13 -> `Trigger`
- Additional Table 14 -> `ME generator`
- Additional Table 15 -> `QCD radiation`
- Additional Table 16 -> `Hadronization`
- Additional Table 17 -> `CMS b hadron B`
- Additional Table 18 -> `Color reconnection`
- Additional Table 19 -> `Underlying event`
- Additional Table 20 -> `PDF`
- Additional Table 21 -> `CMS top quark pT`
- Additional Table 22 -> `Background (data)`
- Additional Table 23 -> `Background (MC)`
- Additional Table 24 -> `Method`
- Additional Table 25 -> `Other`

Important implementation note:
- The raw published per-source correlation matrices are preserved in the JSON fixture, even when some of them are not positive semidefinite.
- This is intentional and matches the published source material.
- NextStat uses the raw matrices for the fixed-variance BLUE covariance construction.
- For nuisance/GVM profiling, NextStat applies the paper-style diagonal-shift regularization internally when a per-source matrix has a negative eigenvalue. The applied shifts are exposed in `diagnostics.corr_regularization_deltas`.

Expected baseline:
- The fixed-variance path reproduces the paper-level rounded result `m_top ~= 172.51 +/- 0.33 GeV` within tight tolerance.
