# pyhf parity audit (NextStat vs pyhf)

Generated: `2026-02-06 22:06:21`

## Summary

- Total: 23
- pyhf workspaces: 8
- Passed NLL parity: 5
- Failed/other: 18

## Results

| Workspace | Status | n_params | Init Δtwice_nll | Shift Δtwice_nll | Δnll_hat | Max |Δ bestfit| | pyhf fit (s) | NextStat fit (s) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `tests/fixtures/bad_histosys_template_length_mismatch.json` | pyhf_model_error | nan | nan | nan | nan | nan | nan | nan |  |
| `tests/fixtures/bad_observations_length_mismatch.json` | nextstat_model_error | nan | nan | nan | nan | nan | nan | nan |  |
| `tests/fixtures/bad_sample_length_mismatch.json` | pyhf_model_error | nan | nan | nan | nan | nan | nan | nan |  |
| `tests/fixtures/complex_workspace.json` | OK | 8 | 3.411e-13 | 3.553e-13 | 2.675e-08 | 3.110e-04 | 0.01 | 0.00 | nextstat nll < pyhf |
| `tests/fixtures/kalman_1d.json` | not_pyhf_workspace | nan | nan | nan | nan | nan | nan | nan | skipped (not pyhf JSON) |
| `tests/fixtures/kalman_1d_missing.json` | not_pyhf_workspace | nan | nan | nan | nan | nan | nan | nan | skipped (not pyhf JSON) |
| `tests/fixtures/kalman_2d_partial_missing.json` | not_pyhf_workspace | nan | nan | nan | nan | nan | nan | nan | skipped (not pyhf JSON) |
| `tests/fixtures/kalman_ar1.json` | not_pyhf_workspace | nan | nan | nan | nan | nan | nan | nan | skipped (not pyhf JSON) |
| `tests/fixtures/kalman_arma11.json` | not_pyhf_workspace | nan | nan | nan | nan | nan | nan | nan | skipped (not pyhf JSON) |
| `tests/fixtures/kalman_local_level.json` | not_pyhf_workspace | nan | nan | nan | nan | nan | nan | nan | skipped (not pyhf JSON) |
| `tests/fixtures/kalman_local_level_seasonal.json` | not_pyhf_workspace | nan | nan | nan | nan | nan | nan | nan | skipped (not pyhf JSON) |
| `tests/fixtures/kalman_local_linear_trend.json` | not_pyhf_workspace | nan | nan | nan | nan | nan | nan | nan | skipped (not pyhf JSON) |
| `tests/fixtures/kalman_local_linear_trend_seasonal.json` | not_pyhf_workspace | nan | nan | nan | nan | nan | nan | nan | skipped (not pyhf JSON) |
| `tests/fixtures/lmm_random_intercept.json` | not_pyhf_workspace | nan | nan | nan | nan | nan | nan | nan | skipped (not pyhf JSON) |
| `tests/fixtures/lmm_random_intercept_slope.json` | not_pyhf_workspace | nan | nan | nan | nan | nan | nan | nan | skipped (not pyhf JSON) |
| `tests/fixtures/ode_rk4_exp_decay.json` | not_pyhf_workspace | nan | nan | nan | nan | nan | nan | nan | skipped (not pyhf JSON) |
| `tests/fixtures/simple_histos_expected.json` | not_pyhf_workspace | nan | nan | nan | nan | nan | nan | nan | skipped (not pyhf JSON) |
| `tests/fixtures/simple_workspace.json` | OK | 3 | 9.095e-13 | 7.958e-13 | 4.678e-08 | 1.596e-04 | 0.00 | 0.00 | nextstat nll < pyhf |
| `tests/fixtures/tchannel_workspace.json` | OK | 277 | 1.467e-09 | 1.251e-09 | 5.391e-02 | 2.734e-01 | 90.26 | 24.69 | nextstat not converged (n_iter=1000); pyhf nll < nextstat |
| `tests/fixtures/tschannel_workspace.json` | json_parse_error | nan | nan | nan | nan | nan | nan | nan | invalid JSON |
| `tests/fixtures/tttt-prod_workspace.json` | OK | 249 | 1.023e-11 | 1.364e-12 | 1.017e-02 | 3.695e-02 | 20.88 | 0.64 | nextstat nll < pyhf |
| `tests/fixtures/workspace-postFit_PTV.json` | not_pyhf_workspace | nan | nan | nan | nan | nan | nan | nan | skipped (not pyhf JSON) |
| `tests/fixtures/workspace_tHu.json` | OK | 184 | 1.023e-12 | 5.457e-12 | 8.098e-02 | 7.747e-02 | 10.02 | 1.65 | nextstat nll < pyhf |

## Best-fit mismatches (details)

### `tests/fixtures/tchannel_workspace.json`

- Init Δtwice_nll: 1.467242e-09
- Shift Δtwice_nll: 1.251465e-09
- nll_hat: pyhf=307.351, nextstat=307.405 (|Δ|=5.390747e-02)
- NextStat fit status: converged=False, n_iter=1000, n_fev=1103, n_gev=2103
- Cross-eval: pyhf_nll(nextstat_hat)=307.405, nextstat_nll(pyhf_hat)=307.351
- Worst bestfit param diffs (top 10):
  - `JET_JER_EffectiveNP_8`: |Δ|=2.734e-01 (pyhf=-0.368344, nextstat=-0.0949694)
  - `JET_JER_EffectiveNP_1`: |Δ|=1.161e-01 (pyhf=+0.0808324, nextstat=+0.1969)
  - `JET_JER_EffectiveNP_11`: |Δ|=9.267e-02 (pyhf=+0.0714596, nextstat=+0.164129)
  - `JET_JER_EffectiveNP_3`: |Δ|=8.651e-02 (pyhf=-0.805153, nextstat=-0.718639)
  - `wc_xsec_pos`: |Δ|=7.827e-02 (pyhf=+0.150891, nextstat=+0.0726169)
  - `MET_SoftTrk_ResoPara`: |Δ|=7.600e-02 (pyhf=+0.444688, nextstat=+0.520685)
  - `weight_bTagSF_DL1r_60_C_0`: |Δ|=7.409e-02 (pyhf=+0.0983257, nextstat=+0.172417)
  - `shower_sitop_sh_CR`: |Δ|=7.221e-02 (pyhf=+0.124977, nextstat=+0.0527692)
  - `wccscale_mur`: |Δ|=7.179e-02 (pyhf=+1.1375, nextstat=+1.20929)
  - `EG_RESOLUTION_ALL`: |Δ|=5.177e-02 (pyhf=-0.301946, nextstat=-0.250173)

### `tests/fixtures/workspace_tHu.json`

- Init Δtwice_nll: 1.023182e-12
- Shift Δtwice_nll: 5.456968e-12
- nll_hat: pyhf=179.485, nextstat=179.404 (|Δ|=8.098023e-02)
- NextStat fit status: converged=True, n_iter=226, n_fev=241, n_gev=467
- Cross-eval: pyhf_nll(nextstat_hat)=179.404, nextstat_nll(pyhf_hat)=179.485
- Worst bestfit param diffs (top 10):
  - `NF_VV3lbc`: |Δ|=7.747e-02 (pyhf=+1.40931, nextstat=+1.33184)
  - `NFttbarW`: |Δ|=6.832e-02 (pyhf=+1.1492, nextstat=+1.08088)
  - `NFHFdecay_e`: |Δ|=5.593e-02 (pyhf=+1.04611, nextstat=+0.990181)
  - `NFttbarZ`: |Δ|=4.635e-02 (pyhf=+1.15896, nextstat=+1.11261)
  - `qmisID_stat`: |Δ|=4.486e-02 (pyhf=+0.0203671, nextstat=-0.0244954)
  - `NFHFdecay_mu`: |Δ|=4.424e-02 (pyhf=+0.93597, nextstat=+0.891729)
  - `JET_Pileup_RhoTopology`: |Δ|=4.246e-02 (pyhf=-0.714037, nextstat=-0.671578)
  - `qmisID_nonClosure`: |Δ|=3.689e-02 (pyhf=-0.01977, nextstat=-0.05666)
  - `lumi`: |Δ|=3.494e-02 (pyhf=+1, nextstat=+1.03494)
  - `qmisID_Zwindow`: |Δ|=3.444e-02 (pyhf=-0.0450114, nextstat=-0.0794469)

### `tests/fixtures/tttt-prod_workspace.json`

- Init Δtwice_nll: 1.023182e-11
- Shift Δtwice_nll: 1.364242e-12
- nll_hat: pyhf=287.512, nextstat=287.502 (|Δ|=1.017162e-02)
- NextStat fit status: converged=True, n_iter=179, n_fev=192, n_gev=371
- Cross-eval: pyhf_nll(nextstat_hat)=287.502, nextstat_nll(pyhf_hat)=287.512
- Worst bestfit param diffs (top 10):
  - `NF_ttW`: |Δ|=3.695e-02 (pyhf=+1.55993, nextstat=+1.59688)
  - `mu_signal`: |Δ|=2.902e-02 (pyhf=+2.01748, nextstat=+2.0465)
  - `NF_ttbar_CO`: |Δ|=2.396e-02 (pyhf=+1.61223, nextstat=+1.63619)
  - `NF_ttbar_HFm`: |Δ|=1.862e-02 (pyhf=+1.07248, nextstat=+1.0911)
  - `NF_ttbar_gstr`: |Δ|=1.577e-02 (pyhf=+0.934421, nextstat=+0.950191)
  - `NF_ttbar_HFe`: |Δ|=1.476e-02 (pyhf=+0.854219, nextstat=+0.868976)
  - `lumi`: |Δ|=1.418e-02 (pyhf=+1, nextstat=+0.985818)
  - `ttZ_Gen`: |Δ|=4.747e-03 (pyhf=+0.180486, nextstat=+0.175738)
  - `ttbar_Xsec_light`: |Δ|=4.666e-03 (pyhf=-0.287899, nextstat=-0.283233)
  - `ATLAS_JES_Flavor_Composition`: |Δ|=4.091e-03 (pyhf=-0.054864, nextstat=-0.0507732)

