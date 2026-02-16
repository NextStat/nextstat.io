# Report Visual Style (HEP 2026)

This guide explains how to produce publication-grade TREx-like plots in NextStat with one command, while keeping a distinct NextStat visual identity.

## One-Command Render

```bash
nextstat report \
  --input tests/fixtures/histfactory/workspace.json \
  --histfactory-xml tests/fixtures/histfactory/combination.xml \
  --out-dir tmp/report \
  --render \
  --svg-dir tmp/report/svg \
  --label-status Internal \
  --sqrt-s-tev 13.0 \
  --show-mc-band true \
  --show-stat-band true \
  --band-hatch "////" \
  --palette hep2026
```

Outputs:

- `tmp/report/report.pdf`
- `tmp/report/svg/*.svg`
- `tmp/report/svg/*.png`

## Style Controls

CLI flags for `nextstat report`:

- `--label-status` (for header text: `Internal`, `Preliminary`, `Public`)
- `--sqrt-s-tev` (header energy text)
- `--show-mc-band` (total uncertainty band)
- `--show-stat-band` (stat-only uncertainty band)
- `--band-hatch` (hatch pattern for total uncertainty)
- `--palette` (`hep2026` or `tableau10`)

## `analysis.yaml` Configuration

```yaml
execution:
  report:
    enabled: true
    out_dir: tmp/report
    overwrite: true
    include_covariance: false
    histfactory_xml: tests/fixtures/histfactory/combination.xml
    render:
      enabled: true
      pdf: tmp/report/report.pdf
      svg_dir: tmp/report/svg
      python: null
      label_status: Internal
      sqrt_s_tev: 13.0
      show_mc_band: true
      show_stat_band: true
      band_hatch: "////"
      palette: hep2026
    skip_uncertainty: false
    uncertainty_grouping: prefix_1
```

## Notes

- `mc_band_postfit` and `ratio_band` represent total postfit uncertainty.
- `mc_band_postfit_stat` and `ratio_band_stat` represent stat-only uncertainty.
- If fit covariance is unavailable, uncertainty bands fall back to diagonal uncertainty propagation.
- Legend order is intentionally stable across channels: `Data`, `Total postfit`, `Total prefit`, stacked samples, then uncertainty bands.
