# Fit Summary Script (`scripts/fit_summary.sh`)

Utility script for extracting compact, analysis-friendly tables from a NextStat fit JSON.

## Usage

```bash
scripts/fit_summary.sh --input fit.json [--out-dir out/] [--prefix fit] [--top 30]
```

## Input contract

The script expects a **fit JSON** (e.g. output of `nextstat fit`) containing:

- `parameter_names` (array)
- `bestfit` (array)
- `uncertainties` (array)

If any are missing (or lengths differ), the script exits with an error.

## Outputs

Given `--prefix fit`, it writes:

- `fit.summary.json` — compact run metadata (`converged`, `n_iter`, `nll`, `poi`, `poi_hat`, `poi_unc`, ...)
- `fit.params.tsv` — all parameters (`name`, `value`, `uncertainty`)
- `fit.top_unc.tsv` — top-N by uncertainty
- `fit.top_pull.tsv` — top-N by absolute pull (`alpha_*`: `value/unc`, `gamma_*`: `(value-1)/unc`)

## Typical workflow

```bash
# 1) Run fit
nextstat fit --input workspace.json > fit.json

# 2) Produce compact summary tables
scripts/fit_summary.sh --input fit.json --out-dir report_tmp --prefix bb4l --top 40

# 3) Produce plotting artifacts
nextstat viz pulls --input workspace.json --fit fit.json --output report_tmp/pulls.json
nextstat viz corr  --input workspace.json --fit fit.json --output report_tmp/corr.json
nextstat viz ranking --input workspace.json --output report_tmp/ranking.json

# 4) Render selected artifacts
nextstat viz render --kind pulls --input report_tmp/pulls.json --output report_tmp/pulls.png
nextstat viz render --kind corr --input report_tmp/corr.json --output report_tmp/corr.png --corr-top-n 40
nextstat viz render --kind ranking --input report_tmp/ranking.json --output report_tmp/ranking.png
```

## Notes

- Raw text logs are not accepted directly. Convert/generate a structured `fit.json` first.
- `jq` is required (`brew install jq` on macOS).
