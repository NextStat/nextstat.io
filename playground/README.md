# NextStat Playground

Static browser demo for running NextStat asymptotic CLs (q~tilde~) directly on a `pyhf`-style `workspace.json`.

## Build (WASM)

Prereqs:
- Rust toolchain (see `rust-toolchain.toml`)
- `wasm32-unknown-unknown` target: `rustup target add wasm32-unknown-unknown`
- `wasm-bindgen` CLI: `cargo install wasm-bindgen-cli --version 0.2.108` (version should match `Cargo.lock`)

Build the JS + WASM bundle into `playground/pkg/`:

From the repo root:

```bash
make playground-build-wasm
```

## Run locally

You must serve the files over HTTP (workers/modules don’t work from `file://`).

```bash
make playground-serve
```

Open `http://localhost:8000/`.

## Features

| Mode | Description |
|------|-------------|
| **Brazil Band (Type A)** | CLs vs μ scan with ±1σ/±2σ expected bands, observed/expected upper limits |
| **Brazil Band (Type B — Mass Scan)** | ATLAS/CMS-style μ_up vs mass with Brazil bands. Auto-generates N mass hypotheses by shifting signal peak across bins, runs CLs at each. Toggle "Mass Scan (Type B)" checkbox. |
| **Profile Scan** | −2Δln L profile likelihood curve with 1σ/95% CL thresholds |
| **MLE Fit** | Maximum likelihood parameter estimates with uncertainties |
| **Hypothesis Test** | Single-point CLs p-value (asymptotic q̃_μ) |
| **GLM Regression** | Linear, logistic, and Poisson regression via L-BFGS-B |

### Type B Mass Scan

When "Mass Scan (Type B)" is enabled in Brazil Band mode, the playground:

1. Parses the workspace and identifies signal samples (those with a `normfactor` modifier matching the POI)
2. Creates N variants (default 7) where the signal yield is redistributed across bins using a Gaussian kernel centered at different positions — simulating different mass hypotheses
3. Runs the full asymptotic CLs computation for each variant
4. Plots 95% CL upper limit on μ vs signal peak position, with ±1σ (green) / ±2σ (gold) expected bands and observed (gold dots)
5. The horizontal red line at μ=1 marks the exclusion threshold — mass points below it are excluded at 95% CL

### Examples

7 preloaded examples available via the dropdown:

- **Simple counting** — 2-bin, 1 channel
- **Shape analysis** — 10-bin with peaked signal
- **Multi-channel** — 2 channels × 5 bins (electron + muon)
- **Discovery** — excess in data (μ̂ > 0)
- **GLM: Linear** — 20 obs × 2 features
- **GLM: Logistic** — 30 obs × 2 features (binary)
- **GLM: Poisson** — 25 obs × 1 feature (counts)

## Native CLI: `nextstat mass-scan`

The playground's Type B mass scan has a native CLI equivalent for production use:

```bash
# Prepare a directory with one workspace.json per mass hypothesis
mkdir mass_points/
# ... generate workspaces (e.g. different signal templates at each mass)

# Run batch CLs upper limits
nextstat mass-scan \
  --workspaces-dir ./mass_points/ \
  --alpha 0.05 \
  --scan-start 0 --scan-stop 5 --scan-points 41 \
  --labels "100 GeV,200 GeV,300 GeV,400 GeV,500 GeV" \
  -o brazil_band.json
```

Output JSON schema:

```json
{
  "command": "mass_scan",
  "alpha": 0.05,
  "scan": { "start": 0.0, "stop": 5.0, "points": 41 },
  "n_mass_points": 5,
  "wall_time_s": 1.23,
  "mass_points": [
    {
      "index": 0,
      "label": "100 GeV",
      "workspace": "mass_100.json",
      "obs_limit": 2.93,
      "exp_limits": [-2σ, -1σ, median, +1σ, +2σ],
      "mu_hat": 1.26,
      "converged": true
    }
  ]
}
```

Comparable to TRExFitter `Limit` action. Files in `--workspaces-dir` are sorted lexicographically — use zero-padded names (`mass_100.json`, `mass_200.json`) for correct ordering.

## Deploy (GitHub Pages)

GitHub Pages can serve the `playground/` directory, but you must ensure `playground/pkg/` is built and present in the published artifact.
Use either:
- commit `playground/pkg/` (quickest for a demo), or
- add a Pages workflow that runs `scripts/playground_build_wasm.sh` and publishes `playground/`.
