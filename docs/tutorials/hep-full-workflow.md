# HEP Full Workflow Tutorial

A complete, end-to-end guide to performing a HistFactory statistical analysis with NextStat.
This tutorial covers every step from building your first workspace JSON to producing
publication-quality exclusion limits — using both the Python API and the CLI.

**Audience:** Graduate students, postdocs, and analysts who want to run HistFactory fits
without ROOT or a heavy C++ framework. Familiarity with basic statistics (likelihood,
p-values) is helpful but not required — we explain everything along the way.

**Time:** ~30 minutes to read; ~10 minutes to run all examples.

---

## Table of Contents

1. [What Is a HistFactory Analysis?](#1-what-is-a-histfactory-analysis)
2. [Workspace Anatomy](#2-workspace-anatomy)
3. [Building Workspaces](#3-building-workspaces)
   - 3.1 Minimal single-channel
   - 3.2 Realistic multi-channel (SR + CR)
   - 3.3 All modifier types
   - 3.4 Measurement configuration
4. [Workspace Auditing](#4-workspace-auditing)
5. [Maximum Likelihood Fit (MLE)](#5-maximum-likelihood-fit-mle)
   - 5.1 Unconditional fit
   - 5.2 Conditional fit (fixed μ)
   - 5.3 Fit to specific regions
   - 5.4 Asimov fit
   - 5.5 Understanding fit results
6. [Hypothesis Testing (CLs)](#6-hypothesis-testing-cls)
   - 6.1 Asymptotic CLs
   - 6.2 Expected bands
   - 6.3 Toy-based CLs
   - 6.4 Discovery significance
7. [Upper Limits (Brazil Band)](#7-upper-limits-brazil-band)
   - 7.1 Bisection mode
   - 7.2 Scan mode
   - 7.3 Profile likelihood scan
8. [Diagnostics & Visualization](#8-diagnostics--visualization)
   - 8.1 Nuisance parameter ranking (impact)
   - 8.2 Pull plots
   - 8.3 Correlation matrix
   - 8.4 Pre/post-fit distributions
   - 8.5 Goodness of fit
   - 8.6 Gamma (stat NP) plots
9. [Advanced Topics](#9-advanced-topics)
   - 9.1 Workspace combination
   - 9.2 Mass scan (multiple signal hypotheses)
   - 9.3 HS3 format (ROOT interop)
   - 9.4 GPU acceleration
   - 9.5 Preprocessing (smoothing & pruning)
   - 9.6 Automated report generation
10. [Complete Analysis Script](#10-complete-analysis-script)

---

## 1. What Is a HistFactory Analysis?

HistFactory is the standard statistical model used by ATLAS, CMS, and other LHC
experiments for binned likelihood analyses. The model describes:

- **Channels** (regions): Signal Region (SR), Control Region (CR), Validation Region (VR)
- **Samples**: Signal, backgrounds (ttbar, W+jets, QCD, etc.)
- **Modifiers**: Systematic uncertainties that change yields or shapes
- **Observations**: The actual measured data in each bin

The likelihood is a product of Poisson terms (one per bin) multiplied by constraint
terms for systematic uncertainties:

```
L(μ, θ) = ∏_bins Poisson(n_i | ν_i(μ, θ)) × ∏_constraints C(θ)
```

Where:
- **μ** (mu) is the signal strength — the parameter of interest (POI)
- **θ** (theta) are nuisance parameters — systematic uncertainties
- **ν_i** is the expected event count in bin i, which depends on μ and θ

NextStat implements this likelihood identically to pyhf, but runs 10-100× faster
thanks to its Rust core and L-BFGS-B optimizer.

---

## 2. Workspace Anatomy

A workspace is a single JSON file with four top-level keys:

```json
{
  "channels": [...],       // regions with samples and modifiers
  "observations": [...],   // measured data per channel
  "measurements": [...],   // POI and parameter config
  "version": "1.0.0"       // always "1.0.0" for pyhf format
}
```

### Channels

Each channel represents a region (SR, CR, VR) with one or more samples:

```json
{
  "name": "SR",
  "samples": [
    {
      "name": "signal",
      "data": [5.0, 10.0, 8.0],           // expected yields per bin
      "modifiers": [...]                    // systematic uncertainties
    },
    {
      "name": "ttbar",
      "data": [50.0, 60.0, 45.0],
      "modifiers": [...]
    }
  ]
}
```

### Observations

One entry per channel, matching the channel name:

```json
{
  "name": "SR",
  "data": [55.0, 72.0, 51.0]    // observed event counts per bin
}
```

### Measurements

Defines the POI and optional parameter overrides:

```json
{
  "name": "MyMeasurement",
  "config": {
    "poi": "mu",
    "parameters": [
      {
        "name": "lumi",
        "auxdata": [1.0],
        "sigmas": [0.02],
        "bounds": [[0.9, 1.1]],
        "inits": [1.0]
      }
    ]
  }
}
```

---

## 3. Building Workspaces

### 3.1 Minimal single-channel workspace

The simplest possible workspace: one SR, one signal, one background, one systematic:

```bash
cat > simple.json << 'EOF'
{
  "channels": [
    {
      "name": "SR",
      "samples": [
        {
          "name": "signal",
          "data": [5.0, 10.0],
          "modifiers": [
            { "name": "mu", "type": "normfactor", "data": null }
          ]
        },
        {
          "name": "background",
          "data": [50.0, 60.0],
          "modifiers": [
            { "name": "uncorr_bkguncrt", "type": "shapesys", "data": [5.0, 12.0] }
          ]
        }
      ]
    }
  ],
  "observations": [
    { "name": "SR", "data": [55.0, 65.0] }
  ],
  "measurements": [
    { "name": "Measurement", "config": { "poi": "mu", "parameters": [] } }
  ],
  "version": "1.0.0"
}
EOF
```

### 3.2 Realistic multi-channel workspace (SR + CR)

A more realistic analysis with:
- **Signal Region (SR)**: where you look for new physics
- **Control Region (CR)**: enriched in background, constrains systematics
- Multiple systematic uncertainties: luminosity, normalization, shape, statistical

```bash
cat > multichannel.json << 'EOF'
{
  "channels": [
    {
      "name": "SR",
      "samples": [
        {
          "name": "signal",
          "data": [12.0, 11.0],
          "modifiers": [
            { "name": "mu", "type": "normfactor", "data": null },
            { "name": "lumi", "type": "lumi", "data": null },
            { "name": "sig_theory", "type": "normsys", "data": { "hi": 1.05, "lo": 0.95 } }
          ]
        },
        {
          "name": "ttbar",
          "data": [100.0, 105.0],
          "modifiers": [
            { "name": "lumi", "type": "lumi", "data": null },
            { "name": "ttbar_xsec", "type": "normsys", "data": { "hi": 1.06, "lo": 0.94 } },
            {
              "name": "jet_energy_scale",
              "type": "histosys",
              "data": {
                "hi_data": [110.0, 115.5],
                "lo_data": [90.0, 94.5]
              }
            },
            { "name": "staterror_SR", "type": "staterror", "data": [10.0, 11.0] }
          ]
        },
        {
          "name": "wjets",
          "data": [20.0, 15.0],
          "modifiers": [
            { "name": "lumi", "type": "lumi", "data": null },
            { "name": "wjets_norm", "type": "normsys", "data": { "hi": 1.30, "lo": 0.70 } }
          ]
        }
      ]
    },
    {
      "name": "CR_ttbar",
      "samples": [
        {
          "name": "ttbar",
          "data": [500.0, 510.0],
          "modifiers": [
            { "name": "lumi", "type": "lumi", "data": null },
            { "name": "ttbar_xsec", "type": "normsys", "data": { "hi": 1.06, "lo": 0.94 } },
            {
              "name": "jet_energy_scale",
              "type": "histosys",
              "data": {
                "hi_data": [550.0, 561.0],
                "lo_data": [450.0, 459.0]
              }
            }
          ]
        },
        {
          "name": "wjets",
          "data": [50.0, 40.0],
          "modifiers": [
            { "name": "lumi", "type": "lumi", "data": null },
            { "name": "wjets_norm", "type": "normsys", "data": { "hi": 1.30, "lo": 0.70 } }
          ]
        }
      ]
    }
  ],
  "observations": [
    { "name": "SR", "data": [130.0, 128.0] },
    { "name": "CR_ttbar", "data": [555.0, 548.0] }
  ],
  "measurements": [
    {
      "name": "search",
      "config": {
        "poi": "mu",
        "parameters": [
          {
            "name": "lumi",
            "auxdata": [1.0],
            "sigmas": [0.017],
            "bounds": [[0.9, 1.1]],
            "inits": [1.0]
          }
        ]
      }
    }
  ],
  "version": "1.0.0"
}
EOF
```

**Key design patterns:**
- The `ttbar_xsec` and `jet_energy_scale` NPs appear in both SR and CR — the CR
  constrains them, reducing their impact on the SR signal extraction.
- `lumi` is shared across all samples — it's a correlated normalization uncertainty.
- `staterror_SR` accounts for limited MC statistics in the SR.

### 3.3 All modifier types

| Modifier | JSON `type` | Constraint | Use Case |
|----------|-------------|-----------|----------|
| **normfactor** | `normfactor` | None (free) | Signal strength (μ), free normalization |
| **normsys** | `normsys` | Gaussian | Cross-section uncertainty, normalization syst. |
| **histosys** | `histosys` | Gaussian | Shape variation (up/down templates) |
| **shapesys** | `shapesys` | Poisson | Per-bin uncorrelated uncertainty |
| **staterror** | `staterror` | Gaussian | MC statistical uncertainty (Barlow-Beeston) |
| **lumi** | `lumi` | Gaussian | Luminosity uncertainty (correlated across samples) |
| **shapefactor** | `shapefactor` | None (free) | Per-bin free normalization (data-driven bkg) |

#### normfactor — free normalization

```json
{ "name": "mu", "type": "normfactor", "data": null }
```

No constraint term. The parameter floats freely in the fit. Used for:
- Signal strength (μ)
- Data-driven background normalization

#### normsys — Gaussian-constrained normalization

```json
{ "name": "ttbar_xsec", "type": "normsys", "data": { "hi": 1.06, "lo": 0.94 } }
```

A ±1σ variation that scales the entire sample yield. At α=+1, yield is multiplied by
`hi` (1.06); at α=-1, by `lo` (0.94). Interpolation between these is smooth (Code4 by
default, matching ROOT/TRExFitter).

#### histosys — shape variation

```json
{
  "name": "jet_energy_scale",
  "type": "histosys",
  "data": {
    "hi_data": [110.0, 115.5],
    "lo_data": [90.0, 94.5]
  }
}
```

Provides per-bin up/down templates. At α=+1, the nominal histogram is replaced by
`hi_data`; at α=-1, by `lo_data`. The interpolation scheme determines the behavior
between these points.

#### shapesys — per-bin Poisson constraint

```json
{ "name": "uncorr_bkguncrt", "type": "shapesys", "data": [5.0, 12.0] }
```

Each bin gets its own nuisance parameter with a Poisson constraint. The `data` array
gives the absolute uncertainty per bin. Use this when bins are statistically independent.

#### staterror — Barlow-Beeston lite

```json
{ "name": "staterror_SR", "type": "staterror", "data": [10.0, 11.0] }
```

Models the MC statistical uncertainty in the signal region. Uses the Barlow-Beeston lite
technique: one NP per bin, Gaussian-constrained. Multiple `staterror` modifiers in the
same channel are merged automatically.

#### lumi — luminosity

```json
{ "name": "lumi", "type": "lumi", "data": null }
```

A correlated normalization uncertainty shared across all samples that reference it. The
actual uncertainty value comes from the measurement's `parameters` block (see Section 3.4).

#### shapefactor — per-bin free normalization

```json
{ "name": "shapefactor_CR", "type": "shapefactor", "data": null }
```

One free (unconstrained) parameter per bin. Used for data-driven background estimation
in control regions.

### 3.4 Measurement configuration

The measurement block configures the POI and parameter overrides:

```json
{
  "name": "search",
  "config": {
    "poi": "mu",
    "parameters": [
      {
        "name": "lumi",
        "auxdata": [1.0],
        "sigmas": [0.017],
        "bounds": [[0.9, 1.1]],
        "inits": [1.0]
      },
      {
        "name": "mu",
        "bounds": [[-5.0, 10.0]],
        "inits": [1.0]
      }
    ]
  }
}
```

| Field | Purpose |
|-------|---------|
| `poi` | Name of the parameter of interest |
| `inits` | Starting value for the fit |
| `bounds` | Allowed range `[lo, hi]` |
| `auxdata` | Auxiliary data for the constraint (e.g., nominal value) |
| `sigmas` | Width of the Gaussian constraint |
| `fixed` | If `true`, parameter is held constant during fit |

---

## 4. Workspace Auditing

Before fitting, always audit your workspace to catch issues:

### CLI

```bash
nextstat audit --input multichannel.json
```

Expected output:

```
Workspace Audit
===============
Format:    pyhf v1.0.0
Channels:  2 (SR, CR_ttbar)
Samples:   3 (signal, ttbar, wjets)
Bins:      4 total (2 per channel)
Modifiers: 6 unique
  normfactor:  1 (mu)
  normsys:     3 (sig_theory, ttbar_xsec, wjets_norm)
  histosys:    1 (jet_energy_scale)
  staterror:   1 (staterror_SR)
  lumi:        1 (lumi)
Parameters:  8 total (1 POI + 7 nuisance)
POI:         mu
Measurements: 1 (search)
```

### Python

```python
import json, nextstat

with open("multichannel.json") as f:
    ws = json.load(f)

model = nextstat.from_pyhf(json.dumps(ws))
print("Number of parameters:", model.n_params())
print("POI index:          ", model.poi_index())
print("Parameter names:    ", model.param_names())
print("Parameter bounds:   ", model.param_bounds())
```

---

## 5. Maximum Likelihood Fit (MLE)

The MLE finds the parameter values that maximize the likelihood (minimize the NLL).

### 5.1 Unconditional fit

All parameters (μ and θ) float freely.

#### Python

```python
import json, nextstat

with open("multichannel.json") as f:
    ws = json.load(f)

model = nextstat.from_pyhf(json.dumps(ws))
result = nextstat.fit(model)

poi = model.poi_index()
names = model.param_names()

print(f"Signal strength (mu): {result.bestfit[poi]:.4f} ± {result.uncertainties[poi]:.4f}")
print(f"NLL at minimum:       {result.nll:.4f}")
print(f"Converged:            {result.converged}")
print(f"Iterations:           {result.n_iter}")
print(f"EDM:                  {result.edm:.2e}")
print()
print("All parameters:")
for i, (name, val, unc) in enumerate(zip(names, result.bestfit, result.uncertainties)):
    marker = " <-- POI" if i == poi else ""
    print(f"  {name:25s} = {val:+.4f} ± {unc:.4f}{marker}")
```

#### CLI

```bash
nextstat fit --input multichannel.json
```

### 5.2 Conditional fit (fixed μ)

Fix μ to a specific value and profile over nuisance parameters only.
This is used internally by hypothesis tests, but you can do it manually:

```python
import json, nextstat

with open("multichannel.json") as f:
    ws = json.load(f)

model = nextstat.from_pyhf(json.dumps(ws))

# Fix mu=0 (background-only fit)
result_bonly = nextstat.fit(model, fixed_params={"mu": 0.0})
print(f"Background-only NLL: {result_bonly.nll:.4f}")

# Fix mu=1 (signal+background fit at nominal strength)
result_sb = nextstat.fit(model, fixed_params={"mu": 1.0})
print(f"S+B NLL (mu=1):      {result_sb.nll:.4f}")

# The test statistic q(mu) = 2 * (NLL_conditional - NLL_unconditional)
result_free = nextstat.fit(model)
q_mu = 2.0 * (result_sb.nll - result_free.nll)
print(f"q(mu=1):             {q_mu:.4f}")
```

### 5.3 Fit to specific regions

Fit only to the signal region, ignoring the control region:

```bash
nextstat fit --input multichannel.json --fit-regions SR
```

Or fit only to the control region (useful for validating background model):

```bash
nextstat fit --input multichannel.json --fit-regions CR_ttbar
```

### 5.4 Asimov fit

An Asimov dataset replaces observed data with the expected yields at specific parameter
values (typically μ=1, θ=nominal). This gives the median expected sensitivity:

```bash
nextstat fit --input multichannel.json --asimov
```

```python
result_asimov = nextstat.fit(model, asimov=True)
print(f"Asimov mu: {result_asimov.bestfit[model.poi_index()]:.4f}")
# Should be very close to 1.0 by construction
```

### 5.5 Understanding fit results

| Field | Meaning | Good Value |
|-------|---------|-----------|
| `converged` | Did the optimizer converge? | `True` |
| `edm` | Estimated Distance to Minimum | < 1e-3 |
| `n_iter` | Number of optimizer iterations | < 500 |
| `final_grad_norm` | Gradient norm at minimum | < 1e-3 |
| `n_active_bounds` | Parameters stuck at boundaries | 0 (ideally) |
| `warnings` | Identifiability warnings | Empty list |

**Red flags:**
- `converged = False` → increase max iterations or check model
- `n_active_bounds > 0` → a parameter hit its boundary, widen bounds
- `warnings` contains "near-singular Hessian" → model may be overparameterized
- Large `edm` → optimizer didn't reach a true minimum

---

## 6. Hypothesis Testing (CLs)

The CLs method is the standard procedure for setting exclusion limits in HEP.
It answers: "Can we exclude a signal of strength μ at 95% confidence?"

### 6.1 Asymptotic CLs

Uses the asymptotic approximation (Wald approximation) — fast, no toys needed.

#### Python

```python
import json, nextstat

with open("multichannel.json") as f:
    ws = json.load(f)

model = nextstat.from_pyhf(json.dumps(ws))

result = nextstat.hypotest(model, mu_test=1.0)

print(f"CLs:            {result.cls:.6f}")
print(f"CLs+b:          {result.clsb:.6f}")
print(f"CLb:            {result.clb:.6f}")
print(f"Excluded (95%)? {'YES' if result.cls < 0.05 else 'NO'}")
```

#### CLI

```bash
nextstat hypotest --input multichannel.json --mu 1.0
```

**Interpreting the result:**
- `CLs < 0.05` → signal of strength μ is **excluded** at 95% CL
- `CLs > 0.05` → signal is **not excluded** (data is compatible with signal)
- `CLs+b` is the p-value of the signal+background hypothesis
- `CLb` is the p-value of the background-only hypothesis
- `CLs = CLs+b / CLb` (the modified frequentist ratio)

### 6.2 Expected bands

The expected CLs values tell you what sensitivity your analysis has *before looking at data*:

#### Python

```python
result = nextstat.hypotest(model, mu_test=1.0, expected_set=True)

print(f"Observed CLs:     {result.cls:.4f}")
print(f"Expected -2σ CLs: {result.expected_set[0]:.4f}")
print(f"Expected -1σ CLs: {result.expected_set[1]:.4f}")
print(f"Expected median:  {result.expected_set[2]:.4f}")
print(f"Expected +1σ CLs: {result.expected_set[3]:.4f}")
print(f"Expected +2σ CLs: {result.expected_set[4]:.4f}")
```

#### CLI

```bash
nextstat hypotest --input multichannel.json --mu 1.0 --expected-set
```

**Band ordering:** `[+2σ, +1σ, median, -1σ, -2σ]` in the `-μ̂/σ` convention
(matching pyhf). The +2σ band corresponds to the strongest expected exclusion.

### 6.3 Toy-based CLs

For small expected yields or when the asymptotic approximation breaks down, use toys:

#### Python

```python
result = nextstat.hypotest_toys(
    model,
    mu_test=1.0,
    n_toys=10000,
    seed=42
)
print(f"Toy CLs: {result.cls:.4f}")
```

#### CLI

```bash
# CPU — all cores
nextstat hypotest-toys --input multichannel.json \
  --mu 1.0 --n-toys 10000 --seed 42 --threads 0

# GPU (NVIDIA)
nextstat hypotest-toys --input multichannel.json \
  --mu 1.0 --n-toys 10000 --seed 42 --gpu cuda

# GPU (Apple Silicon)
nextstat hypotest-toys --input multichannel.json \
  --mu 1.0 --n-toys 10000 --seed 42 --gpu metal
```

**When to use toys vs asymptotic:**
- Asymptotic is fine when expected yields per bin are > ~10
- Use toys when yields are small (< 5 events in any bin)
- Use toys for validating asymptotic results in publications

### 6.4 Discovery significance

Test the background-only hypothesis (μ=0). If rejected, you have evidence for signal:

```bash
nextstat significance --input multichannel.json
```

```python
sig = nextstat.significance(model)
print(f"Observed significance: {sig.observed:.2f} σ")
print(f"Expected significance: {sig.expected:.2f} σ")
# 5σ = discovery, 3σ = evidence
```

---

## 7. Upper Limits (Brazil Band)

An upper limit answers: "What is the maximum signal strength compatible with the data?"

### 7.1 Bisection mode (fast)

Finds the exact μ where CLs = 0.05 using bisection:

```bash
# Observed only
nextstat upper-limit --input multichannel.json

# Observed + expected (Brazil band)
nextstat upper-limit --input multichannel.json --expected
```

```python
limits = nextstat.upper_limit(model)
print(f"Observed:       {limits.observed:.4f}")
print(f"Expected -2σ:   {limits.expected_minus2:.4f}")
print(f"Expected -1σ:   {limits.expected_minus1:.4f}")
print(f"Expected median:{limits.expected:.4f}")
print(f"Expected +1σ:   {limits.expected_plus1:.4f}")
print(f"Expected +2σ:   {limits.expected_plus2:.4f}")
```

### 7.2 Scan mode (for plotting)

Scans CLs across a μ grid — needed for Brazil band plots:

```bash
nextstat upper-limit --input multichannel.json \
  --expected \
  --scan-start 0.0 --scan-stop 5.0 --scan-points 201
```

This returns per-point CLs values that you can plot.

### 7.3 Profile likelihood scan

Compute q(μ) = 2·ΔNLL across a μ grid for the profile likelihood curve:

```bash
nextstat scan --input multichannel.json \
  --start 0.0 --stop 3.0 --points 31
```

```python
mu_values = [i * 0.1 for i in range(31)]  # 0.0 to 3.0
scan = nextstat.profile_scan(model, mu_values)

print(f"Best-fit mu:  {scan['mu_hat']:.4f}")
print(f"NLL at min:   {scan['nll_hat']:.4f}")
for pt in scan["points"][:5]:
    print(f"  mu={pt['mu']:.1f}  q(mu)={pt['q_mu']:.4f}")
```

**Reading a profile likelihood curve:**
- The minimum is at μ̂ (best-fit signal strength)
- Δ(2·NLL) = 1.0 gives the ±1σ interval
- Δ(2·NLL) = 3.84 gives the 95% CL interval

---

## 8. Diagnostics & Visualization

NextStat produces plot-friendly JSON artifacts for every diagnostic.
You can render them with matplotlib, ROOT, or any plotting library.

### 8.1 Nuisance parameter ranking (impact)

The ranking shows which systematics have the largest impact on μ:

```bash
nextstat viz ranking --input multichannel.json --output ranking.json
```

```python
ranking = nextstat.ranking(model)
print("Top impacts on mu:")
for r in ranking[:10]:
    print(f"  {r['name']:25s}  Δμ(+1σ)={r['impact_up']:+.4f}  Δμ(-1σ)={r['impact_down']:+.4f}")
```

**Interpreting ranking:**
- Parameters with large |impact| dominate the uncertainty on μ
- If a nuisance pulls far from 0 and has large impact, investigate further

### 8.2 Pull plots

Pulls show how much each nuisance parameter shifted from its nominal value:

```bash
# First, save the fit result
nextstat fit --input multichannel.json > fit.json

# Then generate pull artifact
nextstat viz pulls --input multichannel.json --fit fit.json --output pulls.json
```

```python
import json
result = nextstat.fit(model)

# Pull = (θ_hat - θ_nominal) / σ_θ
names = model.param_names()
poi = model.poi_index()
for i, (name, val, unc) in enumerate(zip(names, result.bestfit, result.uncertainties)):
    if i == poi:
        continue  # skip POI
    # For constrained NPs, nominal is typically 0 (or 1 for gammas)
    pull = val / unc if unc > 0 else 0.0
    constraint = unc  # post-fit constraint width
    print(f"  {name:25s}  pull={pull:+.2f}  constraint={constraint:.3f}")
```

**Red flags in pulls:**
- |pull| > 2: the data is pulling this NP far from its prior
- Constraint << 1: the data is strongly constraining this NP (good)
- Constraint ≈ 1: the data doesn't constrain this NP beyond the prior

### 8.3 Correlation matrix

```bash
nextstat viz corr --input multichannel.json --fit fit.json --output corr.json
```

```python
corr = nextstat.correlation_matrix(model)
# corr is a dict with 'names' and 'matrix' (2D list)
```

**Interpreting correlations:**
- |ρ| > 0.5 between NPs → they are degenerate, consider reparameterizing
- Large ρ between μ and a NP → that systematic dominates the μ uncertainty

### 8.4 Pre/post-fit distributions

Generate publication-quality distribution comparisons:

```bash
nextstat viz distributions --input multichannel.json \
  --fit fit.json \
  --output distributions.json
```

The output contains pre-fit and post-fit yields per channel, per sample, per bin,
plus systematic uncertainty bands.

### 8.5 Goodness of fit

```bash
nextstat goodness-of-fit --input multichannel.json
```

```python
gof = nextstat.goodness_of_fit(model)
print(f"Saturated q: {gof.statistic:.2f}")
print(f"ndf:         {gof.ndf}")
print(f"p-value:     {gof.pvalue:.4f}")
```

A low p-value (< 0.05) indicates the model doesn't describe the data well.

### 8.6 Gamma plots

For models with staterror/shapesys modifiers, visualize the per-bin gamma NPs:

```bash
nextstat viz gammas --input multichannel.json --fit fit.json --output gammas.json
```

---

## 9. Advanced Topics

### 9.1 Workspace combination

Combine two or more workspaces into a single joint likelihood:

```bash
nextstat combine workspace_ee.json workspace_mumu.json \
  --output combined.json \
  --prefix-channels
```

```python
combined_ws = nextstat.combine([ws_ee_json, ws_mumu_json], prefix_channels=True)
model = nextstat.from_pyhf(combined_ws)
result = nextstat.fit(model)
```

The `--prefix-channels` flag prepends the workspace index to channel names to avoid
collisions. Shared nuisance parameters (same name) are automatically correlated.

### 9.2 Mass scan (multiple signal hypotheses)

Run upper limits across a grid of signal mass points:

```bash
nextstat mass-scan --workspaces-dir mass_workspaces/ \
  --scan-start 0 --scan-stop 5 --scan-points 41 \
  --labels m100,m200,m300,m400,m500
```

Each file in `mass_workspaces/` is a workspace for one mass point. The output is a
JSON with observed and expected limits per mass point — ready for a mass-vs-cross-section
exclusion plot.

### 9.3 HS3 format (ROOT 6.37+)

NextStat natively reads HS3 v0.2 JSON (the HEP Statistics Serialization Standard
produced by ROOT 6.37+). Format is auto-detected:

```bash
# Works transparently — auto-detects pyhf vs HS3
nextstat fit --input workspace-postFit_PTV.json
```

```python
# Auto-detect
model = nextstat.HistFactoryModel.from_workspace(hs3_json_string)

# Explicit HS3
model = nextstat.HistFactoryModel.from_hs3(hs3_json_string, analysis="combPdf_obsData")

# Export back to HS3
hs3_output = model.to_hs3()
```

### 9.4 GPU acceleration

NextStat supports CUDA (NVIDIA) and Metal (Apple Silicon) for:
- Toy-based hypothesis tests
- Batch NLL evaluation
- Profile likelihood scans

```bash
# GPU scan (shares one GpuSession across all points, warm-start)
nextstat scan --input multichannel.json \
  --start 0 --stop 5 --points 21 --gpu cuda

# GPU toys (massive parallelism)
nextstat hypotest-toys --input multichannel.json \
  --mu 1.0 --n-toys 100000 --gpu cuda
```

**Speedup:** 10-50× over CPU for toy-based tests, depending on model complexity.

### 9.5 Preprocessing (smoothing & pruning)

Clean up workspaces before fitting:

```bash
# Smooth small shape variations (reduces fit instabilities)
nextstat preprocess smooth --input multichannel.json --output smoothed.json

# Prune negligible systematics (< 0.5% impact)
nextstat preprocess prune --input multichannel.json --output pruned.json --threshold 0.005
```

### 9.6 Automated report generation

Generate a complete analysis report with all plots:

```bash
nextstat report --input multichannel.json \
  --out-dir report/ \
  --render \
  --pdf report.pdf
```

This generates:
- Pre/post-fit distributions for all channels
- Pull plots
- Correlation matrix
- NP ranking
- Yield tables
- Profile likelihood curve
- CLs Brazil band

---

## 10. Complete Analysis Script

Here is a single Python script that runs the entire HEP workflow from start to finish:

```python
#!/usr/bin/env python3
"""
Complete HEP analysis workflow with NextStat.
Produces: fit results, hypothesis test, upper limits, ranking, pulls.
"""
import json
import nextstat

# ── 1. Load workspace ──────────────────────────────────────────
with open("multichannel.json") as f:
    workspace = json.load(f)

model = nextstat.from_pyhf(json.dumps(workspace))
poi = model.poi_index()
names = model.param_names()

print("=" * 60)
print("NEXTSTAT HEP ANALYSIS")
print("=" * 60)
print(f"Parameters: {model.n_params()} ({len(names)} named)")
print(f"POI:        {names[poi]} (index {poi})")
print()

# ── 2. Unconditional MLE fit ──────────────────────────────────
print("─" * 60)
print("STEP 1: Maximum Likelihood Fit")
print("─" * 60)

result = nextstat.fit(model)
mu_hat = result.bestfit[poi]
mu_unc = result.uncertainties[poi]

print(f"  mu = {mu_hat:.4f} ± {mu_unc:.4f}")
print(f"  NLL = {result.nll:.4f}")
print(f"  Converged: {result.converged}")
print(f"  EDM: {result.edm:.2e}")
print()

# ── 3. Hypothesis test ────────────────────────────────────────
print("─" * 60)
print("STEP 2: Hypothesis Test (CLs at mu=1.0)")
print("─" * 60)

hypo = nextstat.hypotest(model, mu_test=1.0, expected_set=True)

print(f"  Observed CLs:     {hypo.cls:.4f}")
print(f"  Expected -2σ:     {hypo.expected_set[0]:.4f}")
print(f"  Expected -1σ:     {hypo.expected_set[1]:.4f}")
print(f"  Expected median:  {hypo.expected_set[2]:.4f}")
print(f"  Expected +1σ:     {hypo.expected_set[3]:.4f}")
print(f"  Expected +2σ:     {hypo.expected_set[4]:.4f}")
print(f"  Excluded at 95%?  {'YES' if hypo.cls < 0.05 else 'NO'}")
print()

# ── 4. Upper limits ───────────────────────────────────────────
print("─" * 60)
print("STEP 3: Upper Limits (95% CL)")
print("─" * 60)

limits = nextstat.upper_limit(model)

print(f"  Observed:       {limits.observed:.4f}")
print(f"  Expected -2σ:   {limits.expected_minus2:.4f}")
print(f"  Expected -1σ:   {limits.expected_minus1:.4f}")
print(f"  Expected median:{limits.expected:.4f}")
print(f"  Expected +1σ:   {limits.expected_plus1:.4f}")
print(f"  Expected +2σ:   {limits.expected_plus2:.4f}")
print()

# ── 5. NP ranking ─────────────────────────────────────────────
print("─" * 60)
print("STEP 4: Nuisance Parameter Ranking (top 10)")
print("─" * 60)

ranking = nextstat.ranking(model)
for i, r in enumerate(ranking[:10]):
    print(f"  {i+1:2d}. {r['name']:25s}  Δμ(+1σ)={r['impact_up']:+.4f}  Δμ(-1σ)={r['impact_down']:+.4f}")
print()

# ── 6. Pulls ──────────────────────────────────────────────────
print("─" * 60)
print("STEP 5: Nuisance Parameter Pulls")
print("─" * 60)

for i, (name, val, unc) in enumerate(zip(names, result.bestfit, result.uncertainties)):
    if i == poi:
        continue
    pull = val / unc if unc > 0 else 0.0
    bar = "█" * int(min(abs(pull) * 10, 30))
    sign = "+" if pull > 0 else "-"
    print(f"  {name:25s}  {sign}{abs(pull):.2f}σ  {bar}")
print()

# ── 7. Profile likelihood scan ────────────────────────────────
print("─" * 60)
print("STEP 6: Profile Likelihood Scan")
print("─" * 60)

mu_values = [i * 0.2 for i in range(26)]  # 0 to 5
scan = nextstat.profile_scan(model, mu_values)

print(f"  mu_hat = {scan['mu_hat']:.4f}")
for pt in scan["points"]:
    bar = "▓" * int(min(pt["q_mu"] * 5, 40))
    print(f"  mu={pt['mu']:4.1f}  q(mu)={pt['q_mu']:7.3f}  {bar}")
print()

# ── 8. CLs scan for Brazil band ───────────────────────────────
print("─" * 60)
print("STEP 7: CLs Scan (Brazil Band Data)")
print("─" * 60)

scan_mu = [i * 0.05 for i in range(101)]  # 0 to 5
cls_artifact = nextstat.cls_curve(model, scan_mu, alpha=0.05)

print(f"  Observed limit: {cls_artifact['obs_limit']:.4f}")
print(f"  Expected limits: {cls_artifact['exp_limits']}")
print(f"  Scan points: {len(cls_artifact['points'])}")
print()

print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
```

### Plotting the Brazil band (matplotlib)

```python
import json
import matplotlib.pyplot as plt
import numpy as np

# Assuming cls_artifact from the script above
mu = [p["mu"] for p in cls_artifact["points"]]
cls_obs = [p["cls"] for p in cls_artifact["points"]]
cls_exp = np.array([p["expected"] for p in cls_artifact["points"]])

fig, ax = plt.subplots(figsize=(8, 5))

# Yellow band (±2σ)
ax.fill_between(mu, cls_exp[:, 0], cls_exp[:, 4],
                color="#FFC800", alpha=0.4, label="Expected ±2σ")
# Green band (±1σ)
ax.fill_between(mu, cls_exp[:, 1], cls_exp[:, 3],
                color="#00CC00", alpha=0.5, label="Expected ±1σ")
# Expected median
ax.plot(mu, cls_exp[:, 2], "k--", linewidth=1.5, label="Expected (median)")
# Observed
ax.plot(mu, cls_obs, "k-", linewidth=2, label="Observed")
# Alpha line
ax.axhline(0.05, color="red", linestyle=":", linewidth=1, label="95% CL")

ax.set_xlabel("Signal strength (μ)", fontsize=12)
ax.set_ylabel("CLs", fontsize=12)
ax.set_ylim(0, 1)
ax.set_xlim(0, 5)
ax.legend(loc="upper right", fontsize=10)
ax.set_title("CLs vs Signal Strength (Brazil Band)", fontsize=14)
plt.tight_layout()
plt.savefig("brazil_band.png", dpi=150)
plt.show()
```

### Plotting the NP ranking (matplotlib)

```python
import matplotlib.pyplot as plt

# ranking from nextstat.ranking(model)
top_n = ranking[:15]
names_r = [r["name"] for r in top_n][::-1]
up = [r["impact_up"] for r in top_n][::-1]
down = [r["impact_down"] for r in top_n][::-1]

fig, ax = plt.subplots(figsize=(8, 6))
y = range(len(names_r))

ax.barh(y, up, height=0.4, align="center", color="#4488CC", label="Δμ (+1σ)")
ax.barh(y, down, height=0.4, align="center", color="#CC4444", label="Δμ (-1σ)")

ax.set_yticks(y)
ax.set_yticklabels(names_r, fontsize=9)
ax.set_xlabel("Impact on μ", fontsize=12)
ax.set_title("Nuisance Parameter Ranking", fontsize=14)
ax.legend(loc="lower right")
ax.axvline(0, color="black", linewidth=0.5)
plt.tight_layout()
plt.savefig("ranking.png", dpi=150)
plt.show()
```

---

## CLI Quick Reference

| Task | Command |
|------|---------|
| Audit workspace | `nextstat audit --input ws.json` |
| Unconditional fit | `nextstat fit --input ws.json` |
| Fit specific regions | `nextstat fit --input ws.json --fit-regions SR` |
| Asimov fit | `nextstat fit --input ws.json --asimov` |
| Hypothesis test | `nextstat hypotest --input ws.json --mu 1.0` |
| Hypothesis test + bands | `nextstat hypotest --input ws.json --mu 1.0 --expected-set` |
| Toy hypothesis test | `nextstat hypotest-toys --input ws.json --mu 1.0 --n-toys 10000 --seed 42` |
| Discovery significance | `nextstat significance --input ws.json` |
| Upper limit (bisection) | `nextstat upper-limit --input ws.json --expected` |
| Upper limit (scan) | `nextstat upper-limit --input ws.json --scan-start 0 --scan-stop 5 --scan-points 201` |
| Profile scan | `nextstat scan --input ws.json --start 0 --stop 3 --points 31` |
| NP ranking | `nextstat viz ranking --input ws.json` |
| Pull plot | `nextstat viz pulls --input ws.json --fit fit.json` |
| Correlation matrix | `nextstat viz corr --input ws.json --fit fit.json` |
| Distributions | `nextstat viz distributions --input ws.json --fit fit.json` |
| CLs curve (Brazil) | `nextstat viz cls --input ws.json --scan-start 0 --scan-stop 5 --scan-points 201` |
| Profile curve | `nextstat viz profile --input ws.json --start 0 --stop 3 --points 31` |
| Goodness of fit | `nextstat goodness-of-fit --input ws.json` |
| Combine workspaces | `nextstat combine ws1.json ws2.json --output combined.json` |
| Mass scan | `nextstat mass-scan --workspaces-dir dir/ --labels m100,m200` |
| Smooth systematics | `nextstat preprocess smooth --input ws.json --output smooth.json` |
| Prune systematics | `nextstat preprocess prune --input ws.json --output pruned.json` |
| Full report | `nextstat report --input ws.json --out-dir report/ --render --pdf report.pdf` |

---

## What's Next

- [Installation & Quickstart](installation-quickstart.md) — get NextStat installed
- [Python API Reference](/docs/python-api) — all functions and classes
- [CLI Reference](/docs/cli) — all command-line options
- [HistFactory Models](/docs/histfactory) — workspace format details
- [GPU Acceleration](/docs/gpu) — CUDA and Metal setup
- [Bayesian Sampling](/docs/bayesian) — NUTS posterior inference
- [Visualization](/docs/visualization) — all artifact types and schemas
- [TRExFitter Interop](/docs/trexfitter) — import existing TRExFitter configs
