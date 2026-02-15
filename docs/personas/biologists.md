# NextStat for Biologists (PK/PD + NLME)

This guide maps pharmacometrics and biology concepts to NextStat APIs, helping you translate your domain knowledge into working code.

## Quick Start

- Quickstart: `docs/quickstarts/biologist.md`
- PK baseline: `docs/tutorials/phase-13-pk.md`
- NLME baseline: `docs/tutorials/phase-13-nlme.md`
- PK diagnostics: `docs/pharmacometrics/phase2-tutorial.md`
- Python API reference: `docs/references/python-api.md` (Pharmacometrics section)

## Biology → NextStat API Mapping

### Pharmacokinetic Models

| Biology Concept | Parameters | Python API | Notes |
|---|---|---|---|
| 1-compartment oral absorption | CL, V, Ka | `OneCompartmentOralPkModel(times, y, dose=, sigma=)` | First-order absorption, first-order elimination |
| 2-compartment IV bolus | CL, V1, V2, Q | `TwoCompartmentIvPkModel(times, y, dose=, error_model=, sigma=)` | Central + peripheral compartments |
| 2-compartment oral | CL, V1, V2, Q, Ka | `TwoCompartmentOralPkModel(times, y, dose=, error_model=, sigma=)` | Oral absorption + distribution |
| Bioavailability (F) | fraction absorbed | `bioavailability=0.85` in model constructor | F=1.0 for IV, <1.0 for oral with incomplete absorption |

### PK Parameter Interpretation

| Parameter | Biology Meaning | Units (typical) | Where in Results |
|---|---|---|---|
| CL (clearance) | Rate of drug removal from body | L/h | `fit.parameters[0]` or `result["theta"][0]` |
| V (volume of distribution) | Apparent volume drug distributes into | L | `fit.parameters[1]` or `result["theta"][1]` |
| Ka (absorption rate) | Speed of absorption from gut to blood | 1/h | `fit.parameters[2]` or `result["theta"][2]` |
| V1 (central volume) | Volume of central compartment (blood) | L | 2-cpt models |
| V2 (peripheral volume) | Volume of peripheral compartment (tissue) | L | 2-cpt models |
| Q (intercompartmental CL) | Rate of exchange between compartments | L/h | 2-cpt models |

### Error Models (Residual Variability)

| Biology Scenario | Error Model | Python API | When to Use |
|---|---|---|---|
| Assay precision is constant (e.g., LLOQ region) | Additive | `error_model="additive", sigma=0.05` | Low concentrations, biomarkers |
| Assay CV is constant (e.g., LC-MS/MS) | Proportional | `error_model="proportional", sigma=0.10` | Most PK data above LOQ |
| Both assay floor + proportional noise | Combined | `error_model="combined", sigma=0.10, sigma_add=0.02` | Rich sampling with BLQ data |

### Below Limit of Quantification (BLQ/LLOQ)

| Approach | `lloq_policy=` | When to Use |
|---|---|---|
| Drop BLQ observations | `"ignore"` | Quick exploratory analysis |
| Replace BLQ with LLOQ/2 | `"replace_half"` | Simple imputation (NONMEM M3) |
| Left-censored likelihood | `"censored"` | Gold standard (NONMEM M4) |

```python
model = nextstat.OneCompartmentOralPkModel(
    times, y, dose=100.0, sigma=0.05,
    lloq=0.01, lloq_policy="censored"
)
```

### Population PK (Mixed Effects)

| Biology Concept | NextStat API | Description |
|---|---|---|
| Population mean parameters (θ) | `nlme_foce(..., theta_init=[CL, V, Ka])` | Typical values for the population |
| Inter-individual variability (η) | `omega_init=[ω_CL, ω_V, ω_Ka]` | Standard deviations of random effects |
| Random effects (etas) | `result["eta"]` → `[[η_CL, η_V, η_Ka], ...]` | Per-subject deviations from population |
| Individual parameters | `θ_i = θ_pop × exp(η_i)` | Log-normal distribution |
| Objective function value (OFV) | `result["ofv"]` | -2·log(likelihood); lower = better fit |
| Correlated random effects | `nlme_foce(..., omega_init=...)` | CL-V correlation (e.g., larger patients clear faster) |

### Estimation Methods

| Method | Python API | When to Use | Speed |
|---|---|---|---|
| Maximum Likelihood (individual) | `MaximumLikelihoodEstimator().fit(model)` | Single subject, few parameters | Fastest |
| FOCE/FOCEI (population) | `nlme_foce(...)` | Standard population PK (<200 subjects) | Fast |
| SAEM (population) | `nlme_saem(...)` | Complex models, stochastic estimation | Moderate |

### Model Diagnostics

| What You Want | Python API | Output |
|---|---|---|
| Population prediction (PRED) | `pk_gof(...)` → `record["pred"]` | Prediction at η=0 (population mean) |
| Individual prediction (IPRED) | `pk_gof(...)` → `record["ipred"]` | Prediction at individual η̂ |
| Individual weighted residuals | `pk_gof(...)` → `record["iwres"]` | Should be ~N(0,1) if model is correct |
| Conditional weighted residuals | `pk_gof(...)` → `record["cwres"]` | FOCE-based residual (preferred) |
| Visual Predictive Check (VPC) | `pk_vpc(...)` | Simulated prediction intervals vs observed |

### Dose-Response (PD) Models

| Biology Concept | NextStat Model | Parameters |
|---|---|---|
| Emax dose-response | `EmaxModel` | E0, Emax, EC50 |
| Sigmoid Emax (Hill equation) | `SigmoidEmaxModel` | E0, Emax, EC50, Hill coefficient (γ) |
| Indirect response (turnover) | `IndirectResponseModel` | Kin, Kout, IC50/EC50, type (1-4) |

### Data I/O

| Task | Python API | Notes |
|---|---|---|
| Import NONMEM dataset | `read_nonmem(csv_text)` | Standard NONMEM CSV (ID, TIME, AMT, DV, EVID, MDV) |
| Export GOF residuals | `pk_gof(...)` → list of dicts | PRED, IPRED, IWRES, CWRES per observation |
| VPC simulation output | `pk_vpc(...)` → dict with bins | Binned quantiles + prediction intervals |

## Typical Workflow

```python
import nextstat

# 1. Load/generate data
data = nextstat.read_nonmem(open("warfarin.csv").read())

# 2. Population estimation (FOCE)
result = nextstat.nlme_foce(
    data["times"], data["dv"], data["subject_idx"], data["n_subjects"],
    dose=100.0, bioavailability=1.0,
    error_model="proportional", sigma=0.10,
    theta_init=[0.15, 8.0, 1.0],      # CL, V, Ka
    omega_init=[0.20, 0.15, 0.25],     # ω_CL, ω_V, ω_Ka
    max_outer_iter=300, interaction=True,
)

print(f"Population CL: {result['theta'][0]:.3f} L/h")
print(f"Population V:  {result['theta'][1]:.1f} L")
print(f"IIV on CL:     {result['omega'][0]:.1%}")

# 3. Diagnostics
gof = nextstat.pk_gof(
    data["times"], data["dv"], data["subject_idx"],
    dose=100.0, bioavailability=1.0,
    theta=result["theta"], eta=result["eta"],
    error_model="proportional", sigma=0.10,
)

# 4. VPC
vpc = nextstat.pk_vpc(
    data["times"], data["dv"], data["subject_idx"], data["n_subjects"],
    dose=100.0, bioavailability=1.0,
    theta=result["theta"],
    omega_matrix=result["omega_matrix"],
    error_model="proportional", sigma=0.10,
    n_sim=200, n_bins=10,
)
```

## NONMEM ↔ NextStat Translation

| NONMEM Concept | NextStat Equivalent |
|---|---|
| `$THETA` | `theta_init` in `nlme_foce()` / `nlme_saem()` |
| `$OMEGA` | `omega_init` (diagonal SDs) |
| `$SIGMA` | `sigma` / `sigma_add` in error model |
| `ADVAN2 TRANS2` | `OneCompartmentOralPkModel` |
| `ADVAN4 TRANS4` | `TwoCompartmentOralPkModel` |
| `ADVAN3 TRANS4` | `TwoCompartmentIvPkModel` |
| `$EST METHOD=COND INTER` | `nlme_foce(..., interaction=True)` |
| `$EST METHOD=SAEM` | `nlme_saem(...)` |
| `CWRES` | `pk_gof(...)` → `record["cwres"]` |
| `M3` (BLQ handling) | `lloq_policy="censored"` |

## Benchmarks

Reproducible pharma benchmark suite:

- Benchmark runners: `benchmarks/nextstat-public-benchmarks/suites/pharma/`
- Cases: 1-cpt oral, 2-cpt IV/oral, FOCE, SAEM with parameter recovery
- Run: `python suite.py --out-dir out/pharma --fit --seed 42`
- Report: `python report.py --suite out/pharma/pharma_suite.json`

## Reference

- Python API: `docs/references/python-api.md` (Pharmacometrics section)
- Rust API: `docs/references/rust-api.md` (ns-inference crate)
- Type stubs: `bindings/ns-py/python/nextstat/_core.pyi`
