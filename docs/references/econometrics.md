# Econometrics & Causal Inference (Phase 12)

NextStat provides a focused econometrics toolkit covering the most common
social-science/policy-evaluation workflows with defensible defaults.

This page focuses on the **Python** API.

## Panel Fixed Effects

```python
import nextstat

# x: (n x p) design matrix (list of rows)
# y: outcome (length n)
# entity: group identifiers (length n)
fit = nextstat.econometrics.panel_fe_fit(
    x,
    y,
    entity=entity,
    cluster="entity",  # "entity" | "time" | "none"
)

print("beta:", fit.coef)
print("se:", fit.standard_errors)
print("n_entities:", fit.n_entities)
```

**Method:** Entity-demeaned ("within") OLS. Absorbs all entity-level fixed
effects by subtracting entity means from X and y before running OLS.

**Cluster-robust SE:** Liang–Zeger HC0 sandwich estimator with small-sample
correction `G/(G-1) × (N-1)/(N-K)`.

## Difference-in-Differences (DiD)

### Canonical 2×2 DiD

```python
import nextstat

# treat: treatment indicator (0/1)
# post: post-period indicator (0/1)
did = nextstat.econometrics.did_twfe_fit(
    x=None,  # or controls matrix (n x k)
    y=y,
    treat=treat,
    post=post,
    entity=entity,
    time=time,
    cluster="entity",
)

print("ATT:", did.att)
print("SE:", did.att_se)
```

Estimates `y = α + β₁·D + β₂·P + δ·(D×P) + ε` where δ is the ATT.

### Event Study

```python
import nextstat

es = nextstat.econometrics.event_study_twfe_fit(
    y,
    treat=treat,
    time=time,
    event_time=event_time,  # scalar or per-row event time
    entity=entity,
    window=(-3, 3),
    reference=-1,
    x=None,  # or controls matrix (n x k)
    cluster="entity",
)

for k, coef, se in zip(es.rel_times, es.coef, es.standard_errors):
    print(f"k={k:+}: delta={coef:.3f} ± {se:.3f}")
```

**Method:** Two-way FE (entity + time demeaning) with lead/lag indicators.
The reference period (typically k = −1) is omitted for identification.

## Instrumental Variables / 2SLS

```python
import nextstat

fit = nextstat.econometrics.iv_2sls_fit(
    y,
    endog=endog,              # (n x p_endog)
    instruments=instruments,  # (n x q) excluded instruments
    exog=exog,                # optional (n x p_exog)
    endog_names=["d"],
    instrument_names=["z1", "z2"],
    exog_names=["x1", "x2"],
    cov="hc1",               # "homoskedastic" | "hc1" | "cluster"
)

print("beta:", fit.coef)
print("se:", fit.standard_errors)
print("excluded instruments kept:", fit.diagnostics.excluded_instruments)
print("first-stage F:", fit.diagnostics.first_stage_f)
```

**Weak-instrument diagnostics:** Reports first-stage F-statistic, partial R²,
and Stock–Yogo 10% maximal IV size test (F > 16.38 for single instrument /
single endogenous regressor).

## AIPW (Doubly Robust)

```python
import nextstat

fit = nextstat.causal.aipw.aipw_fit(
    x,
    y,
    t,
    estimand="ate",
    trim_eps=1e-6,
)

print("ATE:", fit.estimate)
print("SE:", fit.standard_error)
print("warnings:", fit.propensity_diagnostics.warnings)
```

The AIPW estimator is **doubly robust**: consistent if either the propensity
score model or the outcome model is correctly specified.

**Inputs / hooks:**
- `x`, `y`, `t` — predictors, outcome, and binary treatment indicator.
- `trim_eps` — propensity score clipping epsilon (probabilities are clipped to `[trim_eps, 1-trim_eps]`).
- `propensity_scores` — optional precomputed `P(D=1|X)` to bypass propensity fitting.
- `mu0`, `mu1` — optional precomputed `E[Y(0)|X]`, `E[Y(1)|X]` to bypass outcome regressions.

## Rosenbaum Sensitivity Analysis

`rosenbaum_bounds` is exposed from the native extension as `nextstat._core.rosenbaum_bounds`.

```python
import nextstat

res = nextstat._core.rosenbaum_bounds(
    y_treated,
    y_control,
    [1.0, 1.5, 2.0, 3.0],
)

gamma_c = res.get("gamma_critical")
if gamma_c is None:
    print("Robust to all tested gamma values")
else:
    print(f"Result sensitive at gamma = {gamma_c}")
```

Tests how robust matched-pair results are to a hypothetical unobserved
confounder that changes treatment odds by a factor of Γ. Based on the
Wilcoxon signed-rank test with normal approximation.

## Assumptions & Limitations

| Estimator | Key assumption |
|-----------|---------------|
| Panel FE | Strict exogeneity (no feedback from y to future x) |
| DiD | Parallel trends (absent treatment, trends would be equal) |
| IV/2SLS | Exclusion restriction + instrument relevance |
| AIPW | Unconfoundedness (selection on observables) |

- **No random effects (RE):** Only fixed effects are implemented. Use `lmm`
  module for random-effects models.
- **No heterogeneous treatment effects:** DiD estimates ATT, not CATE. For
  heterogeneous effects with staggered adoption, consider Callaway–Sant'Anna
  or Sun–Abraham estimators (future work).
- **Stock–Yogo critical values** are approximate for configurations beyond
  1 endogenous / 1–2 instruments.
