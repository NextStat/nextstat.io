---
title: "Subscription / Churn Analysis — Pilot Runbook"
status: stable
---

# Subscription / Churn Analysis

NextStat's churn analysis vertical pack for subscription businesses
(SaaS, telco, streaming, marketplaces). The full pipeline is deterministic
and reproducible.

## Pipeline overview

| Step | Command | What it does |
|------|---------|-------------|
| 0 | `generate-data` | Synthetic SaaS cohort data (for testing) |
| 1 | `ingest` | Real-data ingestion from Parquet/CSV/JSON |
| 2 | `analyze` | KM retention curves + log-rank + Cox PH risk model + AIPW uplift |
| 3 | `cohort-matrix` | Life-table per cohort with period retention rates |
| 4 | `compare` | Pairwise log-rank + hazard ratio proxy + MCP correction |
| 5 | `uplift-survival` | RMST, IPW-weighted KM, ΔS(t) at horizons |
| 6 | `diagnostics` | Overlap, balance, censoring, trust gate |

All commands accept `--bundle <dir>` for reproducible run bundles.

## Data requirements

### Minimum fields

| Field | Type | Description |
|-------|------|-------------|
| `times` | `f64[]` | Time to event or censoring (months, days, etc.) |
| `events` | `bool[]` | `true` = churned, `false` = censored |
| `groups` | `i64[]` | Segment/cohort label (e.g. plan type) |

### Optional fields (for causal analysis)

| Field | Type | Description |
|-------|------|-------------|
| `treated` | `u8[]` | Treatment indicator: 1 = treated, 0 = control |
| `covariates` | `f64[][]` | Row-major (n × p) covariate matrix |
| `covariate_names` | `string[]` | Names for covariates |

### Ingestion from real data

For Parquet/CSV, create a mapping config JSON:

```json
{
  "time_column": "tenure_months",
  "event_column": "is_churned",
  "group_column": "plan_type",
  "treatment_column": "received_offer",
  "covariate_columns": ["usage_score", "support_tickets", "age"],
  "observation_end": 24.0
}
```

```bash
nextstat churn ingest -i customer_data.parquet --mapping mapping.json -o ingested.json
```

## Quick start: full pipeline (CLI)

```bash
# Step 1: Generate synthetic data (or use ingest for real data).
nextstat churn generate-data --n-customers 2000 --seed 42 -o data.json

# Step 2: Diagnostics — check data quality FIRST.
nextstat churn diagnostics -i data.json --out-dir diag/
# Look for: Trust gate PASSED, no critical warnings.

# Step 3: Cohort retention matrix.
nextstat churn cohort-matrix -i data.json --periods "1,3,6,12,24" --out-dir cohort/

# Step 4: Segment comparison (pairwise log-rank).
nextstat churn compare -i data.json --correction benjamini_hochberg --out-dir compare/

# Step 5: Survival-native causal uplift.
nextstat churn uplift-survival -i data.json --horizon 12 --eval-horizons "3,6,12,24" --out-dir uplift/

# Step 6: Full analysis (retention + risk model + AIPW uplift).
nextstat churn analyze -i data.json

# Bundle everything for reproducibility.
nextstat churn diagnostics -i data.json --bundle run_bundle/
```

## Quick start: full pipeline (Python)

```python
import nextstat

# 1. Generate synthetic SaaS churn data (2000 customers, seed=42).
ds = nextstat.churn_generate_data(n_customers=2000, seed=42)
print(f"n={ds['n']}, events={ds['n_events']}")

# 2. Retention analysis: KM curves by plan type + log-rank test.
ra = nextstat.churn_retention(ds["times"], ds["events"], ds["groups"])
print(f"Overall median: {ra['overall']['median']:.1f} months")
print(f"Log-rank: χ²={ra['log_rank']['chi_squared']:.2f}, p={ra['log_rank']['p_value']:.4f}")
for g in ra["by_group"]:
    print(f"  Plan {g['group']}: n={g['n']}, events={g['n_events']}, median={g['median']}")

# 3. Risk model: Cox PH hazard ratios.
names = ds["covariate_names"]
rm = nextstat.churn_risk_model(ds["times"], ds["events"], ds["covariates"], names)
for i, name in enumerate(rm["names"]):
    hr = rm["hazard_ratios"][i]
    lo, hi = rm["hr_ci_lower"][i], rm["hr_ci_upper"][i]
    print(f"  {name}: HR={hr:.3f} [{lo:.3f}, {hi:.3f}]")

# 4. Causal uplift: does the intervention reduce churn?
up = nextstat.churn_uplift(
    ds["times"], ds["events"], ds["treated"], ds["covariates"], horizon=12.0
)
print(f"ATE = {up['ate']:.4f} ± {up['se']:.4f}")
print(f"95% CI: [{up['ci_lower']:.4f}, {up['ci_upper']:.4f}]")
if up["gamma_critical"]:
    print(f"Rosenbaum Γ_critical = {up['gamma_critical']:.2f}")
```

## Data-generating process

The synthetic dataset uses an exponential survival model with proportional hazards:

```
h(t | x) = base_hazard × HR_plan × HR_usage^usage_score × HR_treatment^treated
```

Default parameters:
- **base_hazard_free** = 0.08 (8% monthly churn for free plan)
- **HR_basic** = 0.65 (35% reduction vs free)
- **HR_premium** = 0.40 (60% reduction vs free)
- **HR_usage** = 0.80 (20% reduction per 1σ usage increase)
- **HR_treatment** = 0.70 (30% reduction for intervention group)

Customers whose generated time exceeds `max_time` (24 months) are right-censored.

## Interpreting results

### Diagnostics (run first)

- **Trust gate**: PASSED = safe to proceed; FAILED = critical data issues.
- **Censoring by segment**: >80% censoring in a segment → survival estimates unreliable.
- **Covariate balance (SMD)**: |SMD| > 0.25 → treatment/control groups differ on that covariate; IPW may not fully correct.
- **Propensity overlap**: narrow range → weak confounders or near-deterministic treatment assignment.

### Cohort retention matrix

- **Period retention**: fraction surviving from cohort start to each period boundary.
- **Cumulative retention**: product of period retentions (Kaplan-Meier style).
- Compare across cohorts to detect time trends (improving/worsening retention).

### Segment comparison

- **Overall log-rank χ²**: tests whether all segments have the same survival distribution. p < 0.05 → reject null of equal survival.
- **Pairwise p_adjusted**: Bonferroni or Benjamini-Hochberg corrected p-values. Use BH (default) for more power; Bonferroni for stricter family-wise error control.
- **Hazard ratio proxy**: O/E ratio between groups. HR > 1 → group A churns faster than group B.
- **Median diff**: difference in median survival times.

### Retention analysis

- **KM curves**: non-parametric survival probabilities per plan group.
- **Log-rank p-value**: tests whether plan groups have significantly different
  churn distributions. p < 0.05 → reject null of equal survival.

### Risk model

- **Hazard ratios**: HR < 1 means the covariate is protective (reduces churn).
  - `plan_premium HR ≈ 0.40` → premium users churn at 40% the rate of free users.
  - `usage_score HR ≈ 0.80` → each 1σ usage increase reduces hazard by 20%.
- **CIs**: if the 95% CI excludes 1.0, the effect is statistically significant.

### Causal uplift (binary)

- **ATE** (Average Treatment Effect): negative ATE means the intervention reduces
  churn probability within the horizon period.
- **Rosenbaum Γ_critical**: the amount of hidden bias needed to overturn the result.
  Higher Γ → more robust finding.

### Survival-native uplift (RMST)

- **ΔRMST**: difference in Restricted Mean Survival Time between treated and control arms. Positive = treatment extends retention. Unit = same as time input (months).
- **ΔS(t) at horizons**: survival probability difference at specific time points. Positive = treated group retains more customers.
- **IPW**: Inverse Probability Weighting adjusts for confounders. If no covariates are provided, unweighted KM is used.
- **ESS**: Effective Sample Size after IPW. Low ESS → extreme weights → unstable estimates.

## Common failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Trust gate FAILED | >90% overall censoring | Check data extraction; ensure events are captured |
| All p-values = 1.0 | Identical segments | Verify group labels are meaningful |
| HR proxy = NaN | Zero events in one group | Merge small groups or increase sample size |
| ESS < 10 | Extreme propensity weights | Increase trim threshold (e.g. `--trim 0.05`) |
| Propensity range < 0.1 | Near-deterministic treatment | IPW cannot help; consider matching instead |
| SMD > 0.5 | Severe imbalance | Consider stratified analysis or matching |

## Run bundles and reproducibility

Every command supports `--bundle <dir>` which creates:

```
bundle_dir/
├── meta.json           # tool version, git rev, timestamp, command, args
├── manifest.json       # file sizes + SHA-256 per file
├── inputs/
│   └── input.json      # exact input copy + SHA-256 fingerprint
└── outputs/
    └── result.json     # canonicalized JSON output
```

`meta.json` includes:
- **tool_version**: NextStat version (e.g. `0.9.0`)
- **git_rev**: short git commit hash
- **reproducibility.deterministic**: whether output is deterministic
- **reproducibility.bundle_schema_version**: `2`
- **reproducibility.platform**: e.g. `macos/aarch64`

To reproduce a run: use the same NextStat version and pass `inputs/input.json` with the same command and args from `meta.json`.

## Output artifacts

All `--out-dir` commands produce:

| Command | JSON | CSVs |
|---------|------|------|
| `cohort-matrix` | `cohort_matrix.json` | `cohort_matrix.csv` |
| `compare` | `segment_comparison.json` | `segment_summary.csv`, `pairwise_comparisons.csv` |
| `uplift-survival` | `survival_uplift.json` | `uplift_arms.csv`, `uplift_delta_survival.csv` |
| `diagnostics` | `diagnostics.json` | `censoring_by_segment.csv`, `covariate_balance.csv` |

## Notes and limitations

- The synthetic generator uses a simple exponential model. Real data will have
  more complex hazard functions (time-varying covariates, competing risks).
- The propensity model in `churn_uplift` is a logistic regression. For high-dimensional
  covariates, consider gradient-boosted propensity scores externally.
- All results are deterministic given the same seed and inputs.
- RMST is sensitive to the choice of horizon τ. Choose τ within the observed follow-up range.
- Benjamini-Hochberg controls the False Discovery Rate, not the family-wise error rate. Use Bonferroni for stricter control.
