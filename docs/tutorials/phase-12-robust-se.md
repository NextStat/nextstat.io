---
title: "Phase 12: Robust Standard Errors (HC / Cluster) — Tutorial"
status: draft
---

# Robust standard errors (HC / cluster)

NextStat provides dependency-light robust covariance estimators in `nextstat.robust`:

- Heteroskedasticity-consistent OLS (HC0–HC3)
- 1-way cluster-robust covariance (CR0 baseline + finite-sample correction)

These are asymptotic estimators. They do not fix misspecification.

## Quick start (OLS + HC1)

```python
import nextstat

x = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
y = [0.0, 1.0, 1.9, 3.2, 3.9, 5.1]

fit = nextstat.glm.linear.fit(x, y, include_intercept=True)

# HC1 covariance (White / MacKinnon)
cov, se = nextstat.robust.ols_hc_from_fit(fit, x, y, kind="HC1")

print("coef:", fit.coef)
print("HC1 se:", se)
```

Example output (approx):

```text
coef: [-0.019048, 1.014286]
HC1 se: [0.042248, 0.018537]
```

## Cluster-robust SE (1-way)

Use this when residuals are correlated within groups (e.g. repeated measures per entity).

```python
import nextstat

x = [[0.0], [1.0], [2.0], [3.0]]
y = [0.0, 1.0, 2.1, 3.2]
cluster = ["a", "a", "b", "b"]

fit = nextstat.glm.linear.fit(x, y, include_intercept=True)
cov, se = nextstat.robust.ols_cluster_from_fit(fit, x, y, cluster)

print("cluster se:", se)
```

Example output (approx):

```text
cluster se: [0.012247, 0.012247]
```

## Limitations

- **Small G problem**: cluster SE requires enough clusters; with very few clusters,
  inference is unreliable.
- **1-way only**: two-way clustering is not implemented yet.
- **No HAC/Newey-West**: serial correlation robust (HAC) is not implemented yet.
