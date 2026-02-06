# Phase 7: Hierarchical / Multilevel (Random Intercepts)

This tutorial shows the minimal end-to-end flow:

1. Build a hierarchical GLM with a group-indexed random intercept
2. Run MAP fit (`nextstat.fit`)
3. Run NUTS (`nextstat.sample` or `nextstat.bayes.sample`)

## Logistic regression with random intercept

```python
import nextstat

# Toy dataset: 2 groups, 1 feature, binary outcome.
X = [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]]
y = [0, 1, 0, 1, 0, 1]
group_idx = [0, 0, 1, 1, 0, 1]

m = nextstat.hier.logistic_random_intercept(
    x=X,
    y=y,
    group_idx=group_idx,
    n_groups=2,
    coef_prior_mu=0.0,
    coef_prior_sigma=10.0,
)

fit = nextstat.fit(m)
print("nll:", fit.nll)
print("bestfit dim:", len(fit.bestfit))

raw = nextstat.sample(
    m,
    n_chains=4,
    n_warmup=500,
    n_samples=1000,
    seed=42,
    init_jitter_rel=0.10,
)
print(raw["diagnostics"]["quality"])
```

Notes:
- `group_idx[i]` selects the group for observation `i`.
- The random intercept is a partially-pooled Normal effect (group intercepts share hyperparameters).

## Posterior Predictive Checks (PPC)

PPC answers: "Does the fitted model generate data that looks like what we observed?"

NextStat provides lightweight helpers in `nextstat.ppc` that work on the raw dict
returned by `nextstat.sample(...)` (no ArviZ required).

```python
import nextstat

spec = nextstat.data.GlmSpec.logistic_regression(
    x=X,
    y=y,
    include_intercept=False,
    group_idx=group_idx,
    n_groups=2,
    coef_prior_mu=0.0,
    coef_prior_sigma=1.0,
)
m = spec.build()

raw = nextstat.sample(m, n_chains=2, n_warmup=300, n_samples=300, seed=1, init_jitter_rel=0.1)
ppc = nextstat.ppc.ppc_glm_from_sample(spec, raw, n_draws=50, seed=0)
print("observed:", ppc.observed)
print("replicated[0]:", ppc.replicated[0])
```

## Linear regression with random intercept

```python
import nextstat

X = [[0.0], [1.0], [2.0], [3.0]]
y = [0.2, 1.1, 2.0, 3.2]
group_idx = [0, 0, 1, 1]

m = nextstat.hier.linear_random_intercept(
    x=X,
    y=y,
    group_idx=group_idx,
    n_groups=2,
)

fit = nextstat.fit(m)
print("nll:", fit.nll)
```
