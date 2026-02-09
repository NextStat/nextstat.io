# Phase 7.4: Posterior Predictive Checks (PPC) for Hierarchical GLMs

This is a focused tutorial for PPC workflows. It uses `nextstat.data.GlmSpec`
to keep the model's design matrix and group indices available for simulation.

## Logistic random intercept: replicate outcome rates

```python
import nextstat

X = [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]]
y = [0, 1, 0, 1, 0, 1]
group_idx = [0, 0, 1, 1, 0, 1]

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
ppc = nextstat.ppc.ppc_glm_from_sample(spec, raw, n_draws=100, seed=0)

obs_mean = ppc.observed["mean"]
rep_means = [r["mean"] for r in ppc.replicated]
print("observed mean(y):", obs_mean)
print("replicated mean(y) range:", min(rep_means), max(rep_means))
```

Notes:
- Sampling seeds are deterministic per chain: chain `i` uses `seed + i`.
- Initialization options are mutually exclusive (set at most one): `init_jitter`, `init_jitter_rel`, `init_overdispersed_rel`.

## Linear random intercept: replicate mean/variance

```python
import nextstat

X = [[0.0], [1.0], [2.0], [3.0]]
y = [0.2, 1.1, 2.0, 3.2]
group_idx = [0, 0, 1, 1]

spec = nextstat.data.GlmSpec.linear_regression(
    x=X,
    y=y,
    include_intercept=False,
    group_idx=group_idx,
    n_groups=2,
    coef_prior_mu=0.0,
    coef_prior_sigma=1.0,
)
m = spec.build()
raw = nextstat.sample(m, n_chains=2, n_warmup=300, n_samples=300, seed=2, init_jitter_rel=0.1)
ppc = nextstat.ppc.ppc_glm_from_sample(spec, raw, n_draws=100, seed=0)

print("observed stats:", ppc.observed)
print("replicated[0] stats:", ppc.replicated[0])
```
