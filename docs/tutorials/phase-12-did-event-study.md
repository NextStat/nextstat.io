---
title: "Phase 12: Difference-in-Differences (DiD) + Event Study (TWFE) — Tutorial"
status: draft
---

# DiD and event study (TWFE baseline)

NextStat provides minimal DiD and event-study helpers using a **two-way fixed effects (TWFE)**
within transformation:

- DiD regressor: `treat_i * post_t`
- Event study: `treat_i * 1[rel_time == k]` (window of relative times, with a reference bin omitted)

## DiD quick start

```python
import nextstat

data = {
    "y": [10.0, 11.0, 17.0, 18.0, -3.0, -2.0, 2.0, 3.0],
    "entity": ["a", "a", "a", "a", "b", "b", "b", "b"],
    "t": [0, 1, 2, 3, 0, 1, 2, 3],
    "treated": [1, 1, 1, 1, 0, 0, 0, 0],
    "post": [0, 0, 1, 1, 0, 0, 1, 1],
}

did = nextstat.econometrics.did_twfe_from_formula(
    "y ~ 1",
    data,
    entity="entity",
    time="t",
    treat="treated",
    post="post",
    cluster="entity",
)

print("att:", did.att)
print("se:", did.att_se)
```

Example output (approx):

```text
att: 2.0
se: 0.0
```

## Event study quick start

```python
import nextstat

data = {
    "y": [10.0, 11.0, 17.0, 18.0, -3.0, -2.0, 2.0, 3.0],
    "entity": ["a", "a", "a", "a", "b", "b", "b", "b"],
    "t": [0, 1, 2, 3, 0, 1, 2, 3],
    "treated": [1, 1, 1, 1, 0, 0, 0, 0],
}

es = nextstat.econometrics.event_study_twfe_from_formula(
    "y ~ 1",
    data,
    entity="entity",
    time="t",
    treat="treated",
    event_time=2,
    window=(-1, 2),
    reference=-1,
    cluster="entity",
)

print(es.rel_times)
print(es.coef)
print(es.standard_errors)
```

Example output (approx):

```text
[0, 1]
[2.0, 2.0]
[0.0, 0.0]
```

## Identification and limitations

- **Parallel trends**: DiD requires parallel trends. Event-study pre-period coefficients
  should be near 0 as a baseline diagnostic.
- **TWFE limitations**: with staggered adoption or heterogeneous treatment effects,
  simple TWFE can be biased. This tutorial documents only a baseline.
- Only 1-way clustering is implemented.
