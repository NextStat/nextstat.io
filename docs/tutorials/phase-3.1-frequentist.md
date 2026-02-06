---
title: "Phase 3.1 Frequentist Tutorial (Profile Likelihood + CLs)"
status: draft
---

# Phase 3.1 Frequentist Tutorial (Profile Likelihood + CLs)

This tutorial shows the end-to-end workflow for frequentist limits:

1. MLE fit
2. Asymptotic CLs `hypotest` (qtilde)
3. Observed and expected upper limits
4. Profile likelihood scan q(mu)
5. Plot-friendly JSON artifacts for q(mu) and CLs curves (Brazil bands)

The examples use the built-in fixture workspace:

- `tests/fixtures/simple_workspace.json`

## Prerequisites

From the repo root:

```bash
cargo test
```

Optional (Python surface):

```bash
cd bindings/ns-py
maturin develop
```

## 1) Fit (MLE)

CLI:

```bash
cargo run -p ns-cli -- fit --input tests/fixtures/simple_workspace.json --threads 1
```

This returns best-fit parameters, uncertainties, and NLL.

## 2) Hypothesis Test (asymptotic CLs, qtilde)

CLI:

```bash
cargo run -p ns-cli -- hypotest --input tests/fixtures/simple_workspace.json --mu 1.0 --expected-set --threads 1
```

Output:

- `cls`: observed CLs at the tested `mu`
- `clsb`, `clb`: tail probabilities in the asymptotic calculator
- `expected_set.cls`: 5 values (Brazil band), ordered by `expected_set.nsigma_order`

### Expected-set ordering semantics

`expected_set.nsigma_order = [2, 1, 0, -1, -2]` is defined in `-muhat/sigma` space, matching pyhf.

Interpretation:

- `+2` (index 0) corresponds to `muhat` fluctuating down by 2 sigma (stronger expected limit)
- `0` (index 2) is the median expectation (Asimov)
- `-2` (index 4) corresponds to `muhat` fluctuating up by 2 sigma (weaker expected limit)

## 3) Upper Limits (CLs = alpha)

### Root-finding mode (bisection)

Observed only:

```bash
cargo run -p ns-cli -- upper-limit --input tests/fixtures/simple_workspace.json --alpha 0.05 --threads 1
```

Observed + expected limits:

```bash
cargo run -p ns-cli -- upper-limit --input tests/fixtures/simple_workspace.json --alpha 0.05 --expected --threads 1
```

### Scan mode (CLs curve + interpolation)

```bash
cargo run -p ns-cli -- upper-limit --input tests/fixtures/simple_workspace.json --alpha 0.05 --scan-start 0.0 --scan-stop 5.0 --scan-points 201 --threads 1
```

This returns:

- `obs_limit` (`mu_up`): interpolated observed upper limit
- `exp_limits`: interpolated expected upper limits (Brazil band ordering)

## 4) Profile Likelihood Scan (q(mu))

CLI:

```bash
cargo run -p ns-cli -- scan --input tests/fixtures/simple_workspace.json --start 0.0 --stop 2.0 --points 21 --threads 1
```

You get `mu_hat`, `nll_hat` and per-point `q_mu`.

## 5) Visualization Artifacts (Plot-Friendly JSON)

These commands emit JSON that is convenient for plotting: per-point arrays include both observed and expected values.

### 5.1 Profile curve artifact (q(mu))

```bash
cargo run -p ns-cli -- viz profile --input tests/fixtures/simple_workspace.json --start 0.0 --stop 2.0 --points 21 --threads 1
```

Schema:

- `mu_hat`, `nll_hat`
- `points[]: { mu, q_mu, nll_mu, converged, n_iter }`

### 5.2 CLs curve artifact (observed + Brazil bands)

```bash
cargo run -p ns-cli -- viz cls --input tests/fixtures/simple_workspace.json --alpha 0.05 --scan-start 0.0 --scan-stop 5.0 --scan-points 201 --threads 1
```

Schema:

- `alpha`, `nsigma_order`
- `points[]: { mu, cls, expected[5] }`
- `obs_limit`, `exp_limits[5]` (scan interpolation)

## Minimal plotting example (Python)

This example reads the CLs artifact JSON and plots the observed curve and the median expected curve.

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt

artifact = json.loads(Path("cls.json").read_text())

mu = [p["mu"] for p in artifact["points"]]
cls = [p["cls"] for p in artifact["points"]]
exp_median = [p["expected"][2] for p in artifact["points"]]

plt.plot(mu, cls, label="observed CLs")
plt.plot(mu, exp_median, label="expected (median)")
plt.axhline(artifact["alpha"], linestyle="--", color="black", label="alpha")
plt.xlabel("mu")
plt.ylabel("CLs")
plt.legend()
plt.tight_layout()
plt.show()
```

## Python Surface (nextstat.infer)

The Python API mirrors the CLI at a high level.

```python
import json
from pathlib import Path

import nextstat
from nextstat import infer

workspace = json.loads(Path("tests/fixtures/simple_workspace.json").read_text())
model = nextstat.from_pyhf(json.dumps(workspace))

# Observed asymptotic CLs (qtilde) at a tested mu
cls = infer.hypotest(1.0, model)
print("cls(mu=1):", cls)

# Optional: return CLs plus tail probabilities (CLs_b, CLb)
cls, (clsb, clb) = infer.hypotest(1.0, model, return_tail_probs=True)
print("cls/clsb/clb:", cls, clsb, clb)

# Observed upper limit (default alpha=0.05)
mu_up = infer.upper_limit(model, alpha=0.05)
print("mu_up:", mu_up)

# Profile likelihood scan q(mu)
mu_values = [0.0 + i * (2.0 / 20.0) for i in range(21)]
scan = infer.profile_scan(model, mu_values)
print("mu_hat:", scan["mu_hat"])
print("first point:", scan["points"][0])
```

## Python Visualization Artifacts (nextstat.viz)

The `nextstat.viz` module produces the same plot-friendly artifacts as the CLI.

```python
import json
from pathlib import Path

import nextstat
from nextstat import viz

workspace = json.loads(Path("tests/fixtures/simple_workspace.json").read_text())
model = nextstat.from_pyhf(json.dumps(workspace))

scan = [i * 0.05 for i in range(101)]  # 0..5
cls_artifact = viz.cls_curve(model, scan, alpha=0.05)
profile_artifact = viz.profile_curve(model, [i * 0.1 for i in range(21)])  # 0..2
```
