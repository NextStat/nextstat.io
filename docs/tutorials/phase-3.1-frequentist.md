---
title: "Phase 3.1 Frequentist Tutorial (Profile Likelihood + CLs)"
status: stable
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

Optional (Python surface + Apex2 validation runners):

```bash
# Install Python deps used by the validation harness (pyhf, numpy, etc.)
./.venv/bin/pip install -e "bindings/ns-py[validation]"

# Build the `nextstat._core` extension into the active venv (recommended: release build)
./.venv/bin/maturin develop --release -m bindings/ns-py/Cargo.toml
```

Optional (plots in this doc):

```bash
./.venv/bin/pip install -e "bindings/ns-py[viz]"
```

## Validation Harness (Apex2)

The repo includes deterministic validation runners that produce JSON artifacts.

### 0) pyhf parity report (fast, deterministic)

Runs a small suite of HistFactory workspaces through pyhf and NextStat and checks:

- NLL parity (init + random points)
- expected_data parity (full + main-only)

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_pyhf_validation_report.py \
  --out tmp/apex2_pyhf_report.json
```

### 0.1) Master report (pyhf + regression golden + optional ROOT + optional bias/pulls + optional SBC)

This produces one aggregated JSON artifact, and embeds sub-reports.

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_master_report.py \
  --out tmp/apex2_master_report.json
```

In addition to the pyhf + golden regression checks, the master report also includes a
fast, deterministic NUTS/HMC diagnostics smoke test on a small non-HEP model
(GaussianMeanModel). This is intended to catch catastrophic sampling regressions without
requiring `NS_RUN_SLOW=1`.

Optional: embed the full NUTS/HMC quality report into the master report JSON.

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_master_report.py \
  --nuts-quality \
  --nuts-quality-warmup 200 \
  --nuts-quality-samples 200 \
  --out tmp/apex2_master_report.json
```

### 0.2) NUTS/HMC quality report (JSON, thresholds)

This runner produces a standalone JSON artifact focused on Posterior/HMC/NUTS quality:

- GaussianMeanModel (unbounded transform path)
- A small Composed GLM case (generic LogDensityModel sampling path)
- HistFactory simple fixture (bounded transform path)

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_nuts_quality_report.py \
  --out tmp/apex2_nuts_quality_report.json
```

Notes:

- HistFactory posteriors can require longer warmup/sampling to meet tight R-hat/ESS gates.
  By default the runner uses looser R-hat/ESS thresholds for the HistFactory case while
  keeping divergence/treedepth/E-BFMI gates.

Optional: include the slow bias/pulls regression suite (NextStat vs pyhf).

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_master_report.py \
  --bias-pulls \
  --bias-pulls-n-toys 200 \
  --bias-pulls-fixtures simple \
  --bias-pulls-params poi \
  --out tmp/apex2_master_report.json
```

Optional: include synthetic "model zoo" cases in bias/pulls (still slow; intended for nightly).

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_master_report.py \
  --bias-pulls \
  --bias-pulls-include-zoo \
  --bias-pulls-fixtures zoo_multichannel_3 \
  --bias-pulls-n-toys 200 \
  --out tmp/apex2_master_report.json
```

Optional: include the slow SBC (NUTS) report (Phase 5.4.2).

```bash
NS_RUN_SLOW=1 PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_master_report.py \
  --sbc \
  --sbc-n-runs 20 \
  --sbc-warmup 200 \
  --sbc-samples 200 \
  --out tmp/apex2_master_report.json
```

Notes:

- bias/pulls is intended for manual/nightly runs; it is not part of default CI.
- SBC (NUTS) is intended for manual/nightly runs; it requires `NS_RUN_SLOW=1`.
  - For the pytest suite, SBC tests are additionally gated behind `NS_RUN_SBC=1` and the `sbc` marker.
- ROOT parity is recorded as `skipped` unless ROOT prerequisites are present.

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

## Reproducible Run Bundles (Phase 9 reporting baseline)

For regulated workflows (pharma/social science packs) you often need immutable artifacts:

- exact model spec and input data snapshot
- hashes (data + spec) and command args
- output JSON captured alongside inputs

The CLI supports this via a global `--bundle` flag. The target directory must be empty (or not exist).

Example:

```bash
rm -rf tmp/run_bundle
cargo run -p ns-cli -- --bundle tmp/run_bundle fit --input tests/fixtures/simple_workspace.json --threads 1
```

Bundle layout:

- `meta.json`: tool version, args, and input/spec/data SHA-256 hashes (when applicable)
- `inputs/`: `input.json` plus `model_spec.json` + `data.json` (pyhf workspaces)
- `outputs/result.json`: the command output JSON
- `manifest.json`: SHA-256 + size for each file in the bundle

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
