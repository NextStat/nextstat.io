<!--
Reference
Created: 2026-02-08
Updated: 2026-02-24
Scope: nextstat.tools contract usage
-->

# Tool API (nextstat.tools)

This is the user-facing reference for NextStat's **agent/tool-calling** surface.

Determinism semantics: see `tests/python/_tolerances.py` (7-tier tolerance hierarchy).
Tool result schema: `docs/schemas/tools/nextstat_tool_result_v1.schema.json`

## What You Get

1. `nextstat.tools.get_toolkit()` returns OpenAI-compatible tool definitions (JSON Schema input).
2. `nextstat.tools.execute_tool(name, arguments)` executes a tool call and returns a stable envelope.

## Available Tools (31)

### HEP / HistFactory (9 tools)

| Tool | Description |
|------|-------------|
| `nextstat_fit` | MLE fit. Returns bestfit, uncertainties, NLL, covariance, convergence. |
| `nextstat_hypotest` | Asymptotic CLs hypothesis test (qtilde) at given mu. Returns CLs, CLs+b, CLb. |
| `nextstat_hypotest_toys` | Toy-based CLs hypothesis test (qtilde). Stochastic; requires `seed`. |
| `nextstat_upper_limit` | 95% CL upper limit on mu via CLs. Observed + optionally expected (+-1s/+-2s). |
| `nextstat_ranking` | Nuisance parameter ranking (systematic impact on mu). Sorted by impact. |
| `nextstat_discovery_asymptotic` | Asymptotic discovery at mu=0. Returns q0, z0, p0. **Not CLs.** |
| `nextstat_scan` | Profile likelihood scan over mu values. Returns (mu, q_mu, 2*dNLL) arrays. |
| `nextstat_workspace_audit` | Audit a pyhf workspace: channels, samples, modifiers, parameter count. |
| `nextstat_read_root_histogram` | Read a TH1 from a ROOT file (bin edges, content, sumw2, under/overflow). |

### Pharma / NLME (4 tools)

| Tool | Description |
|------|-------------|
| `nextstat_pharma_fit` | Population PK fitting (FOCE/FOCEI/SAEM). 1/2/3-cpt IV or oral. Returns theta, omega, eta, OFV. |
| `nextstat_pharma_vpc` | Visual Predictive Check. Simulates n_sim datasets, returns quantile bands. |
| `nextstat_trial_simulate` | Clinical trial simulation. Returns concentration profiles, AUC, Cmax, Tmax per subject. |
| `nextstat_bioequivalence` | Average bioequivalence (ABE) test, power analysis, or sample size calculation. |

### Safety / Reliability (2 tools)

| Tool | Description |
|------|-------------|
| `nextstat_fault_tree_mc` | Monte Carlo FTA with Birnbaum importance. Bernoulli, uncertain, Weibull modes. |
| `nextstat_fault_tree_ce_is` | Cross-Entropy Importance Sampling FTA for rare events (P < 1e-4). |

### GLM & Bayesian (2 tools)

| Tool | Description |
|------|-------------|
| `nextstat_glm_fit` | Fit GLM (linear, logistic, Poisson, negbin). |
| `nextstat_bayesian_sample` | NUTS/MAMS sampling on any model type. |

### Survival & Competing Risks (3 tools)

| Tool | Description |
|------|-------------|
| `nextstat_survival_fit` | Fit survival models (Cox PH, Weibull, AFT, Exponential). |
| `nextstat_kaplan_meier` | Kaplan-Meier curves with optional log-rank test. |
| `nextstat_competing_risks` | Cumulative incidence (CIF), Gray's test, Fine-Gray regression. |

### Dose-Response (1 tool)

| Tool | Description |
|------|-------------|
| `nextstat_dose_response` | Emax or Sigmoid Emax prediction / NLL evaluation. |

### Econometrics (4 tools)

| Tool | Description |
|------|-------------|
| `nextstat_panel_fe` | Panel data with entity fixed effects (within estimator). |
| `nextstat_did` | Difference-in-Differences via TWFE. |
| `nextstat_iv_2sls` | Instrumental Variables (2SLS) with diagnostics. |
| `nextstat_event_study` | Event study (dynamic DiD) with leads/lags for parallel trends. |

### Causal Inference (1 tool)

| Tool | Description |
|------|-------------|
| `nextstat_aipw` | Doubly-robust AIPW (ATE/ATT). |

### Time Series & Volatility (2 tools)

| Tool | Description |
|------|-------------|
| `nextstat_kalman` | Kalman filter / smooth / forecast. |
| `nextstat_garch_fit` | GARCH(1,1), EGARCH(1,1), or GJR-GARCH(1,1) volatility model. |

### Meta-Analysis & Actuarial (3 tools)

| Tool | Description |
|------|-------------|
| `nextstat_meta_analysis` | Fixed or random-effects meta-analysis. |
| `nextstat_churn_retention` | Churn retention curves. |
| `nextstat_chain_ladder` | Chain ladder (basic or Mack) reserving. |

### Envelope format

```json
{
  "schema_version": "nextstat.tool_result.v1",
  "ok": true,
  "result": {},
  "error": null,
  "meta": {
    "tool_name": "nextstat_fit",
    "nextstat_version": "0.9.7",
    "deterministic": true,
    "eval_mode": "parity",
    "threads_requested": 1
  }
}
```

Schema files:
- `docs/schemas/tools/nextstat_tool_result_v1.schema.json` (envelope)
- `docs/schemas/tools/nextstat_tool_result_strict_v1.schema.json` (envelope + strict result shapes)

## Execution Controls

Most tools accept an `execution` object:

```json
{
  "execution": {
    "deterministic": true,
    "eval_mode": "parity",
    "threads": 1
  }
}
```

Rules:
- `deterministic=true` forces `eval_mode="parity"` and requests `threads=1`.
- Thread control is **best-effort** (Rayon global pool is one-shot). Check `meta.threads_applied` when present.

## Semantics (Avoid Trust-Killers)

- `nextstat_hypotest` returns **CLs** (+ `clsb`, `clb`). It is not a p-value.
- Discovery-style outputs use `nextstat_discovery_asymptotic` which returns `{q0, z0, p0}`.

## Toy Tools and Seeds

Stochastic tools require a `seed` (default 42, but agents should set explicitly):
- `nextstat_hypotest_toys`: returns toy-based CLs as a `"raw"` payload plus the seed metadata.
- `nextstat_pharma_vpc`: simulates n_sim replicate datasets for VPC.
- `nextstat_trial_simulate`: simulates virtual PK trial with IIV.
- `nextstat_fault_tree_mc`: Monte Carlo scenarios for fault tree.
- `nextstat_fault_tree_ce_is`: CE-IS rare event estimation.

## Example: Tool Calling (OpenAI)

```python
import json
from openai import OpenAI
from nextstat.tools import get_toolkit, execute_tool

client = OpenAI()
tools = get_toolkit()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": "Fit the workspace and report the POI."}],
    tools=tools,
)

for call in resp.choices[0].message.tool_calls:
    out = execute_tool(call.function.name, json.loads(call.function.arguments))
    if not out["ok"]:
        raise RuntimeError(out["error"])
    print(out["result"])
```

## Example: Pharma Agent

```python
# Agent can now discover and call pharma tools
tools = get_toolkit()
# Agent sees nextstat_pharma_fit, nextstat_pharma_vpc, etc.
# and can run a complete PopPK analysis pipeline autonomously
```

## Example: ROOT Ingest

Use `nextstat_read_root_histogram` to fetch TH1 content for downstream analysis.

## Prompt Templates (Copy/Paste)

- Fit + report POI:
  - "Fit this workspace and return the POI value and error. Use deterministic mode."
- Upper limit:
  - "Compute the observed and expected 95% CL upper limit (CLs) for this workspace. Use deterministic mode."
- Discovery summary:
  - "Compute the asymptotic discovery significance (q0, z0, p0) at mu=0. Use deterministic mode."
- PopPK analysis:
  - "Fit this warfarin PK data with a 1-compartment oral model using FOCEI, then run a VPC with 500 simulations."
- Fault tree analysis:
  - "Analyze this fault tree with Monte Carlo (1M scenarios) and report the top event failure probability and component importance."
- Bioequivalence:
  - "Test bioequivalence for these AUC values. If BE passes, also compute sample size for 90% power at CV=0.25."
- ROOT quick look:
  - "Read histogram `hist1` from `tests/fixtures/simple_histos.root` and summarize its bin contents and under/overflow."

## Regression Harness

- Golden outputs: `tests/fixtures/tool_goldens/simple_workspace_deterministic.v1.json`
- Generator: `scripts/generate_tool_goldens.py`
- Smoke runner: `scripts/tool_call_smoke.py`

## Server Mode (nextstat-server)

If you run `nextstat-server`, you can also fetch tools and execute them over HTTP:
- Tool registry: `GET /v1/tools/schema`
- Tool execution: `POST /v1/tools/execute`

See: `docs/references/server-api.md`

Python usage:

```python
from nextstat.tools import get_toolkit, execute_tool

server_url = "http://127.0.0.1:3742"
tools = get_toolkit(transport="server", server_url=server_url)

out = execute_tool(
    "nextstat_pharma_fit",
    {"times": [...], "y": [...], "subject_idx": [...], ...},
    transport="server",
    server_url=server_url,
)
```

Notes:
- `server_url` can also be provided via env vars: `NEXTSTAT_SERVER_URL` or `NEXTSTAT_TOOLS_SERVER_URL`.
- `execute_tool(..., transport="server")` falls back to local execution by default when the server call fails.
  - To disable fallback: `fallback_to_local=False`.
