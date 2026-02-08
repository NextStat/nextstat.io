# Physics Assistant Demo (ROOT -> anomaly scan -> p-values + plots)

This demo is a reproducible "agent-shaped" workflow:

- Ingest a TREx-like ROOT export (histograms).
- Build simple HistFactory workspaces (JSON).
- Run an anomaly scan (sliding window) using **asymptotic discovery p-values** (`p0`, `Z0`).
- Run nominal analysis: discovery + upper limit + profile scan.
- Emit versioned JSON artifacts and a tool-call log.

## Run

Local tools mode:

```bash
cd /Users/andresvlc/WebDev/nextstat.io
PYTHONPATH=bindings/ns-py/python .venv/bin/python demos/physics_assistant/run_demo.py \
  --transport local \
  --out-dir demos/physics_assistant/out/local \
  --render
```

Server tools mode (ROOT ingest stays local; analysis calls use `POST /v1/tools/execute`):

```bash
cd /Users/andresvlc/WebDev/nextstat.io
export NEXTSTAT_SERVER_URL=http://127.0.0.1:3742
PYTHONPATH=bindings/ns-py/python .venv/bin/python demos/physics_assistant/run_demo.py \
  --transport server \
  --out-dir demos/physics_assistant/out/server
```

## Outputs

The main artifact is `demo_result.json` in the chosen `--out-dir`, plus:

- `tool_calls.json`: ordered tool call log (requests + envelopes).
- `nominal_workspace.json`, `best_anomaly_workspace.json`: workspaces used for the runs.
- `anomaly_scan_plot_data.json`: plot-ready arrays (JSON).
- `plots/*.png`: optional PNGs when `--render` is set.

