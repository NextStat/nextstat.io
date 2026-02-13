# Physics Assistant Demo (ROOT -> anomaly scan -> p-values + plots)

This demo is a reproducible "agent-shaped" workflow:

- Ingest a TREx-like ROOT export (histograms).
- Build simple HistFactory workspaces (JSON).
- Run an anomaly scan (sliding window) using **asymptotic discovery p-values** (`p0`, `Z0`).
- Run nominal analysis: discovery + upper limit + profile scan.
- Emit versioned JSON artifacts (including `nextstat.figure.v1` plot schemas) and a tool-call log.

## Run

**Local tools mode** (all tools in-process via Python bindings):

```bash
# from repo root:
PYTHONPATH=bindings/ns-py/python python demos/physics_assistant/run_demo.py \
  --transport local \
  --out-dir demos/physics_assistant/out/local \
  --render
```

**Server tools mode** (ROOT ingest stays local; analysis calls use `POST /v1/tools/execute`):

```bash
# from repo root:
export NEXTSTAT_SERVER_URL=http://127.0.0.1:3742
PYTHONPATH=bindings/ns-py/python python demos/physics_assistant/run_demo.py \
  --transport server \
  --out-dir demos/physics_assistant/out/server
```

**Docker mode** (no local NextStat wheel required — agent calls `nextstat-server` over HTTP):

```bash
# from repo root:
docker compose -f demos/physics_assistant/env/docker/docker-compose.yml up --build
```

Outputs appear under `demos/physics_assistant/env/docker/out/` on the host.
The agent container (`agent.Dockerfile`) uses `uproot` + `httpx` for ROOT ingest,
then calls `nextstat-server` (`run_demo_server_only.py`) for all statistical tools.

## Outputs

The main artifact is `demo_result.json` in the chosen `--out-dir`, plus:

- `tool_calls.json`: ordered tool call log (requests + `nextstat.tool_result.v1` envelopes).
- `nominal_workspace.json`, `best_anomaly_workspace.json`: workspaces used for the runs.
- `anomaly_scan_plot_data.json`: plot-ready arrays (JSON).
- `anomaly_scan.figure.v1.json`: anomaly scan plot in `nextstat.figure.v1` schema.
- `nominal_profile_scan.figure.v1.json`: profile scan plot (if available).
- `plots/*.png`: optional PNGs when `--render` is set.

## Files

| File | Description |
|------|-------------|
| `run_demo.py` | Main demo runner (local or server transport). |
| `run_demo_server_only.py` | Server-only variant for Docker (uses `httpx`, no local NextStat wheel). |
| `env/docker/docker-compose.yml` | Two-container setup: `nextstat-server` + agent. |
| `env/docker/agent.Dockerfile` | Lightweight Python agent image. |
| `env/python/requirements.txt` | Pinned Python deps for the agent container. |

