# Physics Assistant Demo

Goal: prove the "NextStat as Oracle" story with a fully reproducible workflow:

- ingest `.root` histograms
- build a HistFactory workspace
- propose an anomaly scan
- compute discovery p-values (`p0`, `Z0`) and upper limits (CLs)
- export plot-friendly artifacts

Implementation:

- Runner: `demos/physics_assistant/run_demo.py`
- Output schema: `docs/schemas/demos/physics_assistant_demo_result_v1.schema.json`

## Runbook

Local mode (all tools in-process via Python bindings):

```bash
cd /path/to/nextstat.io
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python demos/physics_assistant/run_demo.py \
  --transport local \
  --out-dir demos/physics_assistant/out/local \
  --render
```

Docker mode (no local NextStat wheel required for the agent):

The agent container uses `uproot` for `.root` ingest and calls `nextstat-server` over HTTP
for all statistical tools. This avoids compiling the NextStat Python extension inside Docker.

```bash
cd /path/to/nextstat.io
docker compose -f demos/physics_assistant/env/docker/docker-compose.yml up --build
```

Outputs will be written under `demos/physics_assistant/env/docker/out/` on the host.

Server mode (statistical tools over HTTP; ROOT ingest stays local):

```bash
cd /path/to/nextstat.io
export NEXTSTAT_SERVER_URL=http://127.0.0.1:3742
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python demos/physics_assistant/run_demo.py \
  --transport server \
  --out-dir demos/physics_assistant/out/server
```

Why ingest is always local:

- `nextstat-server` does not expose `nextstat_read_root_histogram` by design (file IO is a security surface).
- The demo still exercises server mode fully by shipping only `workspace_json` to the server for inference calls.

## Artifacts

The runner writes:

- `demo_result.json`: main result bundle (validated by the demo schema)
- `tool_calls.json`: ordered tool-call log (requests + `nextstat.tool_result.v1` envelopes)
- `nominal_workspace.json`: nominal model workspace used for discovery/limits/scans
- `best_anomaly_workspace.json`: best window from the anomaly scan
- `anomaly_scan_plot_data.json`: plot-ready arrays (JSON)
- `anomaly_scan.figure.v1.json`: anomaly scan plot in `nextstat.figure.v1` schema (see `docs/schemas/plots/figure_v1.schema.json`)
- `nominal_profile_scan.figure.v1.json`: profile scan plot (if nominal scan produced points)
- `plots/*.png`: optional PNGs when `--render` is enabled

## Files

| File | Description |
|------|-------------|
| `demos/physics_assistant/run_demo.py` | Main runner (local or server transport). |
| `demos/physics_assistant/run_demo_server_only.py` | Server-only variant for Docker (uses `httpx`, no local wheel). |
| `demos/physics_assistant/env/docker/docker-compose.yml` | Two-container setup: `nextstat-server` + agent. |
| `demos/physics_assistant/env/docker/agent.Dockerfile` | Lightweight Python agent image. |
