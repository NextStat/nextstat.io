---
title: "Plot Artifacts"
status: stable
---

# Plot Artifacts (JSON)

NextStat aims to produce **machine-readable, deterministic** artifacts that can be rendered by:

- matplotlib (Python)
- Vega/Vega-Lite (web)
- custom pipelines (reports, dashboards, CI)

The standard plot artifact is a **Figure** JSON:

- Schema: `docs/schemas/plots/figure_v1.schema.json`
- `schema_version = "nextstat.figure.v1"`

## Data Model

A figure has:

- `title`, optional `subtitle`/`description`
- `axes.x` and `axes.y` labels (and optional `scale`/`unit`)
- `series[]`:
  - `kind="line"`: x/y arrays
  - `kind="band"`: x + y_lo/y_hi arrays
  - `kind="vline"` / `kind="hline"`: single reference line

## Reference Renderers

Matplotlib renderer:

```bash
python scripts/plot_artifacts/render_matplotlib.py --in figure.json --out figure.png
```

Vega-Lite adapter (emits a Vega-Lite spec JSON):

```bash
python scripts/plot_artifacts/to_vegalite.py --in figure.json --out figure.vegalite.json
```

