#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _load(path: Path) -> Mapping[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit("invalid JSON (expected object)")
    if obj.get("schema_version") != "nextstat.figure.v1":
        raise SystemExit("unexpected schema_version (expected nextstat.figure.v1)")
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    fig = _load(Path(args.inp))
    axes = fig.get("axes") or {}
    xlab = ((axes.get("x") or {}).get("label")) if isinstance(axes, dict) else None
    ylab = ((axes.get("y") or {}).get("label")) if isinstance(axes, dict) else None

    # Minimal Vega-Lite v5 spec: represent each series as its own layer.
    layers: list[dict[str, Any]] = []
    for s in fig.get("series") or []:
        if not isinstance(s, dict):
            continue
        kind = s.get("kind")
        name = str(s.get("name", ""))
        if kind == "line":
            x = s.get("x") or []
            y = s.get("y") or []
            data = [{"x": float(a), "y": float(b), "series": name} for a, b in zip(x, y)]
            layers.append(
                {
                    "data": {"values": data},
                    "mark": {"type": "line", "point": True},
                    "encoding": {
                        "x": {"field": "x", "type": "quantitative"},
                        "y": {"field": "y", "type": "quantitative"},
                        "color": {"field": "series", "type": "nominal"},
                    },
                }
            )
        elif kind == "band":
            x = s.get("x") or []
            lo = s.get("y_lo") or []
            hi = s.get("y_hi") or []
            data = [{"x": float(a), "y_lo": float(b), "y_hi": float(c), "series": name} for a, b, c in zip(x, lo, hi)]
            layers.append(
                {
                    "data": {"values": data},
                    "mark": {"type": "area", "opacity": 0.2},
                    "encoding": {
                        "x": {"field": "x", "type": "quantitative"},
                        "y": {"field": "y_lo", "type": "quantitative"},
                        "y2": {"field": "y_hi"},
                        "color": {"field": "series", "type": "nominal"},
                    },
                }
            )
        elif kind == "vline":
            x_at = s.get("x_at")
            if isinstance(x_at, (int, float)):
                layers.append(
                    {
                        "data": {"values": [{"x": float(x_at), "series": name}]},
                        "mark": {"type": "rule"},
                        "encoding": {"x": {"field": "x", "type": "quantitative"}, "color": {"field": "series", "type": "nominal"}},
                    }
                )
        elif kind == "hline":
            y_at = s.get("y_at")
            if isinstance(y_at, (int, float)):
                layers.append(
                    {
                        "data": {"values": [{"y": float(y_at), "series": name}]},
                        "mark": {"type": "rule"},
                        "encoding": {"y": {"field": "y", "type": "quantitative"}, "color": {"field": "series", "type": "nominal"}},
                    }
                )

    spec: dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": str(fig.get("title") or ""),
        "layer": layers,
    }
    if isinstance(xlab, str) and xlab:
        for layer in spec.get("layer") or []:
            if isinstance(layer, dict) and "encoding" in layer and "x" in layer["encoding"]:
                layer["encoding"]["x"]["title"] = xlab
    if isinstance(ylab, str) and ylab:
        for layer in spec.get("layer") or []:
            if isinstance(layer, dict) and "encoding" in layer and "y" in layer["encoding"]:
                layer["encoding"]["y"]["title"] = ylab

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

