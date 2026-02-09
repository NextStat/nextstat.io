#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:
        raise SystemExit(f"matplotlib is required: {e}") from e


def _load(path: Path) -> Mapping[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit("invalid JSON (expected object)")
    if obj.get("schema_version") != "nextstat.figure.v1":
        raise SystemExit("unexpected schema_version (expected nextstat.figure.v1)")
    return obj


def _style(s: Mapping[str, Any]) -> dict[str, Any]:
    st = s.get("style") if isinstance(s.get("style"), dict) else {}
    color = st.get("color")
    alpha = st.get("alpha")
    out: dict[str, Any] = {}
    if isinstance(color, str):
        out["color"] = color
    if isinstance(alpha, (int, float)):
        out["alpha"] = float(alpha)
    ls = st.get("line_style")
    if ls == "dash":
        out["linestyle"] = "--"
    elif ls == "dot":
        out["linestyle"] = ":"
    elif ls == "dashdot":
        out["linestyle"] = "-."
    lw = st.get("line_width")
    if isinstance(lw, (int, float)):
        out["linewidth"] = float(lw)
    mk = st.get("marker")
    if isinstance(mk, str) and mk != "none":
        out["marker"] = mk
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input figure JSON")
    ap.add_argument("--out", dest="out", required=True, help="Output image path (.png or .svg)")
    args = ap.parse_args()

    _require_matplotlib()
    import matplotlib.pyplot as plt  # type: ignore

    fig_obj = _load(Path(args.inp))
    title = str(fig_obj.get("title", ""))
    axes = fig_obj.get("axes") or {}
    xlab = ((axes.get("x") or {}).get("label")) if isinstance(axes, dict) else None
    ylab = ((axes.get("y") or {}).get("label")) if isinstance(axes, dict) else None

    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=140)
    ax.set_title(title)
    if isinstance(xlab, str):
        ax.set_xlabel(xlab)
    if isinstance(ylab, str):
        ax.set_ylabel(ylab)

    for s in fig_obj.get("series") or []:
        if not isinstance(s, dict):
            continue
        kind = s.get("kind")
        name = str(s.get("name", ""))
        st = _style(s)
        if kind == "line":
            x = s.get("x") or []
            y = s.get("y") or []
            ax.plot(x, y, label=name, **st)
        elif kind == "band":
            x = s.get("x") or []
            lo = s.get("y_lo") or []
            hi = s.get("y_hi") or []
            st2 = dict(st)
            if "alpha" not in st2:
                st2["alpha"] = 0.2
            ax.fill_between(x, lo, hi, label=name, **st2)
        elif kind == "vline":
            x_at = s.get("x_at")
            if isinstance(x_at, (int, float)):
                ax.axvline(float(x_at), label=name, **st)
        elif kind == "hline":
            y_at = s.get("y_at")
            if isinstance(y_at, (int, float)):
                ax.axhline(float(y_at), label=name, **st)

    if any(isinstance(s, dict) and s.get("name") for s in (fig_obj.get("series") or [])):
        ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, metadata={})
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

