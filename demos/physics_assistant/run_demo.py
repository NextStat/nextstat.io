#!/usr/bin/env python3
"""
Physics Assistant demo (deterministic):

- Ingest a ROOT histogram export (TREx-like) and build simple HistFactory workspaces.
- Run anomaly scan windows via discovery p-values (p0/Z0).
- Run nominal signal discovery + upper limits + profile scans.
- Emit machine-readable artifacts + optional plots.

This is intentionally "agent-shaped": it uses the contract tools layer (local or server)
for all statistical operations, and logs every tool call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _repo_root() -> Path:
    # demos/physics_assistant/run_demo.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _ensure_repo_pythonpath() -> None:
    repo = _repo_root()
    p = repo / "bindings" / "ns-py" / "python"
    if p.exists():
        sys.path.insert(0, str(p))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _stable_json(obj: Any) -> str:
    # Stable for hashing and diffs.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def _pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


@dataclass(frozen=True)
class ExecCfg:
    deterministic: bool
    eval_mode: str
    threads: int


class ToolRunner:
    def __init__(self, *, transport: str, server_url: Optional[str], exec_cfg: ExecCfg):
        self.transport = transport
        self.server_url = server_url
        self.exec_cfg = exec_cfg
        self.calls: list[dict[str, Any]] = []
        self._idx = 0

        _ensure_repo_pythonpath()
        from nextstat.tools import execute_tool  # type: ignore

        self._execute_tool = execute_tool

    def call(self, name: str, arguments: dict[str, Any], *, transport: Optional[str] = None) -> dict[str, Any]:
        t = transport or self.transport
        args = dict(arguments)
        args["execution"] = {
            "deterministic": bool(self.exec_cfg.deterministic),
            "eval_mode": self.exec_cfg.eval_mode,
            "threads": int(self.exec_cfg.threads),
        }
        resp = self._execute_tool(
            name,
            args,
            transport=t,
            server_url=self.server_url,
            fallback_to_local=False,
        )
        self.calls.append(
            {
                "idx": self._idx,
                "transport": t,
                "name": name,
                "arguments": args,
                "response": resp,
            }
        )
        self._idx += 1
        return resp


def _filter_trex_base_keys(keys: list[str], *, channel: str) -> list[str]:
    prefix = f"hist{channel}_"
    out: list[str] = []
    for k in keys:
        name = k.split(";", 1)[0]
        if not name.startswith(prefix):
            continue
        # Keep only central (nominal) histograms.
        if "Low" in name or "High" in name:
            continue
        if "_staterror_" in name:
            continue
        if "shape_stat" in name:
            continue
        if "__" in name:
            continue
        out.append(name)
    out = sorted(set(out))
    return out


def _read_histogram(tool: ToolRunner, *, root_path: str, hist_path: str) -> dict[str, Any]:
    # ROOT ingest stays local even in server mode (server does not expose file IO).
    r = tool.call(
        "nextstat_read_root_histogram",
        {"root_path": root_path, "hist_path": hist_path},
        transport="local",
    )
    if not r.get("ok"):
        raise RuntimeError(f"read_root_histogram failed for {hist_path}: {r.get('error')}")
    result = r.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"invalid read_root_histogram result for {hist_path}")
    return result


def _as_float_list(x: Any) -> list[float]:
    if not isinstance(x, list):
        raise TypeError("expected list")
    return [float(v) for v in x]


def _as_int_counts(x: list[float]) -> list[float]:
    # pyhf/HistFactory expects counts; most ROOT exports are integer-valued,
    # but we keep type float for JSON compatibility while forcing integer values.
    return [float(int(round(v))) for v in x]


def _sum_vectors(vs: list[list[float]]) -> list[float]:
    if not vs:
        return []
    n = len(vs[0])
    out = [0.0] * n
    for v in vs:
        if len(v) != n:
            raise ValueError("vector length mismatch")
        for i, x in enumerate(v):
            out[i] += float(x)
    return out


def _build_workspace_json(
    *,
    channel_name: str,
    data: list[float],
    signal_name: str,
    signal: list[float],
    backgrounds: dict[str, list[float]],
) -> str:
    n = len(data)
    if len(signal) != n:
        raise ValueError("signal length mismatch")
    for k, v in backgrounds.items():
        if len(v) != n:
            raise ValueError(f"background {k} length mismatch")

    samples = []
    samples.append(
        {
            "name": signal_name,
            "data": [float(x) for x in signal],
            "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
        }
    )
    for name, vals in sorted(backgrounds.items()):
        samples.append({"name": name, "data": [float(x) for x in vals], "modifiers": []})

    ws = {
        "channels": [{"name": channel_name, "samples": samples}],
        "observations": [{"name": channel_name, "data": [float(x) for x in data]}],
        "measurements": [{"name": "measurement", "config": {"poi": "mu", "parameters": []}}],
        "version": "1.0.0",
    }
    return _stable_json(ws)


def _tophat_signal(n_bins: int, *, start: int, width: int) -> list[float]:
    if width <= 0:
        raise ValueError("width must be > 0")
    if start < 0 or start + width > n_bins:
        raise ValueError("window out of range")
    s = [0.0] * n_bins
    amp = 1.0 / float(width)
    for i in range(start, start + width):
        s[i] = amp
    return s


def _window_center(bin_edges: list[float], start: int, width: int) -> float:
    # Center in x-space using edges, falling back to bin index if edges are missing.
    if len(bin_edges) >= start + width + 1:
        lo = float(bin_edges[start])
        hi = float(bin_edges[start + width])
        return 0.5 * (lo + hi)
    return float(start) + 0.5 * float(width)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _maybe_render_plots(
    *,
    out_dir: Path,
    anomaly_rows: list[dict[str, Any]],
    nominal_scan_points: Optional[list[dict[str, Any]]],
    render: bool,
) -> dict[str, Optional[str]]:
    if not render:
        return {"anomaly_png": None, "nominal_scan_png": None}

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"--render requested but matplotlib is unavailable: {e}") from e

    # Anomaly scan plot: Z0 vs window center.
    xs = [float(r["center"]) for r in anomaly_rows]
    ys = [float(r["z0"]) for r in anomaly_rows]
    fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=140)
    ax.plot(xs, ys, marker="o", linewidth=1.5)
    ax.set_xlabel("window center")
    ax.set_ylabel("Z0")
    ax.set_title("Anomaly scan (asymptotic discovery significance)")
    ax.grid(True, alpha=0.25)
    anomaly_png = out_dir / "plots" / "anomaly_scan_z0.png"
    anomaly_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(anomaly_png, metadata={})
    plt.close(fig)

    nominal_png_path = None
    if nominal_scan_points:
        mu = [float(p.get("mu")) for p in nominal_scan_points]
        nll_mu = [float(p.get("nll_mu")) for p in nominal_scan_points]
        fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=140)
        ax.plot(mu, nll_mu, marker="o", linewidth=1.5)
        ax.set_xlabel("mu")
        ax.set_ylabel("profile NLL(mu)")
        ax.set_title("Nominal profile scan")
        ax.grid(True, alpha=0.25)
        nominal_png = out_dir / "plots" / "nominal_profile_scan.png"
        nominal_png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(nominal_png, metadata={})
        plt.close(fig)
        nominal_png_path = str(nominal_png)

    return {"anomaly_png": str(anomaly_png), "nominal_scan_png": nominal_png_path}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="demos/physics_assistant/out/demo", help="Output directory.")
    ap.add_argument("--transport", choices=["local", "server"], default="local")
    ap.add_argument("--server-url", default=os.environ.get("NEXTSTAT_SERVER_URL"))

    ap.add_argument("--root-path", default="tests/fixtures/trex_exports/tttt-prod/data.root")
    ap.add_argument("--channel", default="SR_AllBDT", help="TREx export channel name (used in hist prefix).")
    ap.add_argument("--data-hist", default=None, help="Override full histogram path for observed data.")
    ap.add_argument("--signal-hist", default=None, help="Override full histogram path for nominal signal.")

    ap.add_argument("--window-width-bins", type=int, default=3)
    ap.add_argument("--window-step-bins", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--render", action="store_true", help="Also render PNG plots (optional).")

    ap.add_argument("--deterministic", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = (_repo_root() / str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exec_cfg = ExecCfg(deterministic=bool(args.deterministic), eval_mode="parity", threads=1)
    tool = ToolRunner(transport=str(args.transport), server_url=args.server_url, exec_cfg=exec_cfg)

    # List base keys via uproot (read-only; ingestion still uses nextstat_read_root_histogram).
    try:
        import uproot  # type: ignore
    except Exception as e:
        raise RuntimeError(f"uproot is required for key discovery in this demo: {e}") from e

    root_path = str((_repo_root() / str(args.root_path)).resolve())
    root_sha256 = _sha256_file(Path(root_path))
    f = uproot.open(root_path)
    base_keys = _filter_trex_base_keys(list(f.keys()), channel=str(args.channel))
    prefix = f"hist{args.channel}_"
    data_hist = str(args.data_hist) if args.data_hist else prefix + "data"
    signal_hist = str(args.signal_hist) if args.signal_hist else prefix + "tttt"
    if data_hist not in base_keys:
        raise RuntimeError(f"data histogram not found: {data_hist}")
    if signal_hist not in base_keys:
        raise RuntimeError(f"signal histogram not found: {signal_hist}")

    bkg_hists = [k for k in base_keys if k not in (data_hist, signal_hist)]
    if not bkg_hists:
        raise RuntimeError("no background histograms discovered")

    # Ingest.
    data_h = _read_histogram(tool, root_path=root_path, hist_path=data_hist)
    sig_h = _read_histogram(tool, root_path=root_path, hist_path=signal_hist)
    bkg_h: dict[str, dict[str, Any]] = {}
    for k in bkg_hists:
        bkg_h[k] = _read_histogram(tool, root_path=root_path, hist_path=k)

    bin_edges = _as_float_list(data_h.get("bin_edges") or [])
    data_counts = _as_int_counts(_as_float_list(data_h["bin_content"]))
    sig_vals = _as_float_list(sig_h["bin_content"])
    bkg_vals = {k.replace(prefix, ""): _as_float_list(v["bin_content"]) for k, v in bkg_h.items()}

    n_bins = len(data_counts)
    if n_bins <= 0:
        raise RuntimeError("empty data histogram")

    # Nominal workspace: known signal (tttt) + split backgrounds.
    nominal_ws_json = _build_workspace_json(
        channel_name="sr",
        data=data_counts,
        signal_name="signal",
        signal=sig_vals,
        backgrounds=bkg_vals,
    )

    # Workspace audit (uses analysis transport).
    nominal_audit = tool.call("nextstat_workspace_audit", {"workspace_json": nominal_ws_json})
    nominal_disc = tool.call("nextstat_discovery_asymptotic", {"workspace_json": nominal_ws_json})
    nominal_ul = tool.call("nextstat_upper_limit", {"workspace_json": nominal_ws_json, "expected": True})

    mu_hat = None
    if nominal_disc.get("ok"):
        mu_hat = (nominal_disc.get("result") or {}).get("mu_hat")
    try:
        mu_hat_f = float(mu_hat) if mu_hat is not None else 0.0
    except Exception:
        mu_hat_f = 0.0

    scan_stop = max(2.0, mu_hat_f * 3.0 + 1.0)
    nominal_scan = tool.call(
        "nextstat_scan",
        {"workspace_json": nominal_ws_json, "start": 0.0, "stop": float(scan_stop), "points": 21},
    )
    nominal_scan_points = None
    if nominal_scan.get("ok") and isinstance(nominal_scan.get("result"), dict):
        nominal_scan_points = (nominal_scan["result"].get("points") or None)  # type: ignore

    # Anomaly scan: background_total (excluding nominal signal) + tophat window signal.
    bkg_total = _sum_vectors(list(bkg_vals.values()))
    if len(bkg_total) != n_bins:
        raise RuntimeError("background vector length mismatch")

    width = int(args.window_width_bins)
    step = int(args.window_step_bins)
    if width <= 0 or step <= 0:
        raise RuntimeError("window width/step must be > 0")

    anomaly_rows: list[dict[str, Any]] = []
    for start in range(0, n_bins - width + 1, step):
        s = _tophat_signal(n_bins, start=start, width=width)
        ws_json = _build_workspace_json(
            channel_name="sr",
            data=data_counts,
            signal_name="anomaly",
            signal=s,
            backgrounds={"bkg": bkg_total},
        )
        disc = tool.call("nextstat_discovery_asymptotic", {"workspace_json": ws_json})
        if not disc.get("ok"):
            raise RuntimeError(f"discovery failed for window start={start}: {disc.get('error')}")
        r = disc.get("result") or {}
        center = _window_center(bin_edges, start, width)
        anomaly_rows.append(
            {
                "start_bin": int(start),
                "width_bins": int(width),
                "center": float(center),
                "mu_hat": r.get("mu_hat"),
                "z0": float(r.get("z0")),
                "p0": float(r.get("p0")),
            }
        )

    anomaly_rows_sorted = sorted(anomaly_rows, key=lambda x: (-float(x["z0"]), int(x["start_bin"])))
    best = anomaly_rows_sorted[0]
    best_start = int(best["start_bin"])
    best_sig = _tophat_signal(n_bins, start=best_start, width=width)
    best_ws_json = _build_workspace_json(
        channel_name="sr",
        data=data_counts,
        signal_name="anomaly",
        signal=best_sig,
        backgrounds={"bkg": bkg_total},
    )
    best_ul = tool.call("nextstat_upper_limit", {"workspace_json": best_ws_json, "expected": True})
    best_scan = tool.call("nextstat_scan", {"workspace_json": best_ws_json, "start": 0.0, "stop": 10.0, "points": 21})

    # Artifacts.
    ingest_art = {
        "schema_version": "nextstat.physics_assistant_ingest.v1",
        "root_path": root_path,
        "root_sha256": root_sha256,
        "channel": str(args.channel),
        "hist_paths": {
            "data": data_hist,
            "signal_nominal": signal_hist,
            "backgrounds": [prefix + k for k in sorted(bkg_vals.keys())],
        },
        "n_bins": int(n_bins),
    }
    _write(out_dir / "ingest.json", _pretty_json(ingest_art))

    _write(out_dir / "nominal_workspace.json", nominal_ws_json)
    _write(out_dir / "best_anomaly_workspace.json", best_ws_json)

    anomaly_plot_data = {
        "schema_version": "nextstat.physics_assistant_anomaly_plot_data.v1",
        "x": [float(r["center"]) for r in anomaly_rows],
        "z0": [float(r["z0"]) for r in anomaly_rows],
        "p0": [float(r["p0"]) for r in anomaly_rows],
        "start_bin": [int(r["start_bin"]) for r in anomaly_rows],
        "width_bins": int(width),
        "step_bins": int(step),
    }
    _write(out_dir / "anomaly_scan_plot_data.json", _pretty_json(anomaly_plot_data))

    plot_paths = _maybe_render_plots(
        out_dir=out_dir,
        anomaly_rows=anomaly_rows,
        nominal_scan_points=nominal_scan_points if isinstance(nominal_scan_points, list) else None,
        render=bool(args.render),
    )

    tool_calls_path = out_dir / "tool_calls.json"
    _write(tool_calls_path, _pretty_json({"schema_version": "nextstat.tool_calls.v1", "calls": tool.calls}))

    demo_result = {
        "schema_version": "nextstat.physics_assistant_demo_result.v1",
        "deterministic": bool(exec_cfg.deterministic),
        "analysis_transport": str(args.transport),
        "server_url": str(args.server_url) if args.transport == "server" else None,
        "input": {
            "root_path": root_path,
            "root_sha256": root_sha256,
            "channel": str(args.channel),
            "data_hist": data_hist,
            "signal_hist": signal_hist,
            "background_hists": [prefix + k for k in sorted(bkg_vals.keys())],
            "n_bins": int(n_bins),
        },
        "artifacts": {
            "tool_calls_path": os.path.relpath(tool_calls_path, out_dir),
            "ingest_path": "ingest.json",
            "nominal_workspace_path": "nominal_workspace.json",
            "best_anomaly_workspace_path": "best_anomaly_workspace.json",
            "anomaly_scan_plot_data_path": "anomaly_scan_plot_data.json",
            "anomaly_scan_png_path": plot_paths["anomaly_png"] and os.path.relpath(plot_paths["anomaly_png"], out_dir),
            "nominal_scan_png_path": plot_paths["nominal_scan_png"] and os.path.relpath(plot_paths["nominal_scan_png"], out_dir),
        },
        "nominal": {
            "workspace_sha256": _sha256_text(nominal_ws_json),
            "workspace_audit": nominal_audit,
            "discovery": nominal_disc,
            "upper_limit": nominal_ul,
            "profile_scan": nominal_scan,
        },
        "anomaly_scan": {
            "window_width_bins": int(width),
            "window_step_bins": int(step),
            "top_k": int(args.top_k),
            "windows": anomaly_rows_sorted[: int(args.top_k)],
            "best_window": {
                "start_bin": int(best_start),
                "width_bins": int(width),
                "center": float(_window_center(bin_edges, best_start, width)),
                "upper_limit": best_ul,
                "profile_scan": best_scan,
            },
        },
    }

    demo_result_path = out_dir / "demo_result.json"
    _write(demo_result_path, _pretty_json(demo_result))

    # Validate demo_result schema + tool envelopes (best-effort).
    try:
        from jsonschema import validate  # type: ignore

        schema_demo = json.loads(
            (_repo_root() / "docs" / "schemas" / "demos" / "physics_assistant_demo_result_v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        validate(demo_result, schema_demo)

        schema_tool = json.loads(
            (_repo_root() / "docs" / "schemas" / "tools" / "nextstat_tool_result_strict_v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        # Validate every tool response envelope.
        for c in tool.calls:
            validate(c["response"], schema_tool)
    except Exception as e:
        # Keep demo usable even if validation env is incomplete.
        _write(out_dir / "validation_error.txt", str(e) + "\n")

    # Print the primary artifact path (useful in CI logs).
    print(str(demo_result_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

