#!/usr/bin/env python3
"""
Physics Assistant demo runner that does NOT depend on the NextStat Python extension.

It talks to `nextstat-server` over HTTP (`POST /v1/tools/execute`) for all statistical work,
and uses `uproot` to ingest `.root` histograms (pure Python).

This makes it suitable for a minimal "demo agent" Docker image without needing to compile
`nextstat` wheels inside the container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def _pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@dataclass(frozen=True)
class ExecCfg:
    deterministic: bool
    eval_mode: str
    threads: int


class ServerTools:
    def __init__(self, *, server_url: str, exec_cfg: ExecCfg):
        try:
            import httpx  # type: ignore
        except Exception as e:
            raise RuntimeError(f"httpx is required: {e}") from e

        self._httpx = httpx
        self.server_url = server_url.rstrip("/")
        self.exec_cfg = exec_cfg
        self.calls: list[dict[str, Any]] = []
        self._idx = 0

    def _post(self, path: str, payload: dict[str, Any], *, timeout_s: float = 30.0) -> dict[str, Any]:
        r = self._httpx.post(self.server_url + path, json=payload, timeout=timeout_s)
        r.raise_for_status()
        obj = r.json()
        if not isinstance(obj, dict):
            raise RuntimeError("invalid JSON response (expected object)")
        return obj

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        args = dict(arguments)
        args["execution"] = {
            "deterministic": bool(self.exec_cfg.deterministic),
            "eval_mode": self.exec_cfg.eval_mode,
            "threads": int(self.exec_cfg.threads),
        }
        resp = self._post("/v1/tools/execute", {"name": name, "arguments": args})
        self.calls.append({"idx": self._idx, "transport": "server", "name": name, "arguments": args, "response": resp})
        self._idx += 1
        return resp

    def wait_ready(self, *, timeout_s: float = 30.0) -> None:
        t0 = time.time()
        last_err: Optional[str] = None
        while True:
            try:
                obj = self._httpx.get(self.server_url + "/v1/tools/schema", timeout=5.0).json()
                if isinstance(obj, dict) and isinstance(obj.get("tools"), list):
                    return
                last_err = "schema response missing tools[]"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            if time.time() - t0 > timeout_s:
                raise RuntimeError(f"server not ready after {timeout_s}s: {last_err}")
            time.sleep(0.25)


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


def _as_int_counts(x: list[float]) -> list[float]:
    return [float(int(round(v))) for v in x]


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
    if len(bin_edges) >= start + width + 1:
        lo = float(bin_edges[start])
        hi = float(bin_edges[start + width])
        return 0.5 * (lo + hi)
    return float(start) + 0.5 * float(width)


def _read_hist_uproot(root_path: str, hist_path: str) -> dict[str, Any]:
    try:
        import uproot  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError(f"uproot/numpy are required: {e}") from e

    f = uproot.open(root_path)
    obj = f[hist_path]
    # uproot hist: edges + values (+ variances if present)
    edges = obj.axis().edges()
    values = obj.values()
    variances = None
    try:
        variances = obj.variances()
    except Exception:
        variances = None

    edges_l = [float(x) for x in edges.tolist()]
    values_l = [float(x) for x in values.tolist()]
    if variances is None:
        sumw2_l = None
    else:
        sumw2_l = [float(x) for x in np.asarray(variances).tolist()]

    return {
        "name": str(hist_path),
        "title": "",
        "bin_edges": edges_l,
        "bin_content": values_l,
        "sumw2": sumw2_l,
        "underflow": 0.0,
        "overflow": 0.0,
        "underflow_sumw2": None if sumw2_l is None else 0.0,
        "overflow_sumw2": None if sumw2_l is None else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="demos/physics_assistant/out/docker_agent", help="Output directory.")
    ap.add_argument("--server-url", default=os.environ.get("NEXTSTAT_SERVER_URL", "http://server:3742"))

    ap.add_argument("--root-path", default="tests/fixtures/trex_exports/tttt-prod/data.root")
    ap.add_argument("--channel", default="SR_AllBDT", help="TREx export channel name (used in hist prefix).")
    ap.add_argument("--window-width-bins", type=int, default=3)
    ap.add_argument("--window-step-bins", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    out_dir = (_repo_root() / str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    root_path = str((_repo_root() / str(args.root_path)).resolve())
    root_sha256 = _sha256_file(Path(root_path))

    exec_cfg = ExecCfg(deterministic=True, eval_mode="parity", threads=1)
    tools = ServerTools(server_url=str(args.server_url), exec_cfg=exec_cfg)
    tools.wait_ready(timeout_s=60.0)

    # The TREx export fixture uses a simple histogram naming convention.
    prefix = f"hist{args.channel}_"
    data_hist = prefix + "data"
    signal_hist = prefix + "tttt"

    # Read nominal model ingredients.
    data_h = _read_hist_uproot(root_path, data_hist)
    sig_h = _read_hist_uproot(root_path, signal_hist)

    # Background discovery via uproot keys (server does not do file IO).
    import uproot  # type: ignore

    f = uproot.open(root_path)
    keys = [k.split(";", 1)[0] for k in f.keys()]
    bkg_keys = []
    for k in keys:
        if not k.startswith(prefix):
            continue
        if k in (data_hist, signal_hist):
            continue
        if "Low" in k or "High" in k:
            continue
        if "_staterror_" in k or "shape_stat" in k or "__" in k:
            continue
        bkg_keys.append(k)
    bkg_keys = sorted(set(bkg_keys))
    if not bkg_keys:
        raise RuntimeError("no background histograms discovered")

    bkg_vals: dict[str, list[float]] = {}
    for k in bkg_keys:
        h = _read_hist_uproot(root_path, k)
        bkg_vals[k.replace(prefix, "")] = [float(x) for x in h["bin_content"]]

    bin_edges = [float(x) for x in (data_h.get("bin_edges") or [])]
    data_counts = _as_int_counts([float(x) for x in data_h["bin_content"]])
    sig_vals = [float(x) for x in sig_h["bin_content"]]

    n_bins = len(data_counts)
    bkg_total = _sum_vectors(list(bkg_vals.values()))

    nominal_ws_json = _build_workspace_json(
        channel_name="sr",
        data=data_counts,
        signal_name="signal",
        signal=sig_vals,
        backgrounds=bkg_vals,
    )
    _write(out_dir / "nominal_workspace.json", nominal_ws_json)

    nominal_audit = tools.call("nextstat_workspace_audit", {"workspace_json": nominal_ws_json})
    nominal_disc = tools.call("nextstat_discovery_asymptotic", {"workspace_json": nominal_ws_json})
    nominal_ul = tools.call("nextstat_upper_limit", {"workspace_json": nominal_ws_json, "expected": True})
    nominal_scan = tools.call("nextstat_scan", {"workspace_json": nominal_ws_json, "start": 0.0, "stop": 3.0, "points": 21})

    width = int(args.window_width_bins)
    step = int(args.window_step_bins)
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
        disc = tools.call("nextstat_discovery_asymptotic", {"workspace_json": ws_json})
        if not disc.get("ok"):
            raise RuntimeError(f"discovery failed for window start={start}: {disc.get('error')}")
        r = disc.get("result") or {}
        anomaly_rows.append(
            {
                "start_bin": int(start),
                "width_bins": int(width),
                "center": float(_window_center(bin_edges, start, width)),
                "mu_hat": r.get("mu_hat"),
                "z0": float(r.get("z0")),
                "p0": float(r.get("p0")),
            }
        )

    anomaly_rows_sorted = sorted(anomaly_rows, key=lambda x: (-float(x["z0"]), int(x["start_bin"])))
    best = anomaly_rows_sorted[0]
    best_start = int(best["start_bin"])
    best_ws_json = _build_workspace_json(
        channel_name="sr",
        data=data_counts,
        signal_name="anomaly",
        signal=_tophat_signal(n_bins, start=best_start, width=width),
        backgrounds={"bkg": bkg_total},
    )
    _write(out_dir / "best_anomaly_workspace.json", best_ws_json)

    best_ul = tools.call("nextstat_upper_limit", {"workspace_json": best_ws_json, "expected": True})
    best_scan = tools.call("nextstat_scan", {"workspace_json": best_ws_json, "start": 0.0, "stop": 10.0, "points": 21})

    tool_calls_path = out_dir / "tool_calls.json"
    _write(tool_calls_path, _pretty_json({"schema_version": "nextstat.tool_calls.v1", "calls": tools.calls}))

    demo_result = {
        "schema_version": "nextstat.physics_assistant_demo_result.v1",
        "deterministic": True,
        "analysis_transport": "server",
        "server_url": str(args.server_url),
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
            "anomaly_scan_png_path": None,
            "nominal_scan_png_path": None,
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

    print(str(demo_result_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

