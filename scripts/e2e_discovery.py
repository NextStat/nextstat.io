#!/usr/bin/env python3
"""
End-to-end discovery runner (tool-shaped).

Runs a canonical sequence against a workspace JSON:
  - workspace audit
  - MLE fit
  - asymptotic discovery (q0/z0/p0)
  - asymptotic CLs hypotest (qtilde) at a chosen mu
  - 95% CL upper limit (observed + optional expected bands)
  - profile likelihood scan
  - nuisance ranking

Writes:
  - one JSON file per tool call
  - calls.json (full call log)
  - summary.json + summary.md
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def _pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _ensure_parent_dir(path: Path) -> None:
    if path.parent and not str(path.parent) == "":
        path.parent.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, s: str) -> None:
    _ensure_parent_dir(path)
    path.write_text(s, encoding="utf-8")


def _try_import_nextstat() -> None:
    try:
        import nextstat  # noqa: F401

        return
    except Exception:
        # Fall back to repo source tree layout.
        p = _repo_root() / "bindings" / "ns-py" / "python"
        if p.exists():
            import sys

            sys.path.insert(0, str(p))
        import nextstat  # noqa: F401


@dataclass(frozen=True)
class ExecCfg:
    deterministic: bool
    eval_mode: str
    threads: int


def _extract_nested(d: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="Path to a pyhf/HS3 workspace JSON file.")
    ap.add_argument("--out-dir", default="tmp/e2e_discovery", help="Output directory.")

    ap.add_argument(
        "--transport",
        default="local",
        choices=["local", "server"],
        help="Tool execution transport: local (Python bindings) or server (nextstat-server HTTP).",
    )
    ap.add_argument(
        "--server-url",
        default="",
        help="Base URL for --transport=server (e.g. http://127.0.0.1:3742). "
        "If omitted, NEXTSTAT_SERVER_URL / NEXTSTAT_TOOLS_SERVER_URL may be used.",
    )
    ap.add_argument("--timeout-s", type=float, default=30.0, help="HTTP timeout in seconds (server mode).")
    ap.add_argument("--no-fallback", action="store_true", help="Disable fallback to local execution on failures.")

    ap.add_argument("--mu", type=float, default=1.0, help="Signal strength mu to test in `nextstat_hypotest`.")
    ap.add_argument("--expected", action="store_true", help="Include expected bands for `nextstat_upper_limit`.")
    ap.add_argument("--scan-start", type=float, default=0.0, help="Profile scan start.")
    ap.add_argument("--scan-stop", type=float, default=2.0, help="Profile scan stop.")
    ap.add_argument("--scan-points", type=int, default=21, help="Profile scan points.")
    ap.add_argument("--ranking-top-n", type=int, default=10, help="Top N systematics for ranking.")

    ap.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer deterministic parity settings (use --no-deterministic to disable).",
    )
    ap.add_argument("--eval-mode", default="parity", choices=["parity", "fast"], help="Evaluation mode.")
    ap.add_argument("--threads", type=int, default=1, help="Thread count (0 uses library default).")
    cli = ap.parse_args()

    _try_import_nextstat()
    from nextstat.tools import execute_tool  # type: ignore

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ws_path = Path(cli.workspace)
    ws_text = ws_path.read_text(encoding="utf-8")

    exec_cfg = ExecCfg(
        deterministic=bool(cli.deterministic),
        eval_mode=str(cli.eval_mode),
        threads=int(cli.threads),
    )

    calls: list[dict[str, Any]] = []

    def run_tool(name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
        args = dict(tool_args)
        args["execution"] = {
            "deterministic": bool(exec_cfg.deterministic),
            "eval_mode": exec_cfg.eval_mode,
            "threads": int(exec_cfg.threads),
        }
        t0 = time.perf_counter()
        resp = execute_tool(
            name,
            args,
            transport=str(cli.transport),
            server_url=(cli.server_url or None),
            timeout_s=float(cli.timeout_s),
            fallback_to_local=not bool(cli.no_fallback),
        )
        dt = time.perf_counter() - t0
        calls.append(
            {
                "name": name,
                "arguments": args,
                "response": resp,
                "wall_time_s": float(dt),
            }
        )
        _write_text(out_dir / f"{name}.json", _pretty_json(resp))
        return resp

    # Canonical tool chain.
    chain: list[tuple[str, dict[str, Any]]] = [
        ("nextstat_workspace_audit", {"workspace_json": ws_text}),
        ("nextstat_fit", {"workspace_json": ws_text}),
        ("nextstat_discovery_asymptotic", {"workspace_json": ws_text}),
        ("nextstat_hypotest", {"workspace_json": ws_text, "mu": float(cli.mu)}),
        ("nextstat_upper_limit", {"workspace_json": ws_text, "expected": bool(cli.expected)}),
        (
            "nextstat_scan",
            {
                "workspace_json": ws_text,
                "start": float(cli.scan_start),
                "stop": float(cli.scan_stop),
                "points": int(cli.scan_points),
            },
        ),
        ("nextstat_ranking", {"workspace_json": ws_text, "top_n": int(cli.ranking_top_n)}),
    ]

    for name, args in chain:
        resp = run_tool(name, args)
        if not isinstance(resp, dict) or not resp.get("ok", False):
            err = resp.get("error") if isinstance(resp, dict) else None
            raise SystemExit(f"tool failed: {name}: {err}")

    # Summary artifact (best-effort, schema-light).
    fit = json.loads((out_dir / "nextstat_fit.json").read_text(encoding="utf-8"))
    disc = json.loads((out_dir / "nextstat_discovery_asymptotic.json").read_text(encoding="utf-8"))
    ul = json.loads((out_dir / "nextstat_upper_limit.json").read_text(encoding="utf-8"))
    ht = json.loads((out_dir / "nextstat_hypotest.json").read_text(encoding="utf-8"))

    summary = {
        "schema_version": "nextstat.e2e_discovery.v1",
        "inputs": {
            "workspace_path": str(ws_path),
            "transport": str(cli.transport),
            "server_url": (cli.server_url or None),
            "execution": {
                "deterministic": bool(exec_cfg.deterministic),
                "eval_mode": exec_cfg.eval_mode,
                "threads": int(exec_cfg.threads),
            },
            "mu_hypotest": float(cli.mu),
            "upper_limit_expected": bool(cli.expected),
            "scan": {"start": float(cli.scan_start), "stop": float(cli.scan_stop), "points": int(cli.scan_points)},
            "ranking_top_n": int(cli.ranking_top_n),
        },
        "key_results": {
            "poi_value": _extract_nested(fit, ["result", "poi_value"]),
            "poi_error": _extract_nested(fit, ["result", "poi_error"]),
            "nll": _extract_nested(fit, ["result", "nll"]),
            "q0": _extract_nested(disc, ["result", "q0"]),
            "z0": _extract_nested(disc, ["result", "z0"]),
            "p0": _extract_nested(disc, ["result", "p0"]),
            "cls": _extract_nested(ht, ["result", "cls"]),
            "upper_limit_obs": _extract_nested(ul, ["result", "obs_limit"]),
            "upper_limit_exp": _extract_nested(ul, ["result", "exp_limits"]),
        },
        "calls": calls,
    }

    _write_text(out_dir / "calls.json", _pretty_json(calls))
    _write_text(out_dir / "summary.json", _pretty_json(summary))

    md = []
    md.append("# E2E discovery summary")
    md.append("")
    md.append(f"- workspace: `{ws_path}`")
    md.append(f"- transport: `{cli.transport}`")
    if cli.server_url:
        md.append(f"- server_url: `{cli.server_url}`")
    md.append(f"- deterministic: `{exec_cfg.deterministic}` (eval_mode={exec_cfg.eval_mode}, threads={exec_cfg.threads})")
    md.append("")
    md.append("## Key results")
    md.append(f"- poi_value: `{summary['key_results']['poi_value']}`")
    md.append(f"- poi_error: `{summary['key_results']['poi_error']}`")
    md.append(f"- nll: `{summary['key_results']['nll']}`")
    md.append(f"- q0: `{summary['key_results']['q0']}`")
    md.append(f"- z0: `{summary['key_results']['z0']}`")
    md.append(f"- p0: `{summary['key_results']['p0']}`")
    md.append(f"- CLs(mu={cli.mu}): `{summary['key_results']['cls']}`")
    md.append(f"- upper limit (obs): `{summary['key_results']['upper_limit_obs']}`")
    md.append(f"- upper limit (exp): `{summary['key_results']['upper_limit_exp']}`")
    md.append("")
    md.append("## Artifacts")
    md.append(f"- `summary.json`, `calls.json`")
    for name, _args in chain:
        md.append(f"- `{name}.json`")
    md.append("")
    _write_text(out_dir / "summary.md", "\n".join(md) + "\n")

    print(f"Wrote {out_dir}/summary.json and {out_dir}/summary.md")


if __name__ == "__main__":
    main()
