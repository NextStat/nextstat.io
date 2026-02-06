#!/usr/bin/env python3
"""Compare current Apex2 runs against the latest recorded baseline manifest.

This script is designed for the "no cluster yet" workflow: record baselines once
on a reference machine, then repeatedly compare HEAD against that baseline.

Inputs:
- A baseline manifest JSON produced by `tests/record_baseline.py`
  (default: `tmp/baselines/latest_manifest.json`)

What it does:
- Runs the current pyhf parity suite with the same parameters as the baseline
  and compares NextStat performance (time-per-NLL-call, and optional fit timing).
- Runs the P6 GLM benchmark compare wrapper against the baseline JSON.

Usage:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/compare_with_latest_baseline.py

Exit codes:
  0: OK (parity OK and within slowdown thresholds)
  2: Compare failed (parity failure or slowdown threshold exceeded)
  3: Baseline manifest missing/invalid
  4: Runner error
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _with_py_path(env: Dict[str, str]) -> Dict[str, str]:
    repo = _repo_root()
    add = str(repo / "bindings" / "ns-py" / "python")
    cur = env.get("PYTHONPATH", "")
    if cur:
        if add in cur.split(os.pathsep):
            return env
        env["PYTHONPATH"] = cur + os.pathsep + add
    else:
        env["PYTHONPATH"] = add
    return env


def _run(cmd: List[str], *, cwd: Path, env: Dict[str, str]) -> Tuple[int, str]:
    p = subprocess.run(
        cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return p.returncode, p.stdout


def _git_info(repo: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        info["commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(repo), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        info["commit_short"] = info["commit"][:8]
        info["branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(repo),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=str(repo), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        info["dirty"] = bool(dirty)
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["error"] = "git_not_available"
    return info


def _cpu_brand() -> str:
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL
            )
            return out.decode().strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def collect_environment(repo: Path) -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "timestamp": int(time.time()),
        "datetime_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu": _cpu_brand(),
        "git": _git_info(repo),
    }
    return env


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())

def _load_manifest_with_fallbacks(path: Path) -> Dict[str, Any]:
    """Load a baseline manifest and fill missing baseline keys from the same directory.

    This is intentionally forgiving: recording only one baseline type (e.g. ROOT on a cluster)
    can overwrite `latest_manifest.json`. We recover by scanning `baseline_manifest_*.json` for
    the newest manifest that contains the missing baseline key.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    root = _read_json(path)
    if not isinstance(root, dict):
        raise ValueError("manifest_not_object")

    baselines = root.get("baselines")
    if not isinstance(baselines, dict):
        baselines = {}
        root["baselines"] = baselines

    manifest_dir = path.parent
    # Prefer explicit per-type latest manifests if they exist, then fall back to
    # timestamped baseline manifests.
    candidate_paths: List[Path] = []
    for name in (
        "latest_pyhf_manifest.json",
        "latest_p6_glm_manifest.json",
        "latest_root_manifest.json",
    ):
        p = manifest_dir / name
        if p.exists() and p.resolve() != path.resolve():
            candidate_paths.append(p)
    candidate_paths += list(manifest_dir.glob("baseline_manifest_*.json"))
    candidate_paths = sorted(
        {p.resolve() for p in candidate_paths},
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    def find_latest_with_key(key: str) -> Optional[Tuple[Path, Dict[str, Any]]]:
        for p in candidate_paths:
            try:
                d = _read_json(p)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            b = d.get("baselines")
            if isinstance(b, dict) and key in b:
                return p, d
        return None

    want_keys = [
        "pyhf",
        "p6_glm",
        "root_prereq",
        "root_cases",
        "root_suite",
    ]

    merged_from: List[str] = [str(path)]
    for k in want_keys:
        if k in baselines:
            continue
        found = find_latest_with_key(k)
        if found is None:
            continue
        p, d = found
        b = d.get("baselines")
        if isinstance(b, dict) and k in b:
            baselines[k] = b[k]
            merged_from.append(str(p))

    # Record provenance in a stable way (paths are enough; not used by logic).
    root.setdefault("meta", {})
    if isinstance(root["meta"], dict):
        root["meta"]["merged_from"] = list(dict.fromkeys(merged_from))
    return root


def _case_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        name = r.get("name")
        if isinstance(name, str):
            out[name] = r
    return out


def compare_pyhf_perf(
    *,
    baseline_report: Dict[str, Any],
    current_report: Dict[str, Any],
    max_slowdown: float,
    max_slowdown_fit: Optional[float],
    min_baseline_s: float,
    require_same_host: bool,
) -> Dict[str, Any]:
    base_env = baseline_report.get("baseline_env")
    cur_env = current_report.get("baseline_env")

    warn: List[str] = []
    if require_same_host:
        b_host = (base_env or {}).get("hostname")
        c_host = (cur_env or {}).get("hostname")
        if b_host and c_host and b_host != c_host:
            return {
                "status": "error",
                "reason": "hostname_mismatch",
                "baseline_hostname": b_host,
                "current_hostname": c_host,
            }
    else:
        b_host = (base_env or {}).get("hostname")
        c_host = (cur_env or {}).get("hostname")
        if b_host and c_host and b_host != c_host:
            warn.append(f"hostname_mismatch:{b_host}->{c_host}")

    base_cases = baseline_report.get("cases")
    cur_cases = current_report.get("cases")
    if not isinstance(base_cases, list) or not isinstance(cur_cases, list):
        return {"status": "error", "reason": "missing_cases_list"}

    base_idx = _case_index([c for c in base_cases if isinstance(c, dict)])
    cur_idx = _case_index([c for c in cur_cases if isinstance(c, dict)])

    cases_out: List[Dict[str, Any]] = []
    any_failed = False
    max_slow = 0.0
    max_slow_fit = 0.0

    fit_thr = max_slowdown_fit if max_slowdown_fit is not None else max_slowdown

    for name, cur in sorted(cur_idx.items(), key=lambda kv: kv[0]):
        base = base_idx.get(name)
        row: Dict[str, Any] = {"name": name}
        if base is None:
            any_failed = True
            row.update({"ok": False, "reason": "missing_baseline_case"})
            cases_out.append(row)
            continue

        try:
            b = float(base["perf"]["nextstat_nll_wall_s"])
            c = float(cur["perf"]["nextstat_nll_wall_s"])
        except Exception:
            any_failed = True
            row.update({"ok": False, "reason": "missing_perf_nll"})
            cases_out.append(row)
            continue

        if not (b > 0.0) or not (c >= 0.0):
            any_failed = True
            row.update({"ok": False, "reason": "invalid_perf_nll"})
            cases_out.append(row)
            continue

        slow = c / b
        max_slow = max(max_slow, slow)
        checked = b >= float(min_baseline_s)
        ok = (slow <= float(max_slowdown)) if checked else True
        if not ok:
            any_failed = True
        row.update(
            {
                "ok": bool(ok),
                "baseline": {"nextstat_nll_wall_s": float(b)},
                "current": {"nextstat_nll_wall_s": float(c)},
                "slowdown": {"nll": float(slow)},
                "checks": {"nll_checked": bool(checked)},
                "thresholds": {"nll": float(max_slowdown)},
            }
        )

        # Optional fit perf compare (only if present in both).
        b_fit = base.get("perf", {}).get("fit", {}).get("nextstat_wall_s")
        c_fit = cur.get("perf", {}).get("fit", {}).get("nextstat_wall_s")
        if b_fit is not None and c_fit is not None:
            try:
                bfv = float(b_fit)
                cfv = float(c_fit)
            except Exception:
                any_failed = True
                row["ok"] = False
                row["reason"] = "invalid_perf_fit"
                cases_out.append(row)
                continue
            if not (bfv > 0.0) or not (cfv >= 0.0):
                any_failed = True
                row["ok"] = False
                row["reason"] = "invalid_perf_fit"
                cases_out.append(row)
                continue
            sfit = cfv / bfv
            max_slow_fit = max(max_slow_fit, sfit)
            ok_fit = sfit <= float(fit_thr)
            if not ok_fit:
                any_failed = True
                row["ok"] = False
            row["baseline"]["nextstat_fit_wall_s"] = float(bfv)
            row["current"]["nextstat_fit_wall_s"] = float(cfv)
            row["slowdown"]["fit"] = float(sfit)
            row["checks"]["fit_checked"] = True
            row["thresholds"]["fit"] = float(fit_thr)
        cases_out.append(row)

    baseline_only = sorted(set(base_idx.keys()) - set(cur_idx.keys()))
    if baseline_only:
        warn.append(f"baseline_only_cases:{len(baseline_only)}")

    status = "ok" if not any_failed else "fail"
    return {
        "status": status,
        "warnings": warn,
        "summary": {
            "n_cases": int(len(cases_out)),
            "n_ok": int(sum(1 for c in cases_out if c.get("ok") is True)),
            "n_fail": int(sum(1 for c in cases_out if c.get("ok") is False)),
            "max_slowdown_nll": float(max_slow),
            "max_slowdown_fit": float(max_slow_fit),
            "baseline_only": baseline_only,
        },
        "cases": cases_out,
    }


def compare_root_suite_perf(
    *,
    baseline_report: Dict[str, Any],
    current_report: Dict[str, Any],
    max_slowdown: float,
    min_baseline_s: float,
    require_same_host: bool,
) -> Dict[str, Any]:
    base_env = baseline_report.get("baseline_env")
    cur_env = current_report.get("baseline_env")

    warn: List[str] = []
    if require_same_host:
        b_host = (base_env or {}).get("hostname")
        c_host = (cur_env or {}).get("hostname")
        if b_host and c_host and b_host != c_host:
            return {
                "status": "error",
                "reason": "hostname_mismatch",
                "baseline_hostname": b_host,
                "current_hostname": c_host,
            }
    else:
        b_host = (base_env or {}).get("hostname")
        c_host = (cur_env or {}).get("hostname")
        if b_host and c_host and b_host != c_host:
            warn.append(f"hostname_mismatch:{b_host}->{c_host}")

    base_cases = baseline_report.get("cases")
    cur_cases = current_report.get("cases")
    if not isinstance(base_cases, list) or not isinstance(cur_cases, list):
        return {"status": "error", "reason": "missing_cases_list"}

    base_idx = _case_index([c for c in base_cases if isinstance(c, dict)])
    cur_idx = _case_index([c for c in cur_cases if isinstance(c, dict)])

    cases_out: List[Dict[str, Any]] = []
    any_failed = False
    max_slow = 0.0

    for name, cur in sorted(cur_idx.items(), key=lambda kv: kv[0]):
        base = base_idx.get(name)
        row: Dict[str, Any] = {"name": name}
        if base is None:
            any_failed = True
            row.update({"ok": False, "reason": "missing_baseline_case"})
            cases_out.append(row)
            continue

        if cur.get("status") != "ok":
            any_failed = True
            row.update({"ok": False, "reason": f"current_case_status:{cur.get('status')}"})
            cases_out.append(row)
            continue
        if base.get("status") != "ok":
            any_failed = True
            row.update({"ok": False, "reason": f"baseline_case_status:{base.get('status')}"})
            cases_out.append(row)
            continue

        try:
            b = float(base.get("timing_s", {}).get("nextstat_profile_scan"))
            c = float(cur.get("timing_s", {}).get("nextstat_profile_scan"))
        except Exception:
            any_failed = True
            row.update({"ok": False, "reason": "missing_nextstat_profile_scan_timing"})
            cases_out.append(row)
            continue

        if not (b > 0.0) or not (c >= 0.0):
            any_failed = True
            row.update({"ok": False, "reason": "invalid_nextstat_profile_scan_timing"})
            cases_out.append(row)
            continue

        slow = c / b
        max_slow = max(max_slow, slow)
        checked = b >= float(min_baseline_s)
        ok = (slow <= float(max_slowdown)) if checked else True
        if not ok:
            any_failed = True
        row.update(
            {
                "ok": bool(ok),
                "baseline": {"nextstat_profile_scan_s": float(b)},
                "current": {"nextstat_profile_scan_s": float(c)},
                "slowdown": {"nextstat_profile_scan": float(slow)},
                "checks": {"timing_checked": bool(checked)},
                "thresholds": {"max_slowdown": float(max_slowdown)},
            }
        )
        cases_out.append(row)

    baseline_only = sorted(set(base_idx.keys()) - set(cur_idx.keys()))
    if baseline_only:
        warn.append(f"baseline_only_cases:{len(baseline_only)}")

    status = "ok" if not any_failed else "fail"
    return {
        "status": status,
        "warnings": warn,
        "summary": {
            "n_cases": int(len(cases_out)),
            "n_ok": int(sum(1 for c in cases_out if c.get("ok") is True)),
            "n_fail": int(sum(1 for c in cases_out if c.get("ok") is False)),
            "max_slowdown_nextstat_profile_scan": float(max_slow),
            "baseline_only": baseline_only,
        },
        "cases": cases_out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("tmp/baselines/latest_manifest.json"),
        help="Baseline manifest JSON (default: tmp/baselines/latest_manifest.json).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("tmp/baseline_compare_report.json"),
        help="Consolidated compare report output path.",
    )
    ap.add_argument(
        "--workdir",
        type=Path,
        default=Path("tmp/baseline_compare"),
        help="Work directory for intermediate outputs.",
    )
    ap.add_argument(
        "--require-same-host",
        action="store_true",
        help="Fail if baseline hostname != current hostname (recommended for perf gating).",
    )

    # pyhf compare thresholds
    ap.add_argument("--pyhf-max-slowdown", type=float, default=1.30)
    ap.add_argument("--pyhf-max-slowdown-fit", type=float, default=None)
    ap.add_argument("--pyhf-min-baseline-s", type=float, default=0.0)

    # P6 compare thresholds (passed through)
    ap.add_argument("--p6-max-slowdown", type=float, default=1.30)
    ap.add_argument("--p6-min-baseline-fit-s", type=float, default=1e-3)

    # ROOT suite perf compare thresholds
    ap.add_argument("--root-max-slowdown", type=float, default=1.30)
    ap.add_argument("--root-min-baseline-s", type=float, default=0.0)
    args = ap.parse_args()

    repo = _repo_root()
    env_dict = _with_py_path(os.environ.copy())
    cur_env = collect_environment(repo)

    if not args.manifest.exists():
        print(f"Missing baseline manifest: {args.manifest}", file=sys.stderr)
        return 3

    try:
        manifest = _load_manifest_with_fallbacks(args.manifest)
    except Exception as e:
        print(f"Failed to load baseline manifest: {e}", file=sys.stderr)
        return 3

    baselines = manifest.get("baselines")
    if not isinstance(baselines, dict):
        print("Baseline manifest missing key 'baselines'", file=sys.stderr)
        return 3

    args.workdir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "meta": {
            "timestamp": int(time.time()),
            "manifest_path": str(args.manifest),
            "workdir": str(args.workdir),
            "current_env": cur_env,
            "thresholds": {
                "pyhf_max_slowdown": float(args.pyhf_max_slowdown),
                "p6_max_slowdown": float(args.p6_max_slowdown),
                "root_max_slowdown": float(args.root_max_slowdown),
            },
        },
        "baseline_manifest": manifest,
        "pyhf": None,
        "p6_glm": None,
        "root_suite": None,
        "status": None,
        "summary": None,
    }

    any_failed = False
    any_error = False

    # ------------------------------------------------------------------
    # pyhf: run current + compare perf vs baseline
    # ------------------------------------------------------------------
    pyhf_entry = baselines.get("pyhf")
    if isinstance(pyhf_entry, dict) and isinstance(pyhf_entry.get("path"), str):
        baseline_pyhf = Path(pyhf_entry["path"])
        if not baseline_pyhf.exists():
            report["pyhf"] = {"status": "error", "reason": "baseline_missing", "path": str(baseline_pyhf)}
            any_error = True
        else:
            base_rep = _read_json(baseline_pyhf)
            base_meta = (base_rep.get("meta") if isinstance(base_rep, dict) else None) or {}
            base_params = (base_meta.get("params") if isinstance(base_meta, dict) else None) or {}

            sizes_v = base_params.get("sizes", "2,16,64,256")
            if isinstance(sizes_v, list):
                sizes_s = ",".join(str(int(x)) for x in sizes_v)
            else:
                sizes_s = str(sizes_v)

            out_cur = args.workdir / f"pyhf_current_{cur_env['hostname']}_{cur_env['timestamp']}.json"
            runner = repo / "tests" / "apex2_pyhf_validation_report.py"
            cmd = [
                sys.executable,
                str(runner),
                "--out",
                str(out_cur),
                "--sizes",
                sizes_s,
                "--n-random",
                str(int(base_params.get("n_random", 8))),
                "--seed",
                str(int(base_params.get("seed", 0))),
            ]

            # Keep tolerances consistent if present.
            if "nll_atol" in base_params:
                cmd += ["--nll-atol", str(float(base_params["nll_atol"]))]
            if "nll_rtol" in base_params:
                cmd += ["--nll-rtol", str(float(base_params["nll_rtol"]))]
            if "expected_data_atol" in base_params:
                cmd += ["--expected-data-atol", str(float(base_params["expected_data_atol"]))]

            if bool(base_params.get("fit")):
                cmd.append("--fit")

            t0 = time.time()
            rc, stdout = _run(cmd, cwd=repo, env=env_dict)
            wall = time.time() - t0

            if rc != 0 or not out_cur.exists():
                report["pyhf"] = {
                    "status": "error",
                    "reason": "runner_failed",
                    "returncode": int(rc),
                    "stdout_tail": stdout[-4000:],
                    "current_path": str(out_cur),
                }
                any_error = True
            else:
                cur_rep = _read_json(out_cur)
                if isinstance(cur_rep, dict):
                    cur_rep["baseline_env"] = cur_env
                    cur_rep.setdefault("meta", {})
                    if isinstance(cur_rep["meta"], dict):
                        cur_rep["meta"]["wall_s"] = float(wall)
                    out_cur.write_text(json.dumps(cur_rep, indent=2))

                cur_rep = _read_json(out_cur)
                # Parity status is already encoded via exit code, but keep it explicit.
                parity_ok = True
                try:
                    summary = cur_rep.get("summary", {})
                    parity_ok = int(summary.get("n_failed", 0)) == 0
                except Exception:
                    parity_ok = False

                cmp_rep = compare_pyhf_perf(
                    baseline_report=base_rep,
                    current_report=cur_rep,
                    max_slowdown=float(args.pyhf_max_slowdown),
                    max_slowdown_fit=float(args.pyhf_max_slowdown_fit)
                    if args.pyhf_max_slowdown_fit is not None
                    else None,
                    min_baseline_s=float(args.pyhf_min_baseline_s),
                    require_same_host=bool(args.require_same_host),
                )

                pyhf_status = "ok" if (parity_ok and cmp_rep.get("status") == "ok") else "fail"
                if not parity_ok or cmp_rep.get("status") != "ok":
                    any_failed = True

                report["pyhf"] = {
                    "status": pyhf_status,
                    "parity_ok": bool(parity_ok),
                    "baseline_path": str(baseline_pyhf),
                    "current_path": str(out_cur),
                    "compare": cmp_rep,
                }
    else:
        report["pyhf"] = {"status": "skipped", "reason": "no_baseline_in_manifest"}

    # ------------------------------------------------------------------
    # P6 GLM: compare wrapper (runs current benchmark internally)
    # ------------------------------------------------------------------
    p6_entry = baselines.get("p6_glm")
    if isinstance(p6_entry, dict) and isinstance(p6_entry.get("path"), str):
        baseline_p6 = Path(p6_entry["path"])
        if not baseline_p6.exists():
            report["p6_glm"] = {"status": "error", "reason": "baseline_missing", "path": str(baseline_p6)}
            any_error = True
        else:
            base_p6_rep = _read_json(baseline_p6)
            settings = (base_p6_rep.get("meta", {}) or {}).get("settings", {}) if isinstance(base_p6_rep, dict) else {}
            sizes_list = settings.get("sizes", [200, 2000, 20000])
            if isinstance(sizes_list, list):
                sizes = ",".join(str(int(x)) for x in sizes_list)
            else:
                sizes = "200,2000,20000"
            p = int(settings.get("p", 20)) if isinstance(settings, dict) else 20
            l2 = settings.get("l2", None) if isinstance(settings, dict) else None
            l2_f = float(l2) if isinstance(l2, (int, float)) else 0.0
            nb_alpha = float(settings.get("nb_alpha", 0.5)) if isinstance(settings, dict) else 0.5

            out_compare = args.workdir / f"p6_glm_compare_{cur_env['hostname']}_{cur_env['timestamp']}.json"
            out_bench = args.workdir / f"p6_glm_current_bench_{cur_env['hostname']}_{cur_env['timestamp']}.json"
            runner = repo / "tests" / "apex2_p6_glm_benchmark_report.py"
            cmd = [
                sys.executable,
                str(runner),
                "--baseline",
                str(baseline_p6),
                "--bench-out",
                str(out_bench),
                "--out",
                str(out_compare),
                "--max-slowdown",
                str(float(args.p6_max_slowdown)),
                "--min-baseline-fit-s",
                str(float(args.p6_min_baseline_fit_s)),
                "--sizes",
                sizes,
                "--p",
                str(int(p)),
                "--l2",
                str(float(l2_f)),
                "--nb-alpha",
                str(float(nb_alpha)),
            ]
            rc, stdout = _run(cmd, cwd=repo, env=env_dict)
            if rc not in (0, 2) or not out_compare.exists():
                report["p6_glm"] = {
                    "status": "error",
                    "reason": "runner_failed",
                    "returncode": int(rc),
                    "stdout_tail": stdout[-4000:],
                    "baseline_path": str(baseline_p6),
                    "compare_path": str(out_compare),
                    "bench_path": str(out_bench),
                }
                any_error = True
            else:
                p6_rep = _read_json(out_compare)
                p6_status = p6_rep.get("status")
                ok = (p6_status == "ok") and (rc == 0)
                if not ok:
                    any_failed = True
                report["p6_glm"] = {
                    "status": "ok" if ok else "fail",
                    "baseline_path": str(baseline_p6),
                    "bench_path": str(out_bench),
                    "compare_path": str(out_compare),
                    "compare": p6_rep,
                }
    else:
        report["p6_glm"] = {"status": "skipped", "reason": "no_baseline_in_manifest"}

    # ------------------------------------------------------------------
    # ROOT suite: optional compare (only if baseline suite + cases exist)
    # ------------------------------------------------------------------
    root_suite_entry = baselines.get("root_suite")
    root_cases_entry = baselines.get("root_cases")
    if (
        isinstance(root_suite_entry, dict)
        and isinstance(root_suite_entry.get("path"), str)
        and isinstance(root_cases_entry, dict)
        and isinstance(root_cases_entry.get("path"), str)
    ):
        baseline_root_suite = Path(root_suite_entry["path"])
        baseline_root_cases = Path(root_cases_entry["path"])

        if not baseline_root_suite.exists() or not baseline_root_cases.exists():
            report["root_suite"] = {
                "status": "error",
                "reason": "baseline_missing",
                "baseline_suite_path": str(baseline_root_suite),
                "baseline_cases_path": str(baseline_root_cases),
            }
            any_error = True
        else:
            # Check current prereqs (fast)
            prereq_out = args.workdir / f"root_prereq_current_{cur_env['hostname']}_{cur_env['timestamp']}.json"
            prereq_runner = repo / "tests" / "apex2_root_suite_report.py"
            rc_pr, stdout_pr = _run(
                [sys.executable, str(prereq_runner), "--prereq-only", "--out", str(prereq_out)],
                cwd=repo,
                env=env_dict,
            )
            if rc_pr != 0:
                report["root_suite"] = {
                    "status": "skipped",
                    "reason": "missing_prereqs",
                    "prereq_path": str(prereq_out),
                    "stdout_tail": stdout_pr[-4000:],
                }
            else:
                base_rep = _read_json(baseline_root_suite)
                base_meta = (base_rep.get("meta") if isinstance(base_rep, dict) else None) or {}
                thr = (base_meta.get("thresholds") if isinstance(base_meta, dict) else None) or {}
                dq_atol = float(thr.get("dq_atol", 1e-3))
                mu_hat_atol = float(thr.get("mu_hat_atol", 1e-3))

                out_cur = args.workdir / f"root_suite_current_{cur_env['hostname']}_{cur_env['timestamp']}.json"
                workdir = args.workdir / f"root_parity_suite_{cur_env['hostname']}_{cur_env['timestamp']}"
                cmd = [
                    sys.executable,
                    str(prereq_runner),
                    "--cases",
                    str(baseline_root_cases),
                    "--keep-going",
                    "--workdir",
                    str(workdir),
                    "--dq-atol",
                    str(float(dq_atol)),
                    "--mu-hat-atol",
                    str(float(mu_hat_atol)),
                    "--out",
                    str(out_cur),
                ]

                rc, stdout = _run(cmd, cwd=repo, env=env_dict)
                if rc != 0 or not out_cur.exists():
                    report["root_suite"] = {
                        "status": "error",
                        "reason": "runner_failed",
                        "returncode": int(rc),
                        "stdout_tail": stdout[-4000:],
                        "baseline_suite_path": str(baseline_root_suite),
                        "baseline_cases_path": str(baseline_root_cases),
                        "current_path": str(out_cur),
                    }
                    any_error = True
                else:
                    cur_rep = _read_json(out_cur)
                    if isinstance(cur_rep, dict):
                        cur_rep["baseline_env"] = cur_env
                        out_cur.write_text(json.dumps(cur_rep, indent=2))

                    cur_rep = _read_json(out_cur)
                    # Root suite already encodes correctness vs ROOT via its exit code / statuses.
                    root_ok = True
                    try:
                        summ = cur_rep.get("summary", {})
                        root_ok = int(summ.get("n_fail", 0)) == 0 and int(summ.get("n_error", 0)) == 0
                    except Exception:
                        root_ok = False

                    cmp = compare_root_suite_perf(
                        baseline_report=base_rep,
                        current_report=cur_rep,
                        max_slowdown=float(args.root_max_slowdown),
                        min_baseline_s=float(args.root_min_baseline_s),
                        require_same_host=bool(args.require_same_host),
                    )
                    status = "ok" if (root_ok and cmp.get("status") == "ok") else "fail"
                    if status != "ok":
                        any_failed = True
                    report["root_suite"] = {
                        "status": status,
                        "root_ok": bool(root_ok),
                        "baseline_suite_path": str(baseline_root_suite),
                        "baseline_cases_path": str(baseline_root_cases),
                        "current_path": str(out_cur),
                        "compare": cmp,
                    }
    else:
        report["root_suite"] = {"status": "skipped", "reason": "no_baseline_in_manifest"}

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    if any_error:
        report["status"] = "error"
        report["summary"] = {"reason": "runner_error"}
        rc_out = 4
    elif any_failed:
        report["status"] = "fail"
        report["summary"] = {"reason": "compare_failed"}
        rc_out = 2
    else:
        report["status"] = "ok"
        report["summary"] = {"reason": "ok"}
        rc_out = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Wrote: {args.out}")
    return rc_out


if __name__ == "__main__":
    raise SystemExit(main())
