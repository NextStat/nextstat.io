#!/usr/bin/env python3
"""Compare a TREx analysis spec run against the latest recorded baseline.

This is a "no cluster" regression gate:
- Reads a manifest produced by `tests/record_trex_analysis_spec_baseline.py`
- Re-runs the spec in a fresh work dir (outputs redirected)
- Compares fit + expected_data surfaces (numbers-first)
- Optionally checks a slowdown gate (total wall time) from the spec:
  `gates.baseline_compare` (enabled/require_same_host/max_slowdown)
- Optionally checks normalized report artifact hashes if the baseline recorded them

Exit codes:
  0: OK
  2: FAIL (numeric diffs, artifact mismatch, or slowdown gate)
  3: baseline manifest missing/invalid
  4: runner error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_import_paths() -> None:
    repo = _repo_root()
    tests_py = repo / "tests" / "python"
    bindings = repo / "bindings" / "ns-py" / "python"
    for p in [tests_py, bindings]:
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)


_ensure_import_paths()

from _tolerances import EXPECTED_DATA_ATOL, TWICE_NLL_ATOL, TWICE_NLL_RTOL  # type: ignore  # noqa: E402
from _tolerances import PARAM_UNCERTAINTY_ATOL, PARAM_VALUE_ATOL  # type: ignore  # noqa: E402
from _trex_baseline_compare import Tol, compare_baseline_v0, format_report  # type: ignore  # noqa: E402


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


def _cpu_brand() -> str:
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL)
            return out.decode().strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def collect_environment() -> Dict[str, Any]:
    repo = _repo_root()
    out: Dict[str, Any] = {
        "timestamp": int(time.time()),
        "datetime_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu": _cpu_brand(),
    }
    try:
        dirty = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo), stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        out["git_dirty"] = bool(dirty)
    except Exception:
        out["git_dirty"] = None
    return out


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _normalize_artifact_json(obj: Any) -> Any:
    # Strip volatile `meta` (timestamps/tool versions) to focus on numeric content.
    if isinstance(obj, dict):
        out = dict(obj)
        out.pop("meta", None)
        return out
    return obj


def _sha256_normalized_json_file(path: Path) -> str:
    obj = json.loads(path.read_text())
    norm = _normalize_artifact_json(obj)
    b = json.dumps(norm, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return _sha256_bytes(b)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def _validate_spec(spec: Any, schema: Any) -> None:
    try:
        import jsonschema  # type: ignore
    except Exception as e:
        raise SystemExit(f"Missing dependency jsonschema: {e}")
    v = jsonschema.Draft202012Validator(schema)
    errors = sorted(v.iter_errors(spec), key=lambda e: list(e.path))
    if errors:
        lines = [f"FAIL ({len(errors)} error(s))"]
        for e in errors[:50]:
            loc = "$" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in e.path)
            lines.append(f"- {loc}: {e.message}")
        raise SystemExit("\n".join(lines))


def _materialize_spec(spec: dict[str, Any], *, spec_dir: Path, work_dir: Path) -> dict[str, Any]:
    out = json.loads(json.dumps(spec))
    spec_base = spec_dir.resolve()

    def resolve(p: Any) -> Any:
        if not isinstance(p, str) or not p:
            return p
        pp = Path(p)
        if pp.is_absolute():
            return str(pp)
        return str((spec_base / pp).resolve())

    exec_cfg = out.setdefault("execution", {})
    exec_cfg.setdefault("import", {})
    exec_cfg.setdefault("fit", {})
    exec_cfg.setdefault("profile_scan", {})
    exec_cfg.setdefault("report", {})
    exec_cfg.setdefault("determinism", {})

    exec_cfg["import"]["output_json"] = str(work_dir / "workspace.json")
    exec_cfg["fit"]["enabled"] = True
    exec_cfg["fit"]["output_json"] = str(work_dir / "fit.json")
    exec_cfg["profile_scan"]["output_json"] = str(work_dir / "scan.json")
    exec_cfg["report"]["out_dir"] = str(work_dir / "report")

    render = exec_cfg["report"].setdefault("render", {})
    render.setdefault("enabled", False)
    if render.get("enabled"):
        render["pdf"] = str(work_dir / "report" / "report.pdf")
        render["svg_dir"] = str(work_dir / "report" / "svg")
        render.setdefault("python", None)

    # Resolve input/report paths to absolute (relative to the baseline effective spec dir).
    inputs = out.get("inputs") or {}
    if isinstance(inputs, dict):
        mode = inputs.get("mode")
        if mode == "histfactory_xml":
            hf = inputs.get("histfactory") or {}
            if isinstance(hf, dict):
                hf["export_dir"] = resolve(hf.get("export_dir"))
                hf["combination_xml"] = resolve(hf.get("combination_xml"))
        elif mode == "trex_config_txt":
            tc = inputs.get("trex_config_txt") or {}
            if isinstance(tc, dict):
                tc["config_path"] = resolve(tc.get("config_path"))
                tc["base_dir"] = resolve(tc.get("base_dir"))
        elif mode == "trex_config_yaml":
            ty = inputs.get("trex_config_yaml") or {}
            if isinstance(ty, dict):
                ty["base_dir"] = resolve(ty.get("base_dir"))
        elif mode == "workspace_json":
            wj = inputs.get("workspace_json") or {}
            if isinstance(wj, dict):
                wj["path"] = resolve(wj.get("path"))

    rep = exec_cfg.get("report") or {}
    if isinstance(rep, dict):
        rep["histfactory_xml"] = resolve(rep.get("histfactory_xml"))

    return out


def _resolve_nextstat(nextstat_arg: Optional[str]) -> str:
    if nextstat_arg:
        return nextstat_arg
    local = _repo_root() / "target" / "release" / "nextstat"
    if local.exists():
        return str(local)
    return "nextstat"


def _run_spec(*, spec_path: Path, nextstat: str, env: Dict[str, str]) -> Tuple[int, float, str]:
    repo = _repo_root()
    cmd = [sys.executable, "scripts/trex/run_analysis_spec.py", "--spec", str(spec_path), "--nextstat", nextstat]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=str(repo), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - t0
    return p.returncode, wall, p.stdout


def _build_candidate_baseline(*, workspace_json: Path, fit_json: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_import_paths()
    ws_text = workspace_json.read_text()
    fit = _load_json(fit_json)

    names = fit.get("parameter_names") or []
    bestfit = fit.get("bestfit") or []
    uncs = fit.get("uncertainties") or []
    twice_nll = fit.get("twice_nll")

    if not (isinstance(names, list) and isinstance(bestfit, list) and isinstance(uncs, list)):
        raise SystemExit("fit.json missing required arrays: parameter_names/bestfit/uncertainties")
    if len(names) != len(bestfit) or len(names) != len(uncs):
        raise SystemExit(
            f"fit.json length mismatch: parameter_names={len(names)} bestfit={len(bestfit)} uncertainties={len(uncs)}"
        )

    try:
        from nextstat._core import HistFactoryModel  # type: ignore
    except Exception as e:
        raise SystemExit(f"Missing nextstat python bindings (PYTHONPATH?): {e}")

    model = HistFactoryModel.from_workspace(ws_text)
    exp_main = [float(x) for x in model.expected_data(bestfit, include_auxdata=False)]
    exp_with_aux = [float(x) for x in model.expected_data(bestfit, include_auxdata=True)]

    params = [
        {"name": str(n), "value": float(v), "uncertainty": float(u)}
        for (n, v, u) in zip(names, bestfit, uncs)
    ]

    return {
        "schema_version": "trex_baseline_v0",
        "meta": meta,
        "fit": {"twice_nll": float(twice_nll), "parameters": params, "covariance": None},
        "expected_data": {"pyhf_main": exp_main, "pyhf_with_aux": exp_with_aux},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("tmp/baselines/latest_trex_analysis_spec_manifest.json"),
        help="Latest baseline manifest JSON.",
    )
    ap.add_argument(
        "--schema",
        type=Path,
        default=_repo_root() / "docs" / "schemas" / "trex" / "analysis_spec_v0.schema.json",
        help="Analysis spec schema (for validation).",
    )
    ap.add_argument("--nextstat", type=str, default=None, help="Path to nextstat binary.")
    ap.add_argument("--out", type=Path, default=Path("tmp/trex_analysis_spec_compare_report.json"))
    ap.add_argument("--require-same-host", action="store_true", help="Fail if hostname differs from baseline.")
    ap.add_argument(
        "--min-baseline-s",
        type=float,
        default=1e-3,
        help="Skip slowdown gating if baseline total time is below this (timer noise).",
    )
    args = ap.parse_args()

    repo = _repo_root()
    if not args.manifest.exists():
        raise SystemExit(3)

    manifest = _load_json(args.manifest)
    baseline_path = Path(str(manifest.get("baseline_path") or ""))
    spec_effective_path = Path(str(manifest.get("spec_effective_path") or ""))
    spec_effective_sha = str(manifest.get("spec_effective_sha256") or "")

    if not baseline_path.exists():
        raise SystemExit(f"Missing baseline: {baseline_path}")
    if not spec_effective_path.exists():
        raise SystemExit(f"Missing effective spec: {spec_effective_path}")
    if spec_effective_sha:
        got = _sha256_file(spec_effective_path)
        if got != spec_effective_sha:
            raise SystemExit("Effective spec sha256 mismatch (baseline manifest vs file).")

    if not args.schema.exists():
        raise SystemExit(f"Missing schema: {args.schema}")

    baseline = _load_json(baseline_path)
    baseline_total = float(((baseline.get("meta") or {}).get("timings") or {}).get("total_wall_s") or 0.0)
    baseline_host = str(((baseline.get("meta") or {}).get("environment") or {}).get("hostname") or "")
    baseline_artifacts = (baseline.get("meta") or {}).get("artifacts") or {}

    spec0 = _load_yaml(spec_effective_path)
    schema_obj = _load_json(args.schema)
    _validate_spec(spec0, schema_obj)

    now_env = collect_environment()
    if args.require_same_host and baseline_host and now_env.get("hostname") != baseline_host:
        report = {
            "ok": False,
            "reason": "require_same_host",
            "baseline_host": baseline_host,
            "current_host": now_env.get("hostname"),
        }
        _write_json(args.out, report)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = repo / "tmp" / "trex_analysis_spec_compare" / stamp
    work_dir.mkdir(parents=True, exist_ok=True)

    effective = _materialize_spec(spec0, spec_dir=spec_effective_path.parent, work_dir=work_dir)
    eff_path = work_dir / "analysis_spec_effective.yaml"
    eff_path.write_text(yaml.safe_dump(effective, sort_keys=False))

    nextstat = _resolve_nextstat(args.nextstat)
    run_env = _with_py_path(dict(os.environ))
    rc, wall_s, output = _run_spec(spec_path=eff_path, nextstat=nextstat, env=run_env)
    if rc != 0:
        report = {"ok": False, "reason": "runner_error", "exit_code": rc, "output_tail": output[-4000:]}
        _write_json(args.out, report)
        return 4

    ws_path = Path(effective["execution"]["import"]["output_json"])
    fit_path = Path(effective["execution"]["fit"]["output_json"])
    if not ws_path.exists() or not fit_path.exists():
        report = {"ok": False, "reason": "missing_outputs", "workspace": str(ws_path), "fit": str(fit_path)}
        _write_json(args.out, report)
        return 4

    cand_meta: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "deterministic": int(effective["execution"]["determinism"]["threads"]) == 1,
        "threads": int(effective["execution"]["determinism"]["threads"]),
        "inputs": {"baseline_manifest": str(args.manifest), "baseline_path": str(baseline_path)},
        "environment": now_env,
        "timings": {"total_wall_s": float(wall_s)},
    }
    cand = _build_candidate_baseline(workspace_json=ws_path, fit_json=fit_path, meta=cand_meta)

    res = compare_baseline_v0(
        ref=baseline,
        cand=cand,
        tol_twice_nll=Tol(atol=TWICE_NLL_ATOL, rtol=TWICE_NLL_RTOL),
        tol_expected_data=Tol(atol=EXPECTED_DATA_ATOL, rtol=0.0),
        tol_param_value=Tol(atol=PARAM_VALUE_ATOL, rtol=0.0),
        tol_param_unc=Tol(atol=PARAM_UNCERTAINTY_ATOL, rtol=0.0),
    )

    artifacts_ok = True
    artifacts_diffs: list[dict[str, Any]] = []
    want_report_hashes = baseline_artifacts.get("report_files_sha256_normalized")
    if isinstance(want_report_hashes, dict):
        report_dir = Path(effective["execution"]["report"]["out_dir"])
        for name, want_hash in sorted(want_report_hashes.items()):
            p = report_dir / str(name)
            if not p.exists():
                artifacts_ok = False
                artifacts_diffs.append({"file": name, "note": "missing_in_current"})
                continue
            got = _sha256_normalized_json_file(p)
            if str(got) != str(want_hash):
                artifacts_ok = False
                artifacts_diffs.append({"file": name, "note": "sha256_mismatch", "want": want_hash, "got": got})

    want_scan_hash = baseline_artifacts.get("scan_sha256")
    if isinstance(want_scan_hash, str) and want_scan_hash:
        scan_path = Path(effective["execution"]["profile_scan"]["output_json"])
        if not scan_path.exists():
            artifacts_ok = False
            artifacts_diffs.append({"file": "scan.json", "note": "missing_in_current"})
        else:
            got = _sha256_file(scan_path)
            if got != want_scan_hash:
                artifacts_ok = False
                artifacts_diffs.append({"file": "scan.json", "note": "sha256_mismatch", "want": want_scan_hash, "got": got})

    gate = ((spec0.get("gates") or {}).get("baseline_compare") or {}) if isinstance(spec0, dict) else {}
    gate_enabled = bool(gate.get("enabled"))
    max_slowdown = float(gate.get("max_slowdown") or 0.0)
    require_same_host = bool(gate.get("require_same_host"))

    perf_ok = True
    slowdown: Optional[float] = None
    perf_note: Optional[str] = None
    if gate_enabled and max_slowdown > 0.0 and baseline_total >= float(args.min_baseline_s):
        if require_same_host and baseline_host and now_env.get("hostname") != baseline_host:
            perf_ok = False
            perf_note = "baseline_compare.require_same_host"
        else:
            slowdown = float(wall_s) / float(baseline_total) if baseline_total > 0 else None
            if slowdown is not None and slowdown > max_slowdown:
                perf_ok = False
                perf_note = f"slowdown {slowdown:.3f} > {max_slowdown:.3f}"

    ok = bool(res.ok) and bool(perf_ok) and bool(artifacts_ok)
    report = {
        "ok": ok,
        "baseline": {"path": str(baseline_path), "total_wall_s": baseline_total, "host": baseline_host},
        "current": {"total_wall_s": float(wall_s), "host": now_env.get("hostname")},
        "artifacts": {"ok": artifacts_ok, "diffs": artifacts_diffs},
        "perf": {
            "gate_enabled": gate_enabled,
            "max_slowdown": max_slowdown,
            "min_baseline_s": float(args.min_baseline_s),
            "slowdown": slowdown,
            "ok": perf_ok,
            "note": perf_note,
        },
        "numeric": {"ok": res.ok, "diffs": [d.__dict__ for d in res.worst(200)]},
        "numeric_text": format_report(res, top_n=20),
    }
    _write_json(args.out, report)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
