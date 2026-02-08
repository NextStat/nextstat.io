#!/usr/bin/env python3
"""Record a TREx analysis-spec baseline (fit + expected_data + perf).

This is a lightweight "no cluster required" baseline recorder:
- Runs a spec via `scripts/trex/run_analysis_spec.py`
- Extracts the fit surface from `nextstat fit` output JSON
- Computes expected_data surfaces via Python bindings (HistFactoryModel.expected_data)
- Stores total wall time for slowdown gating

Outputs:
- Baseline JSON (`trex_baseline_v0`) under `--out-dir`
- Manifest JSON pointing to the baseline + spec + environment fingerprint

Usage:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/record_trex_analysis_spec_baseline.py \
    --spec docs/specs/trex/analysis_spec_v0.yaml
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

def _local_nextstat_extension_present(repo: Path) -> bool:
    pkg = repo / "bindings" / "ns-py" / "python" / "nextstat"
    if not pkg.exists():
        return False
    pats = ("_core.*.so", "_core.*.pyd", "_core.*.dylib", "_core.*.dll")
    return any(pkg.glob(p) for p in pats)


def _with_py_path(env: Dict[str, str]) -> Dict[str, str]:
    repo = _repo_root()
    if not (_local_nextstat_extension_present(repo) or os.environ.get("NEXTSTAT_FORCE_PYTHONPATH") == "1"):
        return env
    add = str(repo / "bindings" / "ns-py" / "python")
    cur = env.get("PYTHONPATH", "")
    if cur:
        if add in cur.split(os.pathsep):
            return env
        env["PYTHONPATH"] = cur + os.pathsep + add
    else:
        env["PYTHONPATH"] = add
    return env


def _git_info(repo: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        info["commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        info["commit_short"] = info["commit"][:8]
        info["branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        dirty = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo), stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        info["dirty"] = bool(dirty)
    except Exception:
        info["error"] = "git_not_available"
    return info


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
    try:
        import nextstat  # type: ignore

        env["nextstat_version"] = str(nextstat.__version__)
    except Exception:
        env["nextstat_version"] = "unavailable"
    return env


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


def _stamp(hostname: str) -> str:
    return f"{hostname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


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
    """Return a copy of the spec with outputs redirected into work_dir.

    Also forces `execution.fit.enabled=true` so the baseline always has a fit surface.
    """
    out = json.loads(json.dumps(spec))  # cheap deep-copy (JSON-compatible YAML)
    # Resolve all input paths relative to the *original* spec location (not the materialized file).
    # The runner writes the effective YAML under work_dir and then executes `nextstat run --config <effective>`,
    # so any relative paths must be made absolute here to keep behavior stable.
    #
    # We only touch known path fields (schema-driven), leaving non-path strings intact.
    def resolve(spec_base: Path, p: Any) -> Any:
        if not isinstance(p, str) or not p:
            return p
        pp = Path(p)
        if pp.is_absolute():
            return str(pp)
        return str((spec_base / pp).resolve())

    spec_base = spec_dir.resolve()

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

    # Resolve input/report paths to absolute (relative to original spec dir).
    inputs = out.get("inputs") or {}
    if isinstance(inputs, dict):
        mode = inputs.get("mode")
        if mode == "histfactory_xml":
                hf = inputs.get("histfactory") or {}
                if isinstance(hf, dict):
                    hf["export_dir"] = resolve(spec_base, hf.get("export_dir"))
                    hf["combination_xml"] = resolve(spec_base, hf.get("combination_xml"))
        elif mode == "trex_config_txt":
                tc = inputs.get("trex_config_txt") or {}
                if isinstance(tc, dict):
                    tc["config_path"] = resolve(spec_base, tc.get("config_path"))
                    tc["base_dir"] = resolve(spec_base, tc.get("base_dir"))
        elif mode == "trex_config_yaml":
                ty = inputs.get("trex_config_yaml") or {}
                if isinstance(ty, dict):
                    ty["base_dir"] = resolve(spec_base, ty.get("base_dir"))
        elif mode == "workspace_json":
                wj = inputs.get("workspace_json") or {}
                if isinstance(wj, dict):
                    wj["path"] = resolve(spec_base, wj.get("path"))

    rep = exec_cfg.get("report") or {}
    if isinstance(rep, dict):
        rep["histfactory_xml"] = resolve(spec_base, rep.get("histfactory_xml"))

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


def _build_baseline(*, workspace_json: Path, fit_json: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
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

    # Expected data via Python bindings.
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
    ap.add_argument("--spec", type=Path, required=True, help="Path to TREx analysis spec YAML.")
    ap.add_argument(
        "--schema",
        type=Path,
        default=_repo_root() / "docs" / "schemas" / "trex" / "analysis_spec_v0.schema.json",
        help="Path to analysis spec JSON Schema.",
    )
    ap.add_argument("--nextstat", type=str, default=None, help="Path to nextstat binary.")
    ap.add_argument("--out-dir", type=Path, default=Path("tmp/baselines"), help="Baseline output directory.")
    ap.add_argument(
        "--latest-manifest",
        type=Path,
        default=Path("tmp/baselines/latest_trex_analysis_spec_manifest.json"),
        help="Where to write/update the 'latest' manifest pointer.",
    )
    args = ap.parse_args()

    repo = _repo_root()
    env_fp = collect_environment(repo)
    stamp = _stamp(str(env_fp.get("hostname") or socket.gethostname()))

    if not args.spec.exists():
        raise SystemExit(f"Missing --spec: {args.spec}")
    if not args.schema.exists():
        raise SystemExit(f"Missing schema: {args.schema}")

    spec_obj = _load_yaml(args.spec)
    schema_obj = _load_json(args.schema)
    _validate_spec(spec_obj, schema_obj)

    work_dir = repo / "tmp" / "trex_analysis_spec_baseline" / stamp
    work_dir.mkdir(parents=True, exist_ok=True)

    effective = _materialize_spec(spec_obj, spec_dir=args.spec.parent, work_dir=work_dir)
    eff_path = work_dir / "analysis_spec_effective.yaml"
    eff_path.write_text(yaml.safe_dump(effective, sort_keys=False))

    # Record the effective spec hash for reproducibility.
    eff_sha = _sha256_bytes(eff_path.read_bytes())

    nextstat = _resolve_nextstat(args.nextstat)
    run_env = _with_py_path(dict(os.environ))
    rc, wall_s, output = _run_spec(spec_path=eff_path, nextstat=nextstat, env=run_env)
    if rc != 0:
        print(output[-4000:])
        raise SystemExit(rc)

    ws_path = Path(effective["execution"]["import"]["output_json"])
    fit_path = Path(effective["execution"]["fit"]["output_json"])
    if not ws_path.exists():
        raise SystemExit(f"Missing workspace output: {ws_path}")
    if not fit_path.exists():
        raise SystemExit(f"Missing fit output: {fit_path}")

    meta: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "deterministic": int(effective["execution"]["determinism"]["threads"]) == 1,
        "threads": int(effective["execution"]["determinism"]["threads"]),
        "tool_versions": {"nextstat": env_fp.get("nextstat_version")},
        "inputs": {
            "analysis_spec_path": str(args.spec),
            "analysis_spec_effective_sha256": eff_sha,
            "analysis_spec_mode": (effective.get("inputs") or {}).get("mode"),
        },
        "environment": env_fp,
        "timings": {"total_wall_s": float(wall_s)},
        "artifacts": {
            "work_dir": str(work_dir),
            "workspace_json": str(ws_path),
            "fit_json": str(fit_path),
            "scan_json": str(Path(effective["execution"]["profile_scan"]["output_json"])),
            "report_dir": str(Path(effective["execution"]["report"]["out_dir"])),
        },
    }

    # Optional: persist stable hashes for scan/report outputs so the compare step can be strict.
    artifacts = meta["artifacts"]
    scan_cfg = effective.get("execution", {}).get("profile_scan", {})
    if isinstance(scan_cfg, dict) and bool(scan_cfg.get("enabled")):
        scan_path = Path(str(artifacts["scan_json"]))
        if scan_path.exists():
            artifacts["scan_sha256"] = _sha256_file(scan_path)

    report_cfg = effective.get("execution", {}).get("report", {})
    if isinstance(report_cfg, dict) and bool(report_cfg.get("enabled")):
        report_dir = Path(str(artifacts["report_dir"]))
        report_hashes: Dict[str, str] = {}
        for name in ["distributions.json", "pulls.json", "corr.json", "yields.json", "uncertainty.json"]:
            p = report_dir / name
            if p.exists():
                report_hashes[name] = _sha256_normalized_json_file(p)
        if report_hashes:
            artifacts["report_files_sha256_normalized"] = report_hashes

    baseline = _build_baseline(workspace_json=ws_path, fit_json=fit_path, meta=meta)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = out_dir / f"trex_analysis_spec_baseline_{stamp}.json"
    _write_json(baseline_path, baseline)

    # Persist a copy of the effective spec next to the baseline for durability.
    spec_copy_path = out_dir / f"trex_analysis_spec_effective_{stamp}.yaml"
    spec_copy_path.write_text(eff_path.read_text())

    manifest = {
        "schema_version": "trex_analysis_spec_manifest_v0",
        "created_at": meta["created_at"],
        "baseline_path": str(baseline_path),
        "spec_path": str(args.spec),
        "spec_effective_path": str(spec_copy_path),
        "spec_effective_sha256": eff_sha,
        "environment": env_fp,
    }
    manifest_path = out_dir / f"trex_analysis_spec_manifest_{stamp}.json"
    _write_json(manifest_path, manifest)
    _write_json(args.latest_manifest, manifest)

    print(f"OK -> {baseline_path}")
    print(f"manifest -> {manifest_path}")
    print(f"latest -> {args.latest_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
