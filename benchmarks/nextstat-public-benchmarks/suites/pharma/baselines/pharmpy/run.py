#!/usr/bin/env python3
"""Pharmpy baseline runner for pharma suite.

Supported currently:
- dataset kind: pop_pk_1c_oral
- error model: additive
- estimation backend: pharmpy -> nlmixr

Notes:
- pharmpy-core 2.0.0 has known parser issues in nlmixr result extraction on some
  environments. This runner includes a robust fallback: when fit() raises after
  nlmixr has completed, it parses run1.RDATA directly via pyreadr.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any


def _write(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _has_mod(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _extract_recovery_from_param_est(
    param_est: Any, *, true_theta: list[float], true_omega: list[float]
) -> dict[str, Any] | None:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None

    if isinstance(param_est, pd.DataFrame):
        if param_est.shape[1] < 1:
            return None
        param_est = param_est.iloc[:, 0]
    if not isinstance(param_est, pd.Series):
        return None

    def _get(*cands: str) -> float | None:
        for c in cands:
            if c in param_est.index:
                try:
                    return float(param_est[c])
                except Exception:
                    return None
        return None

    cl = _get("THETA_1", "THETA1", "THETA(1)", "POP_CL", "tcl")
    v = _get("THETA_2", "THETA2", "THETA(2)", "POP_V", "tv")
    ka = _get("THETA_3", "THETA3", "THETA(3)", "POP_KA", "tka")
    w_cl = _get("OMEGA_1_1", "OMEGA(1,1)", "OMEGA(1)", "IIV_CL")
    w_v = _get("OMEGA_2_2", "OMEGA(2,2)", "OMEGA(2)", "IIV_V")
    w_ka = _get("OMEGA_3_3", "OMEGA(3,3)", "OMEGA(3)", "IIV_KA")
    if None in (cl, v, ka, w_cl, w_v, w_ka):
        return None

    assert cl is not None and v is not None and ka is not None
    assert w_cl is not None and w_v is not None and w_ka is not None

    w_cl_sd = math.sqrt(max(w_cl, 0.0))
    w_v_sd = math.sqrt(max(w_v, 0.0))
    w_ka_sd = math.sqrt(max(w_ka, 0.0))

    return {
        "CL": {
            "hat": cl,
            "true": float(true_theta[0]),
            "rel_err": abs(cl - float(true_theta[0])) / abs(float(true_theta[0])),
        },
        "V": {
            "hat": v,
            "true": float(true_theta[1]),
            "rel_err": abs(v - float(true_theta[1])) / abs(float(true_theta[1])),
        },
        "Ka": {
            "hat": ka,
            "true": float(true_theta[2]),
            "rel_err": abs(ka - float(true_theta[2])) / abs(float(true_theta[2])),
        },
        "w_CL": {
            "hat": w_cl_sd,
            "true": float(true_omega[0]),
            "rel_err": abs(w_cl_sd - float(true_omega[0])) / abs(float(true_omega[0])),
        },
        "w_V": {
            "hat": w_v_sd,
            "true": float(true_omega[1]),
            "rel_err": abs(w_v_sd - float(true_omega[1])) / abs(float(true_omega[1])),
        },
        "w_Ka": {
            "hat": w_ka_sd,
            "true": float(true_omega[2]),
            "rel_err": abs(w_ka_sd - float(true_omega[2])) / abs(float(true_omega[2])),
        },
    }


def _extract_from_rdata(
    run_name: str, *, true_theta: list[float], true_omega: list[float]
) -> tuple[dict[str, Any] | None, float]:
    try:
        import pyreadr  # type: ignore
    except Exception:
        return None, float("nan")

    run_dir = Path(run_name)
    rdata_files = list(run_dir.glob(".modeldb/*/run1.RDATA"))
    if not rdata_files:
        return None, float("nan")

    try:
        res = pyreadr.read_r(str(rdata_files[0]))
    except Exception:
        return None, float("nan")

    thetas = res.get("thetas")
    omega = res.get("omega")
    ofv_df = res.get("ofv")
    if thetas is None or omega is None:
        return None, float("nan")

    try:
        theta_map = {str(k): float(v) for k, v in zip(thetas.index.tolist(), thetas.iloc[:, 0].tolist())}
        cl = theta_map.get("THETA_1")
        v = theta_map.get("THETA_2")
        ka = theta_map.get("THETA_3")
        if cl is None or v is None or ka is None:
            return None, float("nan")
        w_cl = math.sqrt(max(float(omega.iloc[0, 0]), 0.0))
        w_v = math.sqrt(max(float(omega.iloc[1, 1]), 0.0))
        w_ka = math.sqrt(max(float(omega.iloc[2, 2]), 0.0))
    except Exception:
        return None, float("nan")

    recovery = {
        "CL": {"hat": cl, "true": float(true_theta[0]), "rel_err": abs(cl - float(true_theta[0])) / abs(float(true_theta[0]))},
        "V": {"hat": v, "true": float(true_theta[1]), "rel_err": abs(v - float(true_theta[1])) / abs(float(true_theta[1]))},
        "Ka": {"hat": ka, "true": float(true_theta[2]), "rel_err": abs(ka - float(true_theta[2])) / abs(float(true_theta[2]))},
        "w_CL": {"hat": w_cl, "true": float(true_omega[0]), "rel_err": abs(w_cl - float(true_omega[0])) / abs(float(true_omega[0]))},
        "w_V": {"hat": w_v, "true": float(true_omega[1]), "rel_err": abs(w_v - float(true_omega[1])) / abs(float(true_omega[1]))},
        "w_Ka": {"hat": w_ka, "true": float(true_omega[2]), "rel_err": abs(w_ka - float(true_omega[2])) / abs(float(true_omega[2]))},
    }

    objective = float("nan")
    try:
        if ofv_df is not None and len(ofv_df.index) > 0:
            objective = float(ofv_df.iloc[0, 0])
    except Exception:
        objective = float("nan")
    return recovery, objective


def _cleanup_run_dir(name: str) -> None:
    p = Path(name)
    if p.exists() and p.is_dir():
        shutil.rmtree(p, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    out_path = Path(args.out_path)
    case_id = "unknown"
    try:
        case_obj = json.loads(Path(args.in_path).read_text())
        case_id = str(case_obj.get("case", "unknown"))
    except Exception:
        pass

    if not _has_mod("pharmpy"):
        _write(
            out_path,
            {
                "schema_version": "nextstat.pharma_baseline_result.v1",
                "baseline": "pharmpy",
                "case": case_id,
                "status": "skipped",
                "reason": "pharmpy not installed",
            },
        )
        return 0

    try:
        case_obj = json.loads(Path(args.in_path).read_text())
    except Exception as e:
        _write(
            out_path,
            {
                "schema_version": "nextstat.pharma_baseline_result.v1",
                "baseline": "pharmpy",
                "case": case_id,
                "status": "failed",
                "reason": f"json_parse_error:{type(e).__name__}:{e}",
            },
        )
        return 0

    spec = ((case_obj.get("dataset") or {}).get("spec") or {})
    if not isinstance(spec, dict):
        _write(
            out_path,
            {
                "schema_version": "nextstat.pharma_baseline_result.v1",
                "baseline": "pharmpy",
                "case": case_id,
                "status": "failed",
                "reason": "input case JSON missing dataset.spec",
            },
        )
        return 0

    kind = str(spec.get("kind", ""))
    em = str(spec.get("error_model", ""))
    if kind != "pop_pk_1c_oral":
        _write(
            out_path,
            {
                "schema_version": "nextstat.pharma_baseline_result.v1",
                "baseline": "pharmpy",
                "case": case_id,
                "status": "skipped",
                "reason": f"unsupported dataset kind: {kind}",
            },
        )
        return 0
    if em != "additive":
        _write(
            out_path,
            {
                "schema_version": "nextstat.pharma_baseline_result.v1",
                "baseline": "pharmpy",
                "case": case_id,
                "status": "skipped",
                "reason": f"unsupported error_model for baseline runner: {em}",
            },
        )
        return 0

    os.environ.setdefault("PHARMPYNOCONFIGFILE", "1")
    os.environ.setdefault("R_LIBS_USER", str(Path(__file__).resolve().parents[6] / ".r_libs"))

    try:
        import pandas as pd  # type: ignore
        from pharmpy.modeling import read_model_from_string  # type: ignore
        from pharmpy.tools import fit  # type: ignore
        import pharmpy  # type: ignore
    except Exception as e:
        _write(
            out_path,
            {
                "schema_version": "nextstat.pharma_baseline_result.v1",
                "baseline": "pharmpy",
                "case": case_id,
                "status": "failed",
                "reason": f"import_error:{type(e).__name__}:{e}",
            },
        )
        return 0

    try:
        n_sub = int(spec["n_subjects"])
        ids = [int(x) + 1 for x in spec["subject_idx"]]
        times = [float(x) for x in spec["times"]]
        y = [float(x) for x in spec["y"]]
        dose = float(spec["dose"])
        true_theta = [float(x) for x in spec["true_theta"]]
        true_omega = [float(x) for x in spec["true_omega"]]
    except Exception as e:
        _write(
            out_path,
            {
                "schema_version": "nextstat.pharma_baseline_result.v1",
                "baseline": "pharmpy",
                "case": case_id,
                "status": "failed",
                "reason": f"invalid_spec:{type(e).__name__}:{e}",
                "packages": {"pharmpy": getattr(pharmpy, "__version__", "unknown")},
            },
        )
        return 0

    runs: list[float] = []
    run_notes: list[str] = []
    last_objective = float("nan")
    recovery: dict[str, Any] | None = None
    run_tag = f"{case_id}_{os.getpid()}_{time.time_ns()}"

    with tempfile.TemporaryDirectory(prefix="pharmpy-pharma-") as td:
        data_path = Path(td) / "data.csv"
        df_obs = pd.DataFrame(
            {
                "ID": ids,
                "TIME": times,
                "DV": y,
                "AMT": [0.0] * len(y),
                "EVID": [0] * len(y),
                "CMT": [2] * len(y),
            }
        )
        df_dose = pd.DataFrame(
            {
                "ID": list(range(1, n_sub + 1)),
                "TIME": [0.0] * n_sub,
                "DV": [float("nan")] * n_sub,
                "AMT": [dose] * n_sub,
                "EVID": [1] * n_sub,
                "CMT": [1] * n_sub,
            }
        )
        df = pd.concat([df_dose, df_obs], ignore_index=True).sort_values(
            ["ID", "TIME", "EVID"], ascending=[True, True, False]
        )
        df.to_csv(data_path, index=False)

        # "A = F; Y = A + EPS(1)" is used instead of "Y = F + EPS(1)" to avoid
        # a known pharmpy nlmixr conversion bug ("No resulting term found").
        model_code = (
            "$PROBLEM NextStat pharma baseline via pharmpy\n"
            f"$DATA {data_path} IGNORE=@\n"
            "$INPUT ID TIME DV AMT EVID CMT\n"
            "$SUBROUTINE ADVAN2 TRANS2\n"
            "$PK\n"
            "CL = THETA(1)*EXP(ETA(1))\n"
            "V  = THETA(2)*EXP(ETA(2))\n"
            "KA = THETA(3)*EXP(ETA(3))\n"
            "S2 = V\n"
            "$ERROR\n"
            "A = F\n"
            "Y = A + EPS(1)\n"
            "$THETA (0,0.13)\n"
            "$THETA (0,8.0)\n"
            "$THETA (0,1.0)\n"
            "$OMEGA 0.04\n"
            "$OMEGA 0.0225\n"
            "$OMEGA 0.0625\n"
            "$SIGMA 0.09\n"
            "$ESTIMATION METHOD=1 INTERACTION MAXEVALS=9999\n"
        )
        model = read_model_from_string(model_code)

        warm_name = f"pharmpy_warm_{run_tag}"
        try:
            _ = fit(model, esttool="nlmixr", name=warm_name)
        except Exception as e:
            rec_fb, obj_fb = _extract_from_rdata(
                warm_name, true_theta=true_theta, true_omega=true_omega
            )
            if rec_fb is None:
                _write(
                    out_path,
                    {
                        "schema_version": "nextstat.pharma_baseline_result.v1",
                        "baseline": "pharmpy",
                        "case": case_id,
                        "status": "failed",
                        "reason": f"fit_error:warmup:{type(e).__name__}:{e}",
                        "packages": {"pharmpy": getattr(pharmpy, "__version__", "unknown")},
                    },
                )
                return 0
            run_notes.append(f"warmup_fallback:{type(e).__name__}")
            recovery = rec_fb
            last_objective = obj_fb
        finally:
            _cleanup_run_dir(warm_name)

        for i in range(max(1, int(args.repeat))):
            run_name = f"pharmpy_{run_tag}_{i}"
            t0 = time.perf_counter()
            run_exc: Exception | None = None
            result: Any = None
            try:
                result = fit(model, esttool="nlmixr", name=run_name)
            except Exception as e:
                run_exc = e
            dt = time.perf_counter() - t0
            runs.append(dt)

            if run_exc is None:
                param_est = getattr(result, "parameter_estimates", None)
                rec = _extract_recovery_from_param_est(
                    param_est, true_theta=true_theta, true_omega=true_omega
                )
                if rec is None:
                    rec_fb, obj_fb = _extract_from_rdata(
                        run_name, true_theta=true_theta, true_omega=true_omega
                    )
                    if rec_fb is None:
                        _write(
                            out_path,
                            {
                                "schema_version": "nextstat.pharma_baseline_result.v1",
                                "baseline": "pharmpy",
                                "case": case_id,
                                "status": "failed",
                                "reason": "fit_error:unable_to_extract_parameter_estimates",
                                "packages": {"pharmpy": getattr(pharmpy, "__version__", "unknown")},
                            },
                        )
                        return 0
                    run_notes.append("timed_fallback:missing_param_est")
                    recovery = rec_fb
                    last_objective = obj_fb
                else:
                    recovery = rec
                    try:
                        last_objective = float(getattr(result, "ofv", float("nan")))
                    except Exception:
                        last_objective = float("nan")
            else:
                rec_fb, obj_fb = _extract_from_rdata(
                    run_name, true_theta=true_theta, true_omega=true_omega
                )
                if rec_fb is None:
                    _write(
                        out_path,
                        {
                            "schema_version": "nextstat.pharma_baseline_result.v1",
                            "baseline": "pharmpy",
                            "case": case_id,
                            "status": "failed",
                            "reason": f"fit_error:{type(run_exc).__name__}:{run_exc}",
                            "packages": {"pharmpy": getattr(pharmpy, "__version__", "unknown")},
                        },
                    )
                    return 0
                run_notes.append(f"timed_fallback:{type(run_exc).__name__}")
                recovery = rec_fb
                last_objective = obj_fb
            _cleanup_run_dir(run_name)

    if not runs:
        _write(
            out_path,
            {
                "schema_version": "nextstat.pharma_baseline_result.v1",
                "baseline": "pharmpy",
                "case": case_id,
                "status": "failed",
                "reason": "fit_error:no_timed_runs",
                "packages": {"pharmpy": getattr(pharmpy, "__version__", "unknown")},
            },
        )
        return 0

    if recovery is None:
        _write(
            out_path,
            {
                "schema_version": "nextstat.pharma_baseline_result.v1",
                "baseline": "pharmpy",
                "case": case_id,
                "status": "failed",
                "reason": "fit_error:no_recovery_parsed",
                "packages": {"pharmpy": getattr(pharmpy, "__version__", "unknown")},
            },
        )
        return 0

    _write(
        out_path,
        {
            "schema_version": "nextstat.pharma_baseline_result.v1",
            "baseline": "pharmpy",
            "case": case_id,
            "status": "ok",
            "timing": {
                "fit_time_s": float(min(runs)),
                "raw": {
                    "repeat_n": int(args.repeat),
                    "policy": "min",
                    "per_fit_s": [float(x) for x in runs],
                },
            },
            "meta": {
                "method": "pharmpy+nlmixr",
                "objective": last_objective,
                "runner_notes": run_notes,
            },
            "packages": {"pharmpy": getattr(pharmpy, "__version__", "unknown")},
            "recovery": recovery,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
