#!/usr/bin/env python3
"""Report generator for the pharma suite results (v2).

Handles both individual PK (NLL timing + MLE fit) and population PK (FOCE/SAEM).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _max_rel_errs(recovery: dict) -> tuple[float, float]:
    theta_max = 0.0
    omega_max = 0.0
    for k, v in recovery.items():
        if not isinstance(v, dict):
            continue
        rel = float(v.get("rel_err", 0.0))
        key = str(k).lower()
        if key.startswith("w_") or key.startswith("omega"):
            omega_max = max(omega_max, rel)
        else:
            theta_max = max(theta_max, rel)
    return theta_max, omega_max


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="out/pharma/pharma_suite.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    suite_path = Path(args.suite).resolve()
    out_dir = suite_path.parent
    suite = json.loads(suite_path.read_text())

    # Separate individual PK and population PK cases
    pk_rows: list[tuple] = []
    pop_rows: list[dict] = []

    for e in suite.get("cases", []):
        inst = json.loads((out_dir / e["path"]).read_text())
        model = inst.get("model", {})
        kind = str(model.get("kind", ""))
        case_id = str(inst.get("case", e.get("case", "")))
        n_sub = int(model.get("n_subjects", 0))
        n_obs = int(model.get("n_obs", 0))
        n_par = int(model.get("n_params", 0))

        # Individual PK / NLME LogDensityModel
        if kind in ("pk_1c_oral", "nlme_pk_1c_oral", "pk_2cpt_iv", "pk_2cpt_oral"):
            timing = (inst.get("timing", {}) or {}).get("nll_time_s_per_call", {}) or {}
            nll_t = float(timing.get("nextstat", 0.0))
            fit = inst.get("fit") or {}
            fit_status = "-"
            fit_t = 0.0
            recovery_txt = "-"
            if isinstance(fit, dict):
                fit_status = str(fit.get("status", "-"))
                if fit_status == "ok":
                    fit_t = float((fit.get("time_s", {}) or {}).get("nextstat", 0.0))
                    rec = fit.get("recovery", {})
                    if rec:
                        max_rel = max(v.get("rel_err", 0.0) for v in rec.values())
                        recovery_txt = f"{max_rel:.1%}"
            pk_rows.append((case_id, n_sub, n_obs, n_par, nll_t, fit_status, fit_t, recovery_txt))

        # Population PK (FOCE / SAEM)
        elif kind == "pop_pk_1c_oral":
            foce = inst.get("foce") or {}
            saem = inst.get("saem") or {}

            estimator = "FOCE" if foce else ("SAEM" if saem else "?")
            result = foce if foce else saem

            if isinstance(result, dict) and result.get("status") == "ok":
                fit_t = float((result.get("time_s", {}) or {}).get("nextstat", 0.0))
                meta = result.get("meta", {})
                ofv = float(meta.get("ofv", float("nan")))
                converged = bool(meta.get("converged", False))
                rec = result.get("recovery", {})
                max_theta_rel, max_omega_rel = _max_rel_errs(rec if isinstance(rec, dict) else {})
                repeat_n = int(((result.get("raw", {}) or {}).get("repeat", 0)) if isinstance(result, dict) else 0)
                pop_rows.append({
                    "case": case_id,
                    "method": estimator,
                    "n_sub": n_sub,
                    "n_obs": n_obs,
                    "nextstat_time_s": fit_t,
                    "ofv": ofv,
                    "converged": converged,
                    "nextstat_theta_err": max_theta_rel,
                    "nextstat_omega_err": max_omega_rel,
                    "nextstat_repeat_n": repeat_n,
                    "status": "ok",
                })
            else:
                status = result.get("status", "failed") if isinstance(result, dict) else "failed"
                pop_rows.append({
                    "case": case_id,
                    "method": estimator,
                    "n_sub": n_sub,
                    "n_obs": n_obs,
                    "nextstat_time_s": 0.0,
                    "ofv": 0.0,
                    "converged": False,
                    "nextstat_theta_err": 0.0,
                    "nextstat_omega_err": 0.0,
                    "nextstat_repeat_n": 0,
                    "status": str(status),
                })

    baselines_by_case: dict[str, list[dict]] = {}
    baseline_dir = out_dir / "baselines"
    if baseline_dir.exists():
        for p in sorted(baseline_dir.glob("*.json")):
            try:
                b = json.loads(p.read_text())
            except Exception:
                continue
            baseline_name = str(b.get("baseline", "")).strip()
            case_id = str(b.get("case", "")).strip()
            if not baseline_name or not case_id:
                continue
            item = {
                "baseline": baseline_name,
                "status": str(b.get("status", "unknown")),
            }
            if item["status"] != "ok":
                baselines_by_case.setdefault(case_id, []).append(item)
                continue
            timing = b.get("timing", {}) or {}
            raw = timing.get("raw", {}) or {}
            rec = b.get("recovery", {}) or {}
            theta_err, omega_err = _max_rel_errs(rec if isinstance(rec, dict) else {})
            item["fit_time_s"] = float(timing.get("fit_time_s", 0.0))
            item["repeat_n"] = int(raw.get("repeat_n", raw.get("repeat", 0)))
            item["theta_err"] = theta_err
            item["omega_err"] = omega_err
            baselines_by_case.setdefault(case_id, []).append(item)

    lines: list[str] = []

    # Table 1: Individual PK models
    if pk_rows:
        lines.append("## Individual PK / NLME NLL Timing\n")
        lines.append("| Case | Subj | Obs | Params | NLL/call | Fit | Fit time | Max rel err |")
        lines.append("|---|---:|---:|---:|---:|---|---:|---:|")
        for case, nsub, nobs, npar, nll_t, fit_st, fit_t, rec in pk_rows:
            fit_t_txt = "-" if fit_st != "ok" else f"{fit_t:.4f}s"
            lines.append(f"| `{case}` | {nsub} | {nobs} | {npar} | {nll_t:.6f}s | {fit_st} | {fit_t_txt} | {rec} |")
        lines.append("")

    # Table 2: Population PK (FOCE / SAEM)
    if pop_rows:
        lines.append("## Population PK (FOCE / SAEM)\n")
        lines.append("| Case | Method | Subj | Obs | Fit time | OFV | Conv | Max θ err | Max ω err |")
        lines.append("|---|---|---:|---:|---:|---:|---|---:|---:|")
        for row in pop_rows:
            case = str(row["case"])
            est = str(row["method"])
            nsub = int(row["n_sub"])
            nobs = int(row["n_obs"])
            fit_t = float(row["nextstat_time_s"])
            ofv = float(row["ofv"])
            conv = bool(row["converged"])
            t_err = float(row["nextstat_theta_err"])
            o_err = float(row["nextstat_omega_err"])
            lines.append(
                f"| `{case}` | {est} | {nsub} | {nobs} | {fit_t:.4f}s | {ofv:.2f} | "
                f"{'Y' if conv else 'N'} | {t_err:.1%} | {o_err:.1%} |"
            )
        lines.append("")

    if pop_rows and baselines_by_case:
        lines.append("## NextStat vs External Baselines (Population PK)\n")
        lines.append("| Case | Method | Baseline | NextStat | Baseline | Speedup | NextStat max θ/ω | Baseline max θ/ω | Repeats (N/B) |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for row in pop_rows:
            case = str(row["case"])
            method = str(row["method"])
            ns_t = float(row["nextstat_time_s"])
            ns_theta = float(row["nextstat_theta_err"])
            ns_omega = float(row["nextstat_omega_err"])
            ns_rep = int(row["nextstat_repeat_n"])
            ext = baselines_by_case.get(case, [])
            if not ext:
                lines.append(
                    f"| `{case}` | {method} | - | {ns_t:.4f}s | - | - | {ns_theta:.1%}/{ns_omega:.1%} | - | {ns_rep}/- |"
                )
                continue
            for b in sorted(ext, key=lambda x: str(x.get("baseline", ""))):
                bname = str(b.get("baseline", "external"))
                if str(b.get("status", "")) != "ok":
                    lines.append(
                        f"| `{case}` | {method} | {bname} | {ns_t:.4f}s | {str(b.get('status'))} | - | "
                        f"{ns_theta:.1%}/{ns_omega:.1%} | - | {ns_rep}/- |"
                    )
                    continue
                b_t = float(b.get("fit_time_s", 0.0))
                b_theta = float(b.get("theta_err", 0.0))
                b_omega = float(b.get("omega_err", 0.0))
                b_rep = int(b.get("repeat_n", 0))
                speedup = (b_t / ns_t) if ns_t > 0 else 0.0
                lines.append(
                    f"| `{case}` | {method} | {bname} | {ns_t:.4f}s | {b_t:.4f}s | {speedup:.1f}x | "
                    f"{ns_theta:.1%}/{ns_omega:.1%} | {b_theta:.1%}/{b_omega:.1%} | {ns_rep}/{b_rep} |"
                )
        lines.append("")

    txt = "\n".join(lines) + "\n"
    if str(args.out).strip():
        Path(args.out).resolve().write_text(txt)
    else:
        print(txt, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
