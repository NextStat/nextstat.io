#!/usr/bin/env python3
"""Report generator for the pharma suite results (v2).

Handles both individual PK (NLL timing + MLE fit) and population PK (FOCE/SAEM).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    pop_rows: list[tuple] = []

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
                if rec:
                    theta_errs = {k: v for k, v in rec.items() if not k.startswith("w_")}
                    omega_errs = {k: v for k, v in rec.items() if k.startswith("w_")}
                    max_theta_rel = max((v.get("rel_err", 0.0) for v in theta_errs.values()), default=0.0)
                    max_omega_rel = max((v.get("rel_err", 0.0) for v in omega_errs.values()), default=0.0)
                else:
                    max_theta_rel = max_omega_rel = 0.0
                pop_rows.append((case_id, estimator, n_sub, n_obs, fit_t, ofv, converged,
                                 max_theta_rel, max_omega_rel))
            else:
                status = result.get("status", "failed") if isinstance(result, dict) else "failed"
                pop_rows.append((case_id, estimator, n_sub, n_obs, 0.0, 0.0, False, 0.0, 0.0))

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
        for case, est, nsub, nobs, fit_t, ofv, conv, t_err, o_err in pop_rows:
            lines.append(
                f"| `{case}` | {est} | {nsub} | {nobs} | {fit_t:.4f}s | {ofv:.2f} | "
                f"{'Y' if conv else 'N'} | {t_err:.1%} | {o_err:.1%} |"
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
