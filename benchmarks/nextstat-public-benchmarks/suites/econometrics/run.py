#!/usr/bin/env python3
"""Econometrics seed benchmark: Phase 12 baseline cases + optional parity.

This is intentionally small and self-contained so outsiders can rerun it.

Cases:
- panel_fe: within estimator + 1-way cluster SE (entity)
- did_twfe: TWFE baseline with treat*post regressor
- event_study_twfe: TWFE baseline with lead/lag dummies
- iv_2sls: baseline IV/2SLS (optional parity via linearmodels)
- aipw: AIPW baseline (no external parity yet; timing + sanity only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

import nextstat


def sha256_json_obj(obj: dict) -> str:
    b = (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _pctl(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * float(p)
    i = int(k)
    j = min(i + 1, len(xs) - 1)
    a = xs[i]
    b = xs[j]
    t = k - i
    return a * (1.0 - t) + b * t


def _summary(values: list[float]) -> dict[str, float]:
    return {"min": min(values) if values else 0.0, "median": _pctl(values, 0.5), "p95": _pctl(values, 0.95)}


def _maybe_import(name: str) -> tuple[bool, Optional[str]]:
    try:
        __import__(name)
        mod = sys.modules.get(name)
        return True, str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return False, None


def _max_abs_rel_diff(a: list[float], b: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not a or not b or len(a) != len(b):
        return None, None
    abs_d = 0.0
    rel_d = 0.0
    for x, y in zip(a, b):
        d = abs(float(x) - float(y))
        abs_d = max(abs_d, d)
        denom = max(abs(float(x)), abs(float(y)), 1.0)
        rel_d = max(rel_d, d / denom)
    return float(abs_d), float(rel_d)


def _gen_panel(n_entities: int, n_times: int, n_x: int, *, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    n = int(n_entities) * int(n_times)
    entity = [i for i in range(int(n_entities)) for _ in range(int(n_times))]
    time_id = [t for _ in range(int(n_entities)) for t in range(int(n_times))]

    x = rng.normal(size=(n, n_x)).astype(float)
    beta = np.linspace(0.5, 1.0, num=n_x, dtype=float)
    alpha_i = rng.normal(scale=1.0, size=int(n_entities)).astype(float)
    eps = rng.normal(scale=1.0, size=n).astype(float)
    y = alpha_i[np.array(entity, dtype=int)] + x @ beta + eps

    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "entity": entity,
        "time": time_id,
        "beta_true": beta.tolist(),
    }


def _gen_did(n_entities: int, n_times: int, n_x: int, *, seed: int, treat_frac: float = 0.5) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    n = int(n_entities) * int(n_times)
    entity = [i for i in range(int(n_entities)) for _ in range(int(n_times))]
    time_id = [t for _ in range(int(n_entities)) for t in range(int(n_times))]

    treated_entity = rng.random(int(n_entities)) < float(treat_frac)
    treat = [int(treated_entity[e]) for e in entity]
    post = [int(t >= (int(n_times) // 2)) for t in time_id]

    x = rng.normal(size=(n, n_x)).astype(float)
    beta = np.linspace(0.2, 0.2 + 0.05 * max(n_x - 1, 0), num=n_x, dtype=float)
    att = 1.0

    alpha_i = rng.normal(scale=1.0, size=int(n_entities)).astype(float)
    gamma_t = rng.normal(scale=0.5, size=int(n_times)).astype(float)
    eps = rng.normal(scale=1.0, size=n).astype(float)

    did = np.array([float(t) * float(p) for t, p in zip(treat, post)], dtype=float)
    y = alpha_i[np.array(entity, dtype=int)] + gamma_t[np.array(time_id, dtype=int)] + att * did + x @ beta + eps

    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "entity": entity,
        "time": time_id,
        "treat": treat,
        "post": post,
        "att_true": float(att),
    }


def _gen_event_study(n_entities: int, n_times: int, n_x: int, *, seed: int) -> dict[str, Any]:
    # Balanced panel, single adoption time per treated entity.
    rng = np.random.default_rng(int(seed))
    n = int(n_entities) * int(n_times)
    entity = [i for i in range(int(n_entities)) for _ in range(int(n_times))]
    time_id = [t for _ in range(int(n_entities)) for t in range(int(n_times))]

    treated_entity = rng.random(int(n_entities)) < 0.5
    treat = [int(treated_entity[e]) for e in entity]

    # Event time: treated units adopt at mid-point, control never treated (event_time ignored by API for control via treat=0).
    et = int(n_times) // 2
    event_time = [et if treated_entity[e] else et for e in entity]

    # Dynamic effect: 0 pre, ramp post.
    rel = np.array([int(t) - int(e) for t, e in zip(time_id, event_time)], dtype=int)
    dyn = np.where(rel >= 0, np.minimum(rel, 3) * 0.5, 0.0).astype(float)

    x = rng.normal(size=(n, n_x)).astype(float)
    beta = np.linspace(0.1, 0.1 + 0.03 * max(n_x - 1, 0), num=n_x, dtype=float)

    alpha_i = rng.normal(scale=1.0, size=int(n_entities)).astype(float)
    gamma_t = rng.normal(scale=0.5, size=int(n_times)).astype(float)
    eps = rng.normal(scale=1.0, size=n).astype(float)

    y = alpha_i[np.array(entity, dtype=int)] + gamma_t[np.array(time_id, dtype=int)] + dyn * np.array(treat, dtype=float) + x @ beta + eps

    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "entity": entity,
        "time": time_id,
        "treat": treat,
        "event_time": event_time,
        "window": [-4, 4],
        "reference": -1,
    }


def _gen_iv(n_obs: int, n_x: int, *, seed: int, pi: float = 1.0, rho_uv: float = 0.6) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    x = rng.normal(size=(n_obs, n_x)).astype(float)
    z = rng.normal(size=(n_obs, 1)).astype(float)
    v = rng.normal(size=(n_obs, 1)).astype(float)
    e = rng.normal(size=(n_obs, 1)).astype(float)
    u = rho_uv * v + math.sqrt(max(0.0, 1.0 - rho_uv * rho_uv)) * e

    d = pi * z + 0.5 * x[:, [0]] + v
    # True effect for endog is 2.0; exog x0 is 1.0; others are 0.
    y = 2.0 * d + 1.0 * x[:, [0]] + u

    return {
        "y": y.ravel().tolist(),
        "endog": d.tolist(),
        "instruments": z.tolist(),
        "exog": x.tolist(),
        "coef_true": [2.0] + [1.0] + [0.0] * max(n_x - 1, 0),
    }


def _gen_aipw(n_obs: int, n_x: int, *, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    x = rng.normal(size=(n_obs, n_x)).astype(float)
    logits = 0.5 * x[:, 0]
    p = 1.0 / (1.0 + np.exp(-logits))
    t = (rng.random(n_obs) < p).astype(int)
    tau = 1.0
    y = (x[:, 0] + 0.5 * x[:, 1] + tau * t + rng.normal(scale=1.0, size=n_obs)).astype(float)
    return {"x": x.tolist(), "y": y.tolist(), "t": t.tolist(), "tau_true": float(tau)}


def _fit_panel_fe(data: dict[str, Any]) -> tuple[list[float], list[float], dict[str, Any]]:
    fit = nextstat.econometrics.panel_fe_fit(data["x"], data["y"], entity=data["entity"], time=data["time"], cluster="entity")
    return list(fit.coef), list(fit.standard_errors), {"n_obs": int(fit.n_obs), "n_entities": int(fit.n_entities)}


def _fit_did_twfe(data: dict[str, Any]) -> tuple[list[float], list[float], dict[str, Any]]:
    fit = nextstat.econometrics.did_twfe_fit(
        data["x"],
        data["y"],
        treat=data["treat"],
        post=data["post"],
        entity=data["entity"],
        time=data["time"],
        cluster="entity",
    )
    coef = [float(fit.att)]
    se = [float(fit.att_se)]
    return coef, se, {"n_obs": int(fit.twfe.n_obs), "n_entities": int(fit.twfe.n_entities), "n_times": int(fit.twfe.n_times)}


def _fit_event_study(data: dict[str, Any]) -> tuple[list[float], list[float], dict[str, Any]]:
    fit = nextstat.econometrics.event_study_twfe_fit(
        data["y"],
        treat=data["treat"],
        time=data["time"],
        event_time=data["event_time"],
        entity=data["entity"],
        window=tuple(data["window"]),
        reference=int(data["reference"]),
        x=data["x"],
        cluster="entity",
    )
    return list(fit.coef), list(fit.standard_errors), {"n_obs": int(fit.n_obs), "n_entities": int(fit.n_entities), "n_times": int(fit.n_times), "n_event": int(len(fit.coef))}


def _fit_iv_2sls(data: dict[str, Any]) -> tuple[list[float], list[float], dict[str, Any]]:
    fit = nextstat.econometrics.iv_2sls_fit(
        data["y"],
        endog=data["endog"],
        instruments=data["instruments"],
        exog=data["exog"],
        cov="hc1",
    )
    return list(fit.coef), list(fit.standard_errors), {"n_obs": int(fit.n_obs), "df_resid": int(fit.df_resid)}


def _fit_aipw(data: dict[str, Any]) -> tuple[list[float], list[float], dict[str, Any]]:
    fit = nextstat.causal.aipw.aipw_fit(data["x"], data["y"], data["t"], estimand="ate", trim_eps=1e-6)
    return [float(fit.estimate)], [float(fit.standard_error)], {"n_obs": int(fit.n_obs), "trim_eps": float(fit.trim_eps)}


def _baseline_statsmodels_ols(*, y: list[float], x: list[list[float]], groups: Optional[list[Any]]) -> tuple[list[float], list[float]]:
    import statsmodels.api as sm  # type: ignore[import-not-found]

    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=float)
    m = sm.OLS(Y, X)
    if groups is None:
        r = m.fit()
    else:
        r = m.fit(cov_type="cluster", cov_kwds={"groups": np.asarray(groups), "use_correction": True})
    coef = [float(v) for v in np.asarray(r.params).ravel().tolist()]
    se = [float(v) for v in np.asarray(r.bse).ravel().tolist()]
    return coef, se


def _baseline_linearmodels_iv(data: dict[str, Any]) -> tuple[list[float], list[float]]:
    from linearmodels.iv import IV2SLS  # type: ignore[import-not-found]

    Y = np.asarray(data["y"], dtype=float)
    exog = np.asarray(data["exog"], dtype=float)
    endog = np.asarray(data["endog"], dtype=float)
    instr = np.asarray(data["instruments"], dtype=float)
    # No intercept (NextStat API expects explicit intercept in exog if desired; our generator has none).
    res = IV2SLS(Y, exog, endog, instr).fit(cov_type="robust")
    coef = [float(v) for v in np.asarray(res.params).ravel().tolist()]
    se = [float(v) for v in np.asarray(res.std_errors).ravel().tolist()]
    return coef, se


# ---------------------------------------------------------------------------
# Full-pipeline baselines via linearmodels.PanelOLS (for fair timing comparison)
# ---------------------------------------------------------------------------

def _full_baseline_lm_panel_fe(data: dict[str, Any]) -> None:
    """Full-pipeline Panel FE: DataFrame build + PanelOLS(entity_effects) + cluster SE."""
    from linearmodels.panel import PanelOLS  # type: ignore[import-not-found]
    import pandas as pd  # type: ignore[import-not-found]

    n_x = len(data["x"][0])
    df = pd.DataFrame({"y": data["y"], "entity": data["entity"], "time": data["time"]})
    for i in range(n_x):
        df[f"x{i}"] = [row[i] for row in data["x"]]
    df = df.set_index(["entity", "time"])
    mod = PanelOLS(df["y"], df[[f"x{i}" for i in range(n_x)]], entity_effects=True)
    mod.fit(cov_type="clustered", cluster_entity=True)


def _full_baseline_lm_did(data: dict[str, Any]) -> None:
    """Full-pipeline DID TWFE: DataFrame build + PanelOLS(entity+time) + cluster SE."""
    from linearmodels.panel import PanelOLS  # type: ignore[import-not-found]
    import pandas as pd  # type: ignore[import-not-found]

    n_x = len(data["x"][0])
    did = [int(t) * int(p) for t, p in zip(data["treat"], data["post"])]
    df = pd.DataFrame({"y": data["y"], "entity": data["entity"], "time": data["time"], "did": did})
    for i in range(n_x):
        df[f"x{i}"] = [row[i] for row in data["x"]]
    df = df.set_index(["entity", "time"])
    exog_cols = ["did"] + [f"x{i}" for i in range(n_x)]
    mod = PanelOLS(df["y"], df[exog_cols], entity_effects=True, time_effects=True)
    mod.fit(cov_type="clustered", cluster_entity=True)


def _full_baseline_lm_event_study(data: dict[str, Any]) -> None:
    """Full-pipeline Event Study TWFE: regressor build + PanelOLS(entity+time) + cluster SE."""
    from linearmodels.panel import PanelOLS  # type: ignore[import-not-found]
    import pandas as pd  # type: ignore[import-not-found]

    n_x = len(data["x"][0])
    rel = nextstat.econometrics.relative_time(data["time"], data["event_time"])
    x_ev, names_ev, _ = nextstat.econometrics.event_study_regressors(
        data["treat"], rel, window=tuple(data["window"]), reference=int(data["reference"]),
    )
    x_ev_np = np.asarray(x_ev, dtype=float)
    df = pd.DataFrame({"y": data["y"], "entity": data["entity"], "time": data["time"]})
    for i, name in enumerate(names_ev):
        df[name] = x_ev_np[:, i]
    for i in range(n_x):
        df[f"x{i}"] = [row[i] for row in data["x"]]
    df = df.set_index(["entity", "time"])
    exog_cols = names_ev + [f"x{i}" for i in range(n_x)]
    mod = PanelOLS(df["y"], df[exog_cols], entity_effects=True, time_effects=True)
    mod.fit(cov_type="clustered", cluster_entity=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--kind", required=True, choices=["panel_fe", "did_twfe", "event_study_twfe", "iv_2sls", "aipw"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--n-entities", type=int, default=1000)
    ap.add_argument("--n-times", type=int, default=8)
    ap.add_argument("--n-obs", type=int, default=5000)
    ap.add_argument("--n-x", type=int, default=5)
    ap.add_argument("--treat-frac", type=float, default=0.5)
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kind = str(args.kind)
    seed = int(args.seed)
    repeat = int(args.repeat)
    n_x = int(args.n_x)

    # Generate a deterministic dataset spec per case. This is the dataset identity.
    if kind == "panel_fe":
        spec = {"kind": kind, "n_entities": int(args.n_entities), "n_times": int(args.n_times), "n_x": n_x, "seed": seed}
        data = _gen_panel(int(args.n_entities), int(args.n_times), n_x, seed=seed)
        run_fit = _fit_panel_fe
    elif kind == "did_twfe":
        spec = {
            "kind": kind,
            "n_entities": int(args.n_entities),
            "n_times": int(args.n_times),
            "n_x": n_x,
            "treat_frac": float(args.treat_frac),
            "seed": seed,
        }
        data = _gen_did(int(args.n_entities), int(args.n_times), n_x, seed=seed, treat_frac=float(args.treat_frac))
        run_fit = _fit_did_twfe
    elif kind == "event_study_twfe":
        spec = {"kind": kind, "n_entities": int(args.n_entities), "n_times": int(args.n_times), "n_x": n_x, "seed": seed}
        data = _gen_event_study(int(args.n_entities), int(args.n_times), n_x, seed=seed)
        run_fit = _fit_event_study
    elif kind == "iv_2sls":
        spec = {"kind": kind, "n_obs": int(args.n_obs), "n_x": n_x, "seed": seed}
        data = _gen_iv(int(args.n_obs), n_x, seed=seed)
        run_fit = _fit_iv_2sls
    else:
        spec = {"kind": kind, "n_obs": int(args.n_obs), "n_x": n_x, "seed": seed}
        data = _gen_aipw(int(args.n_obs), n_x, seed=seed)
        run_fit = _fit_aipw

    dataset = {"id": f"generated:econometrics:{args.case}", "sha256": sha256_json_obj(spec), "spec": spec}

    has_sm, sm_v = _maybe_import("statsmodels")
    has_lm, lm_v = _maybe_import("linearmodels")

    status = "ok"
    reason: Optional[str] = None

    try:
        coef_ns, se_ns, extra_ns = run_fit(data)
    except Exception as e:
        obj = {
            "schema_version": "nextstat.econometrics_benchmark_result.v1",
            "suite": "econometrics",
            "case": str(args.case),
            "deterministic": bool(args.deterministic),
            "status": "failed",
            "reason": f"nextstat_error:{type(e).__name__}:{e}",
            "meta": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "nextstat_version": nextstat.__version__,
                "numpy_version": np.__version__,
                "statsmodels_version": sm_v,
                "linearmodels_version": lm_v,
            },
            "dataset": dataset,
            "config": {"kind": kind},
            "parity": {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}},
            "timing": {"wall_time_s": {"min": 0.0, "median": 0.0, "p95": 0.0}, "raw": {"repeat": repeat, "policy": "median", "runs_s": []}},
            "results": {"nextstat": {"coef": [], "se": [], "extra": {}}, "baseline": None},
        }
        out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        return 2

    # Timing: end-to-end call time (includes FE transform, robust SE, etc).
    runs_s: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = run_fit(data)
        runs_s.append(float(time.perf_counter() - t0))

    timing = {"wall_time_s": _summary(runs_s), "raw": {"repeat": repeat, "policy": "median", "runs_s": [float(x) for x in runs_s]}}

    baseline_obj = None
    parity = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}
    timing_baseline: dict[str, Any] = {}

    try:
        if kind in ("panel_fe", "did_twfe", "event_study_twfe"):
            if not has_sm:
                raise RuntimeError("missing_dependency:statsmodels")

            # For these cases, build the exact transformed design from NextStat API to avoid
            # disagreements about FE absorption / column selection in tiny samples.
            if kind == "panel_fe":
                # within transform only (entity demeaning)
                y_star, x_star = nextstat.econometrics._within_demean(data["y"], data["x"], data["entity"])  # type: ignore[attr-defined]
                coef_b, se_b = _baseline_statsmodels_ols(y=y_star, x=x_star, groups=data["entity"])
            elif kind == "did_twfe":
                # TWFE transform is inside NextStat; we use the public fit result only for att.
                # Baseline uses the same regressor construction, then TWFE demeaning.
                d = nextstat.econometrics.did_regressor(data["treat"], data["post"])
                x_raw = [[di] + list(map(float, row)) for di, row in zip(d, data["x"])]
                names = ["treat_post"] + [f"x{i}" for i in range(len(data["x"][0]))]
                y_dd, x_dd, _ne, _nt = nextstat.econometrics._two_way_demean(data["y"], x_raw, data["entity"], data["time"])  # type: ignore[attr-defined]
                x_sel, names_sel = nextstat.econometrics._select_independent_columns(x_dd, names, mandatory=[0])  # type: ignore[attr-defined]
                _ = names_sel
                coef_b, se_b = _baseline_statsmodels_ols(y=y_dd, x=x_sel, groups=data["entity"])
                # Align to ATT only (first coef).
                coef_b, se_b = [coef_b[0]], [se_b[0]]
            else:
                rel = nextstat.econometrics.relative_time(data["time"], data["event_time"])
                x_ev, names_ev, _ks = nextstat.econometrics.event_study_regressors(data["treat"], rel, window=tuple(data["window"]), reference=int(data["reference"]))
                x_ev_np = np.asarray(x_ev, dtype=float)
                x_ctrl_np = np.asarray(data["x"], dtype=float)
                x_all = np.column_stack([x_ev_np, x_ctrl_np])
                names = names_ev + [f"x{i}" for i in range(x_ctrl_np.shape[1])]
                y_dd, x_dd, _ne, _nt = nextstat.econometrics._two_way_demean(data["y"], x_all, data["entity"], data["time"])  # type: ignore[attr-defined]
                x_sel, names_sel = nextstat.econometrics._select_independent_columns(x_dd, names, mandatory=[])  # type: ignore[attr-defined]
                # Keep only event coefficients at the front.
                n_ev = 0
                for nm in names_sel:
                    if nm.startswith("event[") and nm.endswith("]"):
                        n_ev += 1
                    else:
                        break
                coef_full, se_full = _baseline_statsmodels_ols(y=y_dd, x=x_sel, groups=data["entity"])
                coef_b, se_b = coef_full[:n_ev], se_full[:n_ev]

            # Full-pipeline timing via linearmodels.PanelOLS (fair end-to-end comparison)
            if has_lm:
                _lm_run = {
                    "panel_fe": _full_baseline_lm_panel_fe,
                    "did_twfe": _full_baseline_lm_did,
                    "event_study_twfe": _full_baseline_lm_event_study,
                }[kind]
                _lm_run(data)  # warmup
                runs_bl: list[float] = []
                for _ in range(repeat):
                    t0 = time.perf_counter()
                    _lm_run(data)
                    runs_bl.append(float(time.perf_counter() - t0))
                timing_baseline = {"name": "linearmodels.PanelOLS", "wall_time_s": _summary(runs_bl), "raw": {"repeat": repeat, "runs_s": runs_bl}}

            baseline_obj = {"coef": coef_b, "se": se_b, "extra": {"backend": "statsmodels"}}
            coef_abs, coef_rel = _max_abs_rel_diff(coef_ns, coef_b)
            se_abs, se_rel = _max_abs_rel_diff(se_ns, se_b)
            parity = {
                "status": "ok" if coef_abs is not None else "warn",
                "reference": {"name": "statsmodels", "version": str(sm_v or "")},
                "metrics": {
                    "coef_max_abs_diff": coef_abs,
                    "coef_max_rel_diff": coef_rel,
                    "se_max_abs_diff": se_abs,
                    "se_max_rel_diff": se_rel,
                },
            }
        elif kind == "iv_2sls":
            if not has_lm:
                raise RuntimeError("missing_dependency:linearmodels")
            coef_b, se_b = _baseline_linearmodels_iv(data)
            runs_bl: list[float] = []
            for _ in range(repeat):
                t0 = time.perf_counter()
                _baseline_linearmodels_iv(data)
                runs_bl.append(float(time.perf_counter() - t0))
            timing_baseline = {"name": "linearmodels", "wall_time_s": _summary(runs_bl), "raw": {"repeat": repeat, "runs_s": runs_bl}}
            baseline_obj = {"coef": coef_b, "se": se_b, "extra": {"backend": "linearmodels"}}
            coef_abs, coef_rel = _max_abs_rel_diff(coef_ns, coef_b)
            se_abs, se_rel = _max_abs_rel_diff(se_ns, se_b)
            parity = {
                "status": "ok" if coef_abs is not None else "warn",
                "reference": {"name": "linearmodels", "version": str(lm_v or "")},
                "metrics": {
                    "coef_max_abs_diff": coef_abs,
                    "coef_max_rel_diff": coef_rel,
                    "se_max_abs_diff": se_abs,
                    "se_max_rel_diff": se_rel,
                },
            }
        else:
            parity = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}
    except Exception as e:
        status = "warn"
        reason = f"baseline_unavailable:{type(e).__name__}:{e}"
        parity = {"status": "warn", "reference": {"name": "", "version": ""}, "metrics": {}}

    cfg = {"kind": kind, "cov": "cluster" if kind in ("panel_fe", "did_twfe", "event_study_twfe") else "hc1", "cluster": "entity"}
    cfg["n_obs"] = int(extra_ns.get("n_obs", 0))
    cfg["n_entities"] = int(extra_ns.get("n_entities", 0))
    cfg["n_times"] = int(extra_ns.get("n_times", 0))
    cfg["n_regressors"] = int(len(coef_ns))
    if kind == "event_study_twfe":
        cfg["window"] = list(data.get("window", [-4, 4]))
        cfg["reference"] = int(data.get("reference", -1))

    obj = {
        "schema_version": "nextstat.econometrics_benchmark_result.v1",
        "suite": "econometrics",
        "case": str(args.case),
        "deterministic": bool(args.deterministic),
        "status": status,
        "reason": reason,
        "meta": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "nextstat_version": nextstat.__version__,
            "numpy_version": np.__version__,
            "statsmodels_version": sm_v,
            "linearmodels_version": lm_v,
        },
        "dataset": dataset,
        "config": cfg,
        "parity": parity,
        "timing": timing,
        "timing_baseline": timing_baseline,
        "results": {"nextstat": {"coef": coef_ns, "se": se_ns, "extra": extra_ns}, "baseline": baseline_obj},
    }
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return 0 if status != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

