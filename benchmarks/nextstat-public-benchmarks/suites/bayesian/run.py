#!/usr/bin/env python3
"""Bayesian suite runner: NUTS diagnostics + ESS/sec.

Backends:
- `nextstat` (always available)
- `cmdstanpy` (optional)
- `pymc` (optional)

This runner writes a single schema-backed JSON artifact per invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.bench_env import collect_environment


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json_obj(obj: Any) -> str:
    b = (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _median(xs: list[float]) -> float | None:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    if not xs:
        return None
    xs.sort()
    n = len(xs)
    mid = n // 2
    if n % 2:
        return float(xs[mid])
    return 0.5 * (float(xs[mid - 1]) + float(xs[mid]))


def _diag_summary_nextstat(diag: Mapping[str, Any]) -> dict[str, float | None]:
    r_hat = diag.get("r_hat") if isinstance(diag.get("r_hat"), dict) else {}
    ess_bulk = diag.get("ess_bulk") if isinstance(diag.get("ess_bulk"), dict) else {}
    ess_tail = diag.get("ess_tail") if isinstance(diag.get("ess_tail"), dict) else {}
    ebfmi = diag.get("ebfmi") if isinstance(diag.get("ebfmi"), list) else []

    rhat_vals = [v for v in (_safe_float(x) for x in r_hat.values()) if v is not None]
    ess_bulk_vals = [v for v in (_safe_float(x) for x in ess_bulk.values()) if v is not None]
    ess_tail_vals = [v for v in (_safe_float(x) for x in ess_tail.values()) if v is not None]
    ebfmi_vals = [v for v in (_safe_float(x) for x in ebfmi) if v is not None]

    return {
        "divergence_rate": _safe_float(diag.get("divergence_rate")),
        "max_treedepth_rate": _safe_float(diag.get("max_treedepth_rate")),
        "max_r_hat": max(rhat_vals) if rhat_vals else None,
        "min_ess_bulk": min(ess_bulk_vals) if ess_bulk_vals else None,
        "min_ess_tail": min(ess_tail_vals) if ess_tail_vals else None,
        "min_ebfmi": min(ebfmi_vals) if ebfmi_vals else None,
    }


def _posterior_summary_nextstat(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize posterior draws (mean/sd + ESS_bulk) for parity checks.

    Keeps artifacts small by storing only summary stats, not the full draw arrays.
    """
    posterior = raw.get("posterior")
    param_names = raw.get("param_names")
    diag = raw.get("diagnostics") if isinstance(raw.get("diagnostics"), dict) else {}
    ess_bulk = diag.get("ess_bulk") if isinstance(diag.get("ess_bulk"), dict) else {}

    if not isinstance(posterior, Mapping) or not isinstance(param_names, list):
        return {"status": "missing"}

    params: list[dict[str, Any]] = []
    for name_any in param_names:
        name = str(name_any)
        chains = posterior.get(name)
        if not isinstance(chains, list) or not chains:
            continue

        flat: list[float] = []
        for c in chains:
            if isinstance(c, list):
                flat.extend(float(v) for v in c)
        if not flat:
            continue

        n = len(flat)
        mean = sum(flat) / n
        if n > 1:
            var = sum((x - mean) ** 2 for x in flat) / (n - 1)
            sd = math.sqrt(var) if var >= 0 else None
        else:
            sd = None

        params.append(
            {
                "name": name,
                "mean": mean,
                "sd": sd,
                "ess_bulk": _safe_float(ess_bulk.get(name)),
            }
        )

    return {"status": "ok", "params": params}


def _ess_per_sec(*, ess_vals: list[float], wall_time_s: float) -> dict[str, float]:
    eps = 1e-12
    out: dict[str, float] = {}
    if ess_vals:
        out["min"] = float(min(ess_vals) / max(float(wall_time_s), eps))
        med = _median(ess_vals)
        if med is not None:
            out["median"] = float(med / max(float(wall_time_s), eps))
    return out


def _series_to_float_dict(series_like: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        items = series_like.items()
    except Exception:
        return out
    for k, v in items:
        fv = _safe_float(v)
        if fv is None:
            continue
        out[str(k)] = float(fv)
    return out


def _stan_energy_ebfmi(energy_by_chain: dict[int, list[float]]) -> float | None:
    # https://mc-stan.org/docs/reference-manual/analysis.html#ebfmi
    best: float | None = None
    for xs in energy_by_chain.values():
        if len(xs) < 4:
            continue
        mean = sum(xs) / float(len(xs))
        var = sum((x - mean) ** 2 for x in xs) / max(float(len(xs) - 1), 1.0)
        if not math.isfinite(var) or var <= 0:
            continue
        diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        msd = sum(d * d for d in diffs) / max(float(len(diffs)), 1.0)
        ebfmi = float(msd / var)
        if not math.isfinite(ebfmi):
            continue
        best = ebfmi if best is None else min(best, ebfmi)
    return best


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def _base_doc(*, args: argparse.Namespace, nextstat_version: str, dataset: dict[str, Any], model_type: str, cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "nextstat.bayesian_benchmark_result.v1",
        "suite": "bayesian",
        "case": str(args.case),
        "backend": str(args.backend),
        "deterministic": bool(args.deterministic),
        "environment": collect_environment(),
        "meta": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "nextstat_version": str(nextstat_version),
        },
        "dataset": dataset,
        "model": {"model_type": model_type},
        "config": cfg,
    }


def _emit_missing_backend(*, out_path: Path, args: argparse.Namespace, nextstat_version: str, dataset: dict[str, Any], model_type: str, cfg: dict[str, Any], backend_name: str, err: Exception) -> int:
    doc = _base_doc(args=args, nextstat_version=nextstat_version, dataset=dataset, model_type=model_type, cfg=cfg)
    doc.update(
        {
            "status": "warn",
            "reason": f"missing_backend_dep:{backend_name}:{type(err).__name__}:{err}",
            "timing": {"wall_time_s": 0.0},
            "diagnostics_summary": {
                "divergence_rate": None,
                "max_treedepth_rate": None,
                "max_r_hat": None,
                "min_ess_bulk": None,
                "min_ess_tail": None,
                "min_ebfmi": None,
            },
        }
    )
    _write_json(out_path, doc)
    return 0


def _emit_not_supported(*, out_path: Path, args: argparse.Namespace, nextstat_version: str, dataset: dict[str, Any], model_type: str, cfg: dict[str, Any]) -> int:
    doc = _base_doc(args=args, nextstat_version=nextstat_version, dataset=dataset, model_type=model_type, cfg=cfg)
    doc.update(
        {
            "status": "warn",
            "reason": f"backend_not_supported_for_model:{args.backend}:{args.model}",
            "timing": {"wall_time_s": 0.0},
            "diagnostics_summary": {
                "divergence_rate": None,
                "max_treedepth_rate": None,
                "max_r_hat": None,
                "min_ess_bulk": None,
                "min_ess_tail": None,
                "min_ebfmi": None,
            },
        }
    )
    _write_json(out_path, doc)
    return 0


def _logistic(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def make_logistic_regression_dataset(*, n: int, p: int, seed: int) -> dict[str, Any]:
    rng = random.Random(int(seed))
    beta = [0.6, -1.1, 0.3, 0.0, 0.8][:p]
    intercept = -0.2

    x: list[list[float]] = []
    y: list[int] = []
    for _ in range(int(n)):
        row = [rng.gauss(0.0, 1.0) for _ in range(int(p))]
        z = intercept + sum(b * v for b, v in zip(beta, row))
        pr = _logistic(z)
        yi = 1 if rng.random() < pr else 0
        x.append(row)
        y.append(int(yi))

    return {
        "kind": "logistic_regression",
        "n": int(n),
        "p": int(p),
        "seed": int(seed),
        "beta": beta,
        "intercept": float(intercept),
        "x": x,
        "y": y,
    }


def make_hier_random_intercept_dataset(*, n_groups: int, n_per_group: int, seed: int) -> dict[str, Any]:
    rng = random.Random(int(seed))
    beta = [1.0]  # one feature + intercept in the model
    intercept = 0.0
    sigma_alpha = 1.0

    group_alpha = [rng.gauss(0.0, sigma_alpha) for _ in range(int(n_groups))]

    x: list[list[float]] = []
    y: list[int] = []
    group_idx: list[int] = []
    for g in range(int(n_groups)):
        for _ in range(int(n_per_group)):
            row = [rng.gauss(0.0, 1.0)]
            z = intercept + group_alpha[g] + beta[0] * row[0]
            pr = _logistic(z)
            yi = 1 if rng.random() < pr else 0
            x.append(row)
            y.append(int(yi))
            group_idx.append(int(g))

    return {
        "kind": "hier_logistic_random_intercept",
        "n_groups": int(n_groups),
        "n_per_group": int(n_per_group),
        "seed": int(seed),
        "beta": beta,
        "intercept": float(intercept),
        "sigma_alpha": float(sigma_alpha),
        "x": x,
        "y": y,
        "group_idx": group_idx,
    }


def make_eight_schools_dataset() -> dict[str, Any]:
    return {
        "kind": "eight_schools",
        "J": 8,
        "y": [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        "sigma": [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="Case id for reporting.")
    ap.add_argument(
        "--model",
        required=True,
        choices=["histfactory_simple", "glm_logistic", "hier_random_intercept", "eight_schools"],
        help="Which baseline model to run.",
    )
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--deterministic", action="store_true", help="Deterministic output.")
    ap.add_argument(
        "--backend",
        default="nextstat",
        choices=["nextstat", "nextstat_dense", "cmdstanpy", "pymc", "numpyro"],
        help="Which NUTS backend to run (optional).",
    )

    # Sampler settings (kept explicit for reproducibility).
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dataset-seed",
        type=int,
        default=None,
        help="Seed used for dataset generation (generated cases). Defaults to --seed for backward-compat.",
    )
    ap.add_argument("--max-treedepth", type=int, default=10)
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--init-jitter-rel", type=float, default=0.10)
    ap.add_argument("--metric", type=str, default="diagonal",
                    choices=["diagonal", "dense", "auto"],
                    help="(nextstat) Mass matrix type.")

    # Dataset/model-specific knobs (for generated cases).
    ap.add_argument("--n", type=int, default=200, help="(glm_logistic) number of rows.")
    ap.add_argument("--p", type=int, default=5, help="(glm_logistic) number of features.")
    ap.add_argument("--n-groups", type=int, default=20, help="(hier_random_intercept) groups.")
    ap.add_argument("--n-per-group", type=int, default=20, help="(hier_random_intercept) rows per group.")

    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = {
        "n_chains": int(args.n_chains),
        "n_warmup": int(args.warmup),
        "n_samples": int(args.samples),
        "seed": int(args.seed),
        "dataset_seed": int(args.dataset_seed) if args.dataset_seed is not None else int(args.seed),
        "max_treedepth": int(args.max_treedepth),
        "target_accept": float(args.target_accept),
        "init_jitter_rel": float(args.init_jitter_rel),
        "metric": str(args.metric),
    }

    # Eight Schools has funnel geometry — raise target_accept for better exploration.
    if args.model == "eight_schools":
        cfg["target_accept"] = max(cfg["target_accept"], 0.95)

    try:
        import nextstat  # type: ignore
    except Exception as e:
        doc = {
            "schema_version": "nextstat.bayesian_benchmark_result.v1",
            "suite": "bayesian",
            "case": str(args.case),
            "backend": str(args.backend),
            "deterministic": bool(args.deterministic),
            "environment": collect_environment(),
            "status": "failed",
            "reason": f"import_nextstat_failed:{type(e).__name__}:{e}",
            "meta": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "nextstat_version": "unknown",
            },
            "dataset": {"id": "unknown", "sha256": "0" * 64},
            "config": cfg,
            "timing": {"wall_time_s": 0.0},
            "diagnostics_summary": {
                "divergence_rate": None,
                "max_treedepth_rate": None,
                "max_r_hat": None,
                "min_ess_bulk": None,
                "min_ess_tail": None,
                "min_ebfmi": None,
            },
        }
        _write_json(out_path, doc)
        return 2

    nextstat_version = str(getattr(nextstat, "__version__", "unknown"))

    dataset: dict[str, Any]
    generated: dict[str, Any] | None = None
    model_obj: Any
    model_type: str

    if args.model == "histfactory_simple":
        ws_path = Path(__file__).resolve().parent / "datasets" / "simple_workspace.json"
        dataset_sha = sha256_file(ws_path)
        ws = json.loads(ws_path.read_text())
        model_obj = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))
        model_type = "HistFactoryModel(simple_workspace.json)"
        dataset = {"id": str(ws_path), "path": str(ws_path), "sha256": dataset_sha}
    elif args.model == "glm_logistic":
        dataset_seed = int(args.dataset_seed) if args.dataset_seed is not None else int(args.seed)
        raw = make_logistic_regression_dataset(n=int(args.n), p=int(args.p), seed=dataset_seed)
        generated = raw
        dataset_id = f"generated:glm_logistic_n={args.n}_p={args.p}_seed={dataset_seed}"
        dataset_sha = sha256_json_obj(raw)
        spec = nextstat.data.GlmSpec.logistic_regression(
            x=raw["x"],
            y=raw["y"],
            include_intercept=True,
            coef_prior_mu=0.0,
            coef_prior_sigma=1.0,
        )
        model_obj = spec.build()
        model_type = "ComposedGlmModel.logistic_regression"
        dataset = {"id": dataset_id, "sha256": dataset_sha}
    elif args.model == "hier_random_intercept":
        dataset_seed = int(args.dataset_seed) if args.dataset_seed is not None else int(args.seed)
        raw = make_hier_random_intercept_dataset(
            n_groups=int(args.n_groups), n_per_group=int(args.n_per_group), seed=dataset_seed
        )
        generated = raw
        dataset_id = f"generated:hier_logistic_random_intercept_g={args.n_groups}_nper={args.n_per_group}_seed={dataset_seed}"
        dataset_sha = sha256_json_obj(raw)
        spec = nextstat.data.GlmSpec.logistic_regression(
            x=raw["x"],
            y=raw["y"],
            include_intercept=False,
            group_idx=raw["group_idx"],
            n_groups=int(args.n_groups),
            coef_prior_mu=0.0,
            coef_prior_sigma=1.0,
            random_intercept_non_centered=True,
        )
        model_obj = spec.build()
        model_type = "Hierarchical logistic (random intercept, non-centered)"
        dataset = {"id": dataset_id, "sha256": dataset_sha}
    else:
        # eight_schools
        raw = make_eight_schools_dataset()
        generated = raw
        dataset_id = "eight_schools"
        dataset_sha = sha256_json_obj(raw)
        model_obj = nextstat.EightSchoolsModel(y=raw["y"], sigma=raw["sigma"])
        model_type = "Eight Schools (non-centered, funnel geometry)"
        dataset = {"id": dataset_id, "sha256": dataset_sha}

    if args.backend in ("nextstat", "nextstat_dense"):
        t0 = time.perf_counter()
        try:
            r = nextstat.sample(
                model_obj,
                n_chains=cfg["n_chains"],
                n_warmup=cfg["n_warmup"],
                n_samples=cfg["n_samples"],
                seed=cfg["seed"],
                max_treedepth=cfg["max_treedepth"],
                target_accept=cfg["target_accept"],
                metric=("dense" if args.backend == "nextstat_dense" else cfg["metric"]),
                init_jitter_rel=cfg["init_jitter_rel"],
            )
            wall = time.perf_counter() - t0
        except Exception as e:
            wall = time.perf_counter() - t0
            doc = _base_doc(args=args, nextstat_version=nextstat_version, dataset=dataset, model_type=model_type, cfg=cfg)
            doc.update(
                {
                    "status": "failed",
                    "reason": f"sample_failed:{type(e).__name__}:{e}",
                    "timing": {"wall_time_s": float(wall)},
                    "diagnostics_summary": {
                        "divergence_rate": None,
                        "max_treedepth_rate": None,
                        "max_r_hat": None,
                        "min_ess_bulk": None,
                        "min_ess_tail": None,
                        "min_ebfmi": None,
                    },
                }
            )
            _write_json(out_path, doc)
            return 2

        diag = r.get("diagnostics") if isinstance(r.get("diagnostics"), dict) else {}
        summary = _diag_summary_nextstat(diag)
        quality = diag.get("quality") if isinstance(diag.get("quality"), dict) else {}
        status = str(quality.get("status") or "ok")
        if status not in ("ok", "warn", "failed"):
            status = "ok"

        ess_bulk = diag.get("ess_bulk") if isinstance(diag.get("ess_bulk"), dict) else {}
        ess_tail = diag.get("ess_tail") if isinstance(diag.get("ess_tail"), dict) else {}
        ess_bulk_vals = [v for v in (_safe_float(x) for x in ess_bulk.values()) if v is not None]
        ess_tail_vals = [v for v in (_safe_float(x) for x in ess_tail.values()) if v is not None]

        timing = {
            "wall_time_s": float(wall),
            "ess_bulk_per_sec": _ess_per_sec(ess_vals=ess_bulk_vals, wall_time_s=float(wall)),
            "ess_tail_per_sec": _ess_per_sec(ess_vals=ess_tail_vals, wall_time_s=float(wall)),
        }

        param_names = list(map(str, (r.get("param_names") or []))) if isinstance(r.get("param_names"), list) else []

        doc = _base_doc(args=args, nextstat_version=nextstat_version, dataset=dataset, model_type=model_type, cfg=cfg)
        doc["status"] = status
        doc["model"] = {"model_type": model_type, "n_params": int(len(param_names)), "param_names": param_names}
        doc["timing"] = timing
        doc["diagnostics_summary"] = summary
        doc["diagnostics"] = {
            "divergence_rate": diag.get("divergence_rate"),
            "max_treedepth_rate": diag.get("max_treedepth_rate"),
            "r_hat": diag.get("r_hat"),
            "ess_bulk": diag.get("ess_bulk"),
            "ess_tail": diag.get("ess_tail"),
            "ebfmi": diag.get("ebfmi"),
            "quality": diag.get("quality"),
            "posterior_summary": _posterior_summary_nextstat(r),
        }
        _write_json(out_path, doc)
        return 0 if status != "failed" else 2

    if args.model == "histfactory_simple":
        return _emit_not_supported(out_path=out_path, args=args, nextstat_version=nextstat_version, dataset=dataset, model_type=model_type, cfg=cfg)

    if args.backend == "cmdstanpy":
        try:
            import cmdstanpy  # type: ignore
        except Exception as e:
            return _emit_missing_backend(out_path=out_path, args=args, nextstat_version=nextstat_version, dataset=dataset, model_type=model_type, cfg=cfg, backend_name="cmdstanpy", err=e)

        backend_meta: dict[str, Any] = {"cmdstanpy_version": str(getattr(cmdstanpy, "__version__", "unknown"))}

        cache_dir = out_path.parent / "_cmdstan_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        stan_dir = cache_dir / "stan"
        stan_dir.mkdir(parents=True, exist_ok=True)

        if args.model == "glm_logistic":
            stan_code = """
data {
  int<lower=1> N;
  int<lower=1> P;
  matrix[N, P] X;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  real alpha;
  vector[P] beta;
}
model {
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  y ~ bernoulli_logit(alpha + X * beta);
}
"""
            assert generated is not None
            stan_data = {
                "N": int(generated["n"]),
                "P": int(generated["p"]),
                "X": generated["x"],
                "y": generated["y"],
            }
        elif args.model == "hier_random_intercept":
            stan_code = """
data {
  int<lower=1> N;
  int<lower=1> G;
  vector[N] x;
  array[N] int<lower=0, upper=1> y;
  array[N] int<lower=1, upper=G> g;
}
parameters {
  real alpha0;
  real beta;
  vector[G] alpha_raw;
  real<lower=0> sigma_alpha;
}
transformed parameters {
  vector[G] alpha = alpha0 + sigma_alpha * alpha_raw;
}
model {
  alpha0 ~ normal(0, 1);
  beta ~ normal(0, 1);
  alpha_raw ~ normal(0, 1);
  sigma_alpha ~ normal(0, 1);
  for (i in 1:N) {
    y[i] ~ bernoulli_logit(alpha[g[i]] + beta * x[i]);
  }
}
"""
            assert generated is not None
            x = [float(row[0]) for row in generated["x"]]
            g = [int(i) + 1 for i in generated["group_idx"]]  # Stan is 1-indexed
            stan_data = {
                "N": int(len(generated["y"])),
                "G": int(generated["n_groups"]),
                "x": x,
                "y": generated["y"],
                "g": g,
            }
        else:
            # eight_schools (non-centered parameterization)
            stan_code = """
data {
  int<lower=0> J;
  array[J] real y;
  array[J] real<lower=0> sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta_raw;
}
transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
}
model {
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 5);
  theta_raw ~ normal(0, 1);
  y ~ normal(theta, to_vector(sigma));
}
"""
            assert generated is not None
            stan_data = {
                "J": int(generated["J"]),
                "y": generated["y"],
                "sigma": generated["sigma"],
            }

        code_hash = hashlib.sha256(stan_code.encode("utf-8")).hexdigest()
        stan_path = stan_dir / f"{args.model}_{code_hash}.stan"
        if not stan_path.exists():
            stan_path.write_text(stan_code.strip() + "\n")

        try:
            model = cmdstanpy.CmdStanModel(stan_file=str(stan_path))
            try:
                from cmdstanpy.utils import cmdstan as _cmdstan  # type: ignore

                backend_meta["cmdstan_version"] = str(_cmdstan.cmdstan_version())
            except Exception:
                pass
        except Exception as e:
            doc = _base_doc(args=args, nextstat_version=nextstat_version, dataset=dataset, model_type=model_type, cfg=cfg)
            doc.update(
                {
                    "status": "warn",
                    "reason": f"cmdstan_model_build_failed:{type(e).__name__}:{e}",
                    "backend_meta": backend_meta,
                    "timing": {"wall_time_s": 0.0},
                    "diagnostics_summary": {
                        "divergence_rate": None,
                        "max_treedepth_rate": None,
                        "max_r_hat": None,
                        "min_ess_bulk": None,
                        "min_ess_tail": None,
                        "min_ebfmi": None,
                    },
                }
            )
            _write_json(out_path, doc)
            return 0

        t0 = time.perf_counter()
        try:
            fit = model.sample(
                data=stan_data,
                chains=int(cfg["n_chains"]),
                parallel_chains=int(cfg["n_chains"]),
                iter_warmup=int(cfg["n_warmup"]),
                iter_sampling=int(cfg["n_samples"]),
                seed=int(cfg["seed"]),
                max_treedepth=int(cfg["max_treedepth"]),
                adapt_delta=float(cfg["target_accept"]),
                show_progress=False,
                refresh=1,
            )
            wall = time.perf_counter() - t0
        except Exception as e:
            wall = time.perf_counter() - t0
            doc = _base_doc(args=args, nextstat_version=nextstat_version, dataset=dataset, model_type=model_type, cfg=cfg)
            doc.update(
                {
                    "status": "failed",
                    "reason": f"cmdstan_sample_failed:{type(e).__name__}:{e}",
                    "backend_meta": backend_meta,
                    "timing": {"wall_time_s": float(wall)},
                    "diagnostics_summary": {
                        "divergence_rate": None,
                        "max_treedepth_rate": None,
                        "max_r_hat": None,
                        "min_ess_bulk": None,
                        "min_ess_tail": None,
                        "min_ebfmi": None,
                    },
                }
            )
            _write_json(out_path, doc)
            return 2

        max_r_hat: float | None = None
        min_ess_bulk: float | None = None
        min_ess_tail: float | None = None
        r_hat_map: dict[str, float] = {}
        ess_bulk_map: dict[str, float] = {}
        ess_tail_map: dict[str, float] = {}
        ess_bulk_vals: list[float] = []
        ess_tail_vals: list[float] = []
        try:
            df = fit.summary()
            for name, row in df.iterrows():
                k = str(name)
                if k == "lp__":
                    continue
                rhat = _safe_float(row.get("R_hat"))
                eb = _safe_float(row.get("ESS_bulk"))
                et = _safe_float(row.get("ESS_tail"))
                if rhat is not None:
                    r_hat_map[k] = float(rhat)
                if eb is not None:
                    ess_bulk_map[k] = float(eb)
                    ess_bulk_vals.append(float(eb))
                if et is not None:
                    ess_tail_map[k] = float(et)
                    ess_tail_vals.append(float(et))
            max_r_hat = max(r_hat_map.values()) if r_hat_map else None
            min_ess_bulk = min(ess_bulk_vals) if ess_bulk_vals else None
            min_ess_tail = min(ess_tail_vals) if ess_tail_vals else None
        except Exception:
            pass

        divergence_rate: float | None = None
        max_treedepth_rate: float | None = None
        min_ebfmi: float | None = None
        try:
            draws = fit.draws_pd(vars=["divergent__", "treedepth__", "energy__"])
            if "divergent__" in draws.columns:
                divergence_rate = float(draws["divergent__"].mean())
            if "treedepth__" in draws.columns:
                max_treedepth_rate = float((draws["treedepth__"] >= int(cfg["max_treedepth"])).mean())
            if all(c in draws.columns for c in ("chain__", "energy__")):
                energy_by_chain: dict[int, list[float]] = {}
                for chain, e in zip(draws["chain__"].tolist(), draws["energy__"].tolist()):
                    fv = _safe_float(e)
                    if fv is None:
                        continue
                    energy_by_chain.setdefault(int(chain), []).append(float(fv))
                min_ebfmi = _stan_energy_ebfmi(energy_by_chain)
        except Exception:
            pass

        summary = {
            "divergence_rate": divergence_rate,
            "max_treedepth_rate": max_treedepth_rate,
            "max_r_hat": max_r_hat,
            "min_ess_bulk": min_ess_bulk,
            "min_ess_tail": min_ess_tail,
            "min_ebfmi": min_ebfmi,
        }
        timing = {
            "wall_time_s": float(wall),
            "ess_bulk_per_sec": _ess_per_sec(ess_vals=ess_bulk_vals, wall_time_s=float(wall)),
            "ess_tail_per_sec": _ess_per_sec(ess_vals=ess_tail_vals, wall_time_s=float(wall)),
        }

        doc = _base_doc(args=args, nextstat_version=nextstat_version, dataset=dataset, model_type=model_type, cfg=cfg)
        doc.update(
            {
                "status": "ok",
                "backend_meta": backend_meta,
                "timing": timing,
                "diagnostics_summary": summary,
                "diagnostics": {"r_hat": r_hat_map, "ess_bulk": ess_bulk_map, "ess_tail": ess_tail_map},
            }
        )
        _write_json(out_path, doc)
        return 0

    if args.backend == "pymc":
        try:
            import arviz as az  # type: ignore
            import pymc as pm  # type: ignore
        except Exception as e:
            return _emit_missing_backend(out_path=out_path, args=args, nextstat_version=nextstat_version, dataset=dataset, model_type=model_type, cfg=cfg, backend_name="pymc", err=e)

        backend_meta: dict[str, Any] = {
            "pymc_version": str(getattr(pm, "__version__", "unknown")),
            "arviz_version": str(getattr(az, "__version__", "unknown")),
        }

        import numpy as np  # type: ignore

        if args.model == "glm_logistic":
            assert generated is not None
            x = np.asarray(generated["x"], dtype=float)
            y = np.asarray(generated["y"], dtype=int)
            with pm.Model():
                alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
                beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=(int(generated["p"]),))
                logits = alpha + pm.math.dot(x, beta)
                pm.Bernoulli("y", logit_p=logits, observed=y)
                t0 = time.perf_counter()
                idata = pm.sample(
                    draws=int(cfg["n_samples"]),
                    tune=int(cfg["n_warmup"]),
                    chains=int(cfg["n_chains"]),
                    cores=max(1, int(cfg["n_chains"])),
                    random_seed=int(cfg["seed"]),
                    target_accept=float(cfg["target_accept"]),
                    nuts={"max_treedepth": int(cfg["max_treedepth"])},
                    progressbar=False,
                )
                wall = time.perf_counter() - t0
        elif args.model == "hier_random_intercept":
            assert generated is not None
            x = np.asarray([float(row[0]) for row in generated["x"]], dtype=float)
            y = np.asarray(generated["y"], dtype=int)
            g = np.asarray(generated["group_idx"], dtype=int)
            g_n = int(generated["n_groups"])
            with pm.Model():
                alpha0 = pm.Normal("alpha0", mu=0.0, sigma=1.0)
                beta = pm.Normal("beta", mu=0.0, sigma=1.0)
                sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.0)
                alpha_raw = pm.Normal("alpha_raw", mu=0.0, sigma=1.0, shape=(g_n,))
                alpha = pm.Deterministic("alpha", alpha0 + sigma_alpha * alpha_raw)
                logits = alpha[g] + beta * x
                pm.Bernoulli("y", logit_p=logits, observed=y)
                t0 = time.perf_counter()
                idata = pm.sample(
                    draws=int(cfg["n_samples"]),
                    tune=int(cfg["n_warmup"]),
                    chains=int(cfg["n_chains"]),
                    cores=max(1, int(cfg["n_chains"])),
                    random_seed=int(cfg["seed"]),
                    target_accept=float(cfg["target_accept"]),
                    nuts={"max_treedepth": int(cfg["max_treedepth"])},
                    progressbar=False,
                )
                wall = time.perf_counter() - t0
        else:
            # eight_schools (non-centered)
            assert generated is not None
            J = int(generated["J"])
            y_obs = np.asarray(generated["y"], dtype=float)
            sigma_obs = np.asarray(generated["sigma"], dtype=float)
            with pm.Model():
                mu = pm.Normal("mu", mu=0.0, sigma=5.0)
                tau = pm.HalfCauchy("tau", beta=5.0)
                theta_raw = pm.Normal("theta_raw", mu=0.0, sigma=1.0, shape=(J,))
                theta = pm.Deterministic("theta", mu + tau * theta_raw)
                pm.Normal("y", mu=theta, sigma=sigma_obs, observed=y_obs)
                t0 = time.perf_counter()
                idata = pm.sample(
                    draws=int(cfg["n_samples"]),
                    tune=int(cfg["n_warmup"]),
                    chains=int(cfg["n_chains"]),
                    cores=max(1, int(cfg["n_chains"])),
                    random_seed=int(cfg["seed"]),
                    target_accept=float(cfg["target_accept"]),
                    nuts={"max_treedepth": int(cfg["max_treedepth"])},
                    progressbar=False,
                )
                wall = time.perf_counter() - t0

        df = az.summary(idata, round_to=None)
        r_hat_map = _series_to_float_dict(df.get("r_hat", {}))
        ess_bulk_map = _series_to_float_dict(df.get("ess_bulk", {}))
        ess_tail_map = _series_to_float_dict(df.get("ess_tail", {}))

        ess_bulk_vals = list(ess_bulk_map.values())
        ess_tail_vals = list(ess_tail_map.values())
        max_r_hat = max(r_hat_map.values()) if r_hat_map else None
        min_ess_bulk = min(ess_bulk_vals) if ess_bulk_vals else None
        min_ess_tail = min(ess_tail_vals) if ess_tail_vals else None

        divergence_rate: float | None = None
        max_treedepth_rate: float | None = None
        try:
            div = idata.sample_stats.get("diverging")
            if div is not None:
                divergence_rate = float(div.mean().to_numpy())
        except Exception:
            pass
        try:
            td = idata.sample_stats.get("tree_depth")
            if td is not None:
                max_treedepth_rate = float((td >= int(cfg["max_treedepth"])).mean().to_numpy())
        except Exception:
            pass

        min_ebfmi: float | None = None
        try:
            bfmi = az.bfmi(idata)
            vals = [v for v in (float(x) for x in bfmi) if math.isfinite(v)]
            min_ebfmi = min(vals) if vals else None
        except Exception:
            pass

        summary = {
            "divergence_rate": divergence_rate,
            "max_treedepth_rate": max_treedepth_rate,
            "max_r_hat": max_r_hat,
            "min_ess_bulk": min_ess_bulk,
            "min_ess_tail": min_ess_tail,
            "min_ebfmi": min_ebfmi,
        }
        timing = {
            "wall_time_s": float(wall),
            "ess_bulk_per_sec": _ess_per_sec(ess_vals=ess_bulk_vals, wall_time_s=float(wall)),
            "ess_tail_per_sec": _ess_per_sec(ess_vals=ess_tail_vals, wall_time_s=float(wall)),
        }

        doc = _base_doc(args=args, nextstat_version=nextstat_version, dataset=dataset, model_type=model_type, cfg=cfg)
        doc.update(
            {
                "status": "ok",
                "backend_meta": backend_meta,
                "timing": timing,
                "diagnostics_summary": summary,
                "diagnostics": {"r_hat": r_hat_map, "ess_bulk": ess_bulk_map, "ess_tail": ess_tail_map},
            }
        )
        _write_json(out_path, doc)
        return 0

    if args.backend == "numpyro":
        try:
            import arviz as az  # type: ignore
            import jax  # type: ignore
            import jax.numpy as jnp  # type: ignore
            import numpyro  # type: ignore
            import numpyro.distributions as dist  # type: ignore
            from numpyro.infer import MCMC, NUTS  # type: ignore
        except Exception as e:
            return _emit_missing_backend(out_path=out_path, args=args, nextstat_version=nextstat_version, dataset=dataset, model_type=model_type, cfg=cfg, backend_name="numpyro", err=e)

        backend_meta: dict[str, Any] = {
            "numpyro_version": str(getattr(numpyro, "__version__", "unknown")),
            "jax_version": str(getattr(jax, "__version__", "unknown")),
        }

        import numpy as np  # type: ignore

        if args.model == "glm_logistic":
            assert generated is not None
            x_np = np.asarray(generated["x"], dtype=np.float64)
            y_np = np.asarray(generated["y"], dtype=np.int32)

            def _numpyro_glm_logistic(X, y_obs=None):
                N, P = X.shape
                alpha = numpyro.sample("alpha", dist.Normal(0.0, 1.0))
                beta = numpyro.sample("beta", dist.Normal(jnp.zeros(P), jnp.ones(P)).to_event(1))
                logits = alpha + X @ beta
                numpyro.sample("y", dist.BernoulliLogits(logits), obs=y_obs)

            model_fn = _numpyro_glm_logistic
            model_args = (jnp.array(x_np),)
            model_kwargs = {"y_obs": jnp.array(y_np)}

        elif args.model == "hier_random_intercept":
            assert generated is not None
            x_np = np.asarray([float(row[0]) for row in generated["x"]], dtype=np.float64)
            y_np = np.asarray(generated["y"], dtype=np.int32)
            g_np = np.asarray(generated["group_idx"], dtype=np.int32)
            g_n = int(generated["n_groups"])

            def _numpyro_hier(x, g, g_n, y_obs=None):
                alpha0 = numpyro.sample("alpha0", dist.Normal(0.0, 1.0))
                beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
                sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(1.0))
                alpha_raw = numpyro.sample("alpha_raw", dist.Normal(jnp.zeros(g_n), jnp.ones(g_n)).to_event(1))
                alpha = alpha0 + sigma_alpha * alpha_raw
                logits = alpha[g] + beta * x
                numpyro.sample("y", dist.BernoulliLogits(logits), obs=y_obs)

            model_fn = _numpyro_hier
            model_args = (jnp.array(x_np), jnp.array(g_np), g_n)
            model_kwargs = {"y_obs": jnp.array(y_np)}

        else:
            # eight_schools (non-centered)
            assert generated is not None
            J = int(generated["J"])
            y_obs_np = np.asarray(generated["y"], dtype=np.float64)
            sigma_np = np.asarray(generated["sigma"], dtype=np.float64)

            def _numpyro_eight_schools(J, sigma, y_obs=None):
                mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
                tau = numpyro.sample("tau", dist.HalfCauchy(5.0))
                theta_raw = numpyro.sample("theta_raw", dist.Normal(jnp.zeros(J), jnp.ones(J)).to_event(1))
                theta = mu + tau * theta_raw
                numpyro.sample("y", dist.Normal(theta, sigma), obs=y_obs)

            model_fn = _numpyro_eight_schools
            model_args = (J, jnp.array(sigma_np))
            model_kwargs = {"y_obs": jnp.array(y_obs_np)}

        kernel = NUTS(model_fn, target_accept_prob=float(cfg["target_accept"]), max_tree_depth=int(cfg["max_treedepth"]))
        mcmc = MCMC(
            kernel,
            num_warmup=int(cfg["n_warmup"]),
            num_samples=int(cfg["n_samples"]),
            num_chains=int(cfg["n_chains"]),
            chain_method="sequential",
            progress_bar=False,
        )

        rng_key = jax.random.PRNGKey(int(cfg["seed"]))

        t0 = time.perf_counter()
        try:
            mcmc.run(rng_key, *model_args, **model_kwargs)
            wall = time.perf_counter() - t0
        except Exception as e:
            wall = time.perf_counter() - t0
            doc = _base_doc(args=args, nextstat_version=nextstat_version, dataset=dataset, model_type=model_type, cfg=cfg)
            doc.update(
                {
                    "status": "failed",
                    "reason": f"numpyro_sample_failed:{type(e).__name__}:{e}",
                    "backend_meta": backend_meta,
                    "timing": {"wall_time_s": float(wall)},
                    "diagnostics_summary": {
                        "divergence_rate": None,
                        "max_treedepth_rate": None,
                        "max_r_hat": None,
                        "min_ess_bulk": None,
                        "min_ess_tail": None,
                        "min_ebfmi": None,
                    },
                }
            )
            _write_json(out_path, doc)
            return 2

        idata = az.from_numpyro(mcmc)

        df = az.summary(idata, round_to=None)
        r_hat_map = _series_to_float_dict(df.get("r_hat", {}))
        ess_bulk_map = _series_to_float_dict(df.get("ess_bulk", {}))
        ess_tail_map = _series_to_float_dict(df.get("ess_tail", {}))

        ess_bulk_vals = list(ess_bulk_map.values())
        ess_tail_vals = list(ess_tail_map.values())
        max_r_hat = max(r_hat_map.values()) if r_hat_map else None
        min_ess_bulk = min(ess_bulk_vals) if ess_bulk_vals else None
        min_ess_tail = min(ess_tail_vals) if ess_tail_vals else None

        divergence_rate: float | None = None
        max_treedepth_rate: float | None = None
        try:
            div = idata.sample_stats.get("diverging")
            if div is not None:
                divergence_rate = float(div.mean().to_numpy())
        except Exception:
            pass
        try:
            td = idata.sample_stats.get("tree_depth")
            if td is not None:
                max_treedepth_rate = float((td >= int(cfg["max_treedepth"])).mean().to_numpy())
        except Exception:
            pass

        min_ebfmi: float | None = None
        try:
            bfmi = az.bfmi(idata)
            vals = [v for v in (float(x) for x in bfmi) if math.isfinite(v)]
            min_ebfmi = min(vals) if vals else None
        except Exception:
            pass

        summary = {
            "divergence_rate": divergence_rate,
            "max_treedepth_rate": max_treedepth_rate,
            "max_r_hat": max_r_hat,
            "min_ess_bulk": min_ess_bulk,
            "min_ess_tail": min_ess_tail,
            "min_ebfmi": min_ebfmi,
        }
        timing = {
            "wall_time_s": float(wall),
            "ess_bulk_per_sec": _ess_per_sec(ess_vals=ess_bulk_vals, wall_time_s=float(wall)),
            "ess_tail_per_sec": _ess_per_sec(ess_vals=ess_tail_vals, wall_time_s=float(wall)),
        }

        doc = _base_doc(args=args, nextstat_version=nextstat_version, dataset=dataset, model_type=model_type, cfg=cfg)
        doc.update(
            {
                "status": "ok",
                "backend_meta": backend_meta,
                "timing": timing,
                "diagnostics_summary": summary,
                "diagnostics": {"r_hat": r_hat_map, "ess_bulk": ess_bulk_map, "ess_tail": ess_tail_map},
            }
        )
        _write_json(out_path, doc)
        return 0

    doc = _base_doc(args=args, nextstat_version=nextstat_version, dataset=dataset, model_type=model_type, cfg=cfg)
    doc.update(
        {
            "status": "failed",
            "reason": f"unknown_backend:{args.backend}",
            "timing": {"wall_time_s": 0.0},
            "diagnostics_summary": {
                "divergence_rate": None,
                "max_treedepth_rate": None,
                "max_r_hat": None,
                "min_ess_bulk": None,
                "min_ess_tail": None,
                "min_ebfmi": None,
            },
        }
    )
    _write_json(out_path, doc)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
