#!/usr/bin/env python3
"""MAMS benchmark runner: NextStat MAMS vs NUTS vs BlackJAX adjusted_mclmc.

Backends:
- `nextstat_mams` — NextStat MAMS (microcanonical dynamics)
- `nextstat_nuts` — NextStat NUTS (baseline)
- `blackjax_mclmc` — BlackJAX adjusted_mclmc (JAX backend: CPU or GPU)

Cases:
- `std_normal_10d` — N(0, I_10)
- `neal_funnel_2d`  — Neal's funnel (y~N(0,3), x|y~N(0,e^y))
- `eight_schools`   — Non-centered 8-schools hierarchical
- `glm_logistic`    — Synthetic logistic regression (n=200, p=5)

Writes a single JSON artifact per invocation.
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
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.bench_env import collect_environment


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def sha256_json_obj(obj: Any) -> str:
    b = (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _diag_summary_nextstat(diag: dict[str, Any]) -> dict[str, float | None]:
    r_hat = diag.get("r_hat") if isinstance(diag.get("r_hat"), dict) else {}
    ess_bulk = diag.get("ess_bulk") if isinstance(diag.get("ess_bulk"), dict) else {}
    ess_tail = diag.get("ess_tail") if isinstance(diag.get("ess_tail"), dict) else {}

    rhat_vals = [v for v in (_safe_float(x) for x in r_hat.values()) if v is not None]
    ess_bulk_vals = [v for v in (_safe_float(x) for x in ess_bulk.values()) if v is not None]
    ess_tail_vals = [v for v in (_safe_float(x) for x in ess_tail.values()) if v is not None]

    return {
        "max_r_hat": max(rhat_vals) if rhat_vals else None,
        "min_ess_bulk": min(ess_bulk_vals) if ess_bulk_vals else None,
        "min_ess_tail": min(ess_tail_vals) if ess_tail_vals else None,
    }


def _logistic(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def make_logistic_dataset(*, n: int = 200, p: int = 5, seed: int = 42) -> dict[str, Any]:
    rng = random.Random(seed)
    beta = [0.6, -1.1, 0.3, 0.0, 0.8][:p]
    intercept = -0.2
    x: list[list[float]] = []
    y: list[int] = []
    for _ in range(n):
        row = [rng.gauss(0.0, 1.0) for _ in range(p)]
        z = intercept + sum(b * v for b, v in zip(beta, row))
        pr = _logistic(z)
        yi = 1 if rng.random() < pr else 0
        x.append(row)
        y.append(yi)
    return {"x": x, "y": y, "n": n, "p": p, "seed": seed, "beta": beta, "intercept": intercept}


def _extract_stats(stats: dict[str, Any]) -> tuple[int | None, float | None]:
    """Extract total gradient evals and mean accept from sample_stats.

    ``sample_stats`` stores per-chain lists: ``n_leapfrog[chain][sample]``,
    ``accept_prob[chain][sample]``.
    """
    # n_leapfrog
    nl = stats.get("n_leapfrog")
    total_grad: int | None = None
    if isinstance(nl, list) and nl:
        if isinstance(nl[0], list):
            total_grad = sum(int(v) for chain in nl for v in chain)
        else:
            total_grad = sum(int(v) for v in nl)

    # accept_prob
    ap = stats.get("accept_prob")
    mean_accept: float | None = None
    if isinstance(ap, list) and ap:
        if isinstance(ap[0], list):
            flat = [float(v) for chain in ap for v in chain]
        else:
            flat = [float(v) for v in ap]
        if flat:
            mean_accept = sum(flat) / len(flat)

    return total_grad, mean_accept


def _posterior_summary_nextstat(raw: dict[str, Any]) -> dict[str, Any]:
    """Summarize posterior draws for cross-sampler parity checks.

    Uses only stdlib to keep the suite lightweight (no numpy/arviz required).
    """
    posterior = raw.get("posterior")
    param_names = raw.get("param_names")
    diag = raw.get("diagnostics") if isinstance(raw.get("diagnostics"), dict) else {}

    if not isinstance(posterior, dict) or not isinstance(param_names, list):
        return {"status": "missing"}

    r_hat = diag.get("r_hat") if isinstance(diag.get("r_hat"), dict) else {}
    ess_bulk = diag.get("ess_bulk") if isinstance(diag.get("ess_bulk"), dict) else {}
    ess_tail = diag.get("ess_tail") if isinstance(diag.get("ess_tail"), dict) else {}

    params: list[dict[str, Any]] = []
    n_total_draws = 0
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
        n_total_draws = max(n_total_draws, n)
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
                "ess_tail": _safe_float(ess_tail.get(name)),
                "r_hat": _safe_float(r_hat.get(name)),
            }
        )

    return {"status": "ok", "n_draws": n_total_draws, "params": params}


# ---------------------------------------------------------------------------
# NextStat backends
# ---------------------------------------------------------------------------

def _run_nextstat_backend(model_obj: Any, cfg: dict[str, Any], *, method: str) -> dict[str, Any]:
    import nextstat  # type: ignore

    def _run_once(seed: int) -> tuple[float, dict[str, Any]]:
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = dict(
            n_chains=cfg["n_chains"],
            n_warmup=cfg["n_warmup"],
            n_samples=cfg["n_samples"],
            seed=seed,
            target_accept=cfg["target_accept"],
        )
        if method == "mams":
            kwargs["method"] = "mams"
        r = nextstat.sample(model_obj, **kwargs)
        wall = time.perf_counter() - t0
        return wall, r

    cold_wall, _ = _run_once(int(cfg["seed"]))
    warm_wall, r = _run_once(int(cfg["seed"]) + 1)

    diag = r.get("diagnostics") if isinstance(r.get("diagnostics"), dict) else {}
    summary = _diag_summary_nextstat(diag)

    stats = r.get("sample_stats") if isinstance(r.get("sample_stats"), dict) else {}
    total_grad, mean_accept = _extract_stats(stats)

    return {
        "wall_time_s": cold_wall,
        "cold_start_s": cold_wall,
        "warm_start_s": warm_wall,
        "jit_overhead_s": max(0.0, cold_wall - warm_wall),
        "n_grad_evals": total_grad,
        "min_ess_bulk": summary["min_ess_bulk"],
        "min_ess_tail": summary["min_ess_tail"],
        "max_r_hat": summary["max_r_hat"],
        "accept_rate": mean_accept,
        "posterior_summary": _posterior_summary_nextstat(r),
    }


def _run_nextstat_mams(model_obj: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    return _run_nextstat_backend(model_obj, cfg, method="mams")


def _run_nextstat_nuts(model_obj: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    return _run_nextstat_backend(model_obj, cfg, method="nuts")


# ---------------------------------------------------------------------------
# BlackJAX backend
# ---------------------------------------------------------------------------

def _blackjax_log_density(case: str, dataset: dict[str, Any] | None):
    """Return (log_density_fn, dim) for BlackJAX."""
    import jax
    jax.config.update("jax_enable_x64", True)  # match NextStat f64 precision
    import jax.numpy as jnp

    if case == "std_normal_10d":
        dim = 10
        def logp(x):
            return -0.5 * jnp.sum(x ** 2)
        return logp, dim

    elif case == "neal_funnel_2d":
        dim = 2
        def logp(x):
            y, z = x[0], x[1]
            log_py = -0.5 * y * y / 9.0
            log_pz = -0.5 * z * z * jnp.exp(-y) - 0.5 * y
            return log_py + log_pz
        return logp, dim

    elif case == "eight_schools":
        dim = 10  # mu, log_tau, theta_raw[8]
        y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
        sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
        def logp(x):
            mu = x[0]
            log_tau = x[1]
            tau = jnp.exp(log_tau)
            theta_raw = x[2:]
            theta = mu + tau * theta_raw
            lp = -0.5 * mu * mu / 25.0  # N(0,5) prior on mu
            lp += -0.5 * log_tau * log_tau / 25.0 + log_tau  # HalfNormal(5) on tau, Jacobian
            lp += -0.5 * jnp.sum(theta_raw ** 2)  # N(0,1) on theta_raw
            lp += -0.5 * jnp.sum(((y - theta) / sigma) ** 2)  # likelihood
            return lp
        return logp, dim

    elif case == "glm_logistic":
        assert dataset is not None
        dim = int(dataset["p"]) + 1  # intercept + coefs
        x_data = jnp.array(dataset["x"])
        y_data = jnp.array(dataset["y"], dtype=jnp.float64)
        def logp(params):
            intercept = params[0]
            beta = params[1:]
            logits = x_data @ beta + intercept
            ll = jnp.sum(y_data * logits - jnp.log1p(jnp.exp(logits)))
            prior = -0.5 * jnp.sum(params ** 2)
            return ll + prior
        return logp, dim

    else:
        raise ValueError(f"Unknown case for BlackJAX: {case}")


def _run_blackjax_mclmc(case: str, cfg: dict[str, Any], dataset: dict[str, Any] | None) -> dict[str, Any]:
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import blackjax
    except ImportError as e:
        return {"status": "warn", "reason": f"missing_dep:{e}"}

    logp, dim = _blackjax_log_density(case, dataset)
    n_chains = cfg["n_chains"]
    n_samples = cfg["n_samples"]

    def _do_full_run(seed: int) -> tuple:
        """Run full pipeline: init → warmup → scan sampling.

        Returns (state, params, positions, n_int_steps, wall_time).
        positions: (n_chains, n_samples, dim) jax array.
        """
        key = jax.random.PRNGKey(seed)
        init_key, warmup_key, sample_key = jax.random.split(key, 3)
        initial_position = 0.1 * jax.random.normal(init_key, (dim,))

        t0 = time.perf_counter()

        # Init
        init_state = blackjax.adjusted_mclmc.init(
            position=initial_position,
            logdensity_fn=logp,
        )

        # Warmup: tune step_size, L, inv_mass
        raw_kernel = blackjax.adjusted_mclmc.build_kernel(logdensity_fn=logp)

        def warmup_kernel(rng_key, state, step_size, num_integration_steps=10,
                          L_proposal_factor=jnp.inf, **_kwargs):
            return raw_kernel(rng_key, state, step_size, num_integration_steps,
                              L_proposal_factor)

        bj_state, bj_params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
            mclmc_kernel=warmup_kernel,
            num_steps=cfg["n_warmup"],
            state=init_state,
            rng_key=warmup_key,
            target=cfg["target_accept"],
            diagonal_preconditioning=True,
        )

        # Build sampling kernel with tuned params
        L_val = float(bj_params.L)
        step_size_val = float(bj_params.step_size)
        n_int_steps = max(1, int(round(L_val / step_size_val)))

        sampling_alg = blackjax.adjusted_mclmc(
            logdensity_fn=logp,
            step_size=step_size_val,
            num_integration_steps=n_int_steps,
            inverse_mass_matrix=bj_params.inverse_mass_matrix,
        )

        # jax.lax.scan for sequential draws, vmap across chains
        def step_fn(state, rng_key):
            new_state, _ = sampling_alg.step(rng_key, state)
            return new_state, new_state.position

        @jax.jit
        def sample_all_chains(init_st, all_keys):
            def scan_one_chain(keys):
                _, positions = jax.lax.scan(step_fn, init_st, keys)
                return positions
            return jax.vmap(scan_one_chain)(all_keys)

        # Keys: (n_chains, n_samples)
        chain_keys = jax.random.split(sample_key, n_chains)
        per_chain_keys = jax.vmap(lambda k: jax.random.split(k, n_samples))(chain_keys)

        # Execute (triggers JIT on first call, cached on second)
        positions = sample_all_chains(bj_state, per_chain_keys)
        jax.block_until_ready(positions)

        wall = time.perf_counter() - t0
        return bj_state, bj_params, positions, n_int_steps, wall

    try:
        # ---- Cold start: includes all JIT compilation ----
        _, _, positions_cold, n_int_steps, cold_wall = _do_full_run(cfg["seed"])

        # ---- Warm start: JIT cached, re-run with different seed ----
        _, _, positions_warm, _, warm_wall = _do_full_run(cfg["seed"] + 1)

        # Use warm-start draws for ESS (statistically independent from cold)
        import numpy as np
        data = np.array(jax.device_get(positions_warm))  # (n_chains, n_samples, dim)

        total_grad = n_int_steps * n_samples * n_chains

        try:
            import arviz
            ess_bulk_obj = arviz.ess({"x": data}, method="bulk")
            ess_tail_obj = arviz.ess({"x": data}, method="tail")
            r_hat_obj = arviz.rhat({"x": data})
            min_ess_bulk = float(np.min(ess_bulk_obj["x"].values))
            min_ess_tail = float(np.min(ess_tail_obj["x"].values))
            max_r_hat = float(np.max(r_hat_obj["x"].values))
        except Exception:
            min_ess_bulk = None
            min_ess_tail = None
            max_r_hat = None

        return {
            "wall_time_s": cold_wall,
            "cold_start_s": cold_wall,
            "warm_start_s": warm_wall,
            "jit_overhead_s": cold_wall - warm_wall,
            "n_grad_evals": total_grad,
            "n_integration_steps": n_int_steps,
            "min_ess_bulk": min_ess_bulk,
            "min_ess_tail": min_ess_tail,
            "max_r_hat": max_r_hat,
            "accept_rate": None,
        }
    except Exception as e:
        import traceback
        return {"status": "failed", "reason": f"{type(e).__name__}:{e}\n{traceback.format_exc()}", "wall_time_s": 0.0}


# ---------------------------------------------------------------------------
# Case setup
# ---------------------------------------------------------------------------

def _build_nextstat_model(case: str, dataset: dict[str, Any] | None):
    import nextstat  # type: ignore

    if case == "std_normal_10d":
        return nextstat.StdNormalModel(dim=10)
    elif case == "neal_funnel_2d":
        return nextstat.FunnelModel()
    elif case == "eight_schools":
        y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
        sigma = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
        return nextstat.EightSchoolsModel(y=y, sigma=sigma)
    elif case == "glm_logistic":
        assert dataset is not None
        spec = nextstat.data.GlmSpec.logistic_regression(
            x=dataset["x"],
            y=dataset["y"],
            include_intercept=True,
            coef_prior_mu=0.0,
            coef_prior_sigma=1.0,
        )
        return spec.build()
    else:
        raise ValueError(f"Unknown case: {case}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="MAMS benchmark runner")
    ap.add_argument("--case", required=True,
                    choices=["std_normal_10d", "neal_funnel_2d", "eight_schools", "glm_logistic"])
    ap.add_argument("--backend", required=True,
                    choices=["nextstat_mams", "nextstat_nuts", "blackjax_mclmc"])
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = {
        "n_chains": args.n_chains,
        "n_warmup": args.warmup,
        "n_samples": args.samples,
        "seed": args.seed,
        "target_accept": args.target_accept,
    }

    # Dataset (only for glm_logistic)
    dataset: dict[str, Any] | None = None
    if args.case == "glm_logistic":
        dataset = make_logistic_dataset(seed=args.seed)

    dataset_sha = sha256_json_obj(dataset) if dataset else sha256_json_obj({"case": args.case})

    try:
        import nextstat  # type: ignore
        ns_version = str(getattr(nextstat, "__version__", "unknown"))
    except ImportError:
        ns_version = "unknown"

    # Collect backend-specific metadata
    backend_meta: dict[str, str] = {}
    if args.backend == "blackjax_mclmc":
        try:
            import blackjax
            backend_meta["blackjax_version"] = str(getattr(blackjax, "__version__", "unknown"))
        except ImportError:
            pass
        try:
            import jax
            backend_meta["jax_version"] = str(jax.__version__)
            backend_meta["jax_x64_enabled"] = "true"
            backend_meta["jax_devices"] = str(jax.devices())
        except ImportError:
            pass
        try:
            import arviz
            backend_meta["arviz_version"] = str(arviz.__version__)
        except ImportError:
            pass
        backend_meta["sampler"] = "adjusted_mclmc (arXiv:2503.01707)"
        backend_meta["warmup_method"] = "adjusted_mclmc_find_L_and_step_size"
        backend_meta["ess_method"] = "arviz bulk/tail"

    base = {
        "schema_version": "nextstat.mams_benchmark_result.v1",
        "suite": "mams",
        "case": args.case,
        "backend": args.backend,
        "deterministic": bool(args.deterministic),
        "environment": collect_environment(),
        "meta": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "nextstat_version": ns_version,
            **backend_meta,
        },
        "dataset": {"id": args.case, "sha256": dataset_sha},
        "config": cfg,
    }

    # Run backend
    try:
        if args.backend in ("nextstat_mams", "nextstat_nuts"):
            import nextstat  # type: ignore
            model_obj = _build_nextstat_model(args.case, dataset)

            if args.backend == "nextstat_mams":
                result = _run_nextstat_mams(model_obj, cfg)
            else:
                result = _run_nextstat_nuts(model_obj, cfg)
        elif args.backend == "blackjax_mclmc":
            result = _run_blackjax_mclmc(args.case, cfg, dataset)
        else:
            raise ValueError(f"Unknown backend: {args.backend}")
    except Exception as e:
        base["status"] = "failed"
        base["reason"] = f"{type(e).__name__}:{e}"
        base["timing"] = {"wall_time_s": 0.0}
        base["metrics"] = {}
        _write_json(out_path, base)
        return 2

    # Handle warn/failed from backend
    if isinstance(result, dict) and "status" in result:
        base["status"] = result["status"]
        base["reason"] = result.get("reason", "")
        base["timing"] = {"wall_time_s": result.get("wall_time_s", 0.0)}
        base["metrics"] = {}
        _write_json(out_path, base)
        return 0 if result["status"] == "warn" else 2

    # Compute derived metrics
    ess_bulk = _safe_float(result.get("min_ess_bulk"))
    n_grad = result.get("n_grad_evals")
    wall = result.get("wall_time_s", 0.0)

    ess_per_grad = None
    if ess_bulk is not None and n_grad is not None and n_grad > 0:
        ess_per_grad = ess_bulk / n_grad

    ess_per_sec = None
    if ess_bulk is not None and wall > 0:
        ess_per_sec = ess_bulk / wall

    base["status"] = "ok"

    # Timing section — includes cold/warm breakdown for JAX backends
    timing: dict[str, Any] = {"wall_time_s": wall}
    for k in ("cold_start_s", "warm_start_s", "jit_overhead_s"):
        v = _safe_float(result.get(k))
        if v is not None:
            timing[k] = v
    base["timing"] = timing

    # ESS/s uses warm_start_s when available (fair post-JIT comparison)
    warm_wall = _safe_float(result.get("warm_start_s"))
    ess_per_sec_warm = None
    if ess_bulk is not None and warm_wall and warm_wall > 0:
        ess_per_sec_warm = ess_bulk / warm_wall

    base["metrics"] = {
        "wall_time_s": wall,
        "cold_start_s": _safe_float(result.get("cold_start_s")),
        "warm_start_s": _safe_float(result.get("warm_start_s")),
        "jit_overhead_s": _safe_float(result.get("jit_overhead_s")),
        "n_grad_evals": n_grad,
        "n_integration_steps": result.get("n_integration_steps"),
        "min_ess_bulk": ess_bulk,
        "min_ess_tail": _safe_float(result.get("min_ess_tail")),
        "ess_per_grad": ess_per_grad,
        "ess_per_sec": ess_per_sec,
        "ess_per_sec_warm": ess_per_sec_warm,
        "max_r_hat": _safe_float(result.get("max_r_hat")),
        "accept_rate": _safe_float(result.get("accept_rate")),
    }
    if isinstance(result.get("posterior_summary"), dict):
        base["metrics"]["posterior_summary"] = result.get("posterior_summary")

    _write_json(out_path, base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
