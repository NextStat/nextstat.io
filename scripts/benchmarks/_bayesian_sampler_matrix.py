#!/usr/bin/env python3
"""Shared sampler-matrix harness for NextStat Bayesian benchmark scripts.

This harness is for internal sampler discovery and admission review.
Its policy gates are explicit and intentionally stricter than some product-facing
quality contracts; artifacts from this harness must not be read as official
sampler-health verdicts unless that policy is explicitly aligned.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from _bayesian_nextstat_bench import (
    BenchResult,
    DEFAULT_N_CHAINS,
    DEFAULT_N_SAMPLES,
    DEFAULT_N_WARMUP,
    DEFAULT_SEEDS,
    _assert_nextstat_harness_contract,
    _ns_available,
    collect_environment,
    leapfrogs_per_sec,
    max_rhat,
    median_divergence_rate,
    median_mean_accept_prob,
    median_mean_tree_depth,
    median_min_ebfmi,
    median_wall,
    min_ess_per_leapfrog,
    min_ess_per_sec,
    run_nextstat_eight_schools,
    run_nextstat_funnel,
    run_nextstat_funnel_ncp,
    run_nextstat_glm_logistic,
    run_nextstat_glm_negbin,
    run_nextstat_glm_poisson,
    run_nextstat_std_normal,
)


METHOD_ALIASES = {
    "nuts": "NUTS",
    "walnuts": "WALNUTS",
    "mams": "MAMS",
}

DISCOVERY_POLICY = {
    "name": "internal_sampler_matrix_discovery_policy_v1",
    "note": (
        "Internal discovery gate only. Not equivalent to the core diagnostics "
        "contract or the official public sampler suites."
    ),
    "max_divergence_rate": 0.01,
    "max_rhat": 1.01,
    "min_ebfmi": 0.30,
}

CANONICAL_ADMISSION_POLICY = {
    "name": "walnuts_canonical_expansion_admission_v1",
    "note": (
        "Internal policy for deciding whether a posterior family can graduate "
        "from exploratory review into the canonical WALNUTS-vs-NUTS set."
    ),
    "required_contract": {
        "host": "nextstat-bench",
        "runner": "scripts/benchmarks/bench_walnuts_vs_nuts.py",
        "methods": ["nuts", "walnuts"],
        "seeds": [42, 123, 777],
        "n_chains": 4,
        "n_warmup": 1000,
        "n_samples": 1000,
        "metric": "diagonal",
        "target_accept": 0.8,
        "max_treedepth": 10,
        "uses_shipped_product_defaults": True,
    },
    "required_quality_thresholds": {
        "max_divergence_rate": 0.01,
        "max_rhat": 1.01,
        "min_ebfmi": 0.30,
    },
    "fixture_requirements": {
        "reproducible_fixture": True,
        "explicit_priors": True,
        "artifact_must_record_method_specific_config": True,
    },
    "representativeness_requirement": (
        "Candidate must add a materially new posterior geometry class rather "
        "than duplicate an already admitted family."
    ),
    "runtime_budget": {
        "max_two_method_three_seed_wall_secs": 600.0,
    },
}

MODEL_DESCRIPTIONS = {
    "std_normal_10d": "Exact match across methods",
    "eight_schools": "Same non-centered Eight Schools target density across methods",
    "glm_logistic": "Same logistic regression posterior with alpha~N(0,5), beta_j~N(0,2.5) across methods",
    "funnel_ncp_10d": "Same non-centered Neal funnel target density across methods",
    "glm_poisson": "Same Poisson regression posterior with intercept~N(0,5), beta_j~N(0,2.5) across methods",
    "glm_negbin": "Same Negative Binomial regression posterior with intercept~N(0,5), beta_j~N(0,2.5), log_alpha~N(0,1) across methods",
    "funnel_10d": "Same centered Neal funnel target density across methods",
}


@dataclass(frozen=True)
class SamplerMatrixConfig:
    methods: tuple[str, ...]
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    out_dir: Path
    n_chains: int
    n_warmup: int
    n_samples: int
    metric: str
    glm_n: int
    glm_p: int
    target_accept: float
    max_treedepth: int
    mams_max_leapfrog: int
    mams_diagonal_precond: bool


Runner = Callable[[int, str], BenchResult]


def _ratio(num: float, den: float) -> float | None:
    return (num / den) if den > 0 else None


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def _metric_suffix(metric_key: str) -> str:
    mapping = {
        "ess_bulk_per_sec": "ESS_bulk/s",
        "ess_tail_per_sec": "ESS_tail/s",
        "ess_bulk_per_leapfrog": "ESS_bulk/LF",
        "ess_tail_per_leapfrog": "ESS_tail/LF",
        "leapfrogs_per_sec": "LF/s",
        "median_wall_secs": "wall",
    }
    return mapping.get(metric_key, metric_key)


def parse_methods(spec: str, *, allowed_methods: tuple[str, ...] = tuple(METHOD_ALIASES)) -> list[str]:
    methods: list[str] = []
    seen: set[str] = set()
    for part in spec.split(","):
        method = part.strip()
        if not method or method in seen:
            continue
        if method not in allowed_methods:
            raise SystemExit(
                f"Unknown methods: {[method]}. Allowed: {sorted(allowed_methods)}"
            )
        seen.add(method)
        methods.append(method)
    if not methods:
        raise SystemExit("At least one method is required")
    return methods


def parse_models(spec: str, *, allowed_models: tuple[str, ...]) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for part in spec.split(","):
        model = part.strip()
        if not model or model in seen:
            continue
        if model not in allowed_models:
            raise SystemExit(
                f"Unknown models: {[model]}. Allowed: {sorted(allowed_models)}"
            )
        seen.add(model)
        models.append(model)
    if not models:
        raise SystemExit("At least one model is required")
    return models


def parse_seeds(spec: str) -> list[int]:
    seeds = [int(seed) for seed in spec.split(",") if seed.strip()]
    if not seeds:
        raise SystemExit("At least one seed is required")
    return seeds


def quality_summary(results: list[BenchResult]) -> dict[str, Any]:
    div = median_divergence_rate(results)
    rhat = max_rhat(results)
    ebfmi = median_min_ebfmi(results)
    sampling_leapfrogs = [
        int(result.n_leapfrog_sampling_total)
        for result in results
        if result.n_leapfrog_sampling_total is not None
    ]
    warmup_leapfrogs = [
        int(result.n_leapfrog_warmup_total)
        for result in results
        if result.n_leapfrog_warmup_total is not None
    ]
    total_leapfrogs = [
        int(result.n_leapfrog_total)
        for result in results
        if result.n_leapfrog_total is not None
    ]
    return {
        "median_wall_secs": median_wall(results),
        "ess_bulk_per_sec": min_ess_per_sec(results, "ess_bulk"),
        "ess_tail_per_sec": min_ess_per_sec(results, "ess_tail"),
        "ess_bulk_per_leapfrog": min_ess_per_leapfrog(results, "ess_bulk"),
        "ess_tail_per_leapfrog": min_ess_per_leapfrog(results, "ess_tail"),
        "leapfrogs_per_sec": leapfrogs_per_sec(results),
        "median_sampling_leapfrogs": int(statistics.median(sampling_leapfrogs)) if sampling_leapfrogs else None,
        "median_warmup_leapfrogs": int(statistics.median(warmup_leapfrogs)) if warmup_leapfrogs else None,
        "median_total_leapfrogs": int(statistics.median(total_leapfrogs)) if total_leapfrogs else None,
        "median_divergence_rate": div,
        "median_max_rhat": rhat if rhat == rhat else None,
        "median_min_ebfmi": ebfmi,
        "median_mean_tree_depth": median_mean_tree_depth(results),
        "median_mean_accept_prob": median_mean_accept_prob(results),
        "discovery_gate_passed": bool(
            div is not None
            and div <= DISCOVERY_POLICY["max_divergence_rate"]
            and rhat == rhat
            and rhat < DISCOVERY_POLICY["max_rhat"]
            and ebfmi is not None
            and ebfmi >= DISCOVERY_POLICY["min_ebfmi"]
        ),
    }


def pairwise_ratios(
    summaries_by_method: dict[str, dict[str, Any]],
    *,
    methods: list[str],
    baseline_method: str,
) -> dict[str, dict[str, float | None]]:
    comparisons: dict[str, dict[str, float | None]] = {}
    baseline = summaries_by_method[baseline_method]
    metric_keys = [
        "ess_bulk_per_sec",
        "ess_tail_per_sec",
        "ess_bulk_per_leapfrog",
        "ess_tail_per_leapfrog",
        "leapfrogs_per_sec",
        "median_wall_secs",
    ]
    for method in methods:
        if method == baseline_method:
            continue
        method_summary = summaries_by_method[method]
        comparisons[f"{method}_over_{baseline_method}"] = {
            key: _ratio(method_summary[key], baseline[key])
            for key in metric_keys
        }
    return comparisons


def method_sample_kwargs(config: SamplerMatrixConfig, method: str) -> dict[str, Any]:
    if method in {"nuts", "walnuts"}:
        return {"max_treedepth": config.max_treedepth}
    if method == "mams":
        return {
            "max_leapfrog": config.mams_max_leapfrog,
            "diagonal_precond": config.mams_diagonal_precond,
        }
    raise AssertionError(f"Unhandled method: {method}")


def _runner_map(config: SamplerMatrixConfig) -> dict[str, Runner]:
    return {
        "std_normal_10d": lambda seed, method: run_nextstat_std_normal(
            seed,
            dim=10,
            method=method,
            metric=config.metric,
            n_chains=config.n_chains,
            n_warmup=config.n_warmup,
            n_samples=config.n_samples,
            target_accept=config.target_accept,
            max_treedepth=(
                config.max_treedepth if method in {"nuts", "walnuts"} else None
            ),
            sample_kwargs=method_sample_kwargs(config, method),
        ),
        "eight_schools": lambda seed, method: run_nextstat_eight_schools(
            seed,
            method=method,
            metric=config.metric,
            n_chains=config.n_chains,
            n_warmup=config.n_warmup,
            n_samples=config.n_samples,
            target_accept=config.target_accept,
            max_treedepth=(
                config.max_treedepth if method in {"nuts", "walnuts"} else None
            ),
            sample_kwargs=method_sample_kwargs(config, method),
        ),
        "glm_logistic": lambda seed, method: run_nextstat_glm_logistic(
            seed,
            n=config.glm_n,
            p=config.glm_p,
            method=method,
            metric=config.metric,
            n_chains=config.n_chains,
            n_warmup=config.n_warmup,
            n_samples=config.n_samples,
            target_accept=config.target_accept,
            max_treedepth=(
                config.max_treedepth if method in {"nuts", "walnuts"} else None
            ),
            sample_kwargs=method_sample_kwargs(config, method),
        ),
        "funnel_ncp_10d": lambda seed, method: run_nextstat_funnel_ncp(
            seed,
            dim=10,
            method=method,
            metric=config.metric,
            n_chains=config.n_chains,
            n_warmup=config.n_warmup,
            n_samples=config.n_samples,
            target_accept=config.target_accept,
            max_treedepth=(
                config.max_treedepth if method in {"nuts", "walnuts"} else None
            ),
            sample_kwargs=method_sample_kwargs(config, method),
        ),
        "glm_poisson": lambda seed, method: run_nextstat_glm_poisson(
            seed,
            n=config.glm_n,
            p=config.glm_p,
            method=method,
            metric=config.metric,
            n_chains=config.n_chains,
            n_warmup=config.n_warmup,
            n_samples=config.n_samples,
            target_accept=config.target_accept,
            max_treedepth=(
                config.max_treedepth if method in {"nuts", "walnuts"} else None
            ),
            sample_kwargs=method_sample_kwargs(config, method),
        ),
        "glm_negbin": lambda seed, method: run_nextstat_glm_negbin(
            seed,
            n=config.glm_n,
            p=config.glm_p,
            method=method,
            metric=config.metric,
            n_chains=config.n_chains,
            n_warmup=config.n_warmup,
            n_samples=config.n_samples,
            target_accept=config.target_accept,
            max_treedepth=(
                config.max_treedepth if method in {"nuts", "walnuts"} else None
            ),
            sample_kwargs=method_sample_kwargs(config, method),
        ),
        "funnel_10d": lambda seed, method: run_nextstat_funnel(
            seed,
            dim=10,
            method=method,
            metric=config.metric,
            n_chains=config.n_chains,
            n_warmup=config.n_warmup,
            n_samples=config.n_samples,
            target_accept=config.target_accept,
            max_treedepth=(
                config.max_treedepth if method in {"nuts", "walnuts"} else None
            ),
            sample_kwargs=method_sample_kwargs(config, method),
        ),
    }


def run_sampler_matrix(config: SamplerMatrixConfig) -> tuple[dict[str, Any], dict[str, dict[str, list[BenchResult]]]]:
    if not _ns_available():
        raise SystemExit("nextstat not importable; build/install local ns-py first")
    _assert_nextstat_harness_contract()
    import nextstat  # type: ignore

    results_by_model: dict[str, dict[str, list[BenchResult]]] = {
        model: {method: [] for method in config.methods} for model in config.models
    }
    all_runs: list[dict[str, Any]] = []
    runners = _runner_map(config)

    print(f"[ns] module={getattr(nextstat, '__file__', 'unknown')}")
    print(f"[ns] version={getattr(nextstat, '__version__', 'unknown')}")

    for model in config.models:
        runner = runners[model]
        for seed in config.seeds:
            for method in config.methods:
                print(f"[{method}] {model} seed={seed} ...", end=" ", flush=True)
                result = runner(seed, method)
                results_by_model[model][method].append(result)
                all_runs.append(result.__dict__)
                print(f"{result.wall_secs:.2f}s")

    summary: dict[str, Any] = {}
    for model in config.models:
        per_method = {
            method: quality_summary(results_by_model[model][method])
            for method in config.methods
        }
        summary[model] = {
            **per_method,
            **pairwise_ratios(
                per_method,
                methods=list(config.methods),
                baseline_method=config.methods[0],
            ),
        }

    metadata = collect_environment(
        config.metric,
        config.glm_n,
        config.glm_p,
        list(config.seeds),
        models=list(config.models),
        methods=list(config.methods),
        n_chains=config.n_chains,
        n_warmup=config.n_warmup,
        n_samples=config.n_samples,
        target_accept=config.target_accept,
        max_treedepth=config.max_treedepth,
    )
    metadata["target_density_parity"] = {
        model: MODEL_DESCRIPTIONS[model] for model in config.models
    }
    metadata["method_specific_config"] = {
        method: method_sample_kwargs(config, method) for method in config.methods
    }
    metadata["discovery_policy"] = dict(DISCOVERY_POLICY)
    metadata["canonical_admission_policy"] = (
        dict(CANONICAL_ADMISSION_POLICY) if config.metric == "diagonal" else None
    )
    artifact = {
        "metadata": metadata,
        "summary": summary,
        "runs": all_runs,
    }
    return artifact, results_by_model


def format_matrix_table(
    summary: dict[str, Any],
    *,
    models: list[str],
    methods: list[str],
    baseline_method: str,
) -> str:
    header = ["Model"]
    for method in methods:
        header.append(f"{METHOD_ALIASES[method]} ESS_bulk/s")
    for method in methods:
        header.append(f"{METHOD_ALIASES[method]} discovery gate")
    for method in methods:
        if method == baseline_method:
            continue
        header.append(f"{METHOD_ALIASES[method]}/{METHOD_ALIASES[baseline_method]}")
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for model in models:
        row = [model]
        for method in methods:
            row.append(f"{summary[model][method]['ess_bulk_per_sec']:.0f}")
        for method in methods:
            row.append(str(summary[model][method]["discovery_gate_passed"]))
        for method in methods:
            if method == baseline_method:
                continue
            ratio_key = f"{method}_over_{baseline_method}"
            row.append(_format_ratio(summary[model][ratio_key]["ess_bulk_per_sec"]))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_matrix_markdown(
    artifact: dict[str, Any],
    *,
    config: SamplerMatrixConfig,
    output_stem: str,
    title: str,
) -> Path:
    summary = artifact["summary"]
    metadata = artifact["metadata"]
    md_path = config.out_dir / f"{output_stem}.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write(
            f"Seeds: {list(config.seeds)} | Chains: {config.n_chains} | Warmup: {config.n_warmup} | Samples: {config.n_samples}\n\n"
        )
        handle.write(f"Models: {list(config.models)}\n\n")
        handle.write(
            f"Methods: {list(config.methods)} | metric={config.metric} | target_accept={config.target_accept}\n\n"
        )
        handle.write(
            "Scope: internal sampler-discovery artifact only; not an official product-health verdict.\n\n"
        )
        handle.write("Method-specific config:\n")
        for method in config.methods:
            handle.write(f"- `{method}`: {metadata['method_specific_config'][method]}\n")
        handle.write("\n")
        handle.write("Discovery policy:\n")
        handle.write(f"- name: `{metadata['discovery_policy']['name']}`\n")
        handle.write(f"- note: {metadata['discovery_policy']['note']}\n")
        handle.write(
            f"- thresholds: divergence<={metadata['discovery_policy']['max_divergence_rate']}, "
            f"max_rhat<{metadata['discovery_policy']['max_rhat']}, "
            f"min_ebfmi>={metadata['discovery_policy']['min_ebfmi']}\n\n"
        )
        if metadata["canonical_admission_policy"] is None:
            handle.write("Canonical admission policy:\n")
            handle.write(
                "- not applicable for this run because canonical WALNUTS admission review is pinned to metric=diagonal\n\n"
            )
        else:
            handle.write("Canonical admission policy:\n")
            handle.write(f"- name: `{metadata['canonical_admission_policy']['name']}`\n")
            handle.write(f"- note: {metadata['canonical_admission_policy']['note']}\n")
            handle.write(
                f"- required contract: host={metadata['canonical_admission_policy']['required_contract']['host']}, "
                f"runner={metadata['canonical_admission_policy']['required_contract']['runner']}, "
                f"methods={metadata['canonical_admission_policy']['required_contract']['methods']}, "
                f"seeds={metadata['canonical_admission_policy']['required_contract']['seeds']}, "
                f"chains={metadata['canonical_admission_policy']['required_contract']['n_chains']}, "
                f"warmup={metadata['canonical_admission_policy']['required_contract']['n_warmup']}, "
                f"samples={metadata['canonical_admission_policy']['required_contract']['n_samples']}, "
                f"metric={metadata['canonical_admission_policy']['required_contract']['metric']}, "
                f"target_accept={metadata['canonical_admission_policy']['required_contract']['target_accept']}, "
                f"max_treedepth={metadata['canonical_admission_policy']['required_contract']['max_treedepth']}, "
                f"uses_product_defaults={metadata['canonical_admission_policy']['required_contract']['uses_shipped_product_defaults']}\n"
            )
            handle.write(
                f"- required quality thresholds: divergence<={metadata['canonical_admission_policy']['required_quality_thresholds']['max_divergence_rate']}, "
                f"max_rhat<{metadata['canonical_admission_policy']['required_quality_thresholds']['max_rhat']}, "
                f"min_ebfmi>={metadata['canonical_admission_policy']['required_quality_thresholds']['min_ebfmi']}\n"
            )
            handle.write(
                f"- fixture requirements: {metadata['canonical_admission_policy']['fixture_requirements']}\n"
            )
            handle.write(
                f"- runtime budget: two-method/three-seed wall<={metadata['canonical_admission_policy']['runtime_budget']['max_two_method_three_seed_wall_secs']}s\n"
            )
            handle.write(
                f"- representativeness: {metadata['canonical_admission_policy']['representativeness_requirement']}\n\n"
            )
        handle.write("Environment:\n")
        handle.write(f"- Python: {metadata.get('python')}\n")
        handle.write(
            f"- Platform: {metadata.get('platform', {}).get('system')} "
            f"{metadata.get('platform', {}).get('release')} "
            f"({metadata.get('platform', {}).get('machine')})\n"
        )
        handle.write(f"- Hostname: {metadata.get('hostname')}\n")
        handle.write(f"- Git commit: {metadata.get('git_commit')}\n")
        handle.write(f"- NextStat: {metadata.get('nextstat_version')}\n\n")
        lane = metadata.get("execution_lane", {})
        if isinstance(lane, dict) and any(lane.values()):
            handle.write("Execution lane:\n")
            if lane.get("host_policy"):
                handle.write(f"- host_policy: {lane.get('host_policy')}\n")
            if lane.get("submit_host"):
                handle.write(f"- submit_host: {lane.get('submit_host')}\n")
            if lane.get("execute_host"):
                handle.write(f"- execute_host: {lane.get('execute_host')}\n")
            if lane.get("scheduler"):
                handle.write(f"- scheduler: {lane.get('scheduler')}\n")
            handle.write("\n")
        accel = metadata.get("accelerator_runtime", {})
        if isinstance(accel, dict):
            handle.write("Accelerator runtime:\n")
            handle.write(
                f"- cuda_runtime_available: {accel.get('cuda_runtime_available')}\n"
            )
            handle.write(
                f"- metal_runtime_available: {accel.get('metal_runtime_available')}\n"
            )
            handle.write(f"- nvidia_smi_present: {accel.get('nvidia_smi_present')}\n")
            nvidia_gpus = accel.get("nvidia_gpus") or []
            if nvidia_gpus:
                handle.write(f"- nvidia_gpus: {nvidia_gpus}\n")
            else:
                handle.write("- nvidia_gpus: []\n")
            if not accel.get("cuda_runtime_available") and not accel.get("metal_runtime_available"):
                handle.write(
                    "- host_scope_note: this artifact exercises CPU sampler scope only; "
                    "GPU sampler certification is not covered on this runtime\n"
                )
            handle.write("\n")
        handle.write("Interpretation:\n")
        handle.write("- `ESS_bulk/s`: worst-parameter bulk ESS per wall-second (median over seeds)\n")
        handle.write("- `ESS_bulk/LF`: worst-parameter bulk ESS per post-warmup leapfrog/micro-step count\n")
        handle.write("- `LF/s`: total warmup+post-warmup leapfrogs per end-to-end wall-second\n")
        handle.write(
            "- `discovery_gate_passed`: internal discovery-policy check only; do not "
            "read it as the official sampler-health contract\n\n"
        )
        handle.write(
            format_matrix_table(
                summary,
                models=list(config.models),
                methods=list(config.methods),
                baseline_method=config.methods[0],
            )
        )
        handle.write("\n\n")
        for model in config.models:
            handle.write(f"## {model}\n\n")
            for method in config.methods:
                stats = summary[model][method]
                handle.write(
                    f"- {METHOD_ALIASES[method]}: wall={stats['median_wall_secs']:.2f}s, "
                    f"ESS_bulk/s={stats['ess_bulk_per_sec']:.0f}, "
                    f"ESS_bulk/LF={stats['ess_bulk_per_leapfrog']:.6f}, "
                    f"ESS_tail/LF={stats['ess_tail_per_leapfrog']:.6f}, "
                    f"LF/s={stats['leapfrogs_per_sec']:.0f}, "
                    f"sample_LF={stats['median_sampling_leapfrogs']}, "
                    f"warmup_LF={stats['median_warmup_leapfrogs']}, "
                    f"total_LF={stats['median_total_leapfrogs']}, "
                    f"max R-hat={stats['median_max_rhat']}, "
                    f"div={((stats['median_divergence_rate'] or 0.0) * 100):.1f}%, "
                    f"min E-BFMI={stats['median_min_ebfmi']}, "
                    f"discovery_gate_passed={stats['discovery_gate_passed']}\n"
                )
            for method in config.methods:
                if method == config.methods[0]:
                    continue
                ratio_key = f"{method}_over_{config.methods[0]}"
                ratios = summary[model][ratio_key]
                handle.write(
                    f"- Ratios ({METHOD_ALIASES[method]}/{METHOD_ALIASES[config.methods[0]]}): "
                    f"{_metric_suffix('ess_bulk_per_sec')}={_format_ratio(ratios['ess_bulk_per_sec'])}, "
                    f"{_metric_suffix('ess_tail_per_sec')}={_format_ratio(ratios['ess_tail_per_sec'])}, "
                    f"{_metric_suffix('ess_bulk_per_leapfrog')}={_format_ratio(ratios['ess_bulk_per_leapfrog'])}, "
                    f"{_metric_suffix('ess_tail_per_leapfrog')}={_format_ratio(ratios['ess_tail_per_leapfrog'])}, "
                    f"{_metric_suffix('leapfrogs_per_sec')}={_format_ratio(ratios['leapfrogs_per_sec'])}, "
                    f"{_metric_suffix('median_wall_secs')}={_format_ratio(ratios['median_wall_secs'])}\n"
                )
            handle.write("\n")
    return md_path


def build_parser(
    *,
    description: str,
    default_models: str,
    default_methods: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated sampling seeds",
    )
    parser.add_argument("--out-dir", default="results/sampler_matrix", help="Output directory")
    parser.add_argument("--models", default=default_models, help="Comma-separated model set")
    parser.add_argument("--methods", default=default_methods, help="Comma-separated sampler methods")
    parser.add_argument("--n-chains", type=int, default=DEFAULT_N_CHAINS, help="Number of chains")
    parser.add_argument("--n-warmup", type=int, default=DEFAULT_N_WARMUP, help="Warmup iterations")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Sampling iterations")
    parser.add_argument("--metric", default="diagonal", help="Metric type for compared methods")
    parser.add_argument("--glm-n", type=int, default=1000, help="Number of observations for GLM benchmarks")
    parser.add_argument("--glm-p", type=int, default=10, help="Number of predictors for GLM benchmarks")
    parser.add_argument("--target-accept", type=float, default=0.8, help="Target acceptance for all methods")
    parser.add_argument("--max-treedepth", type=int, default=10, help="Tree depth for NUTS/WALNUTS")
    parser.add_argument("--mams-max-leapfrog", type=int, default=1024, help="Maximum leapfrogs for MAMS")
    parser.add_argument(
        "--mams-diagonal-precond",
        dest="mams_diagonal_precond",
        action="store_true",
        help="Enable diagonal preconditioning for MAMS",
    )
    parser.add_argument(
        "--no-mams-diagonal-precond",
        dest="mams_diagonal_precond",
        action="store_false",
        help="Disable diagonal preconditioning for MAMS",
    )
    parser.set_defaults(mams_diagonal_precond=True)
    return parser


def config_from_args(
    args: argparse.Namespace,
    *,
    allowed_models: tuple[str, ...],
    allowed_methods: tuple[str, ...],
) -> SamplerMatrixConfig:
    if args.metric not in {"diagonal", "dense", "auto"}:
        raise SystemExit("metric must be one of: diagonal, dense, auto")
    if args.n_chains <= 0 or args.n_warmup < 0 or args.n_samples <= 0:
        raise SystemExit("Invalid run config: n_chains>0, n_warmup>=0, n_samples>0 required")
    methods = parse_methods(args.methods, allowed_methods=allowed_methods)
    models = parse_models(args.models, allowed_models=allowed_models)
    return SamplerMatrixConfig(
        methods=tuple(methods),
        models=tuple(models),
        seeds=tuple(parse_seeds(args.seeds)),
        out_dir=Path(args.out_dir),
        n_chains=args.n_chains,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        metric=args.metric,
        glm_n=args.glm_n,
        glm_p=args.glm_p,
        target_accept=args.target_accept,
        max_treedepth=args.max_treedepth,
        mams_max_leapfrog=args.mams_max_leapfrog,
        mams_diagonal_precond=bool(args.mams_diagonal_precond),
    )


def run_cli(
    *,
    description: str,
    output_stem: str,
    title: str,
    default_models: str,
    default_methods: str,
    allowed_models: tuple[str, ...],
    allowed_methods: tuple[str, ...],
) -> int:
    parser = build_parser(
        description=description,
        default_models=default_models,
        default_methods=default_methods,
    )
    args = parser.parse_args()
    config = config_from_args(
        args,
        allowed_models=allowed_models,
        allowed_methods=allowed_methods,
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    artifact, _ = run_sampler_matrix(config)
    json_path = config.out_dir / f"{output_stem}.json"
    json_path.write_text(json.dumps(artifact, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n[saved] {json_path}")
    md_path = write_matrix_markdown(
        artifact,
        config=config,
        output_stem=output_stem,
        title=title,
    )
    print(f"[saved] {md_path}")
    print(
        "\n"
        + format_matrix_table(
            artifact["summary"],
            models=list(config.models),
            methods=list(config.methods),
            baseline_method=config.methods[0],
        )
    )
    return 0
