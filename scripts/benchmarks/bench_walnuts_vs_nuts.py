#!/usr/bin/env python3
"""Benchmark WALNUTS against NUTS on identical NextStat targets."""

from __future__ import annotations

from _bayesian_sampler_matrix import run_cli


ALLOWED_MODELS = (
    "std_normal_10d",
    "eight_schools",
    "glm_logistic",
    "funnel_ncp_10d",
    "glm_poisson",
    "glm_negbin",
    "funnel_10d",
)

ALLOWED_METHODS = ("nuts", "walnuts")


def main() -> int:
    return run_cli(
        description="Benchmark WALNUTS vs NUTS on identical NextStat models",
        output_stem="bench_walnuts_vs_nuts",
        title="WALNUTS vs NUTS (NextStat)",
        default_models="std_normal_10d,eight_schools,glm_logistic,funnel_ncp_10d,glm_negbin",
        default_methods="nuts,walnuts",
        allowed_models=ALLOWED_MODELS,
        allowed_methods=ALLOWED_METHODS,
    )


if __name__ == "__main__":
    raise SystemExit(main())
