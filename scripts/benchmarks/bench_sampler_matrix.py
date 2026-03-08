#!/usr/bin/env python3
"""Benchmark multiple NextStat samplers on a shared Bayesian model matrix."""

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

ALLOWED_METHODS = ("nuts", "walnuts", "mams")


def main() -> int:
    return run_cli(
        description="Benchmark a NextStat sampler method matrix on shared Bayesian targets",
        output_stem="bench_sampler_matrix",
        title="Sampler Matrix (NextStat)",
        default_models="std_normal_10d,eight_schools,glm_logistic,funnel_ncp_10d",
        default_methods="nuts,walnuts,mams",
        allowed_models=ALLOWED_MODELS,
        allowed_methods=ALLOWED_METHODS,
    )


if __name__ == "__main__":
    raise SystemExit(main())
