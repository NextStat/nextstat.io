# Simplified Likelihood Exporter Public Validation Surface

**Date**: 2026-03-09  
**Status**: Stable evidence surface  
**Scope**: curated public reinterpretation-style `full -> derived -> reinterpret` exporter evidence

## Purpose

This document defines the stable evidence surface for the curated public
reinterpretation-style exporter matrix.

The case class tracked here is `public_reinterpretation_style`.

It exists so public-style exporter validation is no longer buried only inside
the larger Apex2 exporter artifact.

## What is stable here

The stable claim in this document is about the evidence surface, not the runtime
support class of every case inside that evidence.

Stable evidence surface means:

- a versioned machine-readable report exists:
  `nextstat_simplified_likelihood_export_public_validation_report_v0`
- the report is persisted under the committed `nextstat-bench` current snapshot
  path as `export_public_validation_report.json`
- the report is covered by schema validation, smoke tests, workflow upload, and
  the exporter surface gate

## What this does not claim

This surface does not widen the promoted stable runtime claim for
`nextstat simplify workspace`.

In particular:

- the public validation surface is a separate fidelity-first evidence lane
- its performance threshold is intentionally lower than the exporter stable-review
  control floor because it covers real-world reinterpretation-style cases
- a green public validation report does not by itself promote the wider exporter
  matrix or replace the synthetic control cases used for stable-review gating
- the promoted stable runtime boundary remains:
  `pyhf` source, single-POI, Gaussian-constrained
  `source_model_constraints`, reduced-coordinate derived semantics

## Current committed artifact

Current machine-readable report:

- `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_public_validation_report.json`

Supporting inputs:

- `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/apex2_simplified_likelihood_report.json`
- `docs/specs/apex2_simplified_likelihood_export_public_case_catalog_v0.example.json`

## Required gates

The public validation report is considered healthy only if:

- `public_case_count >= 9`
- `all_schema_valid = true`
- `all_fidelity_gates_pass = true`
- `all_performance_gates_pass = true`
- `max_abs_q_mu_diff <= 0.1`
- `max_upper_limit_ratio_deviation <= 0.05`
- `min_net_end_to_end_upper_limit_speedup >= 0.75x`
- `cases_outside_promoted_stable_runtime_boundary = 0`
- `observed_constraint_covariance_sources = ["source_model_constraints"]`

## Current March 9, 2026 state

The committed `nextstat-bench` report is green.

Current public validation summary:

- `public_case_count = 9`
- `public_case_names = ["atlas_public_dual_sr_dual_cr_gaussian_export_stable_example", "atlas_public_dual_sr_vr_dual_cr_gaussian_export_stable_example", "atlas_public_triple_sr_vr_dual_cr_gaussian_export_stable_example", "atlas_public_sr_cr_gaussian_export_stable_example", "cms_public_sr_cr_export_stable_example", "cms_public_sr_cr_asymmetric_gaussian_export_stable_example", "cms_public_dual_sr_cr_gaussian_export_stable_example", "cms_public_sr_dual_cr_gaussian_export_stable_example", "cms_public_sr_vr_dual_cr_gaussian_export_stable_example"]`
- `cases_outside_promoted_stable_runtime_boundary = 0`
- `observed_constraint_covariance_sources = ["source_model_constraints"]`
- `max_abs_q_mu_diff = 0.09618848026584459`
- `max_upper_limit_ratio_deviation = 0.011190668120821257`
- `min_net_end_to_end_upper_limit_speedup = 0.8777103422927768x`

## Bottom line

This is now a stable evidence surface for public exporter validation.

It is intentionally separate from runtime promotion:

- the evidence report is stable
- the promoted narrow exporter runtime boundary stays explicit
- the current public cases stay inside that promoted runtime boundary
- public validation remains a distinct stable-evidence lane rather than the
  synthetic promotion-control floor used for stable-review gating
