"""LLM Tool Definitions for Agentic Analysis.

Provides OpenAI-compatible function-calling schemas so that AI agents
(GPT-4o, Llama 4, Mistral-Next, Claude, local Ollama models) can
discover and invoke NextStat operations programmatically.

Usage with OpenAI::

    from nextstat.tools import get_toolkit, execute_tool
    import openai

    tools = get_toolkit()  # list of OpenAI function-calling dicts
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Fit this workspace and show me the POI"}],
        tools=tools,
    )

    # When the agent calls a tool:
    for call in response.choices[0].message.tool_calls:
        result = execute_tool(call.function.name, json.loads(call.function.arguments))
        # result is a JSON-serialisable dict

Usage with LangChain / LlamaIndex::

    from nextstat.tools import get_langchain_tools
    tools = get_langchain_tools()  # list of langchain BaseTool

Usage standalone (MCP server)::

    from nextstat.tools import get_mcp_tools
    tools = get_mcp_tools()  # list of MCP tool dicts
"""

from __future__ import annotations

import copy
import json
import math
import os
import urllib.error
import urllib.request
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_EXECUTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Optional execution controls. If deterministic=true (default), NextStat will attempt to "
        "enforce parity-friendly settings (threads=1, eval_mode='parity') where supported."
    ),
    "properties": {
        "deterministic": {
            "type": "boolean",
            "description": "If true, prefer deterministic parity behavior (default: true).",
            "default": True,
        },
        "threads": {
            "type": "integer",
            "description": (
                "Requested thread count. If omitted and deterministic=true, defaults to 1. "
                "If 0, use library default."
            ),
        },
        "eval_mode": {
            "type": "string",
            "description": "Evaluation mode. 'parity' favors numerical stability; 'fast' may use approximations.",
            "enum": ["parity", "fast"],
        },
    },
}

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "nextstat_fit",
            "description": (
                "Run Maximum Likelihood Estimation (MLE) on a HistFactory statistical model. "
                "Returns best-fit parameters, uncertainties, NLL at minimum, and convergence info. "
                "The workspace_json must be a pyhf-style or HS3-style JSON workspace string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace_json": {
                        "type": "string",
                        "description": "JSON string of the pyhf or HS3 workspace. Auto-detected.",
                    },
                    "execution": _EXECUTION_SCHEMA,
                },
                "required": ["workspace_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_hypotest",
            "description": (
                "Run an asymptotic CLs hypothesis test at a given signal strength mu (qtilde). "
                "Returns CLs, CLs+b, and CLb (pyhf-compatible semantics)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace_json": {
                        "type": "string",
                        "description": "JSON string of the pyhf or HS3 workspace.",
                    },
                    "mu": {
                        "type": "number",
                        "description": "Signal strength hypothesis to test (e.g. 1.0 for SM).",
                    },
                    "execution": _EXECUTION_SCHEMA,
                },
                "required": ["workspace_json", "mu"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_hypotest_toys",
            "description": (
                "Run a toy-based CLs hypothesis test at a given signal strength mu (qtilde). "
                "This is stochastic; specify seed for reproducibility."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace_json": {
                        "type": "string",
                        "description": "JSON string of the pyhf or HS3 workspace.",
                    },
                    "mu": {
                        "type": "number",
                        "description": "Signal strength hypothesis to test (e.g. 1.0 for SM).",
                    },
                    "n_toys": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Number of toy pseudo-experiments. Default 1000.",
                        "default": 1000,
                    },
                    "seed": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "RNG seed. Default 42.",
                        "default": 42,
                    },
                    "expected_set": {
                        "type": "boolean",
                        "description": "If true, return expected CLs bands (±1σ, ±2σ). Default false.",
                        "default": False,
                    },
                    "return_meta": {
                        "type": "boolean",
                        "description": "If true, include toy meta/statistics in the result. Default false.",
                        "default": False,
                    },
                    "execution": _EXECUTION_SCHEMA,
                },
                "required": ["workspace_json", "mu"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_upper_limit",
            "description": (
                "Compute the 95% CL upper limit on signal strength via CLs. "
                "Returns observed limit and optionally expected limits (±1σ, ±2σ)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace_json": {
                        "type": "string",
                        "description": "JSON string of the pyhf or HS3 workspace.",
                    },
                    "expected": {
                        "type": "boolean",
                        "description": "If true, return expected limits with ±1σ/±2σ bands.",
                        "default": False,
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Confidence level alpha (default 0.05 for 95% CL).",
                        "default": 0.05,
                    },
                    "lo": {
                        "type": "number",
                        "description": "Lower bracket for root finding (default 0.0).",
                        "default": 0.0,
                    },
                    "hi": {
                        "type": ["number", "null"],
                        "description": "Upper bracket for root finding (default: POI upper bound or 10.0).",
                        "default": None,
                    },
                    "rtol": {
                        "type": "number",
                        "description": "Relative tolerance for root finding (default 1e-4).",
                        "default": 1e-4,
                    },
                    "max_iter": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Max root-finding iterations (default 80).",
                        "default": 80,
                    },
                    "execution": _EXECUTION_SCHEMA,
                },
                "required": ["workspace_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_ranking",
            "description": (
                "Compute nuisance parameter ranking (systematic impact on signal strength). "
                "Returns a sorted list of systematics with their impact (delta_mu_up, delta_mu_down), "
                "pull values, and constraints. This is the physics equivalent of Feature Importance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace_json": {
                        "type": "string",
                        "description": "JSON string of the pyhf or HS3 workspace.",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Return only the top N most impactful systematics. Default: all.",
                    },
                    "execution": _EXECUTION_SCHEMA,
                },
                "required": ["workspace_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_discovery_asymptotic",
            "description": (
                "Compute an asymptotic discovery-style statistic at mu=0 from a profiled likelihood scan. "
                "Returns q0, z0, and p0 (one-sided). This is NOT CLs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace_json": {
                        "type": "string",
                        "description": "JSON string of the pyhf or HS3 workspace.",
                    },
                    "execution": _EXECUTION_SCHEMA,
                },
                "required": ["workspace_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_scan",
            "description": (
                "Run a profile likelihood scan over signal strength values. "
                "Returns (mu, q_mu, 2*delta_NLL) arrays for plotting the likelihood curve."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace_json": {
                        "type": "string",
                        "description": "JSON string of the pyhf or HS3 workspace.",
                    },
                    "start": {
                        "type": "number",
                        "description": "Start of mu scan range. Default 0.0.",
                        "default": 0.0,
                    },
                    "stop": {
                        "type": "number",
                        "description": "End of mu scan range. Default 5.0.",
                        "default": 5.0,
                    },
                    "points": {
                        "type": "integer",
                        "description": "Number of scan points. Default 21.",
                        "default": 21,
                    },
                    "execution": _EXECUTION_SCHEMA,
                },
                "required": ["workspace_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_workspace_audit",
            "description": (
                "Audit a pyhf workspace for compatibility. Reports channel count, sample count, "
                "modifier types, parameter count, and any unsupported features."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace_json": {
                        "type": "string",
                        "description": "JSON string of the pyhf workspace to audit.",
                    },
                    "execution": _EXECUTION_SCHEMA,
                },
                "required": ["workspace_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_read_root_histogram",
            "description": (
                "Read a TH1 histogram from a ROOT file, including sumw2 and under/overflow bins. "
                "Returns bin edges, bin content, and flow bins for downstream analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "root_path": {
                        "type": "string",
                        "description": "Path to a ROOT file on disk.",
                    },
                    "hist_path": {
                        "type": "string",
                        "description": "Histogram path/key inside the ROOT file (e.g. 'dir/hist').",
                    },
                    "execution": _EXECUTION_SCHEMA,
                },
                "required": ["root_path", "hist_path"],
            },
        },
    },
    # -----------------------------------------------------------------------
    # Cross-vertical tools (GLM, Bayesian, Survival, Econometrics, etc.)
    # -----------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "nextstat_glm_fit",
            "description": (
                "Fit a Generalized Linear Model (GLM). Supports linear, logistic, Poisson, and "
                "negative binomial regression. Returns coefficients, standard errors, and predictions. "
                "Input is tabular: X (2D array of features), y (1D array of outcomes)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Feature matrix (2D array, each inner array is one observation).",
                    },
                    "y": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Response variable (1D array).",
                    },
                    "family": {
                        "type": "string",
                        "enum": ["linear", "logistic", "poisson", "negbin"],
                        "description": "GLM family. Default: linear.",
                        "default": "linear",
                    },
                    "include_intercept": {
                        "type": "boolean",
                        "description": "Whether to include an intercept term. Default: true.",
                        "default": True,
                    },
                    "l2": {
                        "type": ["number", "null"],
                        "description": "L2 regularization strength (ridge). Default: none.",
                        "default": None,
                    },
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_bayesian_sample",
            "description": (
                "Run Bayesian NUTS sampling on any NextStat model. Returns posterior draws, "
                "diagnostics (ESS, R-hat, divergences, E-BFMI), and sample statistics. "
                "The model_spec describes which model to build (e.g. logistic regression, survival, etc.)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "enum": [
                            "linear_regression", "logistic_regression", "poisson_regression",
                            "negbin_regression", "cox_ph", "weibull_survival",
                            "lognormal_aft", "ordered_logit", "ordered_probit",
                            "histfactory",
                        ],
                        "description": "Type of model to sample from.",
                    },
                    "x": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Feature matrix (for regression/survival models).",
                    },
                    "y": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Response variable.",
                    },
                    "time": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Event/censoring times (survival models only).",
                    },
                    "event": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Event indicator: 1=event, 0=censored (survival models only).",
                    },
                    "n_levels": {
                        "type": "integer",
                        "description": "Number of ordinal levels (ordered logit/probit only).",
                    },
                    "workspace_json": {
                        "type": "string",
                        "description": "pyhf/HS3 workspace JSON (histfactory model only).",
                    },
                    "n_chains": {
                        "type": "integer",
                        "description": "Number of MCMC chains. Default: 4.",
                        "default": 4,
                    },
                    "n_warmup": {
                        "type": "integer",
                        "description": "Number of warmup iterations per chain. Default: 500.",
                        "default": 500,
                    },
                    "n_samples": {
                        "type": "integer",
                        "description": "Number of post-warmup samples per chain. Default: 1000.",
                        "default": 1000,
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed. Default: 42.",
                        "default": 42,
                    },
                    "target_accept": {
                        "type": "number",
                        "description": "Target acceptance rate for NUTS. Default: 0.8.",
                        "default": 0.8,
                    },
                },
                "required": ["model_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_survival_fit",
            "description": (
                "Fit a survival model via MLE. Supports Cox PH, Weibull, Log-Normal AFT, and "
                "Exponential models. Returns parameter estimates (log-hazard ratios or AFT coefficients), "
                "standard errors, and NLL. Input: X (covariates), time (event/censoring times), "
                "event (1=event, 0=censored)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Covariate matrix.",
                    },
                    "time": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Event or censoring times.",
                    },
                    "event": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Event indicator (1=event, 0=censored).",
                    },
                    "model": {
                        "type": "string",
                        "enum": ["cox_ph", "weibull", "lognormal_aft", "exponential"],
                        "description": "Survival model type. Default: cox_ph.",
                        "default": "cox_ph",
                    },
                },
                "required": ["x", "time", "event"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_kaplan_meier",
            "description": (
                "Compute the Kaplan-Meier survival curve and optionally a log-rank test. "
                "Returns time points, survival probabilities, confidence intervals, and "
                "number at risk. If group labels are provided, also returns a log-rank test."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Event or censoring times.",
                    },
                    "event": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Event indicator (1=event, 0=censored).",
                    },
                    "group": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional group labels for log-rank test (e.g. 0/1 for two arms).",
                    },
                },
                "required": ["time", "event"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_panel_fe",
            "description": (
                "Fit a panel data model with entity fixed effects (within estimator). "
                "Returns coefficients with cluster-robust standard errors. "
                "Equivalent to Stata's xtreg, fe."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Feature matrix (no intercept column needed).",
                    },
                    "y": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Response variable.",
                    },
                    "entity": {
                        "type": "array",
                        "items": {},
                        "description": "Entity/group identifiers (e.g. firm IDs).",
                    },
                    "time": {
                        "type": "array",
                        "items": {},
                        "description": "Optional time identifiers (required for cluster='time' or 'two_way').",
                    },
                    "cluster": {
                        "type": "string",
                        "enum": ["entity", "time", "two_way", "none"],
                        "description": "Cluster-robust SE type. Default: entity.",
                        "default": "entity",
                    },
                },
                "required": ["x", "y", "entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_did",
            "description": (
                "Estimate a Difference-in-Differences (DiD) model via two-way fixed effects (TWFE). "
                "Returns the ATT (average treatment effect on the treated) with cluster-robust SE."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Outcome variable.",
                    },
                    "treat": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Treatment indicator (0/1).",
                    },
                    "post": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Post-treatment indicator (0/1).",
                    },
                    "entity": {
                        "type": "array",
                        "items": {},
                        "description": "Entity identifiers.",
                    },
                    "time": {
                        "type": "array",
                        "items": {},
                        "description": "Time period identifiers.",
                    },
                    "x": {
                        "type": ["array", "null"],
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Optional control covariates.",
                        "default": None,
                    },
                    "cluster": {
                        "type": "string",
                        "enum": ["entity", "time", "two_way", "none"],
                        "default": "entity",
                    },
                },
                "required": ["y", "treat", "post", "entity", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_iv_2sls",
            "description": (
                "Estimate an Instrumental Variables (IV) model via Two-Stage Least Squares (2SLS). "
                "Returns structural coefficients, standard errors (HC1 or cluster-robust), and "
                "first-stage F-statistics for weak instrument diagnostics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Outcome variable.",
                    },
                    "endog": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Endogenous regressors.",
                    },
                    "instruments": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Excluded instruments.",
                    },
                    "exog": {
                        "type": ["array", "null"],
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Optional exogenous controls.",
                        "default": None,
                    },
                    "cov": {
                        "type": "string",
                        "enum": ["homoskedastic", "hc1", "cluster", "hac"],
                        "description": "Covariance estimator. Default: hc1.",
                        "default": "hc1",
                    },
                    "cluster": {
                        "type": ["array", "null"],
                        "items": {},
                        "description": "Cluster labels when cov='cluster'.",
                        "default": None,
                    },
                    "time_index": {
                        "type": ["array", "null"],
                        "items": {},
                        "description": "Ordered time index when cov='hac'.",
                        "default": None,
                    },
                    "max_lag": {
                        "type": ["integer", "null"],
                        "description": "Maximum Newey-West lag when cov='hac'. Default: automatic.",
                        "default": None,
                    },
                },
                "required": ["y", "endog", "instruments"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_aipw",
            "description": (
                "Estimate a doubly-robust Average Treatment Effect (ATE or ATT) via "
                "Augmented Inverse Probability Weighting (AIPW). Returns the treatment effect "
                "estimate, standard error, and propensity diagnostics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Covariate matrix.",
                    },
                    "y": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Outcome variable.",
                    },
                    "treatment": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Treatment indicator (0/1).",
                    },
                    "estimand": {
                        "type": "string",
                        "enum": ["ate", "att"],
                        "description": "Estimand: ATE or ATT. Default: ate.",
                        "default": "ate",
                    },
                },
                "required": ["x", "y", "treatment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_meta_analysis",
            "description": (
                "Run a meta-analysis (fixed or random effects). Input: effect sizes and their "
                "standard errors from multiple studies. Returns pooled estimate, confidence interval, "
                "heterogeneity statistics (I², Q, tau²)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "effects": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Effect sizes from each study.",
                    },
                    "standard_errors": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Standard errors of each study's effect.",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["fixed", "random"],
                        "description": "Meta-analysis method. Default: random.",
                        "default": "random",
                    },
                },
                "required": ["effects", "standard_errors"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_kalman",
            "description": (
                "Run Kalman filtering, smoothing, or forecasting on a linear state-space model. "
                "Returns filtered/smoothed state estimates, log-likelihood, and optionally forecasts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "F": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "State transition matrix.",
                    },
                    "H": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Observation matrix.",
                    },
                    "Q": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Process noise covariance.",
                    },
                    "R": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Observation noise covariance.",
                    },
                    "x0": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Initial state mean.",
                    },
                    "P0": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Initial state covariance.",
                    },
                    "y": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Observation sequence (each element is an observation vector).",
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["filter", "smooth", "forecast"],
                        "description": "Operation to perform. Default: filter.",
                        "default": "filter",
                    },
                    "n_ahead": {
                        "type": "integer",
                        "description": "Number of steps to forecast ahead (forecast mode only). Default: 10.",
                        "default": 10,
                    },
                },
                "required": ["F", "H", "Q", "R", "x0", "P0", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_churn_retention",
            "description": (
                "Compute a churn retention curve from tenure and event data. "
                "Returns survival probabilities at specified time points, "
                "plus optional cohort matrix and diagnostics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tenure": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Subscription tenure (time from signup to event/censoring).",
                    },
                    "event": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Churn indicator (1=churned, 0=still active).",
                    },
                    "time_grid": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Time points at which to evaluate retention. Default: [30, 60, 90, 180, 365].",
                    },
                },
                "required": ["tenure", "event"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_chain_ladder",
            "description": (
                "Run chain ladder or Mack chain ladder reserving on an insurance loss triangle. "
                "Returns ultimate losses, reserves, development factors, and (for Mack) prediction errors."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "triangle": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": ["number", "null"]}},
                        "description": "Loss triangle as 2D array (rows=origin periods, cols=development periods). Use null for missing entries.",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["basic", "mack"],
                        "description": "basic = deterministic chain ladder; mack = with prediction errors. Default: mack.",
                        "default": "mack",
                    },
                },
                "required": ["triangle"],
            },
        },
    },
]


def _http_json_get(url: str, *, timeout_s: float) -> Any:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data)
    except urllib.error.HTTPError as e:
        body = None
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = None
        raise RuntimeError(f"HTTP {e.code} from {url}: {body or e.reason}")
    except Exception as e:
        raise RuntimeError(f"Failed GET {url}: {e}")


def _http_json_post(url: str, payload: dict[str, Any], *, timeout_s: float) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            out = resp.read().decode("utf-8")
            return json.loads(out)
    except urllib.error.HTTPError as e:
        body = None
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = None
        raise RuntimeError(f"HTTP {e.code} from {url}: {body or e.reason}")
    except Exception as e:
        raise RuntimeError(f"Failed POST {url}: {e}")

def _resolve_server_url(server_url: Optional[str]) -> Optional[str]:
    if server_url:
        return server_url.strip()
    for k in ("NEXTSTAT_TOOLS_SERVER_URL", "NEXTSTAT_SERVER_URL"):
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()
    return None


def get_toolkit(
    *,
    transport: str = "local",
    server_url: Optional[str] = None,
    timeout_s: float = 10.0,
) -> list[dict[str, Any]]:
    """Return OpenAI-compatible function-calling tool definitions.

    These can be passed directly to ``openai.chat.completions.create(tools=...)``,
    or adapted for any agent framework that uses the OpenAI tool schema.

    Args:
        transport:
            - ``"local"`` (default): return the in-process Python tool registry.
            - ``"server"``: fetch the registry from ``nextstat-server`` at ``GET /v1/tools/schema``.
        server_url: Base URL for server mode, e.g. ``"http://127.0.0.1:3742"``.
        timeout_s: HTTP timeout (server mode only).

    Returns:
        ``list[dict]`` — each dict has ``type: "function"`` and a ``function``
        key with ``name``, ``description``, and ``parameters`` (JSON Schema).

    Example::

        import openai
        from nextstat.tools import get_toolkit

        tools = get_toolkit()
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Fit this workspace"}],
            tools=tools,
        )
    """
    if transport == "local":
        return copy.deepcopy(_TOOLS)
    if transport != "server":
        raise ValueError(f"Unknown transport: {transport!r}. Use 'local' or 'server'.")
    server_url = _resolve_server_url(server_url)
    if not server_url:
        raise ValueError("server_url is required for transport='server'")
    schema = _http_json_get(f"{server_url.rstrip('/')}/v1/tools/schema", timeout_s=timeout_s)
    tools = schema.get("tools")
    if not isinstance(tools, list):
        raise RuntimeError("Invalid server schema response: missing 'tools' list")
    return tools


def get_tool_names() -> list[str]:
    """Return the list of available tool names."""
    return [t["function"]["name"] for t in _TOOLS]


def get_tool_schema(name: str) -> Optional[dict[str, Any]]:
    """Return the JSON Schema for a specific tool by name.

    Returns ``None`` if the tool is not found.
    """
    for t in _TOOLS:
        if t["function"]["name"] == name:
            return copy.deepcopy(t)
    return None


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _load_model(workspace_json: str):
    """Load a HistFactoryModel from a JSON string (auto-detects pyhf vs HS3)."""
    import nextstat

    model = nextstat.HistFactoryModel.from_workspace(workspace_json)
    return model


def _normal_sf(z: float) -> float:
    """Standard normal survival function (1 - CDF) without SciPy."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _apply_execution(nextstat, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Apply best-effort execution controls. Returns meta additions."""
    exec_cfg = arguments.get("execution") or {}
    deterministic = bool(exec_cfg.get("deterministic", True))

    requested_eval_mode = exec_cfg.get("eval_mode")
    if deterministic:
        eval_mode = "parity"
    else:
        eval_mode = requested_eval_mode

    prev_eval_mode = None
    if eval_mode in ("fast", "parity"):
        try:
            prev_eval_mode = nextstat.get_eval_mode()
            if prev_eval_mode != eval_mode:
                nextstat.set_eval_mode(eval_mode)
        except Exception:
            prev_eval_mode = None

    requested_threads = exec_cfg.get("threads")
    if requested_threads is None and deterministic:
        requested_threads = 1

    threads_applied = None
    if isinstance(requested_threads, int):
        try:
            set_threads = getattr(nextstat, "set_threads", None)
            if callable(set_threads):
                threads_applied = bool(set_threads(int(requested_threads)))
        except Exception:
            threads_applied = False

    return {
        "tool_name": tool_name,
        "deterministic": deterministic,
        "eval_mode_effective": eval_mode if eval_mode in ("fast", "parity") else None,
        "eval_mode_prev": prev_eval_mode,
        "threads_requested": requested_threads,
        "threads_applied": threads_applied,
    }


def _restore_execution(nextstat, meta: dict[str, Any]) -> None:
    prev = meta.get("eval_mode_prev")
    if prev in ("fast", "parity"):
        try:
            if nextstat.get_eval_mode() != prev:
                nextstat.set_eval_mode(prev)
        except Exception:
            pass


def _execute_tool_impl(nextstat, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Implementation body for tools, assuming execution controls are already applied."""
    if name == "nextstat_fit":
        model = _load_model(arguments["workspace_json"])
        result = nextstat.fit(model)
        params = result.parameters
        names = model.parameter_names()
        poi_idx = model.poi_index()
        return {
            "nll": result.nll,
            "converged": result.converged,
            "n_iter": result.n_iter,
            "poi_index": poi_idx,
            "poi_value": params[poi_idx] if poi_idx is not None else None,
            "poi_error": result.uncertainties[poi_idx] if poi_idx is not None else None,
            "parameters": {
                n: {"value": v, "error": e}
                for n, v, e in zip(names, params, result.uncertainties)
            },
        }

    if name == "nextstat_hypotest":
        model = _load_model(arguments["workspace_json"])
        mu = float(arguments["mu"])
        cls_val, tails = nextstat.hypotest(mu, model, return_tail_probs=True)
        clsb, clb = tails
        return {"mu": mu, "cls": float(cls_val), "clsb": float(clsb), "clb": float(clb)}

    if name == "nextstat_hypotest_toys":
        model = _load_model(arguments["workspace_json"])
        mu = float(arguments["mu"])
        n_toys = int(arguments.get("n_toys", 1000))
        seed = int(arguments.get("seed", 42))
        expected_set = bool(arguments.get("expected_set", False))
        return_meta = bool(arguments.get("return_meta", False))
        r = nextstat.hypotest_toys(
            mu,
            model,
            n_toys=n_toys,
            seed=seed,
            expected_set=expected_set,
            return_tail_probs=True,
            return_meta=return_meta,
        )
        # Return shape depends on expected_set/return_meta; keep it explicit and lossless.
        return {
            "mu": mu,
            "n_toys": n_toys,
            "seed": seed,
            "expected_set": expected_set,
            "raw": r,
        }

    if name == "nextstat_upper_limit":
        model = _load_model(arguments["workspace_json"])
        expected = bool(arguments.get("expected", False))
        alpha = float(arguments.get("alpha", 0.05))
        lo = float(arguments.get("lo", 0.0))
        hi = arguments.get("hi", None)
        hi_val = None if hi is None else float(hi)
        rtol = float(arguments.get("rtol", 1e-4))
        max_iter = int(arguments.get("max_iter", 80))
        if expected:
            obs, exp = nextstat.upper_limits_root(
                model, alpha=alpha, lo=lo, hi=hi_val, rtol=rtol, max_iter=max_iter
            )
            return {
                "alpha": alpha,
                "obs_limit": float(obs),
                "exp_limits": [float(x) for x in exp],
            }
        obs = nextstat.upper_limit(model, alpha=alpha, lo=lo, hi=hi_val, rtol=rtol, max_iter=max_iter)
        return {"alpha": alpha, "obs_limit": float(obs)}

    if name == "nextstat_ranking":
        model = _load_model(arguments["workspace_json"])
        from nextstat.interpret import rank_impact

        top_n = arguments.get("top_n")
        table = rank_impact(model, top_n=top_n)
        return {"ranking": table}

    if name == "nextstat_discovery_asymptotic":
        model = _load_model(arguments["workspace_json"])
        fit_res = nextstat.fit(model)
        poi_idx = model.poi_index()
        mu_hat = None
        if poi_idx is not None:
            try:
                mu_hat = float(fit_res.parameters[poi_idx])
            except Exception:
                mu_hat = None
        scan = nextstat.profile_scan(model, [0.0])
        pts = scan.get("points") or []
        if not pts:
            raise RuntimeError("profile_scan returned no points for mu=0")
        nll0 = float(pts[0].get("nll_mu"))
        nll_hat = float(fit_res.nll)
        q0_raw = 2.0 * (nll0 - nll_hat)
        q0 = max(0.0, q0_raw)
        if mu_hat is not None and mu_hat <= 0.0:
            q0 = 0.0
        z0 = math.sqrt(q0)
        p0 = _normal_sf(z0)
        return {
            "mu_hat": mu_hat,
            "nll_hat": nll_hat,
            "nll_mu0": nll0,
            "q0": q0,
            "z0": z0,
            "p0": p0,
        }

    if name == "nextstat_scan":
        model = _load_model(arguments["workspace_json"])
        start = float(arguments.get("start", 0.0))
        stop = float(arguments.get("stop", 5.0))
        points = int(arguments.get("points", 21))
        step = (stop - start) / max(points - 1, 1)
        mu_values = [start + i * step for i in range(points)]
        artifact = nextstat.profile_scan(model, mu_values)
        return dict(artifact)

    if name == "nextstat_workspace_audit":
        ws_json = arguments["workspace_json"]
        result = nextstat.workspace_audit(ws_json)
        return dict(result)

    if name == "nextstat_read_root_histogram":
        root_path = arguments["root_path"]
        hist_path = arguments["hist_path"]
        result = nextstat.read_root_histogram(root_path, hist_path)
        return dict(result)

    # -------------------------------------------------------------------
    # Cross-vertical tools
    # -------------------------------------------------------------------

    if name == "nextstat_glm_fit":
        import nextstat.glm as glm

        x = arguments["x"]
        y = arguments["y"]
        family = arguments.get("family", "linear")
        intercept = arguments.get("include_intercept", True)
        l2 = arguments.get("l2")

        if family == "linear":
            fit = glm.linear.fit(x, y, include_intercept=intercept, l2=l2)
            return {
                "family": "linear",
                "coef": list(fit.coef),
                "standard_errors": list(fit.standard_errors),
                "sigma2_hat": fit.sigma2_hat,
            }
        elif family == "logistic":
            fit = glm.logistic.fit(x, y, include_intercept=intercept, l2=l2)
            return {
                "family": "logistic",
                "coef": list(fit.coef),
                "standard_errors": list(fit.standard_errors),
            }
        elif family == "poisson":
            fit = glm.poisson.fit(x, y, include_intercept=intercept, l2=l2)
            return {
                "family": "poisson",
                "coef": list(fit.coef),
                "standard_errors": list(fit.standard_errors),
            }
        elif family == "negbin":
            fit = glm.negbin.fit(x, y, include_intercept=intercept, l2=l2)
            return {
                "family": "negbin",
                "coef": list(fit.coef),
                "standard_errors": list(fit.standard_errors),
                "alpha": fit.alpha,
            }
        else:
            raise ValueError(f"Unknown GLM family: {family!r}")

    if name == "nextstat_bayesian_sample":
        import nextstat.bayes

        mt = arguments["model_type"]
        x = arguments.get("x")
        y = arguments.get("y")
        time_arr = arguments.get("time")
        event_arr = arguments.get("event")
        n_levels = arguments.get("n_levels")
        ws_json = arguments.get("workspace_json")
        n_chains = int(arguments.get("n_chains", 4))
        n_warmup = int(arguments.get("n_warmup", 500))
        n_samples = int(arguments.get("n_samples", 1000))
        seed = int(arguments.get("seed", 42))
        target_accept = float(arguments.get("target_accept", 0.8))

        model_map = {
            "linear_regression": lambda: nextstat.LinearRegressionModel(x, y),
            "logistic_regression": lambda: nextstat.LogisticRegressionModel(x, y),
            "poisson_regression": lambda: nextstat.PoissonRegressionModel(x, y),
            "negbin_regression": lambda: nextstat.NegativeBinomialRegressionModel(x, y),
            "cox_ph": lambda: nextstat.CoxPhModel(x, time_arr, event_arr),
            "weibull_survival": lambda: nextstat.WeibullSurvivalModel(x, time_arr, event_arr),
            "lognormal_aft": lambda: nextstat.LogNormalAftModel(x, time_arr, event_arr),
            "ordered_logit": lambda: nextstat.OrderedLogitModel(x, y, n_levels),
            "ordered_probit": lambda: nextstat.OrderedProbitModel(x, y, n_levels),
            "histfactory": lambda: nextstat.HistFactoryModel.from_workspace(ws_json),
        }
        if mt not in model_map:
            raise ValueError(f"Unknown model_type: {mt!r}")
        model = model_map[mt]()

        raw = nextstat.bayes.sample(
            model,
            n_chains=n_chains,
            n_warmup=n_warmup,
            n_samples=n_samples,
            seed=seed,
            target_accept=target_accept,
            return_idata=False,
        )
        diag = raw.get("diagnostics", {})
        return {
            "model_type": mt,
            "n_chains": n_chains,
            "n_samples": n_samples,
            "param_names": raw.get("param_names", []),
            "diagnostics": diag,
            "posterior_summary": {
                k: {
                    "mean": sum(sum(ch) for ch in chains) / max(1, sum(len(ch) for ch in chains)),
                }
                for k, chains in (raw.get("posterior") or {}).items()
            },
        }

    if name == "nextstat_survival_fit":
        x = arguments["x"]
        time_arr = arguments["time"]
        event_arr = arguments["event"]
        model_name = arguments.get("model", "cox_ph")

        model_map = {
            "cox_ph": lambda: nextstat.CoxPhModel(x, time_arr, event_arr),
            "weibull": lambda: nextstat.WeibullSurvivalModel(x, time_arr, event_arr),
            "lognormal_aft": lambda: nextstat.LogNormalAftModel(x, time_arr, event_arr),
            "exponential": lambda: nextstat.ExponentialSurvivalModel(x, time_arr, event_arr),
        }
        if model_name not in model_map:
            raise ValueError(f"Unknown survival model: {model_name!r}")
        model = model_map[model_name]()
        result = nextstat.fit(model)
        return {
            "model": model_name,
            "parameters": list(result.parameters),
            "uncertainties": list(result.uncertainties),
            "nll": result.nll,
            "converged": result.converged,
        }

    if name == "nextstat_kaplan_meier":
        time_arr = arguments["time"]
        event_arr = arguments["event"]
        group_arr = arguments.get("group")

        km = nextstat.kaplan_meier(time_arr, event_arr)
        out = dict(km)
        if group_arr is not None:
            lr = nextstat.log_rank_test(time_arr, event_arr, group_arr)
            out["log_rank"] = dict(lr)
        return out

    if name == "nextstat_panel_fe":
        import nextstat.econometrics as econ

        x = arguments["x"]
        y = arguments["y"]
        entity = arguments["entity"]
        time_arr = arguments.get("time")
        cluster = arguments.get("cluster", "entity")

        fit = econ.panel_fe_fit(x, y, entity=entity, time=time_arr, cluster=cluster)
        if fit.cluster == "entity":
            n_clusters = len(set(entity))
        elif fit.cluster == "time":
            n_clusters = 0 if time_arr is None else len(set(time_arr))
        elif fit.cluster == "two_way":
            n_clusters = 0 if time_arr is None else len(set(zip(entity, time_arr)))
        else:
            n_clusters = 0
        return {
            "coef": list(fit.coef),
            "standard_errors": list(fit.standard_errors),
            "n_obs": fit.n_obs,
            "n_entities": fit.n_entities,
            "cluster_kind": fit.cluster,
            "n_clusters": int(n_clusters),
            "cluster": fit.cluster,
        }

    if name == "nextstat_did":
        import nextstat.econometrics as econ

        y = arguments["y"]
        treat = arguments["treat"]
        post = arguments["post"]
        entity = arguments["entity"]
        time_arr = arguments["time"]
        x = arguments.get("x")
        cluster = arguments.get("cluster", "entity")

        did = econ.did_twfe_fit(x, y, treat=treat, post=post, entity=entity, time=time_arr, cluster=cluster)
        return {
            "att": did.att,
            "att_se": did.att_se,
            "coef": list(did.twfe.coef),
            "standard_errors": list(did.twfe.standard_errors),
            "n_obs": did.twfe.n_obs,
            "cluster": did.twfe.cluster,
        }

    if name == "nextstat_iv_2sls":
        import nextstat.econometrics as econ

        y = arguments["y"]
        endog = arguments["endog"]
        instruments = arguments["instruments"]
        exog = arguments.get("exog")
        cov = arguments.get("cov", "hc1")
        cluster = arguments.get("cluster")
        time_index = arguments.get("time_index")
        max_lag = arguments.get("max_lag")

        iv = econ.iv_2sls_fit(
            y,
            endog=endog,
            instruments=instruments,
            exog=exog,
            cov=cov,
            cluster=cluster,
            time_index=time_index,
            max_lag=max_lag,
        )
        out = {
            "coef": list(iv.coef),
            "standard_errors": list(iv.standard_errors),
            "n_obs": iv.n_obs,
        }
        if hasattr(iv, "diagnostics") and iv.diagnostics is not None:
            out["diagnostics"] = {
                "first_stage_f": iv.diagnostics.first_stage_f,
            }
        return out

    if name == "nextstat_aipw":
        from nextstat.causal.aipw import aipw_fit

        x = arguments["x"]
        y = arguments["y"]
        treatment = arguments["treatment"]
        estimand = arguments.get("estimand", "ate")

        result = aipw_fit(x, y, treatment, estimand=estimand)
        return {
            "estimand": result.estimand,
            "estimate": result.estimate,
            "standard_error": result.standard_error,
            "n_obs": result.n_obs,
        }

    if name == "nextstat_meta_analysis":
        effects = arguments["effects"]
        ses = arguments["standard_errors"]
        method = arguments.get("method", "random")

        if method == "fixed":
            result = nextstat.meta_fixed(effects, ses)
        else:
            result = nextstat.meta_random(effects, ses)
        return dict(result)

    if name == "nextstat_kalman":
        F = arguments["F"]
        H = arguments["H"]
        Q = arguments["Q"]
        R = arguments["R"]
        x0 = arguments["x0"]
        P0 = arguments["P0"]
        y = arguments["y"]
        operation = arguments.get("operation", "filter")
        n_ahead = int(arguments.get("n_ahead", 10))

        model = nextstat.KalmanModel(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0, y=y)

        if operation == "filter":
            result = nextstat.kalman_filter(model)
        elif operation == "smooth":
            result = nextstat.kalman_smooth(model)
        elif operation == "forecast":
            result = nextstat.kalman_forecast(model, n_ahead=n_ahead)
        else:
            raise ValueError(f"Unknown kalman operation: {operation!r}")
        return dict(result)

    if name == "nextstat_churn_retention":
        tenure = arguments["tenure"]
        event = arguments["event"]
        time_grid = arguments.get("time_grid", [30, 60, 90, 180, 365])

        result = nextstat.churn_retention(tenure, event, time_grid=time_grid)
        return dict(result)

    if name == "nextstat_chain_ladder":
        triangle = arguments["triangle"]
        method = arguments.get("method", "mack")

        if method == "mack":
            result = nextstat.mack_chain_ladder(triangle)
        else:
            result = nextstat.chain_ladder(triangle)
        return dict(result)

    raise ValueError(f"Unknown tool: {name!r}. Available: {get_tool_names()}")


def execute_tool_raw(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a NextStat tool by name, returning the raw tool result (no envelope)."""
    import nextstat

    meta = _apply_execution(nextstat, name, arguments)
    try:
        return _execute_tool_impl(nextstat, name, arguments)
    finally:
        _restore_execution(nextstat, meta)


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    transport: str = "local",
    server_url: Optional[str] = None,
    timeout_s: float = 30.0,
    fallback_to_local: bool = True,
) -> dict[str, Any]:
    """Execute a NextStat tool call and return a stable response envelope.

    Args:
        transport:
            - ``"local"`` (default): execute in-process via Python bindings.
            - ``"server"``: execute over HTTP via ``nextstat-server`` at ``POST /v1/tools/execute``.
        server_url: Base URL for server mode, e.g. ``"http://127.0.0.1:3742"``.
        timeout_s: HTTP timeout (server mode only).
        fallback_to_local: If true (default), failed server calls fall back to local execution (if available).
    """
    if transport == "server":
        server_url = _resolve_server_url(server_url)
        if not server_url:
            raise ValueError(
                "server_url is required for transport='server' "
                "(or set NEXTSTAT_SERVER_URL / NEXTSTAT_TOOLS_SERVER_URL)"
            )
        try:
            out = _http_json_post(
                f"{server_url.rstrip('/')}/v1/tools/execute",
                {"name": name, "arguments": arguments},
                timeout_s=timeout_s,
            )
            if not isinstance(out, dict) or out.get("schema_version") != "nextstat.tool_result.v1":
                raise RuntimeError("Invalid server tool response (missing tool_result.v1 envelope)")
            return out
        except Exception as e:
            if fallback_to_local:
                try:
                    local = execute_tool(name, arguments, transport="local")
                    meta = local.get("meta")
                    if isinstance(meta, dict):
                        warnings = meta.get("warnings")
                        if not isinstance(warnings, list):
                            warnings = []
                        warnings.append(
                            f"server transport failed ({server_url}): {e.__class__.__name__}: {e}; fell back to local"
                        )
                        meta["warnings"] = warnings
                    return local
                except Exception:
                    # Fall through to a transport error envelope if local execution is unavailable.
                    pass
            return {
                "schema_version": "nextstat.tool_result.v1",
                "ok": False,
                "result": None,
                "error": {"type": e.__class__.__name__, "message": str(e)},
                "meta": {"tool_name": name, "nextstat_version": None, "warnings": [f"server_url={server_url}"]},
            }

    if transport != "local":
        raise ValueError(f"Unknown transport: {transport!r}. Use 'local' or 'server'.")

    import nextstat

    envelope: dict[str, Any] = {
        "schema_version": "nextstat.tool_result.v1",
        "ok": False,
        "result": None,
        "error": None,
        "meta": {
            "tool_name": name,
            "nextstat_version": getattr(nextstat, "__version__", None),
        },
    }

    exec_meta: dict[str, Any] | None = None
    try:
        exec_meta = _apply_execution(nextstat, name, arguments)
        envelope["result"] = _execute_tool_impl(nextstat, name, arguments)
        envelope["ok"] = True
    except Exception as e:
        envelope["error"] = {"type": e.__class__.__name__, "message": str(e)}
    finally:
        if exec_meta is not None:
            _restore_execution(nextstat, exec_meta)

    if exec_meta is not None:
        envelope["meta"]["deterministic"] = exec_meta.get("deterministic")
        ev = exec_meta.get("eval_mode_effective")
        if ev is not None:
            envelope["meta"]["eval_mode"] = ev
        tr = exec_meta.get("threads_requested")
        if tr is not None:
            envelope["meta"]["threads_requested"] = tr
        ta = exec_meta.get("threads_applied")
        if ta is not None:
            envelope["meta"]["threads_applied"] = ta

    return envelope


# ---------------------------------------------------------------------------
# LangChain integration (optional)
# ---------------------------------------------------------------------------


def get_langchain_tools():
    """Return NextStat tools as LangChain ``StructuredTool`` instances.

    Requires ``langchain-core`` to be installed.

    Example::

        from nextstat.tools import get_langchain_tools
        tools = get_langchain_tools()
        agent = create_tool_calling_agent(llm, tools, prompt)
    """
    try:
        from langchain_core.tools import StructuredTool  # type: ignore
    except ImportError:
        raise ImportError(
            "langchain-core is required for get_langchain_tools(). "
            "Install: pip install langchain-core"
        )

    lc_tools = []
    for tool_def in _TOOLS:
        fn_def = tool_def["function"]
        name = fn_def["name"]

        def _make_fn(tool_name):
            def fn(**kwargs):
                return execute_tool(tool_name, kwargs)
            fn.__name__ = tool_name
            fn.__doc__ = fn_def["description"]
            return fn

        lc_tools.append(
            StructuredTool.from_function(
                func=_make_fn(name),
                name=name,
                description=fn_def["description"],
            )
        )

    return lc_tools


# ---------------------------------------------------------------------------
# MCP (Model Context Protocol) integration
# ---------------------------------------------------------------------------


def get_mcp_tools() -> list[dict[str, Any]]:
    """Return NextStat tools as MCP tool definitions.

    Compatible with the Model Context Protocol standard for AI tool servers.

    Returns:
        ``list[dict]`` — each dict has ``name``, ``description``, ``inputSchema``.
    """
    mcp_tools = []
    for tool_def in _TOOLS:
        fn_def = tool_def["function"]
        mcp_tools.append({
            "name": fn_def["name"],
            "description": fn_def["description"],
            "inputSchema": fn_def["parameters"],
        })
    return mcp_tools


def handle_mcp_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle an MCP tool call. Alias for :func:`execute_tool`."""
    return execute_tool(name, arguments)
