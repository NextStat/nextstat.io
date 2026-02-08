//! Server-side Tool API (agent/runtime surface).
//!
//! This is intended to mirror the `nextstat.tools` Python surface:
//! - versioned tool result envelope (`nextstat.tool_result.v1`)
//! - deterministic execution controls (best-effort)
//! - correct semantics for CLs vs discovery p-values

use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use ns_inference::mle::MaximumLikelihoodEstimator;
use ns_translate::pyhf::audit as pyhf_audit;
use ns_translate::pyhf::HistFactoryModel;

use crate::pool::ModelPool;
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Tool envelope (matches docs/schemas/tools/nextstat_tool_result_v1.schema.json)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ToolError {
    #[serde(rename = "type")]
    pub type_name: String,
    pub message: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct ToolMeta {
    pub tool_name: String,
    pub nextstat_version: Option<String>,
    pub deterministic: bool,
    pub eval_mode: String,
    pub threads_requested: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threads_applied: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ToolResultEnvelope {
    pub schema_version: &'static str,
    pub ok: bool,
    pub result: serde_json::Value,
    pub error: Option<ToolError>,
    pub meta: ToolMeta,
}

impl ToolResultEnvelope {
    pub fn ok(tool_name: &str, meta: ToolMeta, result: serde_json::Value) -> Self {
        Self {
            schema_version: "nextstat.tool_result.v1",
            ok: true,
            result,
            error: None,
            meta: ToolMeta { tool_name: tool_name.to_string(), ..meta },
        }
    }

    pub fn err(tool_name: &str, mut meta: ToolMeta, type_name: &str, message: String) -> Self {
        meta.tool_name = tool_name.to_string();
        Self {
            schema_version: "nextstat.tool_result.v1",
            ok: false,
            result: serde_json::Value::Null,
            error: Some(ToolError {
                type_name: type_name.to_string(),
                message,
                extra: serde_json::Map::new(),
            }),
            meta,
        }
    }
}

// ---------------------------------------------------------------------------
// Execution controls
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ExecutionControls {
    #[serde(default = "default_true")]
    pub deterministic: bool,
    pub threads: Option<u64>,
    pub eval_mode: Option<String>, // "parity" | "fast"
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct ToolExecuteRequest {
    pub name: String,
    pub arguments: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Tool schema (OpenAI tool definitions)
// ---------------------------------------------------------------------------

pub fn get_tool_schema() -> serde_json::Value {
    // Keep this self-contained. It mirrors `bindings/ns-py/python/nextstat/tools.py`
    // but can differ if a tool is not supported in server mode.
    let execution_schema = serde_json::json!({
        "type": "object",
        "description": "Optional execution controls. If deterministic=true (default), NextStat will attempt to enforce parity-friendly settings (eval_mode='parity'). Thread control is best-effort on the server (Rayon is global).",
        "properties": {
            "deterministic": { "type": "boolean", "default": true },
            "threads": { "type": "integer", "description": "Requested thread count (best-effort). If omitted and deterministic=true, defaults to 1." },
            "eval_mode": { "type": "string", "enum": ["parity", "fast"], "description": "Evaluation mode: 'parity' favors numerical stability; 'fast' may use approximations." }
        }
    });

    // Note: server mode does not expose `nextstat_read_root_histogram` (file ingest is a security surface).
    let tools = vec![
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "nextstat_fit",
                "description": "Run Maximum Likelihood Estimation (MLE) on a HistFactory model. Input workspace_json is a pyhf-style or HS3-style JSON workspace string.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workspace_json": { "type": "string" },
                        "model_id": { "type": "string", "description": "Optional cached model id (sha256). If present, skips parsing." },
                        "execution": execution_schema
                    },
                    "required": []
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "nextstat_ranking",
                "description": "Compute nuisance parameter ranking (impact on POI).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workspace_json": { "type": "string" },
                        "model_id": { "type": "string" },
                        "top_n": { "type": "integer" },
                        "execution": execution_schema
                    },
                    "required": []
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "nextstat_scan",
                "description": "Run a profile likelihood scan over signal strength values; returns points for plotting q(mu).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workspace_json": { "type": "string" },
                        "model_id": { "type": "string" },
                        "start": { "type": "number", "default": 0.0 },
                        "stop": { "type": "number", "default": 5.0 },
                        "points": { "type": "integer", "default": 21 },
                        "execution": execution_schema
                    },
                    "required": []
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "nextstat_hypotest",
                "description": "Run an asymptotic CLs hypothesis test at a given mu (qtilde). Returns CLs, CLs+b, CLb.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workspace_json": { "type": "string" },
                        "model_id": { "type": "string" },
                        "mu": { "type": "number" },
                        "execution": execution_schema
                    },
                    "required": ["mu"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "nextstat_upper_limit",
                "description": "Compute the 95% CL upper limit on mu via asymptotic CLs (qtilde). Returns observed limit and optional expected bands.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workspace_json": { "type": "string" },
                        "model_id": { "type": "string" },
                        "expected": { "type": "boolean", "default": false },
                        "alpha": { "type": "number", "default": 0.05 },
                        "lo": { "type": "number", "default": 0.0 },
                        "hi": { "type": ["number", "null"], "default": null },
                        "rtol": { "type": "number", "default": 1e-4 },
                        "max_iter": { "type": "integer", "default": 80 },
                        "execution": execution_schema
                    },
                    "required": []
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "nextstat_discovery_asymptotic",
                "description": "Compute asymptotic discovery statistics at mu=0: q0, z0, p0 (one-sided).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workspace_json": { "type": "string" },
                        "model_id": { "type": "string" },
                        "execution": execution_schema
                    },
                    "required": []
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "nextstat_hypotest_toys",
                "description": "Toy-based CLs hypotest at mu (qtilde). Stochastic; specify seed for reproducibility. Server mode currently uses CPU reference implementation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workspace_json": { "type": "string" },
                        "model_id": { "type": "string" },
                        "mu": { "type": "number" },
                        "n_toys": { "type": "integer", "default": 1000, "minimum": 1 },
                        "seed": { "type": "integer", "default": 42, "minimum": 0 },
                        "expected_set": { "type": "boolean", "default": false },
                        "execution": execution_schema
                    },
                    "required": ["mu"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "nextstat_workspace_audit",
                "description": "Audit a pyhf workspace JSON for compatibility and counts; returns warnings for unsupported modifier types.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workspace_json": { "type": "string" }
                    },
                    "required": ["workspace_json"]
                }
            }
        }),
    ];

    serde_json::json!({
        "schema_version": "nextstat.tool_schema.v1",
        "tools": tools
    })
}

// ---------------------------------------------------------------------------
// Execution helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum EffectiveEvalMode {
    Parity,
    Fast,
}

fn parse_eval_mode(s: Option<&str>) -> Option<EffectiveEvalMode> {
    match s {
        Some("parity") => Some(EffectiveEvalMode::Parity),
        Some("fast") => Some(EffectiveEvalMode::Fast),
        _ => None,
    }
}

struct EvalModeGuard {
    prev: ns_compute::EvalMode,
}

impl EvalModeGuard {
    fn set(mode: ns_compute::EvalMode) -> Self {
        let prev = ns_compute::eval_mode();
        ns_compute::set_eval_mode(mode);
        Self { prev }
    }
}

impl Drop for EvalModeGuard {
    fn drop(&mut self) {
        ns_compute::set_eval_mode(self.prev);
    }
}

fn effective_controls(arguments: &serde_json::Value) -> (ExecutionControls, Vec<String>) {
    let mut warnings = Vec::new();
    let exec_val = arguments.get("execution").cloned().unwrap_or(serde_json::Value::Null);
    let mut controls = if exec_val.is_null() {
        ExecutionControls { deterministic: true, threads: None, eval_mode: None }
    } else {
        match serde_json::from_value::<ExecutionControls>(exec_val) {
            Ok(v) => v,
            Err(e) => {
                warnings.push(format!("invalid execution controls; using defaults: {e}"));
                ExecutionControls { deterministic: true, threads: None, eval_mode: None }
            }
        }
    };

    // Deterministic implies parity mode and threads=1 request (best-effort).
    if controls.deterministic {
        if controls.eval_mode.as_deref() != Some("parity") {
            controls.eval_mode = Some("parity".to_string());
        }
        if controls.threads.is_none() {
            controls.threads = Some(1);
        }
    }

    (controls, warnings)
}

fn meta_base(tool_name: &str, controls: &ExecutionControls, warnings: Vec<String>) -> ToolMeta {
    ToolMeta {
        tool_name: tool_name.to_string(),
        nextstat_version: Some(ns_core::VERSION.to_string()),
        deterministic: controls.deterministic,
        eval_mode: controls
            .eval_mode
            .clone()
            .unwrap_or_else(|| match ns_compute::eval_mode() {
                ns_compute::EvalMode::Parity => "parity".to_string(),
                ns_compute::EvalMode::Fast => "fast".to_string(),
            }),
        threads_requested: controls.threads,
        threads_applied: None, // server thread pool is global; we don't mutate it per request
        device: None,
        warnings,
    }
}

// ---------------------------------------------------------------------------
// Model resolution (uses ModelPool cache)
// ---------------------------------------------------------------------------

fn resolve_model_from_args(
    state: &AppState,
    workspace_json: Option<&str>,
    model_id: Option<&str>,
) -> Result<Arc<HistFactoryModel>, String> {
    if let Some(id) = model_id {
        return state
            .model_pool
            .get(id)
            .ok_or_else(|| format!("model {id} not in cache"));
    }

    let ws = workspace_json.ok_or_else(|| "either workspace_json or model_id must be provided".to_string())?;

    let id = ModelPool::hash_workspace(ws);
    if let Some(m) = state.model_pool.get(&id) {
        return Ok(m);
    }

    let model = load_model(ws).map_err(|e| format!("workspace build error: {e}"))?;
    let _ = state.model_pool.insert(ws, model, None);
    state
        .model_pool
        .get(&id)
        .ok_or_else(|| "internal error: model inserted but not found".to_string())
}

fn load_model(json_str: &str) -> anyhow::Result<HistFactoryModel> {
    let format = ns_translate::hs3::detect::detect_format(json_str);
    match format {
        ns_translate::hs3::detect::WorkspaceFormat::Hs3 => {
            ns_translate::hs3::convert::from_hs3_default(json_str).map_err(|e| anyhow::anyhow!(e))
        }
        ns_translate::hs3::detect::WorkspaceFormat::Pyhf
        | ns_translate::hs3::detect::WorkspaceFormat::Unknown => {
            let workspace: ns_translate::pyhf::Workspace = serde_json::from_str(json_str)?;
            HistFactoryModel::from_workspace(&workspace).map_err(|e| anyhow::anyhow!(e))
        }
    }
}

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CommonWorkspaceArgs {
    workspace_json: Option<String>,
    model_id: Option<String>,
}

pub fn execute_tool(state: &AppState, req: ToolExecuteRequest) -> ToolResultEnvelope {
    let name = req.name.clone();

    let (controls, warnings) = effective_controls(&req.arguments);
    let mut meta = meta_base(&name, &controls, warnings);

    // EvalMode is process-wide: set it while holding the compute lock (caller enforces).
    let eff = parse_eval_mode(controls.eval_mode.as_deref());
    let _eval_guard = match eff {
        Some(EffectiveEvalMode::Parity) => Some(EvalModeGuard::set(ns_compute::EvalMode::Parity)),
        Some(EffectiveEvalMode::Fast) => Some(EvalModeGuard::set(ns_compute::EvalMode::Fast)),
        None => None,
    };

    let t0 = Instant::now();

    // Deterministic tool calls should not use GPU (precision + determinism).
    let allow_gpu = !controls.deterministic && state.has_gpu();
    let gpu_device = if allow_gpu { state.gpu_device.as_deref() } else { None };

    let gpu_supported = matches!(name.as_str(), "nextstat_fit" | "nextstat_ranking" | "nextstat_scan");
    if gpu_device.is_some() && !gpu_supported {
        meta.warnings.push("GPU is enabled on this server, but this tool runs on CPU in server mode.".to_string());
    }

    let tool_res = execute_tool_impl(state, &name, req.arguments, gpu_device, t0);

    match tool_res {
        Ok((value, device)) => {
            meta.device = device;
            ToolResultEnvelope::ok(&name, meta, value)
        }
        Err(msg) => ToolResultEnvelope::err(&name, meta, "ToolError", msg),
    }
}

fn execute_tool_impl(
    state: &AppState,
    name: &str,
    arguments: serde_json::Value,
    gpu_device: Option<&str>,
    t0: Instant,
) -> Result<(serde_json::Value, Option<String>), String> {
    match name {
        "nextstat_fit" => {
            #[derive(Debug, Deserialize)]
            struct Args {
                #[serde(flatten)]
                common: CommonWorkspaceArgs,
            }
            let args: Args = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            let model = resolve_model_from_args(
                state,
                args.common.workspace_json.as_deref(),
                args.common.model_id.as_deref(),
            )?;
            let mle = MaximumLikelihoodEstimator::new();

            let (fit, device) = match gpu_device {
                #[cfg(feature = "cuda")]
                Some("cuda") => (mle.fit_gpu(&*model).map_err(|e| e.to_string())?, "cuda".to_string()),
                #[cfg(feature = "metal")]
                Some("metal") => (mle.fit_metal(&*model).map_err(|e| e.to_string())?, "metal".to_string()),
                _ => (mle.fit(&*model).map_err(|e| e.to_string())?, "cpu".to_string()),
            };

            let parameter_names: Vec<String> =
                model.parameters().iter().map(|p| p.name.clone()).collect();
            let poi_index = model.poi_index();
            let (poi_value, poi_error) = if let Some(poi) = poi_index {
                (
                    fit.parameters.get(poi).copied(),
                    fit.uncertainties.get(poi).copied(),
                )
            } else {
                (None, None)
            };

            let mut params_map = serde_json::Map::new();
            let n = parameter_names
                .len()
                .min(fit.parameters.len())
                .min(fit.uncertainties.len());
            for i in 0..n {
                params_map.insert(
                    parameter_names[i].clone(),
                    serde_json::json!({ "value": fit.parameters[i], "error": fit.uncertainties[i] }),
                );
            }

            Ok((
                serde_json::json!({
                    "nll": fit.nll,
                    "converged": fit.converged,
                    "n_iter": fit.n_iter,
                    "poi_index": poi_index,
                    "poi_value": poi_value,
                    "poi_error": poi_error,
                    "parameters": params_map,
                    "wall_time_s": t0.elapsed().as_secs_f64()
                }),
                Some(device),
            ))
        }
        "nextstat_ranking" => {
            #[derive(Debug, Deserialize)]
            struct Args {
                #[serde(flatten)]
                common: CommonWorkspaceArgs,
                top_n: Option<usize>,
            }
            let args: Args = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            let model = resolve_model_from_args(
                state,
                args.common.workspace_json.as_deref(),
                args.common.model_id.as_deref(),
            )?;
            let mle = MaximumLikelihoodEstimator::new();

            let (ranking, device) = match gpu_device {
                #[cfg(feature = "cuda")]
                Some("cuda") => (
                    ns_inference::mle::ranking_gpu(&mle, &*model).map_err(|e| e.to_string())?,
                    "cuda".to_string(),
                ),
                #[cfg(feature = "metal")]
                Some("metal") => (
                    ns_inference::mle::ranking_metal(&mle, &*model).map_err(|e| e.to_string())?,
                    "metal".to_string(),
                ),
                _ => (
                    mle.ranking(&*model).map_err(|e| e.to_string())?,
                    "cpu".to_string(),
                ),
            };

            // Match the Python `nextstat.interpret.rank_impact()` contract:
            // - total_impact = |up| + |down|
            // - sort by total_impact descending (tie-break by name)
            // - assign 1-based rank
            let mut rows: Vec<serde_json::Value> = ranking
                .into_iter()
                .map(|e| {
                    let total_impact = e.delta_mu_up.abs() + e.delta_mu_down.abs();
                    serde_json::json!({
                        "name": e.name,
                        "delta_mu_up": e.delta_mu_up,
                        "delta_mu_down": e.delta_mu_down,
                        "total_impact": total_impact,
                        "pull": e.pull,
                        "constraint": e.constraint
                    })
                })
                .collect();

            rows.sort_by(|a, b| {
                let ia = a.get("total_impact").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let ib = b.get("total_impact").and_then(|x| x.as_f64()).unwrap_or(0.0);
                ib.partial_cmp(&ia)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        let na = a.get("name").and_then(|x| x.as_str()).unwrap_or("");
                        let nb = b.get("name").and_then(|x| x.as_str()).unwrap_or("");
                        na.cmp(nb)
                    })
            });

            for (i, row) in rows.iter_mut().enumerate() {
                if let Some(obj) = row.as_object_mut() {
                    obj.insert("rank".to_string(), serde_json::Value::from((i + 1) as u64));
                }
            }

            if let Some(n) = args.top_n {
                if rows.len() > n {
                    rows.truncate(n);
                }
            }

            Ok((
                serde_json::json!({
                    "ranking": rows,
                    "wall_time_s": t0.elapsed().as_secs_f64()
                }),
                Some(device),
            ))
        }
        "nextstat_scan" => {
            #[derive(Debug, Deserialize)]
            struct Args {
                #[serde(flatten)]
                common: CommonWorkspaceArgs,
                #[serde(default = "default_scan_start")]
                start: f64,
                #[serde(default = "default_scan_stop")]
                stop: f64,
                #[serde(default = "default_scan_points")]
                points: usize,
            }
            fn default_scan_start() -> f64 { 0.0 }
            fn default_scan_stop() -> f64 { 5.0 }
            fn default_scan_points() -> usize { 21 }

            let args: Args = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            let model = resolve_model_from_args(
                state,
                args.common.workspace_json.as_deref(),
                args.common.model_id.as_deref(),
            )?;

            let points = args.points.max(2).min(2001);
            let step = (args.stop - args.start) / ((points - 1) as f64);
            let mu_values: Vec<f64> = (0..points)
                .map(|i| args.start + (i as f64) * step)
                .collect();

            let mle = MaximumLikelihoodEstimator::new();

            let (scan, device) = match gpu_device {
                #[cfg(feature = "cuda")]
                Some("cuda") => (
                    ns_inference::profile_likelihood::scan_gpu(&mle, &*model, &mu_values)
                        .map_err(|e| e.to_string())?,
                    "cuda".to_string(),
                ),
                #[cfg(feature = "metal")]
                Some("metal") => (
                    ns_inference::profile_likelihood::scan_metal(&mle, &*model, &mu_values)
                        .map_err(|e| e.to_string())?,
                    "metal".to_string(),
                ),
                _ => (
                    ns_inference::profile_likelihood::scan_histfactory(&mle, &*model, &mu_values)
                        .map_err(|e| e.to_string())?,
                    "cpu".to_string(),
                ),
            };

            let points_json: Vec<serde_json::Value> = scan
                .points
                .into_iter()
                .map(|p| {
                    serde_json::json!({
                        "mu": p.mu,
                        "q_mu": p.q_mu,
                        "nll_mu": p.nll_mu,
                        "converged": p.converged,
                        "n_iter": p.n_iter
                    })
                })
                .collect();

            Ok((
                serde_json::json!({
                    "poi_index": scan.poi_index,
                    "mu_hat": scan.mu_hat,
                    "nll_hat": scan.nll_hat,
                    "mu_values": mu_values,
                    "points": points_json,
                    "wall_time_s": t0.elapsed().as_secs_f64()
                }),
                Some(device),
            ))
        }
        "nextstat_discovery_asymptotic" => {
            #[derive(Debug, Deserialize)]
            struct Args {
                #[serde(flatten)]
                common: CommonWorkspaceArgs,
            }
            let args: Args = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            let model = resolve_model_from_args(
                state,
                args.common.workspace_json.as_deref(),
                args.common.model_id.as_deref(),
            )?;
            let mle = MaximumLikelihoodEstimator::new();
            let free = mle.fit(&*model).map_err(|e| e.to_string())?;
            let poi = model
                .poi_index()
                .ok_or_else(|| "No POI defined".to_string())?;
            let mu_hat = free.parameters.get(poi).copied();
            let nll_hat = free.nll;

            let scan = ns_inference::profile_likelihood::scan_histfactory(&mle, &*model, &[0.0])
                .map_err(|e| e.to_string())?;
            if scan.points.is_empty() {
                return Err("profile_scan returned no points for mu=0".to_string());
            }
            let nll0 = scan.points[0].nll_mu;

            let mut q0 = 2.0 * (nll0 - nll_hat);
            if q0 < 0.0 {
                q0 = 0.0;
            }
            if let Some(mh) = mu_hat {
                if mh <= 0.0 {
                    q0 = 0.0;
                }
            }
            let z0 = q0.sqrt();
            let p0 = normal_sf(z0);

            Ok((
                serde_json::json!({
                    "mu_hat": mu_hat,
                    "nll_hat": nll_hat,
                    "nll_mu0": nll0,
                    "q0": q0,
                    "z0": z0,
                    "p0": p0,
                    "wall_time_s": t0.elapsed().as_secs_f64()
                }),
                Some("cpu".to_string()),
            ))
        }
        "nextstat_hypotest" => {
            #[derive(Debug, Deserialize)]
            struct Args {
                #[serde(flatten)]
                common: CommonWorkspaceArgs,
                mu: f64,
            }
            let args: Args = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            let model = resolve_model_from_args(
                state,
                args.common.workspace_json.as_deref(),
                args.common.model_id.as_deref(),
            )?;
            let mle = MaximumLikelihoodEstimator::new();
            let ctx = ns_inference::hypotest::AsymptoticCLsContext::new(&mle, &*model)
                .map_err(|e| e.to_string())?;
            let r = ctx
                .hypotest_qtilde(&mle, args.mu)
                .map_err(|e| e.to_string())?;
            Ok((
                serde_json::json!({
                    "mu": args.mu,
                    "cls": r.cls,
                    "clsb": r.clsb,
                    "clb": r.clb,
                    "wall_time_s": t0.elapsed().as_secs_f64()
                }),
                Some("cpu".to_string()),
            ))
        }
        "nextstat_upper_limit" => {
            #[derive(Debug, Deserialize)]
            struct Args {
                #[serde(flatten)]
                common: CommonWorkspaceArgs,
                #[serde(default)]
                expected: bool,
                #[serde(default = "default_alpha")]
                alpha: f64,
                #[serde(default)]
                lo: f64,
                hi: Option<f64>,
                #[serde(default = "default_rtol")]
                rtol: f64,
                #[serde(default = "default_max_iter")]
                max_iter: usize,
            }
            fn default_alpha() -> f64 { 0.05 }
            fn default_rtol() -> f64 { 1e-4 }
            fn default_max_iter() -> usize { 80 }

            let args: Args = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            let model = resolve_model_from_args(
                state,
                args.common.workspace_json.as_deref(),
                args.common.model_id.as_deref(),
            )?;
            let mle = MaximumLikelihoodEstimator::new();
            let ctx = ns_inference::hypotest::AsymptoticCLsContext::new(&mle, &*model)
                .map_err(|e| e.to_string())?;

            let hi = args.hi.unwrap_or(10.0);
            if args.expected {
                let (obs, exp) = ctx
                    .upper_limits_qtilde_bisection(
                        &mle,
                        args.alpha,
                        args.lo,
                        hi,
                        args.rtol,
                        args.max_iter,
                    )
                    .map_err(|e| e.to_string())?;
                Ok((
                    serde_json::json!({
                        "alpha": args.alpha,
                        "obs_limit": obs,
                        "exp_limits": exp,
                        "wall_time_s": t0.elapsed().as_secs_f64()
                    }),
                    Some("cpu".to_string()),
                ))
            } else {
                let obs = ctx
                    .upper_limit_qtilde(&mle, args.alpha, args.lo, hi, args.rtol, args.max_iter)
                    .map_err(|e| e.to_string())?;
                Ok((
                    serde_json::json!({
                        "alpha": args.alpha,
                        "obs_limit": obs,
                        "wall_time_s": t0.elapsed().as_secs_f64()
                    }),
                    Some("cpu".to_string()),
                ))
            }
        }
        "nextstat_hypotest_toys" => {
            #[derive(Debug, Deserialize)]
            struct Args {
                #[serde(flatten)]
                common: CommonWorkspaceArgs,
                mu: f64,
                #[serde(default = "default_n_toys")]
                n_toys: usize,
                #[serde(default = "default_seed")]
                seed: u64,
                #[serde(default)]
                expected_set: bool,
            }
            fn default_n_toys() -> usize { 1000 }
            fn default_seed() -> u64 { 42 }

            let args: Args = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            let model = resolve_model_from_args(
                state,
                args.common.workspace_json.as_deref(),
                args.common.model_id.as_deref(),
            )?;
            let mle = MaximumLikelihoodEstimator::new();

            if args.expected_set {
                let r = ns_inference::toybased::hypotest_qtilde_toys_expected_set(
                    &mle, &*model, args.mu, args.n_toys, args.seed,
                )
                .map_err(|e| e.to_string())?;
                Ok((
                    serde_json::json!({
                        "mu": args.mu,
                        "n_toys": args.n_toys,
                        "seed": args.seed,
                        "expected_set": true,
                        "raw": {
                            "observed": {
                                "cls": r.observed.cls,
                                "clsb": r.observed.clsb,
                                "clb": r.observed.clb
                            },
                            "expected": r.expected
                        },
                        "wall_time_s": t0.elapsed().as_secs_f64()
                    }),
                    Some("cpu".to_string()),
                ))
            } else {
                let r = ns_inference::toybased::hypotest_qtilde_toys(
                    &mle, &*model, args.mu, args.n_toys, args.seed,
                )
                .map_err(|e| e.to_string())?;
                Ok((
                    serde_json::json!({
                        "mu": args.mu,
                        "n_toys": args.n_toys,
                        "seed": args.seed,
                        "expected_set": false,
                        "raw": {
                            "cls": r.cls,
                            "clsb": r.clsb,
                            "clb": r.clb
                        },
                        "wall_time_s": t0.elapsed().as_secs_f64()
                    }),
                    Some("cpu".to_string()),
                ))
            }
        }
        "nextstat_workspace_audit" => {
            #[derive(Debug, Deserialize)]
            struct Args {
                workspace_json: String,
            }
            let args: Args = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            let json: serde_json::Value =
                serde_json::from_str(&args.workspace_json).map_err(|e| e.to_string())?;
            let audit = pyhf_audit::workspace_audit(&json);
            Ok((
                serde_json::to_value(audit).map_err(|e| e.to_string())?,
                Some("cpu".to_string()),
            ))
        }
        other => Err(format!("Unknown tool: {other}")),
    }
}

fn normal_sf(z: f64) -> f64 {
    // One-sided survival function for standard normal:
    // SF(z) = 0.5 * erfc(z / sqrt(2))
    0.5 * statrs::function::erf::erfc(z / std::f64::consts::SQRT_2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize_envelope(mut v: serde_json::Value) -> serde_json::Value {
        fn drop_n_iter(x: &mut serde_json::Value) {
            match x {
                serde_json::Value::Object(obj) => {
                    obj.remove("n_iter");
                    for v in obj.values_mut() {
                        drop_n_iter(v);
                    }
                }
                serde_json::Value::Array(arr) => {
                    for v in arr.iter_mut() {
                        drop_n_iter(v);
                    }
                }
                _ => {}
            }
        }

        drop_n_iter(&mut v);

        // Keep comparisons focused on semantics, not build metadata or timings.
        if let Some(meta) = v.get_mut("meta").and_then(|m| m.as_object_mut()) {
            meta.remove("nextstat_version");
            meta.remove("threads_applied");
            meta.remove("device");
            meta.remove("warnings");
        }
        if let Some(res) = v.get_mut("result").and_then(|r| r.as_object_mut()) {
            res.remove("wall_time_s");
            // Local tool goldens do not include this convenience field (mu is already in points).
            res.remove("mu_values");
        }
        v
    }

    fn assert_json_close(a: &serde_json::Value, b: &serde_json::Value, path: &str) {
        use serde_json::Value;

        const RTOL: f64 = 1e-6;
        const ATOL: f64 = 1e-8;

        match (a, b) {
            (Value::Number(na), Value::Number(nb)) => {
                let af = na.as_f64().unwrap_or(f64::NAN);
                let bf = nb.as_f64().unwrap_or(f64::NAN);
                let diff = (af - bf).abs();
                if diff <= ATOL {
                    return;
                }
                let denom = af.abs().max(bf.abs()).max(1.0);
                if diff / denom <= RTOL {
                    return;
                }
                panic!("{path}: {af} != {bf} (diff={diff}, rtol={RTOL}, atol={ATOL})");
            }
            (Value::Object(oa), Value::Object(ob)) => {
                let ka: std::collections::BTreeSet<_> = oa.keys().collect();
                let kb: std::collections::BTreeSet<_> = ob.keys().collect();
                assert_eq!(ka, kb, "{path}: key mismatch");
                for k in oa.keys() {
                    assert_json_close(&oa[k], &ob[k], &format!("{path}.{k}"));
                }
            }
            (Value::Array(aa), Value::Array(ab)) => {
                assert_eq!(aa.len(), ab.len(), "{path}: length mismatch");
                for (i, (xa, xb)) in aa.iter().zip(ab.iter()).enumerate() {
                    assert_json_close(xa, xb, &format!("{path}[{i}]"));
                }
            }
            _ => {
                assert_eq!(a, b, "{path}: value mismatch");
            }
        }
    }

    #[test]
    fn tool_execute_fit_smoke_ok() {
        let state = AppState::new(None);
        let ws = include_str!("../../../tests/fixtures/simple_workspace.json");

        let req = ToolExecuteRequest {
            name: "nextstat_fit".to_string(),
            arguments: serde_json::json!({
                "workspace_json": ws,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.meta.tool_name, "nextstat_fit");
        let obj = out.result.as_object().expect("result must be an object");
        for k in ["nll", "converged", "n_iter", "poi_index", "poi_value", "poi_error", "parameters"] {
            assert!(obj.contains_key(k), "missing key {k} in result");
        }
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_simple_workspace_deterministic() {
        let state = AppState::new(None);
        let ws = include_str!("../../../tests/fixtures/simple_workspace.json");
        let gold_raw = include_str!("../../../tests/fixtures/tool_goldens/simple_workspace_deterministic.v1.json");
        let gold: serde_json::Value = serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let tools = gold
            .get("tools")
            .and_then(|x| x.as_object())
            .expect("golden must contain tools map");

        // Keep this tight: only tools that should match across local/server in deterministic mode.
        // (server mode intentionally does not expose file ingest tools like ROOT histogram reads).
        let cases: &[(&str, serde_json::Value)] = &[
            (
                "nextstat_fit",
                serde_json::json!({ "workspace_json": ws, "execution": { "deterministic": true } }),
            ),
            (
                "nextstat_hypotest",
                serde_json::json!({ "workspace_json": ws, "mu": 1.0, "execution": { "deterministic": true } }),
            ),
            (
                "nextstat_upper_limit",
                serde_json::json!({ "workspace_json": ws, "expected": true, "execution": { "deterministic": true } }),
            ),
            (
                "nextstat_ranking",
                serde_json::json!({ "workspace_json": ws, "top_n": 5, "execution": { "deterministic": true } }),
            ),
            (
                "nextstat_scan",
                serde_json::json!({ "workspace_json": ws, "start": 0.0, "stop": 2.0, "points": 5, "execution": { "deterministic": true } }),
            ),
            (
                "nextstat_discovery_asymptotic",
                serde_json::json!({ "workspace_json": ws, "execution": { "deterministic": true } }),
            ),
            (
                "nextstat_workspace_audit",
                serde_json::json!({ "workspace_json": ws, "execution": { "deterministic": true } }),
            ),
        ];

        let _guard = state.compute_lock.blocking_lock();
        for (name, args) in cases {
            let req = ToolExecuteRequest {
                name: (*name).to_string(),
                arguments: args.clone(),
            };
            let out = execute_tool(&state, req);
            assert!(
                out.ok,
                "expected ok=true for {name}, got error={:?}",
                out.error
            );
            assert_eq!(out.schema_version, "nextstat.tool_result.v1");
            assert_eq!(out.meta.tool_name, *name);
            assert!(out.meta.deterministic, "meta.deterministic must be true for {name}");
            assert_eq!(out.meta.eval_mode, "parity", "meta.eval_mode must be parity for {name}");
            assert_eq!(
                out.meta.threads_requested,
                Some(1),
                "meta.threads_requested must be 1 for {name}"
            );

            let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
            let want_raw = tools
                .get(*name)
                .unwrap_or_else(|| panic!("missing golden for {name}"));
            let want = normalize_envelope(want_raw.clone());

            assert_json_close(&got, &want, &format!("tool:{name}"));
        }
    }
}
