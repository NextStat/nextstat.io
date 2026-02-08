//! HTTP route handlers for the NextStat server.
//!
//! All endpoints live under `/v1/` and accept/return JSON.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use ns_core::traits::LogDensityModel;

use crate::state::SharedState;
use crate::tools::{ToolExecuteRequest, ToolResultEnvelope};

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

pub fn router() -> Router<SharedState> {
    Router::new()
        .route("/v1/fit", post(fit_handler))
        .route("/v1/ranking", post(ranking_handler))
        .route("/v1/batch/fit", post(batch_fit_handler))
        .route("/v1/batch/toys", post(batch_toys_handler))
        .route("/v1/tools/execute", post(tools_execute_handler))
        .route("/v1/tools/schema", get(tools_schema_handler))
        .route("/v1/models", post(upload_model_handler))
        .route("/v1/models", get(list_models_handler))
        .route("/v1/models/{id}", delete(delete_model_handler))
        .route("/v1/health", get(health_handler))
}

// ---------------------------------------------------------------------------
// POST /v1/fit
// ---------------------------------------------------------------------------

/// Request body for `/v1/fit`.
#[derive(Debug, Deserialize)]
struct FitRequest {
    /// pyhf or HS3 workspace JSON (the full object, not a string).
    /// Can be omitted if `model_id` is provided.
    workspace: Option<serde_json::Value>,

    /// Cached model ID (SHA-256 hash). If provided, skips workspace parsing.
    model_id: Option<String>,

    /// Use GPU if available on this server (default: true).
    #[serde(default = "default_true")]
    gpu: bool,
}

/// Response body for `/v1/fit`.
#[derive(Debug, Serialize)]
struct FitResponse {
    parameter_names: Vec<String>,
    poi_index: Option<usize>,
    bestfit: Vec<f64>,
    uncertainties: Vec<f64>,
    nll: f64,
    twice_nll: f64,
    converged: bool,
    n_iter: usize,
    n_fev: usize,
    n_gev: usize,
    covariance: Option<Vec<f64>>,
    device: String,
    wall_time_s: f64,
}

async fn fit_handler(
    State(state): State<SharedState>,
    Json(req): Json<FitRequest>,
) -> Result<Json<FitResponse>, AppError> {
    state.inflight.fetch_add(1, Ordering::Relaxed);
    let _dec = DecrementOnDrop(&state.inflight);
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    let use_gpu = req.gpu && state.has_gpu();
    let gpu_device = if use_gpu { state.gpu_device.clone() } else { None };
    let gpu_lock = if use_gpu { Some(Arc::clone(&state)) } else { None };

    let pool_ref = Arc::clone(&state);

    // Offload blocking compute to a Rayon/blocking thread
    let result = tokio::task::spawn_blocking(move || {
        let t0 = Instant::now();

        let _compute_guard = pool_ref.compute_lock.blocking_lock();

        let model = resolve_model(&pool_ref, req.workspace.as_ref(), req.model_id.as_deref())?;

        let mle = ns_inference::mle::MaximumLikelihoodEstimator::new();

        // Acquire GPU lock if needed (synchronous — we're in a blocking thread)
        let _gpu_guard = if let Some(ref st) = gpu_lock {
            Some(st.gpu_lock.blocking_lock())
        } else {
            None
        };

        let (fit_result, device) = match gpu_device.as_deref() {
            #[cfg(feature = "cuda")]
            Some("cuda") => {
                let r = mle.fit_gpu(&*model)
                    .map_err(|e| AppError::internal(format!("CUDA fit failed: {e}")))?;
                (r, "cuda".to_string())
            }
            #[cfg(not(feature = "cuda"))]
            Some("cuda") => {
                return Err(AppError::internal("CUDA not compiled".into()));
            }
            #[cfg(feature = "metal")]
            Some("metal") => {
                let r = mle
                    .fit_metal(&*model)
                    .map_err(|e| AppError::internal(format!("Metal fit failed: {e}")))?;
                (r, "metal".to_string())
            }
            #[cfg(not(feature = "metal"))]
            Some("metal") => {
                return Err(AppError::internal("Metal not compiled".into()));
            }
            _ => {
                let r = mle.fit(&*model)
                    .map_err(|e| AppError::internal(format!("CPU fit failed: {e}")))?;
                (r, "cpu".to_string())
            }
        };

        let wall_time_s = t0.elapsed().as_secs_f64();

        let parameter_names: Vec<String> =
            model.parameters().iter().map(|p| p.name.clone()).collect();
        let poi_index = model.poi_index();

        Ok(FitResponse {
            parameter_names,
            poi_index,
            bestfit: fit_result.parameters,
            uncertainties: fit_result.uncertainties,
            nll: fit_result.nll,
            twice_nll: 2.0 * fit_result.nll,
            converged: fit_result.converged,
            n_iter: fit_result.n_iter,
            n_fev: fit_result.n_fev,
            n_gev: fit_result.n_gev,
            covariance: fit_result.covariance,
            device,
            wall_time_s,
        })
    })
    .await
    .map_err(|e| AppError::internal(format!("task panicked: {e}")))?;

    result.map(Json)
}

// ---------------------------------------------------------------------------
// POST /v1/ranking
// ---------------------------------------------------------------------------

/// Request body for `/v1/ranking`.
#[derive(Debug, Deserialize)]
struct RankingRequest {
    /// pyhf or HS3 workspace JSON. Can be omitted if `model_id` is provided.
    workspace: Option<serde_json::Value>,

    /// Cached model ID (SHA-256 hash).
    model_id: Option<String>,

    /// Use GPU if available (default: true).
    #[serde(default = "default_true")]
    gpu: bool,
}

/// Single ranking entry in the response.
#[derive(Debug, Serialize)]
struct RankingEntryResponse {
    name: String,
    delta_mu_up: f64,
    delta_mu_down: f64,
    pull: f64,
    constraint: f64,
}

/// Response body for `/v1/ranking`.
#[derive(Debug, Serialize)]
struct RankingResponse {
    entries: Vec<RankingEntryResponse>,
    device: String,
    wall_time_s: f64,
}

async fn ranking_handler(
    State(state): State<SharedState>,
    Json(req): Json<RankingRequest>,
) -> Result<Json<RankingResponse>, AppError> {
    state.inflight.fetch_add(1, Ordering::Relaxed);
    let _dec = DecrementOnDrop(&state.inflight);
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    let use_gpu = req.gpu && state.has_gpu();
    let gpu_device = if use_gpu { state.gpu_device.clone() } else { None };

    let gpu_lock = if use_gpu { Some(Arc::clone(&state)) } else { None };

    let pool_ref = Arc::clone(&state);

    let result = tokio::task::spawn_blocking(move || {
        let t0 = Instant::now();

        let _compute_guard = pool_ref.compute_lock.blocking_lock();

        let model = resolve_model(&pool_ref, req.workspace.as_ref(), req.model_id.as_deref())?;
        let mle = ns_inference::mle::MaximumLikelihoodEstimator::new();

        let _gpu_guard = if let Some(ref st) = gpu_lock {
            Some(st.gpu_lock.blocking_lock())
        } else {
            None
        };

        let (ranking, device) = match gpu_device.as_deref() {
            #[cfg(feature = "cuda")]
            Some("cuda") => {
                let r = ns_inference::mle::ranking_gpu(&mle, &*model)
                    .map_err(|e| AppError::internal(format!("CUDA ranking failed: {e}")))?;
                (r, "cuda".to_string())
            }
            #[cfg(not(feature = "cuda"))]
            Some("cuda") => {
                return Err(AppError::internal("CUDA not compiled".into()));
            }
            #[cfg(feature = "metal")]
            Some("metal") => {
                let r = ns_inference::mle::ranking_metal(&mle, &*model)
                    .map_err(|e| AppError::internal(format!("Metal ranking failed: {e}")))?;
                (r, "metal".to_string())
            }
            #[cfg(not(feature = "metal"))]
            Some("metal") => {
                return Err(AppError::internal("Metal not compiled".into()));
            }
            _ => {
                let r = mle.ranking(&*model)
                    .map_err(|e| AppError::internal(format!("CPU ranking failed: {e}")))?;
                (r, "cpu".to_string())
            }
        };

        let wall_time_s = t0.elapsed().as_secs_f64();

        let entries = ranking
            .into_iter()
            .map(|e| RankingEntryResponse {
                name: e.name,
                delta_mu_up: e.delta_mu_up,
                delta_mu_down: e.delta_mu_down,
                pull: e.pull,
                constraint: e.constraint,
            })
            .collect();

        Ok(RankingResponse { entries, device, wall_time_s })
    })
    .await
    .map_err(|e| AppError::internal(format!("task panicked: {e}")))?;

    result.map(Json)
}

// ---------------------------------------------------------------------------
// POST /v1/batch/fit
// ---------------------------------------------------------------------------

/// Request body for `/v1/batch/fit`.
#[derive(Debug, Deserialize)]
struct BatchFitRequest {
    /// Array of pyhf/HS3 workspace JSON objects.
    workspaces: Vec<serde_json::Value>,

    /// Use GPU if available (default: true). On GPU, fits run sequentially;
    /// on CPU they run in parallel via Rayon.
    #[serde(default = "default_true")]
    gpu: bool,
}

/// Response body for `/v1/batch/fit`.
#[derive(Debug, Serialize)]
struct BatchFitResponse {
    results: Vec<BatchFitItem>,
    device: String,
    wall_time_s: f64,
}

#[derive(Debug, Serialize)]
struct BatchFitItem {
    index: usize,
    #[serde(flatten)]
    result: Option<FitResponse>,
    error: Option<String>,
}

async fn batch_fit_handler(
    State(state): State<SharedState>,
    Json(req): Json<BatchFitRequest>,
) -> Result<Json<BatchFitResponse>, AppError> {
    if req.workspaces.is_empty() {
        return Err(AppError::bad_request("workspaces array must be non-empty".into()));
    }
    if req.workspaces.len() > 100 {
        return Err(AppError::bad_request("workspaces array exceeds max batch size of 100".into()));
    }

    state.inflight.fetch_add(1, Ordering::Relaxed);
    let _dec = DecrementOnDrop(&state.inflight);
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    let use_gpu = req.gpu && state.has_gpu();
    let gpu_device = if use_gpu { state.gpu_device.clone() } else { None };
    let gpu_lock = if use_gpu { Some(Arc::clone(&state)) } else { None };

    let st = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || {
        let t0 = Instant::now();

        let _compute_guard = st.compute_lock.blocking_lock();

        let _gpu_guard = if let Some(ref st) = gpu_lock {
            Some(st.gpu_lock.blocking_lock())
        } else {
            None
        };

        let device = gpu_device.as_deref().unwrap_or("cpu").to_string();
        let mle = ns_inference::mle::MaximumLikelihoodEstimator::new();

        let items: Vec<BatchFitItem> = if gpu_device.is_some() {
            // GPU: sequential (GPU lock held, single device)
            req.workspaces
                .iter()
                .enumerate()
                .map(|(i, ws)| {
                    match fit_one_workspace(ws, &mle, gpu_device.as_deref()) {
                        Ok(resp) => BatchFitItem { index: i, result: Some(resp), error: None },
                        Err(e) => BatchFitItem { index: i, result: None, error: Some(e.message) },
                    }
                })
                .collect()
        } else {
            // CPU: parallel via Rayon
            req.workspaces
                .par_iter()
                .enumerate()
                .map(|(i, ws)| {
                    match fit_one_workspace(ws, &mle, None) {
                        Ok(resp) => BatchFitItem { index: i, result: Some(resp), error: None },
                        Err(e) => BatchFitItem { index: i, result: None, error: Some(e.message) },
                    }
                })
                .collect()
        };

        let wall_time_s = t0.elapsed().as_secs_f64();
        Ok(BatchFitResponse { results: items, device, wall_time_s })
    })
    .await
    .map_err(|e| AppError::internal(format!("task panicked: {e}")))?;

    result.map(Json)
}

fn fit_one_workspace(
    ws: &serde_json::Value,
    mle: &ns_inference::mle::MaximumLikelihoodEstimator,
    gpu_device: Option<&str>,
) -> Result<FitResponse, AppError> {
    let t0 = Instant::now();
    let json_str = serde_json::to_string(ws)
        .map_err(|e| AppError::bad_request(format!("invalid workspace JSON: {e}")))?;
    let model = load_model(&json_str)?;

    let (fit_result, device) = match gpu_device {
        #[cfg(feature = "cuda")]
        Some("cuda") => {
            let r = mle
                .fit_gpu(&model)
                .map_err(|e| AppError::internal(format!("CUDA fit failed: {e}")))?;
            (r, "cuda".to_string())
        }
        #[cfg(not(feature = "cuda"))]
        Some("cuda") => {
            return Err(AppError::internal("CUDA not compiled".into()));
        }
        #[cfg(feature = "metal")]
        Some("metal") => {
            let r = mle
                .fit_metal(&model)
                .map_err(|e| AppError::internal(format!("Metal fit failed: {e}")))?;
            (r, "metal".to_string())
        }
        #[cfg(not(feature = "metal"))]
        Some("metal") => {
            return Err(AppError::internal("Metal not compiled".into()));
        }
        _ => {
            let r = mle
                .fit(&model)
                .map_err(|e| AppError::internal(format!("CPU fit failed: {e}")))?;
            (r, "cpu".to_string())
        }
    };

    let wall_time_s = t0.elapsed().as_secs_f64();
    let parameter_names: Vec<String> = model.parameters().iter().map(|p| p.name.clone()).collect();
    let poi_index = model.poi_index();

    Ok(FitResponse {
        parameter_names,
        poi_index,
        bestfit: fit_result.parameters,
        uncertainties: fit_result.uncertainties,
        nll: fit_result.nll,
        twice_nll: 2.0 * fit_result.nll,
        converged: fit_result.converged,
        n_iter: fit_result.n_iter,
        n_fev: fit_result.n_fev,
        n_gev: fit_result.n_gev,
        covariance: fit_result.covariance,
        device,
        wall_time_s,
    })
}

// ---------------------------------------------------------------------------
// POST /v1/batch/toys
// ---------------------------------------------------------------------------

/// Request body for `/v1/batch/toys`.
#[derive(Debug, Deserialize)]
struct BatchToysRequest {
    /// pyhf or HS3 workspace JSON.
    workspace: serde_json::Value,

    /// Parameters to generate toys at (e.g., best-fit or Asimov). If omitted,
    /// uses the model's default initial parameters.
    params: Option<Vec<f64>>,

    /// Number of pseudo-experiments (default: 1000).
    #[serde(default = "default_n_toys")]
    n_toys: usize,

    /// Random seed (default: 42).
    #[serde(default = "default_seed")]
    seed: u64,

    /// Use GPU if available (default: true).
    #[serde(default = "default_true")]
    gpu: bool,
}

/// Response body for `/v1/batch/toys`.
#[derive(Debug, Serialize)]
struct BatchToysResponse {
    n_toys: usize,
    n_converged: usize,
    n_failed: usize,
    results: Vec<ToyFitItem>,
    device: String,
    wall_time_s: f64,
}

#[derive(Debug, Serialize)]
struct ToyFitItem {
    bestfit: Vec<f64>,
    nll: f64,
    converged: bool,
    n_iter: usize,
}

async fn batch_toys_handler(
    State(state): State<SharedState>,
    Json(req): Json<BatchToysRequest>,
) -> Result<Json<BatchToysResponse>, AppError> {
    if req.n_toys == 0 || req.n_toys > 100_000 {
        return Err(AppError::bad_request(
            "n_toys must be between 1 and 100000".into(),
        ));
    }

    state.inflight.fetch_add(1, Ordering::Relaxed);
    let _dec = DecrementOnDrop(&state.inflight);
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    let use_gpu = req.gpu && state.has_gpu();
    let gpu_device = if use_gpu { state.gpu_device.clone() } else { None };
    let gpu_lock = if use_gpu { Some(Arc::clone(&state)) } else { None };

    let st = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || {
        let t0 = Instant::now();

        let _compute_guard = st.compute_lock.blocking_lock();

        let json_str = serde_json::to_string(&req.workspace)
            .map_err(|e| AppError::bad_request(format!("invalid workspace JSON: {e}")))?;
        let model = load_model(&json_str)?;

        let params = req.params.unwrap_or_else(|| model.parameter_init());
        if params.len() != model.n_params() {
            return Err(AppError::bad_request(format!(
                "params length {} != model parameters {}",
                params.len(),
                model.n_params()
            )));
        }

        let _gpu_guard = if let Some(ref st) = gpu_lock {
            Some(st.gpu_lock.blocking_lock())
        } else {
            None
        };

        let (fit_results, device) = match gpu_device.as_deref() {
            #[cfg(feature = "cuda")]
            Some("cuda") => {
                let r = ns_inference::gpu_batch::fit_toys_batch_gpu(
                    &model, &params, req.n_toys, req.seed, None,
                )
                .map_err(|e| AppError::internal(format!("CUDA batch toys failed: {e}")))?;
                (r, "cuda".to_string())
            }
            #[cfg(not(feature = "cuda"))]
            Some("cuda") => {
                return Err(AppError::internal("CUDA not compiled".into()));
            }
            #[cfg(feature = "metal")]
            Some("metal") => {
                let r = ns_inference::metal_batch::fit_toys_batch_metal(
                    &model, &params, req.n_toys, req.seed, None,
                )
                .map_err(|e| AppError::internal(format!("Metal batch toys failed: {e}")))?;
                (r, "metal".to_string())
            }
            #[cfg(not(feature = "metal"))]
            Some("metal") => {
                return Err(AppError::internal("Metal not compiled".into()));
            }
            _ => {
                let r = ns_inference::batch::fit_toys_batch(
                    &model, &params, req.n_toys, req.seed, None,
                );
                (r, "cpu".to_string())
            }
        };

        let wall_time_s = t0.elapsed().as_secs_f64();

        let mut n_converged = 0usize;
        let mut n_failed = 0usize;
        let results: Vec<ToyFitItem> = fit_results
            .into_iter()
            .filter_map(|r| match r {
                Ok(fr) => {
                    if fr.converged {
                        n_converged += 1;
                    }
                    Some(ToyFitItem {
                        bestfit: fr.parameters,
                        nll: fr.nll,
                        converged: fr.converged,
                        n_iter: fr.n_iter,
                    })
                }
                Err(_) => {
                    n_failed += 1;
                    None
                }
            })
            .collect();

        Ok(BatchToysResponse {
            n_toys: results.len(),
            n_converged,
            n_failed,
            results,
            device,
            wall_time_s,
        })
    })
    .await
    .map_err(|e| AppError::internal(format!("task panicked: {e}")))?;

    result.map(Json)
}

fn default_n_toys() -> usize {
    1000
}

fn default_seed() -> u64 {
    42
}

// ---------------------------------------------------------------------------
// GET /v1/health
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// POST /v1/models
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct UploadModelRequest {
    workspace: serde_json::Value,
    name: Option<String>,
}

#[derive(Debug, Serialize)]
struct UploadModelResponse {
    model_id: String,
    n_params: usize,
    n_channels: usize,
    cached: bool,
}

async fn upload_model_handler(
    State(state): State<SharedState>,
    Json(req): Json<UploadModelRequest>,
) -> Result<Json<UploadModelResponse>, AppError> {
    let pool_ref = Arc::clone(&state);

    let result = tokio::task::spawn_blocking(move || {
        let json_str = serde_json::to_string(&req.workspace)
            .map_err(|e| AppError::bad_request(format!("invalid workspace JSON: {e}")))?;
        let model = load_model(&json_str)?;
        let n_params = model.parameters().len();
        let n_channels = model.n_channels();
        let model_id = pool_ref.model_pool.insert(&json_str, model, req.name);
        Ok(UploadModelResponse { model_id, n_params, n_channels, cached: true })
    })
    .await
    .map_err(|e| AppError::internal(format!("task panicked: {e}")))?;

    result.map(Json)
}

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------

async fn list_models_handler(
    State(state): State<SharedState>,
) -> Json<Vec<crate::pool::ModelInfo>> {
    Json(state.model_pool.list())
}

// ---------------------------------------------------------------------------
// DELETE /v1/models/:id
// ---------------------------------------------------------------------------

async fn delete_model_handler(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    if state.model_pool.remove(&id) {
        Ok(Json(serde_json::json!({ "deleted": true, "model_id": id })))
    } else {
        Err(AppError::not_found(format!("model {id} not in cache")))
    }
}

// ---------------------------------------------------------------------------
// GET /v1/health
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    uptime_s: f64,
    device: String,
    eval_mode: &'static str,
    inflight: u64,
    total_requests: u64,
    cached_models: usize,
}

async fn health_handler(State(state): State<SharedState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: ns_core::VERSION,
        uptime_s: state.started_at.elapsed().as_secs_f64(),
        device: state.device_str().to_string(),
        eval_mode: match ns_compute::eval_mode() {
            ns_compute::EvalMode::Parity => "parity",
            ns_compute::EvalMode::Fast => "fast",
        },
        inflight: state.inflight.load(Ordering::Relaxed),
        total_requests: state.total_requests.load(Ordering::Relaxed),
        cached_models: state.model_pool.len(),
    })
}

// ---------------------------------------------------------------------------
// Tool API
// ---------------------------------------------------------------------------

async fn tools_schema_handler() -> Json<serde_json::Value> {
    Json(crate::tools::get_tool_schema())
}

async fn tools_execute_handler(
    State(state): State<SharedState>,
    Json(req): Json<ToolExecuteRequest>,
) -> Json<ToolResultEnvelope> {
    state.inflight.fetch_add(1, Ordering::Relaxed);
    let _dec = DecrementOnDrop(&state.inflight);
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    // Execute under compute lock to avoid races with process-wide EvalMode.
    let st = Arc::clone(&state);
    let out = tokio::task::spawn_blocking(move || {
        let _compute_guard = st.compute_lock.blocking_lock();
        crate::tools::execute_tool(&st, req)
    })
    .await
    .unwrap_or_else(|e| {
        // Return an envelope-shaped error even if the task panicked.
        let meta = crate::tools::ToolMeta {
            tool_name: "unknown".to_string(),
            nextstat_version: Some(ns_core::VERSION.to_string()),
            deterministic: true,
            eval_mode: match ns_compute::eval_mode() {
                ns_compute::EvalMode::Parity => "parity".to_string(),
                ns_compute::EvalMode::Fast => "fast".to_string(),
            },
            threads_requested: Some(1),
            threads_applied: None,
            device: Some(state.device_str().to_string()),
            warnings: vec![format!("task panicked: {e}")],
        };
        ToolResultEnvelope::err("unknown", meta, "Panic", "server task panicked".to_string())
    });

    Json(out)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_true() -> bool {
    true
}

/// Resolve a model from either a workspace JSON or a cached model_id.
/// Returns an `Arc` — cheap clone from cache, or wraps a freshly parsed model.
fn resolve_model(
    state: &crate::state::AppState,
    workspace: Option<&serde_json::Value>,
    model_id: Option<&str>,
) -> Result<Arc<ns_translate::pyhf::HistFactoryModel>, AppError> {
    match (model_id, workspace) {
        (Some(id), _) => state
            .model_pool
            .get(id)
            .ok_or_else(|| AppError::not_found(format!("model {id} not in cache"))),
        (None, Some(ws)) => {
            let json_str = serde_json::to_string(ws)
                .map_err(|e| AppError::bad_request(format!("invalid workspace JSON: {e}")))?;
            load_model(&json_str).map(Arc::new)
        }
        (None, None) => Err(AppError::bad_request(
            "either 'workspace' or 'model_id' must be provided".into(),
        )),
    }
}

/// Load a HistFactoryModel from a JSON string (auto-detects HS3 vs pyhf).
fn load_model(json_str: &str) -> Result<ns_translate::pyhf::HistFactoryModel, AppError> {
    let format = ns_translate::hs3::detect::detect_format(json_str);
    match format {
        ns_translate::hs3::detect::WorkspaceFormat::Hs3 => {
            ns_translate::hs3::convert::from_hs3_default(json_str)
                .map_err(|e| AppError::bad_request(format!("HS3 parse error: {e}")))
        }
        ns_translate::hs3::detect::WorkspaceFormat::Pyhf
        | ns_translate::hs3::detect::WorkspaceFormat::Unknown => {
            let workspace: ns_translate::pyhf::Workspace = serde_json::from_str(json_str)
                .map_err(|e| AppError::bad_request(format!("pyhf JSON parse error: {e}")))?;
            ns_translate::pyhf::HistFactoryModel::from_workspace(&workspace)
                .map_err(|e| AppError::bad_request(format!("workspace build error: {e}")))
        }
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Structured JSON error response.
#[derive(Debug)]
struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(msg: String) -> Self {
        Self { status: StatusCode::BAD_REQUEST, message: msg }
    }

    fn internal(msg: String) -> Self {
        Self { status: StatusCode::INTERNAL_SERVER_ERROR, message: msg }
    }

    fn not_found(msg: String) -> Self {
        Self { status: StatusCode::NOT_FOUND, message: msg }
    }

}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let body = serde_json::json!({
            "error": self.message,
        });
        (self.status, Json(body)).into_response()
    }
}

/// RAII guard to decrement an atomic counter on drop.
struct DecrementOnDrop<'a>(&'a std::sync::atomic::AtomicU64);

impl Drop for DecrementOnDrop<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}
