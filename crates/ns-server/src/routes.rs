//! HTTP route handlers for the NextStat server.
//!
//! All endpoints live under `/v1/` and accept/return JSON.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use ns_core::PoiModel;
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
        .route("/v1/unbinned/fit", post(unbinned_fit_handler))
        .route("/v1/nlme/fit", post(nlme_fit_handler))
        .route("/v1/jobs/submit", post(job_submit_handler))
        .route("/v1/jobs/{id}", get(job_status_handler))
        .route("/v1/jobs/{id}", delete(job_cancel_handler))
        .route("/v1/jobs", get(job_list_handler))
        .route("/v1/tools/execute", post(tools_execute_handler))
        .route("/v1/tools/schema", get(tools_schema_handler))
        .route("/v1/models", post(upload_model_handler))
        .route("/v1/models", get(list_models_handler))
        .route("/v1/models/{id}", delete(delete_model_handler))
        .route("/v1/health", get(health_handler))
        .route("/v1/openapi.json", get(openapi_handler))
}

// ---------------------------------------------------------------------------
// POST /v1/fit
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum GpuSelector {
    Bool(bool),
    String(String),
}

impl Default for GpuSelector {
    fn default() -> Self {
        GpuSelector::Bool(true)
    }
}

impl GpuSelector {
    fn should_use_gpu(&self, has_gpu: bool, server_device: Option<&str>) -> Result<bool, String> {
        match self {
            GpuSelector::Bool(b) => Ok(*b && has_gpu),
            GpuSelector::String(s) => {
                let v = s.trim().to_ascii_lowercase();
                match v.as_str() {
                    "" => Err("gpu must be a boolean or a non-empty string".into()),
                    "true" | "auto" => Ok(has_gpu),
                    "false" | "off" | "cpu" | "none" => Ok(false),
                    "cuda" | "metal" => {
                        if !has_gpu {
                            return Err(format!(
                                "requested gpu={v} but server has no GPU configured (start with --gpu cuda|metal)"
                            ));
                        }
                        if server_device != Some(v.as_str()) {
                            return Err(format!(
                                "requested gpu={v} but server device is {}",
                                server_device.unwrap_or("cpu")
                            ));
                        }
                        Ok(true)
                    }
                    _ => Err(format!(
                        "invalid gpu value: {s:?}. Use true/false, 'auto', 'cpu', 'cuda', or 'metal'."
                    )),
                }
            }
        }
    }
}

/// Request body for `/v1/fit`.
#[derive(Debug, Deserialize)]
struct FitRequest {
    /// pyhf or HS3 workspace JSON (the full object, not a string).
    /// Can be omitted if `model_id` is provided.
    workspace: Option<serde_json::Value>,

    /// Cached model ID (SHA-256 hash). If provided, skips workspace parsing.
    model_id: Option<String>,

    /// Use GPU if available on this server (default: true).
    ///
    /// Back-compat: accepts a boolean. For convenience, also accepts a string:
    /// - "auto"/true: use GPU if the server was started with `--gpu ...` (else CPU)
    /// - "cpu"/false: force CPU
    /// - "cuda"/"metal": require that specific server GPU device (case-insensitive)
    #[serde(default)]
    gpu: GpuSelector,
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

    let use_gpu = req
        .gpu
        .should_use_gpu(state.has_gpu(), state.gpu_device.as_deref())
        .map_err(AppError::bad_request)?;
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
        let _gpu_guard = gpu_lock.as_ref().map(|st| st.gpu_lock.blocking_lock());

        let (fit_result, device) = match gpu_device.as_deref() {
            #[cfg(feature = "cuda")]
            Some("cuda") => {
                let r = mle
                    .fit_gpu(model.as_ref())
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
                    .fit_metal(model.as_ref())
                    .map_err(|e| AppError::internal(format!("Metal fit failed: {e}")))?;
                (r, "metal".to_string())
            }
            #[cfg(not(feature = "metal"))]
            Some("metal") => {
                return Err(AppError::internal("Metal not compiled".into()));
            }
            _ => {
                let r = mle
                    .fit(model.as_ref())
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
// POST /v1/unbinned/fit
// ---------------------------------------------------------------------------

/// Request body for `/v1/unbinned/fit`.
#[derive(Debug, Deserialize)]
struct UnbinnedFitRequest {
    /// Unbinned spec JSON (nextstat_unbinned_spec_v0 schema).
    spec: serde_json::Value,

    /// Server-side directory containing data files referenced by the spec.
    /// Relative paths in `spec.channels[].data.file` are resolved against this.
    /// Defaults to "." (server working directory).
    #[serde(default = "default_data_root")]
    data_root: String,
}

fn default_data_root() -> String {
    ".".to_string()
}

/// Response body for `/v1/unbinned/fit`.
#[derive(Debug, Serialize)]
struct UnbinnedFitResponse {
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

async fn unbinned_fit_handler(
    State(state): State<SharedState>,
    Json(req): Json<UnbinnedFitRequest>,
) -> Result<Json<UnbinnedFitResponse>, AppError> {
    state.inflight.fetch_add(1, Ordering::Relaxed);
    let _dec = DecrementOnDrop(&state.inflight);
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    let pool_ref = Arc::clone(&state);

    let result = tokio::task::spawn_blocking(move || {
        let t0 = Instant::now();

        let _compute_guard = pool_ref.compute_lock.blocking_lock();

        // Parse the unbinned spec from the request JSON.
        let spec: ns_unbinned::spec::UnbinnedSpecV0 = serde_json::from_value(req.spec)
            .map_err(|e| AppError::bad_request(format!("invalid unbinned spec JSON: {e}")))?;

        if spec.schema_version != ns_unbinned::spec::UNBINNED_SPEC_V0 {
            return Err(AppError::bad_request(format!(
                "unsupported schema_version: {} (expected {})",
                spec.schema_version,
                ns_unbinned::spec::UNBINNED_SPEC_V0
            )));
        }

        // Validate data_root: reject path traversal via '..' components.
        let data_root = std::path::Path::new(&req.data_root);
        for component in data_root.components() {
            if matches!(component, std::path::Component::ParentDir) {
                return Err(AppError::bad_request(
                    "data_root must not contain '..' components (path traversal rejected)"
                        .to_string(),
                ));
            }
        }
        // compile_unbinned_model uses spec_path.parent() as the base dir;
        // we pass data_root/spec.json so that parent() == data_root.
        let synthetic_spec_path = data_root.join("spec.json");

        let model = ns_unbinned::spec::compile_unbinned_model(&spec, &synthetic_spec_path)
            .map_err(|e| AppError::bad_request(format!("unbinned model compilation error: {e}")))?;

        let mle = ns_inference::mle::MaximumLikelihoodEstimator::new();
        let fit_result = mle
            .fit(&model)
            .map_err(|e| AppError::internal(format!("unbinned CPU fit failed: {e}")))?;

        let wall_time_s = t0.elapsed().as_secs_f64();

        let parameter_names: Vec<String> =
            model.parameters().iter().map(|p| p.name.clone()).collect();
        let poi_index = model.poi_index();

        Ok(UnbinnedFitResponse {
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
            device: "cpu".to_string(),
            wall_time_s,
        })
    })
    .await
    .map_err(|e| AppError::internal(format!("task panicked: {e}")))?;

    result.map(Json)
}

// ---------------------------------------------------------------------------
// POST /v1/nlme/fit
// ---------------------------------------------------------------------------

/// Request body for `/v1/nlme/fit`.
#[derive(Debug, Deserialize)]
struct NlmeFitRequest {
    /// Model type: "pk_1cpt" (individual) or "nlme_1cpt" (population NLME).
    model_type: String,

    /// Observation times (must be >= 0).
    times: Vec<f64>,

    /// Observed concentrations (must be >= 0).
    observations: Vec<f64>,

    /// Dose amount (> 0).
    dose: f64,

    /// Bioavailability (> 0, default: 1.0).
    #[serde(default = "default_bioavailability")]
    bioavailability: f64,

    /// Observation noise standard deviation (> 0).
    sigma: f64,

    /// Subject indices (required for model_type "nlme_1cpt").
    /// Each entry maps the corresponding observation to a subject [0, n_subjects).
    subject_idx: Option<Vec<usize>>,

    /// Number of subjects (required for model_type "nlme_1cpt").
    n_subjects: Option<usize>,

    /// Lower limit of quantification (optional).
    lloq: Option<f64>,

    /// LLOQ handling policy: "ignore", "replace_half", or "censored" (default: "censored").
    #[serde(default = "default_lloq_policy")]
    lloq_policy: String,
}

fn default_bioavailability() -> f64 {
    1.0
}

fn default_lloq_policy() -> String {
    "censored".to_string()
}

/// Response body for `/v1/nlme/fit`.
#[derive(Debug, Serialize)]
struct NlmeFitResponse {
    model_type: String,
    parameter_names: Vec<String>,
    bestfit: Vec<f64>,
    uncertainties: Vec<f64>,
    nll: f64,
    twice_nll: f64,
    converged: bool,
    n_iter: usize,
    n_fev: usize,
    n_gev: usize,
    covariance: Option<Vec<f64>>,
    wall_time_s: f64,
}

fn parse_lloq_policy(s: &str) -> Result<ns_inference::pk::LloqPolicy, AppError> {
    match s {
        "ignore" => Ok(ns_inference::pk::LloqPolicy::Ignore),
        "replace_half" => Ok(ns_inference::pk::LloqPolicy::ReplaceHalf),
        "censored" => Ok(ns_inference::pk::LloqPolicy::Censored),
        other => Err(AppError::bad_request(format!(
            "unknown lloq_policy '{other}'; expected 'ignore', 'replace_half', or 'censored'"
        ))),
    }
}

async fn nlme_fit_handler(
    State(state): State<SharedState>,
    Json(req): Json<NlmeFitRequest>,
) -> Result<Json<NlmeFitResponse>, AppError> {
    state.inflight.fetch_add(1, Ordering::Relaxed);
    let _dec = DecrementOnDrop(&state.inflight);
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    let pool_ref = Arc::clone(&state);

    let result = tokio::task::spawn_blocking(move || {
        let t0 = Instant::now();

        let _compute_guard = pool_ref.compute_lock.blocking_lock();

        let lloq_policy = parse_lloq_policy(&req.lloq_policy)?;
        let mle = ns_inference::mle::MaximumLikelihoodEstimator::new();

        let (fit_result, param_names, model_type) = match req.model_type.as_str() {
            "pk_1cpt" => {
                let model = ns_inference::pk::OneCompartmentOralPkModel::new(
                    req.times,
                    req.observations,
                    req.dose,
                    req.bioavailability,
                    req.sigma,
                    req.lloq,
                    lloq_policy,
                )
                .map_err(|e| AppError::bad_request(format!("PK model error: {e}")))?;

                let names = model.parameter_names();
                let fit = mle
                    .fit(&model)
                    .map_err(|e| AppError::internal(format!("PK fit failed: {e}")))?;
                (fit, names, "pk_1cpt".to_string())
            }
            "nlme_1cpt" => {
                let subject_idx = req.subject_idx.ok_or_else(|| {
                    AppError::bad_request("subject_idx is required for nlme_1cpt".into())
                })?;
                let n_subjects = req.n_subjects.ok_or_else(|| {
                    AppError::bad_request("n_subjects is required for nlme_1cpt".into())
                })?;

                let model = ns_inference::pk::OneCompartmentOralPkNlmeModel::new(
                    req.times,
                    req.observations,
                    subject_idx,
                    n_subjects,
                    req.dose,
                    req.bioavailability,
                    req.sigma,
                    req.lloq,
                    lloq_policy,
                )
                .map_err(|e| AppError::bad_request(format!("NLME model error: {e}")))?;

                let names = model.parameter_names();
                let fit = mle
                    .fit(&model)
                    .map_err(|e| AppError::internal(format!("NLME fit failed: {e}")))?;
                (fit, names, "nlme_1cpt".to_string())
            }
            other => {
                return Err(AppError::bad_request(format!(
                    "unknown model_type '{other}'; expected 'pk_1cpt' or 'nlme_1cpt'"
                )));
            }
        };

        let wall_time_s = t0.elapsed().as_secs_f64();

        Ok(NlmeFitResponse {
            model_type,
            parameter_names: param_names,
            bestfit: fit_result.parameters,
            uncertainties: fit_result.uncertainties,
            nll: fit_result.nll,
            twice_nll: 2.0 * fit_result.nll,
            converged: fit_result.converged,
            n_iter: fit_result.n_iter,
            n_fev: fit_result.n_fev,
            n_gev: fit_result.n_gev,
            covariance: fit_result.covariance,
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
    #[serde(default)]
    gpu: GpuSelector,
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

    let use_gpu = req
        .gpu
        .should_use_gpu(state.has_gpu(), state.gpu_device.as_deref())
        .map_err(AppError::bad_request)?;
    let gpu_device = if use_gpu { state.gpu_device.clone() } else { None };

    let gpu_lock = if use_gpu { Some(Arc::clone(&state)) } else { None };

    let pool_ref = Arc::clone(&state);

    let result = tokio::task::spawn_blocking(move || {
        let t0 = Instant::now();

        let _compute_guard = pool_ref.compute_lock.blocking_lock();

        let model = resolve_model(&pool_ref, req.workspace.as_ref(), req.model_id.as_deref())?;
        let mle = ns_inference::mle::MaximumLikelihoodEstimator::new();

        let _gpu_guard = gpu_lock.as_ref().map(|st| st.gpu_lock.blocking_lock());

        let (ranking, device) = match gpu_device.as_deref() {
            #[cfg(feature = "cuda")]
            Some("cuda") => {
                let r = ns_inference::mle::ranking_gpu(&mle, model.as_ref())
                    .map_err(|e| AppError::internal(format!("CUDA ranking failed: {e}")))?;
                (r, "cuda".to_string())
            }
            #[cfg(not(feature = "cuda"))]
            Some("cuda") => {
                return Err(AppError::internal("CUDA not compiled".into()));
            }
            #[cfg(feature = "metal")]
            Some("metal") => {
                let r = ns_inference::mle::ranking_metal(&mle, model.as_ref())
                    .map_err(|e| AppError::internal(format!("Metal ranking failed: {e}")))?;
                (r, "metal".to_string())
            }
            #[cfg(not(feature = "metal"))]
            Some("metal") => {
                return Err(AppError::internal("Metal not compiled".into()));
            }
            _ => {
                let r = mle
                    .ranking(model.as_ref())
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
    #[serde(default)]
    gpu: GpuSelector,
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

    let use_gpu = req
        .gpu
        .should_use_gpu(state.has_gpu(), state.gpu_device.as_deref())
        .map_err(AppError::bad_request)?;
    let gpu_device = if use_gpu { state.gpu_device.clone() } else { None };
    let gpu_lock = if use_gpu { Some(Arc::clone(&state)) } else { None };

    let st = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || {
        let t0 = Instant::now();

        let _compute_guard = st.compute_lock.blocking_lock();

        let _gpu_guard = gpu_lock.as_ref().map(|st| st.gpu_lock.blocking_lock());

        let device = gpu_device.as_deref().unwrap_or("cpu").to_string();
        let mle = ns_inference::mle::MaximumLikelihoodEstimator::new();

        let items: Vec<BatchFitItem> = if gpu_device.is_some() {
            // GPU: sequential (GPU lock held, single device)
            req.workspaces
                .iter()
                .enumerate()
                .map(|(i, ws)| match fit_one_workspace(ws, &mle, gpu_device.as_deref()) {
                    Ok(resp) => BatchFitItem { index: i, result: Some(resp), error: None },
                    Err(e) => BatchFitItem { index: i, result: None, error: Some(e.message) },
                })
                .collect()
        } else {
            // CPU: parallel via Rayon
            req.workspaces
                .par_iter()
                .enumerate()
                .map(|(i, ws)| match fit_one_workspace(ws, &mle, None) {
                    Ok(resp) => BatchFitItem { index: i, result: Some(resp), error: None },
                    Err(e) => BatchFitItem { index: i, result: None, error: Some(e.message) },
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
            let r =
                mle.fit(&model).map_err(|e| AppError::internal(format!("CPU fit failed: {e}")))?;
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
    #[serde(default)]
    gpu: GpuSelector,
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
        return Err(AppError::bad_request("n_toys must be between 1 and 100000".into()));
    }

    state.inflight.fetch_add(1, Ordering::Relaxed);
    let _dec = DecrementOnDrop(&state.inflight);
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    let use_gpu = req
        .gpu
        .should_use_gpu(state.has_gpu(), state.gpu_device.as_deref())
        .map_err(AppError::bad_request)?;
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

        let _gpu_guard = gpu_lock.as_ref().map(|st| st.gpu_lock.blocking_lock());

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
// Async Jobs API
// ---------------------------------------------------------------------------

async fn job_submit_handler(
    State(state): State<SharedState>,
    Json(req): Json<crate::jobs::JobSubmitRequest>,
) -> Result<Json<crate::jobs::JobSubmitResponse>, AppError> {
    let job_id = state.job_store.create(&req.task_type).await.map_err(AppError::internal)?;

    match req.task_type.as_str() {
        "batch_toys" => {
            let toys_req: BatchToysRequest = serde_json::from_value(req.payload)
                .map_err(|e| AppError::bad_request(format!("invalid batch_toys payload: {e}")))?;

            let st = Arc::clone(&state);
            let jid = job_id.clone();

            tokio::spawn(async move {
                st.job_store.set_running(&jid).await;

                let cancel_token = st.job_store.cancel_token(&jid).await;
                let st2 = Arc::clone(&st);

                let result = tokio::task::spawn_blocking(move || {
                    let t0 = Instant::now();
                    let _compute_guard = st2.compute_lock.blocking_lock();

                    let json_str = serde_json::to_string(&toys_req.workspace)
                        .map_err(|e| format!("invalid workspace JSON: {e}"))?;
                    let model = load_model(&json_str).map_err(|e| e.message)?;

                    let params = toys_req.params.unwrap_or_else(|| model.parameter_init());
                    if params.len() != model.n_params() {
                        return Err(format!(
                            "params length {} != model parameters {}",
                            params.len(),
                            model.n_params()
                        ));
                    }

                    let use_gpu =
                        toys_req.gpu.should_use_gpu(st2.has_gpu(), st2.gpu_device.as_deref())?;
                    let gpu_device = if use_gpu { st2.gpu_device.clone() } else { None };

                    let _gpu_guard =
                        if use_gpu { Some(st2.gpu_lock.blocking_lock()) } else { None };

                    // Check cancellation before starting compute.
                    if let Some(ref token) = cancel_token
                        && token.load(std::sync::atomic::Ordering::Relaxed)
                    {
                        return Err("job cancelled".to_string());
                    }

                    let (fit_results, device) = match gpu_device.as_deref() {
                        #[cfg(feature = "cuda")]
                        Some("cuda") => {
                            let r = ns_inference::gpu_batch::fit_toys_batch_gpu(
                                &model,
                                &params,
                                toys_req.n_toys,
                                toys_req.seed,
                                None,
                            )
                            .map_err(|e| format!("CUDA batch toys failed: {e}"))?;
                            (r, "cuda")
                        }
                        #[cfg(not(feature = "cuda"))]
                        Some("cuda") => {
                            return Err("CUDA not compiled".to_string());
                        }
                        #[cfg(feature = "metal")]
                        Some("metal") => {
                            let r = ns_inference::metal_batch::fit_toys_batch_metal(
                                &model,
                                &params,
                                toys_req.n_toys,
                                toys_req.seed,
                                None,
                            )
                            .map_err(|e| format!("Metal batch toys failed: {e}"))?;
                            (r, "metal")
                        }
                        #[cfg(not(feature = "metal"))]
                        Some("metal") => {
                            return Err("Metal not compiled".to_string());
                        }
                        _ => {
                            let r = ns_inference::batch::fit_toys_batch(
                                &model,
                                &params,
                                toys_req.n_toys,
                                toys_req.seed,
                                None,
                            );
                            (r, "cpu")
                        }
                    };

                    let wall_time_s = t0.elapsed().as_secs_f64();
                    let mut n_converged = 0usize;
                    let mut n_failed = 0usize;
                    let results: Vec<serde_json::Value> = fit_results
                        .into_iter()
                        .filter_map(|r| match r {
                            Ok(fr) => {
                                if fr.converged {
                                    n_converged += 1;
                                }
                                Some(serde_json::json!({
                                    "bestfit": fr.parameters,
                                    "nll": fr.nll,
                                    "converged": fr.converged,
                                    "n_iter": fr.n_iter,
                                }))
                            }
                            Err(_) => {
                                n_failed += 1;
                                None
                            }
                        })
                        .collect();

                    Ok(serde_json::json!({
                        "n_toys": results.len(),
                        "n_converged": n_converged,
                        "n_failed": n_failed,
                        "results": results,
                        "device": device,
                        "wall_time_s": wall_time_s,
                    }))
                })
                .await;

                match result {
                    Ok(Ok(val)) => st.job_store.set_completed(&jid, val).await,
                    Ok(Err(e)) => st.job_store.set_failed(&jid, e).await,
                    Err(e) => st.job_store.set_failed(&jid, format!("task panicked: {e}")).await,
                }
            });
        }
        other => {
            return Err(AppError::bad_request(format!(
                "unknown task_type '{other}'; supported: 'batch_toys'"
            )));
        }
    }

    Ok(Json(crate::jobs::JobSubmitResponse { job_id, status: crate::jobs::JobStatus::Pending }))
}

async fn job_status_handler(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<crate::jobs::JobStatusResponse>, AppError> {
    state
        .job_store
        .get(&id)
        .await
        .map(Json)
        .ok_or_else(|| AppError::not_found(format!("job {id} not found")))
}

async fn job_cancel_handler(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    state.job_store.cancel(&id).await.map_err(AppError::bad_request)?;
    Ok(Json(serde_json::json!({"cancelled": true, "job_id": id})))
}

async fn job_list_handler(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let jobs = state.job_store.list().await;
    Json(serde_json::json!({"jobs": jobs}))
}

// ---------------------------------------------------------------------------
// GET /v1/openapi.json
// ---------------------------------------------------------------------------

async fn openapi_handler() -> Json<serde_json::Value> {
    Json(crate::openapi::openapi_spec())
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
        (None, None) => {
            Err(AppError::bad_request("either 'workspace' or 'model_id' must be provided".into()))
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    /// Build a test router with no auth, no rate limiting.
    fn test_app() -> Router<()> {
        let state = Arc::new(crate::state::AppState::new(None));
        let api_keys = crate::auth::ApiKeys(None);
        let rate_limiter = crate::rate_limit::RateLimiter::disabled();

        Router::new()
            .merge(super::router())
            .layer(axum::middleware::from_fn(crate::rate_limit::rate_limit_middleware))
            .layer(axum::Extension(rate_limiter))
            .layer(axum::middleware::from_fn(crate::auth::auth_middleware))
            .layer(axum::Extension(api_keys))
            .with_state(state)
    }

    /// Build a test router with auth enabled.
    fn test_app_with_auth(keys: std::collections::HashSet<String>) -> Router<()> {
        let state = Arc::new(crate::state::AppState::new(None));
        let api_keys = crate::auth::ApiKeys(Some(Arc::new(keys)));
        let rate_limiter = crate::rate_limit::RateLimiter::disabled();

        Router::new()
            .merge(super::router())
            .layer(axum::middleware::from_fn(crate::rate_limit::rate_limit_middleware))
            .layer(axum::Extension(rate_limiter))
            .layer(axum::middleware::from_fn(crate::auth::auth_middleware))
            .layer(axum::Extension(api_keys))
            .with_state(state)
    }

    /// Build a test router with rate limiting enabled.
    fn test_app_with_rate_limit(rps: u32) -> Router<()> {
        let state = Arc::new(crate::state::AppState::new(None));
        let api_keys = crate::auth::ApiKeys(None);
        let rate_limiter = crate::rate_limit::RateLimiter::new(rps);

        Router::new()
            .merge(super::router())
            .layer(axum::middleware::from_fn(crate::rate_limit::rate_limit_middleware))
            .layer(axum::Extension(rate_limiter))
            .layer(axum::middleware::from_fn(crate::auth::auth_middleware))
            .layer(axum::Extension(api_keys))
            .with_state(state)
    }

    async fn post_json(
        app: Router<()>,
        uri: &str,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        let req = Request::builder()
            .method(Method::POST)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let val: serde_json::Value =
            serde_json::from_slice(&bytes).unwrap_or(serde_json::json!({}));
        (status, val)
    }

    async fn get_json(app: Router<()>, uri: &str) -> (StatusCode, serde_json::Value) {
        let req = Request::builder().method(Method::GET).uri(uri).body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let val: serde_json::Value =
            serde_json::from_slice(&bytes).unwrap_or(serde_json::json!({}));
        (status, val)
    }

    async fn get_json_with_bearer(
        app: Router<()>,
        uri: &str,
        token: &str,
    ) -> (StatusCode, serde_json::Value) {
        let req = Request::builder()
            .method(Method::GET)
            .uri(uri)
            .header("authorization", format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let val: serde_json::Value =
            serde_json::from_slice(&bytes).unwrap_or(serde_json::json!({}));
        (status, val)
    }

    // --- GpuSelector (existing) ---

    #[test]
    fn gpu_selector_accepts_bool_and_string() {
        let req: FitRequest = serde_json::from_value(serde_json::json!({"workspace": {}})).unwrap();
        match req.gpu {
            GpuSelector::Bool(true) => {}
            other => panic!("expected default gpu=true, got: {other:?}"),
        }
        assert!(!req.gpu.should_use_gpu(false, None).unwrap());

        let req: FitRequest =
            serde_json::from_value(serde_json::json!({"workspace": {}, "gpu": "Metal"})).unwrap();
        assert!(req.gpu.should_use_gpu(true, Some("metal")).unwrap());
        assert!(req.gpu.should_use_gpu(true, Some("cuda")).is_err());

        let req: FitRequest =
            serde_json::from_value(serde_json::json!({"workspace": {}, "gpu": "cpu"})).unwrap();
        assert!(!req.gpu.should_use_gpu(true, Some("metal")).unwrap());
    }

    // --- Health ---

    #[tokio::test]
    async fn health_returns_ok() {
        let (status, body) = get_json(test_app(), "/v1/health").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "ok");
        assert!(body["version"].is_string());
        assert_eq!(body["device"], "cpu");
    }

    // --- OpenAPI ---

    #[tokio::test]
    async fn openapi_returns_valid_spec() {
        let (status, body) = get_json(test_app(), "/v1/openapi.json").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["openapi"], "3.1.0");
        assert!(body["info"]["title"].is_string());
        assert!(body["paths"]["/v1/fit"].is_object());
        assert!(body["paths"]["/v1/unbinned/fit"].is_object());
        assert!(body["paths"]["/v1/nlme/fit"].is_object());
        assert!(body["paths"]["/v1/jobs/submit"].is_object());
    }

    // --- NLME / PK ---

    #[tokio::test]
    async fn nlme_pk_1cpt_smoke() {
        let (status, body) = post_json(
            test_app(),
            "/v1/nlme/fit",
            serde_json::json!({
                "model_type": "pk_1cpt",
                "times": [0.5, 1.0, 2.0, 4.0, 8.0],
                "observations": [2.0, 3.5, 3.0, 1.5, 0.3],
                "dose": 100.0,
                "sigma": 0.5
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "body: {body}");
        assert_eq!(body["model_type"], "pk_1cpt");
        assert!(body["converged"].is_boolean());
        assert!(body["nll"].is_number());
        assert_eq!(body["parameter_names"].as_array().unwrap().len(), 3);
        assert!(body["wall_time_s"].as_f64().unwrap() > 0.0);
    }

    #[tokio::test]
    async fn nlme_nlme_1cpt_smoke() {
        let (status, body) = post_json(
            test_app(),
            "/v1/nlme/fit",
            serde_json::json!({
                "model_type": "nlme_1cpt",
                "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
                "observations": [2.0, 3.5, 3.0, 1.5, 1.8, 3.2, 2.8, 1.3],
                "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
                "n_subjects": 2,
                "dose": 100.0,
                "sigma": 0.5
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "body: {body}");
        assert_eq!(body["model_type"], "nlme_1cpt");
        // 6 pop params + 3*2 etas = 12
        assert_eq!(body["parameter_names"].as_array().unwrap().len(), 12);
        assert!(body["nll"].is_number());
    }

    #[tokio::test]
    async fn nlme_bad_model_type_returns_400() {
        let (status, body) = post_json(
            test_app(),
            "/v1/nlme/fit",
            serde_json::json!({
                "model_type": "nonexistent",
                "times": [1.0],
                "observations": [1.0],
                "dose": 100.0,
                "sigma": 0.5
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"].as_str().unwrap().contains("unknown model_type"));
    }

    #[tokio::test]
    async fn nlme_missing_subject_idx_returns_400() {
        let (status, body) = post_json(
            test_app(),
            "/v1/nlme/fit",
            serde_json::json!({
                "model_type": "nlme_1cpt",
                "times": [1.0],
                "observations": [1.0],
                "dose": 100.0,
                "sigma": 0.5
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"].as_str().unwrap().contains("subject_idx"));
    }

    // --- Unbinned fit ---

    #[tokio::test]
    async fn unbinned_fit_bad_spec_returns_400() {
        let (status, body) = post_json(
            test_app(),
            "/v1/unbinned/fit",
            serde_json::json!({ "spec": { "not": "valid" } }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"].as_str().unwrap().contains("invalid unbinned spec"));
    }

    #[tokio::test]
    async fn unbinned_fit_wrong_schema_version_returns_400() {
        let (status, body) = post_json(
            test_app(),
            "/v1/unbinned/fit",
            serde_json::json!({
                "spec": {
                    "schema_version": "wrong_version",
                    "model": { "parameters": [] },
                    "channels": []
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"].as_str().unwrap().contains("unsupported schema_version"));
    }

    // --- Async Jobs ---

    #[tokio::test]
    async fn jobs_bad_task_type_returns_400() {
        let (status, body) = post_json(
            test_app(),
            "/v1/jobs/submit",
            serde_json::json!({ "task_type": "nonexistent", "payload": {} }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"].as_str().unwrap().contains("unknown task_type"));
    }

    #[tokio::test]
    async fn jobs_status_not_found_returns_404() {
        let (status, body) = get_json(test_app(), "/v1/jobs/nonexistent-id").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body["error"].as_str().unwrap().contains("not found"));
    }

    #[tokio::test]
    async fn jobs_list_returns_empty() {
        let (status, body) = get_json(test_app(), "/v1/jobs").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["jobs"].as_array().unwrap().is_empty());
    }

    // --- Auth integration ---

    #[tokio::test]
    async fn auth_health_exempt() {
        let keys: std::collections::HashSet<String> = ["secret-key".to_string()].into();
        let (status, body) = get_json(test_app_with_auth(keys), "/v1/health").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "ok");
    }

    #[tokio::test]
    async fn auth_rejects_missing_token() {
        let keys: std::collections::HashSet<String> = ["secret-key".to_string()].into();
        let (status, body) = get_json(test_app_with_auth(keys), "/v1/openapi.json").await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert!(body["error"].as_str().unwrap().contains("missing"));
    }

    #[tokio::test]
    async fn auth_rejects_bad_token() {
        let keys: std::collections::HashSet<String> = ["secret-key".to_string()].into();
        let (status, body) =
            get_json_with_bearer(test_app_with_auth(keys), "/v1/openapi.json", "wrong-key").await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert!(body["error"].as_str().unwrap().contains("invalid"));
    }

    #[tokio::test]
    async fn auth_accepts_valid_token() {
        let keys: std::collections::HashSet<String> = ["secret-key".to_string()].into();
        let (status, body) =
            get_json_with_bearer(test_app_with_auth(keys), "/v1/openapi.json", "secret-key").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["openapi"], "3.1.0");
    }

    // --- Rate limiting integration ---

    #[tokio::test]
    async fn rate_limit_blocks_after_budget() {
        let app = test_app_with_rate_limit(2);
        let (s1, _) = get_json(app.clone(), "/v1/openapi.json").await;
        assert_eq!(s1, StatusCode::OK);
        let (s2, _) = get_json(app.clone(), "/v1/openapi.json").await;
        assert_eq!(s2, StatusCode::OK);
        let (s3, body) = get_json(app, "/v1/openapi.json").await;
        assert_eq!(s3, StatusCode::TOO_MANY_REQUESTS);
        assert!(body["error"].as_str().unwrap().contains("rate limit"));
    }

    #[tokio::test]
    async fn rate_limit_health_exempt() {
        let app = test_app_with_rate_limit(1);
        // Use up the budget
        let (s1, _) = get_json(app.clone(), "/v1/openapi.json").await;
        assert_eq!(s1, StatusCode::OK);
        // Health should still work
        let (s2, body) = get_json(app, "/v1/health").await;
        assert_eq!(s2, StatusCode::OK);
        assert_eq!(body["status"], "ok");
    }
}
