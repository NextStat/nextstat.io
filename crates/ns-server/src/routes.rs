//! HTTP route handlers for the NextStat server.
//!
//! All endpoints live under `/v1/` and accept/return JSON.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::state::SharedState;

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

pub fn router() -> Router<SharedState> {
    Router::new()
        .route("/v1/fit", post(fit_handler))
        .route("/v1/ranking", post(ranking_handler))
        .route("/v1/health", get(health_handler))
}

// ---------------------------------------------------------------------------
// POST /v1/fit
// ---------------------------------------------------------------------------

/// Request body for `/v1/fit`.
#[derive(Debug, Deserialize)]
struct FitRequest {
    /// pyhf or HS3 workspace JSON (the full object, not a string).
    workspace: serde_json::Value,

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

    // Offload blocking compute to a Rayon/blocking thread
    let result = tokio::task::spawn_blocking(move || {
        let t0 = Instant::now();

        // Parse workspace
        let json_str = serde_json::to_string(&req.workspace)
            .map_err(|e| AppError::bad_request(format!("invalid workspace JSON: {e}")))?;

        let model = load_model(&json_str)?;

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
                let r = mle.fit_gpu(&model)
                    .map_err(|e| AppError::internal(format!("CUDA fit failed: {e}")))?;
                (r, "cuda".to_string())
            }
            #[cfg(not(feature = "cuda"))]
            Some("cuda") => {
                return Err(AppError::internal("CUDA not compiled".into()));
            }
            Some("metal") => {
                return Err(AppError::bad_request(
                    "Metal single-model fit not yet supported; use CPU or CUDA".into(),
                ));
            }
            _ => {
                let r = mle.fit(&model)
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
    /// pyhf or HS3 workspace JSON.
    workspace: serde_json::Value,

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

    let result = tokio::task::spawn_blocking(move || {
        let t0 = Instant::now();

        let json_str = serde_json::to_string(&req.workspace)
            .map_err(|e| AppError::bad_request(format!("invalid workspace JSON: {e}")))?;

        let model = load_model(&json_str)?;
        let mle = ns_inference::mle::MaximumLikelihoodEstimator::new();

        let _gpu_guard = if let Some(ref st) = gpu_lock {
            Some(st.gpu_lock.blocking_lock())
        } else {
            None
        };

        let (ranking, device) = match gpu_device.as_deref() {
            #[cfg(feature = "cuda")]
            Some("cuda") => {
                let r = ns_inference::mle::ranking_gpu(&mle, &model)
                    .map_err(|e| AppError::internal(format!("CUDA ranking failed: {e}")))?;
                (r, "cuda".to_string())
            }
            #[cfg(not(feature = "cuda"))]
            Some("cuda") => {
                return Err(AppError::internal("CUDA not compiled".into()));
            }
            _ => {
                let r = mle.ranking(&model)
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
// GET /v1/health
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    uptime_s: f64,
    device: String,
    inflight: u64,
    total_requests: u64,
}

async fn health_handler(State(state): State<SharedState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: ns_core::VERSION,
        uptime_s: state.started_at.elapsed().as_secs_f64(),
        device: state.device_str().to_string(),
        inflight: state.inflight.load(Ordering::Relaxed),
        total_requests: state.total_requests.load(Ordering::Relaxed),
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_true() -> bool {
    true
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
