//! NextStat Server — self-hosted GPU inference API.
//!
//! Provides a JSON REST API for shared GPU statistical fits so that
//! lab clusters can share one GPU across multiple users/clients
//! without per-user CUDA/Metal setup.
//!
//! # Endpoints
//!
//! **Inference:**
//! - `POST /v1/fit`            — HistFactory MLE fit
//! - `POST /v1/ranking`        — systematic ranking
//! - `POST /v1/unbinned/fit`   — unbinned MLE fit (event-level)
//! - `POST /v1/nlme/fit`       — NLME / PK population fit
//!
//! **Batch:**
//! - `POST /v1/batch/fit`      — batch MLE fit (up to 100 workspaces)
//! - `POST /v1/batch/toys`     — batch toy fits
//!
//! **Async Jobs:**
//! - `POST   /v1/jobs/submit`  — submit long-running task
//! - `GET    /v1/jobs/{id}`    — poll job status
//! - `DELETE /v1/jobs/{id}`    — cancel a job
//! - `GET    /v1/jobs`         — list all jobs
//!
//! **Admin:**
//! - `GET  /v1/health`         — server status, version, GPU info
//! - `GET  /v1/openapi.json`   — OpenAPI 3.1 spec

mod auth;
mod jobs;
mod openapi;
mod pool;
mod rate_limit;
mod routes;
mod state;
mod tools;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware;
use clap::Parser;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

use auth::ApiKeys;
use rate_limit::RateLimiter;

use state::AppState;

/// NextStat inference server — shared GPU statistical fits over HTTP.
#[derive(Parser, Debug)]
#[command(name = "nextstat-server", version = ns_core::VERSION, about)]
struct Cli {
    /// Port to listen on.
    #[arg(short, long, default_value = "3742")]
    port: u16,

    /// Bind address.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// GPU device to use for inference. Values: "cuda", "metal", or omit for CPU-only.
    #[arg(long)]
    gpu: Option<String>,

    /// Maximum number of CPU threads for non-GPU workloads (0 = auto).
    #[arg(long, default_value = "0")]
    threads: usize,

    /// Maximum request body size in MiB (applies to all endpoints).
    ///
    /// Protects the server from accidental or malicious oversized JSON payloads.
    #[arg(long, default_value = "64")]
    max_body_mb: usize,

    /// Path to a file containing API keys (one per line).
    ///
    /// When set, all endpoints except GET /v1/health require
    /// `Authorization: Bearer <key>`.  Alternatively set the
    /// `NS_API_KEYS` environment variable (comma-separated).
    /// If neither is configured, auth is disabled (open mode).
    #[arg(long)]
    api_keys: Option<String>,

    /// Maximum requests per second per IP address (0 = unlimited).
    ///
    /// Simple token-bucket rate limiter. Health endpoint is always exempt.
    #[arg(long, default_value = "0")]
    rate_limit: u32,

    /// Allowed CORS origin(s), comma-separated.
    ///
    /// Examples: "https://app.example.com", "http://localhost:3000,https://app.example.com".
    /// If omitted, defaults to permissive ("*") for development.
    /// Set to restrict allowed origins in production.
    #[arg(long)]
    cors_origin: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,tower_http=debug")),
        )
        .init();

    let cli = Cli::parse();

    // Validate GPU selection at startup
    validate_gpu(cli.gpu.as_deref())?;

    // Configure Rayon thread pool
    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(cli.threads).build_global().ok();
    }

    // Load API keys (auth disabled if none configured)
    let api_keys = ApiKeys::load(cli.api_keys.as_deref()).map_err(|e| anyhow::anyhow!(e))?;

    let state = Arc::new(AppState::new(cli.gpu.clone()));

    let max_body_bytes = mb_to_bytes(cli.max_body_mb);

    let rate_limiter = RateLimiter::new(cli.rate_limit);
    if rate_limiter.is_enabled() {
        tracing::info!(rps = cli.rate_limit, "rate limiting enabled");
    }

    let app = Router::new()
        .merge(routes::router())
        .layer(middleware::from_fn(rate_limit::rate_limit_middleware))
        .layer(axum::Extension(rate_limiter))
        .layer(middleware::from_fn(auth::auth_middleware))
        .layer(axum::Extension(api_keys))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .layer(TraceLayer::new_for_http())
        .layer(match &cli.cors_origin {
            Some(origins) => {
                let parsed: Vec<_> = origins
                    .split(',')
                    .filter_map(|s| s.trim().parse::<axum::http::HeaderValue>().ok())
                    .collect();
                tracing::info!(origins = %origins, "CORS restricted");
                CorsLayer::new()
                    .allow_origin(parsed)
                    .allow_methods([
                        axum::http::Method::GET,
                        axum::http::Method::POST,
                        axum::http::Method::DELETE,
                        axum::http::Method::OPTIONS,
                    ])
                    .allow_headers([
                        axum::http::header::AUTHORIZATION,
                        axum::http::header::CONTENT_TYPE,
                    ])
                    .max_age(std::time::Duration::from_secs(3600))
            }
            None => {
                tracing::warn!("CORS permissive (use --cors-origin in production)");
                CorsLayer::permissive()
            }
        })
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", cli.host, cli.port).parse()?;
    tracing::info!(
        %addr,
        gpu = cli.gpu.as_deref().unwrap_or("cpu"),
        auth = if cli.api_keys.is_some() { "enabled" } else { "disabled" },
        version = ns_core::VERSION,
        "nextstat-server starting"
    );

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

fn mb_to_bytes(mb: usize) -> usize {
    // Clamp overflow to usize::MAX to avoid panics in debug builds.
    mb.saturating_mul(1024).saturating_mul(1024)
}

fn validate_gpu(gpu: Option<&str>) -> anyhow::Result<()> {
    match gpu {
        Some("cuda") => {
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("--gpu cuda requires building with --features cuda");
            #[cfg(feature = "cuda")]
            tracing::info!("GPU mode: CUDA (f64)");
        }
        Some("metal") => {
            #[cfg(not(feature = "metal"))]
            anyhow::bail!("--gpu metal requires building with --features metal");
            #[cfg(feature = "metal")]
            tracing::info!("GPU mode: Metal (f32)");
        }
        Some(other) => anyhow::bail!("unknown --gpu device: {other}. Use 'cuda' or 'metal'"),
        None => {
            tracing::info!("GPU mode: disabled (CPU only)");
        }
    }
    Ok(())
}
