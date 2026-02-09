//! Shared application state for the NextStat server.

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Mutex;

use crate::pool::ModelPool;

/// Shared state available to all request handlers.
pub struct AppState {
    /// GPU device name ("cuda", "metal") or `None` for CPU-only.
    pub gpu_device: Option<String>,

    /// Server start time (for uptime reporting).
    pub started_at: Instant,

    /// GPU serialisation lock. All GPU workloads acquire this mutex so that
    /// only one fit/ranking runs on the GPU at a time. CPU requests bypass it.
    pub gpu_lock: Mutex<()>,

    /// Compute serialisation lock. This is used to prevent races around
    /// process-wide global settings (e.g. `ns_compute::set_eval_mode`).
    ///
    /// This intentionally serializes inference endpoints so per-request
    /// determinism/eval-mode controls can be implemented safely.
    pub compute_lock: Mutex<()>,

    /// In-flight request counter (for /health).
    pub inflight: std::sync::atomic::AtomicU64,

    /// Total requests served (for /health).
    pub total_requests: std::sync::atomic::AtomicU64,

    /// LRU model cache.
    pub model_pool: ModelPool,
}

impl AppState {
    pub fn new(gpu_device: Option<String>) -> Self {
        Self {
            gpu_device,
            started_at: Instant::now(),
            gpu_lock: Mutex::new(()),
            compute_lock: Mutex::new(()),
            inflight: std::sync::atomic::AtomicU64::new(0),
            total_requests: std::sync::atomic::AtomicU64::new(0),
            model_pool: ModelPool::new(None),
        }
    }

    /// Whether GPU is available for this server instance.
    pub fn has_gpu(&self) -> bool {
        self.gpu_device.is_some()
    }

    /// Device string for logging / response metadata.
    pub fn device_str(&self) -> &str {
        self.gpu_device.as_deref().unwrap_or("cpu")
    }
}

/// Type alias used in axum handlers.
pub type SharedState = Arc<AppState>;
