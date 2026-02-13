//! Async job system for long-running tasks (toys, scans).
//!
//! Workflow:
//! 1. Client submits a job via `POST /v1/jobs/submit` → receives a `job_id`.
//! 2. Client polls `GET /v1/jobs/{id}` until status is `completed` or `failed`.
//! 3. Client can cancel with `DELETE /v1/jobs/{id}`.
//!
//! Jobs are stored in memory with a configurable maximum count.
//! Completed jobs are pruned after `JOB_TTL_SECS` seconds.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

/// Maximum number of concurrent + completed jobs in memory.
const MAX_JOBS: usize = 1000;

/// Time-to-live for completed/failed jobs (seconds).
const JOB_TTL_SECS: u64 = 3600;

/// Unique job identifier.
pub type JobId = String;

/// Job status.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Stored job metadata + result.
#[derive(Debug, Clone)]
pub struct Job {
    pub id: JobId,
    pub status: JobStatus,
    pub task_type: String,
    pub created_at: Instant,
    pub started_at: Option<Instant>,
    pub completed_at: Option<Instant>,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
    /// Cancellation token: when set to true, the running task should stop.
    pub cancel_requested: Arc<std::sync::atomic::AtomicBool>,
}

/// Response when querying a job.
#[derive(Debug, Serialize)]
pub struct JobStatusResponse {
    pub id: String,
    pub status: JobStatus,
    pub task_type: String,
    pub elapsed_s: f64,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
}

impl Job {
    fn to_response(&self) -> JobStatusResponse {
        let elapsed = if let Some(completed) = self.completed_at {
            completed.duration_since(self.created_at).as_secs_f64()
        } else {
            self.created_at.elapsed().as_secs_f64()
        };
        JobStatusResponse {
            id: self.id.clone(),
            status: self.status.clone(),
            task_type: self.task_type.clone(),
            elapsed_s: elapsed,
            result: self.result.clone(),
            error: self.error.clone(),
        }
    }
}

/// In-memory job store.
#[derive(Clone)]
pub struct JobStore {
    inner: Arc<Mutex<HashMap<JobId, Job>>>,
}

impl JobStore {
    pub fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(HashMap::new())) }
    }

    /// Create a new pending job and return its ID.
    pub async fn create(&self, task_type: &str) -> Result<JobId, String> {
        let mut store = self.inner.lock().await;

        // Prune expired completed/failed jobs.
        let now = Instant::now();
        let ttl = std::time::Duration::from_secs(JOB_TTL_SECS);
        store.retain(|_, j| {
            if matches!(j.status, JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled)
                && let Some(c) = j.completed_at
            {
                return now.duration_since(c) < ttl;
            }
            true
        });

        if store.len() >= MAX_JOBS {
            return Err("job store full; try again later".to_string());
        }

        let id = generate_job_id();
        let job = Job {
            id: id.clone(),
            status: JobStatus::Pending,
            task_type: task_type.to_string(),
            created_at: now,
            started_at: None,
            completed_at: None,
            result: None,
            error: None,
            cancel_requested: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };
        store.insert(id.clone(), job);
        Ok(id)
    }

    /// Mark a job as running.
    pub async fn set_running(&self, id: &str) {
        if let Some(job) = self.inner.lock().await.get_mut(id) {
            job.status = JobStatus::Running;
            job.started_at = Some(Instant::now());
        }
    }

    /// Mark a job as completed with a result.
    pub async fn set_completed(&self, id: &str, result: serde_json::Value) {
        if let Some(job) = self.inner.lock().await.get_mut(id) {
            job.status = JobStatus::Completed;
            job.completed_at = Some(Instant::now());
            job.result = Some(result);
        }
    }

    /// Mark a job as failed with an error message.
    pub async fn set_failed(&self, id: &str, error: String) {
        if let Some(job) = self.inner.lock().await.get_mut(id) {
            job.status = JobStatus::Failed;
            job.completed_at = Some(Instant::now());
            job.error = Some(error);
        }
    }

    /// Get job status response.
    pub async fn get(&self, id: &str) -> Option<JobStatusResponse> {
        self.inner.lock().await.get(id).map(|j| j.to_response())
    }

    /// Request cancellation of a job.
    pub async fn cancel(&self, id: &str) -> Result<(), String> {
        let mut store = self.inner.lock().await;
        match store.get_mut(id) {
            None => Err(format!("job {id} not found")),
            Some(job) => {
                if matches!(job.status, JobStatus::Completed | JobStatus::Failed) {
                    return Err(format!("job {id} already finished"));
                }
                job.cancel_requested.store(true, std::sync::atomic::Ordering::Relaxed);
                job.status = JobStatus::Cancelled;
                job.completed_at = Some(Instant::now());
                Ok(())
            }
        }
    }

    /// Get cancellation token for a job.
    pub async fn cancel_token(&self, id: &str) -> Option<Arc<std::sync::atomic::AtomicBool>> {
        self.inner.lock().await.get(id).map(|j| Arc::clone(&j.cancel_requested))
    }

    /// List all jobs (for admin/debug).
    pub async fn list(&self) -> Vec<JobStatusResponse> {
        self.inner.lock().await.values().map(|j| j.to_response()).collect()
    }
}

/// Request body for `POST /v1/jobs/submit`.
#[derive(Debug, Deserialize)]
pub struct JobSubmitRequest {
    /// Task type: "batch_toys" (more to come).
    pub task_type: String,

    /// Task-specific payload (same schema as the synchronous endpoint).
    pub payload: serde_json::Value,
}

/// Response body for `POST /v1/jobs/submit`.
#[derive(Debug, Serialize)]
pub struct JobSubmitResponse {
    pub job_id: String,
    pub status: JobStatus,
}

fn generate_job_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    // Simple unique ID: timestamp + random suffix.
    let rand_suffix: u32 = rand_u32();
    format!("job-{ts:x}-{rand_suffix:08x}")
}

fn rand_u32() -> u32 {
    // Use a simple xorshift from the timestamp to avoid pulling in `rand` crate.
    let mut x = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn job_lifecycle() {
        let store = JobStore::new();
        let id = store.create("batch_toys").await.unwrap();

        let status = store.get(&id).await.unwrap();
        assert_eq!(status.status, JobStatus::Pending);

        store.set_running(&id).await;
        let status = store.get(&id).await.unwrap();
        assert_eq!(status.status, JobStatus::Running);

        store.set_completed(&id, serde_json::json!({"n_toys": 100})).await;
        let status = store.get(&id).await.unwrap();
        assert_eq!(status.status, JobStatus::Completed);
        assert!(status.result.is_some());
    }

    #[tokio::test]
    async fn job_cancel() {
        let store = JobStore::new();
        let id = store.create("batch_toys").await.unwrap();
        store.set_running(&id).await;

        store.cancel(&id).await.unwrap();
        let status = store.get(&id).await.unwrap();
        assert_eq!(status.status, JobStatus::Cancelled);
    }

    #[tokio::test]
    async fn job_not_found() {
        let store = JobStore::new();
        assert!(store.get("nonexistent").await.is_none());
    }
}
