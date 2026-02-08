//! LRU model cache for the NextStat server.
//!
//! Models are keyed by SHA-256 hash of the raw workspace JSON.
//! This avoids re-parsing the same workspace on every request.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ns_translate::pyhf::HistFactoryModel;
use sha2::{Digest, Sha256};

/// Maximum number of models held in cache (eviction: oldest by last_used).
const DEFAULT_MAX_MODELS: usize = 64;

/// A cached model entry.
struct CachedModel {
    model: Arc<HistFactoryModel>,
    json_hash: String,
    name: String,
    n_params: usize,
    n_channels: usize,
    created_at: Instant,
    last_used: Instant,
    hit_count: u64,
}

/// Thread-safe LRU model pool.
pub struct ModelPool {
    inner: Mutex<PoolInner>,
}

struct PoolInner {
    models: HashMap<String, CachedModel>,
    max_models: usize,
}

/// Summary of a cached model (for list endpoint).
#[derive(Debug, serde::Serialize)]
pub struct ModelInfo {
    pub model_id: String,
    pub name: String,
    pub n_params: usize,
    pub n_channels: usize,
    pub age_s: f64,
    pub last_used_s: f64,
    pub hit_count: u64,
}

impl ModelPool {
    pub fn new(max_models: Option<usize>) -> Self {
        Self {
            inner: Mutex::new(PoolInner {
                models: HashMap::new(),
                max_models: max_models.unwrap_or(DEFAULT_MAX_MODELS),
            }),
        }
    }

    /// Compute the SHA-256 hash of a workspace JSON string.
    pub fn hash_workspace(json_str: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(json_str.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Insert a model into the pool. Returns the model_id (hash).
    /// If the model already exists, just updates last_used and returns the id.
    pub fn insert(
        &self,
        json_str: &str,
        model: HistFactoryModel,
        name: Option<String>,
    ) -> String {
        let hash = Self::hash_workspace(json_str);
        let mut pool = self.inner.lock().expect("model pool mutex poisoned");

        if let Some(entry) = pool.models.get_mut(&hash) {
            entry.last_used = Instant::now();
            entry.hit_count += 1;
            return hash;
        }

        // Evict oldest if at capacity
        if pool.models.len() >= pool.max_models {
            if let Some(oldest_key) = pool
                .models
                .iter()
                .min_by_key(|(_, v)| v.last_used)
                .map(|(k, _)| k.clone())
            {
                pool.models.remove(&oldest_key);
                tracing::info!(evicted = %oldest_key, "model pool eviction");
            }
        }

        let n_params = model.parameters().len();
        let n_channels = model.n_channels();
        let display_name = name.unwrap_or_else(|| format!("model-{}", &hash[..8]));
        let now = Instant::now();

        pool.models.insert(
            hash.clone(),
            CachedModel {
                model: Arc::new(model),
                json_hash: hash.clone(),
                name: display_name,
                n_params,
                n_channels,
                created_at: now,
                last_used: now,
                hit_count: 0,
            },
        );

        tracing::info!(model_id = %hash, n_params, n_channels, "model cached");
        hash
    }

    /// Look up a model by id, updating last_used on hit.
    /// Returns an `Arc` reference (cheap clone — no model data copied).
    pub fn get(&self, model_id: &str) -> Option<Arc<HistFactoryModel>> {
        let mut pool = self.inner.lock().expect("model pool mutex poisoned");
        if let Some(entry) = pool.models.get_mut(model_id) {
            entry.last_used = Instant::now();
            entry.hit_count += 1;
            Some(Arc::clone(&entry.model))
        } else {
            None
        }
    }

    /// Remove a model from the pool. Returns true if it existed.
    pub fn remove(&self, model_id: &str) -> bool {
        let mut pool = self.inner.lock().expect("model pool mutex poisoned");
        pool.models.remove(model_id).is_some()
    }

    /// List all cached models.
    pub fn list(&self) -> Vec<ModelInfo> {
        let pool = self.inner.lock().expect("model pool mutex poisoned");
        let now = Instant::now();
        pool.models
            .values()
            .map(|entry| ModelInfo {
                model_id: entry.json_hash.clone(),
                name: entry.name.clone(),
                n_params: entry.n_params,
                n_channels: entry.n_channels,
                age_s: now.duration_since(entry.created_at).as_secs_f64(),
                last_used_s: now.duration_since(entry.last_used).as_secs_f64(),
                hit_count: entry.hit_count,
            })
            .collect()
    }

    /// Number of models in the pool.
    pub fn len(&self) -> usize {
        self.inner.lock().expect("model pool mutex poisoned").models.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
