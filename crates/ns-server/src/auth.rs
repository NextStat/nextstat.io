//! API key authentication middleware for the NextStat server.
//!
//! When enabled, all endpoints except `GET /v1/health` require a valid
//! `Authorization: Bearer <key>` header.  Keys are loaded at startup from
//! a file (`--api-keys <path>`, one key per line) or the `NS_API_KEYS`
//! environment variable (comma-separated).
//!
//! If no keys are configured the middleware is a no-op (open mode).

use std::collections::HashSet;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::Request;
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

/// Shared set of valid API keys.  `None` means auth is disabled.
#[derive(Clone)]
pub struct ApiKeys(pub Option<Arc<HashSet<String>>>);

impl ApiKeys {
    /// Load keys from a file (one key per line, blank/comment lines skipped)
    /// or fall back to the `NS_API_KEYS` environment variable.
    /// Returns `None` (auth disabled) if neither source provides keys.
    pub fn load(file_path: Option<&str>) -> Result<Self, String> {
        if let Some(path) = file_path {
            let content = std::fs::read_to_string(path)
                .map_err(|e| format!("failed to read API key file {path}: {e}"))?;
            let keys: HashSet<String> = content
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(String::from)
                .collect();
            if keys.is_empty() {
                return Err(format!("API key file {path} contains no valid keys"));
            }
            tracing::info!(n_keys = keys.len(), source = "file", "API key auth enabled");
            return Ok(Self(Some(Arc::new(keys))));
        }

        if let Ok(env_val) = std::env::var("NS_API_KEYS") {
            let keys: HashSet<String> = env_val
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if !keys.is_empty() {
                tracing::info!(n_keys = keys.len(), source = "env", "API key auth enabled");
                return Ok(Self(Some(Arc::new(keys))));
            }
        }

        tracing::info!("API key auth disabled (no keys configured)");
        Ok(Self(None))
    }

    /// Whether authentication is active.
    pub fn is_enabled(&self) -> bool {
        self.0.is_some()
    }

    /// Check if a key is valid.
    pub fn validate(&self, key: &str) -> bool {
        match &self.0 {
            Some(keys) => keys.contains(key),
            None => true, // auth disabled
        }
    }
}

/// Extract Bearer token from headers.
fn extract_bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|t| t.trim())
}

/// Axum middleware function.  Skips auth for `GET /v1/health`.
pub async fn auth_middleware(request: Request<Body>, next: Next) -> Response {
    // Extract ApiKeys from extensions (injected by the layer).
    let api_keys = request.extensions().get::<ApiKeys>().cloned().unwrap_or(ApiKeys(None));

    // Skip auth if disabled.
    if !api_keys.is_enabled() {
        return next.run(request).await;
    }

    // Always allow health endpoint without auth.
    if request.method() == axum::http::Method::GET && request.uri().path() == "/v1/health" {
        return next.run(request).await;
    }

    match extract_bearer(request.headers()) {
        Some(token) if api_keys.validate(token) => next.run(request).await,
        Some(_) => unauthorized("invalid API key"),
        None => unauthorized("missing Authorization: Bearer <key> header"),
    }
}

fn unauthorized(msg: &str) -> Response {
    let body = serde_json::json!({ "error": msg });
    (StatusCode::UNAUTHORIZED, axum::Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_keys_disabled_accepts_everything() {
        let keys = ApiKeys(None);
        assert!(!keys.is_enabled());
        assert!(keys.validate("any-token"));
    }

    #[test]
    fn api_keys_enabled_validates() {
        let set: HashSet<String> = ["key-abc".to_string(), "key-xyz".to_string()].into();
        let keys = ApiKeys(Some(Arc::new(set)));
        assert!(keys.is_enabled());
        assert!(keys.validate("key-abc"));
        assert!(keys.validate("key-xyz"));
        assert!(!keys.validate("key-bad"));
    }

    #[test]
    fn extract_bearer_parses_header() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer my-secret-key".parse().unwrap());
        assert_eq!(extract_bearer(&headers), Some("my-secret-key"));

        let mut headers2 = HeaderMap::new();
        headers2.insert("authorization", "Basic abc".parse().unwrap());
        assert_eq!(extract_bearer(&headers2), None);
    }
}
