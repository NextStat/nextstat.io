//! Per-IP token-bucket rate limiting middleware for the NextStat server.
//!
//! Configurable via `--rate-limit` CLI flag (requests per second per IP).
//! When set to 0, rate limiting is disabled.  Health endpoint is always exempt.
//!
//! Implementation: fixed-window token bucket stored in a `DashMap`-style
//! structure (we use `std::sync::Mutex<HashMap>` to avoid new dependencies).
//! Stale entries are lazily pruned on each request.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Mutex;
use std::time::Instant;

use axum::body::Body;
use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

/// Rate limiter configuration and state.
#[derive(Clone)]
pub struct RateLimiter(pub Option<std::sync::Arc<RateLimiterInner>>);

pub struct RateLimiterInner {
    /// Maximum requests per window per IP.
    pub max_requests: u32,
    /// Window duration in seconds.
    pub window_secs: u64,
    /// Per-IP buckets.
    buckets: Mutex<HashMap<IpAddr, Bucket>>,
}

struct Bucket {
    count: u32,
    window_start: Instant,
}

impl RateLimiterInner {
    pub fn new(max_requests: u32, window_secs: u64) -> Self {
        Self { max_requests, window_secs, buckets: Mutex::new(HashMap::new()) }
    }

    /// Returns `true` if the request is allowed, `false` if rate-limited.
    pub fn check(&self, ip: IpAddr) -> bool {
        let now = Instant::now();
        let window = std::time::Duration::from_secs(self.window_secs);
        let mut buckets = self.buckets.lock().unwrap_or_else(|e| e.into_inner());

        // Lazy prune: remove stale entries (older than 2× window).
        let prune_threshold = window.saturating_mul(2);
        if buckets.len() > 10_000 {
            buckets.retain(|_, b| now.duration_since(b.window_start) < prune_threshold);
        }

        let bucket = buckets.entry(ip).or_insert_with(|| Bucket { count: 0, window_start: now });

        // Reset window if expired.
        if now.duration_since(bucket.window_start) >= window {
            bucket.count = 0;
            bucket.window_start = now;
        }

        if bucket.count >= self.max_requests {
            return false;
        }

        bucket.count += 1;
        true
    }
}

impl RateLimiter {
    /// Create a disabled rate limiter.
    pub fn disabled() -> Self {
        Self(None)
    }

    /// Create an enabled rate limiter.
    pub fn new(max_requests_per_second: u32) -> Self {
        if max_requests_per_second == 0 {
            return Self::disabled();
        }
        Self(Some(std::sync::Arc::new(RateLimiterInner::new(max_requests_per_second, 1))))
    }

    pub fn is_enabled(&self) -> bool {
        self.0.is_some()
    }
}

/// Extract client IP from request (checks X-Forwarded-For, then ConnectInfo).
fn extract_ip(request: &Request<Body>) -> Option<IpAddr> {
    // Try X-Forwarded-For header first (reverse proxy).
    if let Some(xff) = request.headers().get("x-forwarded-for")
        && let Ok(s) = xff.to_str()
        && let Some(first) = s.split(',').next()
        && let Ok(ip) = first.trim().parse::<IpAddr>()
    {
        return Some(ip);
    }

    // Try axum ConnectInfo.
    request
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0.ip())
}

/// Axum middleware function.
pub async fn rate_limit_middleware(request: Request<Body>, next: Next) -> Response {
    let limiter =
        request.extensions().get::<RateLimiter>().cloned().unwrap_or(RateLimiter::disabled());

    // Skip if disabled.
    let inner = match &limiter.0 {
        Some(inner) => inner,
        None => return next.run(request).await,
    };

    // Always allow health endpoint.
    if request.method() == axum::http::Method::GET && request.uri().path() == "/v1/health" {
        return next.run(request).await;
    }

    let ip = extract_ip(&request).unwrap_or(IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED));

    if inner.check(ip) {
        next.run(request).await
    } else {
        let body = serde_json::json!({
            "error": "rate limit exceeded",
            "retry_after_s": inner.window_secs,
        });
        (StatusCode::TOO_MANY_REQUESTS, axum::Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limiter_allows_within_budget() {
        let inner = RateLimiterInner::new(3, 1);
        let ip: IpAddr = "127.0.0.1".parse().unwrap();
        assert!(inner.check(ip));
        assert!(inner.check(ip));
        assert!(inner.check(ip));
        assert!(!inner.check(ip)); // 4th request blocked
    }

    #[test]
    fn rate_limiter_different_ips_independent() {
        let inner = RateLimiterInner::new(1, 1);
        let ip1: IpAddr = "10.0.0.1".parse().unwrap();
        let ip2: IpAddr = "10.0.0.2".parse().unwrap();
        assert!(inner.check(ip1));
        assert!(!inner.check(ip1));
        assert!(inner.check(ip2)); // different IP, fresh budget
    }

    #[test]
    fn disabled_rate_limiter() {
        let rl = RateLimiter::new(0);
        assert!(!rl.is_enabled());
    }
}
