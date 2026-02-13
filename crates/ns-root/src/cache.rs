//! LRU basket cache for decompressed TTree basket payloads.
//!
//! Inspired by UnROOT.jl's `@memoize LRU` basket caching strategy.
//! ROOT baskets are immutable once written, so caching by file seek position
//! is safe and eliminates redundant decompression when the same branch is read
//! multiple times (e.g. in multi-branch formula evaluation).
//!
//! # Design decisions
//!
//! - **Key**: `u64` basket seek position — unique per basket within a file.
//! - **Value**: `Arc<[u8]>` — shared ownership avoids cloning multi-MB payloads.
//! - **Capacity**: bounded by total decompressed bytes (not entry count).
//! - **Thread-safety**: `Mutex<Inner>` — contention is low because decompression
//!   dominates wall-clock time; a lock-free map would add complexity for negligible gain.
//! - **Scope**: per-`RootFile` instance (no global state, no cross-file interference).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Configuration for the basket cache.
#[derive(Debug, Clone, Copy)]
pub struct CacheConfig {
    /// Maximum total bytes of decompressed payloads to keep in cache.
    /// Default: 256 MiB.
    pub max_bytes: usize,
    /// Whether caching is enabled. When `false`, `get`/`insert` are no-ops.
    pub enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_bytes: 256 * 1024 * 1024, // 256 MiB
            enabled: true,
        }
    }
}

impl CacheConfig {
    /// Create a disabled (no-op) cache config.
    pub fn disabled() -> Self {
        Self { max_bytes: 0, enabled: false }
    }
}

/// A single cached basket entry.
struct CacheEntry {
    /// Decompressed basket payload (shared).
    data: Arc<[u8]>,
    /// Size in bytes (== data.len()), stored separately to avoid Arc deref in eviction.
    size: usize,
    /// Previous key in LRU list.
    prev: Option<u64>,
    /// Next key in LRU list.
    next: Option<u64>,
}

/// Internal LRU state.
///
/// Uses `HashMap` for O(1) lookup and an intrusive doubly-linked list for O(1)
/// promotion/eviction on cache hit/insert.
struct Inner {
    map: HashMap<u64, CacheEntry>,
    /// Least-recently-used key.
    head: Option<u64>,
    /// Most-recently-used key.
    tail: Option<u64>,
    /// Current total bytes stored.
    current_bytes: usize,
    /// Maximum bytes allowed.
    max_bytes: usize,
    // ── Stats ──
    hits: u64,
    misses: u64,
}

impl Inner {
    fn new(max_bytes: usize) -> Self {
        Self {
            map: HashMap::new(),
            head: None,
            tail: None,
            current_bytes: 0,
            max_bytes,
            hits: 0,
            misses: 0,
        }
    }

    fn detach(&mut self, key: u64) {
        let (prev, next) = match self.map.get(&key) {
            Some(entry) => (entry.prev, entry.next),
            None => return,
        };

        if let Some(prev_key) = prev {
            if let Some(prev_entry) = self.map.get_mut(&prev_key) {
                prev_entry.next = next;
            }
        } else {
            self.head = next;
        }

        if let Some(next_key) = next {
            if let Some(next_entry) = self.map.get_mut(&next_key) {
                next_entry.prev = prev;
            }
        } else {
            self.tail = prev;
        }

        if let Some(entry) = self.map.get_mut(&key) {
            entry.prev = None;
            entry.next = None;
        }
    }

    fn push_back(&mut self, key: u64) {
        if self.tail == Some(key) {
            return;
        }
        if !self.map.contains_key(&key) {
            return;
        }

        match self.tail {
            Some(tail_key) => {
                if let Some(tail_entry) = self.map.get_mut(&tail_key) {
                    tail_entry.next = Some(key);
                }
                if let Some(entry) = self.map.get_mut(&key) {
                    entry.prev = Some(tail_key);
                    entry.next = None;
                }
                self.tail = Some(key);
            }
            None => {
                if let Some(entry) = self.map.get_mut(&key) {
                    entry.prev = None;
                    entry.next = None;
                }
                self.head = Some(key);
                self.tail = Some(key);
            }
        }
    }

    fn pop_front(&mut self) -> Option<(u64, CacheEntry)> {
        let key = self.head?;
        self.detach(key);
        self.map.remove_entry(&key)
    }

    fn remove_key(&mut self, key: u64) -> Option<CacheEntry> {
        if !self.map.contains_key(&key) {
            return None;
        }
        self.detach(key);
        self.map.remove(&key)
    }

    fn get(&mut self, key: u64) -> Option<Arc<[u8]>> {
        if self.map.contains_key(&key) {
            self.hits += 1;
            if self.tail != Some(key) {
                self.detach(key);
                self.push_back(key);
            }
            self.map.get(&key).map(|entry| Arc::clone(&entry.data))
        } else {
            self.misses += 1;
            None
        }
    }

    fn insert(&mut self, key: u64, data: Vec<u8>) -> Arc<[u8]> {
        let size = data.len();

        // If this single entry exceeds max capacity, don't cache it
        if size > self.max_bytes {
            return Arc::from(data);
        }

        if let Some(old) = self.remove_key(key) {
            self.current_bytes = self.current_bytes.saturating_sub(old.size);
        }

        // Evict LRU entries until we have room
        while self.current_bytes + size > self.max_bytes {
            if let Some((_, evicted)) = self.pop_front() {
                self.current_bytes -= evicted.size;
            } else {
                break;
            }
        }

        let arc: Arc<[u8]> = Arc::from(data);
        let entry = CacheEntry { data: Arc::clone(&arc), size, prev: None, next: None };
        self.map.insert(key, entry);
        self.push_back(key);

        self.current_bytes += size;
        arc
    }
}

/// Thread-safe LRU cache for decompressed basket payloads.
///
/// # Usage
///
/// ```ignore
/// let cache = BasketCache::new(CacheConfig::default());
///
/// // Try cache first
/// let data = if let Some(cached) = cache.get(seek_pos) {
///     cached
/// } else {
///     let decompressed = decompress_basket(file_data, seek_pos)?;
///     cache.insert(seek_pos, decompressed)
/// };
/// ```
pub struct BasketCache {
    inner: Mutex<Inner>,
    enabled: bool,
}

impl BasketCache {
    /// Create a new basket cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        Self { inner: Mutex::new(Inner::new(config.max_bytes)), enabled: config.enabled }
    }

    /// Look up a cached basket by its file seek position.
    ///
    /// Returns `None` on miss (or if caching is disabled).
    pub fn get(&self, seek: u64) -> Option<Arc<[u8]>> {
        if !self.enabled {
            return None;
        }
        self.inner.lock().unwrap().get(seek)
    }

    /// Insert a decompressed basket into the cache.
    ///
    /// Returns an `Arc` pointing to the data (whether newly inserted or
    /// if the entry was too large to cache).
    pub fn insert(&self, seek: u64, data: Vec<u8>) -> Arc<[u8]> {
        if !self.enabled {
            return Arc::from(data);
        }
        self.inner.lock().unwrap().insert(seek, data)
    }

    /// Get-or-insert: returns cached data or calls `f` to produce it and caches the result.
    pub fn get_or_insert<F, E>(&self, seek: u64, f: F) -> std::result::Result<Arc<[u8]>, E>
    where
        F: FnOnce() -> std::result::Result<Vec<u8>, E>,
    {
        if let Some(cached) = self.get(seek) {
            return Ok(cached);
        }
        let data = f()?;
        Ok(self.insert(seek, data))
    }

    /// Cache statistics snapshot.
    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.lock().unwrap();
        CacheStats {
            entries: inner.map.len(),
            current_bytes: inner.current_bytes,
            max_bytes: inner.max_bytes,
            hits: inner.hits,
            misses: inner.misses,
        }
    }

    /// Clear all cached entries.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.map.clear();
        inner.head = None;
        inner.tail = None;
        inner.current_bytes = 0;
    }
}

impl Default for BasketCache {
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

/// Snapshot of cache statistics.
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    /// Number of cached baskets.
    pub entries: usize,
    /// Current total bytes in cache.
    pub current_bytes: usize,
    /// Maximum configured bytes.
    pub max_bytes: usize,
    /// Total cache hits since creation.
    pub hits: u64,
    /// Total cache misses since creation.
    pub misses: u64,
}

impl CacheStats {
    /// Hit rate as a fraction [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }

    /// Fill ratio as a fraction [0.0, 1.0].
    pub fn fill_ratio(&self) -> f64 {
        if self.max_bytes == 0 { 0.0 } else { self.current_bytes as f64 / self.max_bytes as f64 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_insert_and_get() {
        let cache = BasketCache::new(CacheConfig { max_bytes: 1024, enabled: true });
        assert!(cache.get(100).is_none());

        let data = vec![1u8, 2, 3, 4];
        let arc = cache.insert(100, data.clone());
        assert_eq!(&*arc, &data);

        let cached = cache.get(100).unwrap();
        assert_eq!(&*cached, &data);

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.current_bytes, 4);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn evicts_lru_when_full() {
        let cache = BasketCache::new(CacheConfig { max_bytes: 10, enabled: true });

        cache.insert(1, vec![0u8; 4]); // 4 bytes
        cache.insert(2, vec![0u8; 4]); // 8 bytes total
        assert_eq!(cache.stats().entries, 2);

        // This should evict key=1 (LRU) to make room
        cache.insert(3, vec![0u8; 4]); // would be 12, need to evict
        assert!(cache.get(1).is_none()); // evicted
        assert!(cache.get(2).is_some()); // still here (was MRU before key=3)
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn oversized_entry_not_cached() {
        let cache = BasketCache::new(CacheConfig { max_bytes: 10, enabled: true });
        let arc = cache.insert(1, vec![0u8; 20]); // exceeds max
        assert_eq!(arc.len(), 20); // still returned
        assert!(cache.get(1).is_none()); // but not cached
        assert_eq!(cache.stats().entries, 0);
    }

    #[test]
    fn disabled_cache_is_noop() {
        let cache = BasketCache::new(CacheConfig::disabled());
        let arc = cache.insert(1, vec![1, 2, 3]);
        assert_eq!(&*arc, &[1, 2, 3]);
        assert!(cache.get(1).is_none());
        assert_eq!(cache.stats().entries, 0);
    }

    #[test]
    fn get_or_insert_caches_on_miss() {
        let cache = BasketCache::new(CacheConfig { max_bytes: 1024, enabled: true });
        let result: std::result::Result<_, ()> = cache.get_or_insert(42, || Ok(vec![10, 20, 30]));
        let arc = result.unwrap();
        assert_eq!(&*arc, &[10, 20, 30]);

        // Second call should hit cache (closure not called)
        let result: std::result::Result<_, ()> =
            cache.get_or_insert(42, || panic!("should not be called"));
        assert_eq!(&*result.unwrap(), &[10, 20, 30]);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn clear_empties_cache() {
        let cache = BasketCache::new(CacheConfig { max_bytes: 1024, enabled: true });
        cache.insert(1, vec![1, 2, 3]);
        cache.insert(2, vec![4, 5, 6]);
        assert_eq!(cache.stats().entries, 2);

        cache.clear();
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().current_bytes, 0);
        assert!(cache.get(1).is_none());
    }
}
