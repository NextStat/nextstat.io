//! Lazy (on-demand) branch reader that decompresses only the baskets needed.
//!
//! Unlike [`BranchReader`](crate::branch_reader::BranchReader) which eagerly
//! decompresses **all** baskets, `LazyBranchReader` loads baskets on demand:
//!
//! - **Single entry**: only the one basket containing that entry is touched.
//! - **Entry range**: only the baskets overlapping the range are decompressed.
//! - **Full read**: all baskets, but via cache so repeated reads are free.
//!
//! Combined with [`BasketCache`](crate::cache::BasketCache), this gives
//! UnROOT.jl-style lazy semantics with Rust safety and zero-copy `Arc` sharing.

use std::sync::Arc;

use crate::basket::read_basket_data_cached;
use crate::branch_reader::BranchReader;
use crate::cache::BasketCache;
use crate::chained_slice::ChainedSlice;
use crate::error::{Result, RootError};
use crate::tree::{BranchInfo, LeafType};

/// Lazy reader for a single TTree branch.
///
/// Decompresses baskets on demand, backed by an LRU [`BasketCache`].
/// Provides entry-level and range-level access without materializing the
/// entire branch into memory.
pub struct LazyBranchReader<'a> {
    file_data: &'a [u8],
    branch: &'a BranchInfo,
    is_large: bool,
    cache: &'a BasketCache,
}

impl<'a> LazyBranchReader<'a> {
    /// Create a new lazy branch reader.
    pub fn new(
        file_data: &'a [u8],
        branch: &'a BranchInfo,
        is_large: bool,
        cache: &'a BasketCache,
    ) -> Self {
        Self { file_data, branch, is_large, cache }
    }

    /// Branch metadata.
    #[inline]
    pub fn branch(&self) -> &BranchInfo {
        self.branch
    }

    /// Total number of entries in this branch.
    #[inline]
    pub fn n_entries(&self) -> u64 {
        self.branch.entries
    }

    /// Number of baskets.
    #[inline]
    pub fn n_baskets(&self) -> usize {
        self.branch.n_baskets
    }

    // ── Entry ↔ Basket mapping ────────────────────────────────────

    /// Find which basket contains `entry_idx`.
    ///
    /// Returns `(basket_index, local_entry_offset_within_basket)`.
    /// Returns `Err` if `entry_idx >= self.n_entries()`.
    pub fn basket_for_entry(&self, entry_idx: u64) -> Result<(usize, usize)> {
        if entry_idx >= self.branch.entries {
            return Err(RootError::Deserialization(format!(
                "entry index {} out of range (branch '{}' has {} entries)",
                entry_idx, self.branch.name, self.branch.entries
            )));
        }
        // basket_entry[i] = first entry of basket i. Find the last basket where
        // basket_entry[basket] <= entry_idx.
        let be = &self.branch.basket_entry;
        let basket_idx = match be.binary_search(&entry_idx) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let basket_start = be.get(basket_idx).copied().unwrap_or(0);
        let local = (entry_idx - basket_start) as usize;
        Ok((basket_idx, local))
    }

    /// Find which baskets overlap the entry range `[start, end)`.
    ///
    /// Returns `(first_basket_idx, last_basket_idx_inclusive)`.
    pub fn baskets_for_range(&self, start: u64, end: u64) -> Result<(usize, usize)> {
        if start >= end || start >= self.branch.entries {
            return Err(RootError::Deserialization(format!(
                "invalid entry range [{}, {}) for branch '{}' with {} entries",
                start, end, self.branch.name, self.branch.entries
            )));
        }
        let clamped_end = end.min(self.branch.entries);
        let (first_basket, _) = self.basket_for_entry(start)?;
        let (last_basket, _) = self.basket_for_entry(clamped_end - 1)?;
        Ok((first_basket, last_basket))
    }

    // ── Single basket loading ─────────────────────────────────────

    /// Load and return a single basket payload (cached).
    pub fn load_basket(&self, basket_idx: usize) -> Result<Arc<[u8]>> {
        if basket_idx >= self.branch.n_baskets {
            return Err(RootError::Deserialization(format!(
                "basket index {} out of range (branch '{}' has {} baskets)",
                basket_idx, self.branch.name, self.branch.n_baskets
            )));
        }
        read_basket_data_cached(
            self.file_data,
            self.branch.basket_seek[basket_idx],
            self.is_large,
            Some(self.cache),
        )
    }

    /// Load baskets for a range and return a [`ChainedSlice`] (zero-copy view).
    pub fn load_range_chained(
        &self,
        first_basket: usize,
        last_basket: usize,
    ) -> Result<ChainedSlice> {
        let mut segments = Vec::with_capacity(last_basket - first_basket + 1);
        for i in first_basket..=last_basket {
            segments.push(self.load_basket(i)?);
        }
        Ok(ChainedSlice::new(segments))
    }

    // ── High-level typed access ───────────────────────────────────

    /// Read a single entry as `f64` (scalar branches only, `entry_offset_len == 0`).
    ///
    /// Only decompresses the one basket that contains this entry.
    pub fn read_f64_at(&self, entry_idx: u64) -> Result<f64> {
        let elem_size = self.branch.leaf_type.byte_size();
        if self.branch.entry_offset_len != 0 {
            return Err(RootError::TypeMismatch(
                "read_f64_at only supports scalar (non-jagged) branches".into(),
            ));
        }
        let (basket_idx, local) = self.basket_for_entry(entry_idx)?;
        let payload = self.load_basket(basket_idx)?;
        let offset = local * elem_size;
        if offset + elem_size > payload.len() {
            return Err(RootError::Deserialization(format!(
                "payload too short for entry {}: need {} bytes at offset {}, have {}",
                entry_idx,
                elem_size,
                offset,
                payload.len()
            )));
        }
        Ok(decode_one_f64_from_slice(&payload, offset, self.branch.leaf_type))
    }

    /// Read a contiguous range of entries `[start, end)` as `Vec<f64>` (scalar branches).
    ///
    /// Only decompresses baskets overlapping the requested range.
    pub fn read_f64_range(&self, start: u64, end: u64) -> Result<Vec<f64>> {
        if self.branch.entry_offset_len != 0 {
            return Err(RootError::TypeMismatch(
                "read_f64_range only supports scalar (non-jagged) branches".into(),
            ));
        }
        let clamped_end = end.min(self.branch.entries);
        if start >= clamped_end {
            return Ok(Vec::new());
        }

        let elem_size = self.branch.leaf_type.byte_size();
        let n = (clamped_end - start) as usize;
        let mut out = Vec::with_capacity(n);

        let (first_basket, last_basket) = self.baskets_for_range(start, clamped_end)?;

        for bi in first_basket..=last_basket {
            let payload = self.load_basket(bi)?;
            let basket_start_entry = self.branch.basket_entry.get(bi).copied().unwrap_or(0);
            let basket_end_entry =
                self.branch.basket_entry.get(bi + 1).copied().unwrap_or(self.branch.entries);

            // Clamp to requested range.
            let range_start = start.max(basket_start_entry);
            let range_end = clamped_end.min(basket_end_entry);

            let local_start = (range_start - basket_start_entry) as usize;
            let local_end = (range_end - basket_start_entry) as usize;
            let start_off = local_start * elem_size;
            let end_off = local_end * elem_size;
            if end_off > payload.len() {
                return Err(RootError::Deserialization(format!(
                    "payload too short in basket {}: need {} bytes at offset {}, have {}",
                    bi,
                    elem_size,
                    end_off.saturating_sub(elem_size),
                    payload.len()
                )));
            }
            decode_window_f64_from_slice(
                &payload[start_off..end_off],
                self.branch.leaf_type,
                &mut out,
            );
        }

        Ok(out)
    }

    /// Read all entries as `Vec<f64>` (any scalar branch).
    ///
    /// Functionally equivalent to `BranchReader::as_f64()` but uses lazy loading + cache.
    pub fn read_all_f64(&self) -> Result<Vec<f64>> {
        if self.branch.entry_offset_len != 0 {
            return Err(RootError::TypeMismatch(
                "read_all_f64 only supports scalar (non-jagged) branches".into(),
            ));
        }
        let reader =
            BranchReader::with_cache(self.file_data, self.branch, self.is_large, self.cache);
        reader.as_f64()
    }

    /// Build a [`ChainedSlice`] over all baskets (zero-copy, cached).
    ///
    /// The returned `ChainedSlice` holds `Arc` references to each decompressed
    /// basket. Useful for custom decode pipelines.
    pub fn load_all_chained(&self) -> Result<ChainedSlice> {
        if self.branch.n_baskets == 0 {
            return Ok(ChainedSlice::new(vec![]));
        }
        self.load_range_chained(0, self.branch.n_baskets - 1)
    }
}

/// Decode a single f64 value from a byte slice at the given offset.
#[inline]
fn decode_one_f64_from_slice(data: &[u8], off: usize, leaf_type: LeafType) -> f64 {
    match leaf_type {
        LeafType::F64 => f64::from_be_bytes(data[off..off + 8].try_into().unwrap()),
        LeafType::F32 => f32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as f64,
        LeafType::I32 => i32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as f64,
        LeafType::I64 => i64::from_be_bytes(data[off..off + 8].try_into().unwrap()) as f64,
        LeafType::U32 => u32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as f64,
        LeafType::U64 => u64::from_be_bytes(data[off..off + 8].try_into().unwrap()) as f64,
        LeafType::I16 => i16::from_be_bytes(data[off..off + 2].try_into().unwrap()) as f64,
        LeafType::I8 => data[off] as i8 as f64,
        LeafType::Bool => {
            if data[off] != 0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

#[inline]
unsafe fn read_u64_be_unaligned(ptr: *const u8) -> u64 {
    u64::from_be(unsafe { std::ptr::read_unaligned(ptr as *const u64) })
}

#[inline]
unsafe fn read_u32_be_unaligned(ptr: *const u8) -> u32 {
    u32::from_be(unsafe { std::ptr::read_unaligned(ptr as *const u32) })
}

#[inline]
unsafe fn read_u16_be_unaligned(ptr: *const u8) -> u16 {
    u16::from_be(unsafe { std::ptr::read_unaligned(ptr as *const u16) })
}

#[inline]
fn decode_window_f64_from_slice(data: &[u8], leaf_type: LeafType, out: &mut Vec<f64>) {
    match leaf_type {
        LeafType::F64 => {
            let ptr = data.as_ptr();
            let mut off = 0usize;
            while off + 8 <= data.len() {
                let bits = unsafe { read_u64_be_unaligned(ptr.add(off)) };
                out.push(f64::from_bits(bits));
                off += 8;
            }
        }
        LeafType::F32 => {
            let ptr = data.as_ptr();
            let mut off = 0usize;
            while off + 4 <= data.len() {
                let bits = unsafe { read_u32_be_unaligned(ptr.add(off)) };
                out.push(f32::from_bits(bits) as f64);
                off += 4;
            }
        }
        LeafType::I32 => {
            let ptr = data.as_ptr();
            let mut off = 0usize;
            while off + 4 <= data.len() {
                let v = unsafe { read_u32_be_unaligned(ptr.add(off)) } as i32;
                out.push(v as f64);
                off += 4;
            }
        }
        LeafType::I64 => {
            let ptr = data.as_ptr();
            let mut off = 0usize;
            while off + 8 <= data.len() {
                let v = unsafe { read_u64_be_unaligned(ptr.add(off)) } as i64;
                out.push(v as f64);
                off += 8;
            }
        }
        LeafType::U32 => {
            let ptr = data.as_ptr();
            let mut off = 0usize;
            while off + 4 <= data.len() {
                let v = unsafe { read_u32_be_unaligned(ptr.add(off)) };
                out.push(v as f64);
                off += 4;
            }
        }
        LeafType::U64 => {
            let ptr = data.as_ptr();
            let mut off = 0usize;
            while off + 8 <= data.len() {
                let v = unsafe { read_u64_be_unaligned(ptr.add(off)) };
                out.push(v as f64);
                off += 8;
            }
        }
        LeafType::I16 => {
            let ptr = data.as_ptr();
            let mut off = 0usize;
            while off + 2 <= data.len() {
                let v = unsafe { read_u16_be_unaligned(ptr.add(off)) } as i16;
                out.push(v as f64);
                off += 2;
            }
        }
        LeafType::I8 => {
            let ptr = data.as_ptr();
            for off in 0..data.len() {
                let v = unsafe { *ptr.add(off) } as i8;
                out.push(v as f64);
            }
        }
        LeafType::Bool => {
            let ptr = data.as_ptr();
            for off in 0..data.len() {
                let v = unsafe { *ptr.add(off) };
                out.push(if v != 0 { 1.0 } else { 0.0 });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CacheConfig;

    /// Build a minimal BranchInfo for testing with flat f64 data.
    fn test_branch(n_entries: u64, n_baskets: usize, entries_per_basket: u64) -> BranchInfo {
        let mut basket_entry = Vec::with_capacity(n_baskets);
        let mut basket_seek = Vec::with_capacity(n_baskets);
        for i in 0..n_baskets {
            basket_entry.push(i as u64 * entries_per_basket);
            basket_seek.push((i * 1000) as u64); // dummy seeks
        }
        BranchInfo {
            name: "test".into(),
            leaf_type: LeafType::F64,
            entry_offset_len: 0,
            entries: n_entries,
            basket_bytes: vec![0; n_baskets],
            basket_entry,
            basket_seek,
            n_baskets,
        }
    }

    #[test]
    fn basket_for_entry_single_basket() {
        let branch = test_branch(100, 1, 100);
        let cache = BasketCache::new(CacheConfig::disabled());
        // file_data won't be touched for basket_for_entry
        let lr = LazyBranchReader::new(&[], &branch, false, &cache);

        let (bi, local) = lr.basket_for_entry(0).unwrap();
        assert_eq!(bi, 0);
        assert_eq!(local, 0);

        let (bi, local) = lr.basket_for_entry(50).unwrap();
        assert_eq!(bi, 0);
        assert_eq!(local, 50);

        let (bi, local) = lr.basket_for_entry(99).unwrap();
        assert_eq!(bi, 0);
        assert_eq!(local, 99);

        assert!(lr.basket_for_entry(100).is_err());
    }

    #[test]
    fn basket_for_entry_multi_basket() {
        // 3 baskets: [0..10), [10..20), [20..30)
        let branch = test_branch(30, 3, 10);
        let cache = BasketCache::new(CacheConfig::disabled());
        let lr = LazyBranchReader::new(&[], &branch, false, &cache);

        let (bi, local) = lr.basket_for_entry(0).unwrap();
        assert_eq!((bi, local), (0, 0));

        let (bi, local) = lr.basket_for_entry(9).unwrap();
        assert_eq!((bi, local), (0, 9));

        let (bi, local) = lr.basket_for_entry(10).unwrap();
        assert_eq!((bi, local), (1, 0));

        let (bi, local) = lr.basket_for_entry(25).unwrap();
        assert_eq!((bi, local), (2, 5));

        let (bi, local) = lr.basket_for_entry(29).unwrap();
        assert_eq!((bi, local), (2, 9));
    }

    #[test]
    fn baskets_for_range_spanning() {
        let branch = test_branch(30, 3, 10);
        let cache = BasketCache::new(CacheConfig::disabled());
        let lr = LazyBranchReader::new(&[], &branch, false, &cache);

        // Range [5, 25) spans baskets 0, 1, 2
        let (first, last) = lr.baskets_for_range(5, 25).unwrap();
        assert_eq!(first, 0);
        assert_eq!(last, 2);

        // Range within single basket
        let (first, last) = lr.baskets_for_range(12, 18).unwrap();
        assert_eq!(first, 1);
        assert_eq!(last, 1);
    }
}
