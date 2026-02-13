//! Zero-copy concatenation of multiple byte buffers.
//!
//! [`ChainedSlice`] presents a sequence of `Arc<[u8]>` segments as a single
//! logical byte slice without copying. This avoids the intermediate allocation
//! that eager `read_all_baskets()` → `Vec<Vec<u8>>` → flat decode requires.

use std::sync::Arc;

/// A logical byte slice backed by multiple non-contiguous `Arc<[u8]>` segments.
///
/// Provides random access by logical offset and an iterator over contiguous
/// segments. No data is copied — segments are borrowed via `Arc`.
#[derive(Debug, Clone)]
pub struct ChainedSlice {
    /// The underlying segments (shared ownership, O(1) clone).
    segments: Vec<Arc<[u8]>>,
    /// Cumulative byte offsets: `cum_len[i]` = total bytes in segments `0..i`.
    /// Length is `segments.len() + 1`, with `cum_len[0] == 0`.
    cum_len: Vec<usize>,
}

impl ChainedSlice {
    /// Build from a vector of already-decompressed basket payloads.
    pub fn new(segments: Vec<Arc<[u8]>>) -> Self {
        let mut cum_len = Vec::with_capacity(segments.len() + 1);
        cum_len.push(0);
        for seg in &segments {
            cum_len.push(cum_len.last().unwrap() + seg.len());
        }
        Self { segments, cum_len }
    }

    /// Total logical length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        *self.cum_len.last().unwrap_or(&0)
    }

    /// Whether the chain is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of segments.
    #[inline]
    pub fn n_segments(&self) -> usize {
        self.segments.len()
    }

    /// Access segment `i` as a byte slice.
    #[inline]
    pub fn segment(&self, i: usize) -> &[u8] {
        self.segments[i].as_ref()
    }

    /// Cumulative byte offset of segment `i` (start position in logical space).
    #[inline]
    pub fn segment_offset(&self, i: usize) -> usize {
        self.cum_len[i]
    }

    /// Iterator over `(segment_index, &[u8])` pairs.
    pub fn segments(&self) -> impl Iterator<Item = (usize, &[u8])> {
        self.segments.iter().enumerate().map(|(i, s)| (i, s.as_ref()))
    }

    /// Read a single byte at logical offset `pos`.
    ///
    /// # Panics
    /// Panics if `pos >= self.len()`.
    #[inline]
    pub fn get(&self, pos: usize) -> u8 {
        let (seg_idx, local) = self.locate(pos);
        self.segments[seg_idx][local]
    }

    /// Copy `dst.len()` bytes starting at logical offset `pos` into `dst`.
    ///
    /// # Panics
    /// Panics if `pos + dst.len() > self.len()`.
    pub fn copy_to_slice(&self, pos: usize, dst: &mut [u8]) {
        if dst.is_empty() {
            return;
        }
        assert!(pos + dst.len() <= self.len(), "ChainedSlice: read out of bounds");

        let (mut seg_idx, mut local) = self.locate(pos);
        let mut written = 0;

        while written < dst.len() {
            let seg = &self.segments[seg_idx];
            let avail = seg.len() - local;
            let need = dst.len() - written;
            let n = avail.min(need);
            dst[written..written + n].copy_from_slice(&seg[local..local + n]);
            written += n;
            seg_idx += 1;
            local = 0;
        }
    }

    /// Read `N` bytes at logical offset `pos` as a fixed-size array.
    ///
    /// Fast path: if the range lies within a single segment, no copy.
    /// Slow path: copies across segment boundary.
    #[inline]
    pub fn read_array<const N: usize>(&self, pos: usize) -> [u8; N] {
        let (seg_idx, local) = self.locate(pos);
        let seg = &self.segments[seg_idx];
        if local + N <= seg.len() {
            // Fast path — entirely within one segment.
            let mut arr = [0u8; N];
            arr.copy_from_slice(&seg[local..local + N]);
            arr
        } else {
            // Slow path — crosses segment boundary.
            let mut arr = [0u8; N];
            self.copy_to_slice(pos, &mut arr);
            arr
        }
    }

    /// Locate the segment index and local offset for a given logical position.
    ///
    /// Uses binary search on `cum_len`.
    #[inline]
    fn locate(&self, pos: usize) -> (usize, usize) {
        debug_assert!(pos < self.len(), "ChainedSlice: pos {} >= len {}", pos, self.len());
        // Binary search: find the last segment where cum_len[i] <= pos.
        let seg_idx = match self.cum_len.binary_search(&pos) {
            Ok(i) => {
                // Exact match. If i == segments.len(), back up.
                if i < self.segments.len() { i } else { i - 1 }
            }
            Err(i) => i - 1,
        };
        let local = pos - self.cum_len[seg_idx];
        (seg_idx, local)
    }

    /// Decode a single f64 value at logical offset `pos` given a leaf type.
    ///
    /// This is the primary hot-path method for LazyBranchReader: decode one
    /// element without materializing a contiguous buffer.
    #[inline]
    pub fn decode_f64_at(&self, pos: usize, leaf_type: crate::tree::LeafType) -> f64 {
        use crate::tree::LeafType;
        match leaf_type {
            LeafType::F64 => f64::from_be_bytes(self.read_array::<8>(pos)),
            LeafType::F32 => f32::from_be_bytes(self.read_array::<4>(pos)) as f64,
            LeafType::I32 => i32::from_be_bytes(self.read_array::<4>(pos)) as f64,
            LeafType::I64 => i64::from_be_bytes(self.read_array::<8>(pos)) as f64,
            LeafType::U32 => u32::from_be_bytes(self.read_array::<4>(pos)) as f64,
            LeafType::U64 => u64::from_be_bytes(self.read_array::<8>(pos)) as f64,
            LeafType::I16 => i16::from_be_bytes(self.read_array::<2>(pos)) as f64,
            LeafType::I8 => self.get(pos) as i8 as f64,
            LeafType::Bool => {
                if self.get(pos) != 0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_chain() {
        let c = ChainedSlice::new(vec![]);
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
        assert_eq!(c.n_segments(), 0);
    }

    #[test]
    fn single_segment() {
        let data: Arc<[u8]> = Arc::from(vec![1u8, 2, 3, 4]);
        let c = ChainedSlice::new(vec![data]);
        assert_eq!(c.len(), 4);
        assert_eq!(c.n_segments(), 1);
        assert_eq!(c.get(0), 1);
        assert_eq!(c.get(3), 4);

        let mut buf = [0u8; 4];
        c.copy_to_slice(0, &mut buf);
        assert_eq!(buf, [1, 2, 3, 4]);
    }

    #[test]
    fn multi_segment_read() {
        let s1: Arc<[u8]> = Arc::from(vec![10u8, 20]);
        let s2: Arc<[u8]> = Arc::from(vec![30u8, 40, 50]);
        let s3: Arc<[u8]> = Arc::from(vec![60u8]);
        let c = ChainedSlice::new(vec![s1, s2, s3]);
        assert_eq!(c.len(), 6);
        assert_eq!(c.n_segments(), 3);

        assert_eq!(c.get(0), 10);
        assert_eq!(c.get(1), 20);
        assert_eq!(c.get(2), 30);
        assert_eq!(c.get(4), 50);
        assert_eq!(c.get(5), 60);
    }

    #[test]
    fn cross_segment_copy() {
        let s1: Arc<[u8]> = Arc::from(vec![1u8, 2]);
        let s2: Arc<[u8]> = Arc::from(vec![3u8, 4]);
        let c = ChainedSlice::new(vec![s1, s2]);

        let mut buf = [0u8; 4];
        c.copy_to_slice(0, &mut buf);
        assert_eq!(buf, [1, 2, 3, 4]);

        let mut buf2 = [0u8; 2];
        c.copy_to_slice(1, &mut buf2);
        assert_eq!(buf2, [2, 3]);
    }

    #[test]
    fn read_array_cross_boundary() {
        // Place f32 1.0 (0x3F800000) spanning segments s1 and s2.
        let s1: Arc<[u8]> = Arc::from(vec![0xAA, 0xBB, 0x3F, 0x80]); // last 2 bytes = first half of f32
        let s2: Arc<[u8]> = Arc::from(vec![0x00u8, 0x00, 0xCC]); // first 2 bytes = second half of f32
        let c = ChainedSlice::new(vec![s1, s2]);
        // f32 at offset 2 crosses s1→s2: bytes [0x3F, 0x80, 0x00, 0x00] = 1.0f32
        let arr = c.read_array::<4>(2);
        assert_eq!(f32::from_be_bytes(arr), 1.0);
    }

    #[test]
    fn decode_f64_at_within_segment() {
        use crate::tree::LeafType;
        let val: f64 = std::f64::consts::PI;
        let bytes = val.to_be_bytes();
        let data: Arc<[u8]> = Arc::from(bytes.to_vec());
        let c = ChainedSlice::new(vec![data]);
        let result = c.decode_f64_at(0, LeafType::F64);
        assert!((result - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn segment_iteration() {
        let s1: Arc<[u8]> = Arc::from(vec![1u8, 2]);
        let s2: Arc<[u8]> = Arc::from(vec![3u8]);
        let c = ChainedSlice::new(vec![s1, s2]);
        let segs: Vec<(usize, &[u8])> = c.segments().collect();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0], (0, &[1u8, 2][..]));
        assert_eq!(segs[1], (1, &[3u8][..]));
    }

    #[test]
    fn segment_offset() {
        let s1: Arc<[u8]> = Arc::from(vec![0u8; 100]);
        let s2: Arc<[u8]> = Arc::from(vec![0u8; 200]);
        let s3: Arc<[u8]> = Arc::from(vec![0u8; 50]);
        let c = ChainedSlice::new(vec![s1, s2, s3]);
        assert_eq!(c.segment_offset(0), 0);
        assert_eq!(c.segment_offset(1), 100);
        assert_eq!(c.segment_offset(2), 300);
    }
}
