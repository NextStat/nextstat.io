use crate::common::MAX_BLOCK_SIZE;
use crate::decoding::errors::DecodeBufferError;
use crate::io::{Error, Read, Write};
use alloc::vec::Vec;
#[cfg(feature = "hash")]
use core::hash::Hasher;

/// High-performance flat output buffer for non-streaming zstd decode.
///
/// Unlike `DecodeBuffer` (which wraps a `RingBuffer` with wraparound logic),
/// this uses a simple `Vec<u8>` — eliminating modulo arithmetic, two-slice splits,
/// and branch-heavy extend_from_within on every match copy.
///
/// Used by the fused decode+execute path for maximum throughput.
pub struct FlatDecodeBuffer {
    buf: Vec<u8>,
    /// Logical read head into `buf`.
    ///
    /// We avoid `copy_within()` on every `read()` by advancing this index, and only
    /// compacting occasionally. This keeps reads O(n) in the bytes actually drained.
    start: usize,
    pub dict_content: Vec<u8>,
    pub window_size: usize,
    total_output_counter: u64,
    #[cfg(feature = "hash")]
    pub hash: twox_hash::XxHash64,
}

impl FlatDecodeBuffer {
    pub fn new(window_size: usize) -> Self {
        let cap = window_size.saturating_add(MAX_BLOCK_SIZE as usize + 64);
        FlatDecodeBuffer {
            buf: Vec::with_capacity(cap),
            start: 0,
            dict_content: Vec::new(),
            window_size,
            total_output_counter: 0,
            #[cfg(feature = "hash")]
            hash: twox_hash::XxHash64::with_seed(0),
        }
    }

    pub fn reset(&mut self, window_size: usize) {
        self.window_size = window_size;
        self.buf.clear();
        self.start = 0;
        self.buf.reserve(window_size.saturating_add(MAX_BLOCK_SIZE as usize + 64));
        self.dict_content.clear();
        self.total_output_counter = 0;
        #[cfg(feature = "hash")]
        {
            self.hash = twox_hash::XxHash64::with_seed(0);
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.buf.len().saturating_sub(self.start)
    }

    /// Append literal bytes to the output buffer.
    #[inline(always)]
    pub fn push(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
        self.total_output_counter += data.len() as u64;
    }

    /// Match copy: copy `match_length` bytes from `offset` bytes back in the output.
    /// Uses tiered strategy: memcpy, 8-byte chunks, RLE fill, or copy-doubling.
    #[inline(always)]
    pub fn repeat(&mut self, offset: usize, match_length: usize) -> Result<(), DecodeBufferError> {
        let logical_len = self.len();
        if offset > logical_len {
            return self.repeat_from_dict(offset, match_length);
        }
        if offset == 0 {
            return Err(DecodeBufferError::OffsetTooBig { offset, buf_len: logical_len });
        }

        let old_tail = self.buf.len();
        // Keep a small overshoot (32 bytes) so tiered match-copy routines can safely use
        // wider stores near the end without reallocating or writing out of bounds.
        //
        // Safety: `resize(..., 0)` ensures the whole region is initialized before `fast_copy_match`
        // performs raw pointer writes. We then truncate back to the logical length.
        self.buf.resize(old_tail + match_length + 32, 0);
        unsafe {
            let base = self.buf.as_mut_ptr();
            fast_copy_match(base.add(old_tail - offset), base.add(old_tail), offset, match_length);
        }
        self.buf.truncate(old_tail + match_length);
        self.total_output_counter += match_length as u64;
        Ok(())
    }

    #[cold]
    fn repeat_from_dict(
        &mut self,
        offset: usize,
        match_length: usize,
    ) -> Result<(), DecodeBufferError> {
        if self.total_output_counter <= self.window_size as u64 {
            let bytes_from_dict = offset - self.len();

            if bytes_from_dict > self.dict_content.len() {
                return Err(DecodeBufferError::NotEnoughBytesInDictionary {
                    got: self.dict_content.len(),
                    need: bytes_from_dict,
                });
            }

            if bytes_from_dict < match_length {
                let dict_slice = &self.dict_content[self.dict_content.len() - bytes_from_dict..];
                self.buf.extend_from_slice(dict_slice);
                self.total_output_counter += bytes_from_dict as u64;
                return self.repeat(self.buf.len(), match_length - bytes_from_dict);
            } else {
                let low = self.dict_content.len() - bytes_from_dict;
                let high = low + match_length;
                let dict_slice = &self.dict_content[low..high];
                self.buf.extend_from_slice(dict_slice);
                self.total_output_counter += match_length as u64;
            }
            Ok(())
        } else {
            Err(DecodeBufferError::OffsetTooBig { offset, buf_len: self.len() })
        }
    }

    pub fn can_drain_to_window_size(&self) -> Option<usize> {
        let logical_len = self.len();
        if logical_len > self.window_size {
            Some(logical_len - self.window_size)
        } else {
            None
        }
    }

    pub fn can_drain(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    pub fn reserve_append(&mut self, additional: usize) {
        // Reserve based on the logical window, not a potentially large drained prefix.
        if self.start >= self.window_size {
            self.compact();
        } else {
            self.maybe_compact();
        }
        self.buf.reserve(additional.saturating_add(64));
    }

    pub fn drain_to_window_size(&mut self) -> Option<Vec<u8>> {
        let can_drain = self.can_drain_to_window_size()?;
        #[cfg(feature = "hash")]
        self.hash.write(&self.buf[self.start..self.start + can_drain]);

        let drained: Vec<u8> = self.buf[self.start..self.start + can_drain].to_vec();
        self.start += can_drain;
        self.finish_drain();
        Some(drained)
    }

    pub fn drain_to_window_size_writer(&mut self, mut sink: impl Write) -> Result<usize, Error> {
        match self.can_drain_to_window_size() {
            None => Ok(0),
            Some(can_drain) => {
                #[cfg(feature = "hash")]
                self.hash.write(&self.buf[self.start..self.start + can_drain]);

                write_all_bytes(&mut sink, &self.buf[self.start..self.start + can_drain])?;
                self.start += can_drain;
                self.finish_drain();
                Ok(can_drain)
            }
        }
    }

    pub fn drain(&mut self) -> Vec<u8> {
        #[cfg(feature = "hash")]
        self.hash.write(&self.buf[self.start..]);

        self.compact();
        self.start = 0;
        core::mem::take(&mut self.buf)
    }

    pub fn drain_to_writer(&mut self, mut sink: impl Write) -> Result<usize, Error> {
        let len = self.len();
        if len == 0 {
            return Ok(0);
        }
        #[cfg(feature = "hash")]
        self.hash.write(&self.buf[self.start..]);

        write_all_bytes(&mut sink, &self.buf[self.start..])?;
        self.buf.clear();
        self.start = 0;
        Ok(len)
    }

    pub fn read_all(&mut self, target: &mut [u8]) -> Result<usize, Error> {
        let logical_len = self.len();
        let amount = logical_len.min(target.len());
        if amount > 0 {
            #[cfg(feature = "hash")]
            self.hash.write(&self.buf[self.start..self.start + amount]);

            target[..amount].copy_from_slice(&self.buf[self.start..self.start + amount]);
            self.start += amount;
            self.finish_drain();
        }
        Ok(amount)
    }

    #[inline(always)]
    fn finish_drain(&mut self) {
        if self.start == self.buf.len() {
            self.buf.clear();
            self.start = 0;
            return;
        }
        self.maybe_compact();
    }

    #[inline(always)]
    fn maybe_compact(&mut self) {
        // Compact when the drained prefix dominates, to avoid unbounded growth and
        // keep match sources within the retained window hot in cache.
        if self.start >= 64 * 1024 && self.start >= (self.buf.len() / 2) {
            self.compact();
        }
    }

    fn compact(&mut self) {
        if self.start == 0 {
            return;
        }
        self.buf.copy_within(self.start.., 0);
        self.buf.truncate(self.buf.len() - self.start);
        self.start = 0;
    }
}

impl Read for FlatDecodeBuffer {
    fn read(&mut self, target: &mut [u8]) -> Result<usize, Error> {
        let max_amount = self.can_drain_to_window_size().unwrap_or(0);
        let amount = max_amount.min(target.len());
        if amount > 0 {
            #[cfg(feature = "hash")]
            self.hash.write(&self.buf[self.start..self.start + amount]);

            target[..amount].copy_from_slice(&self.buf[self.start..self.start + amount]);
            self.start += amount;
            self.finish_drain();
        }
        Ok(amount)
    }
}

/// Fast match copy with 3 tiers optimized for common zstd patterns.
///
/// SAFETY:
/// - `src` and `dst` must be within the same allocation.
/// - `dst` must have at least `length` bytes of writable space.
/// - `src` must point to `offset` bytes before `dst` (i.e., `dst - src == offset`).
#[inline(always)]
pub(crate) unsafe fn fast_copy_match(src: *const u8, dst: *mut u8, offset: usize, length: usize) {
    debug_assert!(offset > 0);

    if length <= offset {
        // Tier 0: No overlap at all — straight memcpy (compiler auto-vectorizes).
        // This covers every offset value; the key constraint is length ≤ offset.
        core::ptr::copy_nonoverlapping(src, dst, length);
    } else if offset >= 8 {
        // Tier 2: Mild overlap (8-15 bytes apart).
        // 8-byte unaligned copies are safe: each write starts ≥8 bytes after
        // the read position, so the 8-byte load never touches unwritten bytes.
        let mut i = 0usize;
        while i + 8 <= length {
            let v = (src.add(i) as *const u64).read_unaligned();
            (dst.add(i) as *mut u64).write_unaligned(v);
            i += 8;
        }
        while i < length {
            *dst.add(i) = *src.add(i);
            i += 1;
        }
    } else if offset == 1 {
        // Tier 3a: RLE (offset=1) — most common short offset in zstd.
        // Fill with repeated byte using 8-byte stores.
        let byte = *src;
        let pattern = 0x0101_0101_0101_0101u64.wrapping_mul(byte as u64);
        let mut i = 0usize;
        while i + 8 <= length {
            (dst.add(i) as *mut u64).write_unaligned(pattern);
            i += 8;
        }
        while i < length {
            *dst.add(i) = byte;
            i += 1;
        }
    } else {
        // Tier 3b: Short offset (2-7) — copy-doubling trick.
        //
        // Step 1: Seed first `offset` bytes byte-by-byte from src.
        // (length > offset is guaranteed — Tier 0 handles length ≤ offset.)
        for i in 0..offset {
            *dst.add(i) = *src.add(i);
        }

        // Step 2: Double the written region using non-overlapping copies.
        // Each iteration doubles the amount of established pattern data.
        // copy_nonoverlapping is safe: src [0..copied) doesn't overlap
        // dst [copied..copied+to_copy) since to_copy ≤ copied.
        let mut copied = offset;
        while copied < length {
            let to_copy = copied.min(length - copied);
            core::ptr::copy_nonoverlapping(dst, dst.add(copied), to_copy);
            copied += to_copy;
        }
    }
}

fn write_all_bytes(mut sink: impl Write, buf: &[u8]) -> Result<(), Error> {
    let mut written = 0;
    while written < buf.len() {
        match sink.write(&buf[written..]) {
            Ok(0) => return Ok(()),
            Ok(w) => written += w,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::FlatDecodeBuffer;
    use alloc::vec::Vec;

    #[test]
    fn basic_push_and_repeat() {
        let mut buf = FlatDecodeBuffer::new(1024);
        buf.push(b"ABCDEFGH");
        assert_eq!(buf.len(), 8);
        buf.repeat(8, 8).unwrap();
        assert_eq!(buf.len(), 16);
        assert_eq!(&buf.buf[..16], b"ABCDEFGHABCDEFGH");
    }

    #[test]
    fn short_offset_repeat() {
        let mut buf = FlatDecodeBuffer::new(1024);
        buf.push(b"\x42");
        buf.repeat(1, 100).unwrap();
        assert_eq!(buf.len(), 101);
        assert!(buf.buf.iter().all(|&b| b == 0x42));
    }

    #[test]
    fn offset_3_pattern() {
        let mut buf = FlatDecodeBuffer::new(1024);
        buf.push(b"ABC");
        buf.repeat(3, 30).unwrap();
        assert_eq!(buf.len(), 33);
        for (i, &b) in buf.buf.iter().enumerate() {
            let expected = b"ABC"[i % 3];
            assert_eq!(b, expected, "mismatch at index {}: got {} expected {}", i, b, expected);
        }
    }

    #[test]
    fn offset_4_pattern() {
        let mut buf = FlatDecodeBuffer::new(1024);
        buf.push(b"WXYZ");
        buf.repeat(4, 40).unwrap();
        assert_eq!(buf.len(), 44);
        for (i, &b) in buf.buf.iter().enumerate() {
            let expected = b"WXYZ"[i % 4];
            assert_eq!(b, expected, "mismatch at index {}", i);
        }
    }

    #[test]
    fn large_offset_memcpy() {
        let mut buf = FlatDecodeBuffer::new(4096);
        let data: Vec<u8> = (0..200).collect();
        buf.push(&data);
        buf.repeat(200, 200).unwrap();
        assert_eq!(buf.len(), 400);
        assert_eq!(&buf.buf[..200], &buf.buf[200..400]);
    }

    #[test]
    fn offset_2_pattern() {
        let mut buf = FlatDecodeBuffer::new(1024);
        buf.push(b"AB");
        buf.repeat(2, 20).unwrap();
        assert_eq!(buf.len(), 22);
        for (i, &b) in buf.buf.iter().enumerate() {
            let expected = b"AB"[i % 2];
            assert_eq!(b, expected, "mismatch at index {}", i);
        }
    }

    #[test]
    fn zero_offset_error() {
        let mut buf = FlatDecodeBuffer::new(1024);
        buf.push(b"test");
        assert!(buf.repeat(0, 10).is_err());
    }

    #[test]
    fn drain_and_collect() {
        let mut buf = FlatDecodeBuffer::new(100);
        buf.push(b"hello world");
        let drained = buf.drain();
        assert_eq!(&drained, b"hello world");
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn offset_5_pattern() {
        let mut buf = FlatDecodeBuffer::new(1024);
        buf.push(b"ABCDE");
        buf.repeat(5, 50).unwrap();
        assert_eq!(buf.len(), 55);
        for (i, &b) in buf.buf.iter().enumerate() {
            let expected = b"ABCDE"[i % 5];
            assert_eq!(b, expected, "mismatch at index {}", i);
        }
    }

    #[test]
    fn offset_6_pattern() {
        let mut buf = FlatDecodeBuffer::new(1024);
        buf.push(b"ABCDEF");
        buf.repeat(6, 60).unwrap();
        assert_eq!(buf.len(), 66);
        for (i, &b) in buf.buf.iter().enumerate() {
            let expected = b"ABCDEF"[i % 6];
            assert_eq!(b, expected, "mismatch at index {}", i);
        }
    }

    #[test]
    fn offset_7_pattern() {
        let mut buf = FlatDecodeBuffer::new(1024);
        buf.push(b"ABCDEFG");
        buf.repeat(7, 70).unwrap();
        assert_eq!(buf.len(), 77);
        for (i, &b) in buf.buf.iter().enumerate() {
            let expected = b"ABCDEFG"[i % 7];
            assert_eq!(b, expected, "mismatch at index {}", i);
        }
    }

    #[test]
    fn large_offset_overlap_regression() {
        // Regression: copy_nonoverlapping is UB when length > offset.
        // offset=20 (>= 16), length=100 — must produce repeating 20-byte pattern.
        let mut buf = FlatDecodeBuffer::new(4096);
        let seed: Vec<u8> = (0..20).collect();
        buf.push(&seed);
        buf.repeat(20, 100).unwrap();
        assert_eq!(buf.len(), 120);
        for (i, &b) in buf.buf.iter().enumerate() {
            let expected = (i % 20) as u8;
            assert_eq!(b, expected, "mismatch at index {}: got {} expected {}", i, b, expected);
        }
    }

    #[test]
    fn offset_16_exact_boundary() {
        let mut buf = FlatDecodeBuffer::new(4096);
        let seed: Vec<u8> = (0..16).collect();
        buf.push(&seed);
        buf.repeat(16, 64).unwrap();
        assert_eq!(buf.len(), 80);
        for (i, &b) in buf.buf.iter().enumerate() {
            let expected = (i % 16) as u8;
            assert_eq!(b, expected, "mismatch at index {}", i);
        }
    }

    #[test]
    fn offset_9_mid_range() {
        let mut buf = FlatDecodeBuffer::new(1024);
        buf.push(b"ABCDEFGHI");
        buf.repeat(9, 90).unwrap();
        assert_eq!(buf.len(), 99);
        for (i, &b) in buf.buf.iter().enumerate() {
            let expected = b"ABCDEFGHI"[i % 9];
            assert_eq!(b, expected, "mismatch at index {}", i);
        }
    }
}
