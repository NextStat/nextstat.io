//! Binary reader for ROOT's big-endian serialization format.

use crate::error::{Result, RootError};

/// A cursor-based reader over a byte slice, using ROOT's big-endian conventions.
pub struct RBuffer<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> RBuffer<'a> {
    /// Create a new reader over the given bytes.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Current read position.
    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Total length of underlying buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Remaining bytes from current position.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Set read position absolutely.
    pub fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// Skip `n` bytes forward.
    pub fn skip(&mut self, n: usize) -> Result<()> {
        self.ensure(n)?;
        self.pos += n;
        Ok(())
    }

    /// Read a sub-slice of `n` bytes, advancing the cursor.
    pub fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.ensure(n)?;
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// Read a single byte.
    pub fn read_u8(&mut self) -> Result<u8> {
        self.ensure(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    /// Read a big-endian u16.
    pub fn read_u16(&mut self) -> Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    /// Read a big-endian i16.
    pub fn read_i16(&mut self) -> Result<i16> {
        let b = self.read_bytes(2)?;
        Ok(i16::from_be_bytes([b[0], b[1]]))
    }

    /// Read a big-endian u32.
    pub fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Read a big-endian i32.
    pub fn read_i32(&mut self) -> Result<i32> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Read a big-endian u64.
    pub fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Read a big-endian i64.
    #[allow(dead_code)]
    pub fn read_i64(&mut self) -> Result<i64> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Read a big-endian f32.
    pub fn read_f32(&mut self) -> Result<f32> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Read a big-endian f64.
    pub fn read_f64(&mut self) -> Result<f64> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Read a ROOT-encoded string.
    ///
    /// Format: length byte (if < 255), or 255 + u32 length, then UTF-8 bytes.
    pub fn read_string(&mut self) -> Result<String> {
        let first = self.read_u8()?;
        let len = if first == 255 {
            self.read_u32()? as usize
        } else {
            first as usize
        };
        if len == 0 {
            return Ok(String::new());
        }
        let bytes = self.read_bytes(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    /// Read a ROOT streamer version header.
    ///
    /// Returns `(version, end_pos)` where `end_pos` is the absolute buffer
    /// position where this streamed object ends (`None` if no byte-count header).
    ///
    /// ROOT uses `kByteCountMask = 0x4000_0000` on the first u32 to signal that
    /// a byte count is present. The byte count spans from right after the u32
    /// to the end of the object (i.e. it includes the version u16).
    pub fn read_version(&mut self) -> Result<(u16, Option<usize>)> {
        let start = self.pos;
        let raw = self.read_u32()?;
        if raw & 0x4000_0000 != 0 {
            let byte_count = (raw & !0x4000_0000) as usize;
            let version = self.read_u16()?;
            // byte_count counts from right after the u32
            let end_pos = start + 4 + byte_count;
            Ok((version, Some(end_pos)))
        } else {
            // No byte count — first two bytes are the version.
            let version = (raw >> 16) as u16;
            self.pos -= 2;
            Ok((version, None))
        }
    }

    /// Read a `TObject` header: fUniqueID (u32) + fBits (u32).
    pub fn read_tobject(&mut self) -> Result<(u32, u32)> {
        let _ver = self.read_u16()?; // TObject version
        let unique_id = self.read_u32()?;
        let mut bits = self.read_u32()?;
        bits |= 0x0100_0000; // kIsOnHeap
        if bits & 0x0800_0000 != 0 {
            // kIsReferenced — skip 2-byte pidf
            self.skip(2)?;
        }
        Ok((unique_id, bits))
    }

    /// Read a `TNamed`: TObject + fName + fTitle.
    pub fn read_tnamed(&mut self) -> Result<(String, String)> {
        let (_ver, _end) = self.read_version()?;
        self.read_tobject()?;
        let name = self.read_string()?;
        let title = self.read_string()?;
        Ok((name, title))
    }

    /// Read `N` big-endian f64 values into a Vec.
    pub fn read_array_f64(&mut self, n: usize) -> Result<Vec<f64>> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.read_f64()?);
        }
        Ok(out)
    }

    /// Read `N` big-endian f32 values into a Vec.
    pub fn read_array_f32(&mut self, n: usize) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.read_f32()?);
        }
        Ok(out)
    }

    // ── internal ────────────────────────────────────────────────

    fn ensure(&self, n: usize) -> Result<()> {
        if self.pos + n > self.data.len() {
            return Err(RootError::BufferUnderflow {
                offset: self.pos,
                need: n,
                have: self.data.len().saturating_sub(self.pos),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_primitives() {
        // u32 big-endian: 0x01020304 = 16909060
        let data = [0x01, 0x02, 0x03, 0x04, 0x40, 0x09, 0x21, 0xfb, 0x54, 0x44, 0x2d, 0x18];
        let mut r = RBuffer::new(&data);
        assert_eq!(r.read_u32().unwrap(), 0x0102_0304);
        assert!((r.read_f64().unwrap() - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn read_string_short() {
        let data = [3, b'a', b'b', b'c'];
        let mut r = RBuffer::new(&data);
        assert_eq!(r.read_string().unwrap(), "abc");
    }

    #[test]
    fn read_version_with_bytecount() {
        // byte_count = 0x00000010 (16), version = 3
        let mut data = Vec::new();
        data.extend_from_slice(&0x4000_0010u32.to_be_bytes());
        data.extend_from_slice(&3u16.to_be_bytes());
        data.extend_from_slice(&[0u8; 20]); // padding for end_pos
        let mut r = RBuffer::new(&data);
        let (ver, end) = r.read_version().unwrap();
        assert_eq!(ver, 3);
        // end_pos = start(0) + 4 + 16 = 20
        assert_eq!(end, Some(20));
    }

    #[test]
    fn read_version_without_bytecount() {
        // version = 5, no byte count
        let mut data = Vec::new();
        data.extend_from_slice(&5u16.to_be_bytes());
        data.extend_from_slice(&[0x00, 0x00]); // padding
        let mut r = RBuffer::new(&data);
        let (ver, end) = r.read_version().unwrap();
        assert_eq!(ver, 5);
        assert!(end.is_none());
    }
}
