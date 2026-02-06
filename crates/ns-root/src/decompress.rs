//! ROOT compression block decompression (ZL = zlib, L4 = LZ4, ZS = ZSTD).
//!
//! ROOT writes compressed data as one or more 9-byte-header blocks:
//! ```text
//! bytes 0-1:  algorithm tag ("ZL", "XZ", "L4", "ZS")
//! byte  2:    method (ignored)
//! bytes 3-5:  compressed size   (3-byte little-endian)
//! bytes 6-8:  uncompressed size (3-byte little-endian)
//! ```
//! The compressed payload immediately follows the 9-byte header.

use crate::error::{Result, RootError};

/// Decompress ROOT-compressed data into `expected_len` bytes.
pub fn decompress(src: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(expected_len);
    let mut offset = 0;

    while out.len() < expected_len && offset + 9 <= src.len() {
        let tag = &src[offset..offset + 2];
        // byte 2 is method, skip
        let c_size = read_le24(&src[offset + 3..offset + 6]);
        let u_size = read_le24(&src[offset + 6..offset + 9]);
        offset += 9;

        let end = offset + c_size;
        if end > src.len() {
            return Err(RootError::Decompression(format!(
                "compressed block claims {} bytes but only {} remain",
                c_size,
                src.len() - offset
            )));
        }

        let compressed = &src[offset..end];

        let decompressed = match tag {
            b"ZL" => decompress_zlib(compressed, u_size)?,
            b"L4" => decompress_lz4(compressed, u_size)?,
            _ => {
                return Err(RootError::Decompression(format!(
                    "unsupported compression algorithm: {:?}",
                    std::str::from_utf8(tag).unwrap_or("??")
                )));
            }
        };

        if decompressed.len() != u_size {
            return Err(RootError::Decompression(format!(
                "expected {} uncompressed bytes, got {}",
                u_size,
                decompressed.len()
            )));
        }

        out.extend_from_slice(&decompressed);
        offset = end;
    }

    if out.len() != expected_len {
        return Err(RootError::Decompression(format!(
            "total decompressed length {} != expected {}",
            out.len(),
            expected_len
        )));
    }

    Ok(out)
}

fn decompress_zlib(data: &[u8], expected: usize) -> Result<Vec<u8>> {
    use flate2::read::ZlibDecoder;
    use std::io::Read;

    let mut decoder = ZlibDecoder::new(data);
    let mut out = Vec::with_capacity(expected);
    decoder.read_to_end(&mut out).map_err(|e| {
        RootError::Decompression(format!("zlib: {}", e))
    })?;
    Ok(out)
}

fn decompress_lz4(data: &[u8], expected: usize) -> Result<Vec<u8>> {
    // ROOT LZ4 blocks have an extra 8-byte checksum header before the LZ4 payload.
    // The first 8 bytes are an xxhash64 of the uncompressed data (we skip verification).
    if data.len() < 8 {
        return Err(RootError::Decompression(
            "LZ4 block too small for checksum header".into(),
        ));
    }
    let lz4_data = &data[8..];
    lz4_flex::decompress(lz4_data, expected).map_err(|e| {
        RootError::Decompression(format!("lz4: {}", e))
    })
}

/// Read a 3-byte little-endian unsigned integer.
fn read_le24(b: &[u8]) -> usize {
    b[0] as usize | ((b[1] as usize) << 8) | ((b[2] as usize) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn le24_round_trip() {
        assert_eq!(read_le24(&[0x10, 0x00, 0x00]), 16);
        assert_eq!(read_le24(&[0xff, 0xff, 0xff]), 0xFF_FFFF);
        assert_eq!(read_le24(&[0x00, 0x01, 0x00]), 256);
    }

    #[test]
    fn zlib_round_trip() {
        use flate2::write::ZlibEncoder;
        use flate2::Compression;
        use std::io::Write;

        let original = b"Hello ROOT compression world! Repeated data: AAAAAAAAAA";
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Build a ROOT-style block: ZL header + compressed data
        let mut block = Vec::new();
        block.extend_from_slice(b"ZL"); // tag
        block.push(0x08); // method (zlib default)
        // 3-byte LE compressed size
        let c_len = compressed.len();
        block.push((c_len & 0xFF) as u8);
        block.push(((c_len >> 8) & 0xFF) as u8);
        block.push(((c_len >> 16) & 0xFF) as u8);
        // 3-byte LE uncompressed size
        let u_len = original.len();
        block.push((u_len & 0xFF) as u8);
        block.push(((u_len >> 8) & 0xFF) as u8);
        block.push(((u_len >> 16) & 0xFF) as u8);
        block.extend_from_slice(&compressed);

        let result = decompress(&block, original.len()).unwrap();
        assert_eq!(result, original);
    }
}
