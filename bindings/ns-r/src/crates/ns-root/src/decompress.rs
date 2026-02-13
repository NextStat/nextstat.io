//! ROOT compression block decompression (ZL = zlib, L4 = LZ4, ZS = ZSTD, XZ = LZMA).
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
#[cfg(not(target_arch = "wasm32"))]
use std::io::Cursor;

std::thread_local! {
    static ZSTD_DECODER: std::cell::RefCell<ns_zstd::decoding::FrameDecoder> =
        std::cell::RefCell::new(ns_zstd::decoding::FrameDecoder::new());
}

#[cfg(not(target_arch = "wasm32"))]
std::thread_local! {
    // Native fast-path: C libzstd via zstd-safe.
    static LIBZSTD_DCTX: std::cell::RefCell<zstd_safe::DCtx<'static>> =
        std::cell::RefCell::new(zstd_safe::DCtx::default());
}

// Hard safety limit for untrusted ROOT inputs.
//
// ROOT baskets/blocks are typically O(MiB). If a file claims a gigantic
// uncompressed size, pre-allocations here can become a DoS/OOM vector.
const MAX_DECOMPRESSED_BYTES: usize = 256 * 1024 * 1024; // 256 MiB

/// Decompress ROOT-compressed data into `expected_len` bytes.
pub fn decompress(src: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(expected_len);
    decompress_into(src, expected_len, &mut out)?;
    Ok(out)
}

/// Decompress ROOT-compressed data into a caller-provided buffer.
///
/// This lets callers amortize allocations by reusing `out` across many blocks/baskets.
pub fn decompress_into(src: &[u8], expected_len: usize, out: &mut Vec<u8>) -> Result<()> {
    if expected_len > MAX_DECOMPRESSED_BYTES {
        return Err(RootError::Decompression(format!(
            "refusing to decompress {expected_len} bytes (max {MAX_DECOMPRESSED_BYTES})"
        )));
    }

    out.clear();
    out.reserve(expected_len);

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

        let remaining = expected_len - out.len();
        if u_size > remaining {
            return Err(RootError::Decompression(format!(
                "uncompressed block size {} exceeds remaining expected {}",
                u_size, remaining
            )));
        }

        let compressed = &src[offset..end];
        match tag {
            b"ZL" => {
                let decompressed = decompress_zlib(compressed, u_size)?;
                if decompressed.len() != u_size {
                    return Err(RootError::Decompression(format!(
                        "expected {} uncompressed bytes, got {}",
                        u_size,
                        decompressed.len()
                    )));
                }
                out.extend_from_slice(&decompressed);
            }
            b"L4" => {
                let decompressed = decompress_lz4(compressed, u_size)?;
                if decompressed.len() != u_size {
                    return Err(RootError::Decompression(format!(
                        "expected {} uncompressed bytes, got {}",
                        u_size,
                        decompressed.len()
                    )));
                }
                out.extend_from_slice(&decompressed);
            }
            b"XZ" => {
                let decompressed = decompress_xz(compressed, u_size)?;
                if decompressed.len() != u_size {
                    return Err(RootError::Decompression(format!(
                        "expected {} uncompressed bytes, got {}",
                        u_size,
                        decompressed.len()
                    )));
                }
                out.extend_from_slice(&decompressed);
            }
            b"ZS" => {
                decompress_zstd_append(compressed, u_size, out)?;
            }
            _ => {
                return Err(RootError::Decompression(format!(
                    "unsupported compression algorithm: {:?}",
                    std::str::from_utf8(tag).unwrap_or("??")
                )));
            }
        }

        offset = end;
    }

    if out.len() != expected_len {
        return Err(RootError::Decompression(format!(
            "total decompressed length {} != expected {}",
            out.len(),
            expected_len
        )));
    }

    Ok(())
}

fn decompress_zlib(data: &[u8], expected: usize) -> Result<Vec<u8>> {
    use flate2::read::ZlibDecoder;
    use std::io::Read;

    let mut decoder = ZlibDecoder::new(data);
    let mut out = Vec::with_capacity(expected);
    decoder.read_to_end(&mut out).map_err(|e| RootError::Decompression(format!("zlib: {}", e)))?;
    Ok(out)
}

fn decompress_lz4(data: &[u8], expected: usize) -> Result<Vec<u8>> {
    // ROOT LZ4 blocks have an extra 8-byte checksum header before the LZ4 payload.
    // The first 8 bytes are an xxhash64 of the uncompressed data (we skip verification).
    if data.len() < 8 {
        return Err(RootError::Decompression("LZ4 block too small for checksum header".into()));
    }
    let lz4_data = &data[8..];
    lz4_flex::decompress(lz4_data, expected)
        .map_err(|e| RootError::Decompression(format!("lz4: {}", e)))
}

#[cfg(not(target_arch = "wasm32"))]
fn decompress_zstd_append(data: &[u8], expected: usize, out: &mut Vec<u8>) -> Result<()> {
    if expected == 0 {
        return Ok(());
    }
    let start = out.len();
    out.reserve(expected.max(1));
    let mut cursor = Cursor::new(out);
    cursor.set_position(start as u64);
    let bytes_written =
        LIBZSTD_DCTX
            .with(|cell| cell.borrow_mut().decompress(&mut cursor, data))
            .map_err(|code| RootError::Decompression(format!("zstd(libzstd): error={}", code)))?;
    if bytes_written != expected {
        return Err(RootError::Decompression(format!(
            "zstd(libzstd): expected {} uncompressed bytes, got {}",
            expected, bytes_written
        )));
    }
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn decompress_zstd_append(data: &[u8], expected: usize, out: &mut Vec<u8>) -> Result<()> {
    let start = out.len();
    out.resize(start + expected, 0);
    let dst = &mut out[start..];

    let bytes_written = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ZSTD_DECODER.with(|cell| {
            let mut dec = cell.borrow_mut();
            dec.decode_all(data, dst)
        })
    })) {
        Ok(Ok(n)) => n,
        Ok(Err(e)) => {
            ZSTD_DECODER.with(|cell| {
                *cell.borrow_mut() = ns_zstd::decoding::FrameDecoder::new();
            });
            return Err(RootError::Decompression(format!("zstd: {}", e)));
        }
        Err(payload) => {
            ZSTD_DECODER.with(|cell| {
                *cell.borrow_mut() = ns_zstd::decoding::FrameDecoder::new();
            });
            let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                *s
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.as_str()
            } else {
                "unknown panic payload"
            };
            return Err(RootError::Decompression(format!("zstd: decoder panicked: {}", msg)));
        }
    };

    if bytes_written != expected {
        ZSTD_DECODER.with(|cell| {
            *cell.borrow_mut() = ns_zstd::decoding::FrameDecoder::new();
        });
        return Err(RootError::Decompression(format!(
            "zstd: expected {} uncompressed bytes, got {}",
            expected, bytes_written
        )));
    }

    Ok(())
}

fn decompress_xz(data: &[u8], expected: usize) -> Result<Vec<u8>> {
    let mut input = std::io::BufReader::new(data);
    let mut out = Vec::with_capacity(expected);
    lzma_rs::xz_decompress(&mut input, &mut out)
        .map_err(|e| RootError::Decompression(format!("xz: {}", e)))?;
    Ok(out)
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
        use flate2::Compression;
        use flate2::write::ZlibEncoder;
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

    /// Helper to build a ROOT-style compression block from tag, compressed data, and original len.
    fn make_root_block(tag: &[u8; 2], method: u8, compressed: &[u8], u_len: usize) -> Vec<u8> {
        let mut block = Vec::new();
        block.extend_from_slice(tag);
        block.push(method);
        let c_len = compressed.len();
        block.push((c_len & 0xFF) as u8);
        block.push(((c_len >> 8) & 0xFF) as u8);
        block.push(((c_len >> 16) & 0xFF) as u8);
        block.push((u_len & 0xFF) as u8);
        block.push(((u_len >> 8) & 0xFF) as u8);
        block.push(((u_len >> 16) & 0xFF) as u8);
        block.extend_from_slice(compressed);
        block
    }

    #[test]
    fn zstd_round_trip() {
        let original = b"Hello ROOT ZSTD compression! Repeated data: BBBBBBBBBB";
        let compressed = ns_zstd::encoding::compress_to_vec(
            &original[..],
            ns_zstd::encoding::CompressionLevel::Fastest,
        );
        let block = make_root_block(b"ZS", 0x04, &compressed, original.len());

        let result = decompress(&block, original.len()).unwrap();
        assert_eq!(result, &original[..]);
    }

    #[test]
    fn zstd_malformed_returns_error() {
        let compressed = [0xFFu8; 16];
        let block = make_root_block(b"ZS", 0x04, &compressed, 64);
        assert!(decompress(&block, 64).is_err());
    }

    #[test]
    fn refuses_block_larger_than_total_expected_len() {
        // u_size=1024, but total expected_len=1 => must fail before any algorithm decoder allocates.
        let block = make_root_block(b"ZS", 0x04, &[], 1024);
        assert!(decompress(&block, 1).is_err());
    }

    #[test]
    fn refuses_block_larger_than_remaining_expected_len() {
        // First block consumes 8 bytes, second claims 8 more but only 2 remain.
        let b1 = make_root_block(b"ZL", 0x08, &[], 8);
        let b2 = make_root_block(b"ZL", 0x08, &[], 8);
        let mut src = Vec::new();
        src.extend_from_slice(&b1);
        src.extend_from_slice(&b2);
        assert!(decompress(&src, 10).is_err());
    }

    #[test]
    fn refuses_huge_expected_len() {
        let err = decompress(&[], MAX_DECOMPRESSED_BYTES + 1).unwrap_err();
        match err {
            RootError::Decompression(msg) => {
                assert!(msg.contains("refusing to decompress"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn xz_round_trip() {
        let original = b"Hello ROOT XZ compression! Repeated data: CCCCCCCCCC";
        let mut compressed = Vec::new();
        lzma_rs::xz_compress(&mut std::io::BufReader::new(&original[..]), &mut compressed).unwrap();
        let block = make_root_block(b"XZ", 0x05, &compressed, original.len());

        let result = decompress(&block, original.len()).unwrap();
        assert_eq!(result, &original[..]);
    }
}
