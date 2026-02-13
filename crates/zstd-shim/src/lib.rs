//! Pure-Rust shim for the `zstd` crate API.
//!
//! This is intentionally minimal: it implements the parts of the `zstd` API that
//! Apache Arrow's `parquet` crate uses (`zstd::Encoder` / `zstd::Decoder`).
//!
//! Implementation backend:
//! - wasm32 / `pure-rust`: `ns-zstd` (pure Rust)
//! - native (default): `zstd-safe` (libzstd C backend)

use std::io;
use std::ops::RangeInclusive;

#[cfg(all(not(target_arch = "wasm32"), not(feature = "pure-rust")))]
use zstd_safe as c_zstd;

/// Default compression level in upstream zstd (currently 3).
pub const DEFAULT_COMPRESSION_LEVEL: i32 = 3;

/// The accepted range of compression levels.
///
/// Upstream zstd accepts 1..=22. We keep the same range for compatibility.
pub fn compression_level_range() -> RangeInclusive<i32> {
    1..=22
}

#[cfg(any(target_arch = "wasm32", feature = "pure-rust"))]
fn map_level(level: i32) -> ns_zstd::encoding::CompressionLevel {
    // Upstream zstd levels: 1 (fastest) .. 22 (best), 0 = default (3).
    //
    // ns-zstd currently implements two compression tiers:
    //   Fastest  — lightweight match generation, ~level 1
    //   Default  — full match generation with HC chain, ~level 3
    //
    // Levels 2-22 all map to Default until ns-zstd implements Better/Best tiers.
    // Uncompressed is not exposed (not useful for Parquet/Arrow consumers).
    match level {
        0 => ns_zstd::encoding::CompressionLevel::Default,
        i32::MIN..=1 => ns_zstd::encoding::CompressionLevel::Fastest,
        2.. => ns_zstd::encoding::CompressionLevel::Default,
    }
}

fn map_decode_err(e: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
}

pub mod stream {
    pub mod read {
        use std::io::{self, Read};

        #[cfg(all(not(target_arch = "wasm32"), not(feature = "pure-rust")))]
        pub struct Decoder<R: Read> {
            reader: R,
            dctx: super::super::c_zstd::DCtx<'static>,
            in_storage: Vec<u8>,
            in_pos: usize,
            in_len: usize,
            eof: bool,
            finished: bool,
        }

        #[cfg(all(not(target_arch = "wasm32"), not(feature = "pure-rust")))]
        impl<R: Read> Decoder<R> {
            /// Creates a new decoder.
            pub fn new(reader: R) -> io::Result<Self> {
                let in_size = super::super::c_zstd::DStream::in_size().max(1);
                Ok(Self {
                    reader,
                    dctx: super::super::c_zstd::DCtx::default(),
                    in_storage: vec![0u8; in_size],
                    in_pos: 0,
                    in_len: 0,
                    eof: false,
                    finished: false,
                })
            }

            /// Return the inner reader.
            pub fn finish(self) -> R {
                self.reader
            }

            /// Acquire a reference to the underlying reader.
            pub fn get_ref(&self) -> &R {
                &self.reader
            }

            /// Acquire a mutable reference to the underlying reader.
            pub fn get_mut(&mut self) -> &mut R {
                &mut self.reader
            }
        }

        #[cfg(all(not(target_arch = "wasm32"), not(feature = "pure-rust")))]
        impl<R: Read> Read for Decoder<R> {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                if self.finished || buf.is_empty() {
                    return Ok(0);
                }

                loop {
                    // Refill input if needed.
                    if self.in_pos == self.in_len && !self.eof {
                        let n = self.reader.read(self.in_storage.as_mut_slice())?;
                        self.in_pos = 0;
                        self.in_len = n;
                        if n == 0 {
                            self.eof = true;
                        }
                    }

                    let input = &self.in_storage[self.in_pos..self.in_len];
                    let mut in_buf = super::super::c_zstd::InBuffer::around(input);
                    let mut out_buf = super::super::c_zstd::OutBuffer::around(buf);

                    let hint = self
                        .dctx
                        .decompress_stream(&mut out_buf, &mut in_buf)
                        .map_err(super::super::map_decode_err)?;

                    self.in_pos += in_buf.pos();

                    let wrote = out_buf.pos();
                    if hint == 0 {
                        self.finished = true;
                    }
                    if wrote > 0 {
                        return Ok(wrote);
                    }
                    if self.finished {
                        return Ok(0);
                    }

                    // Need more input but we hit EOF: corrupted frame / truncated input.
                    if self.eof && self.in_pos == self.in_len {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "zstd shim: truncated frame",
                        ));
                    }
                }
            }
        }

        #[cfg(any(target_arch = "wasm32", feature = "pure-rust"))]
        /// Pull-based decoder implementing [`std::io::Read`].
        ///
        /// This shim decodes on demand using `ns-zstd`'s streaming decoder.
        pub struct Decoder<R: Read> {
            inner: ns_zstd::decoding::StreamingDecoder<R, ns_zstd::decoding::FrameDecoder>,
        }

        #[cfg(any(target_arch = "wasm32", feature = "pure-rust"))]
        impl<R: Read> Decoder<R> {
            /// Creates a new decoder.
            pub fn new(reader: R) -> io::Result<Self> {
                let inner = ns_zstd::decoding::StreamingDecoder::new(reader)
                    .map_err(crate::map_decode_err)?;
                Ok(Self { inner })
            }

            /// Return the inner reader.
            pub fn finish(self) -> R {
                self.inner.into_inner()
            }

            /// Acquire a reference to the underlying reader.
            pub fn get_ref(&self) -> &R {
                self.inner.get_ref()
            }

            /// Acquire a mutable reference to the underlying reader.
            pub fn get_mut(&mut self) -> &mut R {
                self.inner.get_mut()
            }
        }

        #[cfg(any(target_arch = "wasm32", feature = "pure-rust"))]
        impl<R: Read> Read for Decoder<R> {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                self.inner.read(buf)
            }
        }
    }

    pub mod write {
        use std::io::{self, Write};

        #[cfg(all(not(target_arch = "wasm32"), not(feature = "pure-rust")))]
        pub struct Encoder<W: Write> {
            writer: W,
            cctx: super::super::c_zstd::CCtx<'static>,
            out_storage: Vec<u8>,
        }

        #[cfg(all(not(target_arch = "wasm32"), not(feature = "pure-rust")))]
        impl<W: Write> Encoder<W> {
            /// Creates a new encoder.
            pub fn new(writer: W, level: i32) -> io::Result<Self> {
                let level =
                    if level == 0 { super::super::DEFAULT_COMPRESSION_LEVEL } else { level };
                let mut cctx = super::super::c_zstd::CCtx::default();
                cctx.set_parameter(super::super::c_zstd::CParameter::CompressionLevel(level))
                    .map_err(super::super::map_decode_err)?;
                let out_size = super::super::c_zstd::CCtx::out_size().max(1);
                Ok(Self { writer, cctx, out_storage: vec![0u8; out_size] })
            }

            /// Finish the stream and return the inner writer.
            pub fn finish(mut self) -> io::Result<W> {
                // End the frame and flush remaining bytes.
                loop {
                    let mut out =
                        super::super::c_zstd::OutBuffer::around(self.out_storage.as_mut_slice());
                    let remaining =
                        self.cctx.end_stream(&mut out).map_err(super::super::map_decode_err)?;
                    let produced = out.as_slice();
                    if !produced.is_empty() {
                        self.writer.write_all(produced)?;
                    }
                    if remaining == 0 {
                        break;
                    }
                }
                Ok(self.writer)
            }

            pub fn get_ref(&self) -> &W {
                &self.writer
            }

            pub fn get_mut(&mut self) -> &mut W {
                &mut self.writer
            }
        }

        #[cfg(all(not(target_arch = "wasm32"), not(feature = "pure-rust")))]
        impl<W: Write> Write for Encoder<W> {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                if buf.is_empty() {
                    return Ok(0);
                }

                let mut in_buf = super::super::c_zstd::InBuffer::around(buf);
                while in_buf.pos() < buf.len() {
                    let mut out =
                        super::super::c_zstd::OutBuffer::around(self.out_storage.as_mut_slice());
                    self.cctx
                        .compress_stream(&mut out, &mut in_buf)
                        .map_err(super::super::map_decode_err)?;
                    let produced = out.as_slice();
                    if !produced.is_empty() {
                        self.writer.write_all(produced)?;
                    }
                }

                Ok(buf.len())
            }

            fn flush(&mut self) -> io::Result<()> {
                // Flush intermediate buffers to match `zstd` crate semantics.
                loop {
                    let mut out =
                        super::super::c_zstd::OutBuffer::around(self.out_storage.as_mut_slice());
                    let remaining =
                        self.cctx.flush_stream(&mut out).map_err(super::super::map_decode_err)?;
                    let produced = out.as_slice();
                    if !produced.is_empty() {
                        self.writer.write_all(produced)?;
                    }
                    if remaining == 0 {
                        break;
                    }
                }
                self.writer.flush()
            }
        }

        #[cfg(any(target_arch = "wasm32", feature = "pure-rust"))]
        /// Push-based encoder implementing [`std::io::Write`].
        ///
        /// This shim encodes incrementally into a single zstd frame, writing blocks to the
        /// underlying writer as they become available.
        pub struct Encoder<W: Write> {
            inner: ns_zstd::encoding::StreamingEncoder<W>,
        }

        #[cfg(any(target_arch = "wasm32", feature = "pure-rust"))]
        impl<W: Write> Encoder<W> {
            /// Creates a new encoder.
            ///
            /// `level`: zstd compression level (1-22). A level of `0` uses the default (3).
            pub fn new(writer: W, level: i32) -> io::Result<Self> {
                Ok(Self {
                    inner: ns_zstd::encoding::StreamingEncoder::new(
                        writer,
                        crate::map_level(level),
                    ),
                })
            }

            /// Finish the stream and return the inner writer.
            pub fn finish(self) -> io::Result<W> {
                self.inner.finish()
            }

            /// Acquire a reference to the underlying writer.
            pub fn get_ref(&self) -> &W {
                self.inner.get_ref()
            }

            /// Acquire a mutable reference to the underlying writer.
            pub fn get_mut(&mut self) -> &mut W {
                self.inner.get_mut()
            }
        }

        #[cfg(any(target_arch = "wasm32", feature = "pure-rust"))]
        impl<W: Write> Write for Encoder<W> {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.inner.write(buf)
            }

            fn flush(&mut self) -> io::Result<()> {
                self.inner.flush()
            }
        }
    }

    pub use read::Decoder;
    pub use write::Encoder;
}

pub use stream::read::Decoder;
pub use stream::write::Encoder;

pub mod bulk {
    use std::io;

    /// Compress a block of data and return a zstd frame.
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "pure-rust")))]
    pub fn compress(data: &[u8], level: i32) -> io::Result<Vec<u8>> {
        let level = if level == 0 { crate::DEFAULT_COMPRESSION_LEVEL } else { level };
        let bound = crate::c_zstd::compress_bound(data.len());
        let mut out = Vec::with_capacity(bound.max(1));
        let mut cctx = crate::c_zstd::CCtx::default();
        cctx.compress(&mut out, data, level).map_err(crate::map_decode_err)?;
        Ok(out)
    }

    /// Compress a block of data and return a zstd frame.
    #[cfg(any(target_arch = "wasm32", feature = "pure-rust"))]
    pub fn compress(data: &[u8], level: i32) -> io::Result<Vec<u8>> {
        Ok(ns_zstd::encoding::compress_to_vec(data, crate::map_level(level)))
    }

    /// Decompress a zstd frame, returning at most `capacity` bytes.
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "pure-rust")))]
    pub fn decompress(data: &[u8], capacity: usize) -> io::Result<Vec<u8>> {
        let mut out = Vec::with_capacity(capacity.max(1));
        let mut dctx = crate::c_zstd::DCtx::default();
        dctx.decompress(&mut out, data).map_err(crate::map_decode_err)?;
        Ok(out)
    }

    /// Decompress a zstd frame, returning at most `capacity` bytes.
    #[cfg(any(target_arch = "wasm32", feature = "pure-rust"))]
    pub fn decompress(data: &[u8], capacity: usize) -> io::Result<Vec<u8>> {
        let mut out = Vec::with_capacity(capacity.max(1));
        let mut dec = ns_zstd::decoding::FrameDecoder::new();
        dec.decode_all_to_vec(data, &mut out).map_err(crate::map_decode_err)?;
        Ok(out)
    }

    /// Compress to a pre-allocated buffer.
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "pure-rust")))]
    pub fn compress_to_buffer(
        source: &[u8],
        destination: &mut [u8],
        level: i32,
    ) -> io::Result<usize> {
        let level = if level == 0 { crate::DEFAULT_COMPRESSION_LEVEL } else { level };
        let mut cctx = crate::c_zstd::CCtx::default();
        let n = cctx.compress(destination, source, level).map_err(crate::map_decode_err)?;
        Ok(n)
    }

    /// Compress to a pre-allocated buffer.
    #[cfg(any(target_arch = "wasm32", feature = "pure-rust"))]
    pub fn compress_to_buffer(
        source: &[u8],
        destination: &mut [u8],
        level: i32,
    ) -> io::Result<usize> {
        let compressed = compress(source, level)?;
        if compressed.len() > destination.len() {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "zstd shim: destination buffer too small",
            ));
        }
        destination[..compressed.len()].copy_from_slice(&compressed);
        Ok(compressed.len())
    }

    /// Decompress to a pre-allocated buffer.
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "pure-rust")))]
    pub fn decompress_to_buffer(source: &[u8], destination: &mut [u8]) -> io::Result<usize> {
        let mut dctx = crate::c_zstd::DCtx::default();
        let n = dctx.decompress(destination, source).map_err(crate::map_decode_err)?;
        Ok(n)
    }

    /// Decompress to a pre-allocated buffer.
    #[cfg(any(target_arch = "wasm32", feature = "pure-rust"))]
    pub fn decompress_to_buffer(source: &[u8], destination: &mut [u8]) -> io::Result<usize> {
        let out = decompress(source, destination.len())?;
        destination[..out.len()].copy_from_slice(&out);
        Ok(out.len())
    }
}
