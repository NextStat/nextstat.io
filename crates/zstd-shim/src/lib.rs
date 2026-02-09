//! Pure-Rust shim for the `zstd` crate API.
//!
//! This is intentionally minimal: it implements the parts of the `zstd` API that
//! Apache Arrow's `parquet` crate uses (`zstd::Encoder` / `zstd::Decoder`).
//!
//! Implementation backend: `ns-zstd` (pure Rust).

use std::io;
use std::ops::RangeInclusive;

/// Default compression level in upstream zstd (currently 3).
pub const DEFAULT_COMPRESSION_LEVEL: i32 = 3;

/// The accepted range of compression levels.
///
/// Upstream zstd accepts 1..=22. We keep the same range for compatibility.
pub fn compression_level_range() -> RangeInclusive<i32> {
    1..=22
}

fn map_level(level: i32) -> ns_zstd::encoding::CompressionLevel {
    // zstd semantics: 0 = "default" (currently level 3).
    // We only support two quality tiers right now: Fastest (~1) and Default (~3).
    match level {
        0 => ns_zstd::encoding::CompressionLevel::Default,
        i32::MIN..=1 => ns_zstd::encoding::CompressionLevel::Fastest,
        _ => ns_zstd::encoding::CompressionLevel::Default,
    }
}

fn map_decode_err(e: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
}

pub mod stream {
    pub mod read {
        use std::io::{self, Read};

        /// Pull-based decoder implementing [`std::io::Read`].
        ///
        /// This shim decodes on demand using `ns-zstd`'s streaming decoder.
        pub struct Decoder<R: Read> {
            inner: ns_zstd::decoding::StreamingDecoder<R, ns_zstd::decoding::FrameDecoder>,
        }

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

        impl<R: Read> Read for Decoder<R> {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                self.inner.read(buf)
            }
        }
    }

    pub mod write {
        use std::io::{self, Write};

        /// Push-based encoder implementing [`std::io::Write`].
        ///
        /// This shim encodes incrementally into a single zstd frame, writing blocks to the
        /// underlying writer as they become available.
        pub struct Encoder<W: Write> {
            inner: ns_zstd::encoding::StreamingEncoder<W>,
        }

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
    pub fn compress(data: &[u8], level: i32) -> io::Result<Vec<u8>> {
        Ok(ns_zstd::encoding::compress_to_vec(data, crate::map_level(level)))
    }

    /// Decompress a zstd frame, returning at most `capacity` bytes.
    pub fn decompress(data: &[u8], capacity: usize) -> io::Result<Vec<u8>> {
        let mut out = Vec::with_capacity(capacity.max(1));
        let mut dec = ns_zstd::decoding::FrameDecoder::new();
        dec.decode_all_to_vec(data, &mut out).map_err(crate::map_decode_err)?;
        Ok(out)
    }

    /// Compress to a pre-allocated buffer.
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
    pub fn decompress_to_buffer(source: &[u8], destination: &mut [u8]) -> io::Result<usize> {
        let out = decompress(source, destination.len())?;
        destination[..out.len()].copy_from_slice(&out);
        Ok(out.len())
    }
}
