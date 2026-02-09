//! Streaming (push-based) zstd encoder.
//!
//! This provides a `Write` interface similar to `zstd::stream::write::Encoder`:
//! bytes written are buffered into zstd blocks (128KiB) and emitted to the underlying writer.
//!
//! Unlike calling `compress_to_vec` repeatedly, this produces a single zstd frame and
//! preserves match history across blocks until `finish()` is called.

use alloc::vec::Vec;
use core::convert::TryInto;

#[cfg(feature = "hash")]
use core::hash::Hasher;
#[cfg(feature = "hash")]
use twox_hash::XxHash64;

use crate::io::{Error, Write};

use super::block_header::BlockHeader;
use super::frame_compressor::{CompressState, FseTables};
use super::frame_header::FrameHeader;
use super::levels::{compress_default, compress_fastest};
use super::{CompressionLevel, MatchGeneratorDriver, Matcher};

/// Streaming zstd encoder which writes a single zstd frame to an underlying writer.
///
/// Call [`Write::write`] to supply uncompressed bytes, [`Write::flush`] to force
/// emitting a (non-final) block, and [`StreamingEncoder::finish`] to close the
/// frame and recover the inner writer.
pub struct StreamingEncoder<W: Write> {
    drain: Option<W>,
    compression_level: CompressionLevel,
    state: CompressState<MatchGeneratorDriver>,
    output: Vec<u8>,
    cur_block: Vec<u8>,
    cur_filled: usize,
    started: bool,
    finished: bool,
    #[cfg(feature = "hash")]
    hasher: XxHash64,
}

impl<W: Write> StreamingEncoder<W> {
    #[inline(always)]
    fn err_finished() -> Error {
        #[cfg(feature = "std")]
        {
            Error::other("ns-zstd streaming encoder: already finished")
        }
        #[cfg(not(feature = "std"))]
        {
            Error::from(ErrorKind::Other)
        }
    }

    /// Create a new streaming encoder writing into `drain`.
    pub fn new(drain: W, compression_level: CompressionLevel) -> Self {
        Self {
            drain: Some(drain),
            compression_level,
            state: CompressState {
                matcher: MatchGeneratorDriver::new(1024 * 128, 1),
                last_huff_table: None,
                fse_tables: FseTables::new(),
                offset_hist: [1, 4, 8],
                tmp_literals: Vec::new(),
                tmp_sequences: Vec::new(),
                tmp_block: Vec::new(),
                #[cfg(feature = "std")]
                perf: super::frame_compressor::EncodePerf::new(),
            },
            output: Vec::with_capacity(1024 * 130),
            cur_block: Vec::new(),
            cur_filled: 0,
            started: false,
            finished: false,
            #[cfg(feature = "hash")]
            hasher: XxHash64::with_seed(0),
        }
    }

    /// Finish the stream (write the last block and optional checksum) and return the inner writer.
    pub fn finish(mut self) -> Result<W, Error> {
        self.finish_inner()?;
        Ok(self.drain.take().expect("drain must exist after finish"))
    }

    /// Acquire a reference to the underlying writer.
    pub fn get_ref(&self) -> &W {
        self.drain.as_ref().expect("encoder drain already taken")
    }

    /// Acquire a mutable reference to the underlying writer.
    pub fn get_mut(&mut self) -> &mut W {
        self.drain.as_mut().expect("encoder drain already taken")
    }

    fn ensure_started(&mut self) -> Result<(), Error> {
        if self.started {
            return Ok(());
        }

        self.state.matcher.reset(self.compression_level);
        self.state.last_huff_table = None;
        self.state.offset_hist = [1, 4, 8];
        #[cfg(feature = "std")]
        self.state.perf.reset_from_env();

        let header = FrameHeader {
            frame_content_size: None,
            single_segment: false,
            content_checksum: cfg!(feature = "hash"),
            dictionary_id: None,
            window_size: Some(self.state.matcher.window_size()),
        };

        self.output.clear();
        header.serialize(&mut self.output);
        let out = self.output.as_slice();
        self.drain.as_mut().expect("drain must exist").write_all(out)?;
        self.output.clear();

        self.cur_block = self.state.matcher.get_next_space();
        self.cur_filled = 0;
        self.started = true;
        Ok(())
    }

    fn emit_block(&mut self, last_block: bool, uncompressed_data: Vec<u8>) -> Result<(), Error> {
        let read_bytes = uncompressed_data.len();

        #[cfg(feature = "hash")]
        self.hasher.write(&uncompressed_data);

        // Special handling for a 0-byte terminating block (required when the total size is an
        // exact multiple of block size, or when the whole stream is empty).
        if uncompressed_data.is_empty() {
            let header = BlockHeader {
                last_block: true,
                block_type: crate::blocks::block::BlockType::Raw,
                block_size: 0,
            };
            header.serialize(&mut self.output);
            let out = self.output.as_slice();
            self.drain.as_mut().expect("drain must exist").write_all(out)?;
            self.output.clear();
            return Ok(());
        }

        match self.compression_level {
            CompressionLevel::Uncompressed => {
                let header = BlockHeader {
                    last_block,
                    block_type: crate::blocks::block::BlockType::Raw,
                    block_size: read_bytes.try_into().unwrap(),
                };
                header.serialize(&mut self.output);
                self.output.extend_from_slice(&uncompressed_data);
            }
            CompressionLevel::Fastest => {
                compress_fastest(&mut self.state, last_block, uncompressed_data, &mut self.output)
            }
            CompressionLevel::Default => {
                compress_default(&mut self.state, last_block, uncompressed_data, &mut self.output)
            }
            _ => {
                unimplemented!();
            }
        }

        let out = self.output.as_slice();
        self.drain.as_mut().expect("drain must exist").write_all(out)?;
        self.output.clear();
        Ok(())
    }

    fn flush_full_blocks_from_current(&mut self) -> Result<(), Error> {
        let block_size = self.cur_block.len();
        while self.cur_filled == block_size && block_size != 0 {
            let mut block = core::mem::take(&mut self.cur_block);
            // `cur_block` is always fully initialized; we only emit the filled region.
            block.truncate(self.cur_filled);
            self.emit_block(false, block)?;
            self.cur_block = self.state.matcher.get_next_space();
            self.cur_filled = 0;
        }
        Ok(())
    }

    fn flush_partial_block(&mut self) -> Result<(), Error> {
        if self.cur_filled == 0 {
            return Ok(());
        }
        let mut block = core::mem::take(&mut self.cur_block);
        block.truncate(self.cur_filled);
        self.emit_block(false, block)?;
        self.cur_block = self.state.matcher.get_next_space();
        self.cur_filled = 0;
        Ok(())
    }

    fn finish_inner(&mut self) -> Result<(), Error> {
        if self.finished {
            return Ok(());
        }
        self.ensure_started()?;

        if self.cur_filled == 0 {
            // Terminate the frame with an empty last block (matches FrameCompressor behavior).
            self.emit_block(true, Vec::new())?;
        } else {
            let mut block = core::mem::take(&mut self.cur_block);
            block.truncate(self.cur_filled);
            self.emit_block(true, block)?;
            self.cur_block = Vec::new();
            self.cur_filled = 0;
        }

        #[cfg(feature = "hash")]
        {
            let checksum = self.hasher.finish();
            self.get_mut().write_all(&(checksum as u32).to_le_bytes())?;
        }

        #[cfg(feature = "std")]
        {
            self.state.perf.finish_total();
            let level_name = match self.compression_level {
                CompressionLevel::Uncompressed => "uncompressed",
                CompressionLevel::Fastest => "fastest",
                CompressionLevel::Default => "default",
                CompressionLevel::Better => "better",
                CompressionLevel::Best => "best",
            };
            self.state.perf.print(level_name);
        }

        self.finished = true;
        Ok(())
    }
}

impl<W: Write> Write for StreamingEncoder<W> {
    fn write(&mut self, mut buf: &[u8]) -> Result<usize, Error> {
        if self.finished {
            return Err(Self::err_finished());
        }
        self.ensure_started()?;

        let total = buf.len();
        if total == 0 {
            return Ok(0);
        }

        let block_size = self.cur_block.len();
        while !buf.is_empty() {
            let dst = &mut self.cur_block[self.cur_filled..block_size];
            let n = dst.len().min(buf.len());
            dst[..n].copy_from_slice(&buf[..n]);
            self.cur_filled += n;
            buf = &buf[n..];

            if self.cur_filled == block_size {
                self.flush_full_blocks_from_current()?;
            }
        }

        Ok(total)
    }

    fn flush(&mut self) -> Result<(), Error> {
        if self.finished {
            return Ok(());
        }
        self.ensure_started()?;
        self.flush_partial_block()?;
        self.get_mut().flush()
    }
}
