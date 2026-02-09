//! Utilities and interfaces for encoding an entire frame. Allows reusing resources

use alloc::vec::Vec;
use core::convert::TryInto;
#[cfg(feature = "hash")]
use twox_hash::XxHash64;

#[cfg(feature = "hash")]
use core::hash::Hasher;

use super::{
    block_header::BlockHeader, frame_header::FrameHeader, levels::*,
    match_generator::MatchGeneratorDriver, CompressionLevel, Matcher,
};
use crate::fse::fse_encoder::{default_ll_table, default_ml_table, default_of_table, FSETable};

use crate::io::{Read, Write};

#[cfg(feature = "std")]
pub(crate) struct EncodePerf {
    enabled: bool,
    total_start: Option<std::time::Instant>,
    total_ns: u128,
    match_ns: u128,
    literals_ns: u128,
    sequences_ns: u128,
    blocks: u64,
    in_bytes: u64,
    out_bytes: u64,
    literals_bytes: u64,
    sequences: u64,
}

#[cfg(feature = "std")]
impl EncodePerf {
    pub fn new() -> Self {
        let mut s = Self {
            enabled: false,
            total_start: None,
            total_ns: 0,
            match_ns: 0,
            literals_ns: 0,
            sequences_ns: 0,
            blocks: 0,
            in_bytes: 0,
            out_bytes: 0,
            literals_bytes: 0,
            sequences: 0,
        };
        s.reset_from_env();
        s
    }

    #[inline]
    pub fn reset_from_env(&mut self) {
        self.enabled = std::env::var_os("NS_ZSTD_ENC_TIMING").is_some();
        self.total_start = self.enabled.then(std::time::Instant::now);
        self.total_ns = 0;
        self.match_ns = 0;
        self.literals_ns = 0;
        self.sequences_ns = 0;
        self.blocks = 0;
        self.in_bytes = 0;
        self.out_bytes = 0;
        self.literals_bytes = 0;
        self.sequences = 0;
    }

    #[inline]
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    #[inline]
    pub fn add_match(&mut self, d: std::time::Duration) {
        if self.enabled {
            self.match_ns += d.as_nanos();
        }
    }

    #[inline]
    pub fn add_literals(&mut self, d: std::time::Duration) {
        if self.enabled {
            self.literals_ns += d.as_nanos();
        }
    }

    #[inline]
    pub fn add_sequences(&mut self, d: std::time::Duration) {
        if self.enabled {
            self.sequences_ns += d.as_nanos();
        }
    }

    #[inline]
    pub fn on_block_end(&mut self, in_len: usize, out_len: usize, lit_len: usize, seqs: usize) {
        if self.enabled {
            self.blocks += 1;
            self.in_bytes += in_len as u64;
            self.out_bytes += out_len as u64;
            self.literals_bytes += lit_len as u64;
            self.sequences += seqs as u64;
        }
    }

    #[inline]
    pub fn finish_total(&mut self) {
        if self.enabled {
            if let Some(start) = self.total_start.take() {
                self.total_ns += start.elapsed().as_nanos();
            }
        }
    }

    pub fn print(&self, level_name: &str) {
        if !self.enabled {
            return;
        }
        let total_s = (self.total_ns as f64) / 1e9;
        let match_s = (self.match_ns as f64) / 1e9;
        let lit_s = (self.literals_ns as f64) / 1e9;
        let seq_s = (self.sequences_ns as f64) / 1e9;
        let in_mib = (self.in_bytes as f64) / (1024.0 * 1024.0);
        let out_mib = (self.out_bytes as f64) / (1024.0 * 1024.0);
        let throughput = if total_s > 0.0 { in_mib / total_s } else { 0.0 };
        let ratio =
            if self.in_bytes > 0 { (self.out_bytes as f64) / (self.in_bytes as f64) } else { 0.0 };
        std::eprintln!(
            "ns-zstd encode timing (level={}): blocks={} in={:.2} MiB out={:.2} MiB ratio={:.3} throughput={:.1} MiB/s",
            level_name, self.blocks, in_mib, out_mib, ratio, throughput
        );
        std::eprintln!(
            "  match={:.3}s literals={:.3}s sequences={:.3}s total={:.3}s (seqs={} literal_bytes={})",
            match_s, lit_s, seq_s, total_s, self.sequences, self.literals_bytes
        );
    }
}

/// An interface for compressing arbitrary data with the ZStandard compression algorithm.
///
/// `FrameCompressor` will generally be used by:
/// 1. Initializing a compressor by providing a buffer of data using `FrameCompressor::new()`
/// 2. Starting compression and writing that compression into a vec using `FrameCompressor::begin`
///
/// # Examples
/// ```
/// use ns_zstd::encoding::{FrameCompressor, CompressionLevel};
/// let mock_data: &[_] = &[0x1, 0x2, 0x3, 0x4];
/// let mut output = std::vec::Vec::new();
/// // Initialize a compressor.
/// let mut compressor = FrameCompressor::new(CompressionLevel::Uncompressed);
/// compressor.set_source(mock_data);
/// compressor.set_drain(&mut output);
///
/// // `compress` writes the compressed output into the provided buffer.
/// compressor.compress();
/// ```
pub struct FrameCompressor<R: Read, W: Write, M: Matcher> {
    uncompressed_data: Option<R>,
    compressed_data: Option<W>,
    compression_level: CompressionLevel,
    state: CompressState<M>,
    #[cfg(feature = "hash")]
    hasher: XxHash64,
}

pub(crate) struct FseTables {
    pub(crate) ll_default: FSETable,
    pub(crate) ll_previous: Option<FSETable>,
    pub(crate) ml_default: FSETable,
    pub(crate) ml_previous: Option<FSETable>,
    pub(crate) of_default: FSETable,
    pub(crate) of_previous: Option<FSETable>,
}

impl FseTables {
    pub fn new() -> Self {
        Self {
            ll_default: default_ll_table(),
            ll_previous: None,
            ml_default: default_ml_table(),
            ml_previous: None,
            of_default: default_of_table(),
            of_previous: None,
        }
    }
}

pub(crate) struct CompressState<M: Matcher> {
    pub(crate) matcher: M,
    pub(crate) last_huff_table: Option<crate::huff0::huff0_encoder::HuffmanTable>,
    pub(crate) fse_tables: FseTables,
    pub(crate) offset_hist: [u32; 3],
    // Scratch buffers reused across blocks to avoid per-block allocations.
    pub(crate) tmp_literals: Vec<u8>,
    pub(crate) tmp_sequences: Vec<EncodedSequence>,
    pub(crate) tmp_block: Vec<u8>,
    #[cfg(feature = "std")]
    pub(crate) perf: EncodePerf,
}

/// Pre-encoded sequence data used by the encoder hot path.
///
/// Stores the FSE symbol (code) for LL/ML/OF plus their extra bits.
/// This avoids recomputing `encode_literal_length/encode_match_len/encode_offset` multiple times
/// per sequence during table selection and sequence encoding.
#[derive(Clone, Copy)]
pub(crate) struct EncodedSequence {
    pub ll_add_bits: u16,
    pub ml_add_bits: u16,
    pub of_add_bits: u32,
    pub ll_code: u8,
    pub ml_code: u8,
    pub of_code: u8,
    pub ll_num_bits: u8,
    pub ml_num_bits: u8,
    pub of_num_bits: u8,
}

impl<R: Read, W: Write> FrameCompressor<R, W, MatchGeneratorDriver> {
    /// Create a new `FrameCompressor`
    pub fn new(compression_level: CompressionLevel) -> Self {
        Self {
            uncompressed_data: None,
            compressed_data: None,
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
                perf: EncodePerf::new(),
            },
            #[cfg(feature = "hash")]
            hasher: XxHash64::with_seed(0),
        }
    }
}

impl<R: Read, W: Write, M: Matcher> FrameCompressor<R, W, M> {
    /// Create a new `FrameCompressor` with a custom matching algorithm implementation
    pub fn new_with_matcher(matcher: M, compression_level: CompressionLevel) -> Self {
        Self {
            uncompressed_data: None,
            compressed_data: None,
            state: CompressState {
                matcher,
                last_huff_table: None,
                fse_tables: FseTables::new(),
                offset_hist: [1, 4, 8],
                tmp_literals: Vec::new(),
                tmp_sequences: Vec::new(),
                tmp_block: Vec::new(),
                #[cfg(feature = "std")]
                perf: EncodePerf::new(),
            },
            compression_level,
            #[cfg(feature = "hash")]
            hasher: XxHash64::with_seed(0),
        }
    }

    /// Before calling [FrameCompressor::compress] you need to set the source.
    ///
    /// This is the data that is compressed and written into the drain.
    pub fn set_source(&mut self, uncompressed_data: R) -> Option<R> {
        self.uncompressed_data.replace(uncompressed_data)
    }

    /// Before calling [FrameCompressor::compress] you need to set the drain.
    ///
    /// As the compressor compresses data, the drain serves as a place for the output to be writte.
    pub fn set_drain(&mut self, compressed_data: W) -> Option<W> {
        self.compressed_data.replace(compressed_data)
    }

    /// Compress the uncompressed data from the provided source as one Zstd frame and write it to the provided drain
    ///
    /// This will repeatedly call [Read::read] on the source to fill up blocks until the source returns 0 on the read call.
    /// Also [Write::write_all] will be called on the drain after each block has been encoded.
    ///
    /// To avoid endlessly encoding from a potentially endless source (like a network socket) you can use the
    /// [Read::take] function
    pub fn compress(&mut self) {
        // Clearing buffers to allow re-using of the compressor
        self.state.matcher.reset(self.compression_level);
        self.state.last_huff_table = None;
        self.state.offset_hist = [1, 4, 8];
        #[cfg(feature = "std")]
        self.state.perf.reset_from_env();
        let source = self.uncompressed_data.as_mut().unwrap();
        let drain = self.compressed_data.as_mut().unwrap();
        // As the frame is compressed, it's stored here
        let output: &mut Vec<u8> = &mut Vec::with_capacity(1024 * 130);
        // First write the frame header
        let header = FrameHeader {
            frame_content_size: None,
            single_segment: false,
            content_checksum: cfg!(feature = "hash"),
            dictionary_id: None,
            window_size: Some(self.state.matcher.window_size()),
        };
        header.serialize(output);
        // Now compress block by block
        loop {
            // Read a single block's worth of uncompressed data from the input
            let mut uncompressed_data = self.state.matcher.get_next_space();
            let mut read_bytes = 0;
            let last_block;
            'read_loop: loop {
                let new_bytes = source.read(&mut uncompressed_data[read_bytes..]).unwrap();
                if new_bytes == 0 {
                    last_block = true;
                    break 'read_loop;
                }
                read_bytes += new_bytes;
                if read_bytes == uncompressed_data.len() {
                    last_block = false;
                    break 'read_loop;
                }
            }
            uncompressed_data.resize(read_bytes, 0);
            // As we read, hash that data too
            #[cfg(feature = "hash")]
            self.hasher.write(&uncompressed_data);
            // Special handling is needed for compression of a totally empty file (why you'd want to do that, I don't know)
            if uncompressed_data.is_empty() {
                let header = BlockHeader {
                    last_block: true,
                    block_type: crate::blocks::block::BlockType::Raw,
                    block_size: 0,
                };
                // Write the header, then the block
                header.serialize(output);
                drain.write_all(output).unwrap();
                output.clear();
                break;
            }

            match self.compression_level {
                CompressionLevel::Uncompressed => {
                    let header = BlockHeader {
                        last_block,
                        block_type: crate::blocks::block::BlockType::Raw,
                        block_size: read_bytes.try_into().unwrap(),
                    };
                    // Write the header, then the block
                    header.serialize(output);
                    output.extend_from_slice(&uncompressed_data);
                }
                CompressionLevel::Fastest => {
                    compress_fastest(&mut self.state, last_block, uncompressed_data, output)
                }
                CompressionLevel::Default => {
                    compress_default(&mut self.state, last_block, uncompressed_data, output)
                }
                _ => {
                    unimplemented!();
                }
            }
            drain.write_all(output).unwrap();
            output.clear();
            if last_block {
                break;
            }
        }

        // If the `hash` feature is enabled, then `content_checksum` is set to true in the header
        // and a 32 bit hash is written at the end of the data.
        #[cfg(feature = "hash")]
        {
            // Because we only have the data as a reader, we need to read all of it to calculate the checksum
            // Possible TODO: create a wrapper around self.uncompressed data that hashes the data as it's read?
            let content_checksum = self.hasher.finish();
            drain.write_all(&(content_checksum as u32).to_le_bytes()).unwrap();
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
    }

    /// Get a mutable reference to the source
    pub fn source_mut(&mut self) -> Option<&mut R> {
        self.uncompressed_data.as_mut()
    }

    /// Get a mutable reference to the drain
    pub fn drain_mut(&mut self) -> Option<&mut W> {
        self.compressed_data.as_mut()
    }

    /// Get a reference to the source
    pub fn source(&self) -> Option<&R> {
        self.uncompressed_data.as_ref()
    }

    /// Get a reference to the drain
    pub fn drain(&self) -> Option<&W> {
        self.compressed_data.as_ref()
    }

    /// Retrieve the source
    pub fn take_source(&mut self) -> Option<R> {
        self.uncompressed_data.take()
    }

    /// Retrieve the drain
    pub fn take_drain(&mut self) -> Option<W> {
        self.compressed_data.take()
    }

    /// Before calling [FrameCompressor::compress] you can replace the matcher
    pub fn replace_matcher(&mut self, mut match_generator: M) -> M {
        core::mem::swap(&mut match_generator, &mut self.state.matcher);
        match_generator
    }

    /// Before calling [FrameCompressor::compress] you can replace the compression level
    pub fn set_compression_level(
        &mut self,
        compression_level: CompressionLevel,
    ) -> CompressionLevel {
        let old = self.compression_level;
        self.compression_level = compression_level;
        old
    }

    /// Get the current compression level
    pub fn compression_level(&self) -> CompressionLevel {
        self.compression_level
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::FrameCompressor;
    use crate::common::MAGIC_NUM;
    use crate::decoding::FrameDecoder;
    use alloc::vec::Vec;

    #[test]
    fn frame_starts_with_magic_num() {
        let mock_data = [1_u8, 2, 3].as_slice();
        let mut output: Vec<u8> = Vec::new();
        let mut compressor = FrameCompressor::new(super::CompressionLevel::Uncompressed);
        compressor.set_source(mock_data);
        compressor.set_drain(&mut output);

        compressor.compress();
        assert!(output.starts_with(&MAGIC_NUM.to_le_bytes()));
    }

    #[test]
    fn very_simple_raw_compress() {
        let mock_data = [1_u8, 2, 3].as_slice();
        let mut output: Vec<u8> = Vec::new();
        let mut compressor = FrameCompressor::new(super::CompressionLevel::Uncompressed);
        compressor.set_source(mock_data);
        compressor.set_drain(&mut output);

        compressor.compress();
    }

    #[test]
    fn very_simple_compress() {
        let mut mock_data = vec![0; 1 << 17];
        mock_data.extend(vec![1; (1 << 17) - 1]);
        mock_data.extend(vec![2; (1 << 18) - 1]);
        mock_data.extend(vec![2; 1 << 17]);
        mock_data.extend(vec![3; (1 << 17) - 1]);
        let mut output: Vec<u8> = Vec::new();
        let mut compressor = FrameCompressor::new(super::CompressionLevel::Uncompressed);
        compressor.set_source(mock_data.as_slice());
        compressor.set_drain(&mut output);

        compressor.compress();

        let mut decoder = FrameDecoder::new();
        let mut decoded = Vec::with_capacity(mock_data.len());
        decoder.decode_all_to_vec(&output, &mut decoded).unwrap();
        assert_eq!(mock_data, decoded);
    }

    #[test]
    fn rle_compress() {
        let mock_data = vec![0; 1 << 19];
        let mut output: Vec<u8> = Vec::new();
        let mut compressor = FrameCompressor::new(super::CompressionLevel::Uncompressed);
        compressor.set_source(mock_data.as_slice());
        compressor.set_drain(&mut output);

        compressor.compress();

        let mut decoder = FrameDecoder::new();
        let mut decoded = Vec::with_capacity(mock_data.len());
        decoder.decode_all_to_vec(&output, &mut decoded).unwrap();
        assert_eq!(mock_data, decoded);
    }

    #[test]
    fn aaa_compress() {
        let mock_data = vec![0, 1, 3, 4, 5];
        let mut output: Vec<u8> = Vec::new();
        let mut compressor = FrameCompressor::new(super::CompressionLevel::Uncompressed);
        compressor.set_source(mock_data.as_slice());
        compressor.set_drain(&mut output);

        compressor.compress();

        let mut decoder = FrameDecoder::new();
        let mut decoded = Vec::with_capacity(mock_data.len());
        decoder.decode_all_to_vec(&output, &mut decoded).unwrap();
        assert_eq!(mock_data, decoded);
    }

    #[cfg(feature = "std")]
    #[test]
    fn fuzz_targets() {
        use std::io::Read;
        fn decode_ns_zstd(data: &mut dyn std::io::Read) -> Vec<u8> {
            let mut decoder = crate::decoding::StreamingDecoder::new(data).unwrap();
            let mut result: Vec<u8> = Vec::new();
            decoder.read_to_end(&mut result).expect("Decoding failed");
            result
        }

        fn decode_ns_zstd_writer(mut data: impl Read) -> Vec<u8> {
            let mut decoder = crate::decoding::FrameDecoder::new();
            decoder.reset(&mut data).unwrap();
            let mut result = vec![];
            while !decoder.is_finished() || decoder.can_collect() > 0 {
                decoder
                    .decode_blocks(
                        &mut data,
                        crate::decoding::BlockDecodingStrategy::UptoBytes(1024 * 1024),
                    )
                    .unwrap();
                decoder.collect_to_writer(&mut result).unwrap();
            }
            result
        }

        fn encode_zstd(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
            crate::encoding::compress_to_vec(data, crate::encoding::CompressionLevel::Fastest);
            Ok(crate::encoding::compress_to_vec(data, crate::encoding::CompressionLevel::Fastest))
        }

        fn encode_ns_zstd_uncompressed(data: &mut dyn std::io::Read) -> Vec<u8> {
            let mut input = Vec::new();
            data.read_to_end(&mut input).unwrap();

            crate::encoding::compress_to_vec(
                input.as_slice(),
                crate::encoding::CompressionLevel::Uncompressed,
            )
        }

        fn encode_ns_zstd_compressed(data: &mut dyn std::io::Read) -> Vec<u8> {
            let mut input = Vec::new();
            data.read_to_end(&mut input).unwrap();

            crate::encoding::compress_to_vec(
                input.as_slice(),
                crate::encoding::CompressionLevel::Fastest,
            )
        }

        fn decode_zstd(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
            let mut decoder = crate::decoding::StreamingDecoder::new(data).unwrap();
            let mut output = Vec::new();
            decoder.read_to_end(&mut output).expect("Decoding failed");
            Ok(output)
        }
        if std::fs::exists("fuzz/artifacts/interop").unwrap_or(false) {
            for file in std::fs::read_dir("fuzz/artifacts/interop").unwrap() {
                if file.as_ref().unwrap().file_type().unwrap().is_file() {
                    let data = std::fs::read(file.unwrap().path()).unwrap();
                    let data = data.as_slice();
                    // Decoding
                    let compressed = encode_zstd(data).unwrap();
                    let decoded = decode_ns_zstd(&mut compressed.as_slice());
                    let decoded2 = decode_ns_zstd_writer(&mut compressed.as_slice());
                    assert!(
                        decoded == data,
                        "Decoded data did not match the original input during decompression"
                    );
                    assert_eq!(
                        decoded2, data,
                        "Decoded data did not match the original input during decompression"
                    );

                    // Encoding
                    // Uncompressed encoding
                    let mut input = data;
                    let compressed = encode_ns_zstd_uncompressed(&mut input);
                    let decoded = decode_zstd(&compressed).unwrap();
                    assert_eq!(
                        decoded, data,
                        "Decoded data did not match the original input during compression"
                    );
                    // Compressed encoding
                    let mut input = data;
                    let compressed = encode_ns_zstd_compressed(&mut input);
                    let decoded = decode_zstd(&compressed).unwrap();
                    assert_eq!(
                        decoded, data,
                        "Decoded data did not match the original input during compression"
                    );
                }
            }
        }
    }
}
