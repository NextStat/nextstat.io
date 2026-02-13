use super::super::blocks::block::BlockHeader;
use super::super::blocks::block::BlockType;
use super::super::blocks::literals_section::LiteralsSection;
use super::super::blocks::literals_section::LiteralsSectionType;
use super::super::blocks::sequence_section::SequencesHeader;
use super::literals_section_decoder::decode_literals;
use crate::common::MAX_BLOCK_SIZE;
use crate::decoding::errors::DecodeSequenceError;
use crate::decoding::errors::ExecuteSequencesError;
use crate::decoding::errors::{
    BlockHeaderReadError, BlockSizeError, BlockTypeError, DecodeBlockContentError,
    DecompressBlockError,
};
use crate::decoding::fused::{
    decode_and_execute_sequences, decode_and_execute_sequences_to, FusedOutputBuffer,
};
use crate::decoding::scratch::DecoderScratch;
use crate::decoding::slice_output_buffer::SliceOutputBuffer;
use crate::io::{Error, ErrorKind, Read};

pub struct BlockDecoder {
    header_buffer: [u8; 3],
    internal_state: DecoderState,
}

enum DecoderState {
    ReadyToDecodeNextHeader,
    ReadyToDecodeNextBody,
    #[allow(dead_code)]
    Failed, //TODO put "self.internal_state = DecoderState::Failed;" everywhere an unresolvable error occurs
}

/// Create a new [BlockDecoder].
pub fn new() -> BlockDecoder {
    BlockDecoder { internal_state: DecoderState::ReadyToDecodeNextHeader, header_buffer: [0u8; 3] }
}

impl BlockDecoder {
    pub(crate) fn decode_block_content_from_slice_to(
        &mut self,
        header: &BlockHeader,
        workspace: &mut DecoderScratch,
        out: &mut impl FusedOutputBuffer,
        source: &mut &[u8],
    ) -> Result<u64, DecodeBlockContentError> {
        match self.internal_state {
            DecoderState::ReadyToDecodeNextBody => { /* Happy :) */ }
            DecoderState::Failed => return Err(DecodeBlockContentError::DecoderStateIsFailed),
            DecoderState::ReadyToDecodeNextHeader => {
                return Err(DecodeBlockContentError::ExpectedHeaderOfPreviousBlock)
            }
        }

        #[inline(always)]
        fn take_bytes<'a>(src: &mut &'a [u8], n: usize) -> Result<&'a [u8], Error> {
            if src.len() < n {
                return Err(Error::from(ErrorKind::UnexpectedEof));
            }
            let (head, rest) = src.split_at(n);
            *src = rest;
            Ok(head)
        }

        let block_type = header.block_type;
        match block_type {
            BlockType::RLE => {
                let b = take_bytes(source, 1).map_err(|err| {
                    DecodeBlockContentError::ReadError { step: block_type, source: err }
                })?;
                self.internal_state = DecoderState::ReadyToDecodeNextHeader;

                let mut remaining = header.decompressed_size as usize;
                const BATCH_SIZE: usize = 512;
                let mut buf = [0u8; BATCH_SIZE];
                buf.fill(b[0]);
                while remaining != 0 {
                    let chunk = remaining.min(BATCH_SIZE);
                    out.ensure_capacity(chunk)
                        .map_err(|e| DecodeBlockContentError::DecompressBlockError(e.into()))?;
                    out.push_unchecked(&buf[..chunk])
                        .map_err(|e| DecodeBlockContentError::DecompressBlockError(e.into()))?;
                    remaining -= chunk;
                }

                Ok(1)
            }
            BlockType::Raw => {
                let mut remaining = header.decompressed_size as usize;
                while remaining != 0 {
                    let chunk = remaining;
                    out.ensure_capacity(chunk)
                        .map_err(|e| DecodeBlockContentError::DecompressBlockError(e.into()))?;

                    let bytes = take_bytes(source, chunk).map_err(|err| {
                        DecodeBlockContentError::ReadError { step: block_type, source: err }
                    })?;
                    out.push_unchecked(bytes)
                        .map_err(|e| DecodeBlockContentError::DecompressBlockError(e.into()))?;
                    remaining -= chunk;
                }

                self.internal_state = DecoderState::ReadyToDecodeNextHeader;
                Ok(u64::from(header.decompressed_size))
            }
            BlockType::Reserved => {
                panic!("How did you even get this. The decoder should error out if it detects a reserved-type block");
            }
            BlockType::Compressed => {
                let size = header.content_size as usize;
                let raw = take_bytes(source, size).map_err(DecompressBlockError::from)?;
                self.decompress_block_from_slice_to(header, workspace, out, raw)?;

                self.internal_state = DecoderState::ReadyToDecodeNextHeader;
                Ok(u64::from(header.content_size))
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn decode_block_content_to(
        &mut self,
        header: &BlockHeader,
        workspace: &mut DecoderScratch,
        out: &mut SliceOutputBuffer<'_>,
        mut source: impl Read,
    ) -> Result<u64, DecodeBlockContentError> {
        match self.internal_state {
            DecoderState::ReadyToDecodeNextBody => { /* Happy :) */ }
            DecoderState::Failed => return Err(DecodeBlockContentError::DecoderStateIsFailed),
            DecoderState::ReadyToDecodeNextHeader => {
                return Err(DecodeBlockContentError::ExpectedHeaderOfPreviousBlock)
            }
        }

        let block_type = header.block_type;
        match block_type {
            BlockType::RLE => {
                let mut b = [0u8; 1];
                source.read_exact(&mut b).map_err(|err| DecodeBlockContentError::ReadError {
                    step: block_type,
                    source: err,
                })?;
                self.internal_state = DecoderState::ReadyToDecodeNextHeader;

                let mut remaining = header.decompressed_size as usize;
                while remaining != 0 {
                    let chunk = remaining.min(out.remaining_mut().len());
                    if chunk == 0 {
                        return Err(DecodeBlockContentError::DecompressBlockError(
                            ExecuteSequencesError::NotEnoughBytesForSequence {
                                wanted: out.bytes_written() + remaining,
                                have: out.bytes_written(),
                            }
                            .into(),
                        ));
                    }
                    out.remaining_mut()[..chunk].fill(b[0]);
                    out.advance(chunk);
                    remaining -= chunk;
                }

                Ok(1)
            }
            BlockType::Raw => {
                let mut remaining = header.decompressed_size as usize;
                while remaining != 0 {
                    let chunk = remaining.min(out.remaining_mut().len());
                    if chunk == 0 {
                        return Err(DecodeBlockContentError::DecompressBlockError(
                            ExecuteSequencesError::NotEnoughBytesForSequence {
                                wanted: out.bytes_written() + remaining,
                                have: out.bytes_written(),
                            }
                            .into(),
                        ));
                    }
                    source.read_exact(&mut out.remaining_mut()[..chunk]).map_err(|err| {
                        DecodeBlockContentError::ReadError { step: block_type, source: err }
                    })?;
                    out.advance(chunk);
                    remaining -= chunk;
                }

                self.internal_state = DecoderState::ReadyToDecodeNextHeader;
                Ok(u64::from(header.decompressed_size))
            }
            BlockType::Reserved => {
                panic!("How did you even get this. The decoder should error out if it detects a reserved-type block");
            }
            BlockType::Compressed => {
                self.decompress_block_to(header, workspace, out, source)?;

                self.internal_state = DecoderState::ReadyToDecodeNextHeader;
                Ok(u64::from(header.content_size))
            }
        }
    }

    pub fn decode_block_content(
        &mut self,
        header: &BlockHeader,
        workspace: &mut DecoderScratch, //reuse this as often as possible. Not only if the trees are reused but also reuse the allocations when building new trees
        mut source: impl Read,
    ) -> Result<u64, DecodeBlockContentError> {
        match self.internal_state {
            DecoderState::ReadyToDecodeNextBody => { /* Happy :) */ }
            DecoderState::Failed => return Err(DecodeBlockContentError::DecoderStateIsFailed),
            DecoderState::ReadyToDecodeNextHeader => {
                return Err(DecodeBlockContentError::ExpectedHeaderOfPreviousBlock)
            }
        }

        let block_type = header.block_type;
        workspace.buffer.reserve_append(header.decompressed_size as usize);
        match block_type {
            BlockType::RLE => {
                const BATCH_SIZE: usize = 512;
                let mut buf = [0u8; BATCH_SIZE];
                let full_reads = header.decompressed_size / BATCH_SIZE as u32;
                let single_read_size = header.decompressed_size % BATCH_SIZE as u32;

                source.read_exact(&mut buf[0..1]).map_err(|err| {
                    DecodeBlockContentError::ReadError { step: block_type, source: err }
                })?;
                self.internal_state = DecoderState::ReadyToDecodeNextHeader;

                for i in 1..BATCH_SIZE {
                    buf[i] = buf[0];
                }

                for _ in 0..full_reads {
                    workspace.buffer.push(&buf[..]);
                }
                let smaller = &mut buf[..single_read_size as usize];
                workspace.buffer.push(smaller);

                Ok(1)
            }
            BlockType::Raw => {
                const BATCH_SIZE: usize = 128 * 1024;
                let mut buf = [0u8; BATCH_SIZE];
                let full_reads = header.decompressed_size / BATCH_SIZE as u32;
                let single_read_size = header.decompressed_size % BATCH_SIZE as u32;

                for _ in 0..full_reads {
                    source.read_exact(&mut buf[..]).map_err(|err| {
                        DecodeBlockContentError::ReadError { step: block_type, source: err }
                    })?;
                    workspace.buffer.push(&buf[..]);
                }

                let smaller = &mut buf[..single_read_size as usize];
                source.read_exact(smaller).map_err(|err| DecodeBlockContentError::ReadError {
                    step: block_type,
                    source: err,
                })?;
                workspace.buffer.push(smaller);

                self.internal_state = DecoderState::ReadyToDecodeNextHeader;
                Ok(u64::from(header.decompressed_size))
            }

            BlockType::Reserved => {
                panic!("How did you even get this. The decoder should error out if it detects a reserved-type block");
            }

            BlockType::Compressed => {
                self.decompress_block(header, workspace, source)?;

                self.internal_state = DecoderState::ReadyToDecodeNextHeader;
                Ok(u64::from(header.content_size))
            }
        }
    }

    fn decompress_block(
        &mut self,
        header: &BlockHeader,
        workspace: &mut DecoderScratch, //reuse this as often as possible. Not only if the trees are reused but also reuse the allocations when building new trees
        source: impl Read,
    ) -> Result<(), DecompressBlockError> {
        workspace.buffer.reserve_append(header.decompressed_size as usize);
        let size = header.content_size as usize;
        workspace.block_content_buffer.clear();
        workspace.block_content_buffer.resize(size, 0);
        // `read_exact` is both simpler (fixed-size) and avoids relying on `read_to_end`'s growth loop.
        // We keep the buffer zero-initialized for safety (no uninit bytes ever become visible).
        if let Err(e) =
            source.take(size as u64).read_exact(workspace.block_content_buffer.as_mut_slice())
        {
            workspace.block_content_buffer.clear();
            return Err(DecompressBlockError::from(e));
        }
        let raw = workspace.block_content_buffer.as_slice();

        let mut section = LiteralsSection::new();
        let bytes_in_literals_header = section.parse_from_header(raw)?;
        let raw = &raw[bytes_in_literals_header as usize..];
        vprintln!(
            "Found {} literalssection with regenerated size: {}, and compressed size: {:?}",
            section.ls_type,
            section.regenerated_size,
            section.compressed_size
        );

        let upper_limit_for_literals = match section.compressed_size {
            Some(x) => x as usize,
            None => match section.ls_type {
                LiteralsSectionType::RLE => 1,
                LiteralsSectionType::Raw => section.regenerated_size as usize,
                _ => panic!("Bug in this library"),
            },
        };

        if raw.len() < upper_limit_for_literals {
            return Err(DecompressBlockError::MalformedSectionHeader {
                expected_len: upper_limit_for_literals,
                remaining_bytes: raw.len(),
            });
        }

        let raw_literals = &raw[..upper_limit_for_literals];
        vprintln!("Slice for literals: {}", raw_literals.len());

        // Avoid copying raw literals into an intermediate buffer: for Raw sections we can
        // execute sequences directly from the slice within `block_content_buffer`.
        let literals: &[u8];
        let bytes_used_in_literals_section = if matches!(section.ls_type, LiteralsSectionType::Raw)
        {
            literals = raw_literals;
            section.regenerated_size
        } else {
            workspace.literals_buffer.clear(); // defensive: previous block literals must have been consumed
            let used = decode_literals(
                &section,
                &mut workspace.huf,
                raw_literals,
                &mut workspace.literals_buffer,
            )?;
            literals = &workspace.literals_buffer;
            used
        };
        debug_assert!(
            section.regenerated_size == literals.len() as u32,
            "Wrong number of literals: {}, Should have been: {}",
            literals.len(),
            section.regenerated_size
        );
        debug_assert!(bytes_used_in_literals_section == upper_limit_for_literals as u32);

        let raw = &raw[upper_limit_for_literals..];
        vprintln!("Slice for sequences with headers: {}", raw.len());

        let mut seq_section = SequencesHeader::new();
        let bytes_in_sequence_header = seq_section.parse_from_header(raw)?;
        let raw = &raw[bytes_in_sequence_header as usize..];
        vprintln!(
            "Found sequencessection with sequences: {} and size: {}",
            seq_section.num_sequences,
            raw.len()
        );

        debug_assert!(
            u32::from(bytes_in_literals_header)
                + bytes_used_in_literals_section
                + u32::from(bytes_in_sequence_header)
                + raw.len() as u32
                == header.content_size
        );
        vprintln!("Slice for sequences: {}", raw.len());

        if seq_section.num_sequences != 0 {
            vprintln!("Fused decode+execute sequences");
            decode_and_execute_sequences(
                &seq_section,
                raw,
                &mut workspace.fse,
                &mut workspace.buffer,
                &mut workspace.offset_hist,
                literals,
            )?;
        } else {
            if !raw.is_empty() {
                return Err(DecompressBlockError::DecodeSequenceError(
                    DecodeSequenceError::ExtraBits { bits_remaining: raw.len() as isize * 8 },
                ));
            }
            workspace.buffer.push(literals);
            workspace.sequences.clear();
        }

        Ok(())
    }

    fn decompress_block_from_slice_to(
        &mut self,
        header: &BlockHeader,
        workspace: &mut DecoderScratch,
        out: &mut impl FusedOutputBuffer,
        raw: &[u8],
    ) -> Result<(), DecompressBlockError> {
        let mut section = LiteralsSection::new();
        let bytes_in_literals_header = section.parse_from_header(raw)?;
        let raw = &raw[bytes_in_literals_header as usize..];

        let upper_limit_for_literals = match section.compressed_size {
            Some(x) => x as usize,
            None => match section.ls_type {
                LiteralsSectionType::RLE => 1,
                LiteralsSectionType::Raw => section.regenerated_size as usize,
                _ => panic!("Bug in this library"),
            },
        };

        if raw.len() < upper_limit_for_literals {
            return Err(DecompressBlockError::MalformedSectionHeader {
                expected_len: upper_limit_for_literals,
                remaining_bytes: raw.len(),
            });
        }

        let raw_literals = &raw[..upper_limit_for_literals];

        let literals: &[u8];
        let bytes_used_in_literals_section = if matches!(section.ls_type, LiteralsSectionType::Raw)
        {
            literals = raw_literals;
            section.regenerated_size
        } else {
            workspace.literals_buffer.clear();
            let used = decode_literals(
                &section,
                &mut workspace.huf,
                raw_literals,
                &mut workspace.literals_buffer,
            )?;
            literals = &workspace.literals_buffer;
            used
        };
        debug_assert!(
            section.regenerated_size == literals.len() as u32,
            "Wrong number of literals: {}, Should have been: {}",
            literals.len(),
            section.regenerated_size
        );
        debug_assert!(bytes_used_in_literals_section == upper_limit_for_literals as u32);

        let raw = &raw[upper_limit_for_literals..];

        let mut seq_section = SequencesHeader::new();
        let bytes_in_sequence_header = seq_section.parse_from_header(raw)?;
        let raw = &raw[bytes_in_sequence_header as usize..];

        debug_assert!(
            u32::from(bytes_in_literals_header)
                + bytes_used_in_literals_section
                + u32::from(bytes_in_sequence_header)
                + raw.len() as u32
                == header.content_size
        );

        if seq_section.num_sequences != 0 {
            decode_and_execute_sequences_to(
                &seq_section,
                raw,
                &mut workspace.fse,
                out,
                &mut workspace.offset_hist,
                literals,
            )?;
        } else {
            if !raw.is_empty() {
                return Err(DecompressBlockError::DecodeSequenceError(
                    DecodeSequenceError::ExtraBits { bits_remaining: raw.len() as isize * 8 },
                ));
            }
            out.push(literals)?;
            workspace.sequences.clear();
        }

        Ok(())
    }

    #[allow(dead_code)]
    fn decompress_block_to(
        &mut self,
        header: &BlockHeader,
        workspace: &mut DecoderScratch,
        out: &mut SliceOutputBuffer<'_>,
        source: impl Read,
    ) -> Result<(), DecompressBlockError> {
        let size = header.content_size as usize;
        workspace.block_content_buffer.clear();
        workspace.block_content_buffer.resize(size, 0);
        if let Err(e) =
            source.take(size as u64).read_exact(workspace.block_content_buffer.as_mut_slice())
        {
            workspace.block_content_buffer.clear();
            return Err(DecompressBlockError::from(e));
        }
        let raw = workspace.block_content_buffer.as_slice();

        let mut section = LiteralsSection::new();
        let bytes_in_literals_header = section.parse_from_header(raw)?;
        let raw = &raw[bytes_in_literals_header as usize..];

        let upper_limit_for_literals = match section.compressed_size {
            Some(x) => x as usize,
            None => match section.ls_type {
                LiteralsSectionType::RLE => 1,
                LiteralsSectionType::Raw => section.regenerated_size as usize,
                _ => panic!("Bug in this library"),
            },
        };

        if raw.len() < upper_limit_for_literals {
            return Err(DecompressBlockError::MalformedSectionHeader {
                expected_len: upper_limit_for_literals,
                remaining_bytes: raw.len(),
            });
        }

        let raw_literals = &raw[..upper_limit_for_literals];

        let literals: &[u8];
        let bytes_used_in_literals_section = if matches!(section.ls_type, LiteralsSectionType::Raw)
        {
            literals = raw_literals;
            section.regenerated_size
        } else {
            workspace.literals_buffer.clear();
            let used = decode_literals(
                &section,
                &mut workspace.huf,
                raw_literals,
                &mut workspace.literals_buffer,
            )?;
            literals = &workspace.literals_buffer;
            used
        };
        debug_assert!(
            section.regenerated_size == literals.len() as u32,
            "Wrong number of literals: {}, Should have been: {}",
            literals.len(),
            section.regenerated_size
        );
        debug_assert!(bytes_used_in_literals_section == upper_limit_for_literals as u32);

        let raw = &raw[upper_limit_for_literals..];

        let mut seq_section = SequencesHeader::new();
        let bytes_in_sequence_header = seq_section.parse_from_header(raw)?;
        let raw = &raw[bytes_in_sequence_header as usize..];

        debug_assert!(
            u32::from(bytes_in_literals_header)
                + bytes_used_in_literals_section
                + u32::from(bytes_in_sequence_header)
                + raw.len() as u32
                == header.content_size
        );

        if seq_section.num_sequences != 0 {
            decode_and_execute_sequences_to(
                &seq_section,
                raw,
                &mut workspace.fse,
                out,
                &mut workspace.offset_hist,
                literals,
            )?;
        } else {
            if !raw.is_empty() {
                return Err(DecompressBlockError::DecodeSequenceError(
                    DecodeSequenceError::ExtraBits { bits_remaining: raw.len() as isize * 8 },
                ));
            }
            out.push(literals)?;
            workspace.sequences.clear();
        }

        Ok(())
    }

    /// Reads 3 bytes from the provided reader and returns
    /// the deserialized header and the number of bytes read.
    pub fn read_block_header(
        &mut self,
        mut r: impl Read,
    ) -> Result<(BlockHeader, u8), BlockHeaderReadError> {
        //match self.internal_state {
        //    DecoderState::ReadyToDecodeNextHeader => {/* Happy :) */},
        //    DecoderState::Failed => return Err(format!("Cant decode next block if failed along the way. Results will be nonsense")),
        //    DecoderState::ReadyToDecodeNextBody => return Err(format!("Cant decode next block header, while expecting to decode the body of the previous block. Results will be nonsense")),
        //}

        r.read_exact(&mut self.header_buffer[0..3])?;

        let btype = self.block_type()?;
        if let BlockType::Reserved = btype {
            return Err(BlockHeaderReadError::FoundReservedBlock);
        }

        let block_size = self.block_content_size()?;
        let decompressed_size = match btype {
            BlockType::Raw => block_size,
            BlockType::RLE => block_size,
            BlockType::Reserved => 0, //should be caught above, this is an error state
            BlockType::Compressed => 0, //unknown but will be smaller than 128kb (or window_size if that is smaller than 128kb)
        };
        let content_size = match btype {
            BlockType::Raw => block_size,
            BlockType::Compressed => block_size,
            BlockType::RLE => 1,
            BlockType::Reserved => 0, //should be caught above, this is an error state
        };

        let last_block = self.is_last();

        self.reset_buffer();
        self.internal_state = DecoderState::ReadyToDecodeNextBody;

        //just return 3. Blockheaders always take 3 bytes
        Ok((BlockHeader { last_block, block_type: btype, decompressed_size, content_size }, 3))
    }

    fn reset_buffer(&mut self) {
        self.header_buffer[0] = 0;
        self.header_buffer[1] = 0;
        self.header_buffer[2] = 0;
    }

    fn is_last(&self) -> bool {
        self.header_buffer[0] & 0x1 == 1
    }

    fn block_type(&self) -> Result<BlockType, BlockTypeError> {
        let t = (self.header_buffer[0] >> 1) & 0x3;
        match t {
            0 => Ok(BlockType::Raw),
            1 => Ok(BlockType::RLE),
            2 => Ok(BlockType::Compressed),
            3 => Ok(BlockType::Reserved),
            other => Err(BlockTypeError::InvalidBlocktypeNumber { num: other }),
        }
    }

    fn block_content_size(&self) -> Result<u32, BlockSizeError> {
        let val = self.block_content_size_unchecked();
        if val > MAX_BLOCK_SIZE {
            Err(BlockSizeError::BlockSizeTooLarge { size: val })
        } else {
            Ok(val)
        }
    }

    fn block_content_size_unchecked(&self) -> u32 {
        u32::from(self.header_buffer[0] >> 3) //push out type and last_block flags. Retain 5 bit
            | (u32::from(self.header_buffer[1]) << 5)
            | (u32::from(self.header_buffer[2]) << 13)
    }
}
