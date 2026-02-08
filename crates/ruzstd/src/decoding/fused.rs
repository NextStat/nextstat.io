//! Fused sequence decode + execute.
//!
//! Instead of two passes (decode all sequences into `Vec<Sequence>`, then execute),
//! this module decodes and executes each sequence immediately. This eliminates:
//! - The intermediate `Vec<Sequence>` allocation
//! - A full second pass over all decoded sequences
//! - Double memory traffic through the sequence data

use crate::bit_io::BitReaderReversed;
use crate::blocks::sequence_section::{SequencesHeader, MAX_OFFSET_CODE};
use crate::decoding::flat_decode_buffer::FlatDecodeBuffer;
use crate::decoding::errors::{
    DecodeSequenceError, DecompressBlockError, ExecuteSequencesError,
};
use crate::decoding::scratch::FSEScratch;
use crate::decoding::sequence_execution::do_offset_history;
use crate::decoding::sequence_section_decoder::{
    lookup_ll_code, lookup_ml_code, maybe_update_fse_tables,
};
use crate::fse::FSEDecoder;

pub trait FusedOutputBuffer {
    fn push(&mut self, data: &[u8]) -> Result<(), ExecuteSequencesError>;
    fn repeat(&mut self, offset: usize, match_length: usize) -> Result<(), ExecuteSequencesError>;
}

impl FusedOutputBuffer for FlatDecodeBuffer {
    #[inline(always)]
    fn push(&mut self, data: &[u8]) -> Result<(), ExecuteSequencesError> {
        self.push(data);
        Ok(())
    }

    #[inline(always)]
    fn repeat(&mut self, offset: usize, match_length: usize) -> Result<(), ExecuteSequencesError> {
        self.repeat(offset, match_length)
            .map_err(ExecuteSequencesError::DecodebufferError)
    }
}

/// Fused decode + execute: decodes sequences from the bitstream and immediately
/// executes them (copy literals + match copy) without buffering into `Vec<Sequence>`.
///
/// Takes individual fields to allow the borrow checker to see that
/// `source` (from `block_content_buffer`) and `fse`/`buffer`/etc. are disjoint.
pub fn decode_and_execute_sequences(
    section: &SequencesHeader,
    source: &[u8],
    fse: &mut FSEScratch,
    buffer: &mut FlatDecodeBuffer,
    offset_hist: &mut [u32; 3],
    literals_buffer: &[u8],
) -> Result<(), DecompressBlockError> {
    decode_and_execute_sequences_impl(section, source, fse, buffer, offset_hist, literals_buffer)
}

pub(crate) fn decode_and_execute_sequences_to(
    section: &SequencesHeader,
    source: &[u8],
    fse: &mut FSEScratch,
    buffer: &mut impl FusedOutputBuffer,
    offset_hist: &mut [u32; 3],
    literals_buffer: &[u8],
) -> Result<(), DecompressBlockError> {
    decode_and_execute_sequences_impl(section, source, fse, buffer, offset_hist, literals_buffer)
}

fn decode_and_execute_sequences_impl(
    section: &SequencesHeader,
    source: &[u8],
    fse: &mut FSEScratch,
    buffer: &mut impl FusedOutputBuffer,
    offset_hist: &mut [u32; 3],
    literals_buffer: &[u8],
) -> Result<(), DecompressBlockError> {
    let bytes_read = maybe_update_fse_tables(section, source, fse)?;

    let bit_stream = &source[bytes_read..];
    let mut br = BitReaderReversed::new(bit_stream);

    // Skip the 0 padding at the end of the last byte and the sentinel 1 bit
    let mut skipped_bits = 0;
    loop {
        let val = br.get_bits(1);
        skipped_bits += 1;
        if val == 1 || skipped_bits > 8 {
            break;
        }
    }
    if skipped_bits > 8 {
        return Err(DecodeSequenceError::ExtraPadding { skipped_bits }.into());
    }

    if fse.ll_rle.is_some() || fse.ml_rle.is_some() || fse.of_rle.is_some() {
        fused_with_rle(section, &mut br, fse, buffer, offset_hist, literals_buffer)
    } else {
        fused_without_rle(section, &mut br, fse, buffer, offset_hist, literals_buffer)
    }
}

fn fused_without_rle(
    section: &SequencesHeader,
    br: &mut BitReaderReversed<'_>,
    fse: &FSEScratch,
    buffer: &mut impl FusedOutputBuffer,
    offset_hist: &mut [u32; 3],
    literals_buffer: &[u8],
) -> Result<(), DecompressBlockError> {
    let mut ll_dec = FSEDecoder::new(&fse.literal_lengths);
    let mut ml_dec = FSEDecoder::new(&fse.match_lengths);
    let mut of_dec = FSEDecoder::new(&fse.offsets);

    ll_dec.init_state(br).map_err(DecodeSequenceError::from)?;
    of_dec.init_state(br).map_err(DecodeSequenceError::from)?;
    ml_dec.init_state(br).map_err(DecodeSequenceError::from)?;

    let num_sequences = section.num_sequences as usize;
    let mut literals_copy_counter = 0usize;

    for seq_idx in 0..num_sequences {
        // --- Decode ---
        let ll_code = ll_dec.decode_symbol();
        let ml_code = ml_dec.decode_symbol();
        let of_code = of_dec.decode_symbol();

        let (ll_value, ll_num_bits) = lookup_ll_code(ll_code);
        let (ml_value, ml_num_bits) = lookup_ml_code(ml_code);

        if of_code > MAX_OFFSET_CODE {
            return Err(DecodeSequenceError::UnsupportedOffset {
                offset_code: of_code,
            }
            .into());
        }

        let (obits, ml_add, ll_add) = br.get_bits_triple(of_code, ml_num_bits, ll_num_bits);
        let offset = obits as u32 + (1u32 << of_code);

        if offset == 0 {
            return Err(DecodeSequenceError::ZeroOffset.into());
        }

        let ll = ll_value + ll_add as u32;
        let ml = ml_value + ml_add as u32;

        // --- Execute immediately ---

        // 1. Copy literals
        if ll > 0 {
            let high = literals_copy_counter + ll as usize;
            if high > literals_buffer.len() {
                return Err(ExecuteSequencesError::NotEnoughBytesForSequence {
                    wanted: high,
                    have: literals_buffer.len(),
                }
                .into());
            }
            buffer
                .push(&literals_buffer[literals_copy_counter..high])
                .map_err(DecompressBlockError::from)?;
            literals_copy_counter += ll as usize;
        }

        // 2. Match copy
        let actual_offset = do_offset_history(offset, ll, offset_hist);
        if actual_offset == 0 {
            return Err(ExecuteSequencesError::ZeroOffset.into());
        }
        if ml > 0 {
            buffer
                .repeat(actual_offset as usize, ml as usize)
                .map_err(DecompressBlockError::from)?;
        }

        // Interleaved FSE state update: read all 3 bit values in one refill,
        // then do 3 independent table lookups that the CPU can pipeline.
        // Max total bits: 9 (LL) + 9 (ML) + 8 (OF) = 26, well within 56-bit refill.
        if seq_idx + 1 < num_sequences {
            let ll_nb = ll_dec.state.num_bits;
            let ml_nb = ml_dec.state.num_bits;
            let of_nb = of_dec.state.num_bits;

            let (ll_bits, ml_bits, of_bits) = br.get_bits_triple(ll_nb, ml_nb, of_nb);

            ll_dec.update_state_from_bits(ll_bits);
            ml_dec.update_state_from_bits(ml_bits);
            of_dec.update_state_from_bits(of_bits);
        }

        if br.bits_remaining() < 0 {
            return Err(DecodeSequenceError::NotEnoughBytesForNumSequences.into());
        }
    }

    // Copy remaining literals after last sequence
    if literals_copy_counter < literals_buffer.len() {
        buffer
            .push(&literals_buffer[literals_copy_counter..])
            .map_err(DecompressBlockError::from)?;
    }

    if br.bits_remaining() > 0 {
        Err(DecodeSequenceError::ExtraBits {
            bits_remaining: br.bits_remaining(),
        }
        .into())
    } else {
        Ok(())
    }
}

fn fused_with_rle(
    section: &SequencesHeader,
    br: &mut BitReaderReversed<'_>,
    fse: &FSEScratch,
    buffer: &mut impl FusedOutputBuffer,
    offset_hist: &mut [u32; 3],
    literals_buffer: &[u8],
) -> Result<(), DecompressBlockError> {
    let mut ll_dec = FSEDecoder::new(&fse.literal_lengths);
    let mut ml_dec = FSEDecoder::new(&fse.match_lengths);
    let mut of_dec = FSEDecoder::new(&fse.offsets);

    if fse.ll_rle.is_none() {
        ll_dec.init_state(br).map_err(DecodeSequenceError::from)?;
    }
    if fse.of_rle.is_none() {
        of_dec.init_state(br).map_err(DecodeSequenceError::from)?;
    }
    if fse.ml_rle.is_none() {
        ml_dec.init_state(br).map_err(DecodeSequenceError::from)?;
    }

    let num_sequences = section.num_sequences as usize;
    let mut literals_copy_counter = 0usize;

    for seq_idx in 0..num_sequences {
        // --- Decode ---
        let ll_code = if let Some(rle) = fse.ll_rle {
            rle
        } else {
            ll_dec.decode_symbol()
        };
        let ml_code = if let Some(rle) = fse.ml_rle {
            rle
        } else {
            ml_dec.decode_symbol()
        };
        let of_code = if let Some(rle) = fse.of_rle {
            rle
        } else {
            of_dec.decode_symbol()
        };

        let (ll_value, ll_num_bits) = lookup_ll_code(ll_code);
        let (ml_value, ml_num_bits) = lookup_ml_code(ml_code);

        if of_code > MAX_OFFSET_CODE {
            return Err(DecodeSequenceError::UnsupportedOffset {
                offset_code: of_code,
            }
            .into());
        }

        let (obits, ml_add, ll_add) = br.get_bits_triple(of_code, ml_num_bits, ll_num_bits);
        let offset = obits as u32 + (1u32 << of_code);

        if offset == 0 {
            return Err(DecodeSequenceError::ZeroOffset.into());
        }

        let ll = ll_value + ll_add as u32;
        let ml = ml_value + ml_add as u32;

        // --- Execute immediately ---

        // 1. Copy literals
        if ll > 0 {
            let high = literals_copy_counter + ll as usize;
            if high > literals_buffer.len() {
                return Err(ExecuteSequencesError::NotEnoughBytesForSequence {
                    wanted: high,
                    have: literals_buffer.len(),
                }
                .into());
            }
            buffer
                .push(&literals_buffer[literals_copy_counter..high])
                .map_err(DecompressBlockError::from)?;
            literals_copy_counter += ll as usize;
        }

        // 2. Match copy
        let actual_offset = do_offset_history(offset, ll, offset_hist);
        if actual_offset == 0 {
            return Err(ExecuteSequencesError::ZeroOffset.into());
        }
        if ml > 0 {
            buffer
                .repeat(actual_offset as usize, ml as usize)
                .map_err(DecompressBlockError::from)?;
        }

        // Interleaved FSE state update for non-RLE decoders.
        // Collect bit widths, read all bits in one batch, then do independent table lookups.
        if seq_idx + 1 < num_sequences {
            let ll_nb = if fse.ll_rle.is_none() { ll_dec.state.num_bits } else { 0 };
            let ml_nb = if fse.ml_rle.is_none() { ml_dec.state.num_bits } else { 0 };
            let of_nb = if fse.of_rle.is_none() { of_dec.state.num_bits } else { 0 };

            let (ll_bits, ml_bits, of_bits) = br.get_bits_triple(ll_nb, ml_nb, of_nb);

            if fse.ll_rle.is_none() { ll_dec.update_state_from_bits(ll_bits); }
            if fse.ml_rle.is_none() { ml_dec.update_state_from_bits(ml_bits); }
            if fse.of_rle.is_none() { of_dec.update_state_from_bits(of_bits); }
        }

        if br.bits_remaining() < 0 {
            return Err(DecodeSequenceError::NotEnoughBytesForNumSequences.into());
        }
    }

    // Copy remaining literals after last sequence
    if literals_copy_counter < literals_buffer.len() {
        buffer
            .push(&literals_buffer[literals_copy_counter..])
            .map_err(DecompressBlockError::from)?;
    }

    if br.bits_remaining() > 0 {
        Err(DecodeSequenceError::ExtraBits {
            bits_remaining: br.bits_remaining(),
        }
        .into())
    } else {
        Ok(())
    }
}
