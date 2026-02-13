//! Fused sequence decode + execute.
//!
//! Instead of two passes (decode all sequences into `Vec<Sequence>`, then execute),
//! this module decodes and executes each sequence immediately. This eliminates:
//! - The intermediate `Vec<Sequence>` allocation
//! - A full second pass over all decoded sequences
//! - Double memory traffic through the sequence data

use crate::bit_io::BitReaderReversed;
use crate::blocks::sequence_section::SequencesHeader;
use crate::decoding::errors::{DecodeSequenceError, DecompressBlockError, ExecuteSequencesError};
use crate::decoding::flat_decode_buffer::FlatDecodeBuffer;
use crate::decoding::scratch::FSEScratch;
use crate::decoding::sequence_execution::do_offset_history;
use crate::decoding::sequence_section_decoder::finish_sequence_bitstream;
use crate::decoding::sequence_section_decoder::{build_seq_symbols, maybe_update_fse_tables};

pub trait FusedOutputBuffer {
    fn push(&mut self, data: &[u8]) -> Result<(), ExecuteSequencesError>;
    fn repeat(&mut self, offset: usize, match_length: usize) -> Result<(), ExecuteSequencesError>;

    #[inline(always)]
    fn ensure_capacity(&mut self, _extra: usize) -> Result<(), ExecuteSequencesError> {
        Ok(())
    }

    #[inline(always)]
    fn push_unchecked(&mut self, data: &[u8]) -> Result<(), ExecuteSequencesError> {
        self.push(data)
    }

    #[inline(always)]
    fn repeat_unchecked(
        &mut self,
        offset: usize,
        match_length: usize,
    ) -> Result<(), ExecuteSequencesError> {
        self.repeat(offset, match_length)
    }
}

impl FusedOutputBuffer for FlatDecodeBuffer {
    #[inline(always)]
    fn push(&mut self, data: &[u8]) -> Result<(), ExecuteSequencesError> {
        self.push(data);
        Ok(())
    }

    #[inline(always)]
    fn repeat(&mut self, offset: usize, match_length: usize) -> Result<(), ExecuteSequencesError> {
        self.repeat(offset, match_length).map_err(ExecuteSequencesError::DecodebufferError)
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

    // Build merged SeqSymbol tables (one lookup = base_value + extra bits + state transition).
    // For RLE modes, this creates a 1-entry table with nb_bits=0.
    build_seq_symbols(fse);

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

    fused_unified(section, &mut br, fse, buffer, offset_hist, literals_buffer)
}

/// Unified fused decode+execute using merged SeqSymbol tables.
/// Handles both RLE and non-RLE modes without branching — RLE modes
/// have accuracy_log=0 (no init bits) and nb_bits=0 (state stays at 0).
fn fused_unified(
    section: &SequencesHeader,
    br: &mut BitReaderReversed<'_>,
    fse: &FSEScratch,
    buffer: &mut impl FusedOutputBuffer,
    offset_hist: &mut [u32; 3],
    literals_buffer: &[u8],
) -> Result<(), DecompressBlockError> {
    let seq_ll = &fse.seq_ll;
    let seq_ml = &fse.seq_ml;
    let seq_of = &fse.seq_of;

    // Init states: for RLE, accuracy_log=0 → get_bits(0)=0 → state=0
    let mut ll_state = br.get_bits(fse.seq_ll_log) as usize;
    let mut of_state = br.get_bits(fse.seq_of_log) as usize;
    let mut ml_state = br.get_bits(fse.seq_ml_log) as usize;

    let num_sequences = section.num_sequences as usize;
    let mut literals_copy_counter = 0usize;

    // Main loop: all sequences except the last (avoids `seq_idx + 1 < num_sequences` branch).
    // Also removes `ml > 0` check (zstd guarantees ml >= MINMATCH=3 for all sequences)
    // and merges the two `ll > 0` checks into one block.
    let last_seq = if num_sequences > 0 { num_sequences - 1 } else { 0 };

    for seq_idx in 0..num_sequences {
        let ll_entry = unsafe { *seq_ll.get_unchecked(ll_state) };
        let ml_entry = unsafe { *seq_ml.get_unchecked(ml_state) };
        let of_entry = unsafe { *seq_of.get_unchecked(of_state) };

        let (obits, ml_add, ll_add) = br.get_bits_triple(
            of_entry.nb_additional_bits,
            ml_entry.nb_additional_bits,
            ll_entry.nb_additional_bits,
        );

        let offset = obits as u32 + of_entry.base_value;
        let ll = ll_entry.base_value + ll_add as u32;
        let ml = ml_entry.base_value + ml_add as u32;

        let actual_offset = do_offset_history(offset, ll, offset_hist);
        if actual_offset == 0 {
            return Err(ExecuteSequencesError::ZeroOffset.into());
        }

        let ll_usize = ll as usize;
        let ml_usize = ml as usize;

        buffer.ensure_capacity(ll_usize + ml_usize).map_err(DecompressBlockError::from)?;

        if ll_usize > 0 {
            let high = literals_copy_counter + ll_usize;
            if high > literals_buffer.len() {
                return Err(ExecuteSequencesError::NotEnoughBytesForSequence {
                    wanted: high,
                    have: literals_buffer.len(),
                }
                .into());
            }
            buffer
                .push_unchecked(&literals_buffer[literals_copy_counter..high])
                .map_err(DecompressBlockError::from)?;
            literals_copy_counter = high;
        }

        buffer
            .repeat_unchecked(actual_offset as usize, ml_usize)
            .map_err(DecompressBlockError::from)?;

        // FSE state update (skipped for last sequence)
        if seq_idx < last_seq {
            let (ll_bits, ml_bits, of_bits) =
                br.get_bits_triple(ll_entry.nb_bits, ml_entry.nb_bits, of_entry.nb_bits);

            ll_state = ll_entry.next_state_base as usize + ll_bits as usize;
            ml_state = ml_entry.next_state_base as usize + ml_bits as usize;
            of_state = of_entry.next_state_base as usize + of_bits as usize;
        }
    }

    // Copy remaining literals after last sequence
    if literals_copy_counter < literals_buffer.len() {
        let remaining = literals_buffer.len() - literals_copy_counter;
        buffer.ensure_capacity(remaining).map_err(DecompressBlockError::from)?;
        buffer
            .push_unchecked(&literals_buffer[literals_copy_counter..])
            .map_err(DecompressBlockError::from)?;
    }

    if br.bits_remaining() > 0 {
        finish_sequence_bitstream(br).map_err(DecompressBlockError::from)?;
    }
    Ok(())
}
