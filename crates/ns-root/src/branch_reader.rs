//! Column-oriented data extraction from TTree branches.

use rayon::prelude::*;

use crate::basket::read_basket_data;
use crate::error::{Result, RootError};
use crate::tree::{BranchInfo, LeafType};

/// A jagged (variable-length) column: flat values + per-entry offsets.
///
/// `offsets` has length `n_entries + 1`. Entry `i` has values
/// `flat[offsets[i]..offsets[i+1]]`.
#[derive(Debug, Clone)]
pub struct JaggedCol {
    /// Flat array of all values across all entries.
    pub flat: Vec<f64>,
    /// Entry boundaries: `offsets.len() == n_entries + 1`.
    pub offsets: Vec<usize>,
}

impl JaggedCol {
    /// Get element `index` of entry `row`. Returns `oor` for out-of-range.
    pub fn get(&self, row: usize, index: usize, oor: f64) -> f64 {
        let start = self.offsets[row];
        let end = self.offsets[row + 1];
        let len = end - start;
        if index >= len {
            oor
        } else {
            self.flat[start + index]
        }
    }

    /// Number of entries.
    pub fn n_entries(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
}

/// Reader for extracting column data from a TTree branch.
pub struct BranchReader<'a> {
    file_data: &'a [u8],
    branch: &'a BranchInfo,
    is_large: bool,
}

impl<'a> BranchReader<'a> {
    /// Create a new branch reader.
    pub fn new(file_data: &'a [u8], branch: &'a BranchInfo, is_large: bool) -> Self {
        Self { file_data, branch, is_large }
    }

    /// Read all entries as `f64`, converting from the native type.
    pub fn as_f64(&self) -> Result<Vec<f64>> {
        let raw_baskets = self.read_all_baskets()?;
        decode_as_f64(&raw_baskets, self.branch.leaf_type)
    }

    /// Read a single indexed element per entry as `f64` for array/jagged branches.
    ///
    /// This supports branch names referenced from expressions like `jet_pt[0]` by
    /// materializing a scalar column.
    ///
    /// If an entry has fewer than `index + 1` elements, `out_of_range_value` is used.
    pub fn as_f64_indexed(&self, index: usize, out_of_range_value: f64) -> Result<Vec<f64>> {
        // Fast path: fixed-size arrays (no entry offsets, but basket contains N = entries * len).
        if self.branch.entry_offset_len == 0 {
            let flat = self.as_f64()?;
            let entries = self.branch.entries as usize;
            if entries == 0 {
                return Ok(Vec::new());
            }

            if flat.len() == entries {
                if index == 0 {
                    return Ok(flat);
                }
                return Err(RootError::TypeMismatch(format!(
                    "indexed access [{}] requested for scalar branch '{}'",
                    index, self.branch.name
                )));
            }

            if flat.len() % entries != 0 {
                return Err(RootError::Deserialization(format!(
                    "branch '{}' decoded to {} values, not divisible by entries={}",
                    self.branch.name,
                    flat.len(),
                    entries
                )));
            }

            let len = flat.len() / entries;
            if index >= len {
                // ROOT/TTreeFormula returns 0.0 for OOR on both fixed and jagged arrays.
                return Ok(vec![out_of_range_value; entries]);
            }

            let mut out_col: Vec<f64> = Vec::with_capacity(entries);
            for i in 0..entries {
                out_col.push(flat[i * len + index]);
            }
            return Ok(out_col);
        }

        let elem_size = self.branch.leaf_type.byte_size();
        if elem_size == 0 {
            return Err(RootError::TypeMismatch(format!(
                "unsupported leaf type for indexed access: {:?}",
                self.branch.leaf_type
            )));
        }

        let raw_baskets = self.read_all_baskets()?;
        let mut out: Vec<f64> = Vec::with_capacity(self.branch.entries as usize);

        for (i, payload) in raw_baskets.iter().enumerate() {
            let n_entries = self
                .branch
                .basket_entry
                .get(i + 1)
                .copied()
                .unwrap_or(self.branch.entries)
                .saturating_sub(self.branch.basket_entry.get(i).copied().unwrap_or(0))
                as usize;
            if n_entries == 0 {
                continue;
            }
            decode_indexed_from_payload(
                payload,
                self.branch.leaf_type,
                self.branch.entry_offset_len,
                n_entries,
                index,
                out_of_range_value,
                &mut out,
            )?;
        }

        Ok(out)
    }

    /// Read all entries as `f64` using parallel basket decompression.
    pub fn as_f64_par(&self) -> Result<Vec<f64>> {
        let raw_baskets = self.read_all_baskets_par()?;
        decode_as_f64(&raw_baskets, self.branch.leaf_type)
    }

    /// Read all entries as `f32`.
    pub fn as_f32(&self) -> Result<Vec<f32>> {
        let raw_baskets = self.read_all_baskets()?;
        decode_as_f32(&raw_baskets, self.branch.leaf_type)
    }

    /// Read all entries as `i32`.
    pub fn as_i32(&self) -> Result<Vec<i32>> {
        let raw_baskets = self.read_all_baskets()?;
        decode_as_i32(&raw_baskets, self.branch.leaf_type)
    }

    /// Read all entries as `i64`.
    pub fn as_i64(&self) -> Result<Vec<i64>> {
        let raw_baskets = self.read_all_baskets()?;
        decode_as_i64(&raw_baskets, self.branch.leaf_type)
    }

    /// Read all entries as a jagged column (variable-length per entry).
    ///
    /// Requires a branch with entry-offset tables (`entry_offset_len > 0`).
    /// For fixed-size arrays, all entries will have the same length.
    pub fn as_jagged_f64(&self) -> Result<JaggedCol> {
        if self.branch.entry_offset_len == 0 {
            // Fixed-size array — synthesize offsets
            let flat = self.as_f64()?;
            let entries = self.branch.entries as usize;
            if entries == 0 {
                return Ok(JaggedCol { flat: Vec::new(), offsets: vec![0] });
            }
            let elem_per_entry = if flat.len() == entries { 1 } else { flat.len() / entries };
            let mut offsets = Vec::with_capacity(entries + 1);
            for i in 0..=entries {
                offsets.push(i * elem_per_entry);
            }
            return Ok(JaggedCol { flat, offsets });
        }

        let elem_size = self.branch.leaf_type.byte_size();
        if elem_size == 0 {
            return Err(RootError::TypeMismatch(format!(
                "unsupported leaf type for jagged access: {:?}",
                self.branch.leaf_type
            )));
        }

        let raw_baskets = self.read_all_baskets()?;
        let mut flat = Vec::new();
        let mut offsets = vec![0usize];

        for (i, payload) in raw_baskets.iter().enumerate() {
            let n_entries = self
                .branch
                .basket_entry
                .get(i + 1)
                .copied()
                .unwrap_or(self.branch.entries)
                .saturating_sub(self.branch.basket_entry.get(i).copied().unwrap_or(0))
                as usize;
            if n_entries == 0 {
                continue;
            }
            decode_jagged_from_payload(
                payload,
                self.branch.leaf_type,
                self.branch.entry_offset_len,
                n_entries,
                &mut flat,
                &mut offsets,
            )?;
        }

        Ok(JaggedCol { flat, offsets })
    }

    /// Read and decompress all baskets sequentially.
    fn read_all_baskets(&self) -> Result<Vec<Vec<u8>>> {
        let mut baskets = Vec::with_capacity(self.branch.n_baskets);
        for i in 0..self.branch.n_baskets {
            let data = read_basket_data(self.file_data, self.branch.basket_seek[i], self.is_large)?;
            baskets.push(data);
        }
        Ok(baskets)
    }

    /// Read and decompress all baskets in parallel via rayon.
    fn read_all_baskets_par(&self) -> Result<Vec<Vec<u8>>> {
        let results: Vec<Result<Vec<u8>>> = (0..self.branch.n_baskets)
            .into_par_iter()
            .map(|i| read_basket_data(self.file_data, self.branch.basket_seek[i], self.is_large))
            .collect();

        results.into_iter().collect()
    }
}

// ── Decoding big-endian baskets to typed arrays ────────────────

fn decode_indexed_from_payload(
    payload: &[u8],
    leaf_type: LeafType,
    entry_offset_len: usize,
    n_entries: usize,
    index: usize,
    out_of_range_value: f64,
    out: &mut Vec<f64>,
) -> Result<()> {
    // ROOT stores an entry-offset table at the end of the basket payload for jagged
    // (variable-length) branches. In TBranch, `fEntryOffsetLen` is typically the
    // number of *bits* per offset entry (e.g. 16/32/64). We store it as-is.
    if entry_offset_len == 0 {
        return Err(RootError::TypeMismatch(
            "indexed decoding requested without entry-offset table".into(),
        ));
    }
    if !entry_offset_len.is_multiple_of(8) {
        return Err(RootError::TypeMismatch(format!(
            "unsupported fEntryOffsetLen={} (expected multiple of 8)",
            entry_offset_len
        )));
    }

    let bytes_per_offset = entry_offset_len / 8;
    if !matches!(bytes_per_offset, 2 | 4 | 8) {
        return Err(RootError::TypeMismatch(format!(
            "unsupported offset width: {} bytes (from fEntryOffsetLen={})",
            bytes_per_offset, entry_offset_len
        )));
    }

    // Observed (uproot) basket layout for jagged leaflist branches:
    //   [data bytes...][count = (n_entries+1)][offset_0..offset_n]
    // where offsets are absolute to the full TBasket buffer; subtract `offset_0` to
    // get indices into `data`.
    let n_offsets = n_entries + 1;
    let tail_words = n_offsets + 1; // +1 for the count word
    let tail_bytes = tail_words
        .checked_mul(bytes_per_offset)
        .ok_or_else(|| RootError::Deserialization("offset table size overflow".into()))?;
    if payload.len() < tail_bytes {
        return Err(RootError::Deserialization(format!(
            "basket payload too small for offset table: have {} need {}",
            payload.len(),
            tail_bytes
        )));
    }

    let data_end = payload.len() - tail_bytes;
    let data = &payload[..data_end];
    let tail = &payload[data_end..];

    let read_word_be = |b: &[u8]| -> usize {
        match bytes_per_offset {
            2 => u16::from_be_bytes(b.try_into().unwrap()) as usize,
            4 => u32::from_be_bytes(b.try_into().unwrap()) as usize,
            8 => u64::from_be_bytes(b.try_into().unwrap()) as usize,
            _ => unreachable!(),
        }
    };

    let count = read_word_be(&tail[..bytes_per_offset]);
    if count != n_entries + 1 {
        return Err(RootError::Deserialization(format!(
            "unexpected entry-offset count word: got {} want {}",
            count,
            n_entries + 1
        )));
    }

    let mut offsets: Vec<usize> = Vec::with_capacity(n_offsets);
    for i in 0..n_offsets {
        let start = bytes_per_offset * (1 + i);
        let end = start + bytes_per_offset;
        offsets.push(read_word_be(&tail[start..end]));
    }

    let base = offsets[0];
    let elem_size = leaf_type.byte_size();

    for i in 0..n_entries {
        let start = offsets[i].saturating_sub(base);
        let end = offsets[i + 1].saturating_sub(base);
        if end > data.len() || start > end {
            return Err(RootError::Deserialization(format!(
                "invalid entry offsets in basket: start={start} end={end} data_len={}",
                data.len()
            )));
        }

        let need = (index + 1) * elem_size;
        if start + need > end {
            out.push(out_of_range_value);
            continue;
        }

        let off = start + index * elem_size;
        let val = match leaf_type {
            LeafType::F64 => f64::from_be_bytes(data[off..off + 8].try_into().unwrap()),
            LeafType::F32 => f32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as f64,
            LeafType::I32 => i32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as f64,
            LeafType::I64 => i64::from_be_bytes(data[off..off + 8].try_into().unwrap()) as f64,
            LeafType::U32 => u32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as f64,
            LeafType::U64 => u64::from_be_bytes(data[off..off + 8].try_into().unwrap()) as f64,
            LeafType::I16 => i16::from_be_bytes(data[off..off + 2].try_into().unwrap()) as f64,
            LeafType::I8 => data[off] as i8 as f64,
            LeafType::Bool => {
                if data[off] != 0 {
                    1.0
                } else {
                    0.0
                }
            }
        };
        out.push(val);
    }

    Ok(())
}

/// Decode all elements of each entry into flat + offsets for jagged columns.
fn decode_jagged_from_payload(
    payload: &[u8],
    leaf_type: LeafType,
    entry_offset_len: usize,
    n_entries: usize,
    flat: &mut Vec<f64>,
    offsets: &mut Vec<usize>,
) -> Result<()> {
    if entry_offset_len == 0 {
        return Err(RootError::TypeMismatch(
            "jagged decoding requested without entry-offset table".into(),
        ));
    }
    if !entry_offset_len.is_multiple_of(8) {
        return Err(RootError::TypeMismatch(format!(
            "unsupported fEntryOffsetLen={} (expected multiple of 8)",
            entry_offset_len
        )));
    }

    let bytes_per_offset = entry_offset_len / 8;
    if !matches!(bytes_per_offset, 2 | 4 | 8) {
        return Err(RootError::TypeMismatch(format!(
            "unsupported offset width: {} bytes (from fEntryOffsetLen={})",
            bytes_per_offset, entry_offset_len
        )));
    }

    let n_offsets = n_entries + 1;
    let tail_words = n_offsets + 1;
    let tail_bytes = tail_words
        .checked_mul(bytes_per_offset)
        .ok_or_else(|| RootError::Deserialization("offset table size overflow".into()))?;
    if payload.len() < tail_bytes {
        return Err(RootError::Deserialization(format!(
            "basket payload too small for offset table: have {} need {}",
            payload.len(),
            tail_bytes
        )));
    }

    let data_end = payload.len() - tail_bytes;
    let data = &payload[..data_end];
    let tail = &payload[data_end..];

    let read_word_be = |b: &[u8]| -> usize {
        match bytes_per_offset {
            2 => u16::from_be_bytes(b.try_into().unwrap()) as usize,
            4 => u32::from_be_bytes(b.try_into().unwrap()) as usize,
            8 => u64::from_be_bytes(b.try_into().unwrap()) as usize,
            _ => unreachable!(),
        }
    };

    let count = read_word_be(&tail[..bytes_per_offset]);
    if count != n_entries + 1 {
        return Err(RootError::Deserialization(format!(
            "unexpected entry-offset count word: got {} want {}",
            count,
            n_entries + 1
        )));
    }

    let mut entry_offsets: Vec<usize> = Vec::with_capacity(n_offsets);
    for i in 0..n_offsets {
        let start = bytes_per_offset * (1 + i);
        let end = start + bytes_per_offset;
        entry_offsets.push(read_word_be(&tail[start..end]));
    }

    let base = entry_offsets[0];
    let elem_size = leaf_type.byte_size();

    for i in 0..n_entries {
        let start = entry_offsets[i].saturating_sub(base);
        let end = entry_offsets[i + 1].saturating_sub(base);
        if end > data.len() || start > end {
            return Err(RootError::Deserialization(format!(
                "invalid entry offsets in basket: start={start} end={end} data_len={}",
                data.len()
            )));
        }

        let n_elems = (end - start) / elem_size;
        for j in 0..n_elems {
            let off = start + j * elem_size;
            let val = decode_one_f64(data, off, leaf_type);
            flat.push(val);
        }
        offsets.push(flat.len());
    }

    Ok(())
}

/// Decode a single f64 value from big-endian bytes at `off`.
fn decode_one_f64(data: &[u8], off: usize, leaf_type: LeafType) -> f64 {
    match leaf_type {
        LeafType::F64 => f64::from_be_bytes(data[off..off + 8].try_into().unwrap()),
        LeafType::F32 => f32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as f64,
        LeafType::I32 => i32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as f64,
        LeafType::I64 => i64::from_be_bytes(data[off..off + 8].try_into().unwrap()) as f64,
        LeafType::U32 => u32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as f64,
        LeafType::U64 => u64::from_be_bytes(data[off..off + 8].try_into().unwrap()) as f64,
        LeafType::I16 => i16::from_be_bytes(data[off..off + 2].try_into().unwrap()) as f64,
        LeafType::I8 => data[off] as i8 as f64,
        LeafType::Bool => {
            if data[off] != 0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

fn decode_as_f64(baskets: &[Vec<u8>], leaf_type: LeafType) -> Result<Vec<f64>> {
    let elem_size = leaf_type.byte_size();
    let mut out = Vec::new();

    for basket in baskets {
        let data = basket.as_slice();
        // Number of elements based on element size
        let n = data.len() / elem_size;

        for i in 0..n {
            let offset = i * elem_size;
            let val = match leaf_type {
                LeafType::F64 => f64::from_be_bytes(data[offset..offset + 8].try_into().unwrap()),
                LeafType::F32 => {
                    f32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as f64
                }
                LeafType::I32 => {
                    i32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as f64
                }
                LeafType::I64 => {
                    i64::from_be_bytes(data[offset..offset + 8].try_into().unwrap()) as f64
                }
                LeafType::U32 => {
                    u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as f64
                }
                LeafType::U64 => {
                    u64::from_be_bytes(data[offset..offset + 8].try_into().unwrap()) as f64
                }
                LeafType::I16 => {
                    i16::from_be_bytes(data[offset..offset + 2].try_into().unwrap()) as f64
                }
                LeafType::I8 => data[offset] as i8 as f64,
                LeafType::Bool => {
                    if data[offset] != 0 {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
            out.push(val);
        }
    }

    Ok(out)
}

fn decode_as_f32(baskets: &[Vec<u8>], leaf_type: LeafType) -> Result<Vec<f32>> {
    let f64s = decode_as_f64(baskets, leaf_type)?;
    Ok(f64s.into_iter().map(|v| v as f32).collect())
}

fn decode_as_i32(baskets: &[Vec<u8>], leaf_type: LeafType) -> Result<Vec<i32>> {
    let elem_size = leaf_type.byte_size();
    let mut out = Vec::new();

    for basket in baskets {
        let data = basket.as_slice();
        let n = data.len() / elem_size;

        for i in 0..n {
            let offset = i * elem_size;
            let val = match leaf_type {
                LeafType::I32 => i32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()),
                LeafType::I16 => {
                    i16::from_be_bytes(data[offset..offset + 2].try_into().unwrap()) as i32
                }
                LeafType::I8 => data[offset] as i8 as i32,
                LeafType::Bool => {
                    if data[offset] != 0 {
                        1
                    } else {
                        0
                    }
                }
                other => {
                    return Err(RootError::TypeMismatch(format!("cannot read {:?} as i32", other)));
                }
            };
            out.push(val);
        }
    }

    Ok(out)
}

fn decode_as_i64(baskets: &[Vec<u8>], leaf_type: LeafType) -> Result<Vec<i64>> {
    let elem_size = leaf_type.byte_size();
    let mut out = Vec::new();

    for basket in baskets {
        let data = basket.as_slice();
        let n = data.len() / elem_size;

        for i in 0..n {
            let offset = i * elem_size;
            let val = match leaf_type {
                LeafType::I64 => i64::from_be_bytes(data[offset..offset + 8].try_into().unwrap()),
                LeafType::I32 => {
                    i32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as i64
                }
                LeafType::I16 => {
                    i16::from_be_bytes(data[offset..offset + 2].try_into().unwrap()) as i64
                }
                LeafType::I8 => data[offset] as i8 as i64,
                LeafType::Bool => {
                    if data[offset] != 0 {
                        1
                    } else {
                        0
                    }
                }
                other => {
                    return Err(RootError::TypeMismatch(format!("cannot read {:?} as i64", other)));
                }
            };
            out.push(val);
        }
    }

    Ok(out)
}
