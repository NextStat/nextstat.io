//! Basket (compressed data block) reading for TTree branches.

use crate::decompress::decompress;
use crate::error::{Result, RootError};
use crate::key::Key;
use crate::rbuffer::RBuffer;

/// Read and decompress a single basket from the file.
///
/// Returns the decompressed payload (big-endian encoded values).
pub fn read_basket_data(
    file_data: &[u8],
    seek: u64,
    is_large: bool,
) -> Result<Vec<u8>> {
    let pos = seek as usize;
    if pos >= file_data.len() {
        return Err(RootError::BufferUnderflow {
            offset: pos,
            need: 1,
            have: 0,
        });
    }

    // Read the TKey header for this basket
    let mut r = RBuffer::new(file_data);
    r.set_pos(pos);
    let key = Key::read(&mut r, is_large)?;

    let key_end = pos + key.n_bytes as usize;
    if key_end > file_data.len() {
        return Err(RootError::BufferUnderflow {
            offset: pos,
            need: key.n_bytes as usize,
            have: file_data.len() - pos,
        });
    }

    let obj_start = pos + key.key_len as usize;
    let compressed_data = &file_data[obj_start..key_end];
    let compressed_len = key.n_bytes as usize - key.key_len as usize;

    if key.obj_len as usize != compressed_len {
        // Compressed — decompress
        let full = decompress(compressed_data, key.obj_len as usize)?;
        // The decompressed payload may contain entry offsets at the end.
        // For fixed-size leaf types, it's just flat big-endian data.
        // We return the full payload and let the caller decide.
        Ok(full)
    } else {
        // Uncompressed
        Ok(compressed_data.to_vec())
    }
}

/// Extract the raw data portion of a decompressed basket payload.
///
/// If `entry_offset_len > 0`, the last portion of the payload is an array
/// of entry offsets (for variable-length data). We strip those for
/// fixed-size types.
pub fn strip_entry_offsets(
    payload: &[u8],
    n_entries: u64,
    entry_offset_len: usize,
) -> &[u8] {
    if entry_offset_len > 0 && n_entries > 0 {
        // Entry offsets are i32 (4 bytes each) at the end
        let offset_bytes = n_entries as usize * 4;
        if payload.len() > offset_bytes {
            return &payload[..payload.len() - offset_bytes];
        }
    }
    payload
}
