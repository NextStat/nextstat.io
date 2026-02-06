//! TDirectory parsing and key-list navigation.

use crate::error::Result;
use crate::key::Key;
use crate::rbuffer::RBuffer;

/// A parsed TDirectory: an ordered list of TKeys.
#[derive(Debug, Clone)]
pub struct Directory {
    keys: Vec<Key>,
}

impl Directory {
    /// Read the key list from the file at `seek_keys`.
    ///
    /// The key list starts with a TKey header for the list itself, then
    /// a u32 `nkeys`, followed by `nkeys` TKey records.
    pub fn read_key_list(
        file_data: &[u8],
        seek_keys: usize,
        _nbytes_keys: usize,
        is_large: bool,
    ) -> Result<Self> {
        let mut r = RBuffer::new(file_data);
        r.set_pos(seek_keys);

        // The key-list itself is stored as a TKey; skip its header.
        let _list_key = Key::read(&mut r, is_large)?;

        // Number of keys
        let nkeys = r.read_u32()? as usize;

        let mut keys = Vec::with_capacity(nkeys);
        for _ in 0..nkeys {
            keys.push(Key::read(&mut r, is_large)?);
        }

        Ok(Directory { keys })
    }

    /// Read directory from the decompressed payload of a TDirectoryFile key.
    ///
    /// The payload starts with a TDirectory streamer that contains
    /// `seek_keys` and `nbytes_keys` for the subdirectory's key list.
    pub fn read_from_payload(
        payload: &[u8],
        _parent_key: &Key,
        is_large: bool,
        file_data: &[u8],
    ) -> Result<Self> {
        let mut r = RBuffer::new(payload);

        // TDirectory streamer
        let dir_version = r.read_u16()?;
        let _datime_c = r.read_u32()?;
        let _datime_m = r.read_u32()?;
        let _nbytes_keys = r.read_u32()?;
        let _nbytes_name = r.read_u32()?;

        let is_dir_large = dir_version > 1000;

        let (_seek_dir, _seek_parent, seek_keys);
        if is_dir_large {
            _seek_dir = r.read_u64()?;
            _seek_parent = r.read_u64()?;
            seek_keys = r.read_u64()?;
        } else {
            _seek_dir = r.read_u32()? as u64;
            _seek_parent = r.read_u32()? as u64;
            seek_keys = r.read_u32()? as u64;
        }

        if seek_keys == 0 {
            return Ok(Directory { keys: Vec::new() });
        }

        Self::read_key_list(file_data, seek_keys as usize, _nbytes_keys as usize, is_large)
    }

    /// Access the list of keys.
    pub fn keys(&self) -> &[Key] {
        &self.keys
    }

    /// Find a key by name (returns the last cycle — highest version).
    pub fn find_key(&self, name: &str) -> Option<&Key> {
        self.keys
            .iter()
            .filter(|k| k.name == name)
            .max_by_key(|k| k.cycle)
    }
}
