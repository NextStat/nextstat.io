//! TKey parsing — the record header used by ROOT to locate objects.

use crate::error::Result;
use crate::rbuffer::RBuffer;

/// A parsed TKey record.
#[derive(Debug, Clone)]
pub struct Key {
    /// Total number of bytes in compressed object + key header.
    pub n_bytes: u32,
    /// Version of key class.
    pub version: u16,
    /// Uncompressed object length.
    pub obj_len: u32,
    /// Key creation time (ROOT datime).
    #[allow(dead_code)]
    pub datime: u32,
    /// Length of the key header itself.
    pub key_len: u16,
    /// Cycle number (ROOT versioning within a directory).
    #[allow(dead_code)]
    pub cycle: u16,
    /// Absolute position of this key in the file.
    pub seek_key: u64,
    /// Parent directory seek position.
    #[allow(dead_code)]
    pub seek_pdir: u64,
    /// Class name of the stored object.
    pub class_name: String,
    /// Object name.
    pub name: String,
    /// Object title.
    #[allow(dead_code)]
    pub title: String,
}

/// Public info about a key (for `list_keys()`).
#[derive(Debug, Clone)]
pub struct KeyInfo {
    /// Object name.
    pub name: String,
    /// Object class name (e.g. "TH1D", "TDirectoryFile").
    pub class_name: String,
    /// Cycle number.
    pub cycle: u16,
}

impl KeyInfo {
    /// Create from an internal Key.
    pub fn from_key(key: &Key) -> Self {
        Self {
            name: key.name.clone(),
            class_name: key.class_name.clone(),
            cycle: key.cycle,
        }
    }
}

impl Key {
    /// Read a TKey from the buffer at the current position.
    pub fn read(r: &mut RBuffer, is_large: bool) -> Result<Self> {
        let n_bytes = r.read_u32()?;
        let version = r.read_u16()?;
        let obj_len = r.read_u32()?;
        let datime = r.read_u32()?;
        let key_len = r.read_u16()?;
        let cycle = r.read_u16()?;

        let is_key_large = version > 1000;

        let (seek_key, seek_pdir) = if is_key_large || is_large {
            let sk = r.read_u64()?;
            let sp = r.read_u64()?;
            (sk, sp)
        } else {
            let sk = r.read_u32()? as u64;
            let sp = r.read_u32()? as u64;
            (sk, sp)
        };

        let class_name = r.read_string()?;
        let name = r.read_string()?;
        let title = r.read_string()?;

        Ok(Key {
            n_bytes,
            version,
            obj_len,
            datime,
            key_len,
            cycle,
            seek_key,
            seek_pdir,
            class_name,
            name,
            title,
        })
    }
}
