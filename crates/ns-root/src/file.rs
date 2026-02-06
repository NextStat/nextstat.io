//! TFile header parsing and top-level ROOT file interface.

use std::fs;
use std::path::{Path, PathBuf};

use crate::decompress::decompress;
use crate::directory::Directory;
use crate::error::{Result, RootError};
use crate::histogram::Histogram;
use crate::key::{Key, KeyInfo};
use crate::objects;
use crate::rbuffer::RBuffer;

/// Parsed ROOT file header.
struct FileHeader {
    /// ROOT file format version.
    #[allow(dead_code)]
    version: u32,
    /// Offset of first data record.
    #[allow(dead_code)]
    begin: u64,
    /// Offset of first free-segment record.
    #[allow(dead_code)]
    end: u64,
    /// Offset where top-level directory keys are stored.
    seek_keys: u64,
    /// Number of bytes in the key list.
    nbytes_keys: u32,
    /// Whether the file uses large (64-bit) seek pointers (version >= 1000000).
    is_large: bool,
}

/// A ROOT file opened for reading histograms.
pub struct RootFile {
    /// Raw file bytes.
    data: Vec<u8>,
    /// Parsed header.
    header: FileHeader,
    /// Path for diagnostics.
    #[allow(dead_code)]
    path: PathBuf,
}

const ROOT_MAGIC: &[u8; 4] = b"root";

impl RootFile {
    /// Open and parse a ROOT file from disk.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let data = fs::read(&path)?;
        Self::from_bytes(data, path)
    }

    /// Parse a ROOT file from a byte vector (for testing).
    pub fn from_bytes(data: Vec<u8>, path: PathBuf) -> Result<Self> {
        if data.len() < 64 {
            return Err(RootError::BadMagic);
        }
        if &data[0..4] != ROOT_MAGIC {
            return Err(RootError::BadMagic);
        }

        let header = Self::parse_header(&data)?;
        Ok(Self { data, header, path })
    }

    fn parse_header(data: &[u8]) -> Result<FileHeader> {
        let mut r = RBuffer::new(data);
        r.skip(4)?; // magic

        let version = r.read_u32()?;
        let is_large = version >= 1_000_000;

        let begin = r.read_u32()? as u64;
        let (end, seek_free, nbytes_free, _nfree);
        if is_large {
            end = r.read_u64()?;
            seek_free = r.read_u64()?;
            nbytes_free = r.read_u32()?;
            _nfree = r.read_u32()?;
        } else {
            end = r.read_u32()? as u64;
            seek_free = r.read_u32()? as u64;
            nbytes_free = r.read_u32()?;
            _nfree = r.read_u32()?;
        }

        let _nbytes_name = r.read_u8()?;
        let _units = r.read_u8()?;
        let _compress = r.read_u32()?;

        let (seek_keys, nbytes_keys);
        if is_large {
            let _seek_info = r.read_u64()?;
            let _nbytes_info = r.read_u32()?;
            // For large files, seek_keys is at a different offset.
            // We need to navigate the top-level TDirectory to find it.
            // Fall through to TDirectory parsing below.
            seek_keys = 0;
            nbytes_keys = 0;
        } else {
            let _seek_info = r.read_u32()? as u64;
            let _nbytes_info = r.read_u32()?;
            seek_keys = 0;
            nbytes_keys = 0;
        }

        let _ = seek_free;
        let _ = nbytes_free;

        // Parse the top-level TDirectory header (embedded right after the file header).
        // The TDirectory at `begin` contains seek_keys and nbytes_keys.
        let (real_seek_keys, real_nbytes_keys) = Self::parse_top_directory(data, begin, is_large)?;

        Ok(FileHeader {
            version,
            begin,
            end,
            seek_keys: if real_seek_keys != 0 { real_seek_keys } else { seek_keys },
            nbytes_keys: if real_nbytes_keys != 0 { real_nbytes_keys } else { nbytes_keys },
            is_large,
        })
    }

    /// Parse the top-level TKey + TDirectory at `begin` to get seek_keys/nbytes_keys.
    fn parse_top_directory(data: &[u8], begin: u64, is_large: bool) -> Result<(u64, u32)> {
        let mut r = RBuffer::new(data);
        r.set_pos(begin as usize);

        // Read the TKey header
        let _key = Key::read(&mut r, is_large)?;

        // Now we're at the start of the TDirectory payload.
        // TDirectory streamer: version, datime_c, datime_m, nbytes_keys, nbytes_name, seek_dir, seek_parent, seek_keys
        let dir_version = r.read_u16()?;
        let _datime_c = r.read_u32()?;
        let _datime_m = r.read_u32()?;
        let nbytes_keys = r.read_u32()?;
        let _nbytes_name = r.read_u32()?;

        let is_dir_large = dir_version > 1000;

        if is_dir_large {
            let _seek_dir = r.read_u64()?;
            let _seek_parent = r.read_u64()?;
            let seek_keys = r.read_u64()?;
            Ok((seek_keys, nbytes_keys))
        } else {
            let _seek_dir = r.read_u32()? as u64;
            let _seek_parent = r.read_u32()? as u64;
            let seek_keys = r.read_u32()? as u64;
            Ok((seek_keys, nbytes_keys))
        }
    }

    /// List all keys in the top-level directory.
    pub fn list_keys(&self) -> Result<Vec<KeyInfo>> {
        let dir = self.read_top_directory()?;
        Ok(dir.keys().iter().map(KeyInfo::from_key).collect())
    }

    /// Get a histogram by its full path (e.g. `"subdir/hist_name"`).
    pub fn get_histogram(&self, path: &str) -> Result<Histogram> {
        let dir = self.read_top_directory()?;
        self.resolve_histogram(&dir, path)
    }

    fn read_top_directory(&self) -> Result<Directory> {
        Directory::read_key_list(
            &self.data,
            self.header.seek_keys as usize,
            self.header.nbytes_keys as usize,
            self.header.is_large,
        )
    }

    fn resolve_histogram(&self, dir: &Directory, path: &str) -> Result<Histogram> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        if parts.is_empty() {
            return Err(RootError::KeyNotFound(path.to_string()));
        }

        if parts.len() == 1 {
            return self.read_histogram_from_dir(dir, parts[0]);
        }

        // Navigate subdirectories
        let mut current_dir = dir.clone();
        for &part in &parts[..parts.len() - 1] {
            let key = current_dir.find_key(part).ok_or_else(|| {
                RootError::KeyNotFound(format!("{} (in path {})", part, path))
            })?;

            if key.class_name != "TDirectoryFile" && key.class_name != "TDirectory" {
                return Err(RootError::Deserialization(format!(
                    "'{}' is not a directory (class: {})",
                    part, key.class_name
                )));
            }

            let payload = self.read_key_payload(key)?;
            current_dir = Directory::read_from_payload(&payload, key, self.header.is_large, &self.data)?;
        }

        self.read_histogram_from_dir(&current_dir, parts[parts.len() - 1])
    }

    fn read_histogram_from_dir(&self, dir: &Directory, name: &str) -> Result<Histogram> {
        let key = dir.find_key(name).ok_or_else(|| {
            RootError::KeyNotFound(name.to_string())
        })?;

        let payload = self.read_key_payload(key)?;
        objects::read_histogram(&payload, &key.class_name)
    }

    fn read_key_payload(&self, key: &Key) -> Result<Vec<u8>> {
        let seek = key.seek_key as usize;
        if seek + key.n_bytes as usize > self.data.len() {
            return Err(RootError::BufferUnderflow {
                offset: seek,
                need: key.n_bytes as usize,
                have: self.data.len().saturating_sub(seek),
            });
        }

        let key_slice = &self.data[seek..seek + key.n_bytes as usize];

        // Read past the key header to get the raw object data.
        let obj_start = key.key_len as usize;
        let compressed_data = &key_slice[obj_start..];

        if key.obj_len as usize != (key.n_bytes as usize - key.key_len as usize) {
            // Data is compressed
            decompress(compressed_data, key.obj_len as usize)
        } else {
            // Data is uncompressed
            Ok(compressed_data.to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_non_root_file() {
        let data = vec![0u8; 100];
        let result = RootFile::from_bytes(data, PathBuf::from("test.root"));
        assert!(matches!(result, Err(RootError::BadMagic)));
    }

    #[test]
    fn reject_too_small() {
        let data = b"root".to_vec();
        let result = RootFile::from_bytes(data, PathBuf::from("test.root"));
        assert!(matches!(result, Err(RootError::BadMagic)));
    }
}
