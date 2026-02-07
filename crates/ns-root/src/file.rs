//! TFile header parsing and top-level ROOT file interface.

use std::fs;
use std::path::{Path, PathBuf};

use crate::branch_reader::{BranchReader, JaggedCol};
use crate::datasource::DataSource;
use crate::decompress::decompress;
use crate::directory::Directory;
use crate::error::{Result, RootError};
use crate::histogram::{Histogram, HistogramWithFlows};
use crate::key::{Key, KeyInfo};
use crate::objects;
use crate::rbuffer::RBuffer;
use crate::tree::Tree;

/// Parsed ROOT file header.
#[allow(dead_code)]
struct FileHeader {
    /// Offset of first data record (also where top-level TKey sits).
    begin: u64,
    /// Whether the file uses large (64-bit) seek pointers (version >= 1000000).
    is_large: bool,
    /// Number of bytes for the name record (TKey + TNamed) at begin.
    nbytes_name: u32,
    /// Offset where top-level directory keys are stored.
    seek_keys: u64,
    /// Number of bytes in the key list.
    nbytes_keys: u32,
}

/// A ROOT file opened for reading histograms and trees.
pub struct RootFile {
    /// Raw file bytes (owned or memory-mapped).
    data: DataSource,
    /// Parsed header.
    header: FileHeader,
    /// Path for diagnostics.
    #[allow(dead_code)]
    path: PathBuf,
}

const ROOT_MAGIC: &[u8; 4] = b"root";

impl RootFile {
    /// Open and parse a ROOT file from disk using memory mapping.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = fs::File::open(&path)?;
        // SAFETY: We only read the file, and rely on the OS to handle
        // concurrent modifications (which is UB for mmap but acceptable
        // for our read-only scientific-data use case).
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let data = DataSource::Mmap(mmap);
        Self::from_datasource(data, path)
    }

    /// Parse a ROOT file from a byte vector (for testing).
    pub fn from_bytes(data: Vec<u8>, path: PathBuf) -> Result<Self> {
        Self::from_datasource(DataSource::Owned(data), path)
    }

    /// Internal constructor from any DataSource.
    fn from_datasource(data: DataSource, path: PathBuf) -> Result<Self> {
        if data.len() < 64 {
            return Err(RootError::BadMagic);
        }
        if &data[0..4] != ROOT_MAGIC {
            return Err(RootError::BadMagic);
        }

        let header = Self::parse_header(&data)?;
        Ok(Self { data, header, path })
    }

    /// Parse the file-level header (first ~63 bytes) and the embedded TDirectory.
    ///
    /// ROOT file header layout (small file, version < 1000000):
    /// ```text
    /// offset  size  field
    ///    0      4   magic "root"
    ///    4      4   fVersion
    ///    8      4   fBEGIN
    ///   12      4   fEND
    ///   16      4   fSeekFree
    ///   20      4   fNbytesFree
    ///   24      4   nfree
    ///   28      4   fNbytesName
    ///   32      1   fUnits
    ///   33      4   fCompress
    ///   37      4   fSeekInfo
    ///   41      4   fNbytesInfo
    ///   45     18   fUUID
    ///   63         (end of file header)
    /// ```
    ///
    /// The TDirectory streamer is located at `fBEGIN + fNbytesName`.
    fn parse_header(data: &[u8]) -> Result<FileHeader> {
        let mut r = RBuffer::new(data);
        r.skip(4)?; // magic

        let version = r.read_u32()?;
        let is_large = version >= 1_000_000;

        let begin = r.read_u32()? as u64;

        if is_large {
            let _end = r.read_u64()?;
            let _seek_free = r.read_u64()?;
        } else {
            let _end = r.read_u32()?;
            let _seek_free = r.read_u32()?;
        }
        let _nbytes_free = r.read_u32()?;
        let _nfree = r.read_u32()?;
        let nbytes_name = r.read_u32()?;
        let _units = r.read_u8()?;
        let _compress = r.read_u32()?;
        if is_large {
            let _seek_info = r.read_u64()?;
        } else {
            let _seek_info = r.read_u32()?;
        }
        let _nbytes_info = r.read_u32()?;
        // 18-byte UUID follows — skip it

        // Parse the top-level TDirectory located at fBEGIN + fNbytesName.
        let (seek_keys, nbytes_keys) =
            Self::parse_top_directory(data, begin as usize, nbytes_name as usize, is_large)?;

        Ok(FileHeader { begin, is_large, nbytes_name, seek_keys, nbytes_keys })
    }

    /// Parse the TDirectory streamer at `begin + nbytes_name` to extract seek_keys.
    fn parse_top_directory(
        data: &[u8],
        begin: usize,
        nbytes_name: usize,
        _is_large: bool,
    ) -> Result<(u64, u32)> {
        let dir_offset = begin + nbytes_name;
        if dir_offset >= data.len() {
            return Err(RootError::Deserialization("TDirectory offset past end of file".into()));
        }

        let mut r = RBuffer::new(data);
        r.set_pos(dir_offset);

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

    /// Get a histogram by its full path, preserving under/overflow bins.
    pub fn get_histogram_with_flows(&self, path: &str) -> Result<HistogramWithFlows> {
        let dir = self.read_top_directory()?;
        self.resolve_histogram_with_flows(&dir, path)
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
            let key = current_dir
                .find_key(part)
                .ok_or_else(|| RootError::KeyNotFound(format!("{} (in path {})", part, path)))?;

            if key.class_name != "TDirectoryFile" && key.class_name != "TDirectory" {
                return Err(RootError::Deserialization(format!(
                    "'{}' is not a directory (class: {})",
                    part, key.class_name
                )));
            }

            current_dir = self.read_subdirectory(key)?;
        }

        self.read_histogram_from_dir(&current_dir, parts[parts.len() - 1])
    }

    fn resolve_histogram_with_flows(
        &self,
        dir: &Directory,
        path: &str,
    ) -> Result<HistogramWithFlows> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        if parts.is_empty() {
            return Err(RootError::KeyNotFound(path.to_string()));
        }

        if parts.len() == 1 {
            return self.read_histogram_from_dir_with_flows(dir, parts[0]);
        }

        // Navigate subdirectories
        let mut current_dir = dir.clone();
        for &part in &parts[..parts.len() - 1] {
            let key = current_dir
                .find_key(part)
                .ok_or_else(|| RootError::KeyNotFound(format!("{} (in path {})", part, path)))?;

            if key.class_name != "TDirectoryFile" && key.class_name != "TDirectory" {
                return Err(RootError::Deserialization(format!(
                    "'{}' is not a directory (class: {})",
                    part, key.class_name
                )));
            }

            current_dir = self.read_subdirectory(key)?;
        }

        self.read_histogram_from_dir_with_flows(&current_dir, parts[parts.len() - 1])
    }

    fn read_subdirectory(&self, key: &Key) -> Result<Directory> {
        // For TDirectoryFile, the seek_keys is stored in the directory payload.
        // We need to read the key payload, then parse the TDirectory streamer.
        let payload = self.read_key_payload(key)?;
        Directory::read_from_payload(&payload, key, self.header.is_large, &self.data)
    }

    fn read_histogram_from_dir(&self, dir: &Directory, name: &str) -> Result<Histogram> {
        let key = dir.find_key(name).ok_or_else(|| RootError::KeyNotFound(name.to_string()))?;

        let payload = self.read_key_payload(key)?;
        objects::read_histogram(&payload, &key.class_name)
    }

    fn read_histogram_from_dir_with_flows(
        &self,
        dir: &Directory,
        name: &str,
    ) -> Result<HistogramWithFlows> {
        let key = dir.find_key(name).ok_or_else(|| RootError::KeyNotFound(name.to_string()))?;

        let payload = self.read_key_payload(key)?;
        objects::read_histogram_with_flows(&payload, &key.class_name)
    }

    /// Read and decompress the payload of a TKey.
    pub(crate) fn read_key_payload(&self, key: &Key) -> Result<Vec<u8>> {
        read_key_payload_from(&self.data, key)
    }

    /// Access the raw file data.
    #[allow(dead_code)]
    pub(crate) fn file_data(&self) -> &[u8] {
        &self.data
    }

    /// Whether file uses 64-bit seek pointers.
    #[allow(dead_code)]
    pub(crate) fn is_large(&self) -> bool {
        self.header.is_large
    }

    // ── TTree API ──────────────────────────────────────────────

    /// Read a TTree by name from the top-level directory.
    pub fn get_tree(&self, name: &str) -> Result<Tree> {
        let dir = self.read_top_directory()?;
        let key = dir.find_key(name).ok_or_else(|| RootError::TreeNotFound(name.to_string()))?;

        if key.class_name != "TTree" {
            return Err(RootError::TreeNotFound(format!(
                "'{}' is {} not TTree",
                name, key.class_name
            )));
        }

        let payload = self.read_key_payload(key)?;
        objects::read_tree(&payload)
    }

    /// Create a [`BranchReader`] for the named branch.
    pub fn branch_reader<'a>(&'a self, tree: &'a Tree, branch: &str) -> Result<BranchReader<'a>> {
        let info = tree
            .find_branch(branch)
            .ok_or_else(|| RootError::BranchNotFound(branch.to_string()))?;
        Ok(BranchReader::new(&self.data, info, self.header.is_large))
    }

    /// Read a branch as a jagged (variable-length) column.
    pub fn branch_data_jagged(&self, tree: &Tree, branch: &str) -> Result<JaggedCol> {
        let reader = self.branch_reader(tree, branch)?;
        reader.as_jagged_f64()
    }

    /// Convenience: read all entries from a branch as `f64`.
    pub fn branch_data(&self, tree: &Tree, branch: &str) -> Result<Vec<f64>> {
        match self.branch_reader(tree, branch) {
            Ok(r) => r.as_f64(),
            Err(RootError::BranchNotFound(_)) => {
                if let Some((base, idx)) = parse_indexed_branch_name(branch) {
                    let r = self.branch_reader(tree, base)?;
                    // ROOT/TTreeFormula convention for out-of-range indexing is effectively 0.0
                    // for numeric types (common in analysis selections/weights).
                    return r.as_f64_indexed(idx, 0.0);
                }
                Err(RootError::BranchNotFound(branch.to_string()))
            }
            Err(e) => Err(e),
        }
    }
}

fn parse_indexed_branch_name(s: &str) -> Option<(&str, usize)> {
    // Accept `name[0]` where name may contain dots/underscores/etc.
    let rb = s.strip_suffix(']')?;
    let lb_pos = rb.rfind('[')?;
    let (base, idx_str) = rb.split_at(lb_pos);
    let idx_str = &idx_str[1..]; // drop '['
    if base.is_empty() || idx_str.is_empty() {
        return None;
    }
    let idx: usize = idx_str.parse().ok()?;
    Some((base, idx))
}

/// Shared helper: read and decompress a TKey payload from raw file bytes.
pub(crate) fn read_key_payload_from(data: &[u8], key: &Key) -> Result<Vec<u8>> {
    let seek = key.seek_key as usize;
    if seek + key.n_bytes as usize > data.len() {
        return Err(RootError::BufferUnderflow {
            offset: seek,
            need: key.n_bytes as usize,
            have: data.len().saturating_sub(seek),
        });
    }

    let key_slice = &data[seek..seek + key.n_bytes as usize];

    // Object data starts after the key header.
    let obj_start = key.key_len as usize;
    let compressed_data = &key_slice[obj_start..];

    let compressed_len = key.n_bytes as usize - key.key_len as usize;
    if key.obj_len as usize != compressed_len {
        // Data is compressed
        decompress(compressed_data, key.obj_len as usize)
    } else {
        // Data is uncompressed
        Ok(compressed_data.to_vec())
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
