//! TFile header parsing and top-level ROOT file interface.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::branch_reader::{BranchReader, JaggedCol};
use crate::cache::{BasketCache, CacheConfig};
use crate::datasource::DataSource;
use crate::decompress::decompress;
use crate::directory::Directory;
use crate::error::{Result, RootError};
use crate::histogram::{Histogram, HistogramWithFlows};
use crate::key::{Key, KeyInfo};
use crate::lazy_branch_reader::LazyBranchReader;
use crate::objects;
use crate::rbuffer::RBuffer;
use crate::rntuple::{
    RNTUPLE_ENVELOPE_TYPE_FOOTER, RNTUPLE_ENVELOPE_TYPE_HEADER, RNTUPLE_ENVELOPE_TYPE_PAGELIST,
    RNTupleAnchor, RNTupleClusterGroupSummary, RNTupleFieldKind, RNTupleFooterSummary,
    RNTupleMetadataSummary, RNTuplePageListSummary, RNTuplePageSummary, RNTupleScalarType,
    RNTupleSchemaSummary, parse_rntuple_anchor_payload, parse_rntuple_envelope,
    parse_rntuple_footer_summary, parse_rntuple_header_summary, parse_rntuple_pagelist_summary,
    parse_rntuple_schema_summary, parse_rntuple_schema_summary_with_footer,
};
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
    /// LRU cache for decompressed basket payloads (shared across all branch readers).
    basket_cache: BasketCache,
}

/// Public info about an RNTuple-like object discovered in a ROOT directory key list.
#[derive(Debug, Clone)]
pub struct RNTupleInfo {
    /// Object name.
    pub name: String,
    /// Raw ROOT class name as stored in TKey metadata.
    pub class_name: String,
    /// Cycle number.
    pub cycle: u16,
}

/// Decompressed RNTuple metadata envelopes resolved from the anchor.
#[derive(Debug, Clone)]
pub struct RNTupleEnvelopeBytes {
    /// Parsed anchor values used for locating metadata blobs.
    pub anchor: RNTupleAnchor,
    /// Decompressed header envelope bytes.
    pub header: Vec<u8>,
    /// Decompressed footer envelope bytes.
    pub footer: Vec<u8>,
}

/// Decompressed page-list envelope bytes for one cluster group.
#[derive(Debug, Clone)]
pub struct RNTuplePageListEnvelopeBytes {
    /// Cluster-group summary the page-list belongs to.
    pub cluster_group: RNTupleClusterGroupSummary,
    /// Decompressed page-list envelope bytes.
    pub page_list: Vec<u8>,
}

/// Raw on-storage page bytes loaded through a page-list descriptor.
#[derive(Debug, Clone)]
pub struct RNTuplePageBlobBytes {
    /// Parsed page descriptor selected from page-list summary.
    pub page: RNTuplePageSummary,
    /// Raw on-storage bytes for this page (compressed or plain, as stored).
    pub page_blob: Vec<u8>,
}

/// Decoded primitive RNTuple column for one cluster group.
#[derive(Debug, Clone, PartialEq)]
pub struct RNTuplePrimitiveColumnF64 {
    /// Field name from RNTuple schema.
    pub field_name: String,
    /// Scalar type declared in RNTuple schema.
    pub scalar_type: RNTupleScalarType,
    /// Source page index in page-list summary.
    pub page_index: usize,
    /// Decoded numeric values represented as `f64`.
    pub values: Vec<f64>,
}

/// Decoded fixed-array RNTuple column for one cluster group.
#[derive(Debug, Clone, PartialEq)]
pub struct RNTupleFixedArrayColumnF64 {
    /// Field name from RNTuple schema.
    pub field_name: String,
    /// Element scalar type declared in schema.
    pub element_scalar_type: RNTupleScalarType,
    /// Fixed number of elements per entry.
    pub fixed_len: usize,
    /// Source page index in page-list summary.
    pub page_index: usize,
    /// Decoded values grouped by entry.
    pub values: Vec<Vec<f64>>,
}

/// Decoded variable-array RNTuple column for one cluster group.
#[derive(Debug, Clone, PartialEq)]
pub struct RNTupleVariableArrayColumnF64 {
    /// Field name from RNTuple schema.
    pub field_name: String,
    /// Element scalar type declared in schema.
    pub element_scalar_type: RNTupleScalarType,
    /// Source offset page index in page-list summary.
    pub offset_page_index: usize,
    /// Source data page index in page-list summary.
    pub data_page_index: usize,
    /// Decoded jagged values grouped by entry.
    pub values: Vec<Vec<f64>>,
}

/// Decoded `std::pair<primitive, primitive>` nested RNTuple column.
#[derive(Debug, Clone, PartialEq)]
pub struct RNTuplePairColumnF64 {
    /// Field name from RNTuple schema.
    pub field_name: String,
    /// Left element scalar type.
    pub left_scalar_type: RNTupleScalarType,
    /// Right element scalar type.
    pub right_scalar_type: RNTupleScalarType,
    /// Source left page index.
    pub left_page_index: usize,
    /// Source right page index.
    pub right_page_index: usize,
    /// Decoded pair values grouped by entry.
    pub values: Vec<(f64, f64)>,
}

/// Decoded `std::pair<primitive, std::vector<primitive>>` nested RNTuple column.
#[derive(Debug, Clone, PartialEq)]
pub struct RNTuplePairScalarVariableColumnF64 {
    /// Field name from RNTuple schema.
    pub field_name: String,
    /// Left element scalar type.
    pub left_scalar_type: RNTupleScalarType,
    /// Right variable element scalar type.
    pub right_element_scalar_type: RNTupleScalarType,
    /// Source left page index.
    pub left_page_index: usize,
    /// Source right-offset page index.
    pub right_offset_page_index: usize,
    /// Source right-data page index.
    pub right_data_page_index: usize,
    /// Decoded pair values grouped by entry.
    pub values: Vec<(f64, Vec<f64>)>,
}

/// Decoded `std::pair<std::vector<primitive>, primitive>` nested RNTuple column.
#[derive(Debug, Clone, PartialEq)]
pub struct RNTuplePairVariableScalarColumnF64 {
    /// Field name from RNTuple schema.
    pub field_name: String,
    /// Left variable element scalar type.
    pub left_element_scalar_type: RNTupleScalarType,
    /// Right scalar type.
    pub right_scalar_type: RNTupleScalarType,
    /// Source left-offset page index.
    pub left_offset_page_index: usize,
    /// Source left-data page index.
    pub left_data_page_index: usize,
    /// Source right page index.
    pub right_page_index: usize,
    /// Decoded pair values grouped by entry.
    pub values: Vec<(Vec<f64>, f64)>,
}

/// Decoded `std::pair<std::vector<primitive>, std::vector<primitive>>` nested RNTuple column.
#[derive(Debug, Clone, PartialEq)]
pub struct RNTuplePairVariableVariableColumnF64 {
    /// Field name from RNTuple schema.
    pub field_name: String,
    /// Left variable element scalar type.
    pub left_element_scalar_type: RNTupleScalarType,
    /// Right variable element scalar type.
    pub right_element_scalar_type: RNTupleScalarType,
    /// Source left-offset page index.
    pub left_offset_page_index: usize,
    /// Source left-data page index.
    pub left_data_page_index: usize,
    /// Source right-offset page index.
    pub right_offset_page_index: usize,
    /// Source right-data page index.
    pub right_data_page_index: usize,
    /// Decoded pair values grouped by entry.
    pub values: Vec<(Vec<f64>, Vec<f64>)>,
}

/// Decoded supported RNTuple columns for one cluster group.
#[derive(Debug, Clone, PartialEq)]
pub struct RNTupleDecodedColumnsF64 {
    /// Primitive scalar fields.
    pub primitive: Vec<RNTuplePrimitiveColumnF64>,
    /// Fixed-size array fields.
    pub fixed_arrays: Vec<RNTupleFixedArrayColumnF64>,
    /// Variable-size array fields.
    pub variable_arrays: Vec<RNTupleVariableArrayColumnF64>,
    /// Nested `std::pair<primitive,primitive>` fields.
    pub pairs: Vec<RNTuplePairColumnF64>,
    /// Nested `std::pair<primitive,std::vector<primitive>>` fields.
    pub pair_scalar_variable: Vec<RNTuplePairScalarVariableColumnF64>,
    /// Nested `std::pair<std::vector<primitive>,primitive>` fields.
    pub pair_variable_scalar: Vec<RNTuplePairVariableScalarColumnF64>,
    /// Nested `std::pair<std::vector<primitive>,std::vector<primitive>>` fields.
    pub pair_variable_variable: Vec<RNTuplePairVariableVariableColumnF64>,
}

/// Decoded supported columns for one specific cluster group.
#[derive(Debug, Clone, PartialEq)]
pub struct RNTupleClusterDecodedColumnsF64 {
    /// Cluster-group index in footer order.
    pub cluster_group_index: usize,
    /// First global entry index covered by this cluster group.
    pub min_entry: u64,
    /// Number of entries covered by this cluster group.
    pub entry_span: u64,
    /// Decoded supported columns for this cluster group.
    pub columns: RNTupleDecodedColumnsF64,
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
        Ok(Self { data, header, path, basket_cache: BasketCache::default() })
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

    /// List top-level keys that look like RNTuple objects.
    ///
    /// ROOT class names vary across producers, so detection is best-effort and
    /// based on class-name matching (`*RNTuple*`, case-insensitive).
    pub fn list_rntuples(&self) -> Result<Vec<RNTupleInfo>> {
        let keys = self.list_keys()?;
        Ok(keys.into_iter().filter_map(|k| to_rntuple_info(&k)).collect())
    }

    /// Fast check for whether this file exposes any top-level RNTuple key.
    pub fn has_rntuples(&self) -> Result<bool> {
        Ok(!self.list_rntuples()?.is_empty())
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

    fn read_file_range(&self, seek: u64, n_bytes: u64) -> Result<&[u8]> {
        let start: usize = seek
            .try_into()
            .map_err(|_| RootError::Deserialization(format!("seek offset too large: {}", seek)))?;
        let len: usize = n_bytes.try_into().map_err(|_| {
            RootError::Deserialization(format!("range length too large: {}", n_bytes))
        })?;
        let end = start.checked_add(len).ok_or_else(|| {
            RootError::Deserialization(format!(
                "range overflow for seek={} n_bytes={}",
                seek, n_bytes
            ))
        })?;
        if end > self.data.len() {
            return Err(RootError::BufferUnderflow {
                offset: start,
                need: len,
                have: self.data.len().saturating_sub(start),
            });
        }
        Ok(&self.data[start..end])
    }

    /// Access the raw file data.
    #[allow(dead_code)]
    pub fn file_data(&self) -> &[u8] {
        &self.data
    }

    /// Whether file uses 64-bit seek pointers.
    #[allow(dead_code)]
    pub fn is_large(&self) -> bool {
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

    // ── RNTuple API (foundation) ─────────────────────────────

    /// Read raw payload bytes for an RNTuple-like key by name.
    ///
    /// This is a foundation API used by upcoming metadata/schema parser stages.
    /// It validates the key class as RNTuple-like and returns decompressed object bytes.
    pub fn read_rntuple_payload(&self, name: &str) -> Result<Vec<u8>> {
        let dir = self.read_top_directory()?;
        let key = dir.find_key(name).ok_or_else(|| RootError::KeyNotFound(name.to_string()))?;
        if !is_rntuple_class_name(&key.class_name) {
            return Err(RootError::UnsupportedClass(format!(
                "'{}' is {} not an RNTuple-like class",
                name, key.class_name
            )));
        }
        self.read_key_payload(key)
    }

    /// Parse anchor metadata from an RNTuple-like key by name.
    pub fn read_rntuple_anchor(&self, name: &str) -> Result<RNTupleAnchor> {
        let payload = self.read_rntuple_payload(name)?;
        parse_rntuple_anchor_payload(&payload)
    }

    /// Read and decompress RNTuple header/footer envelopes referenced by the anchor.
    pub fn read_rntuple_envelopes(&self, name: &str) -> Result<RNTupleEnvelopeBytes> {
        let anchor = self.read_rntuple_anchor(name)?;

        let header_comp = self.read_file_range(anchor.seek_header, anchor.nbytes_header)?;
        let footer_comp = self.read_file_range(anchor.seek_footer, anchor.nbytes_footer)?;

        let header = decompress_ntuple_blob(header_comp, anchor.len_header as usize)?;
        let footer = decompress_ntuple_blob(footer_comp, anchor.len_footer as usize)?;

        Ok(RNTupleEnvelopeBytes { anchor, header, footer })
    }

    /// Parse metadata summary from anchor and header/footer envelopes.
    pub fn read_rntuple_metadata_summary(&self, name: &str) -> Result<RNTupleMetadataSummary> {
        let envelopes = self.read_rntuple_envelopes(name)?;
        let header_envelope =
            parse_rntuple_envelope(&envelopes.header, Some(RNTUPLE_ENVELOPE_TYPE_HEADER))?;
        let footer_envelope =
            parse_rntuple_envelope(&envelopes.footer, Some(RNTUPLE_ENVELOPE_TYPE_FOOTER))?;
        let header_summary = parse_rntuple_header_summary(&envelopes.header)?;
        Ok(RNTupleMetadataSummary {
            anchor: envelopes.anchor,
            header_envelope,
            footer_envelope,
            header_summary,
        })
    }

    /// Parse best-effort schema summary from RNTuple metadata.
    pub fn read_rntuple_schema_summary(&self, name: &str) -> Result<RNTupleSchemaSummary> {
        let envelopes = self.read_rntuple_envelopes(name)?;
        let header_summary = parse_rntuple_header_summary(&envelopes.header)?;
        parse_rntuple_schema_summary_with_footer(&header_summary, &envelopes.footer)
    }

    /// Parse footer summary with cluster-group/page-list locators.
    pub fn read_rntuple_footer_summary(&self, name: &str) -> Result<RNTupleFooterSummary> {
        let envelopes = self.read_rntuple_envelopes(name)?;
        parse_rntuple_footer_summary(&envelopes.footer)
    }

    /// Load and decompress the page-list envelope for a given cluster group.
    pub fn read_rntuple_pagelist_envelope(
        &self,
        name: &str,
        cluster_group_index: usize,
    ) -> Result<RNTuplePageListEnvelopeBytes> {
        let footer = self.read_rntuple_footer_summary(name)?;
        let cluster_group = footer
            .cluster_groups
            .get(cluster_group_index)
            .ok_or_else(|| {
                RootError::Deserialization(format!(
                    "RNTuple cluster_group_index {} out of bounds (n={})",
                    cluster_group_index,
                    footer.cluster_groups.len()
                ))
            })?
            .clone();

        let comp = self.read_file_range(
            cluster_group.page_list_locator.position,
            cluster_group.page_list_locator.nbytes_on_storage,
        )?;
        let page_list =
            decompress_ntuple_blob(comp, cluster_group.page_list_envelope_len as usize)?;
        parse_rntuple_envelope(&page_list, Some(RNTUPLE_ENVELOPE_TYPE_PAGELIST))?;

        Ok(RNTuplePageListEnvelopeBytes { cluster_group, page_list })
    }

    /// Parse page-list summary (page descriptors) for a given cluster group.
    pub fn read_rntuple_pagelist_summary(
        &self,
        name: &str,
        cluster_group_index: usize,
    ) -> Result<RNTuplePageListSummary> {
        let page_list = self.read_rntuple_pagelist_envelope(name, cluster_group_index)?;
        parse_rntuple_pagelist_summary(&page_list.page_list)
    }

    /// Load a raw on-storage page blob by page index from one cluster group's page list.
    ///
    /// This method intentionally returns bytes as stored and does not attempt page-level
    /// value decoding yet.
    pub fn read_rntuple_page_blob(
        &self,
        name: &str,
        cluster_group_index: usize,
        page_index: usize,
    ) -> Result<RNTuplePageBlobBytes> {
        let summary = self.read_rntuple_pagelist_summary(name, cluster_group_index)?;
        let page = summary.pages.get(page_index).ok_or_else(|| {
            RootError::Deserialization(format!(
                "RNTuple page_index {} out of bounds (n={})",
                page_index,
                summary.pages.len()
            ))
        })?;
        let page_blob =
            self.read_file_range(page.position, page.nbytes_on_storage as u64)?.to_vec();
        Ok(RNTuplePageBlobBytes { page: page.clone(), page_blob })
    }

    /// Decode all currently supported schema fields for one cluster group.
    ///
    /// Supported decode set:
    /// - primitive scalars
    /// - fixed arrays
    /// - variable arrays (offset + data pages)
    /// - nested `std::pair<primitive,primitive>`
    /// - nested `std::pair<primitive,std::vector<primitive>>`
    /// - nested `std::pair<std::vector<primitive>,primitive>`
    /// - nested `std::pair<std::vector<primitive>,std::vector<primitive>>`
    ///
    /// Unsupported nested kinds return a deterministic `UnsupportedClass` error.
    pub fn read_rntuple_decoded_columns_f64(
        &self,
        name: &str,
        cluster_group_index: usize,
    ) -> Result<RNTupleDecodedColumnsF64> {
        let (schema, required_field_names) = self.read_rntuple_decode_schema_context(name)?;
        let footer = self.read_rntuple_footer_summary(name)?;
        let cluster_group = footer.cluster_groups.get(cluster_group_index).ok_or_else(|| {
            RootError::Deserialization(format!(
                "RNTuple cluster_group_index {} out of bounds (n={})",
                cluster_group_index,
                footer.cluster_groups.len()
            ))
        })?;
        self.decode_rntuple_fields_f64_internal(
            name,
            cluster_group_index,
            &schema,
            &required_field_names,
            cluster_group.entry_span,
        )
    }

    /// Decode all currently supported schema fields across all cluster groups.
    pub fn read_rntuple_decoded_columns_all_clusters_f64(
        &self,
        name: &str,
    ) -> Result<Vec<RNTupleClusterDecodedColumnsF64>> {
        let (schema, required_field_names) = self.read_rntuple_decode_schema_context(name)?;
        let footer = self.read_rntuple_footer_summary(name)?;
        let mut out = Vec::with_capacity(footer.cluster_groups.len());
        for (idx, cg) in footer.cluster_groups.iter().enumerate() {
            let columns = self.decode_rntuple_fields_f64_internal(
                name,
                idx,
                &schema,
                &required_field_names,
                cg.entry_span,
            )?;
            out.push(RNTupleClusterDecodedColumnsF64 {
                cluster_group_index: idx,
                min_entry: cg.min_entry,
                entry_span: cg.entry_span,
                columns,
            });
        }
        Ok(out)
    }

    /// Decode primitive schema fields for one cluster group into `f64` vectors.
    pub fn read_rntuple_primitive_columns_f64(
        &self,
        name: &str,
        cluster_group_index: usize,
    ) -> Result<Vec<RNTuplePrimitiveColumnF64>> {
        Ok(self.read_rntuple_decoded_columns_f64(name, cluster_group_index)?.primitive)
    }

    /// Decode fixed-array schema fields for one cluster group.
    pub fn read_rntuple_fixed_array_columns_f64(
        &self,
        name: &str,
        cluster_group_index: usize,
    ) -> Result<Vec<RNTupleFixedArrayColumnF64>> {
        Ok(self.read_rntuple_decoded_columns_f64(name, cluster_group_index)?.fixed_arrays)
    }

    /// Decode variable-array schema fields for one cluster group.
    pub fn read_rntuple_variable_array_columns_f64(
        &self,
        name: &str,
        cluster_group_index: usize,
    ) -> Result<Vec<RNTupleVariableArrayColumnF64>> {
        Ok(self.read_rntuple_decoded_columns_f64(name, cluster_group_index)?.variable_arrays)
    }

    /// Decode nested `std::pair<primitive, primitive>` schema fields for one cluster group.
    pub fn read_rntuple_pair_columns_f64(
        &self,
        name: &str,
        cluster_group_index: usize,
    ) -> Result<Vec<RNTuplePairColumnF64>> {
        Ok(self.read_rntuple_decoded_columns_f64(name, cluster_group_index)?.pairs)
    }

    /// Decode nested `std::pair<primitive, std::vector<primitive>>` fields for one cluster group.
    pub fn read_rntuple_pair_scalar_variable_columns_f64(
        &self,
        name: &str,
        cluster_group_index: usize,
    ) -> Result<Vec<RNTuplePairScalarVariableColumnF64>> {
        Ok(self.read_rntuple_decoded_columns_f64(name, cluster_group_index)?.pair_scalar_variable)
    }

    /// Decode nested `std::pair<std::vector<primitive>, primitive>` fields for one cluster group.
    pub fn read_rntuple_pair_variable_scalar_columns_f64(
        &self,
        name: &str,
        cluster_group_index: usize,
    ) -> Result<Vec<RNTuplePairVariableScalarColumnF64>> {
        Ok(self.read_rntuple_decoded_columns_f64(name, cluster_group_index)?.pair_variable_scalar)
    }

    /// Decode nested `std::pair<std::vector<primitive>, std::vector<primitive>>` fields.
    pub fn read_rntuple_pair_variable_variable_columns_f64(
        &self,
        name: &str,
        cluster_group_index: usize,
    ) -> Result<Vec<RNTuplePairVariableVariableColumnF64>> {
        Ok(self.read_rntuple_decoded_columns_f64(name, cluster_group_index)?.pair_variable_variable)
    }

    fn read_rntuple_decode_schema_context(
        &self,
        name: &str,
    ) -> Result<(RNTupleSchemaSummary, HashSet<String>)> {
        let envelopes = self.read_rntuple_envelopes(name)?;
        let header_summary = parse_rntuple_header_summary(&envelopes.header)?;
        let header_schema = parse_rntuple_schema_summary(&header_summary);
        let required_field_names =
            header_schema.fields.into_iter().map(|field| field.name).collect::<HashSet<_>>();
        let merged_schema =
            parse_rntuple_schema_summary_with_footer(&header_summary, &envelopes.footer)?;
        Ok((merged_schema, required_field_names))
    }

    fn decode_rntuple_fields_f64_internal(
        &self,
        name: &str,
        cluster_group_index: usize,
        schema: &RNTupleSchemaSummary,
        required_field_names: &HashSet<String>,
        entry_span: u64,
    ) -> Result<RNTupleDecodedColumnsF64> {
        let entry_count = usize::try_from(entry_span).map_err(|_| {
            RootError::Deserialization(format!(
                "RNTuple entry_span {} does not fit usize",
                entry_span
            ))
        })?;

        let pagelist = self.read_rntuple_pagelist_summary(name, cluster_group_index)?;
        let mut used_pages = vec![false; pagelist.pages.len()];
        let mut cursor = 0usize;
        let mut primitive = Vec::new();
        let mut fixed_arrays = Vec::new();
        let mut variable_arrays = Vec::new();
        let mut pairs = Vec::new();
        let mut pair_scalar_variable = Vec::new();
        let mut pair_variable_scalar = Vec::new();
        let mut pair_variable_variable = Vec::new();

        for field in &schema.fields {
            match field.kind {
                RNTupleFieldKind::Primitive => {
                    let scalar_type = field.scalar_type.ok_or_else(|| {
                        RootError::Deserialization(format!(
                            "RNTuple primitive field '{}' has no scalar type",
                            field.name
                        ))
                    })?;
                    let byte_width = scalar_type_byte_width(scalar_type);
                    let total_elems = entry_count;
                    let Some(page_idx) =
                        find_matching_page_index(&pagelist.pages, &used_pages, cursor, |page| {
                            page.record_tag_raw == -40 && abs_element_count(page) == total_elems
                        })
                    else {
                        if required_field_names.contains(&field.name) {
                            return Err(RootError::Deserialization(format!(
                                "RNTuple primitive field '{}' page not found for entry_count={} width={}",
                                field.name, entry_count, byte_width
                            )));
                        }
                        continue;
                    };
                    let page = &pagelist.pages[page_idx];
                    let values = decode_rntuple_numeric_page_f64(
                        self,
                        page,
                        scalar_type,
                        total_elems,
                        &field.name,
                    )?;
                    used_pages[page_idx] = true;
                    cursor = page_idx.saturating_add(1);
                    primitive.push(RNTuplePrimitiveColumnF64 {
                        field_name: field.name.clone(),
                        scalar_type,
                        page_index: page_idx,
                        values,
                    });
                }
                RNTupleFieldKind::FixedArray => {
                    let elem_ty = field.element_scalar_type.ok_or_else(|| {
                        RootError::Deserialization(format!(
                            "RNTuple fixed-array field '{}' has no element scalar type",
                            field.name
                        ))
                    })?;
                    let fixed_len = field.fixed_len.ok_or_else(|| {
                        RootError::Deserialization(format!(
                            "RNTuple fixed-array field '{}' has no fixed_len",
                            field.name
                        ))
                    })?;
                    let total_elems = entry_count.checked_mul(fixed_len).ok_or_else(|| {
                        RootError::Deserialization(format!(
                            "RNTuple fixed-array total element overflow for field '{}'",
                            field.name
                        ))
                    })?;
                    let byte_width = scalar_type_byte_width(elem_ty);
                    let Some(page_idx) =
                        find_matching_page_index(&pagelist.pages, &used_pages, cursor, |page| {
                            page.record_tag_raw == -40 && abs_element_count(page) == total_elems
                        })
                    else {
                        if required_field_names.contains(&field.name) {
                            return Err(RootError::Deserialization(format!(
                                "RNTuple fixed-array field '{}' page not found (elems={} width={})",
                                field.name, total_elems, byte_width
                            )));
                        }
                        continue;
                    };
                    let page = &pagelist.pages[page_idx];
                    let flat_values = decode_rntuple_numeric_page_f64(
                        self,
                        page,
                        elem_ty,
                        total_elems,
                        &field.name,
                    )?;
                    let values = flat_values.chunks_exact(fixed_len).map(|c| c.to_vec()).collect();
                    used_pages[page_idx] = true;
                    cursor = page_idx.saturating_add(1);
                    fixed_arrays.push(RNTupleFixedArrayColumnF64 {
                        field_name: field.name.clone(),
                        element_scalar_type: elem_ty,
                        fixed_len,
                        page_index: page_idx,
                        values,
                    });
                }
                RNTupleFieldKind::VariableArray => {
                    let elem_ty = field.element_scalar_type.ok_or_else(|| {
                        RootError::Deserialization(format!(
                            "RNTuple variable-array field '{}' has no element scalar type",
                            field.name
                        ))
                    })?;
                    let bundle = decode_variable_array_bundle_f64(
                        self,
                        &pagelist,
                        &used_pages,
                        cursor,
                        entry_count,
                        &field.name,
                        elem_ty,
                    )?;
                    let Some(bundle) = bundle else {
                        if required_field_names.contains(&field.name) {
                            return Err(RootError::Deserialization(format!(
                                "RNTuple variable-array field '{}' offset page not found (entry_count={})",
                                field.name, entry_count
                            )));
                        }
                        continue;
                    };

                    used_pages[bundle.offset_page_index] = true;
                    used_pages[bundle.data_page_index] = true;
                    cursor = bundle.data_page_index.saturating_add(1);
                    variable_arrays.push(RNTupleVariableArrayColumnF64 {
                        field_name: field.name.clone(),
                        element_scalar_type: elem_ty,
                        offset_page_index: bundle.offset_page_index,
                        data_page_index: bundle.data_page_index,
                        values: bundle.values,
                    });
                }
                RNTupleFieldKind::Nested => {
                    let Some(pair_layout) = parse_pair_layout(&field.type_name) else {
                        return Err(RootError::UnsupportedClass(format!(
                            "RNTuple nested field '{}' has unsupported type '{}'",
                            field.name, field.type_name
                        )));
                    };
                    match pair_layout {
                        PairLayout::ScalarScalar {
                            left_scalar_type: left_ty,
                            right_scalar_type: right_ty,
                        } => {
                            let left_total = entry_count;
                            let right_total = entry_count;
                            let left_page_idx = find_matching_page_index(
                                &pagelist.pages,
                                &used_pages,
                                cursor,
                                |page| {
                                    page.record_tag_raw == -40
                                        && abs_element_count(page) == left_total
                                },
                            );
                            let Some(left_page_idx) = left_page_idx else {
                                if required_field_names.contains(&field.name) {
                                    return Err(RootError::Deserialization(format!(
                                        "RNTuple pair field '{}' left page not found",
                                        field.name
                                    )));
                                }
                                continue;
                            };
                            let right_page_idx = find_matching_page_index(
                                &pagelist.pages,
                                &used_pages,
                                left_page_idx.saturating_add(1),
                                |page| {
                                    page.record_tag_raw == -40
                                        && abs_element_count(page) == right_total
                                },
                            )
                            .ok_or_else(|| {
                                RootError::Deserialization(format!(
                                    "RNTuple pair field '{}' right page not found",
                                    field.name
                                ))
                            })?;
                            let left_page = &pagelist.pages[left_page_idx];
                            let right_page = &pagelist.pages[right_page_idx];
                            let left_values = decode_rntuple_numeric_page_f64(
                                self,
                                left_page,
                                left_ty,
                                left_total,
                                &field.name,
                            )?;
                            let right_values = decode_rntuple_numeric_page_f64(
                                self,
                                right_page,
                                right_ty,
                                right_total,
                                &field.name,
                            )?;
                            let values =
                                left_values.into_iter().zip(right_values.into_iter()).collect();
                            used_pages[left_page_idx] = true;
                            used_pages[right_page_idx] = true;
                            cursor = right_page_idx.saturating_add(1);
                            pairs.push(RNTuplePairColumnF64 {
                                field_name: field.name.clone(),
                                left_scalar_type: left_ty,
                                right_scalar_type: right_ty,
                                left_page_index: left_page_idx,
                                right_page_index: right_page_idx,
                                values,
                            });
                        }
                        PairLayout::ScalarVariableRight {
                            left_scalar_type: left_ty,
                            right_element_scalar_type: right_elem_ty,
                        } => {
                            let left_total = entry_count;
                            let left_page_idx = find_matching_page_index(
                                &pagelist.pages,
                                &used_pages,
                                cursor,
                                |page| {
                                    page.record_tag_raw == -40
                                        && abs_element_count(page) == left_total
                                },
                            );
                            let Some(left_page_idx) = left_page_idx else {
                                if required_field_names.contains(&field.name) {
                                    return Err(RootError::Deserialization(format!(
                                        "RNTuple pair field '{}' left page not found",
                                        field.name
                                    )));
                                }
                                continue;
                            };
                            let left_page = &pagelist.pages[left_page_idx];
                            let left_values = decode_rntuple_numeric_page_f64(
                                self,
                                left_page,
                                left_ty,
                                left_total,
                                &field.name,
                            )?;
                            let mut local_used = used_pages.clone();
                            local_used[left_page_idx] = true;
                            let right_bundle = decode_pair_variable_side_f64(
                                self,
                                &pagelist,
                                &local_used,
                                left_page_idx.saturating_add(1),
                                entry_count,
                                &field.name,
                                "right",
                                right_elem_ty,
                            )?
                            .ok_or_else(|| {
                                RootError::Deserialization(format!(
                                    "RNTuple pair field '{}' right offset page not found",
                                    field.name
                                ))
                            })?;
                            let mut values = Vec::with_capacity(entry_count);
                            for (idx, right_vals) in right_bundle.values.into_iter().enumerate() {
                                values.push((left_values[idx], right_vals));
                            }
                            used_pages[left_page_idx] = true;
                            used_pages[right_bundle.offset_page_index] = true;
                            used_pages[right_bundle.data_page_index] = true;
                            cursor = right_bundle.data_page_index.saturating_add(1);
                            pair_scalar_variable.push(RNTuplePairScalarVariableColumnF64 {
                                field_name: field.name.clone(),
                                left_scalar_type: left_ty,
                                right_element_scalar_type: right_elem_ty,
                                left_page_index: left_page_idx,
                                right_offset_page_index: right_bundle.offset_page_index,
                                right_data_page_index: right_bundle.data_page_index,
                                values,
                            });
                        }
                        PairLayout::VariableLeftScalar {
                            left_element_scalar_type: left_elem_ty,
                            right_scalar_type: right_ty,
                        } => {
                            let left_bundle = decode_pair_variable_side_f64(
                                self,
                                &pagelist,
                                &used_pages,
                                cursor,
                                entry_count,
                                &field.name,
                                "left",
                                left_elem_ty,
                            )?;
                            let Some(left_bundle) = left_bundle else {
                                if required_field_names.contains(&field.name) {
                                    return Err(RootError::Deserialization(format!(
                                        "RNTuple pair field '{}' left offset page not found",
                                        field.name
                                    )));
                                }
                                continue;
                            };
                            let right_total = entry_count;
                            let mut local_used = used_pages.clone();
                            local_used[left_bundle.offset_page_index] = true;
                            local_used[left_bundle.data_page_index] = true;
                            let right_page_idx = find_matching_page_index(
                                &pagelist.pages,
                                &local_used,
                                left_bundle.data_page_index.saturating_add(1),
                                |page| {
                                    page.record_tag_raw == -40
                                        && abs_element_count(page) == right_total
                                },
                            )
                            .ok_or_else(|| {
                                RootError::Deserialization(format!(
                                    "RNTuple pair field '{}' right page not found",
                                    field.name
                                ))
                            })?;
                            let right_page = &pagelist.pages[right_page_idx];
                            let right_values = decode_rntuple_numeric_page_f64(
                                self,
                                right_page,
                                right_ty,
                                right_total,
                                &field.name,
                            )?;
                            let mut values = Vec::with_capacity(entry_count);
                            for (idx, left_vals) in left_bundle.values.into_iter().enumerate() {
                                values.push((left_vals, right_values[idx]));
                            }
                            used_pages[left_bundle.offset_page_index] = true;
                            used_pages[left_bundle.data_page_index] = true;
                            used_pages[right_page_idx] = true;
                            cursor = right_page_idx.saturating_add(1);
                            pair_variable_scalar.push(RNTuplePairVariableScalarColumnF64 {
                                field_name: field.name.clone(),
                                left_element_scalar_type: left_elem_ty,
                                right_scalar_type: right_ty,
                                left_offset_page_index: left_bundle.offset_page_index,
                                left_data_page_index: left_bundle.data_page_index,
                                right_page_index: right_page_idx,
                                values,
                            });
                        }
                        PairLayout::VariableVariable {
                            left_element_scalar_type: left_elem_ty,
                            right_element_scalar_type: right_elem_ty,
                        } => {
                            let left_bundle = decode_pair_variable_side_f64(
                                self,
                                &pagelist,
                                &used_pages,
                                cursor,
                                entry_count,
                                &field.name,
                                "left",
                                left_elem_ty,
                            )?;
                            let Some(left_bundle) = left_bundle else {
                                if required_field_names.contains(&field.name) {
                                    return Err(RootError::Deserialization(format!(
                                        "RNTuple pair field '{}' left offset page not found",
                                        field.name
                                    )));
                                }
                                continue;
                            };
                            let mut local_used = used_pages.clone();
                            local_used[left_bundle.offset_page_index] = true;
                            local_used[left_bundle.data_page_index] = true;
                            let right_bundle = decode_pair_variable_side_f64(
                                self,
                                &pagelist,
                                &local_used,
                                left_bundle.data_page_index.saturating_add(1),
                                entry_count,
                                &field.name,
                                "right",
                                right_elem_ty,
                            )?
                            .ok_or_else(|| {
                                RootError::Deserialization(format!(
                                    "RNTuple pair field '{}' right offset page not found",
                                    field.name
                                ))
                            })?;
                            let mut values = Vec::with_capacity(entry_count);
                            for (left_vals, right_vals) in
                                left_bundle.values.into_iter().zip(right_bundle.values.into_iter())
                            {
                                values.push((left_vals, right_vals));
                            }
                            used_pages[left_bundle.offset_page_index] = true;
                            used_pages[left_bundle.data_page_index] = true;
                            used_pages[right_bundle.offset_page_index] = true;
                            used_pages[right_bundle.data_page_index] = true;
                            cursor = right_bundle.data_page_index.saturating_add(1);
                            pair_variable_variable.push(RNTuplePairVariableVariableColumnF64 {
                                field_name: field.name.clone(),
                                left_element_scalar_type: left_elem_ty,
                                right_element_scalar_type: right_elem_ty,
                                left_offset_page_index: left_bundle.offset_page_index,
                                left_data_page_index: left_bundle.data_page_index,
                                right_offset_page_index: right_bundle.offset_page_index,
                                right_data_page_index: right_bundle.data_page_index,
                                values,
                            });
                        }
                    }
                }
                RNTupleFieldKind::Unknown => {}
            }
        }
        Ok(RNTupleDecodedColumnsF64 {
            primitive,
            fixed_arrays,
            variable_arrays,
            pairs,
            pair_scalar_variable,
            pair_variable_scalar,
            pair_variable_variable,
        })
    }

    /// Create a [`BranchReader`] for the named branch (with basket caching).
    pub fn branch_reader<'a>(&'a self, tree: &'a Tree, branch: &str) -> Result<BranchReader<'a>> {
        let info = tree
            .find_branch(branch)
            .ok_or_else(|| RootError::BranchNotFound(branch.to_string()))?;
        Ok(BranchReader::with_cache(&self.data, info, self.header.is_large, &self.basket_cache))
    }

    /// Create a [`LazyBranchReader`] for the named branch.
    ///
    /// Unlike [`branch_reader`](Self::branch_reader) which eagerly loads all baskets,
    /// this returns a lazy reader that decompresses baskets on demand.
    pub fn lazy_branch_reader<'a>(
        &'a self,
        tree: &'a Tree,
        branch: &str,
    ) -> Result<LazyBranchReader<'a>> {
        let info = tree
            .find_branch(branch)
            .ok_or_else(|| RootError::BranchNotFound(branch.to_string()))?;
        Ok(LazyBranchReader::new(&self.data, info, self.header.is_large, &self.basket_cache))
    }

    /// Access the basket cache (e.g. for stats or clearing).
    pub fn basket_cache(&self) -> &BasketCache {
        &self.basket_cache
    }

    /// Configure the basket cache. Clears existing entries.
    pub fn set_cache_config(&mut self, config: CacheConfig) {
        self.basket_cache = BasketCache::new(config);
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

fn is_rntuple_class_name(class_name: &str) -> bool {
    class_name.to_ascii_lowercase().contains("rntuple")
}

fn to_rntuple_info(key: &KeyInfo) -> Option<RNTupleInfo> {
    if !is_rntuple_class_name(&key.class_name) {
        return None;
    }
    Some(RNTupleInfo {
        name: key.name.clone(),
        class_name: key.class_name.clone(),
        cycle: key.cycle,
    })
}

fn decompress_ntuple_blob(blob: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    if blob.len() == expected_len { Ok(blob.to_vec()) } else { decompress(blob, expected_len) }
}

fn scalar_type_byte_width(scalar_type: RNTupleScalarType) -> usize {
    match scalar_type {
        RNTupleScalarType::Bool | RNTupleScalarType::I8 | RNTupleScalarType::U8 => 1,
        RNTupleScalarType::I16 | RNTupleScalarType::U16 => 2,
        RNTupleScalarType::I32 | RNTupleScalarType::U32 | RNTupleScalarType::F32 => 4,
        RNTupleScalarType::I64 | RNTupleScalarType::U64 | RNTupleScalarType::F64 => 8,
    }
}

fn decode_primitive_page_f64(
    page_blob: &[u8],
    scalar_type: RNTupleScalarType,
    n_values: usize,
) -> Result<Vec<f64>> {
    let byte_width = scalar_type_byte_width(scalar_type);
    let expected_len = n_values.checked_mul(byte_width).ok_or_else(|| {
        RootError::Deserialization("RNTuple primitive decode length overflow".into())
    })?;
    if page_blob.len() != expected_len {
        return Err(RootError::Deserialization(format!(
            "RNTuple primitive page length mismatch: got={} expected={}",
            page_blob.len(),
            expected_len
        )));
    }

    let data = if n_values > 1 && byte_width > 1 {
        decode_byte_shuffle(page_blob, byte_width, n_values)?
    } else {
        page_blob.to_vec()
    };

    let mut out = Vec::with_capacity(n_values);
    match scalar_type {
        RNTupleScalarType::Bool => {
            out.extend(data.into_iter().map(|v| if v == 0 { 0.0 } else { 1.0 }));
        }
        RNTupleScalarType::U8 => {
            out.extend(data.into_iter().map(|v| v as f64));
        }
        RNTupleScalarType::I8 => {
            out.extend(data.into_iter().map(|v| zigzag_decode_u8(v) as f64));
        }
        RNTupleScalarType::U16 => {
            for chunk in data.chunks_exact(2) {
                out.push(u16::from_le_bytes([chunk[0], chunk[1]]) as f64);
            }
        }
        RNTupleScalarType::I16 => {
            for chunk in data.chunks_exact(2) {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(zigzag_decode_u16(raw) as f64);
            }
        }
        RNTupleScalarType::U32 => {
            for chunk in data.chunks_exact(4) {
                out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64);
            }
        }
        RNTupleScalarType::I32 => {
            for chunk in data.chunks_exact(4) {
                let raw = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push(zigzag_decode_u32(raw) as f64);
            }
        }
        RNTupleScalarType::U64 => {
            for chunk in data.chunks_exact(8) {
                out.push(u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as f64);
            }
        }
        RNTupleScalarType::I64 => {
            for chunk in data.chunks_exact(8) {
                let raw = u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                out.push(zigzag_decode_u64(raw) as f64);
            }
        }
        RNTupleScalarType::F32 => {
            for chunk in data.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64);
            }
        }
        RNTupleScalarType::F64 => {
            for chunk in data.chunks_exact(8) {
                out.push(f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]));
            }
        }
    }

    Ok(out)
}

fn decode_byte_shuffle(data: &[u8], elem_size: usize, n_values: usize) -> Result<Vec<u8>> {
    if data.len() != elem_size * n_values {
        return Err(RootError::Deserialization(format!(
            "RNTuple byte-shuffle length mismatch: got={} expected={}",
            data.len(),
            elem_size * n_values
        )));
    }
    let mut out = vec![0u8; data.len()];
    for byte_idx in 0..elem_size {
        let src = &data[byte_idx * n_values..(byte_idx + 1) * n_values];
        for (val_idx, byte) in src.iter().enumerate() {
            out[val_idx * elem_size + byte_idx] = *byte;
        }
    }
    Ok(out)
}

fn zigzag_decode_u8(v: u8) -> i8 {
    ((v >> 1) as i8) ^ (-((v & 1) as i8))
}

fn zigzag_decode_u16(v: u16) -> i16 {
    ((v >> 1) as i16) ^ (-((v & 1) as i16))
}

fn zigzag_decode_u32(v: u32) -> i32 {
    ((v >> 1) as i32) ^ (-((v & 1) as i32))
}

fn zigzag_decode_u64(v: u64) -> i64 {
    ((v >> 1) as i64) ^ (-((v & 1) as i64))
}

fn abs_element_count(page: &RNTuplePageSummary) -> usize {
    page.element_count_raw.unsigned_abs() as usize
}

fn find_matching_page_index<F>(
    pages: &[RNTuplePageSummary],
    used_pages: &[bool],
    start: usize,
    pred: F,
) -> Option<usize>
where
    F: Fn(&RNTuplePageSummary) -> bool,
{
    let from_start = (start..pages.len()).find(|&idx| !used_pages[idx] && pred(&pages[idx]));
    from_start
        .or_else(|| (0..start.min(pages.len())).find(|&idx| !used_pages[idx] && pred(&pages[idx])))
}

fn decode_u64_le_blob(page_blob: &[u8]) -> Result<Vec<u64>> {
    if !page_blob.len().is_multiple_of(8) {
        return Err(RootError::Deserialization(format!(
            "RNTuple offset blob length {} is not multiple of 8",
            page_blob.len()
        )));
    }
    let mut out = Vec::with_capacity(page_blob.len() / 8);
    for chunk in page_blob.chunks_exact(8) {
        out.push(u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    Ok(out)
}

fn derive_end_offsets(
    raw_offsets: &[u64],
    entry_count: usize,
    field_name: &str,
) -> Result<Vec<usize>> {
    let ends_u64: Vec<u64> = if raw_offsets.len() == entry_count {
        raw_offsets.to_vec()
    } else if raw_offsets.len() == entry_count.saturating_add(1) {
        if is_non_decreasing_u64(&raw_offsets[1..]) {
            raw_offsets[1..].to_vec()
        } else if raw_offsets.last().copied() == Some(0)
            && is_non_decreasing_u64(&raw_offsets[..entry_count])
        {
            raw_offsets[..entry_count].to_vec()
        } else {
            raw_offsets[1..].to_vec()
        }
    } else {
        return Err(RootError::Deserialization(format!(
            "RNTuple variable-array field '{}' unexpected offset count {} for entry_count={}",
            field_name,
            raw_offsets.len(),
            entry_count
        )));
    };
    let mut out = Vec::with_capacity(entry_count);
    let mut prev = 0usize;
    for end in ends_u64 {
        let end_usize = usize::try_from(end).map_err(|_| {
            RootError::Deserialization(format!(
                "RNTuple variable-array field '{}' offset {} does not fit usize",
                field_name, end
            ))
        })?;
        if end_usize < prev {
            return Err(RootError::Deserialization(format!(
                "RNTuple variable-array field '{}' offsets are not monotonic: {} < {}",
                field_name, end_usize, prev
            )));
        }
        out.push(end_usize);
        prev = end_usize;
    }
    Ok(out)
}

fn is_non_decreasing_u64(values: &[u64]) -> bool {
    values.windows(2).all(|w| w[1] >= w[0])
}

fn counts_to_end_offsets(values: &[u64]) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(values.len());
    let mut acc = 0usize;
    for count in values {
        let c = usize::try_from(*count).ok()?;
        acc = acc.checked_add(c)?;
        out.push(acc);
    }
    Some(out)
}

fn derive_end_offsets_candidates(
    raw_offsets: &[u64],
    entry_count: usize,
    field_name: &str,
) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if let Ok(ends) = derive_end_offsets(raw_offsets, entry_count, field_name) {
        out.push(ends);
    }

    let count_slices: Vec<&[u64]> = if raw_offsets.len() == entry_count {
        vec![raw_offsets]
    } else if raw_offsets.len() == entry_count.saturating_add(1) {
        vec![&raw_offsets[1..], &raw_offsets[..entry_count]]
    } else {
        Vec::new()
    };
    for slice in count_slices {
        let Some(ends) = counts_to_end_offsets(slice) else {
            continue;
        };
        if !out.iter().any(|existing| existing == &ends) {
            out.push(ends);
        }
    }
    out
}

fn is_integer_scalar_type(scalar_type: RNTupleScalarType) -> bool {
    matches!(
        scalar_type,
        RNTupleScalarType::I8
            | RNTupleScalarType::U8
            | RNTupleScalarType::I16
            | RNTupleScalarType::U16
            | RNTupleScalarType::I32
            | RNTupleScalarType::U32
            | RNTupleScalarType::I64
            | RNTupleScalarType::U64
    )
}

fn decode_packed_integer_page_f64(
    page_blob: &[u8],
    scalar_type: RNTupleScalarType,
    total_elems: usize,
    stored_width: usize,
) -> Result<Vec<f64>> {
    let expected_len = total_elems.checked_mul(stored_width).ok_or_else(|| {
        RootError::Deserialization("RNTuple packed-integer page size overflow".to_string())
    })?;
    if page_blob.len() != expected_len {
        return Err(RootError::Deserialization(format!(
            "RNTuple packed-integer page length mismatch: got={} expected={}",
            page_blob.len(),
            expected_len
        )));
    }
    let signed = matches!(
        scalar_type,
        RNTupleScalarType::I8
            | RNTupleScalarType::I16
            | RNTupleScalarType::I32
            | RNTupleScalarType::I64
    );
    let mut out = Vec::with_capacity(total_elems);
    for chunk in page_blob.chunks_exact(stored_width) {
        let value = if signed {
            match stored_width {
                1 => i8::from_le_bytes([chunk[0]]) as f64,
                2 => i16::from_le_bytes([chunk[0], chunk[1]]) as f64,
                4 => i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64,
                8 => i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as f64,
                _ => {
                    return Err(RootError::Deserialization(format!(
                        "RNTuple packed-integer unsupported signed width {}",
                        stored_width
                    )));
                }
            }
        } else {
            match stored_width {
                1 => u8::from_le_bytes([chunk[0]]) as f64,
                2 => u16::from_le_bytes([chunk[0], chunk[1]]) as f64,
                4 => u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64,
                8 => u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as f64,
                _ => {
                    return Err(RootError::Deserialization(format!(
                        "RNTuple packed-integer unsupported unsigned width {}",
                        stored_width
                    )));
                }
            }
        };
        out.push(value);
    }
    Ok(out)
}

fn decode_rntuple_numeric_page_f64(
    file: &RootFile,
    page: &RNTuplePageSummary,
    scalar_type: RNTupleScalarType,
    total_elems: usize,
    field_name: &str,
) -> Result<Vec<f64>> {
    decode_rntuple_numeric_page_f64_with_options(
        file,
        page,
        scalar_type,
        total_elems,
        field_name,
        true,
    )
}

fn decode_rntuple_numeric_page_f64_with_options(
    file: &RootFile,
    page: &RNTuplePageSummary,
    scalar_type: RNTupleScalarType,
    total_elems: usize,
    field_name: &str,
    allow_packed_integer_fallback: bool,
) -> Result<Vec<f64>> {
    let expected_nbytes =
        total_elems.checked_mul(scalar_type_byte_width(scalar_type)).ok_or_else(|| {
            RootError::Deserialization(format!(
                "RNTuple field '{}' expected page size overflow",
                field_name
            ))
        })?;
    let page_blob = file.read_file_range(page.position, page.nbytes_on_storage as u64)?;
    match decompress_ntuple_blob(page_blob, expected_nbytes)
        .and_then(|decompressed| decode_primitive_page_f64(&decompressed, scalar_type, total_elems))
    {
        Ok(values) => Ok(values),
        Err(primary_err) => {
            if !allow_packed_integer_fallback || !is_integer_scalar_type(scalar_type) {
                return Err(primary_err);
            }
            let native_width = scalar_type_byte_width(scalar_type);
            for packed_width in [1usize, 2, 4, 8] {
                if packed_width >= native_width {
                    continue;
                }
                let expected_packed = match total_elems.checked_mul(packed_width) {
                    Some(v) => v,
                    None => continue,
                };
                let packed = match decompress_ntuple_blob(page_blob, expected_packed) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if let Ok(values) =
                    decode_packed_integer_page_f64(&packed, scalar_type, total_elems, packed_width)
                {
                    return Ok(values);
                }
            }
            Err(primary_err)
        }
    }
}

fn decode_rntuple_offset_page_u64(
    file: &RootFile,
    page: &RNTuplePageSummary,
    entry_count: usize,
    field_name: &str,
) -> Result<Vec<u64>> {
    let blob = file.read_file_range(page.position, page.nbytes_on_storage as u64)?;
    let expected_direct = [entry_count, entry_count.saturating_add(1)];

    if blob.len() % 8 == 0 {
        let direct = decode_u64_le_blob(blob)?;
        if expected_direct.contains(&direct.len())
            && derive_end_offsets(&direct, entry_count, field_name).is_ok()
        {
            return Ok(direct);
        }
    }

    let expected_sizes = [entry_count.checked_mul(8), entry_count.saturating_add(1).checked_mul(8)];
    for expected in expected_sizes.into_iter().flatten() {
        if let Ok(decompressed) = decompress_ntuple_blob(blob, expected) {
            if decompressed.len() % 8 != 0 {
                continue;
            }
            let decoded = decode_u64_le_blob(&decompressed)?;
            if expected_direct.contains(&decoded.len())
                && derive_end_offsets(&decoded, entry_count, field_name).is_ok()
            {
                return Ok(decoded);
            }
        }
    }

    // Fallback: some layouts encode offsets as compressed primitive pages
    // (e.g. u32/u64 with byte-shuffle), not raw LE u64 blobs.
    let candidate_lens = [entry_count, entry_count.saturating_add(1)];
    let candidate_types = [
        RNTupleScalarType::U64,
        RNTupleScalarType::I64,
        RNTupleScalarType::U32,
        RNTupleScalarType::I32,
    ];
    for total_elems in candidate_lens {
        for scalar_type in candidate_types {
            let expected_nbytes = match total_elems.checked_mul(scalar_type_byte_width(scalar_type))
            {
                Some(v) => v,
                None => continue,
            };
            let decompressed = match decompress_ntuple_blob(blob, expected_nbytes) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let decoded = match decode_primitive_page_f64(&decompressed, scalar_type, total_elems) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let mut out = Vec::with_capacity(decoded.len());
            let mut ok = true;
            for value in decoded {
                if !value.is_finite() || value < 0.0 {
                    ok = false;
                    break;
                }
                let rounded = value.round();
                if (rounded - value).abs() > 1e-9 {
                    ok = false;
                    break;
                }
                if rounded > (u64::MAX as f64) {
                    ok = false;
                    break;
                }
                out.push(rounded as u64);
            }
            if ok
                && expected_direct.contains(&out.len())
                && derive_end_offsets(&out, entry_count, field_name).is_ok()
            {
                return Ok(out);
            }
        }
    }

    Err(RootError::Deserialization(format!(
        "RNTuple variable-array field '{}' offset decode failed for page bytes={} entry_count={}",
        field_name,
        blob.len(),
        entry_count
    )))
}

struct DecodedVariablePageBundleF64 {
    offset_page_index: usize,
    data_page_index: usize,
    values: Vec<Vec<f64>>,
}

#[allow(clippy::too_many_arguments)]
fn decode_pair_variable_side_f64(
    file: &RootFile,
    pagelist: &RNTuplePageListSummary,
    used_pages: &[bool],
    start: usize,
    entry_count: usize,
    field_name: &str,
    _side_label: &str,
    elem_ty: RNTupleScalarType,
) -> Result<Option<DecodedVariablePageBundleF64>> {
    decode_variable_array_bundle_f64(
        file,
        pagelist,
        used_pages,
        start,
        entry_count,
        field_name,
        elem_ty,
    )
}

fn decode_variable_array_bundle_f64(
    file: &RootFile,
    pagelist: &RNTuplePageListSummary,
    used_pages: &[bool],
    start: usize,
    entry_count: usize,
    field_name: &str,
    elem_ty: RNTupleScalarType,
) -> Result<Option<DecodedVariablePageBundleF64>> {
    for offset_page_idx in start..pagelist.pages.len() {
        if used_pages[offset_page_idx] {
            continue;
        }
        let offset_page = &pagelist.pages[offset_page_idx];
        if offset_page.record_tag_raw >= 0 {
            continue;
        }

        let raw_offsets =
            match decode_rntuple_offset_page_u64(file, offset_page, entry_count, field_name) {
                Ok(v) => v,
                Err(_) => continue,
            };
        let end_offsets_candidates =
            derive_end_offsets_candidates(&raw_offsets, entry_count, field_name);
        if end_offsets_candidates.is_empty() {
            continue;
        }

        for end_offsets in end_offsets_candidates {
            let total_elems = end_offsets.last().copied().unwrap_or(0usize);
            let allow_packed_integer_fallback = entry_count > 1;

            for (data_page_idx, data_page) in
                pagelist.pages.iter().enumerate().skip(offset_page_idx.saturating_add(1))
            {
                if used_pages[data_page_idx] {
                    continue;
                }
                if data_page.record_tag_raw >= 0 {
                    continue;
                }
                let flat_values = match decode_rntuple_numeric_page_f64_with_options(
                    file,
                    data_page,
                    elem_ty,
                    total_elems,
                    field_name,
                    allow_packed_integer_fallback,
                ) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let mut values = Vec::with_capacity(entry_count);
                let mut begin = 0usize;
                for end in end_offsets.iter().copied() {
                    values.push(flat_values[begin..end].to_vec());
                    begin = end;
                }
                return Ok(Some(DecodedVariablePageBundleF64 {
                    offset_page_index: offset_page_idx,
                    data_page_index: data_page_idx,
                    values,
                }));
            }
        }
    }
    Ok(None)
}

enum PairLayout {
    ScalarScalar {
        left_scalar_type: RNTupleScalarType,
        right_scalar_type: RNTupleScalarType,
    },
    ScalarVariableRight {
        left_scalar_type: RNTupleScalarType,
        right_element_scalar_type: RNTupleScalarType,
    },
    VariableLeftScalar {
        left_element_scalar_type: RNTupleScalarType,
        right_scalar_type: RNTupleScalarType,
    },
    VariableVariable {
        left_element_scalar_type: RNTupleScalarType,
        right_element_scalar_type: RNTupleScalarType,
    },
}

fn parse_pair_layout(type_name: &str) -> Option<PairLayout> {
    let compact = type_name.replace(' ', "");
    let inner =
        template_payload(&compact, "std::pair<").or_else(|| template_payload(&compact, "pair<"))?;
    let (left, right) = split_top_level_once(inner, ',')?;
    let left_scalar_ty = parse_scalar_type_token(left);
    let right_scalar_ty = parse_scalar_type_token(right);
    let left_vec_elem_ty = parse_std_vector_like_token(left).and_then(parse_scalar_type_token);
    let right_vec_elem_ty = parse_std_vector_like_token(right).and_then(parse_scalar_type_token);

    match (left_scalar_ty, right_scalar_ty, left_vec_elem_ty, right_vec_elem_ty) {
        (Some(left), Some(right), _, _) => {
            Some(PairLayout::ScalarScalar { left_scalar_type: left, right_scalar_type: right })
        }
        (Some(left), None, _, Some(right_elem)) => Some(PairLayout::ScalarVariableRight {
            left_scalar_type: left,
            right_element_scalar_type: right_elem,
        }),
        (None, Some(right), Some(left_elem), _) => Some(PairLayout::VariableLeftScalar {
            left_element_scalar_type: left_elem,
            right_scalar_type: right,
        }),
        (None, None, Some(left_elem), Some(right_elem)) => Some(PairLayout::VariableVariable {
            left_element_scalar_type: left_elem,
            right_element_scalar_type: right_elem,
        }),
        _ => None,
    }
}

fn parse_std_vector_like_token(type_name: &str) -> Option<&str> {
    let inner = template_payload(type_name, "std::vector<")
        .or_else(|| template_payload(type_name, "vector<"))
        .or_else(|| template_payload(type_name, "ROOT::VecOps::RVec<"))
        .or_else(|| template_payload(type_name, "RVec<"))?;
    let (elem_ty, _) = split_top_level_once(inner, ',').unwrap_or((inner, ""));
    Some(elem_ty)
}

fn template_payload<'a>(type_name: &'a str, prefix: &str) -> Option<&'a str> {
    if !type_name.starts_with(prefix) || !type_name.ends_with('>') {
        return None;
    }
    Some(&type_name[prefix.len()..type_name.len() - 1])
}

fn split_top_level_once(s: &str, delim: char) -> Option<(&str, &str)> {
    let mut depth = 0usize;
    for (idx, ch) in s.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            _ if ch == delim && depth == 0 => {
                let left = &s[..idx];
                let right = &s[idx + ch.len_utf8()..];
                return Some((left, right));
            }
            _ => {}
        }
    }
    None
}

fn parse_scalar_type_token(type_name: &str) -> Option<RNTupleScalarType> {
    let t = type_name.to_ascii_lowercase();
    match t.as_str() {
        "bool" | "std::bool_t" => Some(RNTupleScalarType::Bool),
        "uint64_t" | "std::uint64_t" | "u64" | "unsignedlonglong" => Some(RNTupleScalarType::U64),
        "int64_t" | "std::int64_t" | "i64" | "longlong" => Some(RNTupleScalarType::I64),
        "uint32_t" | "std::uint32_t" | "u32" | "unsignedint" => Some(RNTupleScalarType::U32),
        "int32_t" | "std::int32_t" | "i32" | "int" => Some(RNTupleScalarType::I32),
        "uint16_t" | "std::uint16_t" | "u16" | "unsignedshort" => Some(RNTupleScalarType::U16),
        "int16_t" | "std::int16_t" | "i16" | "short" => Some(RNTupleScalarType::I16),
        "uint8_t" | "std::uint8_t" | "u8" | "unsignedchar" => Some(RNTupleScalarType::U8),
        "int8_t" | "std::int8_t" | "i8" | "char" | "signedchar" => Some(RNTupleScalarType::I8),
        "double" | "f64" => Some(RNTupleScalarType::F64),
        "float" | "f32" => Some(RNTupleScalarType::F32),
        _ => None,
    }
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
    use std::collections::HashSet;

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(name)
    }

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

    #[test]
    fn rntuple_class_name_matching_is_case_insensitive() {
        assert!(is_rntuple_class_name("ROOT::Experimental::RNTuple"));
        assert!(is_rntuple_class_name("rntuple"));
        assert!(!is_rntuple_class_name("TTree"));
    }

    #[test]
    fn to_rntuple_info_filters_non_rntuple_keys() {
        let keep = KeyInfo {
            name: "Events".into(),
            class_name: "ROOT::Experimental::RNTuple".into(),
            cycle: 1,
        };
        let drop = KeyInfo { name: "tree".into(), class_name: "TTree".into(), cycle: 1 };

        let info = to_rntuple_info(&keep).expect("expected rntuple key");
        assert_eq!(info.name, "Events");
        assert_eq!(info.class_name, "ROOT::Experimental::RNTuple");
        assert_eq!(info.cycle, 1);
        assert!(to_rntuple_info(&drop).is_none());
    }

    #[test]
    fn derive_end_offsets_accepts_entry_count_or_plus_one_layouts() {
        let one = derive_end_offsets(&[3], 1, "arr").expect("single-count offsets should parse");
        assert_eq!(one, vec![3]);

        let plus_one =
            derive_end_offsets(&[0, 3, 5], 2, "arr").expect("plus-one offsets should parse");
        assert_eq!(plus_one, vec![3, 5]);
    }

    #[test]
    fn derive_end_offsets_rejects_non_monotonic_or_bad_lengths() {
        let err =
            derive_end_offsets(&[4, 2], 2, "arr").expect_err("expected non-monotonic failure");
        let msg = err.to_string();
        assert!(msg.contains("not monotonic"), "unexpected error: {}", msg);

        let err = derive_end_offsets(&[1, 2, 3], 1, "arr").expect_err("expected bad-len failure");
        let msg = err.to_string();
        assert!(msg.contains("unexpected offset count"), "unexpected error: {}", msg);
    }

    #[test]
    fn parse_pair_layout_supports_scalar_scalar() {
        let layout =
            parse_pair_layout("std::pair<float,std::int32_t>").expect("expected pair layout");
        assert!(matches!(
            layout,
            PairLayout::ScalarScalar {
                left_scalar_type: RNTupleScalarType::F32,
                right_scalar_type: RNTupleScalarType::I32
            }
        ));
    }

    #[test]
    fn parse_pair_layout_supports_scalar_variable() {
        let layout = parse_pair_layout("std::pair<float,std::vector<std::int32_t>>")
            .expect("expected pair layout");
        assert!(matches!(
            layout,
            PairLayout::ScalarVariableRight {
                left_scalar_type: RNTupleScalarType::F32,
                right_element_scalar_type: RNTupleScalarType::I32
            }
        ));
    }

    #[test]
    fn parse_pair_layout_supports_variable_scalar() {
        let layout = parse_pair_layout("std::pair<std::vector<std::int32_t>,float>")
            .expect("expected pair layout");
        assert!(matches!(
            layout,
            PairLayout::VariableLeftScalar {
                left_element_scalar_type: RNTupleScalarType::I32,
                right_scalar_type: RNTupleScalarType::F32
            }
        ));
    }

    #[test]
    fn parse_pair_layout_supports_variable_variable() {
        let layout = parse_pair_layout("std::pair<std::vector<std::int32_t>,std::vector<float>>")
            .expect("expected pair layout");
        assert!(matches!(
            layout,
            PairLayout::VariableVariable {
                left_element_scalar_type: RNTupleScalarType::I32,
                right_element_scalar_type: RNTupleScalarType::F32
            }
        ));
    }

    #[test]
    fn parse_pair_layout_rejects_unsupported_children() {
        assert!(parse_pair_layout("std::pair<MyType,float>").is_none());
        assert!(parse_pair_layout("std::pair<float,std::array<float,2>>").is_none());
    }

    #[test]
    fn decode_fails_when_required_schema_field_has_no_matching_pages() {
        let path = fixture_path("rntuple_simple.root");
        assert!(path.exists(), "missing fixture: {}", path.display());

        let f = RootFile::open(&path).expect("failed to open simple rntuple fixture");
        let footer = f.read_rntuple_footer_summary("Events").expect("footer summary failed");
        let entry_span = footer.cluster_groups[0].entry_span;

        // Request a field with entry_span * 2 entries — no page in the fixture
        // has that many elements, so the decoder must fail for the required field.
        // (Using the real entry_span would match an existing page by shape, since
        // pages carry no field-name metadata in the RNTuple format.)
        let schema = RNTupleSchemaSummary {
            ntuple_name: Some("Events".to_string()),
            fields: vec![crate::rntuple::RNTupleSchemaField {
                name: "pt_missing".to_string(),
                type_name: "float".to_string(),
                kind: RNTupleFieldKind::Primitive,
                scalar_type: Some(RNTupleScalarType::F32),
                element_scalar_type: None,
                fixed_len: None,
            }],
        };
        let required_field_names = HashSet::from(["pt_missing".to_string()]);
        let err = f
            .decode_rntuple_fields_f64_internal(
                "Events",
                0,
                &schema,
                &required_field_names,
                entry_span * 2,
            )
            .expect_err("expected deserialization failure for schema/page mismatch");
        assert!(matches!(err, RootError::Deserialization(_)));
        let msg = err.to_string();
        assert!(msg.contains("pt_missing"), "unexpected error message: {}", msg);
        assert!(msg.contains("page not found"), "unexpected error message: {}", msg);
    }
}
