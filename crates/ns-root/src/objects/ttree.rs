//! TTree and TBranch binary deserialization from ROOT streamer format.

use crate::error::{Result, RootError};
use crate::rbuffer::RBuffer;
use crate::tree::{BranchInfo, LeafType, Tree};

/// Skip a versioned object by reading its byte-count and jumping to end_pos.
fn skip_versioned(r: &mut RBuffer) -> Result<()> {
    let (_ver, end) = r.read_version()?;
    if let Some(end_pos) = end {
        r.set_pos(end_pos);
    }
    Ok(())
}

/// Read a TTree from a decompressed TKey payload.
pub fn read_ttree(payload: &[u8]) -> Result<Tree> {
    let mut r = RBuffer::new(payload);

    // TTree version header
    let (tree_ver, tree_end) = r.read_version()?;
    let tree_end =
        tree_end.ok_or_else(|| RootError::Deserialization("TTree missing byte count".into()))?;

    // TNamed (name, title)
    let (name, _title) = r.read_tnamed()?;

    // TAttLine, TAttFill, TAttMarker — skip
    skip_versioned(&mut r)?;
    skip_versioned(&mut r)?;
    skip_versioned(&mut r)?;

    // TTree fields
    let entries = r.read_i64()? as u64; // fEntries
    let _tot_bytes = r.read_i64()?; // fTotBytes
    let _zip_bytes = r.read_i64()?; // fZipBytes
    let _saved_bytes = r.read_i64()?; // fSavedBytes

    if tree_ver >= 18 {
        let _flushed_bytes = r.read_i64()?; // fFlushedBytes
    }

    let _weight = r.read_f64()?; // fWeight
    let _timer_interval = r.read_i32()?; // fTimerInterval
    let _scan_field = r.read_i32()?; // fScanField
    let _update = r.read_i32()?; // fUpdate

    if tree_ver >= 18 {
        let _default_entry_offset_len = r.read_i32()?;
    }

    let n_cluster_range = if tree_ver >= 19 { r.read_i32()? } else { 0 };

    let _max_entries = r.read_i64()?; // fMaxEntries
    let _max_entry_loop = r.read_i64()?; // fMaxEntryLoop
    let _max_virtual_size = r.read_i64()?; // fMaxVirtualSize
    let _auto_save = r.read_i64()?; // fAutoSave

    if tree_ver >= 18 {
        let _auto_flush = r.read_i64()?; // fAutoFlush
    }

    let _estimate = r.read_i64()?; // fEstimate

    // Cluster range arrays (v19+)
    // ROOT always writes the UChar_t array size header, even when n_cluster_range == 0
    if tree_ver >= 19 {
        // fClusterRangeEnd: array of i64
        let _n = r.read_u8()?;
        for _ in 0..n_cluster_range {
            let _val = r.read_i64()?;
        }
        // fClusterSize: array of i64
        let _n = r.read_u8()?;
        for _ in 0..n_cluster_range {
            let _val = r.read_i64()?;
        }
    }

    // IOBits (v20+)
    if tree_ver >= 20 {
        // skip TBits: versioned object
        skip_versioned(&mut r)?;
    }

    // fBranches: TObjArray of TBranch
    let branches = read_tobjarray_branches(&mut r)?;

    // We have what we need — skip rest to tree_end
    r.set_pos(tree_end);

    Ok(Tree { name, entries, branches })
}

// ── TObjArray parsing ──────────────────────────────────────────

/// ROOT reference system constants.
const K_BYTE_COUNT_MASK: u32 = 0x4000_0000;
const K_NEW_CLASS_TAG: u32 = 0xFFFF_FFFF;
const K_CLASS_MASK: u32 = 0x8000_0000;

/// State for tracking class tags in ROOT's reference system.
///
/// ROOT uses a byte-offset based reference system for class names in TObjArray:
/// - `kNewClassTag (0xFFFFFFFF)` introduces a new class name (null-terminated C string)
/// - `kClassMask (0x80000000) | offset` references a previously registered class
/// - `kByteCountMask (0x40000000)` wraps objects with a byte count
///
/// Classes are registered by their byte offset in the stream for later reference.
struct ClassRefTracker {
    /// Maps byte offset → class name for the ROOT reference system.
    classes: Vec<(usize, String)>,
}

impl ClassRefTracker {
    fn new() -> Self {
        Self { classes: Vec::new() }
    }

    /// Look up a class by its byte offset tag.
    fn lookup(&self, offset: usize) -> Option<&str> {
        self.classes.iter().find(|(off, _)| *off == offset).map(|(_, name)| name.as_str())
    }

    /// Read a single element from a TObjArray.
    /// Returns (class_name, object_end_pos) — object_end_pos is the absolute
    /// position where this element's data ends (from byte-count wrapper).
    fn read_element(&mut self, r: &mut RBuffer) -> Result<Option<(String, usize)>> {
        let tag = r.read_u32()?;

        if tag == 0 {
            return Ok(None);
        }

        // Objects in TObjArray are wrapped with kByteCountMask
        if tag & K_BYTE_COUNT_MASK != 0 {
            let byte_count = (tag & !K_BYTE_COUNT_MASK) as usize;
            let obj_start = r.pos() - 4; // start of this u32
            let obj_end = obj_start + 4 + byte_count;

            // Now read the class tag
            let class_tag_pos = r.pos(); // position where class tag is stored
            let class_tag = r.read_u32()?;

            let class_name = if class_tag == K_NEW_CLASS_TAG {
                // New class: read null-terminated class name
                let name_start = r.pos();
                let name = r.read_cstring()?;
                // Register this class with the offset of the class tag
                // ROOT uses (class_tag_pos | kClassMask) for back-references
                self.classes.push((class_tag_pos, name.clone()));
                // Also register at name_start - some ROOT versions reference differently
                let _ = name_start;
                name
            } else if class_tag & K_CLASS_MASK != 0 {
                // Reference to existing class by offset
                let ref_offset = (class_tag & !K_CLASS_MASK) as usize;
                match self.lookup(ref_offset) {
                    Some(name) => name.to_string(),
                    None => {
                        return Err(RootError::Deserialization(format!(
                            "class ref offset {} not found (tag={:#010x}, known: {:?})",
                            ref_offset, class_tag, self.classes
                        )));
                    }
                }
            } else {
                return Err(RootError::Deserialization(format!(
                    "unexpected class tag {:#010x} at pos {}",
                    class_tag, class_tag_pos
                )));
            };

            return Ok(Some((class_name, obj_end)));
        }

        Err(RootError::Deserialization(format!(
            "unexpected tag {:#010x} in TObjArray at pos {}",
            tag,
            r.pos() - 4
        )))
    }
}

/// Read a TObjArray of TBranch objects.
fn read_tobjarray_branches(r: &mut RBuffer) -> Result<Vec<BranchInfo>> {
    let (_ver, arr_end) = r.read_version()?;
    let arr_end =
        arr_end.ok_or_else(|| RootError::Deserialization("TObjArray missing byte count".into()))?;

    // TObject header
    r.read_tobject()?;

    // Name (often empty)
    let _name = r.read_string()?;

    let count = r.read_i32()?;
    let _low_bound = r.read_i32()?;

    let mut branches = Vec::new();
    let mut tracker = ClassRefTracker::new();

    for _ in 0..count {
        let element = tracker.read_element(r)?;

        match element {
            None => {
                // Null entry — skip
            }
            Some((ref class_name, obj_end)) if class_name == "TBranch" => match read_tbranch(r) {
                Ok(branch) => branches.push(branch),
                Err(_) => {
                    r.set_pos(obj_end);
                }
            },
            Some((class_name, obj_end)) => {
                // Unknown branch type — try to read as TBranch
                log::debug!("skipping unknown branch class: {}", class_name);
                match read_tbranch(r) {
                    Ok(branch) => branches.push(branch),
                    Err(_) => {
                        r.set_pos(obj_end);
                    }
                }
            }
        }
    }

    r.set_pos(arr_end);
    Ok(branches)
}

/// Skip a TObjArray (for arrays we don't care about).
fn skip_tobjarray(r: &mut RBuffer) -> Result<()> {
    let (_ver, end) = r.read_version()?;
    if let Some(end_pos) = end {
        r.set_pos(end_pos);
    }
    Ok(())
}

// ── TBranch parsing ────────────────────────────────────────────

/// Read a single TBranch from the buffer.
fn read_tbranch(r: &mut RBuffer) -> Result<BranchInfo> {
    let (branch_ver, branch_end) = r.read_version()?;
    let branch_end = branch_end
        .ok_or_else(|| RootError::Deserialization("TBranch missing byte count".into()))?;

    // TNamed
    let (name, _title) = r.read_tnamed()?;

    // TAttFill — skip
    skip_versioned(r)?;

    // Branch fields
    let _compress = r.read_i32()?; // fCompress
    let _basket_size = r.read_i32()?; // fBasketSize
    let _entry_offset_len = r.read_i32()?; // fEntryOffsetLen
    let write_basket = r.read_i32()?; // fWriteBasket (= number of valid baskets)
    let _entry_number = r.read_i64()?; // fEntryNumber

    // IOBits (v13+): TIOFeatures streamed as a versioned object
    if branch_ver >= 13 {
        skip_versioned(r)?;
    }

    let _offset = r.read_i32()?; // fOffset
    let max_baskets = r.read_i32()?; // fMaxBaskets
    let _split_level = r.read_i32()?; // fSplitLevel
    let entries = r.read_i64()? as u64; // fEntries

    if branch_ver >= 11 {
        let _first_entry = r.read_i64()?; // fFirstEntry
    }

    let _tot_bytes = r.read_i64()?; // fTotBytes
    let _zip_bytes = r.read_i64()?; // fZipBytes

    // fBranches: TObjArray of sub-branches
    skip_tobjarray(r)?;

    // fLeaves: TObjArray of TLeaf
    let leaf_type = read_tobjarray_leaves(r)?;

    // fBaskets: TObjArray (in-memory baskets, usually empty or references)
    skip_tobjarray(r)?;

    // Now arrays of size fMaxBaskets; first fWriteBasket entries are valid.
    let n_baskets = write_basket as usize;
    let max = max_baskets as usize;

    // fBasketBytes: i32[fMaxBaskets] — but stored as TArrayI w/ 1-byte header
    let _n_bytes_arr = r.read_u8()?; // TArray count byte (should match max)
    let mut basket_bytes = Vec::with_capacity(n_baskets);
    for i in 0..max {
        let v = r.read_i32()? as u32;
        if i < n_baskets {
            basket_bytes.push(v);
        }
    }

    // fBasketEntry: i64[fMaxBaskets] — first fWriteBasket+1 valid
    let _n_entry_arr = r.read_u8()?;
    let mut basket_entry = Vec::with_capacity(n_baskets + 1);
    for i in 0..max {
        let v = r.read_i64()? as u64;
        if i <= n_baskets {
            basket_entry.push(v);
        }
    }

    // fBasketSeek: i64[fMaxBaskets] — first fWriteBasket valid
    let _n_seek_arr = r.read_u8()?;
    let mut basket_seek = Vec::with_capacity(n_baskets);
    for i in 0..max {
        let v = r.read_i64()? as u64;
        if i < n_baskets {
            basket_seek.push(v);
        }
    }

    // Skip to end of branch
    if branch_end > r.pos() {
        r.set_pos(branch_end);
    }

    // Determine leaf type (default to F64 if no leaves found)
    let leaf = leaf_type.unwrap_or(LeafType::F64);

    Ok(BranchInfo {
        name,
        leaf_type: leaf,
        entries,
        basket_bytes,
        basket_entry,
        basket_seek,
        n_baskets,
    })
}

// ── TLeaf parsing ──────────────────────────────────────────────

/// Read a TObjArray of TLeaf objects and return the leaf type of the first one.
fn read_tobjarray_leaves(r: &mut RBuffer) -> Result<Option<LeafType>> {
    let (_ver, arr_end) = r.read_version()?;
    let arr_end = arr_end.ok_or_else(|| {
        RootError::Deserialization("TObjArray (leaves) missing byte count".into())
    })?;

    // TObject header
    r.read_tobject()?;

    // Name
    let _name = r.read_string()?;

    let count = r.read_i32()?;
    let _low_bound = r.read_i32()?;

    let mut leaf_type = None;
    let mut tracker = ClassRefTracker::new();

    for _ in 0..count {
        let element = tracker.read_element(r)?;

        match element {
            None => {}
            Some((class_name, obj_end)) => {
                let lt = leaf_type_from_class(&class_name);

                // The element byte-count wrapper already includes the versioned
                // TLeaf data. The TLeaf versioned header follows next in the stream.
                // Skip to obj_end to move past this leaf.
                if leaf_type.is_none() {
                    leaf_type = lt;
                }
                r.set_pos(obj_end);
            }
        }
    }

    r.set_pos(arr_end);
    Ok(leaf_type)
}

/// Map a TLeaf class name to a `LeafType`.
fn leaf_type_from_class(class_name: &str) -> Option<LeafType> {
    match class_name {
        "TLeafF" => Some(LeafType::F32),
        "TLeafD" => Some(LeafType::F64),
        "TLeafI" => Some(LeafType::I32),
        "TLeafL" => Some(LeafType::I64),
        "TLeafS" => Some(LeafType::I16),
        "TLeafB" => Some(LeafType::I8),
        "TLeafO" => Some(LeafType::Bool),
        _ => None,
    }
}
