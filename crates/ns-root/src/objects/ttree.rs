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
    let tree_end = tree_end.ok_or_else(|| {
        RootError::Deserialization("TTree missing byte count".into())
    })?;

    // TNamed (name, title)
    let (name, _title) = r.read_tnamed()?;

    // TAttLine, TAttFill, TAttMarker — skip
    skip_versioned(&mut r)?;
    skip_versioned(&mut r)?;
    skip_versioned(&mut r)?;

    // TTree fields
    let entries = r.read_i64()? as u64;     // fEntries
    let _tot_bytes = r.read_i64()?;          // fTotBytes
    let _zip_bytes = r.read_i64()?;          // fZipBytes
    let _saved_bytes = r.read_i64()?;        // fSavedBytes

    if tree_ver >= 18 {
        let _flushed_bytes = r.read_i64()?;  // fFlushedBytes
    }

    let _weight = r.read_f64()?;             // fWeight
    let _timer_interval = r.read_i32()?;     // fTimerInterval
    let _scan_field = r.read_i32()?;         // fScanField
    let _update = r.read_i32()?;             // fUpdate

    if tree_ver >= 18 {
        let _default_entry_offset_len = r.read_i32()?;
    }

    let n_cluster_range = if tree_ver >= 19 {
        r.read_i32()?
    } else {
        0
    };

    let _max_entries = r.read_i64()?;        // fMaxEntries
    let _max_entry_loop = r.read_i64()?;     // fMaxEntryLoop
    let _max_virtual_size = r.read_i64()?;   // fMaxVirtualSize
    let _auto_save = r.read_i64()?;          // fAutoSave

    if tree_ver >= 18 {
        let _auto_flush = r.read_i64()?;     // fAutoFlush
    }

    let _estimate = r.read_i64()?;           // fEstimate

    // Cluster range arrays (v19+)
    if tree_ver >= 19 && n_cluster_range > 0 {
        // fClusterRangeEnd: array of i64
        let _n = r.read_u8()?; // TArray header byte
        for _ in 0..n_cluster_range {
            let _val = r.read_i64()?;
        }
        // fClusterSize: array of i64
        let _n = r.read_u8()?; // TArray header byte
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

/// State for tracking class tags in ROOT's reference system.
struct ClassRefTracker {
    classes: Vec<String>,
}

impl ClassRefTracker {
    fn new() -> Self {
        Self { classes: Vec::new() }
    }

    /// Read a class tag and return the class name.
    fn read_class_tag(&mut self, r: &mut RBuffer) -> Result<Option<String>> {
        let tag = r.read_u32()?;

        if tag == 0 {
            // Null pointer
            return Ok(None);
        }

        // New class tag: bit 30 set, top 2 bits = 0b10
        if tag & 0xC000_0000 == 0x8000_0000 {
            // New class — read class name
            let class_name = r.read_string()?;
            self.classes.push(class_name.clone());
            return Ok(Some(class_name));
        }

        // Existing class reference (bits 31:30 = 0b11)
        if tag & 0xC000_0000 == 0xC000_0000 {
            let idx = (tag & !0xC000_0000) as usize;
            // ROOT indexes class tags by byte offset, but we just index
            // sequentially. Fall back to lookup.
            if let Some(name) = self.classes.last() {
                return Ok(Some(name.clone()));
            }
            return Err(RootError::Deserialization(format!(
                "class ref tag {:#x} not found (idx={})", tag, idx
            )));
        }

        // Object tag — has byte count, then class tag
        // High bit set means object with byte count
        if tag & 0x4000_0000 != 0 {
            // This is a byte-count — the object starts here
            let _byte_count = tag & !0x4000_0000;
            // Next read the actual class tag
            return self.read_class_tag(r);
        }

        Err(RootError::Deserialization(format!(
            "unexpected tag {:#010x} in TObjArray", tag
        )))
    }
}

/// Read a TObjArray of TBranch objects.
fn read_tobjarray_branches(r: &mut RBuffer) -> Result<Vec<BranchInfo>> {
    let (_ver, arr_end) = r.read_version()?;
    let arr_end = arr_end.ok_or_else(|| {
        RootError::Deserialization("TObjArray missing byte count".into())
    })?;

    // TObject header
    r.read_tobject()?;

    // Name (often empty)
    let _name = r.read_string()?;

    let count = r.read_i32()?;
    let _low_bound = r.read_i32()?;

    let mut branches = Vec::new();
    let mut tracker = ClassRefTracker::new();

    for _ in 0..count {
        let class = tracker.read_class_tag(r)?;

        match class.as_deref() {
            None => {
                // Null entry — skip
            }
            Some("TBranch") => {
                let branch = read_tbranch(r)?;
                branches.push(branch);
            }
            Some(other) => {
                // Unknown branch type — try to skip via byte count
                // The class tag already consumed a versioned header for
                // the object, so we need to skip it.
                log::debug!("skipping unknown branch class: {}", other);
                // Try to read as TBranch anyway (many subclasses share layout)
                match read_tbranch(r) {
                    Ok(branch) => branches.push(branch),
                    Err(_) => {
                        // If parsing fails, just skip to array end
                        r.set_pos(arr_end);
                        return Ok(branches);
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
    let branch_end = branch_end.ok_or_else(|| {
        RootError::Deserialization("TBranch missing byte count".into())
    })?;

    // TNamed
    let (name, _title) = r.read_tnamed()?;

    // TAttFill — skip
    skip_versioned(r)?;

    // Branch fields
    let _compress = r.read_i32()?;      // fCompress
    let _basket_size = r.read_i32()?;    // fBasketSize
    let _entry_offset_len = r.read_i32()?; // fEntryOffsetLen
    let write_basket = r.read_i32()?;    // fWriteBasket (= number of valid baskets)
    let _entry_number = r.read_i64()?;   // fEntryNumber

    // IOBits (v13+)
    if branch_ver >= 13 {
        // fIOBits — just a u32 in practice
        let _io_bits = r.read_u32()?;
    }

    let _offset = r.read_i32()?;        // fOffset
    let max_baskets = r.read_i32()?;     // fMaxBaskets
    let _split_level = r.read_i32()?;    // fSplitLevel
    let entries = r.read_i64()? as u64;  // fEntries

    if branch_ver >= 11 {
        let _first_entry = r.read_i64()?; // fFirstEntry
    }

    let _tot_bytes = r.read_i64()?;      // fTotBytes
    let _zip_bytes = r.read_i64()?;      // fZipBytes

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
        let class = tracker.read_class_tag(r)?;

        match class.as_deref() {
            None => {}
            Some(cls) => {
                let lt = leaf_type_from_class(cls);

                // Read TLeaf to skip it properly
                let (_leaf_ver, leaf_end) = r.read_version()?;
                if let Some(end) = leaf_end {
                    // We only need the class name for the type
                    r.set_pos(end);
                } else {
                    // No byte count — can't skip reliably
                    // Just set leaf type and break to array end
                    if leaf_type.is_none() {
                        leaf_type = lt;
                    }
                    r.set_pos(arr_end);
                    return Ok(leaf_type);
                }

                if leaf_type.is_none() {
                    leaf_type = lt;
                }
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
