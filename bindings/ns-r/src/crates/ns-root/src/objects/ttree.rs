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
    /// Best-effort intervals that cover where a class tag/name was observed.
    ///
    /// Some ROOT writers appear to reference offsets that fall within the element's
    /// byte-count wrapper but don't match the exact positions we register in `classes`.
    /// When an exact lookup fails, we fall back to resolving by interval containment.
    intervals: Vec<(usize, usize, String)>,
}

impl ClassRefTracker {
    fn new() -> Self {
        Self { classes: Vec::new(), intervals: Vec::new() }
    }

    /// Look up a class by its byte offset tag.
    fn lookup(&self, offset: usize) -> Option<&str> {
        self.classes.iter().find(|(off, _)| *off == offset).map(|(_, name)| name.as_str())
    }

    fn lookup_by_interval(&self, offset: usize) -> Option<&str> {
        // Choose the smallest interval that contains the offset (most specific).
        let mut best: Option<(usize, &str)> = None;
        for (start, end, name) in &self.intervals {
            if offset < *start || offset > *end {
                continue;
            }
            let len = end.saturating_sub(*start);
            match best {
                None => best = Some((len, name.as_str())),
                Some((best_len, _)) if len < best_len => best = Some((len, name.as_str())),
                _ => {}
            }
        }
        best.map(|(_, n)| n)
    }

    fn unique_class_name(&self) -> Option<&str> {
        let mut it = self.classes.iter();
        let (_, first) = it.next()?;
        if it.all(|(_, name)| name == first) { Some(first.as_str()) } else { None }
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
                // Register this class for later back-references.
                // Different ROOT writers may use different base offsets for the reference tag.
                // We register a few plausible offsets to maximize compatibility.
                // - class_tag_pos: offset of the u32 class tag
                // - obj_start: offset of the enclosing byte-count tag
                // - name_start: offset where the class name bytes start
                self.classes.push((class_tag_pos, name.clone()));
                self.classes.push((obj_start, name.clone()));
                self.classes.push((name_start, name.clone()));
                // Some ROOT versions appear to reference near the end of the byte-count wrapper.
                // These two offsets are added defensively.
                self.classes.push((obj_end.saturating_sub(4), name.clone()));
                self.classes.push((obj_end, name.clone()));
                // Interval fallback: cover the full element wrapper for this class.
                self.intervals.push((obj_start, obj_end, name.clone()));
                name
            } else if class_tag & K_CLASS_MASK != 0 {
                // Reference to existing class by offset
                let ref_offset = (class_tag & !K_CLASS_MASK) as usize;
                match self.lookup(ref_offset) {
                    Some(name) => name.to_string(),
                    None => {
                        if let Some(name) = self.lookup_by_interval(ref_offset) {
                            let name = name.to_string();
                            self.classes.push((ref_offset, name.clone()));
                            name
                        } else if let Some(unique) = self.unique_class_name().map(|s| s.to_string())
                        {
                            self.classes.push((ref_offset, unique.clone()));
                            unique
                        } else {
                            return Err(RootError::Deserialization(format!(
                                "class ref offset {} not found (tag={:#010x}, known: {:?})",
                                ref_offset, class_tag, self.classes
                            )));
                        }
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
    let debug = std::env::var("NS_ROOT_TTREE_DEBUG").is_ok();

    for idx in 0..count {
        let elem_pos = r.pos();
        let element = tracker.read_element(r)?;

        match element {
            None => {
                // Null entry — skip
                if debug {
                    eprintln!("[ttree] branch elem[{idx}] NULL at pos={elem_pos}");
                }
            }
            Some((ref class_name, obj_end)) if class_name == "TBranch" => match read_tbranch(r) {
                Ok(branch) => branches.push(branch),
                Err(e) => {
                    if debug {
                        eprintln!(
                            "[ttree] branch elem[{idx}] TBranch parse failed at pos={} end={}: {e}",
                            elem_pos, obj_end
                        );
                    }
                    r.set_pos(obj_end);
                }
            },
            Some((ref class_name, obj_end)) if class_name == "TBranchElement" => {
                match read_tbranch_element(r, obj_end) {
                    Ok(subs) => branches.extend(subs),
                    Err(e) => {
                        if debug {
                            eprintln!(
                                "[ttree] branch elem[{idx}] TBranchElement parse failed at pos={} end={}: {e}",
                                elem_pos, obj_end
                            );
                        }
                        r.set_pos(obj_end);
                    }
                }
            }
            Some((class_name, obj_end)) => {
                // Unknown branch type — try to read as TBranch
                log::debug!("skipping unknown branch class: {}", class_name);
                match read_tbranch(r) {
                    Ok(branch) => branches.push(branch),
                    Err(e) => {
                        if debug {
                            eprintln!(
                                "[ttree] branch elem[{idx}] {} parse failed at pos={} end={}: {e}",
                                class_name, elem_pos, obj_end
                            );
                        }
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

/// Intermediate representation of TBranch base-class fields.
/// Used by both `read_tbranch()` and `read_tbranch_element()`.
struct TBranchFields {
    name: String,
    leaf_type: Option<LeafType>,
    entry_offset_len: i32,
    entries: u64,
    basket_bytes: Vec<u32>,
    basket_entry: Vec<u64>,
    basket_seek: Vec<u64>,
    n_baskets: usize,
    sub_branches: Vec<BranchInfo>,
    /// Absolute end position of this TBranch object in the buffer.
    branch_end: usize,
}

impl TBranchFields {
    fn into_branch_info(self) -> BranchInfo {
        BranchInfo {
            name: self.name,
            leaf_type: self.leaf_type.unwrap_or(LeafType::F64),
            entry_offset_len: self.entry_offset_len.max(0) as usize,
            entries: self.entries,
            basket_bytes: self.basket_bytes,
            basket_entry: self.basket_entry,
            basket_seek: self.basket_seek,
            n_baskets: self.n_baskets,
        }
    }
}

/// Read TBranch base-class fields from the buffer.
///
/// If `recurse_sub` is true, sub-branches (fBranches TObjArray) are recursively
/// parsed; otherwise they are skipped. This controls whether we collect
/// sub-branches for TBranchElement (split vector\<T\>) or skip them for plain TBranch.
fn read_tbranch_base(r: &mut RBuffer, recurse_sub: bool) -> Result<TBranchFields> {
    let (branch_ver, branch_end) = r.read_version()?;
    let branch_end = branch_end
        .ok_or_else(|| RootError::Deserialization("TBranch missing byte count".into()))?;

    // TNamed
    let (name, title) = r.read_tnamed()?;

    // TAttFill — skip
    skip_versioned(r)?;

    // Branch fields
    let _compress = r.read_i32()?; // fCompress
    let _basket_size = r.read_i32()?; // fBasketSize
    let entry_offset_len = r.read_i32()?; // fEntryOffsetLen
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
    let sub_branches = if recurse_sub {
        read_tobjarray_branches(r)?
    } else {
        skip_tobjarray(r)?;
        Vec::new()
    };

    // fLeaves: TObjArray of TLeaf
    let leaf_type = read_tobjarray_leaves(r)?.or_else(|| leaf_type_from_branch_title(&title));

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

    Ok(TBranchFields {
        name,
        leaf_type,
        entry_offset_len,
        entries,
        basket_bytes,
        basket_entry,
        basket_seek,
        n_baskets,
        sub_branches,
        branch_end,
    })
}

/// Read a single TBranch from the buffer.
fn read_tbranch(r: &mut RBuffer) -> Result<BranchInfo> {
    let fields = read_tbranch_base(r, false)?;
    // Skip any trailing bytes we didn't parse (e.g. fFileName in some versions)
    if fields.branch_end > r.pos() {
        r.set_pos(fields.branch_end);
    }
    Ok(fields.into_branch_info())
}

/// Read a TBranchElement from the buffer.
///
/// TBranchElement extends TBranch with additional fields (fClassName, fParentName,
/// fClonesName, fCheckSum, fClassVersion, fID, fType, fStreamerType, fMaximum,
/// fBranchCount, fBranchCount2). For vector\<T\> branches, the data is stored in
/// sub-branches when the split level > 0.
///
/// Returns a Vec\<BranchInfo\> — either the sub-branches (split case) or a single
/// branch info (unsplit / leaf case).
fn read_tbranch_element(r: &mut RBuffer, element_end: usize) -> Result<Vec<BranchInfo>> {
    // TBranchElement version header
    let (_elem_ver, _elem_end) = r.read_version()?;

    // TBranch base fields with sub-branch recursion
    let mut fields = read_tbranch_base(r, true)?;
    // Ensure we're positioned at the end of the TBranch base-class portion before
    // reading TBranchElement-specific fields.
    if fields.branch_end > r.pos() {
        r.set_pos(fields.branch_end);
    }

    // Best-effort: infer leaf type for unsplit `std::vector<T>` (TBranchElement)
    // branches. These often use `TLeafElement` (which we don't fully parse yet),
    // so `leaf_type` may be missing unless we read `fClassName` here.
    if fields.leaf_type.is_none()
        && r.pos() < element_end
        && let Ok(class_name) = r.read_string()
    {
        fields.leaf_type = leaf_type_from_vector_class_name(&class_name);
    }

    // Skip TBranchElement-specific fields to element_end
    r.set_pos(element_end);

    // If has sub-branches → return them (split vector<T>)
    // Otherwise → return self as leaf
    if !fields.sub_branches.is_empty() {
        Ok(fields.sub_branches)
    } else {
        Ok(vec![fields.into_branch_info()])
    }
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
    let debug = std::env::var("NS_ROOT_TTREE_DEBUG").is_ok();

    for idx in 0..count {
        let element = match tracker.read_element(r) {
            Ok(e) => e,
            Err(e) => {
                // Some ROOT writers use the global reference system for class tags, which can
                // point outside this TObjArray. Since leaf types are also encoded in the branch
                // title (leaflist), treat leaf parsing as best-effort: skip the array and let
                // callers fall back to the title.
                if debug {
                    eprintln!("[ttree] leaf elem[{idx}] parse failed at pos={}: {e}", r.pos());
                }
                r.set_pos(arr_end);
                return Ok(leaf_type);
            }
        };

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

fn leaf_type_from_branch_title(title: &str) -> Option<LeafType> {
    let t = title.trim();

    // Leaflist syntax: `name/F`, `name/I`, `name/D`, `name[10]/F`, etc.
    // ROOT encodes the leaf type as a single character after the last '/'.
    if let Some((_, ty)) = t.rsplit_once('/') {
        let code = ty.trim().as_bytes().first().copied()? as char;
        return match code {
            'F' => Some(LeafType::F32),
            'D' => Some(LeafType::F64),
            'I' => Some(LeafType::I32),
            'i' => Some(LeafType::U32),
            'L' => Some(LeafType::I64),
            'l' => Some(LeafType::U64),
            'S' => Some(LeafType::I16),
            'B' => Some(LeafType::I8),
            'O' => Some(LeafType::Bool),
            _ => None,
        };
    }

    let t = t.strip_prefix("std::").unwrap_or(t);
    let inner = t.strip_prefix("vector<")?.strip_suffix('>')?.trim();
    match inner {
        "float" | "Float_t" => Some(LeafType::F32),
        "double" | "Double_t" => Some(LeafType::F64),
        "int" | "Int_t" => Some(LeafType::I32),
        "unsigned int" | "UInt_t" => Some(LeafType::U32),
        "long" | "Long64_t" => Some(LeafType::I64),
        "unsigned long" | "ULong64_t" => Some(LeafType::U64),
        "short" | "Short_t" => Some(LeafType::I16),
        "char" | "Char_t" => Some(LeafType::I8),
        "bool" | "Bool_t" => Some(LeafType::Bool),
        _ => None,
    }
}

fn leaf_type_from_vector_class_name(class_name: &str) -> Option<LeafType> {
    let t = class_name.trim();
    let t = t.strip_prefix("std::").unwrap_or(t);
    let inner = t.strip_prefix("vector<")?.strip_suffix('>')?.trim();
    match inner {
        "float" | "Float_t" => Some(LeafType::F32),
        "double" | "Double_t" => Some(LeafType::F64),
        "int" | "Int_t" => Some(LeafType::I32),
        "unsigned int" | "UInt_t" => Some(LeafType::U32),
        "long" | "Long64_t" => Some(LeafType::I64),
        "unsigned long" | "ULong64_t" => Some(LeafType::U64),
        "short" | "Short_t" => Some(LeafType::I16),
        "char" | "Char_t" => Some(LeafType::I8),
        "bool" | "Bool_t" => Some(LeafType::Bool),
        _ => None,
    }
}
