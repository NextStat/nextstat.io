//! Public types for TTree branch metadata.

/// Leaf data type (maps to ROOT TLeaf class names).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafType {
    /// `TLeafF` — 32-bit float.
    F32,
    /// `TLeafD` — 64-bit float.
    F64,
    /// `TLeafI` — 32-bit signed integer.
    I32,
    /// `TLeafL` — 64-bit signed integer.
    I64,
    /// `TLeafI` unsigned variant.
    U32,
    /// `TLeafL` unsigned variant.
    U64,
    /// `TLeafS` — 16-bit signed integer.
    I16,
    /// `TLeafB` — 8-bit signed integer.
    I8,
    /// `TLeafO` — boolean (1 byte).
    Bool,
}

impl LeafType {
    /// Size in bytes of one element.
    pub fn byte_size(self) -> usize {
        match self {
            LeafType::F32 | LeafType::I32 | LeafType::U32 => 4,
            LeafType::F64 | LeafType::I64 | LeafType::U64 => 8,
            LeafType::I16 => 2,
            LeafType::I8 | LeafType::Bool => 1,
        }
    }
}

/// Metadata for a single TBranch in a TTree.
#[derive(Debug, Clone)]
pub struct BranchInfo {
    /// Branch name.
    pub name: String,
    /// Data type of leaves.
    pub leaf_type: LeafType,
    /// Total number of entries in this branch.
    pub entries: u64,
    /// Compressed byte sizes for each basket.
    pub basket_bytes: Vec<u32>,
    /// Entry boundaries for each basket.
    pub basket_entry: Vec<u64>,
    /// Absolute file offsets (seek positions) for each basket.
    pub basket_seek: Vec<u64>,
    /// Number of valid baskets (`fWriteBasket`).
    pub n_baskets: usize,
}

/// A parsed TTree with branch metadata.
#[derive(Debug, Clone)]
pub struct Tree {
    /// Tree name.
    pub name: String,
    /// Total number of entries.
    pub entries: u64,
    /// Flat list of all branches (including sub-branches).
    pub branches: Vec<BranchInfo>,
}

impl Tree {
    /// Find a branch by name.
    pub fn find_branch(&self, name: &str) -> Option<&BranchInfo> {
        self.branches.iter().find(|b| b.name == name)
    }

    /// List all branch names.
    pub fn branch_names(&self) -> Vec<&str> {
        self.branches.iter().map(|b| b.name.as_str()).collect()
    }
}
