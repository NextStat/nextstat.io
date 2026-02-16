//! # ns-root
//!
//! Native ROOT file reader for NextStat.
//!
//! Reads TH1D/TH1F histograms and TTrees from `.root` files without requiring
//! Python or external ROOT libraries. Supports zlib, LZ4, ZSTD, and XZ compression.
//!
//! ## Example
//!
//! ```no_run
//! use ns_root::RootFile;
//!
//! let f = RootFile::open("data.root").unwrap();
//! for key in f.list_keys().unwrap() {
//!     println!("{} ({})", key.name, key.class_name);
//! }
//! let h = f.get_histogram("signal").unwrap();
//! println!("bins: {}, entries: {}", h.n_bins, h.entries);
//!
//! // TTree access
//! let tree = f.get_tree("events").unwrap();
//! let pt: Vec<f64> = f.branch_data(&tree, "pt").unwrap();
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod basket;
pub mod branch_reader;
pub mod cache;
pub mod chained_slice;
pub mod datasource;
pub mod decompress;
pub mod directory;
pub mod error;
pub mod expr;
pub mod file;
pub mod filler;
pub mod histogram;
pub mod key;
pub mod lazy_branch_reader;
pub mod objects;
pub mod rbuffer;
pub mod rntuple;
pub mod tree;

pub use branch_reader::{BranchReader, JaggedCol};
pub use cache::{BasketCache, CacheConfig, CacheStats};
pub use chained_slice::ChainedSlice;
pub use error::{Result, RootError};
pub use expr::{CompiledExpr, DEFAULT_CHUNK_SIZE};
pub use file::{
    RNTupleClusterDecodedColumnsF64, RNTupleDecodedColumnsF64, RNTupleEnvelopeBytes,
    RNTupleFixedArrayColumnF64, RNTupleInfo, RNTuplePageBlobBytes, RNTuplePageListEnvelopeBytes,
    RNTuplePairColumnF64, RNTuplePairScalarVariableColumnF64, RNTuplePairVariableScalarColumnF64,
    RNTuplePairVariableVariableColumnF64, RNTuplePrimitiveColumnF64, RNTupleVariableArrayColumnF64,
    RootFile,
};
pub use filler::{
    FilledHistogram, FlowPolicy, HistogramSpec, NegativeWeightPolicy, fill_histograms,
    fill_histograms_with_jagged,
};
pub use histogram::{Histogram, HistogramWithFlows};
pub use key::KeyInfo;
pub use lazy_branch_reader::LazyBranchReader;
pub use rntuple::{
    RNTUPLE_ENVELOPE_TYPE_FOOTER, RNTUPLE_ENVELOPE_TYPE_HEADER, RNTUPLE_ENVELOPE_TYPE_PAGELIST,
    RNTupleAnchor, RNTupleClusterGroupSummary, RNTupleEnvelopeInfo, RNTupleFieldKind,
    RNTupleFieldToken, RNTupleFooterSummary, RNTupleHeaderSummary, RNTupleLocatorSummary,
    RNTupleMetadataSummary, RNTuplePageListSummary, RNTuplePageSummary, RNTupleScalarType,
    RNTupleSchemaField, RNTupleSchemaSummary, parse_rntuple_anchor_payload, parse_rntuple_envelope,
    parse_rntuple_footer_summary, parse_rntuple_header_summary, parse_rntuple_pagelist_summary,
    parse_rntuple_schema_summary,
};
pub use tree::{BranchInfo, LeafType, Tree};
