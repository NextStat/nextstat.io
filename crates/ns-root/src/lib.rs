//! # ns-root
//!
//! Native ROOT file reader for NextStat.
//!
//! Reads TH1D/TH1F histograms and TTrees from `.root` files without requiring
//! Python or external ROOT libraries. Supports zlib and LZ4 compression.
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
pub mod datasource;
pub mod decompress;
pub mod directory;
pub mod error;
pub mod expr;
pub mod file;
pub mod filler;
pub mod histogram;
pub mod key;
pub mod objects;
pub mod rbuffer;
pub mod tree;

pub use branch_reader::BranchReader;
pub use error::{Result, RootError};
pub use expr::CompiledExpr;
pub use file::RootFile;
pub use filler::{
    FilledHistogram, FlowPolicy, HistogramSpec, NegativeWeightPolicy, fill_histograms,
};
pub use histogram::{Histogram, HistogramWithFlows};
pub use key::KeyInfo;
pub use tree::{BranchInfo, LeafType, Tree};
