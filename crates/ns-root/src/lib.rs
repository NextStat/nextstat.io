//! # ns-root
//!
//! Native ROOT file reader for NextStat.
//!
//! Reads TH1D and TH1F histograms from `.root` files without requiring
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
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod error;
pub mod rbuffer;
pub mod file;
pub mod key;
pub mod decompress;
pub mod directory;
pub mod objects;
pub mod histogram;

pub use error::{RootError, Result};
pub use file::RootFile;
pub use histogram::Histogram;
pub use key::KeyInfo;
