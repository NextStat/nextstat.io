//! HistFactory XML+ROOT → Workspace converter.
//!
//! Parses CERN HistFactory `combination.xml` + `channel_*.xml` files,
//! reads histograms from ROOT files, and produces the same `Workspace`
//! struct as the pyhf JSON path.

mod builder;
mod channel;
mod combination;

pub use builder::bin_edges_by_channel_from_xml;
pub use builder::from_xml;
