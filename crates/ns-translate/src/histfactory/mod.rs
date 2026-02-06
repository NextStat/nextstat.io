//! HistFactory XML+ROOT → Workspace converter.
//!
//! Parses CERN HistFactory `combination.xml` + `channel_*.xml` files,
//! reads histograms from ROOT files, and produces the same `Workspace`
//! struct as the pyhf JSON path.

mod combination;
mod channel;
mod builder;

pub use builder::from_xml;
