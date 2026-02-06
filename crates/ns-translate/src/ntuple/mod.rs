//! Ntuple (TTree) → Workspace pipeline.
//!
//! High-level API for building a pyhf [`Workspace`] directly from ROOT ntuple
//! files, eliminating the need to pre-generate histograms.

mod config;
mod processor;

pub use config::{ChannelConfig, NtupleModifier, SampleConfig};
pub use processor::NtupleWorkspaceBuilder;
