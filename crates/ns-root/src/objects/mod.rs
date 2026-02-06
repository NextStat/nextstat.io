//! ROOT object deserialization dispatch.

mod th1;
pub(crate) mod ttree;

use crate::error::{Result, RootError};
use crate::histogram::{Histogram, HistogramWithFlows};
use crate::tree::Tree;

/// Read a histogram from a decompressed object payload, given its class name.
pub fn read_histogram(payload: &[u8], class_name: &str) -> Result<Histogram> {
    match class_name {
        "TH1D" => th1::read_th1d(payload),
        "TH1F" => th1::read_th1f(payload),
        _ => Err(RootError::UnsupportedClass(class_name.to_string())),
    }
}

/// Read a histogram from a decompressed object payload, preserving under/overflow bins.
pub fn read_histogram_with_flows(payload: &[u8], class_name: &str) -> Result<HistogramWithFlows> {
    match class_name {
        "TH1D" => th1::read_th1d_with_flows(payload),
        "TH1F" => th1::read_th1f_with_flows(payload),
        _ => Err(RootError::UnsupportedClass(class_name.to_string())),
    }
}

/// Read a TTree from a decompressed TKey payload.
pub fn read_tree(payload: &[u8]) -> Result<Tree> {
    ttree::read_ttree(payload)
}
