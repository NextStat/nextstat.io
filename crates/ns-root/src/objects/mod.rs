//! ROOT object deserialization dispatch.

mod th1;
pub(crate) mod ttree;

use crate::error::{Result, RootError};
use crate::histogram::Histogram;
use crate::tree::Tree;

/// Read a histogram from a decompressed object payload, given its class name.
pub fn read_histogram(payload: &[u8], class_name: &str) -> Result<Histogram> {
    match class_name {
        "TH1D" => th1::read_th1d(payload),
        "TH1F" => th1::read_th1f(payload),
        _ => Err(RootError::UnsupportedClass(class_name.to_string())),
    }
}

/// Read a TTree from a decompressed TKey payload.
pub fn read_tree(payload: &[u8]) -> Result<Tree> {
    ttree::read_ttree(payload)
}
