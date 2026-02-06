//! ROOT object deserialization dispatch.

mod th1;

use crate::error::{Result, RootError};
use crate::histogram::Histogram;

/// Read a histogram from a decompressed object payload, given its class name.
pub fn read_histogram(payload: &[u8], class_name: &str) -> Result<Histogram> {
    match class_name {
        "TH1D" => th1::read_th1d(payload),
        "TH1F" => th1::read_th1f(payload),
        _ => Err(RootError::UnsupportedClass(class_name.to_string())),
    }
}
