//! Build-time SHA-256 checksums for CUDA kernel sources.
//!
//! Used for kernel traceability: benchmark artifacts, rsync verification,
//! and CI/wheel provenance. Empty map when compiled without `cuda` feature.

use std::collections::BTreeMap;

/// Return SHA-256 hex digests for CUDA kernel source files.
///
/// Keys are file names (e.g. `"mams_engine.cuh"`), values are lowercase hex SHA-256.
/// Returns an empty map when compiled without the `cuda` feature.
pub fn kernel_checksums() -> BTreeMap<&'static str, &'static str> {
    #[allow(unused_mut)]
    let mut m = BTreeMap::new();
    #[cfg(feature = "cuda")]
    {
        m.insert("mams_engine.cuh", env!("CUDA_MAMS_ENGINE_SHA256"));
        m.insert("mams_leapfrog.cu", env!("CUDA_MAMS_LEAPFROG_SHA256"));
    }
    m
}
