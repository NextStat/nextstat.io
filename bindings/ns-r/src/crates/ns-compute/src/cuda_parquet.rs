//! Direct Parquet → CUDA GPU upload utilities.
//!
//! Provides zero-extra-copy upload of pre-decoded event data (SoA `f64` layout)
//! to CUDA device memory. Designed to work with `ns_translate::arrow::parquet::ParquetEventData`.
//!
//! # Data flow
//!
//! ```text
//! Parquet file
//!   → mmap (memmap2)
//!   → predicate pushdown (row group pruning)
//!   → parallel decode (rayon)
//!   → Arrow Float64Array → SoA f64 buffer
//!   → [THIS MODULE] clone_htod → CudaSlice<f64>
//! ```
//!
//! The SoA buffer from `read_parquet_events_soa` is passed directly to
//! `CudaStream::clone_htod`, avoiding any intermediate `Vec` allocation.

use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

fn cuda_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("CUDA (parquet upload): {msg}"))
}

/// Event data uploaded to CUDA device memory in SoA layout.
///
/// Mirrors `ParquetEventData` but with GPU-resident buffers.
pub struct CudaEventData {
    /// Device buffer `[n_obs × n_events]` in SoA layout.
    pub d_obs_soa: CudaSlice<f64>,
    /// Optional per-event weights on device (length `n_events`).
    pub d_weights: Option<CudaSlice<f64>>,
    /// Number of events.
    pub n_events: usize,
    /// Number of observables.
    pub n_obs: usize,
}

/// Upload pre-decoded SoA event data to CUDA device memory.
///
/// `soa` must have length `n_obs * n_events` in SoA layout:
/// `soa[obs_idx * n_events + event_idx]`.
///
/// `weights` is an optional per-event weight vector of length `n_events`.
///
/// This is the final stage of the Parquet → GPU pipeline:
/// ```text
/// let data = ns_translate::arrow::parquet::read_parquet_events_soa(...)?;
/// let gpu  = ns_compute::cuda_parquet::upload_events_to_cuda(&stream, &data.soa, data.weights.as_deref(), data.n_events, data.n_obs)?;
/// ```
pub fn upload_events_to_cuda(
    stream: &Arc<CudaStream>,
    soa: &[f64],
    weights: Option<&[f64]>,
    n_events: usize,
    n_obs: usize,
) -> ns_core::Result<CudaEventData> {
    if soa.len() != n_obs * n_events {
        return Err(ns_core::Error::Validation(format!(
            "SoA length mismatch: expected {} ({}×{}), got {}",
            n_obs * n_events,
            n_obs,
            n_events,
            soa.len()
        )));
    }

    if let Some(w) = weights {
        if w.len() != n_events {
            return Err(ns_core::Error::Validation(format!(
                "weights length mismatch: expected {}, got {}",
                n_events,
                w.len()
            )));
        }
    }

    let d_obs_soa = stream.clone_htod(soa).map_err(cuda_err)?;

    let d_weights = match weights {
        Some(w) => Some(stream.clone_htod(w).map_err(cuda_err)?),
        None => None,
    };

    Ok(CudaEventData { d_obs_soa, d_weights, n_events, n_obs })
}

/// Upload pre-decoded SoA event data to CUDA with observable bounds.
///
/// This is the full-featured variant that also uploads per-observable
/// lower/upper bounds (needed by the unbinned NLL kernel).
pub fn upload_events_with_bounds_to_cuda(
    stream: &Arc<CudaStream>,
    soa: &[f64],
    weights: Option<&[f64]>,
    obs_bounds: &[(f64, f64)],
    n_events: usize,
    n_obs: usize,
) -> ns_core::Result<CudaEventDataWithBounds> {
    if obs_bounds.len() != n_obs {
        return Err(ns_core::Error::Validation(format!(
            "obs_bounds length mismatch: expected {}, got {}",
            n_obs,
            obs_bounds.len()
        )));
    }

    let base = upload_events_to_cuda(stream, soa, weights, n_events, n_obs)?;

    let obs_lo: Vec<f64> = obs_bounds.iter().map(|&(lo, _)| lo).collect();
    let obs_hi: Vec<f64> = obs_bounds.iter().map(|&(_, hi)| hi).collect();

    let d_obs_lo = stream.clone_htod(&obs_lo).map_err(cuda_err)?;
    let d_obs_hi = stream.clone_htod(&obs_hi).map_err(cuda_err)?;

    Ok(CudaEventDataWithBounds {
        d_obs_soa: base.d_obs_soa,
        d_weights: base.d_weights,
        d_obs_lo,
        d_obs_hi,
        n_events,
        n_obs,
    })
}

/// Event data with observable bounds on CUDA device.
pub struct CudaEventDataWithBounds {
    /// Device buffer `[n_obs × n_events]` in SoA layout.
    pub d_obs_soa: CudaSlice<f64>,
    /// Optional per-event weights on device.
    pub d_weights: Option<CudaSlice<f64>>,
    /// Per-observable lower bounds on device.
    pub d_obs_lo: CudaSlice<f64>,
    /// Per-observable upper bounds on device.
    pub d_obs_hi: CudaSlice<f64>,
    /// Number of events.
    pub n_events: usize,
    /// Number of observables.
    pub n_obs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;

    fn get_stream() -> Option<Arc<CudaStream>> {
        let ctx = std::panic::catch_unwind(|| CudaContext::new(0)).ok()?.ok()?;
        Some(ctx.default_stream())
    }

    #[test]
    fn test_upload_soa_roundtrip() {
        let stream = match get_stream() {
            Some(s) => s,
            None => {
                eprintln!("No CUDA device — skipping");
                return;
            }
        };

        // 3 events, 2 observables: mass=[100,110,120], pt=[20,40,60]
        let soa = vec![100.0, 110.0, 120.0, 20.0, 40.0, 60.0];
        let result = upload_events_to_cuda(&stream, &soa, None, 3, 2).unwrap();

        assert_eq!(result.n_events, 3);
        assert_eq!(result.n_obs, 2);
        assert!(result.d_weights.is_none());

        // Read back from device and verify.
        let mut readback = vec![0.0f64; 6];
        stream.memcpy_dtoh(&result.d_obs_soa, &mut readback).unwrap();
        for (i, (&orig, &got)) in soa.iter().zip(readback.iter()).enumerate() {
            assert!((orig - got).abs() < 1e-12, "mismatch at {i}: expected {orig}, got {got}");
        }
    }

    #[test]
    fn test_upload_with_weights() {
        let stream = match get_stream() {
            Some(s) => s,
            None => return,
        };

        let soa = vec![1.0, 2.0, 3.0];
        let weights = vec![0.5, 1.0, 1.5];
        let result = upload_events_to_cuda(&stream, &soa, Some(&weights), 3, 1).unwrap();

        assert!(result.d_weights.is_some());
        let mut w_back = vec![0.0f64; 3];
        stream.memcpy_dtoh(result.d_weights.as_ref().unwrap(), &mut w_back).unwrap();
        for (i, (&orig, &got)) in weights.iter().zip(w_back.iter()).enumerate() {
            assert!(
                (orig - got).abs() < 1e-12,
                "weight mismatch at {i}: expected {orig}, got {got}"
            );
        }
    }

    #[test]
    fn test_upload_with_bounds() {
        let stream = match get_stream() {
            Some(s) => s,
            None => return,
        };

        let soa = vec![100.0, 110.0, 20.0, 40.0];
        let bounds = vec![(50.0, 200.0), (0.0, 100.0)];
        let result = upload_events_with_bounds_to_cuda(&stream, &soa, None, &bounds, 2, 2).unwrap();

        assert_eq!(result.n_events, 2);
        assert_eq!(result.n_obs, 2);

        let mut lo = vec![0.0f64; 2];
        let mut hi = vec![0.0f64; 2];
        stream.memcpy_dtoh(&result.d_obs_lo, &mut lo).unwrap();
        stream.memcpy_dtoh(&result.d_obs_hi, &mut hi).unwrap();
        assert!((lo[0] - 50.0).abs() < 1e-12);
        assert!((lo[1] - 0.0).abs() < 1e-12);
        assert!((hi[0] - 200.0).abs() < 1e-12);
        assert!((hi[1] - 100.0).abs() < 1e-12);
    }

    #[test]
    fn test_validation_soa_length_mismatch() {
        let stream = match get_stream() {
            Some(s) => s,
            None => return,
        };

        let soa = vec![1.0, 2.0, 3.0]; // 3 elements but claim 2×2=4
        let result = upload_events_to_cuda(&stream, &soa, None, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_weights_length_mismatch() {
        let stream = match get_stream() {
            Some(s) => s,
            None => return,
        };

        let soa = vec![1.0, 2.0];
        let weights = vec![0.5]; // 1 weight but 2 events
        let result = upload_events_to_cuda(&stream, &soa, Some(&weights), 2, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_bounds_length_mismatch() {
        let stream = match get_stream() {
            Some(s) => s,
            None => return,
        };

        let soa = vec![1.0, 2.0];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)]; // 2 bounds but n_obs=1
        let result = upload_events_with_bounds_to_cuda(&stream, &soa, None, &bounds, 2, 1);
        assert!(result.is_err());
    }
}
