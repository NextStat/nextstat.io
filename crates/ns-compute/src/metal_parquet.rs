//! Direct Parquet → Metal GPU upload utilities.
//!
//! Provides zero-extra-copy upload of pre-decoded event data (SoA layout)
//! to Metal device memory. Designed to work with `ns_translate::arrow::parquet::ParquetEventData`.
//!
//! Metal kernels operate in **f32** precision. This module handles the f64→f32 conversion
//! during upload, so the caller works exclusively with f64 (matching Arrow/Parquet).
//!
//! # Data flow
//!
//! ```text
//! Parquet file
//!   → mmap (memmap2)
//!   → predicate pushdown (row group pruning)
//!   → parallel decode (rayon)
//!   → Arrow Float64Array → SoA f64 buffer
//!   → [THIS MODULE] f64→f32 + new_buffer_with_data → MTLBuffer
//! ```

use metal::{Buffer, Device, MTLResourceOptions};
use std::mem;

/// Event data uploaded to Metal device memory in SoA layout (f32 precision).
pub struct MetalEventData {
    /// Device buffer `[n_obs × n_events]` in SoA layout (f32).
    pub buf_obs_soa: Buffer,
    /// Optional per-event weights on device (f32, length `n_events`).
    pub buf_weights: Option<Buffer>,
    /// Number of events.
    pub n_events: usize,
    /// Number of observables.
    pub n_obs: usize,
}

/// Event data with observable bounds on Metal device (f32 precision).
pub struct MetalEventDataWithBounds {
    /// Device buffer `[n_obs × n_events]` in SoA layout (f32).
    pub buf_obs_soa: Buffer,
    /// Optional per-event weights on device (f32).
    pub buf_weights: Option<Buffer>,
    /// Per-observable lower bounds on device (f32).
    pub buf_obs_lo: Buffer,
    /// Per-observable upper bounds on device (f32).
    pub buf_obs_hi: Buffer,
    /// Number of events.
    pub n_events: usize,
    /// Number of observables.
    pub n_obs: usize,
}

/// Upload pre-decoded SoA event data to Metal device memory.
///
/// Converts f64 → f32 during upload (Metal kernels use f32 precision).
///
/// `soa` must have length `n_obs * n_events` in SoA layout:
/// `soa[obs_idx * n_events + event_idx]`.
///
/// Usage:
/// ```text
/// let data = ns_translate::arrow::parquet::read_parquet_events_soa(...)?;
/// let gpu  = ns_compute::metal_parquet::upload_events_to_metal(&device, &data.soa, data.weights.as_deref(), data.n_events, data.n_obs)?;
/// ```
pub fn upload_events_to_metal(
    device: &Device,
    soa: &[f64],
    weights: Option<&[f64]>,
    n_events: usize,
    n_obs: usize,
) -> ns_core::Result<MetalEventData> {
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

    let opts = MTLResourceOptions::StorageModeShared;

    let soa_f32: Vec<f32> = soa.iter().map(|&v| v as f32).collect();
    let buf_obs_soa = create_buffer_f32(device, &soa_f32, opts);

    let buf_weights = weights.map(|w| {
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        create_buffer_f32(device, &w_f32, opts)
    });

    Ok(MetalEventData { buf_obs_soa, buf_weights, n_events, n_obs })
}

/// Upload pre-decoded SoA event data with observable bounds to Metal device memory.
///
/// Full-featured variant that also uploads per-observable bounds (needed by unbinned NLL kernel).
pub fn upload_events_with_bounds_to_metal(
    device: &Device,
    soa: &[f64],
    weights: Option<&[f64]>,
    obs_bounds: &[(f64, f64)],
    n_events: usize,
    n_obs: usize,
) -> ns_core::Result<MetalEventDataWithBounds> {
    if obs_bounds.len() != n_obs {
        return Err(ns_core::Error::Validation(format!(
            "obs_bounds length mismatch: expected {}, got {}",
            n_obs,
            obs_bounds.len()
        )));
    }

    let base = upload_events_to_metal(device, soa, weights, n_events, n_obs)?;

    let opts = MTLResourceOptions::StorageModeShared;

    let lo_f32: Vec<f32> = obs_bounds.iter().map(|&(lo, _)| lo as f32).collect();
    let hi_f32: Vec<f32> = obs_bounds.iter().map(|&(_, hi)| hi as f32).collect();

    let buf_obs_lo = create_buffer_f32(device, &lo_f32, opts);
    let buf_obs_hi = create_buffer_f32(device, &hi_f32, opts);

    Ok(MetalEventDataWithBounds {
        buf_obs_soa: base.buf_obs_soa,
        buf_weights: base.buf_weights,
        buf_obs_lo,
        buf_obs_hi,
        n_events,
        n_obs,
    })
}

/// Read a Metal f32 buffer back to f64 (for verification/debug).
pub fn read_metal_soa_to_f64(buffer: &Buffer, count: usize) -> Vec<f64> {
    let ptr = buffer.contents() as *const f32;
    // SAFETY: `buffer` was allocated with capacity >= `count * size_of::<f32>()`.
    let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
    slice.iter().map(|&v| v as f64).collect()
}

fn create_buffer_f32(device: &Device, data: &[f32], opts: MTLResourceOptions) -> Buffer {
    if data.is_empty() {
        return device.new_buffer(mem::size_of::<f32>().max(4) as u64, opts);
    }
    device.new_buffer_with_data(
        data.as_ptr() as *const std::ffi::c_void,
        (data.len() * mem::size_of::<f32>()) as u64,
        opts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_device() -> Option<Device> {
        Device::system_default()
    }

    #[test]
    fn test_upload_soa_roundtrip() {
        let device = match get_device() {
            Some(d) => d,
            None => {
                eprintln!("No Metal device — skipping");
                return;
            }
        };

        // 3 events, 2 observables: mass=[100,110,120], pt=[20,40,60]
        let soa = vec![100.0, 110.0, 120.0, 20.0, 40.0, 60.0];
        let result = upload_events_to_metal(&device, &soa, None, 3, 2).unwrap();

        assert_eq!(result.n_events, 3);
        assert_eq!(result.n_obs, 2);
        assert!(result.buf_weights.is_none());

        // Read back and verify (f64 → f32 → f64, so check ~1e-6 tolerance).
        let readback = read_metal_soa_to_f64(&result.buf_obs_soa, 6);
        for (i, (&orig, &got)) in soa.iter().zip(readback.iter()).enumerate() {
            assert!((orig - got).abs() < 0.01, "mismatch at {i}: expected {orig}, got {got}");
        }
    }

    #[test]
    fn test_upload_with_weights() {
        let device = match get_device() {
            Some(d) => d,
            None => return,
        };

        let soa = vec![1.0, 2.0, 3.0];
        let weights = vec![0.5, 1.0, 1.5];
        let result = upload_events_to_metal(&device, &soa, Some(&weights), 3, 1).unwrap();

        assert!(result.buf_weights.is_some());
        let w_back = read_metal_soa_to_f64(result.buf_weights.as_ref().unwrap(), 3);
        for (i, (&orig, &got)) in weights.iter().zip(w_back.iter()).enumerate() {
            assert!(
                (orig - got).abs() < 0.01,
                "weight mismatch at {i}: expected {orig}, got {got}"
            );
        }
    }

    #[test]
    fn test_upload_with_bounds() {
        let device = match get_device() {
            Some(d) => d,
            None => return,
        };

        let soa = vec![100.0, 110.0, 20.0, 40.0];
        let bounds = vec![(50.0, 200.0), (0.0, 100.0)];
        let result =
            upload_events_with_bounds_to_metal(&device, &soa, None, &bounds, 2, 2).unwrap();

        assert_eq!(result.n_events, 2);
        assert_eq!(result.n_obs, 2);

        let lo = read_metal_soa_to_f64(&result.buf_obs_lo, 2);
        let hi = read_metal_soa_to_f64(&result.buf_obs_hi, 2);
        assert!((lo[0] - 50.0).abs() < 0.01);
        assert!((lo[1] - 0.0).abs() < 0.01);
        assert!((hi[0] - 200.0).abs() < 0.01);
        assert!((hi[1] - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_validation_soa_length_mismatch() {
        let device = match get_device() {
            Some(d) => d,
            None => return,
        };

        let soa = vec![1.0, 2.0, 3.0]; // 3 elements but claim 2×2=4
        let result = upload_events_to_metal(&device, &soa, None, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_weights_length_mismatch() {
        let device = match get_device() {
            Some(d) => d,
            None => return,
        };

        let soa = vec![1.0, 2.0];
        let weights = vec![0.5]; // 1 weight but 2 events
        let result = upload_events_to_metal(&device, &soa, Some(&weights), 2, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_bounds_length_mismatch() {
        let device = match get_device() {
            Some(d) => d,
            None => return,
        };

        let soa = vec![1.0, 2.0];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)]; // 2 bounds but n_obs=1
        let result = upload_events_with_bounds_to_metal(&device, &soa, None, &bounds, 2, 1);
        assert!(result.is_err());
    }
}
