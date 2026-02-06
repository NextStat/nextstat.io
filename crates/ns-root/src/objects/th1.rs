//! TH1D and TH1F deserialization.
//!
//! ROOT TH1D/TH1F serialization layout (simplified):
//! ```text
//! TH1D (or TH1F)
//!   └─ TH1 (base)
//!        ├─ TNamed (name, title)
//!        ├─ TAttLine (skip 10 bytes: color u16, style u16, width u16 — but version-dependent)
//!        ├─ TAttFill (skip: color u16, style u16)
//!        ├─ TAttMarker (skip: color u16, style u16, size f32)
//!        ├─ fNcells (i32)
//!        ├─ fXaxis (TAxis)
//!        ├─ fYaxis (TAxis)
//!        ├─ fZaxis (TAxis)
//!        ├─ various scalar stats (fBarOffset, fBarWidth, fEntries, fTsumw, ...)
//!        ├─ fSumw2 (TArrayD — n i32 + n×f64)
//!        ├─ fOption (TString)
//!        ├─ fFunctions (TList — skip via byte_count)
//!        ├─ (fBufferSize, fBuffer if version >= 4) — usually empty
//!        ├─ fBinStatErrOpt, fBinStatErrOpt2 (if th1_version >= 8)
//!        └─ fNormFactor (if th1_version ≥ 2)
//!   └─ TArrayD / TArrayF (the actual bin contents)
//! ```

use crate::error::{Result, RootError};
use crate::histogram::Histogram;
use crate::rbuffer::RBuffer;

/// Read a TH1D from decompressed object bytes.
pub fn read_th1d(data: &[u8]) -> Result<Histogram> {
    let mut r = RBuffer::new(data);

    // TH1D version header
    let (th1d_ver, _bc) = r.read_version()?;
    if th1d_ver < 1 {
        return Err(RootError::Deserialization(format!(
            "unsupported TH1D version: {}",
            th1d_ver
        )));
    }

    // TH1 base
    let (name, title, n_cells, axis, sumw2) = read_th1_base(&mut r)?;

    // TArrayD: the bin contents (n_cells entries including under/overflow)
    let arr_n = r.read_u32()? as usize;
    if arr_n != n_cells as usize {
        return Err(RootError::Deserialization(format!(
            "TH1D array size {} != fNcells {}",
            arr_n, n_cells
        )));
    }
    let bin_content_raw = r.read_array_f64(arr_n)?;

    build_histogram(name, title, n_cells, &axis, &bin_content_raw, sumw2)
}

/// Read a TH1F from decompressed object bytes.
pub fn read_th1f(data: &[u8]) -> Result<Histogram> {
    let mut r = RBuffer::new(data);

    // TH1F version header
    let (th1f_ver, _bc) = r.read_version()?;
    if th1f_ver < 1 {
        return Err(RootError::Deserialization(format!(
            "unsupported TH1F version: {}",
            th1f_ver
        )));
    }

    // TH1 base
    let (name, title, n_cells, axis, sumw2) = read_th1_base(&mut r)?;

    // TArrayF: bin contents (f32)
    let arr_n = r.read_u32()? as usize;
    if arr_n != n_cells as usize {
        return Err(RootError::Deserialization(format!(
            "TH1F array size {} != fNcells {}",
            arr_n, n_cells
        )));
    }
    let bin_content_f32 = r.read_array_f32(arr_n)?;
    let bin_content_raw: Vec<f64> = bin_content_f32.iter().map(|&v| v as f64).collect();

    build_histogram(name, title, n_cells, &axis, &bin_content_raw, sumw2)
}

/// Axis info extracted from TAxis.
struct AxisInfo {
    n_bins: i32,
    x_min: f64,
    x_max: f64,
    /// Variable-width bin edges (empty for uniform binning).
    bin_edges: Vec<f64>,
}

/// Read the TH1 base class.
///
/// Returns (name, title, n_cells, axis_info, sumw2).
fn read_th1_base(r: &mut RBuffer) -> Result<(String, String, i32, AxisInfo, Option<Vec<f64>>)> {
    // TH1 version header
    let (th1_ver, th1_bc) = r.read_version()?;
    let th1_end = th1_bc.map(|bc| r.pos() + bc as usize);

    // TNamed
    let (name, title) = r.read_tnamed()?;

    // TAttLine (version + body)
    skip_streamer_object(r)?;
    // TAttFill
    skip_streamer_object(r)?;
    // TAttMarker
    skip_streamer_object(r)?;

    // fNcells
    let n_cells = r.read_i32()?;

    // fXaxis
    let axis = read_taxis(r)?;
    // fYaxis (skip)
    skip_taxis(r)?;
    // fZaxis (skip)
    skip_taxis(r)?;

    // Scalar stats (version-dependent count)
    let _bar_offset = r.read_i16()?;
    let _bar_width = r.read_i16()?;
    let _entries = r.read_f64()?;
    let _tsumw = r.read_f64()?;
    let _tsumw2 = r.read_f64()?;
    let _tsumwx = r.read_f64()?;
    let _tsumwx2 = r.read_f64()?;
    if th1_ver >= 2 {
        // fMaximum, fMinimum
        let _max = r.read_f64()?;
        let _min = r.read_f64()?;
    }
    if th1_ver >= 3 {
        // fNormFactor
        let _norm = r.read_f64()?;
    }

    // fContour (TArrayD)
    let contour_n = r.read_u32()? as usize;
    if contour_n > 0 {
        r.skip(contour_n * 8)?; // f64 array
    }

    // fSumw2 (TArrayD)
    let sumw2_n = r.read_u32()? as usize;
    let sumw2 = if sumw2_n > 0 {
        Some(r.read_array_f64(sumw2_n)?)
    } else {
        None
    };

    // fOption (TString)
    let _option = r.read_string()?;

    // fFunctions (TList) — skip via byte_count
    skip_streamer_object(r)?;

    // fBufferSize (i32) — if th1_ver >= 4 the full fBuffer is here but usually empty
    if th1_ver >= 4 {
        let buf_size = r.read_i32()?;
        if buf_size > 0 {
            // fBuffer: buf_size f64 values (very rare, skip)
            r.skip(buf_size as usize * 8)?;
        }
    }

    // fBinStatErrOpt (i32) if th1_ver >= 7
    if th1_ver >= 7 {
        let _err_opt = r.read_i32()?;
    }
    // fStatOverflows (i32) if th1_ver >= 8
    if th1_ver >= 8 {
        let _stat_overflows = r.read_i32()?;
    }

    // If there's a known byte_count end, seek to it to skip any fields we missed.
    if let Some(end) = th1_end {
        if end > r.pos() {
            r.set_pos(end);
        }
    }

    Ok((name, title, n_cells, axis, sumw2))
}

/// Read a TAxis.
fn read_taxis(r: &mut RBuffer) -> Result<AxisInfo> {
    let (_ver, bc) = r.read_version()?;
    let axis_end = bc.map(|b| r.pos() + b as usize);

    // TNamed
    let (_name, _title) = r.read_tnamed()?;

    // TAttAxis
    skip_streamer_object(r)?;

    let n_bins = r.read_i32()?;
    let x_min = r.read_f64()?;
    let x_max = r.read_f64()?;

    // fXbins (TArrayD) — variable bin edges
    let xbins_n = r.read_u32()? as usize;
    let bin_edges = if xbins_n > 0 {
        r.read_array_f64(xbins_n)?
    } else {
        Vec::new()
    };

    // Skip remaining axis fields to axis_end
    if let Some(end) = axis_end {
        if end > r.pos() {
            r.set_pos(end);
        }
    }

    Ok(AxisInfo {
        n_bins,
        x_min,
        x_max,
        bin_edges,
    })
}

/// Skip a TAxis without extracting data.
fn skip_taxis(r: &mut RBuffer) -> Result<()> {
    let (_ver, bc) = r.read_version()?;
    if let Some(b) = bc {
        r.set_pos(r.pos() + b as usize);
    } else {
        // Without byte count we can't reliably skip; attempt to read and discard.
        let _ = read_taxis_inner_skip(r)?;
    }
    Ok(())
}

/// Fallback: read axis fields without byte_count.
fn read_taxis_inner_skip(r: &mut RBuffer) -> Result<()> {
    // TNamed
    r.read_tnamed()?;
    // TAttAxis
    skip_streamer_object(r)?;
    let _n_bins = r.read_i32()?;
    let _x_min = r.read_f64()?;
    let _x_max = r.read_f64()?;
    let xbins_n = r.read_u32()? as usize;
    if xbins_n > 0 {
        r.skip(xbins_n * 8)?;
    }
    // The remaining fields (fFirst, fLast, fBits2, fTimeDisplay, fTimeFormat, fLabels, fModLabs)
    // are hard to skip without byte_count. This path is rarely hit since ROOT >= 4 always writes byte_count.
    Ok(())
}

/// Skip a streamer object that has a version header with byte_count.
///
/// ROOT writes most embedded objects with `(version | 0x40000000, byte_count)`.
/// We read the version header and skip `byte_count` bytes.
fn skip_streamer_object(r: &mut RBuffer) -> Result<()> {
    let (_ver, bc) = r.read_version()?;
    if let Some(b) = bc {
        r.set_pos(r.pos() + b as usize);
    }
    // If no byte_count, the object is minimal (version-only).
    // For TAttLine/TAttFill/TAttMarker the old format is very short and
    // we skip nothing extra — but this is fragile. In practice all modern ROOT
    // files use byte_count headers.
    Ok(())
}

/// Convert raw bin array (with under/overflow) to a Histogram.
fn build_histogram(
    name: String,
    title: String,
    n_cells: i32,
    axis: &AxisInfo,
    bin_content_raw: &[f64],
    sumw2_raw: Option<Vec<f64>>,
) -> Result<Histogram> {
    let n_bins = axis.n_bins as usize;

    // bin_content_raw has n_cells = n_bins + 2 entries:
    // [underflow, bin1, bin2, ..., binN, overflow]
    if bin_content_raw.len() != n_cells as usize {
        return Err(RootError::Deserialization(format!(
            "bin content length {} != n_cells {}",
            bin_content_raw.len(),
            n_cells
        )));
    }

    // Extract main bins (skip underflow=0 and overflow=last)
    let bin_content: Vec<f64> = bin_content_raw[1..1 + n_bins].to_vec();

    // Extract sumw2 for main bins
    let sumw2 = sumw2_raw.map(|sw2| {
        if sw2.len() == n_cells as usize {
            sw2[1..1 + n_bins].to_vec()
        } else {
            sw2
        }
    });

    // Compute bin edges
    let bin_edges = if !axis.bin_edges.is_empty() {
        axis.bin_edges.clone()
    } else {
        // Uniform binning
        let width = (axis.x_max - axis.x_min) / n_bins as f64;
        (0..=n_bins)
            .map(|i| axis.x_min + i as f64 * width)
            .collect()
    };

    // Compute entries from sum of bin contents
    let entries: f64 = bin_content.iter().sum();

    Ok(Histogram {
        name,
        title,
        n_bins,
        x_min: axis.x_min,
        x_max: axis.x_max,
        bin_edges,
        bin_content,
        sumw2,
        entries,
    })
}
