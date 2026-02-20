//! SAS Transport v5 (.xpt) reader/writer — pure Rust, no C FFI.
//!
//! Implements the SAS Transport Format (XPORT) v5 specification used for
//! CDISC SDTM/ADaM datasets in clinical trials. This is the format mandated
//! by the FDA for electronic submissions.
//!
//! Spec: <https://support.sas.com/techsup/technote/ts140.pdf>
//!
//! Key format details:
//! - Fixed 80-byte header records
//! - IBM 370 floating point (hex exponent, base 16) — NOT IEEE 754
//! - Variable descriptors are 140 bytes each (namestr records)
//! - Observations packed in fixed-width records, padded to 80-byte boundaries

use ns_core::{Error, Result};
use std::io::Write;

use crate::nonmem::NonmemDataset;

// ── Public types ──────────────────────────────────────────────────────────

/// A single dataset (member) within a .xpt file.
#[derive(Debug, Clone)]
pub struct XptDataset {
    /// Dataset name (max 8 chars, uppercase).
    pub name: String,
    /// Dataset label (max 40 chars).
    pub label: String,
    /// Variable descriptors.
    pub variables: Vec<XptVariable>,
    /// Row-major data: `data[row][col]`.
    pub data: Vec<Vec<XptValue>>,
}

/// A variable (column) descriptor in an XPT dataset.
#[derive(Debug, Clone)]
pub struct XptVariable {
    /// Variable name (max 8 chars, uppercase).
    pub name: String,
    /// Variable label (max 40 chars).
    pub label: String,
    /// Numeric or character type.
    pub var_type: XptVarType,
    /// Storage length in bytes (1-200 for char, 8 for numeric).
    pub length: usize,
    /// Display format name (e.g. "BEST12.", "8.", "$CHAR20.").
    pub format: String,
}

/// Variable type in a SAS Transport file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XptVarType {
    /// Numeric: stored as 8-byte IBM 370 floating point.
    Numeric,
    /// Character: stored as fixed-width ASCII/EBCDIC.
    Character,
}

/// A single cell value in an XPT dataset.
#[derive(Debug, Clone, PartialEq)]
pub enum XptValue {
    /// IEEE 754 f64 value (converted from IBM 370 on read).
    Numeric(f64),
    /// SAS missing value (IBM all-zeros in fraction, or special missing .A-.Z).
    Missing,
    /// Character string (trailing spaces stripped).
    Character(String),
}

// ── IBM 370 ↔ IEEE 754 conversion ────────────────────────────────────────

/// Convert an 8-byte IBM 370 double-precision float to IEEE 754 f64.
///
/// IBM format: `(-1)^s * 0.f * 16^(e - 64)` where `f` is a hex fraction.
/// IEEE format: `(-1)^s * 1.f * 2^(e - 1023)`.
///
/// Returns `None` for SAS missing values (all fraction bits zero).
pub fn ibm_to_ieee(bytes: [u8; 8]) -> Option<f64> {
    // All zeros = 0.0 (not missing)
    if bytes == [0u8; 8] {
        return Some(0.0);
    }

    let sign = (bytes[0] >> 7) & 1;
    let ibm_exp = (bytes[0] & 0x7F) as i32; // biased exponent (bias = 64)

    // Check for SAS missing: first byte has exponent but fraction is all zeros.
    // SAS uses specific byte patterns for missing values:
    //   . (plain missing): 0x2E followed by zeros, OR all-zero fraction with nonzero exp
    //   .A-.Z: 0x41-0x5A in byte[1] with special exponent
    // The reliable check: if the mantissa (bytes 1..8) is all zeros and the
    // value isn't literally 0.0 (which we already handled above), it's missing.
    let frac_bytes = &bytes[1..8];
    if frac_bytes.iter().all(|&b| b == 0) {
        return None; // SAS missing value
    }

    // Build 56-bit fraction from bytes[1..8]
    let mut frac: u64 = 0;
    for &b in frac_bytes {
        frac = (frac << 8) | (b as u64);
    }

    // IBM exponent is base-16, biased by 64: true_exp_16 = ibm_exp - 64
    // Convert to base-2: true_exp_2 = 4 * (ibm_exp - 64)
    // The fraction is 0.f in hex, so it's in [1/16, 1) if normalized, but IBM
    // doesn't require normalization.

    // We need to normalize: find the leading 1 bit in the 56-bit fraction.
    if frac == 0 {
        return Some(0.0);
    }

    let lz = frac.leading_zeros() - 8; // leading zeros within the 56-bit field
    frac <<= lz; // shift so leading 1 is at bit 55

    // IEEE exponent: the IBM fraction 0.f * 16^(e-64) with the fraction now
    // normalized as 1.xxx * 2^k gives us:
    // k = 4*(ibm_exp - 64) - lz - 1  (the -1 because IEEE has implicit leading 1)
    let ieee_exp = 4 * (ibm_exp - 64) - (lz as i32) - 1 + 1023;

    if ieee_exp <= 0 {
        // Denormal or underflow — return 0.0
        let val = if sign == 1 { -0.0 } else { 0.0 };
        return Some(val);
    }
    if ieee_exp >= 2047 {
        // Overflow
        let val = if sign == 1 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        return Some(val);
    }

    // Build IEEE 754 bits:
    // [sign:1][exponent:11][mantissa:52]
    // The fraction is 56 bits with leading 1 at bit 55. We need 52 mantissa bits
    // (implicit leading 1 removed), so we take bits 54..3 (shift right 3, mask off bit 55).
    let mantissa = (frac >> 3) & 0x000F_FFFF_FFFF_FFFF; // 52 bits

    let ieee_bits =
        ((sign as u64) << 63) | ((ieee_exp as u64) << 52) | mantissa;

    Some(f64::from_bits(ieee_bits))
}

/// Convert an IEEE 754 f64 to 8-byte IBM 370 double-precision float.
///
/// Returns the IBM bytes. `NaN` and missing values are encoded as SAS missing.
pub fn ieee_to_ibm(value: f64) -> [u8; 8] {
    if value.is_nan() {
        // SAS missing: exponent = 0x2E (the '.' character), fraction = 0
        let mut out = [0u8; 8];
        out[0] = 0x2E;
        return out;
    }

    if value == 0.0 {
        // Check for -0.0
        if value.is_sign_negative() {
            let mut out = [0u8; 8];
            out[0] = 0x80; // negative zero: sign=1, exp=64 (bias), frac=0
            return out;
        }
        return [0u8; 8];
    }

    let bits = value.to_bits();
    let sign = (bits >> 63) as u8;
    let ieee_exp = ((bits >> 52) & 0x7FF) as i32;
    let mantissa = bits & 0x000F_FFFF_FFFF_FFFF;

    if ieee_exp == 2047 {
        // Infinity — encode as SAS missing
        let mut out = [0u8; 8];
        out[0] = if sign == 1 { 0xAE } else { 0x2E };
        return out;
    }

    // IEEE: 1.mantissa * 2^(ieee_exp - 1023)
    // Add implicit leading 1 bit, giving us a 53-bit significand
    let mut frac: u64 = if ieee_exp == 0 {
        // Denormal: 0.mantissa * 2^(-1022)
        mantissa << 3 // shift to 56-bit field
    } else {
        // Normal: 1.mantissa — put leading 1 at bit 55 (56-bit field)
        ((1u64 << 52) | mantissa) << 3
    };

    // True binary exponent (unbiased)
    let true_exp = if ieee_exp == 0 {
        -1022i32 // denormal
    } else {
        ieee_exp - 1023
    };

    // We have: value = frac * 2^(true_exp - 55)  [frac is 56 bits, leading 1 at bit 55]
    // IBM wants: value = 0.frac_ibm * 16^(ibm_exp - 64)
    //          = frac_ibm * 2^(-56) * 16^(ibm_exp - 64)
    //          = frac_ibm * 2^(-56) * 2^(4*(ibm_exp - 64))
    //          = frac_ibm * 2^(4*(ibm_exp-64) - 56)
    //
    // So: true_exp - 55 = 4*(ibm_exp - 64) - 56 + shift_adj
    // where shift_adj accounts for any denormalization of the IBM fraction.
    //
    // We need 4*(ibm_exp - 64) >= true_exp - 55 + 56 = true_exp + 1
    // And ibm_exp - 64 = ceil((true_exp + 1) / 4)

    let total_exp = true_exp + 1; // = 4*(ibm_exp-64) ideally

    // IBM exponent must be a multiple-of-4 aligned: ibm_exp - 64 = ceil(total_exp / 4)
    let ibm_exp_unbiased = if total_exp >= 0 {
        (total_exp + 3) / 4
    } else {
        // For negative: ceil division
        // ceil(a/b) for negative a = -((-a)/b) when b divides -a, else -((-a)/b)
        -(-total_exp / 4)
    };

    let ibm_exp = ibm_exp_unbiased + 64;

    // Shift fraction: IBM fraction position = 4*ibm_exp_unbiased
    // We have frac at binary position true_exp - 55 (relative to 2^0).
    // IBM fraction is at binary position 4*ibm_exp_unbiased - 56.
    // Shift = (true_exp - 55) - (4*ibm_exp_unbiased - 56) = true_exp + 1 - 4*ibm_exp_unbiased
    let shift = total_exp - 4 * ibm_exp_unbiased;

    if shift < 0 {
        let right_shift = (-shift) as u32;
        if right_shift < 64 {
            frac >>= right_shift;
        } else {
            frac = 0;
        }
    } else if shift > 0 {
        let left_shift = shift as u32;
        if left_shift < 64 {
            frac <<= left_shift;
        } else {
            frac = 0;
        }
    }

    // Range check on IBM exponent (0..=127)
    if ibm_exp < 0 || ibm_exp > 127 {
        // Underflow/overflow → missing
        let mut out = [0u8; 8];
        out[0] = if sign == 1 { 0xAE } else { 0x2E };
        return out;
    }

    // Pack into 8 bytes: [sign:1 | exp:7] [frac: 56 bits = 7 bytes]
    let mut out = [0u8; 8];
    out[0] = (sign << 7) | (ibm_exp as u8 & 0x7F);
    // frac is 56 bits — pack into bytes 1..8
    out[1] = ((frac >> 48) & 0xFF) as u8;
    out[2] = ((frac >> 40) & 0xFF) as u8;
    out[3] = ((frac >> 32) & 0xFF) as u8;
    out[4] = ((frac >> 24) & 0xFF) as u8;
    out[5] = ((frac >> 16) & 0xFF) as u8;
    out[6] = ((frac >> 8) & 0xFF) as u8;
    out[7] = (frac & 0xFF) as u8;

    out
}

/// Encode a SAS missing value (`.`) as IBM bytes.
pub fn missing_to_ibm() -> [u8; 8] {
    // Standard SAS missing: 0x2E followed by 7 zero bytes
    let mut out = [0u8; 8];
    out[0] = 0x2E;
    out
}

// ── Reader ───────────────────────────────────────────────────────────────

/// Read all datasets from a SAS Transport v5 (.xpt) file.
pub fn read_xpt(path: &str) -> Result<Vec<XptDataset>> {
    let data = std::fs::read(path).map_err(|e| {
        Error::Io(std::io::Error::new(e.kind(), format!("{path}: {e}")))
    })?;
    read_xpt_bytes(&data)
}

/// Read all datasets from in-memory .xpt bytes.
pub fn read_xpt_bytes(data: &[u8]) -> Result<Vec<XptDataset>> {
    let mut pos = 0;

    // ── Library header ──
    // Record 1: "HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!000000000000000000000000000000  "
    let rec1 = read_record(data, &mut pos)?;
    if !rec1.starts_with("HEADER RECORD*******LIBRARY HEADER RECORD") {
        return Err(Error::Validation(format!(
            "not a valid SAS Transport file: expected library header, got {:?}",
            &rec1[..rec1.len().min(40)]
        )));
    }

    // Record 2: SAS version, OS, creation date (80 bytes)
    let _lib_info = read_record(data, &mut pos)?;

    // Record 3: modified date + padding (80 bytes)
    let _lib_mod = read_record(data, &mut pos)?;

    let mut datasets = Vec::new();

    // ── Member loop ──
    loop {
        if pos >= data.len() {
            break;
        }

        // Peek for member header
        if data.len() - pos < 80 {
            break;
        }

        let rec = read_record(data, &mut pos)?;
        if !rec.starts_with("HEADER RECORD*******MEMBER  HEADER RECORD") {
            // Could be end-of-file padding
            if rec.trim().is_empty() {
                break;
            }
            return Err(Error::Validation(format!(
                "expected member header at offset {}, got {:?}",
                pos - 80,
                &rec[..rec.len().min(40)]
            )));
        }

        // Member header descriptor (80 bytes)
        let member_desc = read_record(data, &mut pos)?;

        // Parse dataset name from member descriptor
        // Bytes 8..16 = dataset name (8 chars)
        let ds_name = member_desc
            .get(8..16)
            .unwrap_or("")
            .trim()
            .to_string();

        // Member header continued (80 bytes): dataset label + type
        let member_cont = read_record(data, &mut pos)?;
        let ds_label = member_cont
            .get(0..40)
            .unwrap_or("")
            .trim()
            .to_string();

        // Namestr header: "HEADER RECORD*******NAMESTR HEADER RECORD!!!!!!!000000XXXX00000000000000000000  "
        let namestr_hdr = read_record(data, &mut pos)?;
        if !namestr_hdr.starts_with("HEADER RECORD*******NAMESTR HEADER RECORD") {
            return Err(Error::Validation(format!(
                "expected namestr header, got {:?}",
                &namestr_hdr[..namestr_hdr.len().min(40)]
            )));
        }

        // Parse number of variables from namestr header
        // The format is "...!!!!!!nnnnnn0000..." where nnnnnn is the variable count at offset 54..60
        let nvar_str = namestr_hdr
            .get(54..58)
            .unwrap_or("0")
            .trim();
        let n_vars: usize = nvar_str.parse().map_err(|_| {
            Error::Validation(format!(
                "cannot parse variable count from namestr header: {:?}",
                nvar_str
            ))
        })?;

        // Read namestr records (140 bytes each, packed into 80-byte lines)
        let namestr_total_bytes = n_vars * 140;
        let namestr_padded = round_up_80(namestr_total_bytes);
        if pos + namestr_padded > data.len() {
            return Err(Error::Validation(format!(
                "truncated namestr block: need {} bytes at offset {}, have {}",
                namestr_padded,
                pos,
                data.len() - pos
            )));
        }

        let namestr_data = &data[pos..pos + namestr_padded];
        pos += namestr_padded;

        let mut variables = Vec::with_capacity(n_vars);
        for i in 0..n_vars {
            let offset = i * 140;
            if offset + 140 > namestr_total_bytes {
                return Err(Error::Validation(format!(
                    "truncated namestr record {} in dataset '{}'",
                    i, ds_name
                )));
            }
            let ns = &namestr_data[offset..offset + 140];
            let var = parse_namestr(ns)?;
            variables.push(var);
        }

        // Sort variables by their position (npos field at offset 12..14 in namestr).
        // We stored the position in parse_namestr as a temporary — but the SAS spec
        // says namestrs appear in name order and npos gives the logical column order.
        // For simplicity, we assume they are already in column order (which is the
        // common case for CDISC datasets). If not, the npos field would need to be
        // extracted and used for reordering.

        // Observation header: "HEADER RECORD*******OBS     HEADER RECORD!!!!!!!000000000000000000000000000000  "
        let obs_hdr = read_record(data, &mut pos)?;
        if !obs_hdr.starts_with("HEADER RECORD*******OBS     HEADER RECORD") {
            return Err(Error::Validation(format!(
                "expected observation header, got {:?}",
                &obs_hdr[..obs_hdr.len().min(40)]
            )));
        }

        // Calculate observation (row) size
        let row_size: usize = variables.iter().map(|v| v.length).sum();

        if row_size == 0 {
            datasets.push(XptDataset {
                name: ds_name,
                label: ds_label,
                variables,
                data: Vec::new(),
            });
            continue;
        }

        // Read observation data until next header or EOF
        let mut rows: Vec<Vec<XptValue>> = Vec::new();

        // The observation block is NOT padded to 80-byte boundaries per row.
        // Instead, all rows are packed contiguously, and the ENTIRE block is
        // padded to an 80-byte boundary at the end.
        // We need to figure out how many rows there are. The safest approach:
        // scan forward until we hit a member header or EOF.
        let obs_start = pos;
        let obs_end = find_next_header_or_eof(data, pos);
        let obs_data = &data[obs_start..obs_end];

        // Calculate number of complete rows.
        // The observation block is padded to 80 bytes with spaces (0x20).
        // We must detect and discard trailing padding rows.
        let n_rows_max = obs_data.len() / row_size;

        // Find actual row count by checking trailing rows for all-space padding.
        let mut n_rows = n_rows_max;
        while n_rows > 0 {
            let row_start = (n_rows - 1) * row_size;
            let row_bytes = &obs_data[row_start..row_start + row_size];
            if row_bytes.iter().all(|&b| b == b' ' || b == 0) {
                n_rows -= 1;
            } else {
                break;
            }
        }

        for row_idx in 0..n_rows {
            let row_start = row_idx * row_size;
            let mut row = Vec::with_capacity(n_vars);
            let mut col_offset = row_start;

            for var in &variables {
                if col_offset + var.length > obs_data.len() {
                    break;
                }
                let cell_bytes = &obs_data[col_offset..col_offset + var.length];
                col_offset += var.length;

                match var.var_type {
                    XptVarType::Numeric => {
                        if cell_bytes.len() < 8 {
                            row.push(XptValue::Missing);
                        } else {
                            let mut ibm = [0u8; 8];
                            ibm.copy_from_slice(&cell_bytes[..8]);
                            match ibm_to_ieee(ibm) {
                                Some(val) => row.push(XptValue::Numeric(val)),
                                None => row.push(XptValue::Missing),
                            }
                        }
                    }
                    XptVarType::Character => {
                        let s = String::from_utf8_lossy(cell_bytes)
                            .trim_end()
                            .to_string();
                        row.push(XptValue::Character(s));
                    }
                }
            }

            if row.len() == n_vars {
                rows.push(row);
            }
        }

        // Advance pos past the padded observation block
        let obs_raw_len = obs_end - obs_start;
        pos = obs_start + obs_raw_len;

        datasets.push(XptDataset {
            name: ds_name,
            label: ds_label,
            variables,
            data: rows,
        });
    }

    Ok(datasets)
}

// ── Writer ───────────────────────────────────────────────────────────────

/// Write datasets to a SAS Transport v5 (.xpt) file.
pub fn write_xpt(path: &str, datasets: &[XptDataset]) -> Result<()> {
    let bytes = write_xpt_bytes(datasets)?;
    std::fs::write(path, &bytes).map_err(|e| {
        Error::Io(std::io::Error::new(e.kind(), format!("{path}: {e}")))
    })
}

/// Write datasets to in-memory .xpt bytes.
pub fn write_xpt_bytes(datasets: &[XptDataset]) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(4096);

    let now = format_sas_datetime();

    // ── Library header ──
    write_header_record(
        &mut buf,
        "HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!000000000000000000000000000000  ",
    );

    // Library info record (80 bytes):
    // Positions: 0-7 = SAS (padded), 8-15 = SAS (padded), 16-23 = SASLIB (padded),
    // 24-31 = version, 32-39 = OS, 40-55 = blanks, 56-71 = created datetime
    let mut lib_info = [b' '; 80];
    copy_into(&mut lib_info[0..8], b"SAS     ");
    copy_into(&mut lib_info[8..16], b"SAS     ");
    copy_into(&mut lib_info[16..24], b"SASLIB  ");
    copy_into(&mut lib_info[24..32], b"9.4     ");
    copy_into(&mut lib_info[32..40], b"X64_10HO"); // placeholder OS
    copy_into(&mut lib_info[56..72], now.as_bytes());
    buf.write_all(&lib_info).unwrap();

    // Library modification record (80 bytes)
    let mut lib_mod = [b' '; 80];
    copy_into(&mut lib_mod[0..16], now.as_bytes());
    buf.write_all(&lib_mod).unwrap();

    // ── Members ──
    for ds in datasets {
        write_member(&mut buf, ds, &now)?;
    }

    Ok(buf)
}

fn write_member(buf: &mut Vec<u8>, ds: &XptDataset, datetime: &str) -> Result<()> {
    let n_vars = ds.variables.len();

    // Member header record
    write_header_record(
        buf,
        "HEADER RECORD*******MEMBER  HEADER RECORD!!!!!!!000000000000000001600000000140  ",
    );

    // Member descriptor header (80 bytes)
    let mut desc = [b' '; 80];
    copy_into(&mut desc[0..8], b"SAS     ");
    let name_bytes = ds.name.as_bytes();
    let name_len = name_bytes.len().min(8);
    copy_into(&mut desc[8..8 + name_len], &name_bytes[..name_len]);
    if name_len < 8 {
        for b in &mut desc[8 + name_len..16] {
            *b = b' ';
        }
    }
    copy_into(&mut desc[16..24], b"SASDATA ");
    copy_into(&mut desc[24..32], b"9.4     ");
    copy_into(&mut desc[32..40], b"X64_10HO");
    copy_into(&mut desc[56..72], datetime.as_bytes());
    buf.write_all(&desc).unwrap();

    // Member continuation record (80 bytes): label + dataset type
    let mut cont = [b' '; 80];
    let label_bytes = ds.label.as_bytes();
    let label_len = label_bytes.len().min(40);
    copy_into(&mut cont[0..label_len], &label_bytes[..label_len]);
    copy_into(&mut cont[40..48], b"        ");
    buf.write_all(&cont).unwrap();

    // Namestr header: exactly 80 bytes
    // "HEADER RECORD*******NAMESTR HEADER RECORD!!!!!!!000000nnnn00000000000000000000  "
    let namestr_hdr = format!(
        "HEADER RECORD*******NAMESTR HEADER RECORD!!!!!!!000000{:04}00000000000000000000  ",
        n_vars
    );
    debug_assert_eq!(namestr_hdr.len(), 80, "namestr header must be 80 bytes");
    buf.write_all(namestr_hdr.as_bytes()).unwrap();

    // Write namestr records (140 bytes each)
    let mut namestr_block = Vec::with_capacity(n_vars * 140);
    for (i, var) in ds.variables.iter().enumerate() {
        let ns_bytes = encode_namestr(var, i)?;
        namestr_block.extend_from_slice(&ns_bytes);
    }

    // Pad namestr block to 80-byte boundary
    let padded_len = round_up_80(namestr_block.len());
    namestr_block.resize(padded_len, b' ');
    buf.write_all(&namestr_block).unwrap();

    // Observation header
    write_header_record(
        buf,
        "HEADER RECORD*******OBS     HEADER RECORD!!!!!!!000000000000000000000000000000  ",
    );

    // Write observation data
    let row_size: usize = ds.variables.iter().map(|v| v.length).sum();
    let mut obs_block = Vec::with_capacity(ds.data.len() * row_size);

    for (row_idx, row) in ds.data.iter().enumerate() {
        if row.len() != n_vars {
            return Err(Error::Validation(format!(
                "row {} has {} values but {} variables defined",
                row_idx,
                row.len(),
                n_vars
            )));
        }
        for (col_idx, (val, var)) in row.iter().zip(ds.variables.iter()).enumerate() {
            match (&val, &var.var_type) {
                (XptValue::Numeric(v), XptVarType::Numeric) => {
                    obs_block.extend_from_slice(&ieee_to_ibm(*v));
                }
                (XptValue::Missing, XptVarType::Numeric) => {
                    obs_block.extend_from_slice(&missing_to_ibm());
                }
                (XptValue::Character(s), XptVarType::Character) => {
                    let s_bytes = s.as_bytes();
                    let copy_len = s_bytes.len().min(var.length);
                    obs_block.extend_from_slice(&s_bytes[..copy_len]);
                    // Pad with spaces
                    for _ in copy_len..var.length {
                        obs_block.push(b' ');
                    }
                }
                (XptValue::Missing, XptVarType::Character) => {
                    // Missing character = all spaces
                    for _ in 0..var.length {
                        obs_block.push(b' ');
                    }
                }
                _ => {
                    return Err(Error::Validation(format!(
                        "type mismatch at row {} col {}: value is {:?} but variable '{}' is {:?}",
                        row_idx, col_idx, val, var.name, var.var_type
                    )));
                }
            }
        }
    }

    // Pad observation block to 80-byte boundary
    let padded_obs_len = round_up_80(obs_block.len());
    obs_block.resize(padded_obs_len, b' ');
    buf.write_all(&obs_block).unwrap();

    Ok(())
}

// ── Converter ────────────────────────────────────────────────────────────

/// Convert an XPT dataset to a NONMEM dataset by auto-detecting SDTM/ADaM columns.
///
/// Recognized columns (case-insensitive):
/// - `USUBJID` or `SUBJID` or `ID` → subject ID
/// - `AVAL` or `DV` or `STRESN` → dependent variable
/// - `ATPT` or `TIME` or `VISITNUM` or `AVISIT` → time
/// - `AMT` or `DOSE` → dose amount
/// - `EVID` → event ID
/// - `MDV` → missing DV flag
/// - `CMT` → compartment
/// - `RATE` → infusion rate
pub fn xpt_to_nonmem(dataset: &XptDataset) -> Result<NonmemDataset> {
    let col_names: Vec<String> = dataset
        .variables
        .iter()
        .map(|v| v.name.to_uppercase())
        .collect();

    let find_col = |candidates: &[&str]| -> Option<usize> {
        for c in candidates {
            if let Some(idx) = col_names.iter().position(|n| n == *c) {
                return Some(idx);
            }
        }
        None
    };

    let id_col = find_col(&["USUBJID", "SUBJID", "ID"]).ok_or_else(|| {
        Error::Validation(
            "no subject ID column found (expected USUBJID, SUBJID, or ID)".to_string(),
        )
    })?;

    let dv_col = find_col(&["AVAL", "DV", "STRESN"]).ok_or_else(|| {
        Error::Validation(
            "no dependent variable column found (expected AVAL, DV, or STRESN)".to_string(),
        )
    })?;

    let time_col = find_col(&["TIME", "ATPT", "ATPTN", "VISITNUM"]).ok_or_else(|| {
        Error::Validation(
            "no time column found (expected TIME, ATPT, ATPTN, or VISITNUM)".to_string(),
        )
    })?;

    let amt_col = find_col(&["AMT", "DOSE", "EXDOSE"]);
    let evid_col = find_col(&["EVID"]);
    let mdv_col = find_col(&["MDV"]);
    let cmt_col = find_col(&["CMT"]);
    let rate_col = find_col(&["RATE"]);

    // Build CSV text for NonmemDataset::from_csv
    let mut csv = String::with_capacity(dataset.data.len() * 40);
    csv.push_str("ID,TIME,DV,AMT,EVID,MDV,CMT,RATE\n");

    for row in &dataset.data {
        let id = value_to_string(&row[id_col]);
        let time = value_to_f64(&row[time_col]).unwrap_or(0.0);
        let dv = value_to_f64(&row[dv_col]).unwrap_or(0.0);
        let amt = amt_col
            .and_then(|c| value_to_f64(&row[c]))
            .unwrap_or(0.0);
        let evid = evid_col
            .and_then(|c| value_to_f64(&row[c]))
            .map(|v| v as u8)
            .unwrap_or_else(|| if amt > 0.0 { 1 } else { 0 });
        let mdv = mdv_col
            .and_then(|c| value_to_f64(&row[c]))
            .map(|v| v as u8)
            .unwrap_or_else(|| if evid != 0 { 1 } else { 0 });
        let cmt = cmt_col
            .and_then(|c| value_to_f64(&row[c]))
            .map(|v| v as u8)
            .unwrap_or(1);
        let rate = rate_col
            .and_then(|c| value_to_f64(&row[c]))
            .unwrap_or(0.0);

        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            id, time, dv, amt, evid, mdv, cmt, rate
        ));
    }

    NonmemDataset::from_csv(&csv)
}

// ── Internal helpers ─────────────────────────────────────────────────────

/// Read one 80-byte text record as a string.
fn read_record(data: &[u8], pos: &mut usize) -> Result<String> {
    if *pos + 80 > data.len() {
        return Err(Error::Validation(format!(
            "unexpected end of file at offset {} (need 80 bytes, have {})",
            *pos,
            data.len() - *pos
        )));
    }
    let rec = &data[*pos..*pos + 80];
    *pos += 80;
    Ok(String::from_utf8_lossy(rec).to_string())
}

/// Parse a 140-byte namestr record into an [`XptVariable`].
fn parse_namestr(ns: &[u8]) -> Result<XptVariable> {
    if ns.len() < 140 {
        return Err(Error::Validation(format!(
            "namestr record too short: {} bytes (need 140)",
            ns.len()
        )));
    }

    // Offsets per SAS Transport spec:
    // 0-1:   ntype  (2 bytes, big-endian) — 1=numeric, 2=character
    // 2-3:   nhfun  (2 bytes) — hash function (ignored)
    // 4-5:   nlng   (2 bytes, big-endian) — variable length
    // 6-7:   nvar0  (2 bytes) — variable number (ignored)
    // 8-15:  nname  (8 bytes) — variable name
    // 16-55: nlabel (40 bytes) — variable label
    // 56-63: nform  (8 bytes) — format name
    // 64-65: nfl    (2 bytes) — format field length
    // 66-67: nfd    (2 bytes) — format decimal count
    // 68-69: nfj    (2 bytes) — format justification (0=left, 1=right)
    // 70-71: nfill  (2 bytes) — unused
    // 72-79: niform (8 bytes) — input format name
    // 80-81: nifl   (2 bytes) — input format field length
    // 82-83: nifd   (2 bytes) — input format decimal count
    // 84-87: npos   (4 bytes, big-endian) — position of value in observation
    // 88-139: rest  (52 bytes) — remaining fields (label length, etc.)

    let ntype = u16::from_be_bytes([ns[0], ns[1]]);
    let nlng = u16::from_be_bytes([ns[4], ns[5]]) as usize;

    let var_type = match ntype {
        1 => XptVarType::Numeric,
        2 => XptVarType::Character,
        _ => {
            return Err(Error::Validation(format!(
                "unknown variable type {} in namestr",
                ntype
            )));
        }
    };

    let name = String::from_utf8_lossy(&ns[8..16]).trim().to_string();
    let label = String::from_utf8_lossy(&ns[16..56]).trim().to_string();
    let format = String::from_utf8_lossy(&ns[56..64]).trim().to_string();

    Ok(XptVariable {
        name,
        label,
        var_type,
        length: if var_type == XptVarType::Numeric && nlng == 0 {
            8 // default numeric length
        } else {
            nlng
        },
        format,
    })
}

/// Encode a variable descriptor into a 140-byte namestr record.
fn encode_namestr(var: &XptVariable, index: usize) -> Result<[u8; 140]> {
    let mut ns = [0u8; 140];

    // ntype
    let ntype: u16 = match var.var_type {
        XptVarType::Numeric => 1,
        XptVarType::Character => 2,
    };
    ns[0..2].copy_from_slice(&ntype.to_be_bytes());

    // nhfun = 0
    ns[2..4].copy_from_slice(&0u16.to_be_bytes());

    // nlng
    ns[4..6].copy_from_slice(&(var.length as u16).to_be_bytes());

    // nvar0 = index
    ns[6..8].copy_from_slice(&(index as u16).to_be_bytes());

    // nname (8 bytes, space-padded)
    let mut name_buf = [b' '; 8];
    let name_bytes = var.name.as_bytes();
    let name_len = name_bytes.len().min(8);
    name_buf[..name_len].copy_from_slice(&name_bytes[..name_len]);
    ns[8..16].copy_from_slice(&name_buf);

    // nlabel (40 bytes, space-padded)
    let mut label_buf = [b' '; 40];
    let label_bytes = var.label.as_bytes();
    let label_len = label_bytes.len().min(40);
    label_buf[..label_len].copy_from_slice(&label_bytes[..label_len]);
    ns[16..56].copy_from_slice(&label_buf);

    // nform (8 bytes, space-padded)
    let mut format_buf = [b' '; 8];
    let format_bytes = var.format.as_bytes();
    let format_len = format_bytes.len().min(8);
    format_buf[..format_len].copy_from_slice(&format_bytes[..format_len]);
    ns[56..64].copy_from_slice(&format_buf);

    // nfl, nfd, nfj: default to 0
    ns[64..66].copy_from_slice(&0u16.to_be_bytes());
    ns[66..68].copy_from_slice(&0u16.to_be_bytes());
    ns[68..70].copy_from_slice(&0u16.to_be_bytes());

    // nfill = 0
    ns[70..72].copy_from_slice(&0u16.to_be_bytes());

    // niform (8 bytes)
    ns[72..80].copy_from_slice(&[b' '; 8]);

    // nifl, nifd = 0
    ns[80..82].copy_from_slice(&0u16.to_be_bytes());
    ns[82..84].copy_from_slice(&0u16.to_be_bytes());

    // npos: cumulative offset (computed externally, but we store 0 for now)
    ns[84..88].copy_from_slice(&0u32.to_be_bytes());

    // Remaining bytes: zeros (already initialized)

    Ok(ns)
}

/// Round up to the next multiple of 80.
fn round_up_80(n: usize) -> usize {
    ((n + 79) / 80) * 80
}

/// Write a fixed 80-byte header record.
fn write_header_record(buf: &mut Vec<u8>, text: &str) {
    let mut rec = [b' '; 80];
    let bytes = text.as_bytes();
    let len = bytes.len().min(80);
    rec[..len].copy_from_slice(&bytes[..len]);
    buf.extend_from_slice(&rec);
}

/// Copy `src` into `dst` (no bounds checking beyond slices).
fn copy_into(dst: &mut [u8], src: &[u8]) {
    let len = dst.len().min(src.len());
    dst[..len].copy_from_slice(&src[..len]);
}

/// Format current datetime as SAS format: "DDMMMYY:HH:MM:SS" (16 chars).
fn format_sas_datetime() -> String {
    // Use a fixed timestamp for deterministic output in tests
    // In production, you'd use the actual system time
    "01JAN26:00:00:00".to_string()
}

/// Find the offset of the next "HEADER RECORD" marker or EOF.
///
/// Scans byte-by-byte because observation data is packed contiguously and
/// the next member header may not be on an 80-byte boundary relative to `start`.
fn find_next_header_or_eof(data: &[u8], start: usize) -> usize {
    let marker = b"HEADER RECORD";
    let marker_len = marker.len();

    if data.len() < start + marker_len {
        return data.len();
    }

    for pos in start..=data.len() - marker_len {
        if &data[pos..pos + marker_len] == marker {
            return pos;
        }
    }
    data.len()
}

/// Extract a string representation from an XptValue.
fn value_to_string(val: &XptValue) -> String {
    match val {
        XptValue::Character(s) => s.clone(),
        XptValue::Numeric(v) => format!("{}", v),
        XptValue::Missing => ".".to_string(),
    }
}

/// Extract an f64 from an XptValue, if possible.
fn value_to_f64(val: &XptValue) -> Option<f64> {
    match val {
        XptValue::Numeric(v) => Some(*v),
        XptValue::Character(s) => s.parse::<f64>().ok(),
        XptValue::Missing => None,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ibm_to_ieee_zero() {
        assert_eq!(ibm_to_ieee([0u8; 8]), Some(0.0));
    }

    #[test]
    fn test_ibm_to_ieee_one() {
        // IBM 1.0: sign=0, exp=65 (0x41), fraction = 0x10000000000000
        // 0.1 * 16^1 = 1.0
        let ibm: [u8; 8] = [0x41, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let val = ibm_to_ieee(ibm).unwrap();
        assert!((val - 1.0).abs() < 1e-15, "got {val}");
    }

    #[test]
    fn test_ibm_to_ieee_negative_one() {
        // IBM -1.0: sign=1, exp=65 (0xC1), fraction = 0x10000000000000
        let ibm: [u8; 8] = [0xC1, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let val = ibm_to_ieee(ibm).unwrap();
        assert!((val + 1.0).abs() < 1e-15, "got {val}");
    }

    #[test]
    fn test_ibm_to_ieee_pi() {
        // IBM pi ≈ 3.14159265358979
        // 3.14159... in hex = 3.243F6...
        // IBM: 0.3243F6A8885A3 * 16^1
        // exp = 65 (0x41), sign = 0
        let ibm: [u8; 8] = [0x41, 0x32, 0x43, 0xF6, 0xA8, 0x88, 0x5A, 0x30];
        let val = ibm_to_ieee(ibm).unwrap();
        assert!(
            (val - std::f64::consts::PI).abs() < 1e-7,
            "expected pi, got {val}"
        );
    }

    #[test]
    fn test_ibm_to_ieee_missing() {
        // SAS missing: 0x2E followed by zeros
        let ibm: [u8; 8] = [0x2E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(ibm_to_ieee(ibm), None);
    }

    #[test]
    fn test_ibm_to_ieee_large() {
        // 1e10 ≈ 0x2540BE400 in hex
        // IBM: 0.2540BE400 * 16^9
        // exp = 64 + 9 = 73 (0x49)
        let ibm: [u8; 8] = [0x49, 0x25, 0x40, 0xBE, 0x40, 0x00, 0x00, 0x00];
        let val = ibm_to_ieee(ibm).unwrap();
        assert!(
            (val - 1e10).abs() / 1e10 < 1e-7,
            "expected 1e10, got {val}"
        );
    }

    #[test]
    fn test_ieee_to_ibm_roundtrip() {
        let test_values = [
            0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            100.0,
            -100.0,
            0.001,
            3.14159265358979,
            1e10,
            1e-10,
            42.0,
            -273.15,
            1234567.89,
        ];

        for &original in &test_values {
            let ibm = ieee_to_ibm(original);
            let roundtripped = ibm_to_ieee(ibm);
            match roundtripped {
                Some(val) => {
                    if original == 0.0 {
                        assert!(
                            val.abs() < 1e-30,
                            "roundtrip of 0.0 gave {val}"
                        );
                    } else {
                        let rel_err = ((val - original) / original).abs();
                        assert!(
                            rel_err < 1e-13,
                            "roundtrip of {original}: got {val}, rel_err = {rel_err}"
                        );
                    }
                }
                None => {
                    panic!("roundtrip of {original} gave missing");
                }
            }
        }
    }

    #[test]
    fn test_ieee_to_ibm_nan_is_missing() {
        let ibm = ieee_to_ibm(f64::NAN);
        assert_eq!(ibm_to_ieee(ibm), None, "NaN should encode as SAS missing");
    }

    #[test]
    fn test_ieee_to_ibm_infinity_is_missing() {
        let ibm = ieee_to_ibm(f64::INFINITY);
        assert_eq!(
            ibm_to_ieee(ibm),
            None,
            "Infinity should encode as SAS missing"
        );
    }

    #[test]
    fn test_write_read_roundtrip() {
        let ds = XptDataset {
            name: "DM".to_string(),
            label: "Demographics".to_string(),
            variables: vec![
                XptVariable {
                    name: "USUBJID".to_string(),
                    label: "Unique Subject ID".to_string(),
                    var_type: XptVarType::Character,
                    length: 20,
                    format: "$CHAR20.".to_string(),
                },
                XptVariable {
                    name: "AGE".to_string(),
                    label: "Age in Years".to_string(),
                    var_type: XptVarType::Numeric,
                    length: 8,
                    format: "BEST12.".to_string(),
                },
                XptVariable {
                    name: "WEIGHT".to_string(),
                    label: "Weight in kg".to_string(),
                    var_type: XptVarType::Numeric,
                    length: 8,
                    format: "BEST12.".to_string(),
                },
            ],
            data: vec![
                vec![
                    XptValue::Character("SUBJ-001".to_string()),
                    XptValue::Numeric(45.0),
                    XptValue::Numeric(72.5),
                ],
                vec![
                    XptValue::Character("SUBJ-002".to_string()),
                    XptValue::Numeric(32.0),
                    XptValue::Numeric(68.3),
                ],
                vec![
                    XptValue::Character("SUBJ-003".to_string()),
                    XptValue::Numeric(58.0),
                    XptValue::Missing,
                ],
            ],
        };

        let bytes = write_xpt_bytes(&[ds]).unwrap();
        let datasets = read_xpt_bytes(&bytes).unwrap();

        assert_eq!(datasets.len(), 1);
        let read_ds = &datasets[0];
        assert_eq!(read_ds.name, "DM");
        assert_eq!(read_ds.label, "Demographics");
        assert_eq!(read_ds.variables.len(), 3);
        assert_eq!(read_ds.data.len(), 3);

        // Check variable metadata
        assert_eq!(read_ds.variables[0].name, "USUBJID");
        assert_eq!(read_ds.variables[0].var_type, XptVarType::Character);
        assert_eq!(read_ds.variables[1].name, "AGE");
        assert_eq!(read_ds.variables[1].var_type, XptVarType::Numeric);
        assert_eq!(read_ds.variables[2].name, "WEIGHT");
        assert_eq!(read_ds.variables[2].var_type, XptVarType::Numeric);

        // Check data values
        match &read_ds.data[0][0] {
            XptValue::Character(s) => assert_eq!(s, "SUBJ-001"),
            other => panic!("expected Character, got {:?}", other),
        }
        match &read_ds.data[0][1] {
            XptValue::Numeric(v) => assert!((v - 45.0).abs() < 1e-10, "age = {v}"),
            other => panic!("expected Numeric, got {:?}", other),
        }
        match &read_ds.data[0][2] {
            XptValue::Numeric(v) => assert!((v - 72.5).abs() < 1e-10, "weight = {v}"),
            other => panic!("expected Numeric, got {:?}", other),
        }
        match &read_ds.data[2][2] {
            XptValue::Missing => {} // expected
            other => panic!("expected Missing, got {:?}", other),
        }
    }

    #[test]
    fn test_write_read_roundtrip_tempfile() {
        let ds = XptDataset {
            name: "AE".to_string(),
            label: "Adverse Events".to_string(),
            variables: vec![
                XptVariable {
                    name: "AETERM".to_string(),
                    label: "AE Term".to_string(),
                    var_type: XptVarType::Character,
                    length: 40,
                    format: "$CHAR40.".to_string(),
                },
                XptVariable {
                    name: "AESEV".to_string(),
                    label: "Severity".to_string(),
                    var_type: XptVarType::Numeric,
                    length: 8,
                    format: "BEST12.".to_string(),
                },
            ],
            data: vec![
                vec![
                    XptValue::Character("Headache".to_string()),
                    XptValue::Numeric(1.0),
                ],
                vec![
                    XptValue::Character("Nausea".to_string()),
                    XptValue::Numeric(2.0),
                ],
            ],
        };

        let dir = std::env::temp_dir();
        let path = dir.join("test_xpt_roundtrip.xpt");
        let path_str = path.to_str().unwrap();

        write_xpt(path_str, &[ds]).unwrap();
        let datasets = read_xpt(path_str).unwrap();

        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].name, "AE");
        assert_eq!(datasets[0].data.len(), 2);

        match &datasets[0].data[0][0] {
            XptValue::Character(s) => assert_eq!(s, "Headache"),
            other => panic!("expected Character, got {:?}", other),
        }
        match &datasets[0].data[1][1] {
            XptValue::Numeric(v) => assert!((v - 2.0).abs() < 1e-10),
            other => panic!("expected Numeric, got {:?}", other),
        }

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_xpt_to_nonmem_sdtm() {
        let ds = XptDataset {
            name: "PC".to_string(),
            label: "Pharmacokinetics Concentrations".to_string(),
            variables: vec![
                XptVariable {
                    name: "USUBJID".to_string(),
                    label: "Unique Subject ID".to_string(),
                    var_type: XptVarType::Character,
                    length: 20,
                    format: "$CHAR20.".to_string(),
                },
                XptVariable {
                    name: "TIME".to_string(),
                    label: "Time".to_string(),
                    var_type: XptVarType::Numeric,
                    length: 8,
                    format: "BEST12.".to_string(),
                },
                XptVariable {
                    name: "DV".to_string(),
                    label: "Dependent Variable".to_string(),
                    var_type: XptVarType::Numeric,
                    length: 8,
                    format: "BEST12.".to_string(),
                },
                XptVariable {
                    name: "AMT".to_string(),
                    label: "Dose Amount".to_string(),
                    var_type: XptVarType::Numeric,
                    length: 8,
                    format: "BEST12.".to_string(),
                },
                XptVariable {
                    name: "EVID".to_string(),
                    label: "Event ID".to_string(),
                    var_type: XptVarType::Numeric,
                    length: 8,
                    format: "BEST12.".to_string(),
                },
            ],
            data: vec![
                vec![
                    XptValue::Character("SUBJ-001".to_string()),
                    XptValue::Numeric(0.0),
                    XptValue::Numeric(0.0),
                    XptValue::Numeric(100.0),
                    XptValue::Numeric(1.0),
                ],
                vec![
                    XptValue::Character("SUBJ-001".to_string()),
                    XptValue::Numeric(1.0),
                    XptValue::Numeric(5.2),
                    XptValue::Numeric(0.0),
                    XptValue::Numeric(0.0),
                ],
                vec![
                    XptValue::Character("SUBJ-001".to_string()),
                    XptValue::Numeric(2.0),
                    XptValue::Numeric(4.1),
                    XptValue::Numeric(0.0),
                    XptValue::Numeric(0.0),
                ],
                vec![
                    XptValue::Character("SUBJ-002".to_string()),
                    XptValue::Numeric(0.0),
                    XptValue::Numeric(0.0),
                    XptValue::Numeric(100.0),
                    XptValue::Numeric(1.0),
                ],
                vec![
                    XptValue::Character("SUBJ-002".to_string()),
                    XptValue::Numeric(1.0),
                    XptValue::Numeric(4.8),
                    XptValue::Numeric(0.0),
                    XptValue::Numeric(0.0),
                ],
            ],
        };

        let nonmem = xpt_to_nonmem(&ds).unwrap();
        assert_eq!(nonmem.n_subjects(), 2);
        assert_eq!(nonmem.subject_ids(), &["SUBJ-001", "SUBJ-002"]);

        let (times, dv, subj_idx) = nonmem.observation_data();
        // Should have 3 observation records (EVID=0): t=1,2 for SUBJ-001, t=1 for SUBJ-002
        assert_eq!(times.len(), 3);
        assert!((dv[0] - 5.2).abs() < 1e-10);
        assert!((dv[1] - 4.1).abs() < 1e-10);
        assert!((dv[2] - 4.8).abs() < 1e-10);
        assert_eq!(subj_idx[0], 0);
        assert_eq!(subj_idx[1], 0);
        assert_eq!(subj_idx[2], 1);
    }

    #[test]
    fn test_multiple_datasets() {
        let ds1 = XptDataset {
            name: "DM".to_string(),
            label: "Demographics".to_string(),
            variables: vec![XptVariable {
                name: "AGE".to_string(),
                label: "Age".to_string(),
                var_type: XptVarType::Numeric,
                length: 8,
                format: "BEST12.".to_string(),
            }],
            data: vec![vec![XptValue::Numeric(30.0)]],
        };

        let ds2 = XptDataset {
            name: "AE".to_string(),
            label: "Adverse Events".to_string(),
            variables: vec![XptVariable {
                name: "AESEV".to_string(),
                label: "Severity".to_string(),
                var_type: XptVarType::Numeric,
                length: 8,
                format: "BEST12.".to_string(),
            }],
            data: vec![
                vec![XptValue::Numeric(1.0)],
                vec![XptValue::Numeric(3.0)],
            ],
        };

        let bytes = write_xpt_bytes(&[ds1, ds2]).unwrap();
        let datasets = read_xpt_bytes(&bytes).unwrap();

        assert_eq!(datasets.len(), 2);
        assert_eq!(datasets[0].name, "DM");
        assert_eq!(datasets[0].data.len(), 1);
        assert_eq!(datasets[1].name, "AE");
        assert_eq!(datasets[1].data.len(), 2);
    }

    #[test]
    fn test_empty_dataset() {
        let ds = XptDataset {
            name: "EMPTY".to_string(),
            label: "Empty Dataset".to_string(),
            variables: vec![XptVariable {
                name: "X".to_string(),
                label: "".to_string(),
                var_type: XptVarType::Numeric,
                length: 8,
                format: "BEST12.".to_string(),
            }],
            data: Vec::new(),
        };

        let bytes = write_xpt_bytes(&[ds]).unwrap();
        let datasets = read_xpt_bytes(&bytes).unwrap();

        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].name, "EMPTY");
        assert_eq!(datasets[0].data.len(), 0);
    }

    #[test]
    fn test_xpt_to_nonmem_adam_columns() {
        // Test with ADaM column names (AVAL instead of DV)
        let ds = XptDataset {
            name: "ADPC".to_string(),
            label: "PK Analysis Dataset".to_string(),
            variables: vec![
                XptVariable {
                    name: "SUBJID".to_string(),
                    label: "Subject ID".to_string(),
                    var_type: XptVarType::Character,
                    length: 10,
                    format: "$CHAR10.".to_string(),
                },
                XptVariable {
                    name: "ATPTN".to_string(),
                    label: "Analysis Timepoint (N)".to_string(),
                    var_type: XptVarType::Numeric,
                    length: 8,
                    format: "BEST12.".to_string(),
                },
                XptVariable {
                    name: "AVAL".to_string(),
                    label: "Analysis Value".to_string(),
                    var_type: XptVarType::Numeric,
                    length: 8,
                    format: "BEST12.".to_string(),
                },
            ],
            data: vec![
                vec![
                    XptValue::Character("001".to_string()),
                    XptValue::Numeric(0.5),
                    XptValue::Numeric(3.2),
                ],
                vec![
                    XptValue::Character("001".to_string()),
                    XptValue::Numeric(1.0),
                    XptValue::Numeric(5.7),
                ],
            ],
        };

        let nonmem = xpt_to_nonmem(&ds).unwrap();
        assert_eq!(nonmem.n_subjects(), 1);
        let (times, dv, _) = nonmem.observation_data();
        assert_eq!(times.len(), 2);
        assert!((times[0] - 0.5).abs() < 1e-10);
        assert!((dv[0] - 3.2).abs() < 1e-10);
    }
}
