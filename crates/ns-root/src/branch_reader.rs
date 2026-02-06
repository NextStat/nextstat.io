//! Column-oriented data extraction from TTree branches.

use rayon::prelude::*;

use crate::basket::read_basket_data;
use crate::error::{Result, RootError};
use crate::tree::{BranchInfo, LeafType};

/// Reader for extracting column data from a TTree branch.
pub struct BranchReader<'a> {
    file_data: &'a [u8],
    branch: &'a BranchInfo,
    is_large: bool,
}

impl<'a> BranchReader<'a> {
    /// Create a new branch reader.
    pub fn new(file_data: &'a [u8], branch: &'a BranchInfo, is_large: bool) -> Self {
        Self { file_data, branch, is_large }
    }

    /// Read all entries as `f64`, converting from the native type.
    pub fn as_f64(&self) -> Result<Vec<f64>> {
        let raw_baskets = self.read_all_baskets()?;
        decode_as_f64(&raw_baskets, self.branch.leaf_type)
    }

    /// Read all entries as `f64` using parallel basket decompression.
    pub fn as_f64_par(&self) -> Result<Vec<f64>> {
        let raw_baskets = self.read_all_baskets_par()?;
        decode_as_f64(&raw_baskets, self.branch.leaf_type)
    }

    /// Read all entries as `f32`.
    pub fn as_f32(&self) -> Result<Vec<f32>> {
        let raw_baskets = self.read_all_baskets()?;
        decode_as_f32(&raw_baskets, self.branch.leaf_type)
    }

    /// Read all entries as `i32`.
    pub fn as_i32(&self) -> Result<Vec<i32>> {
        let raw_baskets = self.read_all_baskets()?;
        decode_as_i32(&raw_baskets, self.branch.leaf_type)
    }

    /// Read all entries as `i64`.
    pub fn as_i64(&self) -> Result<Vec<i64>> {
        let raw_baskets = self.read_all_baskets()?;
        decode_as_i64(&raw_baskets, self.branch.leaf_type)
    }

    /// Read and decompress all baskets sequentially.
    fn read_all_baskets(&self) -> Result<Vec<Vec<u8>>> {
        let mut baskets = Vec::with_capacity(self.branch.n_baskets);
        for i in 0..self.branch.n_baskets {
            let data = read_basket_data(self.file_data, self.branch.basket_seek[i], self.is_large)?;
            baskets.push(data);
        }
        Ok(baskets)
    }

    /// Read and decompress all baskets in parallel via rayon.
    fn read_all_baskets_par(&self) -> Result<Vec<Vec<u8>>> {
        let results: Vec<Result<Vec<u8>>> = (0..self.branch.n_baskets)
            .into_par_iter()
            .map(|i| read_basket_data(self.file_data, self.branch.basket_seek[i], self.is_large))
            .collect();

        results.into_iter().collect()
    }
}

// ── Decoding big-endian baskets to typed arrays ────────────────

fn decode_as_f64(baskets: &[Vec<u8>], leaf_type: LeafType) -> Result<Vec<f64>> {
    let elem_size = leaf_type.byte_size();
    let mut out = Vec::new();

    for basket in baskets {
        let data = basket.as_slice();
        // Number of elements based on element size
        let n = data.len() / elem_size;

        for i in 0..n {
            let offset = i * elem_size;
            let val = match leaf_type {
                LeafType::F64 => f64::from_be_bytes(data[offset..offset + 8].try_into().unwrap()),
                LeafType::F32 => {
                    f32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as f64
                }
                LeafType::I32 => {
                    i32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as f64
                }
                LeafType::I64 => {
                    i64::from_be_bytes(data[offset..offset + 8].try_into().unwrap()) as f64
                }
                LeafType::U32 => {
                    u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as f64
                }
                LeafType::U64 => {
                    u64::from_be_bytes(data[offset..offset + 8].try_into().unwrap()) as f64
                }
                LeafType::I16 => {
                    i16::from_be_bytes(data[offset..offset + 2].try_into().unwrap()) as f64
                }
                LeafType::I8 => data[offset] as i8 as f64,
                LeafType::Bool => {
                    if data[offset] != 0 {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
            out.push(val);
        }
    }

    Ok(out)
}

fn decode_as_f32(baskets: &[Vec<u8>], leaf_type: LeafType) -> Result<Vec<f32>> {
    let f64s = decode_as_f64(baskets, leaf_type)?;
    Ok(f64s.into_iter().map(|v| v as f32).collect())
}

fn decode_as_i32(baskets: &[Vec<u8>], leaf_type: LeafType) -> Result<Vec<i32>> {
    let elem_size = leaf_type.byte_size();
    let mut out = Vec::new();

    for basket in baskets {
        let data = basket.as_slice();
        let n = data.len() / elem_size;

        for i in 0..n {
            let offset = i * elem_size;
            let val = match leaf_type {
                LeafType::I32 => i32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()),
                LeafType::I16 => {
                    i16::from_be_bytes(data[offset..offset + 2].try_into().unwrap()) as i32
                }
                LeafType::I8 => data[offset] as i8 as i32,
                LeafType::Bool => {
                    if data[offset] != 0 {
                        1
                    } else {
                        0
                    }
                }
                other => {
                    return Err(RootError::TypeMismatch(format!("cannot read {:?} as i32", other)));
                }
            };
            out.push(val);
        }
    }

    Ok(out)
}

fn decode_as_i64(baskets: &[Vec<u8>], leaf_type: LeafType) -> Result<Vec<i64>> {
    let elem_size = leaf_type.byte_size();
    let mut out = Vec::new();

    for basket in baskets {
        let data = basket.as_slice();
        let n = data.len() / elem_size;

        for i in 0..n {
            let offset = i * elem_size;
            let val = match leaf_type {
                LeafType::I64 => i64::from_be_bytes(data[offset..offset + 8].try_into().unwrap()),
                LeafType::I32 => {
                    i32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as i64
                }
                LeafType::I16 => {
                    i16::from_be_bytes(data[offset..offset + 2].try_into().unwrap()) as i64
                }
                LeafType::I8 => data[offset] as i8 as i64,
                LeafType::Bool => {
                    if data[offset] != 0 {
                        1
                    } else {
                        0
                    }
                }
                other => {
                    return Err(RootError::TypeMismatch(format!("cannot read {:?} as i64", other)));
                }
            };
            out.push(val);
        }
    }

    Ok(out)
}
