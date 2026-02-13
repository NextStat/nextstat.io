//! NextStat-optimized Zstandard decoder/encoder (fork of ruzstd).
//!
//! Based on ruzstd 0.8.2 by Moritz Borcherding.
//! Optimizations: static lookup tables, pre-sized Huffman decode, inlined hot paths,
//! removed #[cold] from BitReaderReversed::refill.
#![no_std]
#![deny(trivial_casts, trivial_numeric_casts, rust_2018_idioms)]
#![allow(unused_mut)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

#[cfg(feature = "std")]
pub(crate) const VERBOSE: bool = false;

macro_rules! vprintln {
    ($($x:expr),*) => {
        #[cfg(feature = "std")]
        if crate::VERBOSE {
            std::println!($($x),*);
        }
    }
}

mod bit_io;
mod common;
pub mod decoding;
pub mod encoding;

pub(crate) mod blocks;
pub(crate) mod fse;
pub(crate) mod huff0;

#[cfg(feature = "std")]
pub mod io_std;

#[cfg(feature = "std")]
pub use io_std as io;

#[cfg(not(feature = "std"))]
pub mod io_nostd;

#[cfg(not(feature = "std"))]
pub use io_nostd as io;
