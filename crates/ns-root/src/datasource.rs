//! Memory-mapped or owned data backing for ROOT file reads.

use std::ops::Deref;

/// Backing storage for a ROOT file.
///
/// `Mmap` avoids copying multi-GB ntuple files into RAM;
/// `Owned` is used for `from_bytes()` and testing.
pub enum DataSource {
    /// File bytes owned in a `Vec<u8>`.
    Owned(Vec<u8>),
    /// Memory-mapped file.
    Mmap(memmap2::Mmap),
}

impl Deref for DataSource {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        match self {
            DataSource::Owned(v) => v,
            DataSource::Mmap(m) => m,
        }
    }
}

impl AsRef<[u8]> for DataSource {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self
    }
}
