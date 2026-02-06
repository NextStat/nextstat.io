use nalgebra::DMatrix;

/// Natural log of `2*pi` as an f64 constant.
///
/// We keep this as a literal because `ln()` is not a `const fn` on stable Rust.
pub(super) const LN_2PI: f64 = 1.837_877_066_409_345_3;

#[inline]
pub(super) fn symmetrize(p: &DMatrix<f64>) -> DMatrix<f64> {
    0.5 * (p + p.transpose())
}
