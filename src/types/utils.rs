//! Crate private core logic implementations

#[inline]
pub(crate) fn mut_translate_impl<T: num::Num + Copy>(out: &mut [T], b: &[T]) {
    for (out_elem, &b_elem) in out.iter_mut().zip(b) {
        *out_elem = *out_elem + b_elem;
    }
}

#[inline]
pub(crate) fn translate_impl<T: num::Num + Copy>(a: &[T], b: &[T], out: &mut [T]) {
    // Copy a into out, then add b in-place
    out.copy_from_slice(a);
    mut_translate_impl(out, b);
}

#[inline]
pub(crate) fn dot_impl<T>(a: &[T], b: &[T]) -> T
where
    T: num::Num + Copy + std::iter::Sum<T>,
{
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
}
