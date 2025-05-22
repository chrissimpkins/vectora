//! Core logic implementations

use crate::errors::VectorError;

use num::Complex;

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

#[inline]
pub(crate) fn dot_to_f64_impl<T>(a: &[T], b: &[T]) -> f64
where
    T: num::ToPrimitive,
{
    a.iter().zip(b.iter()).map(|(x, y)| x.to_f64().unwrap() * y.to_f64().unwrap()).sum()
}

#[inline]
pub(crate) fn hermitian_dot_impl<N>(a: &[Complex<N>], b: &[Complex<N>]) -> Complex<N>
where
    N: num::Num + Copy + std::iter::Sum<N> + std::ops::Neg<Output = N>,
{
    a.iter().zip(b.iter()).map(|(x, y)| *x * y.conj()).sum()
}

#[inline]
pub(crate) fn cross_impl<T>(a: &[T], b: &[T]) -> [T; 3]
where
    T: num::Num + Copy,
{
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

#[inline]
pub(crate) fn lerp_impl<T>(a: &[T], b: &[T], weight: T, out: &mut [T])
where
    T: Copy
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + num::One,
{
    let one_minus_w = T::one() - weight;
    for ((out_elem, &a_elem), &b_elem) in out.iter_mut().zip(a).zip(b) {
        *out_elem = one_minus_w * a_elem + weight * b_elem;
    }
}

#[inline]
pub(crate) fn mut_lerp_impl<T>(out: &mut [T], end: &[T], weight: T)
where
    T: Copy
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + num::One,
{
    let one_minus_w = T::one() - weight;
    for (a, &b) in out.iter_mut().zip(end) {
        *a = one_minus_w * *a + weight * b;
    }
}

#[inline]
pub(crate) fn distance_impl<T>(a: &[T], b: &[T]) -> T
where
    T: num::Float + Clone + std::iter::Sum<T>,
{
    a.iter().zip(b.iter()).map(|(a, b)| (*a - *b).powi(2)).sum::<T>().sqrt()
}

#[inline]
pub(crate) fn distance_complex_impl<N>(a: &[num::Complex<N>], b: &[num::Complex<N>]) -> N
where
    N: num::Float + Clone + std::iter::Sum<N>,
{
    a.iter().zip(b.iter()).map(|(a, b)| (*a - *b).norm_sqr()).sum::<N>().sqrt()
}

#[inline]
pub(crate) fn manhattan_distance_impl<T>(a: &[T], b: &[T]) -> T
where
    T: num::Float + Clone + std::iter::Sum<T>,
{
    a.iter().zip(b.iter()).map(|(a, b)| (*a - *b).abs()).sum()
}

pub(crate) fn manhattan_distance_complex_impl<N>(a: &[num::Complex<N>], b: &[num::Complex<N>]) -> N
where
    N: num::Float + Clone + std::iter::Sum<N>,
{
    a.iter().zip(b.iter()).map(|(a, b)| (*a - *b).norm()).sum()
}

#[inline]
pub(crate) fn chebyshev_distance_impl<T>(a: &[T], b: &[T]) -> T
where
    T: num::Float + Clone + PartialOrd,
{
    a.iter().zip(b.iter()).map(|(a, b)| (*a - *b).abs()).fold(T::zero(), |acc, x| acc.max(x))
}

#[inline]
pub(crate) fn chebyshev_distance_complex_impl<N>(a: &[num::Complex<N>], b: &[num::Complex<N>]) -> N
where
    N: num::Float + Clone + PartialOrd,
{
    a.iter().zip(b.iter()).map(|(a, b)| (*a - *b).norm()).fold(N::zero(), |acc, x| acc.max(x))
}

#[inline]
pub(crate) fn minkowski_distance_impl<T>(a: &[T], b: &[T], p: T) -> T
where
    T: num::Float + Clone + std::iter::Sum<T>,
{
    a.iter().zip(b.iter()).map(|(a, b)| (*a - *b).abs().powf(p)).sum::<T>().powf(T::one() / p)
}

#[inline]
pub(crate) fn minkowski_distance_complex_impl<N>(
    a: &[num::Complex<N>],
    b: &[num::Complex<N>],
    p: N,
) -> N
where
    N: num::Float + Clone + std::iter::Sum<N>,
{
    a.iter().zip(b.iter()).map(|(a, b)| (*a - *b).norm().powf(p)).sum::<N>().powf(N::one() / p)
}

#[inline]
pub(crate) fn angle_with_impl<T>(a: &[T], b: &[T], norm_a: T, norm_b: T) -> T
where
    T: num::Float + Clone + std::iter::Sum<T>,
{
    let dot = dot_impl(a, b);
    let cos_theta = dot / (norm_a * norm_b);
    let cos_theta = cos_theta.max(-T::one()).min(T::one());
    cos_theta.acos()
}

#[inline]
pub(crate) fn project_onto_impl<T, Out>(b: &[T], scalar: T) -> Out
where
    T: Copy + std::ops::Mul<T, Output = T>,
    Out: std::iter::FromIterator<T>,
{
    b.iter().map(|x| *x * scalar).collect()
}

#[inline]
pub(crate) fn normalize_impl<T, Out>(slice: &[T], norm: T) -> Result<Out, VectorError>
where
    T: Copy + PartialEq + std::ops::Div<T, Output = T> + num::Zero,
    Out: std::iter::FromIterator<T>,
{
    if norm == T::zero() {
        return Err(VectorError::ZeroVectorError("Cannot normalize a zero vector".to_string()));
    }
    Ok(slice.iter().map(|&a| a / norm).collect())
}

#[inline]
pub(crate) fn mut_normalize_impl<T>(slice: &mut [T], norm: T) -> Result<(), VectorError>
where
    T: Copy + PartialEq + std::ops::Div<T, Output = T> + num::Zero,
{
    if norm == T::zero() {
        return Err(VectorError::ZeroVectorError("Cannot normalize a zero vector".to_string()));
    }
    for a in slice.iter_mut() {
        *a = *a / norm;
    }
    Ok(())
}

#[inline]
pub(crate) fn normalize_to_impl<T, Out>(
    slice: &[T],
    norm: T,
    magnitude: T,
) -> Result<Out, VectorError>
where
    T: Copy + PartialEq + std::ops::Div<T, Output = T> + std::ops::Mul<T, Output = T> + num::Zero,
    Out: std::iter::FromIterator<T>,
{
    if norm == T::zero() {
        return Err(VectorError::ZeroVectorError("Cannot normalize a zero vector".to_string()));
    }
    let scale = magnitude / norm;
    Ok(slice.iter().map(|&a| a * scale).collect())
}

#[inline]
pub(crate) fn mut_normalize_to_impl<T>(
    slice: &mut [T],
    norm: T,
    magnitude: T,
) -> Result<(), VectorError>
where
    T: Copy + PartialEq + std::ops::Div<T, Output = T> + std::ops::Mul<T, Output = T> + num::Zero,
{
    if norm == T::zero() {
        return Err(VectorError::ZeroVectorError("Cannot normalize a zero vector".to_string()));
    }
    let scale = magnitude / norm;
    for a in slice.iter_mut() {
        *a = *a * scale;
    }
    Ok(())
}

#[inline]
pub(crate) fn cosine_similarity_impl<T>(a: &[T], b: &[T], norm_a: T, norm_b: T) -> T
where
    T: num::Float + Clone + std::iter::Sum<T> + std::ops::Div<Output = T>,
{
    let dot = dot_impl(a, b);
    dot / (norm_a * norm_b)
}

#[inline]
pub(crate) fn cosine_similarity_complex_impl<N>(
    a: &[num::Complex<N>],
    b: &[num::Complex<N>],
    norm_a: N,
    norm_b: N,
) -> num::Complex<N>
where
    N: num::Float + Clone + std::iter::Sum<N> + std::ops::Neg<Output = N>,
    num::Complex<N>: std::ops::Div<Output = num::Complex<N>>,
{
    let dot = hermitian_dot_impl(a, b);
    dot / num::Complex::new(norm_a * norm_b, N::zero())
}

#[cfg(test)]
mod tests {
    use num::Complex;

    use crate::FlexVector;

    use super::*;

    // -- mut_translate_impl ==
    #[test]
    fn test_mut_translate_impl_basic() {
        let mut out = [1, 2, 3];
        let b = [4, 5, 6];
        mut_translate_impl(&mut out, &b);
        assert_eq!(out, [5, 7, 9]);
    }

    #[test]
    fn test_mut_translate_impl_empty() {
        let mut out: [i32; 0] = [];
        let b: [i32; 0] = [];
        mut_translate_impl(&mut out, &b);
        assert_eq!(out, []);
    }

    #[test]
    fn test_mut_translate_impl_partial() {
        let mut out = [1, 2, 3];
        let b = [10, 20];
        mut_translate_impl(&mut out, &b);
        // Only the first two elements are updated
        assert_eq!(out, [11, 22, 3]);
    }

    // -- translate_impl --
    #[test]
    fn test_translate_impl_basic() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        let mut out = [0; 3];
        translate_impl(&a, &b, &mut out);
        assert_eq!(out, [5, 7, 9]);
    }

    #[test]
    fn test_translate_impl_empty() {
        let a: [i32; 0] = [];
        let b: [i32; 0] = [];
        let mut out: [i32; 0] = [];
        translate_impl(&a, &b, &mut out);
        assert_eq!(out, []);
    }

    #[test]
    fn test_translate_impl_partial() {
        let a = [1, 2, 3];
        let b = [10, 20];
        let mut out = [0; 3];
        translate_impl(&a, &b, &mut out);
        // Only the first two elements are updated
        assert_eq!(out, [11, 22, 3]);
    }

    // -- dot_impl --
    #[test]
    fn test_dot_impl_basic() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        let dot = dot_impl(&a, &b);
        assert_eq!(dot, 1 * 4 + 2 * 5 + 3 * 6);
    }

    #[test]
    fn test_dot_impl_empty() {
        let a: [i32; 0] = [];
        let b: [i32; 0] = [];
        let dot = dot_impl(&a, &b);
        assert_eq!(dot, 0);
    }

    #[test]
    fn test_dot_impl_partial() {
        let a = [1, 2, 3];
        let b = [10, 20];
        let dot = dot_impl(&a, &b);
        // this is not a valid dot product and the reason why this function
        // must be protected with vector dimension checks at compile or run time.
        assert_eq!(dot, 1 * 10 + 2 * 20);
    }

    // -- dot_to_f64_impl --
    #[test]
    fn test_dot_to_f64_impl_basic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let dot = dot_to_f64_impl(&a, &b);
        assert!((dot - (1.0_f64 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0)).abs() < 1e-12);
    }

    #[test]
    fn test_dot_to_f64_impl_empty() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        let dot = dot_to_f64_impl(&a, &b);
        assert_eq!(dot, 0.0_f64);
    }

    #[test]
    fn test_dot_to_f64_impl_partial() {
        let a = [1i32, 2, 3];
        let b = [10i32, 20];
        let dot = dot_to_f64_impl(&a, &b);
        assert_eq!(dot, 1.0_f64 * 10.0 + 2.0 * 20.0);
    }

    #[test]
    fn test_dot_to_f64_impl_mixed_types() {
        let a = [1u8, 2u8, 3u8];
        let b = [4u8, 5u8, 6u8];
        let dot = dot_to_f64_impl(&a, &b);
        assert_eq!(dot, 1.0_f64 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);
    }

    // -- hermitian_dot_impl --
    #[test]
    fn test_hermitian_dot_impl_basic() {
        let a = [Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(5.0, -1.0), Complex::new(-2.0, 2.0)];
        // Hermitian dot: sum_i a_i * conj(b_i)
        // conj(b_0) = 5.0 + 1.0i, conj(b_1) = -2.0 - 2.0i
        // a_0 * conj(b_0) = (1+2i)*(5+1i) = 1*5 + 1*1i + 2i*5 + 2i*1i = 5 + 1i + 10i + 2i^2 = 5 + 11i - 2 = 3 + 11i
        // a_1 * conj(b_1) = (3+4i)*(-2-2i) = 3*-2 + 3*-2i + 4i*-2 + 4i*-2i = -6 -6i -8i -8i^2 = -6 -14i +8 = 2 -14i
        // sum = (3+11i) + (2-14i) = 5 -3i
        let result = super::hermitian_dot_impl(&a, &b);
        assert!((result.re - 5.0).abs() < 1e-12);
        assert!((result.im + 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_hermitian_dot_impl_empty() {
        let a: [Complex<f64>; 0] = [];
        let b: [Complex<f64>; 0] = [];
        let result = super::hermitian_dot_impl(&a, &b);
        assert_eq!(result, Complex::new(0.0, 0.0));
    }

    #[test]
    fn test_hermitian_dot_impl_partial() {
        let a = [Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(5.0, -1.0)];
        // Only the first element is used
        let result = super::hermitian_dot_impl(&a, &b);
        // a_0 * conj(b_0) = (1+2i)*(5+1i) = 3 + 11i (see above)
        assert!((result.re - 3.0).abs() < 1e-12);
        assert!((result.im - 11.0).abs() < 1e-12);
    }

    #[test]
    fn test_hermitian_dot_impl_identical() {
        let a = [Complex::new(2.0_f64, -3.0), Complex::new(-1.0, 4.0)];
        let b = [Complex::new(2.0, -3.0), Complex::new(-1.0, 4.0)];
        // Hermitian dot with self: sum_i a_i * conj(a_i) = sum_i |a_i|^2 (real, >= 0)
        let result = super::hermitian_dot_impl(&a, &b);
        let expected =
            Complex::new((2.0 * 2.0 + (-3.0) * (-3.0)) + ((-1.0) * (-1.0) + 4.0 * 4.0), 0.0);
        assert!((result.re - expected.re).abs() < 1e-12);
        assert!((result.im - expected.im).abs() < 1e-12);
    }

    // -- cross_impl --
    #[test]
    fn test_cross_impl_basic() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        let cross = cross_impl(&a, &b);
        // [2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4] = [12-15, 12-6, 5-8] = [-3, 6, -3]
        assert_eq!(cross, [-3, 6, -3]);
    }

    #[test]
    fn test_cross_impl_zero_vector() {
        let a = [0, 0, 0];
        let b = [1, 2, 3];
        let cross = cross_impl(&a, &b);
        assert_eq!(cross, [0, 0, 0]);
    }

    #[test]
    fn test_cross_impl_parallel_vectors() {
        let a = [1, 2, 3];
        let b = [2, 4, 6];
        let cross = cross_impl(&a, &b);
        assert_eq!(cross, [0, 0, 0]);
    }

    #[test]
    fn test_cross_impl_orthogonal_vectors() {
        let a = [1, 0, 0];
        let b = [0, 1, 0];
        let cross = cross_impl(&a, &b);
        assert_eq!(cross, [0, 0, 1]);
    }

    // -- lerp_impl --
    #[test]
    fn test_lerp_impl_basic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let mut out = [0.0f32; 3];
        lerp_impl(&a, &b, 0.5, &mut out);
        assert_eq!(out, [2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_lerp_impl_weight_zero() {
        let a = [1.0f64, 2.0, 3.0];
        let b = [4.0f64, 5.0, 6.0];
        let mut out = [0.0f64; 3];
        lerp_impl(&a, &b, 0.0, &mut out);
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_lerp_impl_weight_one() {
        let a = [1.0f64, 2.0, 3.0];
        let b = [4.0f64, 5.0, 6.0];
        let mut out = [0.0f64; 3];
        lerp_impl(&a, &b, 1.0, &mut out);
        assert_eq!(out, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_lerp_impl_empty() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        lerp_impl(&a, &b, 0.5, &mut out);
        assert_eq!(out, []);
    }

    #[test]
    fn test_lerp_impl_partial() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0];
        let mut out = [0.0f32; 3];
        lerp_impl(&a, &b, 0.5, &mut out);
        // Only the first two elements are updated
        assert_eq!(out, [2.5, 3.5, 0.0]);
    }

    // -- mut_lerp_impl --
    #[test]
    fn test_mut_lerp_impl_basic() {
        let mut out = [1.0f32, 2.0, 3.0];
        let end = [4.0f32, 5.0, 6.0];
        mut_lerp_impl(&mut out, &end, 0.5);
        assert_eq!(out, [2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_mut_lerp_impl_weight_zero() {
        let mut out = [1.0f64, 2.0, 3.0];
        let end = [4.0f64, 5.0, 6.0];
        mut_lerp_impl(&mut out, &end, 0.0);
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mut_lerp_impl_weight_one() {
        let mut out = [1.0f64, 2.0, 3.0];
        let end = [4.0f64, 5.0, 6.0];
        mut_lerp_impl(&mut out, &end, 1.0);
        assert_eq!(out, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_mut_lerp_impl_empty() {
        let mut out: [f32; 0] = [];
        let end: [f32; 0] = [];
        mut_lerp_impl(&mut out, &end, 0.5);
        assert_eq!(out, []);
    }

    #[test]
    fn test_mut_lerp_impl_partial() {
        let mut out = [1.0f32, 2.0, 3.0];
        let end = [4.0f32, 5.0];
        mut_lerp_impl(&mut out, &end, 0.5);
        // Only the first two elements are updated
        assert_eq!(out, [2.5, 3.5, 3.0]);
    }

    #[test]
    fn test_mut_lerp_impl_complex_basic() {
        let mut out = [Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        let end = [Complex::new(5.0, -1.0), Complex::new(-2.0, 2.0)];
        let weight = Complex::new(0.5, 0.0);
        mut_lerp_impl(&mut out, &end, weight);
        let expected = [
            Complex::new(0.5 * 1.0 + 0.5 * 5.0, 0.5 * 2.0 + 0.5 * -1.0),
            Complex::new(0.5 * 3.0 + 0.5 * -2.0, 0.5 * 4.0 + 0.5 * 2.0),
        ];
        assert!((out[0].re - expected[0].re).abs() < 1e-12);
        assert!((out[0].im - expected[0].im).abs() < 1e-12);
        assert!((out[1].re - expected[1].re).abs() < 1e-12);
        assert!((out[1].im - expected[1].im).abs() < 1e-12);
    }

    // -- distance_impl --
    #[test]
    fn test_distance_impl_basic() {
        let a = [1.0f64, 2.0, 3.0];
        let b = [4.0f64, 6.0, 8.0];
        let dist = distance_impl(&a, &b);
        // sqrt((1-4)^2 + (2-6)^2 + (3-8)^2) = sqrt(9 + 16 + 25) = sqrt(50)
        assert!((dist - 50f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_distance_impl_empty() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        let dist = distance_impl(&a, &b);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_distance_impl_partial() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 6.0];
        let dist = distance_impl(&a, &b);
        // Only the first two elements are used: sqrt((1-4)^2 + (2-6)^2) = sqrt(9 + 16) = 5
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_impl_identical() {
        let a = [1.23f64, 4.56, 7.89];
        let b = [1.23f64, 4.56, 7.89];
        let dist = distance_impl(&a, &b);
        assert_eq!(dist, 0.0);
    }

    // -- distance_complex_impl --
    #[test]
    fn test_distance_complex_impl_basic() {
        use num::Complex;
        let a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(4.0, 6.0), Complex::new(8.0, 8.0)];
        // sqrt(|1+2i - 4+6i|^2 + |3+4i - 8+8i|^2)
        // = sqrt(|-3-4i|^2 + |-5-4i|^2)
        // = sqrt((3^2+4^2) + (5^2+4^2)) = sqrt(25 + 41) = sqrt(66)
        let dist = distance_complex_impl(&a, &b);
        assert!((dist - 66f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_distance_complex_impl_empty() {
        use num::Complex;
        let a: [Complex<f64>; 0] = [];
        let b: [Complex<f64>; 0] = [];
        let dist = distance_complex_impl(&a, &b);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_distance_complex_impl_partial() {
        use num::Complex;
        let a = [Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(4.0, 6.0)];
        // Only the first element is used: sqrt(|-3-4i|^2) = sqrt(25) = 5
        let dist = distance_complex_impl(&a, &b);
        assert!((dist - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_distance_complex_impl_identical() {
        use num::Complex;
        let a = [Complex::new(1.23, 4.56), Complex::new(7.89, 0.12)];
        let b = [Complex::new(1.23, 4.56), Complex::new(7.89, 0.12)];
        let dist = distance_complex_impl(&a, &b);
        assert_eq!(dist, 0.0);
    }

    // -- manhattan_distance_impl --
    #[test]
    fn test_manhattan_distance_impl_basic() {
        let a = [1.0f64, 2.0, 3.0];
        let b = [4.0f64, 6.0, 8.0];
        let dist = manhattan_distance_impl(&a, &b);
        // |1-4| + |2-6| + |3-8| = 3 + 4 + 5 = 12
        assert!((dist - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_manhattan_distance_impl_empty() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        let dist = manhattan_distance_impl(&a, &b);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_manhattan_distance_impl_partial() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 6.0];
        let dist = manhattan_distance_impl(&a, &b);
        // Only the first two elements are used: |1-4| + |2-6| = 3 + 4 = 7
        assert!((dist - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance_impl_identical() {
        let a = [1.23f64, 4.56, 7.89];
        let b = [1.23f64, 4.56, 7.89];
        let dist = manhattan_distance_impl(&a, &b);
        assert_eq!(dist, 0.0);
    }

    // -- manhattan_distance_complex_impl --
    #[test]
    fn test_manhattan_distance_complex_impl_basic() {
        use num::Complex;
        let a = [Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(4.0, 6.0), Complex::new(8.0, 8.0)];
        // |1+2i - 4+6i| + |3+4i - 8+8i| = |-3-4i| + |-5-4i| = 5 + sqrt(41)
        let dist = manhattan_distance_complex_impl(&a, &b);
        assert!((dist - (5.0 + 41f64.sqrt())).abs() < 1e-12);
    }

    #[test]
    fn test_manhattan_distance_complex_impl_empty() {
        use num::Complex;
        let a: [Complex<f64>; 0] = [];
        let b: [Complex<f64>; 0] = [];
        let dist = manhattan_distance_complex_impl(&a, &b);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_manhattan_distance_complex_impl_partial() {
        use num::Complex;
        let a = [Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(4.0, 6.0)];
        // Only the first element is used: |-3-4i| = 5
        let dist = manhattan_distance_complex_impl(&a, &b);
        assert!((dist - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_manhattan_distance_complex_impl_identical() {
        use num::Complex;
        let a = [Complex::new(1.23_f64, 4.56), Complex::new(7.89, 0.12)];
        let b = [Complex::new(1.23, 4.56), Complex::new(7.89, 0.12)];
        let dist = manhattan_distance_complex_impl(&a, &b);
        assert_eq!(dist, 0.0);
    }

    // -- chebyshev_distance_impl --
    #[test]
    fn test_chebyshev_distance_impl_basic() {
        let a = [1.0f64, 2.0, 3.0];
        let b = [4.0f64, 6.0, 8.0];
        let dist = chebyshev_distance_impl(&a, &b);
        // max(|1-4|, |2-6|, |3-8|) = max(3, 4, 5) = 5
        assert!((dist - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_chebyshev_distance_impl_empty() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        let dist = chebyshev_distance_impl(&a, &b);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_chebyshev_distance_impl_partial() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 6.0];
        let dist = chebyshev_distance_impl(&a, &b);
        // Only the first two elements are used: max(|1-4|, |2-6|) = max(3, 4) = 4
        assert!((dist - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_chebyshev_distance_impl_identical() {
        let a = [1.23f64, 4.56, 7.89];
        let b = [1.23f64, 4.56, 7.89];
        let dist = chebyshev_distance_impl(&a, &b);
        assert_eq!(dist, 0.0);
    }

    // -- chebyshev_distance_complex_impl --
    #[test]
    fn test_chebyshev_distance_complex_impl_basic() {
        use num::Complex;
        let a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(4.0, 6.0), Complex::new(8.0, 8.0)];
        // max(|1+2i - 4+6i|, |3+4i - 8+8i|) = max(|-3-4i|, |-5-4i|) = max(5, sqrt(41))
        let dist = chebyshev_distance_complex_impl(&a, &b);
        assert!((dist - 41f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_chebyshev_distance_complex_impl_empty() {
        use num::Complex;
        let a: [Complex<f64>; 0] = [];
        let b: [Complex<f64>; 0] = [];
        let dist = chebyshev_distance_complex_impl(&a, &b);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_chebyshev_distance_complex_impl_partial() {
        use num::Complex;
        let a = [Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(4.0, 6.0)];
        // Only the first element is used: |-3-4i| = 5
        let dist = chebyshev_distance_complex_impl(&a, &b);
        assert!((dist - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_chebyshev_distance_complex_impl_identical() {
        use num::Complex;
        let a = [Complex::new(1.23, 4.56), Complex::new(7.89, 0.12)];
        let b = [Complex::new(1.23, 4.56), Complex::new(7.89, 0.12)];
        let dist = chebyshev_distance_complex_impl(&a, &b);
        assert_eq!(dist, 0.0);
    }

    // -- minkowski_distance_impl --
    #[test]
    fn test_minkowski_distance_impl_basic() {
        let a = [1.0f64, 2.0, 3.0];
        let b = [4.0f64, 6.0, 8.0];
        let p = 3.0;
        let dist = minkowski_distance_impl(&a, &b, p);
        // ((|1-4|^3 + |2-6|^3 + |3-8|^3))^(1/3) = (27 + 64 + 125)^(1/3) = (216)^(1/3) = 6
        assert!((dist - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_minkowski_distance_impl_p1() {
        let a = [1.0f64, 2.0, 3.0];
        let b = [4.0f64, 6.0, 8.0];
        let dist = minkowski_distance_impl(&a, &b, 1.0);
        // Should match manhattan distance: 3 + 4 + 5 = 12
        assert!((dist - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_minkowski_distance_impl_p2() {
        let a = [1.0f64, 2.0, 3.0];
        let b = [4.0f64, 6.0, 8.0];
        let dist = minkowski_distance_impl(&a, &b, 2.0);
        // Should match euclidean distance: sqrt(9 + 16 + 25) = sqrt(50)
        assert!((dist - 50f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_minkowski_distance_impl_empty() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        let dist = minkowski_distance_impl(&a, &b, 2.0);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_minkowski_distance_impl_partial() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 6.0];
        let dist = minkowski_distance_impl(&a, &b, 2.0);
        // Only the first two elements: sqrt((1-4)^2 + (2-6)^2) = sqrt(9 + 16) = 5
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_minkowski_distance_impl_identical() {
        let a = [1.23f64, 4.56, 7.89];
        let b = [1.23f64, 4.56, 7.89];
        let dist = minkowski_distance_impl(&a, &b, 2.0);
        assert_eq!(dist, 0.0);
    }

    // -- minkowski_distance_complex_impl --
    #[test]
    fn test_minkowski_distance_complex_impl_basic() {
        use num::Complex;
        let a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(4.0, 6.0), Complex::new(8.0, 8.0)];
        let p = 3.0;
        // ((|-3-4i|^3 + |-5-4i|^3))^(1/3) = (5^3 + 41.sqrt()^3)^(1/3)
        let expected = (5.0_f64.powf(3.0) + 41f64.sqrt().powf(3.0)).powf(1.0 / 3.0);
        let dist = minkowski_distance_complex_impl(&a, &b, p);
        assert!((dist - expected).abs() < 1e-12);
    }

    #[test]
    fn test_minkowski_distance_complex_impl_p1() {
        use num::Complex;
        let a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(4.0, 6.0), Complex::new(8.0, 8.0)];
        // Should match manhattan distance: 5 + sqrt(41)
        let dist = minkowski_distance_complex_impl(&a, &b, 1.0);
        assert!((dist - (5.0 + 41f64.sqrt())).abs() < 1e-12);
    }

    #[test]
    fn test_minkowski_distance_complex_impl_p2() {
        use num::Complex;
        let a = [Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(4.0, 6.0), Complex::new(8.0, 8.0)];
        // Should match euclidean distance: sqrt(25 + 41) = sqrt(66)
        let dist = minkowski_distance_complex_impl(&a, &b, 2.0);
        assert!((dist - 66f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_minkowski_distance_complex_impl_empty() {
        use num::Complex;
        let a: [Complex<f64>; 0] = [];
        let b: [Complex<f64>; 0] = [];
        let dist = minkowski_distance_complex_impl(&a, &b, 2.0);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_minkowski_distance_complex_impl_partial() {
        use num::Complex;
        let a = [Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(4.0, 6.0)];
        // Only the first element is used: |-3-4i| = 5
        let dist = minkowski_distance_complex_impl(&a, &b, 2.0);
        assert!((dist - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_minkowski_distance_complex_impl_identical() {
        use num::Complex;
        let a = [Complex::new(1.23, 4.56), Complex::new(7.89, 0.12)];
        let b = [Complex::new(1.23, 4.56), Complex::new(7.89, 0.12)];
        let dist = minkowski_distance_complex_impl(&a, &b, 2.0);
        assert_eq!(dist, 0.0);
    }

    // -- angle_with_impl --
    #[test]
    fn test_angle_with_impl_orthogonal() {
        let a = [1.0f64, 0.0];
        let b = [0.0f64, 1.0];
        let norm_a = (a[0].powi(2) + a[1].powi(2)).sqrt();
        let norm_b = (b[0].powi(2) + b[1].powi(2)).sqrt();
        let angle = angle_with_impl(&a, &b, norm_a, norm_b);
        // Orthogonal vectors: angle should be pi/2
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn test_angle_with_impl_parallel() {
        let a = [1.0f64, 2.0];
        let b = [2.0f64, 4.0];
        let norm_a = (a[0].powi(2) + a[1].powi(2)).sqrt();
        let norm_b = (b[0].powi(2) + b[1].powi(2)).sqrt();
        let angle = angle_with_impl(&a, &b, norm_a, norm_b);
        // Parallel vectors: angle should be 0 (allow for floating-point error)
        assert!(angle.abs() < 1e-7, "angle was {}", angle);
    }

    #[test]
    fn test_angle_with_impl_opposite() {
        let a = [1.0f64, 0.0];
        let b = [-1.0f64, 0.0];
        let norm_a = (a[0].powi(2) + a[1].powi(2)).sqrt();
        let norm_b = (b[0].powi(2) + b[1].powi(2)).sqrt();
        let angle = angle_with_impl(&a, &b, norm_a, norm_b);
        // Opposite vectors: angle should be pi
        assert!((angle - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn test_angle_with_impl_identical() {
        let a = [3.0f64, 4.0];
        let b = [3.0f64, 4.0];
        let norm_a = (a[0].powi(2) + a[1].powi(2)).sqrt();
        let norm_b = (b[0].powi(2) + b[1].powi(2)).sqrt();
        let angle = angle_with_impl(&a, &b, norm_a, norm_b);
        // Identical vectors: angle should be 0
        assert!(angle.abs() < 1e-12);
    }

    #[test]
    fn test_angle_with_impl_arbitrary() {
        let a = [1.0f64, 2.0];
        let b = [2.0f64, 1.0];
        let norm_a = (a[0].powi(2) + a[1].powi(2)).sqrt();
        let norm_b = (b[0].powi(2) + b[1].powi(2)).sqrt();
        let angle = angle_with_impl(&a, &b, norm_a, norm_b);
        // Check that the angle is between 0 and pi
        assert!(angle > 0.0 && angle < std::f64::consts::PI);
    }

    // -- project_onto_impl --
    #[test]
    fn test_project_onto_impl_basic() {
        let b = [1.0f64, 0.0];
        let scalar = 3.0;
        let proj: FlexVector<f64> = project_onto_impl(&b, scalar);
        // Projection of [3,4] onto [1,0] is [3,0], so scalar = 3, b = [1,0]
        assert!((proj[0] - 3.0).abs() < 1e-12);
        assert!((proj[1] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_project_onto_impl_parallel() {
        let b = [1.0f64, 2.0];
        let scalar = 2.0;
        let proj: FlexVector<f64> = project_onto_impl(&b, scalar);
        // scalar = 2, b = [1,2] => [2,4]
        assert!((proj[0] - 2.0).abs() < 1e-12);
        assert!((proj[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_project_onto_impl_orthogonal() {
        let b = [1.0f64, 0.0];
        let scalar = 0.0;
        let proj: FlexVector<f64> = project_onto_impl(&b, scalar);
        // scalar = 0, so projection is [0,0]
        assert!((proj[0] - 0.0).abs() < 1e-12);
        assert!((proj[1] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_project_onto_impl_identical() {
        let b = [5.0f64, 5.0];
        let scalar = 1.0;
        let proj: FlexVector<f64> = project_onto_impl(&b, scalar);
        // scalar = 1, so projection is b itself
        assert!((proj[0] - 5.0).abs() < 1e-12);
        assert!((proj[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_project_onto_impl_zero_vector() {
        let b = [0.0f64, 0.0];
        let scalar = 42.0;
        let proj: FlexVector<f64> = project_onto_impl(&b, scalar);
        // b is zero vector, so projection is [0,0]
        assert!((proj[0] - 0.0).abs() < 1e-12);
        assert!((proj[1] - 0.0).abs() < 1e-12);
    }

    // -- normalize_impl --
    #[test]
    fn test_normalize_impl_f64_basic() {
        let v = [3.0f64, 4.0];
        let norm = (3.0f64 * 3.0 + 4.0 * 4.0).sqrt();
        let result: FlexVector<f64> = normalize_impl(&v, norm).unwrap();
        assert!((result[0] - 0.6).abs() < 1e-12);
        assert!((result[1] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_normalize_impl_f64_zero_vector() {
        let v = [0.0f64, 0.0];
        let norm = 0.0f64;
        let result: Result<FlexVector<f64>, _> = normalize_impl(&v, norm);
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_impl_f64_single_element() {
        let v = [5.0f64];
        let norm = 5.0f64;
        let result: FlexVector<f64> = normalize_impl(&v, norm).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_normalize_impl_f64_empty() {
        let v: [f64; 0] = [];
        let norm = 1.0f64;
        let result: FlexVector<f64> = normalize_impl(&v, norm).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_normalize_impl_complex_basic() {
        let v = [Complex::new(3.0, 4.0)];
        let norm = Complex::new((3.0f64 * 3.0 + 4.0 * 4.0).sqrt(), 0.0);
        let result: FlexVector<Complex<f64>> = normalize_impl(&v, norm).unwrap();
        assert!((result[0].re - 0.6).abs() < 1e-12);
        assert!((result[0].im - 0.8).abs() < 1e-12);
    }

    // -- mut_normalize_impl --
    #[test]
    fn test_mut_normalize_impl_f64_basic() {
        let mut v = [3.0f64, 4.0];
        let norm = (3.0f64 * 3.0 + 4.0 * 4.0).sqrt();
        mut_normalize_impl(&mut v, norm).unwrap();
        assert!((v[0] - 0.6).abs() < 1e-12);
        assert!((v[1] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_mut_normalize_impl_f64_zero_vector() {
        let mut v = [0.0f64, 0.0];
        let norm = 0.0f64;
        let result = mut_normalize_impl(&mut v, norm);
        assert!(result.is_err());
    }

    #[test]
    fn test_mut_normalize_impl_f64_single_element() {
        let mut v = [5.0f64];
        let norm = 5.0f64;
        mut_normalize_impl(&mut v, norm).unwrap();
        assert!((v[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mut_normalize_impl_f64_empty() {
        let mut v: [f64; 0] = [];
        let norm = 1.0f64;
        mut_normalize_impl(&mut v, norm).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn test_mut_normalize_impl_complex_basic() {
        use num::Complex;
        let mut v = [Complex::new(3.0, 4.0)];
        let norm = Complex::new((3.0f64 * 3.0 + 4.0 * 4.0).sqrt(), 0.0);
        mut_normalize_impl(&mut v, norm).unwrap();
        assert!((v[0].re - 0.6).abs() < 1e-12);
        assert!((v[0].im - 0.8).abs() < 1e-12);
    }

    // -- normalize_to_impl --
    #[test]
    fn test_normalize_to_impl_f64_basic() {
        let v = [3.0f64, 4.0];
        let norm = (3.0f64 * 3.0 + 4.0 * 4.0).sqrt();
        let magnitude = 10.0f64;
        let result: FlexVector<f64> = normalize_to_impl(&v, norm, magnitude).unwrap();
        // The normalized vector should have the same direction as v and norm 10
        let expected = [3.0 / 5.0 * 10.0, 4.0 / 5.0 * 10.0];
        assert!((result[0] - expected[0]).abs() < 1e-12);
        assert!((result[1] - expected[1]).abs() < 1e-12);
    }

    #[test]
    fn test_normalize_to_impl_f64_zero_vector() {
        let v = [0.0f64, 0.0];
        let norm = 0.0f64;
        let magnitude = 10.0f64;
        let result: Result<FlexVector<f64>, _> = normalize_to_impl(&v, norm, magnitude);
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_to_impl_f64_single_element() {
        let v = [5.0f64];
        let norm = 5.0f64;
        let magnitude = 2.0f64;
        let result: FlexVector<f64> = normalize_to_impl(&v, norm, magnitude).unwrap();
        assert!((result[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_normalize_to_impl_f64_empty() {
        let v: [f64; 0] = [];
        let norm = 1.0f64;
        let magnitude = 2.0f64;
        let result: FlexVector<f64> = normalize_to_impl(&v, norm, magnitude).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_normalize_to_impl_complex_basic() {
        let v = [Complex::new(3.0, 4.0)];
        let norm = Complex::new((3.0f64 * 3.0 + 4.0 * 4.0).sqrt(), 0.0);
        let magnitude = Complex::new(10.0, 0.0);
        let result: FlexVector<Complex<f64>> = normalize_to_impl(&v, norm, magnitude).unwrap();
        let expected = Complex::new(3.0 / 5.0 * 10.0, 4.0 / 5.0 * 10.0);
        assert!((result[0].re - expected.re).abs() < 1e-12);
        assert!((result[0].im - expected.im).abs() < 1e-12);
    }

    // -- mut_normalize_to_impl --
    #[test]
    fn test_mut_normalize_to_impl_f64_basic() {
        let mut v = [3.0f64, 4.0];
        let norm = (3.0f64 * 3.0 + 4.0 * 4.0).sqrt();
        let magnitude = 10.0f64;
        mut_normalize_to_impl(&mut v, norm, magnitude).unwrap();
        // The normalized vector should have the same direction as v and norm 10
        let expected = [3.0 / 5.0 * 10.0, 4.0 / 5.0 * 10.0];
        assert!((v[0] - expected[0]).abs() < 1e-12);
        assert!((v[1] - expected[1]).abs() < 1e-12);
    }

    #[test]
    fn test_mut_normalize_to_impl_f64_zero_vector() {
        let mut v = [0.0f64, 0.0];
        let norm = 0.0f64;
        let magnitude = 10.0f64;
        let result = mut_normalize_to_impl(&mut v, norm, magnitude);
        assert!(result.is_err());
    }

    #[test]
    fn test_mut_normalize_to_impl_f64_single_element() {
        let mut v = [5.0f64];
        let norm = 5.0f64;
        let magnitude = 2.0f64;
        mut_normalize_to_impl(&mut v, norm, magnitude).unwrap();
        assert!((v[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_mut_normalize_to_impl_f64_empty() {
        let mut v: [f64; 0] = [];
        let norm = 1.0f64;
        let magnitude = 2.0f64;
        mut_normalize_to_impl(&mut v, norm, magnitude).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn test_mut_normalize_to_impl_complex_basic() {
        use num::Complex;
        let mut v = [Complex::new(3.0, 4.0)];
        let norm = Complex::new((3.0f64 * 3.0 + 4.0 * 4.0).sqrt(), 0.0);
        let magnitude = Complex::new(10.0, 0.0);
        mut_normalize_to_impl(&mut v, norm, magnitude).unwrap();
        let expected = Complex::new(3.0 / 5.0 * 10.0, 4.0 / 5.0 * 10.0);
        assert!((v[0].re - expected.re).abs() < 1e-12);
        assert!((v[0].im - expected.im).abs() < 1e-12);
    }

    // -- cosine_similarity_impl --
    #[test]
    fn test_cosine_similarity_impl_basic() {
        let a = [1.0f64, 0.0];
        let b = [0.0f64, 1.0];
        let norm_a = (a[0].powi(2) + a[1].powi(2)).sqrt();
        let norm_b = (b[0].powi(2) + b[1].powi(2)).sqrt();
        let cos_sim = cosine_similarity_impl(&a, &b, norm_a, norm_b);
        // Orthogonal vectors: cosine similarity should be 0
        assert!((cos_sim - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_similarity_impl_parallel() {
        let a = [1.0f64, 2.0];
        let b = [2.0f64, 4.0];
        let norm_a = (a[0].powi(2) + a[1].powi(2)).sqrt();
        let norm_b = (b[0].powi(2) + b[1].powi(2)).sqrt();
        let cos_sim = cosine_similarity_impl(&a, &b, norm_a, norm_b);
        // Parallel vectors: cosine similarity should be 1
        assert!((cos_sim - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_similarity_impl_opposite() {
        let a = [1.0f64, 0.0];
        let b = [-1.0f64, 0.0];
        let norm_a = (a[0].powi(2) + a[1].powi(2)).sqrt();
        let norm_b = (b[0].powi(2) + b[1].powi(2)).sqrt();
        let cos_sim = cosine_similarity_impl(&a, &b, norm_a, norm_b);
        // Opposite vectors: cosine similarity should be -1
        assert!((cos_sim + 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_similarity_impl_identical() {
        let a = [3.0f64, 4.0];
        let b = [3.0f64, 4.0];
        let norm_a = (a[0].powi(2) + a[1].powi(2)).sqrt();
        let norm_b = (b[0].powi(2) + b[1].powi(2)).sqrt();
        let cos_sim = cosine_similarity_impl(&a, &b, norm_a, norm_b);
        // Identical vectors: cosine similarity should be 1
        assert!((cos_sim - 1.0).abs() < 1e-12);
    }

    // -- cosine_similarity_complex_impl --
    #[test]
    fn test_cosine_similarity_complex_impl_basic() {
        use num::Complex;
        let a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(5.0, -1.0), Complex::new(-2.0, 2.0)];
        let norm_a = a.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        let norm_b = b.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        let cos_sim = cosine_similarity_complex_impl(&a, &b, norm_a, norm_b);
        // Compare to direct calculation
        let dot = hermitian_dot_impl(&a, &b);
        let expected = dot / Complex::new(norm_a * norm_b, 0.0);
        assert!((cos_sim.re - expected.re).abs() < 1e-12);
        assert!((cos_sim.im - expected.im).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_similarity_complex_impl_identical() {
        use num::Complex;
        let a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let norm_a = a.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        let cos_sim = cosine_similarity_complex_impl(&a, &a, norm_a, norm_a);
        // For identical vectors, cosine similarity should be 1+0i
        assert!((cos_sim.re - 1.0).abs() < 1e-12);
        assert!(cos_sim.im.abs() < 1e-12);
    }

    #[test]
    fn test_cosine_similarity_complex_impl_zero_vector() {
        use num::Complex;
        let a = [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let b = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let norm_a = a.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        let norm_b = b.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        // This will produce NaN due to division by zero, but we check for zero norm in the FlexVector impl.
        let cos_sim = cosine_similarity_complex_impl(&a, &b, norm_a, norm_b);
        assert!(cos_sim.re.is_nan() && cos_sim.im.is_nan());
    }
}
