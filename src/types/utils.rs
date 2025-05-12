//! Core logic implementations

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
pub(crate) fn cross_impl<T>(a: &[T], b: &[T]) -> [T; 3]
where
    T: num::Num + Copy,
{
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

#[inline]
pub(crate) fn lerp_impl<T>(a: &[T], b: &[T], weight: T, out: &mut [T])
where
    T: num::Float + Copy,
{
    out.copy_from_slice(a);
    mut_lerp_impl(out, b, weight);
}

#[inline]
pub(crate) fn mut_lerp_impl<T>(out: &mut [T], end: &[T], weight: T)
where
    T: num::Float + Copy,
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
pub(crate) fn manhattan_distance_impl<T>(a: &[T], b: &[T]) -> T
where
    T: num::Float + Clone + std::iter::Sum<T>,
{
    a.iter().zip(b.iter()).map(|(a, b)| (*a - *b).abs()).sum()
}

#[inline]
pub(crate) fn chebyshev_distance_impl<T>(a: &[T], b: &[T]) -> T
where
    T: num::Float + Clone + PartialOrd,
{
    a.iter().zip(b.iter()).map(|(a, b)| (*a - *b).abs()).fold(T::zero(), |acc, x| acc.max(x))
}

#[inline]
pub(crate) fn minkowski_distance_impl<T>(a: &[T], b: &[T], p: T) -> T
where
    T: num::Float + Clone + std::iter::Sum<T>,
{
    a.iter().zip(b.iter()).map(|(a, b)| (*a - *b).abs().powf(p)).sum::<T>().powf(T::one() / p)
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
pub(crate) fn project_onto_impl<T, Out>(a: &[T], b: &[T], denom: T) -> Out
where
    T: num::Float + Clone + std::iter::Sum<T>,
    Out: std::iter::FromIterator<T>,
{
    let scalar = dot_impl(a, b) / denom;
    b.iter().map(|x| *x * scalar).collect()
}

#[cfg(test)]
mod tests {
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
        assert_eq!(out, [2.5, 3.5, 3.0]);
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
        let a = [3.0f64, 4.0];
        let b = [1.0f64, 0.0];
        let denom = dot_impl(&b, &b);
        let proj: Vec<f64> = project_onto_impl(&a, &b, denom);
        // Projection of [3,4] onto [1,0] is [3,0]
        assert!((proj[0] - 3.0).abs() < 1e-12);
        assert!((proj[1] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_project_onto_impl_parallel() {
        let a = [2.0f64, 4.0];
        let b = [1.0f64, 2.0];
        let denom = dot_impl(&b, &b);
        let proj: Vec<f64> = project_onto_impl(&a, &b, denom);
        // a is parallel to b, so projection should be a itself
        assert!((proj[0] - 2.0).abs() < 1e-12);
        assert!((proj[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_project_onto_impl_orthogonal() {
        let a = [0.0f64, 1.0];
        let b = [1.0f64, 0.0];
        let denom = dot_impl(&b, &b);
        let proj: Vec<f64> = project_onto_impl(&a, &b, denom);
        // Orthogonal vectors: projection should be [0,0]
        assert!((proj[0] - 0.0).abs() < 1e-12);
        assert!((proj[1] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_project_onto_impl_identical() {
        let a = [5.0f64, 5.0];
        let b = [5.0f64, 5.0];
        let denom = dot_impl(&b, &b);
        let proj: Vec<f64> = project_onto_impl(&a, &b, denom);
        // Should be a itself
        assert!((proj[0] - 5.0).abs() < 1e-12);
        assert!((proj[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_project_onto_impl_zero_vector() {
        let a = [1.0f64, 2.0];
        let b = [0.0f64, 0.0];
        let denom = dot_impl(&b, &b);
        // denom is zero, so projection would be NaN or inf; check for NaN
        let proj: Vec<f64> = project_onto_impl(&a, &b, denom);
        assert!(proj.iter().all(|x| x.is_nan()));
    }
}
