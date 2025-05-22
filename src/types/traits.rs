//! Traits.

use crate::types::orientation::VectorOrientation;
use num::Complex;
use std::borrow::Cow;

use crate::errors::VectorError;

/// ...
pub trait VectorBase<T> {
    // --- Core accessors ---
    /// ...
    fn as_slice(&self) -> &[T];

    /// ...
    fn as_mut_slice(&mut self) -> &mut [T];

    /// ...
    #[inline]
    fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// ...
    #[inline]
    fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    // --- Element access ---
    /// ...
    #[inline]
    fn get(&self, index: usize) -> Option<&T> {
        self.as_slice().get(index)
    }

    /// ...
    #[inline]
    fn first(&self) -> Option<&T> {
        self.as_slice().first()
    }

    /// ...
    #[inline]
    fn last(&self) -> Option<&T> {
        self.as_slice().last()
    }

    // --- Iteration ---
    /// ...
    #[inline]
    fn iter(&self) -> std::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// ...
    #[inline]
    fn iter_rev(&self) -> std::iter::Rev<std::slice::Iter<'_, T>> {
        self.as_slice().iter().rev()
    }

    /// ...
    #[inline]
    fn enumerate(&self) -> std::iter::Enumerate<std::slice::Iter<'_, T>> {
        self.iter().enumerate()
    }

    // --- Conversion ---
    /// ...
    #[inline]
    fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.as_slice().to_vec()
    }

    /// Returns a boxed slice containing a clone of the vector's data.
    #[inline]
    fn to_boxed_slice(&self) -> Box<[T]>
    where
        T: Clone,
    {
        self.as_slice().to_vec().into_boxed_slice()
    }

    /// Returns an Arc slice containing a clone of the vector's data.
    #[cfg(target_has_atomic = "ptr")]
    #[inline]
    fn to_arc_slice(&self) -> std::sync::Arc<[T]>
    where
        T: Clone,
    {
        std::sync::Arc::from(self.as_slice().to_vec())
    }

    /// Returns an Rc slice containing a clone of the vector's data.
    #[inline]
    fn to_rc_slice(&self) -> std::rc::Rc<[T]>
    where
        T: Clone,
    {
        std::rc::Rc::from(self.as_slice().to_vec())
    }

    /// Returns a Cow<[T]> of the vector's data.
    #[inline]
    fn as_cow(&self) -> Cow<'_, [T]>
    where
        T: Clone,
    {
        Cow::Borrowed(self.as_slice())
    }

    /// ...
    #[inline]
    fn pretty(&self) -> String
    where
        T: std::fmt::Debug,
    {
        format!("{:#?}", self.as_slice())
    }

    // --- Search/containment ---
    /// ...
    #[inline]
    fn contains(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        self.as_slice().contains(x)
    }

    /// ...
    #[inline]
    fn starts_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        self.as_slice().starts_with(needle)
    }

    /// ...
    #[inline]
    fn ends_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        self.as_slice().ends_with(needle)
    }

    /// ...
    #[inline]
    fn position<P>(&self, predicate: P) -> Option<usize>
    where
        P: FnMut(&T) -> bool,
    {
        self.as_slice().iter().position(predicate)
    }

    /// ...
    #[inline]
    fn rposition<P>(&self, predicate: P) -> Option<usize>
    where
        P: FnMut(&T) -> bool,
    {
        self.as_slice().iter().rposition(predicate)
    }

    // --- Slicing/chunking ---
    /// ...
    #[inline]
    fn windows(&self, size: usize) -> std::slice::Windows<'_, T> {
        self.as_slice().windows(size)
    }

    /// ...
    #[inline]
    fn chunks(&self, size: usize) -> std::slice::Chunks<'_, T> {
        self.as_slice().chunks(size)
    }

    /// ...
    #[inline]
    fn split_at(&self, mid: usize) -> (&[T], &[T]) {
        self.as_slice().split_at(mid)
    }

    /// ...
    #[inline]
    fn split<F>(&self, pred: F) -> std::slice::Split<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        self.as_slice().split(pred)
    }

    /// ...
    #[inline]
    fn splitn<F>(&self, n: usize, pred: F) -> std::slice::SplitN<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        self.as_slice().splitn(n, pred)
    }

    /// ...
    #[inline]
    fn rsplit<F>(&self, pred: F) -> std::slice::RSplit<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        self.as_slice().rsplit(pred)
    }

    /// ...
    #[inline]
    fn rsplitn<F>(&self, n: usize, pred: F) -> std::slice::RSplitN<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        self.as_slice().rsplitn(n, pred)
    }

    // --- Pointer access ---
    /// ...
    #[inline]
    fn as_ptr(&self) -> *const T {
        self.as_slice().as_ptr()
    }
}

/// A trait for types that can be transposed between row and column orientation.
pub trait Transposable {
    /// The type returned by transposing.
    type Transposed;

    /// Returns a new value with the opposite orientation.
    fn transpose(self) -> Self::Transposed;
}

/// ...
pub trait VectorOps<T>: VectorBase<T> {
    /// ...
    type Output;

    /// ...
    fn translate(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: num::Num + Copy;

    /// ...
    fn mut_translate(&mut self, other: &Self) -> Result<(), VectorError>
    where
        T: num::Num + Copy;

    /// Returns a new vector scaled by the given scalar.
    fn scale(&self, scalar: T) -> Self::Output
    where
        T: num::Num + Copy,
        Self::Output: std::iter::FromIterator<T>;

    /// Scales the vector in place by the given scalar.
    #[inline]
    fn mut_scale(&mut self, scalar: T)
    where
        T: num::Num + Copy,
    {
        for a in self.as_mut_slice().iter_mut() {
            *a = *a * scalar;
        }
    }

    /// Returns a new vector with all elements negated.
    fn negate(&self) -> Self::Output
    where
        T: std::ops::Neg<Output = T> + Clone,
        Self::Output: std::iter::FromIterator<T>;

    /// Negates all elements in place.
    #[inline]
    fn mut_negate(&mut self)
    where
        T: std::ops::Neg<Output = T> + Clone,
    {
        for a in self.as_mut_slice().iter_mut() {
            *a = -a.clone();
        }
    }

    /// Sets all elements to zero in place.
    #[inline]
    fn mut_zero(&mut self)
    where
        T: num::Zero + Clone,
    {
        for a in self.as_mut_slice().iter_mut() {
            *a = T::zero();
        }
    }

    /// ...
    fn dot(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Num + Copy + std::iter::Sum<T>;

    /// Dot product as f64 (for integer and float types).
    fn dot_to_f64(&self, other: &Self) -> Result<f64, VectorError>
    where
        T: num::ToPrimitive;

    ///...
    fn cross(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: num::Num + Copy,
        Self::Output: std::iter::FromIterator<T>;

    /// ...
    #[inline]
    fn sum(&self) -> T
    where
        T: num::Num + Copy + std::iter::Sum<T>,
    {
        self.as_slice().iter().copied().sum()
    }

    /// ...
    #[inline]
    fn product(&self) -> T
    where
        T: num::Num + Copy + std::iter::Product<T>,
    {
        self.as_slice().iter().copied().product()
    }

    /// ...
    #[inline]
    fn minimum(&self) -> Option<T>
    where
        T: PartialOrd + Copy,
    {
        self.as_slice().iter().copied().reduce(|a, b| if a < b { a } else { b })
    }

    /// Element-wise minimum
    fn elementwise_min(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: PartialOrd + Clone;

    /// ...
    #[inline]
    fn maximum(&self) -> Option<T>
    where
        T: PartialOrd + Copy,
    {
        self.as_slice().iter().copied().reduce(|a, b| if a > b { a } else { b })
    }

    /// Element-wise maximum
    fn elementwise_max(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: PartialOrd + Clone;

    /// Returns a new vector where each element is clamped to the [min, max] range.
    #[inline]
    fn elementwise_clamp(&self, min: T, max: T) -> Self::Output
    where
        T: PartialOrd + Clone,
        Self::Output: std::iter::FromIterator<T>,
    {
        self.as_slice()
            .iter()
            .map(|x| {
                if *x < min {
                    min.clone()
                } else if *x > max {
                    max.clone()
                } else {
                    x.clone()
                }
            })
            .collect()
    }

    /// L1 norm (sum of absolute values).
    #[inline]
    fn l1_norm(&self) -> T
    where
        T: num::Signed + Copy + std::iter::Sum<T>,
    {
        self.as_slice().iter().map(|a| a.abs()).sum()
    }

    /// L∞ norm (maximum absolute value).
    #[inline]
    fn linf_norm(&self) -> T
    where
        T: num::Signed + Copy + PartialOrd,
    {
        self.as_slice()
            .iter()
            .map(|a| a.abs())
            .fold(T::zero(), |acc, x| if acc > x { acc } else { x })
    }
}

/// ...
pub trait VectorOpsFloat<T>: VectorBase<T> {
    /// ...
    type Output;

    /// ...
    fn normalize(&self) -> Result<Self::Output, VectorError>
    where
        T: Copy + PartialEq + std::ops::Div<T, Output = T>,
        Self::Output: std::iter::FromIterator<T>;

    /// ...
    fn mut_normalize(&mut self) -> Result<(), VectorError>
    where
        T: Copy + PartialEq + std::ops::Div<T, Output = T>;

    /// Returns a new vector with the same direction and the given magnitude.
    fn normalize_to(&self, magnitude: T) -> Result<Self::Output, VectorError>
    where
        T: Copy + PartialEq + std::ops::Div<T, Output = T> + std::ops::Mul<T, Output = T>,
        Self::Output: std::iter::FromIterator<T>;

    /// ...
    fn mut_normalize_to(&mut self, magnitude: T) -> Result<(), VectorError>
    where
        T: Copy
            + PartialEq
            + std::ops::Div<T, Output = T>
            + std::ops::Mul<T, Output = T>
            + num::Zero;

    /// Linear interpolation between self and end by weight in [0, 1].
    fn lerp(&self, end: &Self, weight: T) -> Result<Self::Output, VectorError>
    where
        T: num::Float + Clone + PartialOrd;

    /// In-place linear interpolation between self and end by weight in [0, 1].
    fn mut_lerp(&mut self, end: &Self, weight: T) -> Result<(), VectorError>
    where
        T: num::Float + Copy + PartialOrd;

    /// Midpoint
    fn midpoint(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: num::Float + Clone;

    /// Euclidean distance between self and other.
    fn distance(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + Clone + std::iter::Sum<T>;

    /// Manhattan (L1) distance between self and other.
    fn manhattan_distance(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + Clone + std::iter::Sum<T>;

    /// Chebyshev (L∞) distance between self and other.
    fn chebyshev_distance(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + Clone + PartialOrd;

    /// Minkowski (Lp) distance between self and other.
    fn minkowski_distance(&self, other: &Self, p: T) -> Result<T, VectorError>
    where
        T: num::Float + Clone + std::iter::Sum<T>;

    /// Euclidean norm (magnitude) of the vector.
    #[inline]
    fn norm(&self) -> T
    where
        T: num::Float + Clone + std::iter::Sum<T>,
    {
        self.as_slice().iter().map(|a| (*a).powi(2)).sum::<T>().sqrt()
    }

    /// Alias for norm (magnitude).
    #[inline]
    fn magnitude(&self) -> T
    where
        T: num::Float + Clone + std::iter::Sum<T>,
    {
        self.norm()
    }

    /// Lp norm (generalized Minkowski norm).
    #[inline]
    fn lp_norm(&self, p: T) -> Result<T, VectorError>
    where
        T: num::Float + Copy + std::iter::Sum<T>,
    {
        if p < T::one() {
            return Err(VectorError::OutOfRangeError("p must be >= 1".to_string()));
        }
        Ok(self.as_slice().iter().map(|a| a.abs().powf(p)).sum::<T>().powf(T::one() / p))
    }

    /// ...
    fn angle_with(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + Clone + std::iter::Sum<T>;

    /// Projects self onto other.
    /// Returns an error if `other` is the zero vector.
    fn project_onto(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: num::Float + Clone + std::iter::Sum<T>,
        Self::Output: std::iter::FromIterator<T>;

    /// ...
    fn cosine_similarity(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + Clone + std::iter::Sum<T> + std::ops::Div<Output = T>;
}

/// ...
pub trait VectorOpsComplex<N>: VectorBase<Complex<N>> {
    /// ...
    type Output;

    /// ...
    fn normalize(&self) -> Result<Self::Output, VectorError>
    where
        Complex<N>: Copy + PartialEq + std::ops::Div<Complex<N>, Output = Complex<N>>,
        Self::Output: std::iter::FromIterator<Complex<N>>;

    /// ...
    fn mut_normalize(&mut self) -> Result<(), VectorError>
    where
        Complex<N>: Copy + PartialEq + std::ops::Div<Complex<N>, Output = Complex<N>>;

    /// Returns a new vector with the same direction and the given magnitude (real).
    fn normalize_to(&self, magnitude: N) -> Result<Self::Output, VectorError>
    where
        Complex<N>: Copy
            + PartialEq
            + std::ops::Div<Complex<N>, Output = Complex<N>>
            + std::ops::Mul<Complex<N>, Output = Complex<N>>,
        Self::Output: std::iter::FromIterator<Complex<N>>;

    /// Scales the complex vector in place to the given (real) magnitude.
    fn mut_normalize_to(&mut self, magnitude: N) -> Result<(), VectorError>
    where
        Complex<N>: Copy
            + PartialEq
            + std::ops::Div<Complex<N>, Output = Complex<N>>
            + std::ops::Mul<Complex<N>, Output = Complex<N>>
            + num::Zero;

    /// Hermitian dot product: for all complex types
    fn dot(&self, other: &Self) -> Result<Complex<N>, VectorError>
    where
        N: num::Num + Copy + std::iter::Sum<N> + std::ops::Neg<Output = N>;

    /// Linear interpolation between self and end by real weight in [0, 1].
    fn lerp(&self, end: &Self, weight: N) -> Result<Self::Output, VectorError>
    where
        N: num::Float + Clone + PartialOrd;

    /// In-place linear interpolation between self and end by real weight in [0, 1].
    fn mut_lerp(&mut self, end: &Self, weight: N) -> Result<(), VectorError>
    where
        N: num::Float + Copy + PartialOrd;

    /// Midpoint
    fn midpoint(&self, end: &Self) -> Result<Self::Output, VectorError>
    where
        N: num::Float + Clone;

    /// Euclidean distance (L2 norm) between self and other (returns real).
    fn distance(&self, other: &Self) -> Result<N, VectorError>
    where
        N: num::Float + Clone + std::iter::Sum<N>;

    /// Manhattan (L1) distance between self and other (sum of magnitudes of differences).
    fn manhattan_distance(&self, other: &Self) -> Result<N, VectorError>
    where
        N: num::Float + Clone + std::iter::Sum<N>;

    /// Chebyshev (L∞) distance between self and other (maximum magnitude of differences).
    fn chebyshev_distance(&self, other: &Self) -> Result<N, VectorError>
    where
        N: num::Float + Clone + PartialOrd;

    /// Minkowski (Lp) distance between self and other.
    fn minkowski_distance(&self, other: &Self, p: N) -> Result<N, VectorError>
    where
        N: num::Float + Clone + std::iter::Sum<N>;

    /// Euclidean norm (magnitude) of the vector (returns real).
    #[inline]
    fn norm(&self) -> N
    where
        N: num::Float + Clone + std::iter::Sum<N>,
    {
        self.as_slice().iter().map(|a| a.norm_sqr()).sum::<N>().sqrt()
    }

    /// Alias for norm (magnitude).
    #[inline]
    fn magnitude(&self) -> N
    where
        N: num::Float + Clone + std::iter::Sum<N>,
    {
        self.norm()
    }

    /// L1 norm (sum of magnitudes).
    #[inline]
    fn l1_norm(&self) -> N
    where
        N: num::Float + Clone + std::iter::Sum<N>,
    {
        self.as_slice().iter().map(|a| a.norm()).sum()
    }

    /// L∞ norm (maximum magnitude).
    #[inline]
    fn linf_norm(&self) -> N
    where
        N: num::Float + Clone + PartialOrd,
    {
        self.as_slice().iter().map(|a| a.norm()).fold(N::zero(), |acc, x| acc.max(x))
    }

    /// Lp norm (generalized Minkowski norm for complex).
    #[inline]
    fn lp_norm(&self, p: N) -> Result<N, VectorError>
    where
        N: num::Float + Clone + std::iter::Sum<N>,
    {
        if p < N::one() {
            return Err(VectorError::OutOfRangeError("p must be >= 1".to_string()));
        }
        Ok(self.as_slice().iter().map(|a| a.norm().powf(p)).sum::<N>().powf(N::one() / p))
    }

    /// ...
    fn project_onto(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        N: num::Float + Clone + std::iter::Sum<N>,
        Self::Output: std::iter::FromIterator<Complex<N>>;

    /// ...
    fn cosine_similarity(&self, other: &Self) -> Result<Complex<N>, VectorError>
    where
        N: num::Float + Clone + std::iter::Sum<N> + std::ops::Neg<Output = N>,
        Complex<N>: std::ops::Div<Output = Complex<N>>;
}

/// ...
pub trait VectorHasOrientation {
    /// ...
    fn orientation(&self) -> VectorOrientation;
}

// ================================
//
// pub(crate) trait impls
//
// ================================

/// Helper trait for orientation name.
pub(crate) trait VectorOrientationName {
    /// Returns orientation name.
    fn orientation_name() -> &'static str;
}
