//! VectorSlice types.

use crate::errors::VectorError;
use crate::types::flexvector::FlexVector;
use crate::types::orientation::Column;
use crate::types::traits::{
    VectorBase, VectorBaseMut, VectorOps, VectorOpsComplex, VectorOpsFloat, VectorOpsFloatMut,
    VectorOpsMut, VectorOrientationName,
};
use crate::types::utils::{
    angle_with_impl, chebyshev_distance_complex_impl, chebyshev_distance_impl,
    cosine_similarity_complex_impl, cosine_similarity_impl, cross_impl, cross_into_impl,
    distance_complex_impl, distance_impl, dot_impl, dot_to_f64_impl, elementwise_max_impl,
    elementwise_max_into_impl, elementwise_min_impl, elementwise_min_into_impl, hermitian_dot_impl,
    lerp_impl, manhattan_distance_complex_impl, manhattan_distance_impl,
    minkowski_distance_complex_impl, minkowski_distance_impl, mut_lerp_impl, mut_normalize_impl,
    mut_normalize_to_impl, mut_translate_impl, normalize_impl, normalize_into_impl,
    normalize_to_impl, normalize_to_into_impl, project_onto_impl, project_onto_into_impl,
    translate_impl,
};

use std::fmt;
use std::marker::PhantomData;

use num::{Complex, Zero};

// /////////////////////////////////
// ================================
//
// VectorSlice type
//
// ================================
// /////////////////////////////////

/// ...
#[derive(Clone, PartialEq, Eq)]
pub struct VectorSlice<'a, T, O = Column> {
    /// ...
    pub elements: &'a [T],
    _orientation: PhantomData<O>,
}

// ================================
//
// Constructors
//
// ================================

impl<'a, T, O> VectorSlice<'a, T, O> {
    /// ...
    #[inline]
    pub fn new(slice: &'a [T]) -> Self {
        VectorSlice { elements: slice, _orientation: PhantomData }
    }

    /// ...
    #[inline]
    pub fn from_range(parent: &'a [T], range: std::ops::Range<usize>) -> Self {
        Self::new(&parent[range])
    }
}

// ================================
//
// Traits
//
// ================================

impl<'a, T, O> std::ops::Deref for VectorSlice<'a, T, O> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.elements
    }
}

impl<'a, T, O> AsRef<[T]> for VectorSlice<'a, T, O> {
    fn as_ref(&self) -> &[T] {
        self.elements
    }
}

impl<'a, T, O> IntoIterator for VectorSlice<'a, T, O> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.iter()
    }
}

impl<'a, T, O> IntoIterator for &'a VectorSlice<'a, T, O> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.iter()
    }
}

impl<'a, T, O> fmt::Display for VectorSlice<'a, T, O>
where
    T: fmt::Debug,
    O: VectorOrientationName + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} VectorSlice {:?}", O::orientation_name(), self.elements)
    }
}

impl<'a, T, O> std::fmt::Debug for VectorSlice<'a, T, O>
where
    T: std::fmt::Debug,
    O: VectorOrientationName + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorSlice")
            .field("orientation", &O::orientation_name())
            .field("elements", &self.elements)
            .finish()
    }
}

impl<'a, T, O> std::hash::Hash for VectorSlice<'a, T, O>
where
    T: std::hash::Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.elements.hash(state);
    }
}

impl<'a, T, O> Ord for VectorSlice<'a, T, O>
where
    T: Ord,
    O: Eq,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.elements.cmp(other.elements)
    }
}

impl<'a, T, O> PartialOrd for VectorSlice<'a, T, O>
where
    T: PartialOrd,
    O: PartialEq,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.elements.partial_cmp(other.elements)
    }
}

// ================================
//
// Crate trait impls
//
// ================================

impl<'a, T, O> VectorBase<T> for VectorSlice<'a, T, O> {
    #[inline]
    fn as_slice(&self) -> &[T] {
        self.elements
    }
}

impl<'a, T, O> VectorOps<T> for VectorSlice<'a, T, O>
where
    T: Copy,
{
    type Output = FlexVector<T, O>;

    #[inline]
    fn translate(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: num::Num + Copy,
    {
        self.check_same_length_and_raise(other)?;
        let mut out = FlexVector::zero(self.len());
        translate_impl(self.as_slice(), other.as_slice(), out.as_mut_slice());
        Ok(out)
    }

    #[inline]
    fn translate_into(&self, other: &Self, out: &mut [T]) -> Result<(), VectorError>
    where
        T: num::Num + Copy,
    {
        self.check_same_length_and_raise(other)?;
        if out.len() != self.len() {
            return Err(VectorError::MismatchedLengthError(
                "Output buffer has wrong length".to_string(),
            ));
        }
        translate_impl(self, other, out);
        Ok(())
    }

    #[inline]
    fn dot(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Num + Copy + std::iter::Sum<T>,
    {
        self.check_same_length_and_raise(other)?;
        Ok(dot_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn dot_to_f64(&self, other: &Self) -> Result<f64, VectorError>
    where
        T: num::ToPrimitive,
    {
        self.check_same_length_and_raise(other)?;
        Ok(dot_to_f64_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn cross(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: num::Num + Copy,
        Self::Output: std::iter::FromIterator<T>,
    {
        if self.len() != 3 || other.len() != 3 {
            return Err(VectorError::OutOfRangeError(
                "Cross product is only defined for 3D vectors".to_string(),
            ));
        }
        let a = self.as_slice();
        let b = other.as_slice();
        let result = cross_impl(a, b);
        Ok(result.into_iter().collect())
    }

    #[inline]
    fn cross_into(&self, other: &Self, out: &mut [T]) -> Result<(), VectorError>
    where
        T: num::Num + Copy,
    {
        if self.len() != 3 || other.len() != 3 || out.len() != 3 {
            return Err(VectorError::OutOfRangeError(
                "Cross product is only defined for 3D vectors".to_string(),
            ));
        }
        let a = self.as_slice();
        let b = other.as_slice();
        cross_into_impl(a, b, out);
        Ok(())
    }

    #[inline]
    fn elementwise_min(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: PartialOrd + Copy,
    {
        self.check_same_length_and_raise(other)?;
        Ok(FlexVector::from_vec(elementwise_min_impl(self.as_slice(), other.as_slice())))
    }

    #[inline]
    fn elementwise_min_into(&self, other: &Self, out: &mut [T]) -> Result<(), VectorError>
    where
        T: PartialOrd + Copy,
    {
        if self.len() != other.len() || out.len() != self.len() {
            return Err(VectorError::MismatchedLengthError(
                "Vectors must have the same length".to_string(),
            ));
        }
        elementwise_min_into_impl(self.as_slice(), other.as_slice(), out);
        Ok(())
    }

    #[inline]
    fn elementwise_max(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: PartialOrd + Copy,
    {
        self.check_same_length_and_raise(other)?;
        Ok(FlexVector::from_vec(elementwise_max_impl(self.as_slice(), other.as_slice())))
    }

    #[inline]
    fn elementwise_max_into(&self, other: &Self, out: &mut [T]) -> Result<(), VectorError>
    where
        T: PartialOrd + Copy,
    {
        if self.len() != other.len() || out.len() != self.len() {
            return Err(VectorError::MismatchedLengthError(
                "Vectors must have the same length".to_string(),
            ));
        }
        elementwise_max_into_impl(self.as_slice(), other.as_slice(), out);
        Ok(())
    }
}

impl<'a, T, O> VectorOpsFloat<T> for VectorSlice<'a, T, O>
where
    T: num::Float + std::iter::Sum<T>,
{
    type Output = FlexVector<T, O>;

    #[inline]
    fn normalize(&self) -> Result<Self::Output, VectorError>
    where
        T: num::Float + std::ops::Div<T, Output = T> + num::Zero,
        Self::Output: std::iter::FromIterator<T>,
    {
        normalize_impl(self.as_slice(), self.norm())
    }

    #[inline]
    fn normalize_into(&self, out: &mut [T]) -> Result<(), VectorError>
    where
        T: num::Float + std::ops::Div<T, Output = T> + num::Zero,
    {
        let norm = self.norm();
        normalize_into_impl(self.as_slice(), norm, out)
    }

    #[inline]
    fn normalize_to(&self, magnitude: T) -> Result<Self::Output, VectorError>
    where
        T: num::Float + std::ops::Div<T, Output = T> + std::ops::Mul<T, Output = T> + num::Zero,
        Self::Output: std::iter::FromIterator<T>,
    {
        normalize_to_impl(self.as_slice(), self.norm(), magnitude)
    }

    #[inline]
    fn normalize_to_into(&self, magnitude: T, out: &mut [T]) -> Result<(), VectorError>
    where
        T: num::Float + std::ops::Div<T, Output = T> + std::ops::Mul<T, Output = T> + num::Zero,
    {
        let norm = self.norm();
        normalize_to_into_impl(self.as_slice(), norm, magnitude, out)
    }

    #[inline]
    fn lerp(&self, end: &Self, weight: T) -> Result<Self::Output, VectorError>
    where
        T: num::Float,
    {
        self.check_same_length_and_raise(end)?;
        if weight < T::zero() || weight > T::one() {
            return Err(VectorError::OutOfRangeError("weight must be in [0, 1]".to_string()));
        }
        let mut out = FlexVector::zero(self.len());
        lerp_impl(self.as_slice(), end.as_slice(), weight, out.as_mut_slice());
        Ok(out)
    }

    #[inline]
    fn lerp_into(&self, end: &Self, weight: T, out: &mut [T]) -> Result<(), VectorError>
    where
        T: num::Float,
    {
        self.check_same_length_and_raise(end)?;
        if self.len() != out.len() {
            return Err(VectorError::MismatchedLengthError(
                "Output buffer has different length than input vectors".to_string(),
            ));
        }
        if weight < T::zero() || weight > T::one() {
            return Err(VectorError::OutOfRangeError("weight must be in [0, 1]".to_string()));
        }
        lerp_impl(self.as_slice(), end.as_slice(), weight, out);
        Ok(())
    }

    #[inline]
    fn midpoint(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: num::Float,
    {
        self.check_same_length_and_raise(other)?;
        let mut out = FlexVector::zero(self.len());
        lerp_impl(self.as_slice(), other.as_slice(), T::from(0.5).unwrap(), out.as_mut_slice());
        Ok(out)
    }

    #[inline]
    fn midpoint_into(&self, end: &Self, out: &mut [T]) -> Result<(), VectorError>
    where
        T: num::Float,
    {
        self.check_same_length_and_raise(end)?;
        if self.len() != out.len() {
            return Err(VectorError::MismatchedLengthError(
                "Output buffer has different length than input vectors".to_string(),
            ));
        }
        lerp_impl(self.as_slice(), end.as_slice(), T::from(0.5).unwrap(), out);
        Ok(())
    }

    #[inline]
    fn distance(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + std::iter::Sum<T>,
    {
        self.check_same_length_and_raise(other)?;
        Ok(distance_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn manhattan_distance(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + std::iter::Sum<T>,
    {
        self.check_same_length_and_raise(other)?;
        Ok(manhattan_distance_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn chebyshev_distance(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + PartialOrd,
    {
        self.check_same_length_and_raise(other)?;
        Ok(chebyshev_distance_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn minkowski_distance(&self, other: &Self, p: T) -> Result<T, VectorError>
    where
        T: num::Float + std::iter::Sum<T>,
    {
        self.check_same_length_and_raise(other)?;
        if p < T::one() {
            return Err(VectorError::OutOfRangeError("p must be >= 1".to_string()));
        }
        Ok(minkowski_distance_impl(self.as_slice(), other.as_slice(), p))
    }

    #[inline]
    fn angle_with(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + std::iter::Sum<T>,
    {
        self.check_same_length_and_raise(other)?;
        let norm_self = self.norm();
        let norm_other = other.norm();
        if norm_self == T::zero() || norm_other == T::zero() {
            return Err(VectorError::ZeroVectorError(
                "Cannot compute angle with zero vector".to_string(),
            ));
        }
        Ok(angle_with_impl(self.as_slice(), other.as_slice(), norm_self, norm_other))
    }

    #[inline]
    fn project_onto(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: num::Float + std::iter::Sum<T>,
        Self::Output: std::iter::FromIterator<T>,
    {
        self.check_same_length_and_raise(other)?;
        let denom = dot_impl(other.as_slice(), other.as_slice());
        if denom == T::zero() {
            return Err(VectorError::ZeroVectorError(
                "Cannot project onto zero vector".to_string(),
            ));
        }
        let scalar = dot_impl(self.as_slice(), other.as_slice()) / denom;
        Ok(project_onto_impl(other.as_slice(), scalar))
    }

    #[inline]
    fn project_onto_into(&self, other: &Self, out: &mut [T]) -> Result<(), VectorError>
    where
        T: num::Float + std::iter::Sum<T>,
    {
        self.check_same_length_and_raise(other)?;
        if out.len() != self.len() {
            return Err(VectorError::MismatchedLengthError(
                "Output buffer has different length than input vectors".to_string(),
            ));
        }
        let denom = dot_impl(other.as_slice(), other.as_slice());
        if denom == T::zero() {
            return Err(VectorError::ZeroVectorError(
                "Cannot project onto zero vector".to_string(),
            ));
        }
        let scalar = dot_impl(self.as_slice(), other.as_slice()) / denom;
        project_onto_into_impl(other.as_slice(), scalar, out);
        Ok(())
    }

    #[inline]
    fn cosine_similarity(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + std::iter::Sum<T> + std::ops::Div<Output = T>,
    {
        self.check_same_length_and_raise(other)?;
        let norm_self = self.norm();
        let norm_other = other.norm();
        if norm_self == T::zero() || norm_other == T::zero() {
            return Err(VectorError::ZeroVectorError(
                "Cannot compute cosine similarity with zero vector".to_string(),
            ));
        }
        Ok(cosine_similarity_impl(self.as_slice(), other.as_slice(), norm_self, norm_other))
    }
}

impl<'a, N, O> VectorOpsComplex<N> for VectorSlice<'a, Complex<N>, O>
where
    N: num::Num + Copy + std::iter::Sum<N>,
{
    type Output = FlexVector<Complex<N>, O>;

    #[inline]
    fn normalize(&self) -> Result<Self::Output, VectorError>
    where
        N: num::Float,
        Complex<N>: Copy + PartialEq + std::ops::Div<Complex<N>, Output = Complex<N>>,
        Self::Output: std::iter::FromIterator<Complex<N>>,
    {
        normalize_impl(self.as_slice(), Complex::new(self.norm(), N::zero()))
    }

    #[inline]
    fn normalize_to(&self, magnitude: N) -> Result<Self::Output, VectorError>
    where
        N: num::Float,
        Complex<N>: Copy
            + PartialEq
            + std::ops::Div<Complex<N>, Output = Complex<N>>
            + std::ops::Mul<Complex<N>, Output = Complex<N>>
            + num::Zero,
        Self::Output: std::iter::FromIterator<Complex<N>>,
    {
        normalize_to_impl(
            self.as_slice(),
            Complex::new(self.norm(), N::zero()),
            Complex::new(magnitude, N::zero()),
        )
    }

    #[inline]
    fn dot(&self, other: &Self) -> Result<Complex<N>, VectorError>
    where
        N: num::Num + Copy + std::iter::Sum<N> + std::ops::Neg<Output = N>,
    {
        self.check_same_length_and_raise(other)?;
        Ok(hermitian_dot_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn lerp(&self, end: &Self, weight: N) -> Result<Self::Output, VectorError>
    where
        N: num::Float,
        Complex<N>: Copy
            + std::ops::Add<Output = Complex<N>>
            + std::ops::Mul<Output = Complex<N>>
            + std::ops::Sub<Output = Complex<N>>
            + num::One,
    {
        self.check_same_length_and_raise(end)?;
        if weight < N::zero() || weight > N::one() {
            return Err(VectorError::OutOfRangeError("weight must be in [0, 1]".to_string()));
        }
        let w = Complex::new(weight, N::zero());
        let mut out = FlexVector::zero(self.len());
        lerp_impl(self.as_slice(), end.as_slice(), w, out.as_mut_slice());
        Ok(out)
    }

    #[inline]
    fn midpoint(&self, end: &Self) -> Result<Self::Output, VectorError>
    where
        N: num::Float,
    {
        self.check_same_length_and_raise(end)?;
        self.lerp(end, num::cast(0.5).unwrap())
    }

    #[inline]
    fn distance(&self, other: &Self) -> Result<N, VectorError>
    where
        N: num::Float + std::iter::Sum<N>,
    {
        self.check_same_length_and_raise(other)?;
        Ok(distance_complex_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn manhattan_distance(&self, other: &Self) -> Result<N, VectorError>
    where
        N: num::Float + std::iter::Sum<N>,
    {
        self.check_same_length_and_raise(other)?;
        Ok(manhattan_distance_complex_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn chebyshev_distance(&self, other: &Self) -> Result<N, VectorError>
    where
        N: num::Float,
    {
        self.check_same_length_and_raise(other)?;
        Ok(chebyshev_distance_complex_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn minkowski_distance(&self, other: &Self, p: N) -> Result<N, VectorError>
    where
        N: num::Float + std::iter::Sum<N>,
    {
        self.check_same_length_and_raise(other)?;
        if p < N::one() {
            return Err(VectorError::OutOfRangeError("p must be >= 1".to_string()));
        }
        Ok(minkowski_distance_complex_impl(self.as_slice(), other.as_slice(), p))
    }

    #[inline]
    fn project_onto(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        N: num::Float + std::iter::Sum<N> + std::ops::Neg<Output = N>,
        Complex<N>: Copy
            + std::ops::Mul<Output = Complex<N>>
            + std::ops::Add<Output = Complex<N>>
            + std::ops::Div<Complex<N>, Output = Complex<N>>
            + num::Zero,
        Self::Output: std::iter::FromIterator<Complex<N>>,
    {
        self.check_same_length_and_raise(other)?;
        let denom = hermitian_dot_impl(other.as_slice(), other.as_slice());
        if denom == Complex::<N>::zero() {
            return Err(VectorError::ZeroVectorError(
                "Cannot project onto zero vector".to_string(),
            ));
        }
        let scalar = hermitian_dot_impl(self.as_slice(), other.as_slice()) / denom;
        Ok(project_onto_impl(other.as_slice(), scalar))
    }

    #[inline]
    fn cosine_similarity(&self, other: &Self) -> Result<num::Complex<N>, VectorError>
    where
        N: num::Float + std::iter::Sum<N> + std::ops::Neg<Output = N>,
        Complex<N>: std::ops::Div<Output = Complex<N>>,
    {
        self.check_same_length_and_raise(other)?;
        let norm_self = self.norm();
        let norm_other = other.norm();
        if norm_self == N::zero() || norm_other == N::zero() {
            return Err(VectorError::ZeroVectorError(
                "Cannot compute cosine similarity with zero vector".to_string(),
            ));
        }
        Ok(cosine_similarity_complex_impl(self.as_slice(), other.as_slice(), norm_self, norm_other))
    }
}

// ================================
//
// Methods
//
// ================================

impl<'a, T, O> VectorSlice<'a, T, O> {
    // ================================
    //
    // Private methods
    //
    // ================================

    /// Returns Ok(()) if self and other have the same length (i.e. vector dimensionality),
    /// otherwise returns a VectorError.
    #[inline]
    fn check_same_length_and_raise(&self, other: &Self) -> Result<(), VectorError> {
        if self.len() != other.len() {
            Err(VectorError::MismatchedLengthError("Vectors must have the same length".to_string()))
        } else {
            Ok(())
        }
    }
}

// /////////////////////////////////
// ================================
//
// VectorSliceMut type
//
// ================================
// /////////////////////////////////

/// ...
pub struct VectorSliceMut<'a, T, O = Column> {
    /// ...
    pub elements: &'a mut [T],
    _orientation: PhantomData<O>,
}

// ================================
//
// Constructors
//
// ================================

impl<'a, T, O> VectorSliceMut<'a, T, O> {
    /// ...
    #[inline]
    pub fn new(slice: &'a mut [T]) -> Self {
        VectorSliceMut { elements: slice, _orientation: PhantomData }
    }

    /// Creates a mutable VectorSliceMut from a parent mutable slice and a range.
    #[inline]
    pub fn from_range(parent: &'a mut [T], range: std::ops::Range<usize>) -> Self {
        Self::new(&mut parent[range])
    }
}

// ================================
//
// Traits
//
// ================================

impl<'a, T, O> std::ops::Deref for VectorSliceMut<'a, T, O> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.elements
    }
}

impl<'a, T, O> std::ops::DerefMut for VectorSliceMut<'a, T, O> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.elements
    }
}

impl<'a, T, O> AsMut<[T]> for VectorSliceMut<'a, T, O> {
    fn as_mut(&mut self) -> &mut [T] {
        self.elements
    }
}

impl<'a, T, O> IntoIterator for VectorSliceMut<'a, T, O> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.iter_mut()
    }
}

impl<'a, T, O> IntoIterator for &'a mut VectorSliceMut<'a, T, O> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.iter_mut()
    }
}

impl<'a, T, O> fmt::Display for VectorSliceMut<'a, T, O>
where
    T: fmt::Debug,
    O: VectorOrientationName + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} VectorSliceMut {:?}", O::orientation_name(), self.elements)
    }
}

impl<'a, T, O> std::fmt::Debug for VectorSliceMut<'a, T, O>
where
    T: std::fmt::Debug,
    O: VectorOrientationName + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorSliceMut")
            .field("orientation", &O::orientation_name())
            .field("elements", &self.elements)
            .finish()
    }
}

// ================================
//
// Crate trait impls
//
// ================================

impl<'a, T, O> VectorBase<T> for VectorSliceMut<'a, T, O> {
    #[inline]
    fn as_slice(&self) -> &[T] {
        self.elements
    }
}

impl<'a, T, O> VectorBaseMut<T> for VectorSliceMut<'a, T, O> {
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.elements
    }
}

impl<'a, T, O> VectorOps<T> for VectorSliceMut<'a, T, O>
where
    T: Copy,
{
    type Output = FlexVector<T, O>;

    #[inline]
    fn translate(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: num::Num + Copy,
    {
        self.check_same_length_and_raise(other)?;
        let mut out = FlexVector::zero(self.len());
        translate_impl(self.as_slice(), other.as_slice(), out.as_mut_slice());
        Ok(out)
    }

    #[inline]
    fn translate_into(&self, other: &Self, out: &mut [T]) -> Result<(), VectorError>
    where
        T: num::Num + Copy,
    {
        self.check_same_length_and_raise(other)?;
        if out.len() != self.len() {
            return Err(VectorError::MismatchedLengthError(
                "Output buffer has wrong length".to_string(),
            ));
        }
        translate_impl(self, other, out);
        Ok(())
    }

    #[inline]
    fn dot(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Num + Copy + std::iter::Sum<T>,
    {
        self.check_same_length_and_raise(other)?;
        Ok(dot_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn dot_to_f64(&self, other: &Self) -> Result<f64, VectorError>
    where
        T: num::ToPrimitive,
    {
        self.check_same_length_and_raise(other)?;
        Ok(dot_to_f64_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn cross(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: num::Num + Copy,
        Self::Output: std::iter::FromIterator<T>,
    {
        if self.len() != 3 || other.len() != 3 {
            return Err(VectorError::OutOfRangeError(
                "Cross product is only defined for 3D vectors".to_string(),
            ));
        }
        let a = self.as_slice();
        let b = other.as_slice();
        let result = cross_impl(a, b);
        Ok(result.into_iter().collect())
    }

    #[inline]
    fn cross_into(&self, other: &Self, out: &mut [T]) -> Result<(), VectorError>
    where
        T: num::Num + Copy,
    {
        if self.len() != 3 || other.len() != 3 || out.len() != 3 {
            return Err(VectorError::OutOfRangeError(
                "Cross product is only defined for 3D vectors".to_string(),
            ));
        }
        let a = self.as_slice();
        let b = other.as_slice();
        cross_into_impl(a, b, out);
        Ok(())
    }

    #[inline]
    fn elementwise_min(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: PartialOrd + Copy,
    {
        self.check_same_length_and_raise(other)?;
        Ok(FlexVector::from_vec(elementwise_min_impl(self.as_slice(), other.as_slice())))
    }

    #[inline]
    fn elementwise_min_into(&self, other: &Self, out: &mut [T]) -> Result<(), VectorError>
    where
        T: PartialOrd + Copy,
    {
        if self.len() != other.len() || out.len() != self.len() {
            return Err(VectorError::MismatchedLengthError(
                "Vectors must have the same length".to_string(),
            ));
        }
        elementwise_min_into_impl(self.as_slice(), other.as_slice(), out);
        Ok(())
    }

    #[inline]
    fn elementwise_max(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: PartialOrd + Copy,
    {
        self.check_same_length_and_raise(other)?;
        Ok(FlexVector::from_vec(elementwise_max_impl(self.as_slice(), other.as_slice())))
    }

    #[inline]
    fn elementwise_max_into(&self, other: &Self, out: &mut [T]) -> Result<(), VectorError>
    where
        T: PartialOrd + Copy,
    {
        if self.len() != other.len() || out.len() != self.len() {
            return Err(VectorError::MismatchedLengthError(
                "Vectors must have the same length".to_string(),
            ));
        }
        elementwise_max_into_impl(self.as_slice(), other.as_slice(), out);
        Ok(())
    }
}

impl<'a, T, O> VectorOpsMut<T> for VectorSliceMut<'a, T, O>
where
    T: Copy,
{
    type Output = Self;

    #[inline]
    fn mut_translate(&mut self, other: &Self) -> Result<(), VectorError>
    where
        T: num::Num + Copy,
    {
        self.check_same_length_and_raise(other)?;
        mut_translate_impl(self.as_mut_slice(), other.as_slice());
        Ok(())
    }
}

impl<'a, T, O> VectorOpsFloat<T> for VectorSliceMut<'a, T, O>
where
    T: num::Float + std::iter::Sum<T>,
{
    type Output = FlexVector<T, O>;

    #[inline]
    fn normalize(&self) -> Result<Self::Output, VectorError>
    where
        T: num::Float + std::ops::Div<T, Output = T>,
        Self::Output: std::iter::FromIterator<T>,
    {
        normalize_impl(self.as_slice(), self.norm())
    }

    #[inline]
    fn normalize_into(&self, out: &mut [T]) -> Result<(), VectorError>
    where
        T: num::Float + std::ops::Div<T, Output = T> + num::Zero,
    {
        let norm = self.norm();
        normalize_into_impl(self.as_slice(), norm, out)
    }

    #[inline]
    fn normalize_to(&self, magnitude: T) -> Result<Self::Output, VectorError>
    where
        T: num::Float + std::ops::Div<T, Output = T> + std::ops::Mul<T, Output = T>,
        Self::Output: std::iter::FromIterator<T>,
    {
        normalize_to_impl(self.as_slice(), self.norm(), magnitude)
    }

    #[inline]
    fn normalize_to_into(&self, magnitude: T, out: &mut [T]) -> Result<(), VectorError>
    where
        T: num::Float + std::ops::Div<T, Output = T> + std::ops::Mul<T, Output = T> + num::Zero,
    {
        let norm = self.norm();
        normalize_to_into_impl(self.as_slice(), norm, magnitude, out)
    }

    #[inline]
    fn lerp(&self, end: &Self, weight: T) -> Result<Self::Output, VectorError>
    where
        T: num::Float,
    {
        self.check_same_length_and_raise(end)?;
        if weight < T::zero() || weight > T::one() {
            return Err(VectorError::OutOfRangeError("weight must be in [0, 1]".to_string()));
        }
        let mut out = FlexVector::zero(self.len());
        lerp_impl(self.as_slice(), end.as_slice(), weight, out.as_mut_slice());
        Ok(out)
    }

    #[inline]
    fn lerp_into(&self, end: &Self, weight: T, out: &mut [T]) -> Result<(), VectorError>
    where
        T: num::Float,
    {
        self.check_same_length_and_raise(end)?;
        if self.len() != out.len() {
            return Err(VectorError::MismatchedLengthError(
                "Output buffer has different length than input vectors".to_string(),
            ));
        }
        if weight < T::zero() || weight > T::one() {
            return Err(VectorError::OutOfRangeError("weight must be in [0, 1]".to_string()));
        }
        lerp_impl(self.as_slice(), end.as_slice(), weight, out);
        Ok(())
    }

    #[inline]
    fn midpoint(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: num::Float,
    {
        self.check_same_length_and_raise(other)?;
        let mut out = FlexVector::zero(self.len());
        lerp_impl(self.as_slice(), other.as_slice(), T::from(0.5).unwrap(), out.as_mut_slice());
        Ok(out)
    }

    #[inline]
    fn midpoint_into(&self, end: &Self, out: &mut [T]) -> Result<(), VectorError>
    where
        T: num::Float,
    {
        self.check_same_length_and_raise(end)?;
        if self.len() != out.len() {
            return Err(VectorError::MismatchedLengthError(
                "Output buffer has different length than input vectors".to_string(),
            ));
        }
        lerp_impl(self.as_slice(), end.as_slice(), T::from(0.5).unwrap(), out);
        Ok(())
    }

    #[inline]
    fn distance(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + std::iter::Sum<T>,
    {
        self.check_same_length_and_raise(other)?;
        Ok(distance_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn manhattan_distance(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + std::iter::Sum<T>,
    {
        self.check_same_length_and_raise(other)?;
        Ok(manhattan_distance_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn chebyshev_distance(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + PartialOrd,
    {
        self.check_same_length_and_raise(other)?;
        Ok(chebyshev_distance_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn minkowski_distance(&self, other: &Self, p: T) -> Result<T, VectorError>
    where
        T: num::Float + std::iter::Sum<T>,
    {
        self.check_same_length_and_raise(other)?;
        if p < T::one() {
            return Err(VectorError::OutOfRangeError("p must be >= 1".to_string()));
        }
        Ok(minkowski_distance_impl(self.as_slice(), other.as_slice(), p))
    }

    #[inline]
    fn angle_with(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + std::iter::Sum<T>,
    {
        self.check_same_length_and_raise(other)?;
        let norm_self = self.norm();
        let norm_other = other.norm();
        if norm_self == T::zero() || norm_other == T::zero() {
            return Err(VectorError::ZeroVectorError(
                "Cannot compute angle with zero vector".to_string(),
            ));
        }
        Ok(angle_with_impl(self.as_slice(), other.as_slice(), norm_self, norm_other))
    }

    #[inline]
    fn project_onto(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: num::Float + std::iter::Sum<T>,
        Self::Output: std::iter::FromIterator<T>,
    {
        self.check_same_length_and_raise(other)?;
        let denom = dot_impl(other.as_slice(), other.as_slice());
        if denom == T::zero() {
            return Err(VectorError::ZeroVectorError(
                "Cannot project onto zero vector".to_string(),
            ));
        }
        let scalar = dot_impl(self.as_slice(), other.as_slice()) / denom;
        Ok(project_onto_impl(other.as_slice(), scalar))
    }

    #[inline]
    fn project_onto_into(&self, other: &Self, out: &mut [T]) -> Result<(), VectorError>
    where
        T: num::Float + std::iter::Sum<T>,
    {
        self.check_same_length_and_raise(other)?;
        if out.len() != self.len() {
            return Err(VectorError::MismatchedLengthError(
                "Output buffer has different length than input vectors".to_string(),
            ));
        }
        let denom = dot_impl(other.as_slice(), other.as_slice());
        if denom == T::zero() {
            return Err(VectorError::ZeroVectorError(
                "Cannot project onto zero vector".to_string(),
            ));
        }
        let scalar = dot_impl(self.as_slice(), other.as_slice()) / denom;
        project_onto_into_impl(other.as_slice(), scalar, out);
        Ok(())
    }

    #[inline]
    fn cosine_similarity(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + std::iter::Sum<T> + std::ops::Div<Output = T>,
    {
        self.check_same_length_and_raise(other)?;
        let norm_self = self.norm();
        let norm_other = other.norm();
        if norm_self == T::zero() || norm_other == T::zero() {
            return Err(VectorError::ZeroVectorError(
                "Cannot compute cosine similarity with zero vector".to_string(),
            ));
        }
        Ok(cosine_similarity_impl(self.as_slice(), other.as_slice(), norm_self, norm_other))
    }
}

impl<'a, T, O> VectorOpsFloatMut<T> for VectorSliceMut<'a, T, O>
where
    T: num::Float + std::iter::Sum<T>,
{
    type Output = Self;

    #[inline]
    fn mut_normalize(&mut self) -> Result<(), VectorError>
    where
        T: num::Float + std::ops::Div<T, Output = T>,
    {
        let norm = self.norm();
        mut_normalize_impl(self.as_mut_slice(), norm)
    }

    #[inline]
    fn mut_normalize_to(&mut self, magnitude: T) -> Result<(), VectorError>
    where
        T: num::Float + std::ops::Div<T, Output = T> + std::ops::Mul<T, Output = T> + num::Zero,
    {
        let n = self.norm();
        mut_normalize_to_impl(self.as_mut_slice(), n, magnitude)
    }

    #[inline]
    fn mut_lerp(&mut self, end: &Self, weight: T) -> Result<(), VectorError>
    where
        T: num::Float,
    {
        self.check_same_length_and_raise(end)?;
        if weight < T::zero() || weight > T::one() {
            return Err(VectorError::OutOfRangeError("weight must be in [0, 1]".to_string()));
        }
        mut_lerp_impl(self.as_mut_slice(), end.as_slice(), weight);
        Ok(())
    }
}

impl<'a, N, O> VectorOpsComplex<N> for VectorSliceMut<'a, Complex<N>, O>
where
    N: num::Num + Copy + std::iter::Sum<N>,
{
    type Output = FlexVector<Complex<N>, O>;

    #[inline]
    fn normalize(&self) -> Result<Self::Output, VectorError>
    where
        N: num::Float,
        Complex<N>: Copy + PartialEq + std::ops::Div<Complex<N>, Output = Complex<N>>,
        Self::Output: std::iter::FromIterator<Complex<N>>,
    {
        normalize_impl(self.as_slice(), Complex::new(self.norm(), N::zero()))
    }

    #[inline]
    fn normalize_to(&self, magnitude: N) -> Result<Self::Output, VectorError>
    where
        N: num::Float,
        Complex<N>: Copy
            + PartialEq
            + std::ops::Div<Complex<N>, Output = Complex<N>>
            + std::ops::Mul<Complex<N>, Output = Complex<N>>,
        Self::Output: std::iter::FromIterator<Complex<N>>,
    {
        normalize_to_impl(
            self.as_slice(),
            Complex::new(self.norm(), N::zero()),
            Complex::new(magnitude, N::zero()),
        )
    }

    #[inline]
    fn dot(&self, other: &Self) -> Result<Complex<N>, VectorError>
    where
        N: num::Num + Copy + std::iter::Sum<N> + std::ops::Neg<Output = N>,
    {
        self.check_same_length_and_raise(other)?;
        Ok(hermitian_dot_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn lerp(&self, end: &Self, weight: N) -> Result<Self::Output, VectorError>
    where
        N: num::Float,
    {
        self.check_same_length_and_raise(end)?;
        if weight < N::zero() || weight > N::one() {
            return Err(VectorError::OutOfRangeError("weight must be in [0, 1]".to_string()));
        }
        let w = Complex::new(weight, N::zero());
        let mut out = FlexVector::zero(self.len());
        lerp_impl(self.as_slice(), end.as_slice(), w, out.as_mut_slice());
        Ok(out)
    }

    #[inline]
    fn midpoint(&self, end: &Self) -> Result<Self::Output, VectorError>
    where
        N: num::Float,
    {
        self.check_same_length_and_raise(end)?;
        self.lerp(end, num::cast(0.5).unwrap())
    }

    #[inline]
    fn distance(&self, other: &Self) -> Result<N, VectorError>
    where
        N: num::Float + std::iter::Sum<N>,
    {
        self.check_same_length_and_raise(other)?;
        Ok(distance_complex_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn manhattan_distance(&self, other: &Self) -> Result<N, VectorError>
    where
        N: num::Float + std::iter::Sum<N>,
    {
        self.check_same_length_and_raise(other)?;
        Ok(manhattan_distance_complex_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn chebyshev_distance(&self, other: &Self) -> Result<N, VectorError>
    where
        N: num::Float,
    {
        self.check_same_length_and_raise(other)?;
        Ok(chebyshev_distance_complex_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn minkowski_distance(&self, other: &Self, p: N) -> Result<N, VectorError>
    where
        N: num::Float + std::iter::Sum<N>,
    {
        self.check_same_length_and_raise(other)?;
        if p < N::one() {
            return Err(VectorError::OutOfRangeError("p must be >= 1".to_string()));
        }
        Ok(minkowski_distance_complex_impl(self.as_slice(), other.as_slice(), p))
    }

    #[inline]
    fn project_onto(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        N: num::Float + std::iter::Sum<N>,
        Self::Output: std::iter::FromIterator<Complex<N>>,
    {
        self.check_same_length_and_raise(other)?;
        let denom = hermitian_dot_impl(other.as_slice(), other.as_slice());
        if denom == Complex::<N>::zero() {
            return Err(VectorError::ZeroVectorError(
                "Cannot project onto zero vector".to_string(),
            ));
        }
        let scalar = hermitian_dot_impl(self.as_slice(), other.as_slice()) / denom;
        Ok(project_onto_impl(other.as_slice(), scalar))
    }

    #[inline]
    fn cosine_similarity(&self, other: &Self) -> Result<num::Complex<N>, VectorError>
    where
        N: num::Float + std::iter::Sum<N> + std::ops::Neg<Output = N>,
        Complex<N>: std::ops::Div<Output = Complex<N>>,
    {
        self.check_same_length_and_raise(other)?;
        let norm_self = self.norm();
        let norm_other = other.norm();
        if norm_self == N::zero() || norm_other == N::zero() {
            return Err(VectorError::ZeroVectorError(
                "Cannot compute cosine similarity with zero vector".to_string(),
            ));
        }
        Ok(cosine_similarity_complex_impl(self.as_slice(), other.as_slice(), norm_self, norm_other))
    }
}

// ================================
//
// Methods
//
// ================================

impl<'a, T, O> VectorSliceMut<'a, T, O> {
    // ================================
    //
    // Private methods
    //
    // ================================

    /// Returns Ok(()) if self and other have the same length (i.e. vector dimensionality),
    /// otherwise returns a VectorError.
    #[inline]
    fn check_same_length_and_raise(&self, other: &Self) -> Result<(), VectorError> {
        if self.len() != other.len() {
            Err(VectorError::MismatchedLengthError("Vectors must have the same length".to_string()))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{FlexVector, VectorBase};
    use crate::types::orientation::{Column, Row};
    use num::Complex;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // /////////////////////////////////
    // ================================
    //
    // VectorSlice type
    //
    // ================================
    // /////////////////////////////////

    // -- new --
    #[test]
    fn test_vector_slice_new() {
        let data = [1, 2, 3, 4, 5];
        let vslice: VectorSlice<'_, i32, Column> = VectorSlice::new(&data);
        assert_eq!(vslice.elements, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_vector_slice_new_complex() {
        let data = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0), Complex::new(5.0, 6.0)];
        let vslice: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::new(&data);
        assert_eq!(
            vslice.elements,
            &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0), Complex::new(5.0, 6.0),]
        );
    }

    // -- from_range --
    #[test]
    fn test_vector_slice_from_range_middle() {
        let data = [10, 20, 30, 40, 50];
        let vslice: VectorSlice<'_, i32, Row> = VectorSlice::from_range(&data, 1..4);
        assert_eq!(vslice.elements, &[20, 30, 40]);
    }

    #[test]
    fn test_vector_slice_from_range_full() {
        let data = [7, 8, 9];
        let vslice: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 0..3);
        assert_eq!(vslice.elements, &[7, 8, 9]);
    }

    #[test]
    fn test_vector_slice_from_range_empty() {
        let data = [1, 2, 3];
        let vslice: VectorSlice<'_, i32, Row> = VectorSlice::from_range(&data, 1..1);
        assert_eq!(vslice.elements, &[]);
    }

    #[test]
    fn test_vector_slice_from_range_middle_complex() {
        let data = [
            Complex::new(10.0, 1.0),
            Complex::new(20.0, 2.0),
            Complex::new(30.0, 3.0),
            Complex::new(40.0, 4.0),
            Complex::new(50.0, 5.0),
        ];
        let vslice: VectorSlice<'_, Complex<f64>, Row> = VectorSlice::from_range(&data, 1..4);
        assert_eq!(
            vslice.elements,
            &[Complex::new(20.0, 2.0), Complex::new(30.0, 3.0), Complex::new(40.0, 4.0),]
        );
    }

    #[test]
    fn test_vector_slice_from_range_full_complex() {
        let data = [Complex::new(7.0, 0.0), Complex::new(8.0, 1.0), Complex::new(9.0, 2.0)];
        let vslice: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&data, 0..3);
        assert_eq!(
            vslice.elements,
            &[Complex::new(7.0, 0.0), Complex::new(8.0, 1.0), Complex::new(9.0, 2.0),]
        );
    }

    #[test]
    fn test_vector_slice_from_range_empty_complex() {
        let data = [Complex::new(1.0, 1.0), Complex::new(2.0, 2.0), Complex::new(3.0, 3.0)];
        let vslice: VectorSlice<'_, Complex<f64>, Row> = VectorSlice::from_range(&data, 1..1);
        assert_eq!(vslice.elements, &[]);
    }

    // -- Deref trait for VectorSlice --
    #[test]
    fn test_vector_slice_deref_access() {
        let data = [1, 2, 3, 4, 5];
        let vslice: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 1..4);
        // Deref allows direct indexing
        assert_eq!(vslice[0], 2);
        assert_eq!(vslice[1], 3);
        assert_eq!(vslice[2], 4);
        // Deref allows using slice methods
        assert_eq!(vslice.len(), 3);
        assert!(vslice.contains(&3));
        assert_eq!(vslice.iter().sum::<i32>(), 9);
    }

    #[test]
    fn test_vector_slice_deref_complex() {
        let data = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0), Complex::new(5.0, 6.0)];
        let vslice: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&data, 0..2);
        // Deref allows direct indexing
        assert_eq!(vslice[0], Complex::new(1.0, 2.0));
        assert_eq!(vslice[1], Complex::new(3.0, 4.0));
        // Deref allows using slice methods
        assert_eq!(vslice.len(), 2);
        assert!(vslice.contains(&Complex::new(3.0, 4.0)));
    }

    // -- AsRef trait for VectorSlice --

    #[test]
    fn test_vector_slice_as_ref_basic() {
        let data = [1, 2, 3, 4, 5];
        let vslice: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 2..5);
        let slice_ref: &[i32] = vslice.as_ref();
        assert_eq!(slice_ref, &[3, 4, 5]);
    }

    #[test]
    fn test_vector_slice_as_ref_complex() {
        let data = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0), Complex::new(5.0, 6.0)];
        let vslice: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&data, 1..3);
        let slice_ref: &[Complex<f64>] = vslice.as_ref();
        assert_eq!(slice_ref, &[Complex::new(3.0, 4.0), Complex::new(5.0, 6.0)]);
    }

    #[test]
    fn test_vector_slice_as_ref_with_std_function() {
        let data = [10, 20, 30, 40];
        let vslice: VectorSlice<'_, i32, Row> = VectorSlice::from_range(&data, 1..4);
        // Use a standard library function that takes AsRef<[i32]>
        let sum: i32 = vslice.as_ref().iter().sum();
        assert_eq!(sum, 20 + 30 + 40);
    }

    // -- IntoIterator trait for VectorSlice --
    #[test]
    fn test_vector_slice_into_iter_basic() {
        let data = [1, 2, 3, 4];
        let vslice: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 1..4);
        let collected: Vec<i32> = vslice.into_iter().copied().collect();
        assert_eq!(collected, vec![2, 3, 4]);
    }

    #[test]
    fn test_vector_slice_into_iter_ref() {
        let data = [10, 20, 30];
        let vslice: VectorSlice<'_, i32, Row> = VectorSlice::from_range(&data, 0..2);
        let mut sum = 0;
        for val in &vslice {
            sum += *val;
        }
        assert_eq!(sum, 10 + 20);
    }

    #[test]
    fn test_vector_slice_into_iter_complex() {
        let data = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0), Complex::new(5.0, 6.0)];
        let vslice: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&data, 0..3);
        let collected: Vec<Complex<f64>> = vslice.into_iter().cloned().collect();
        assert_eq!(
            collected,
            vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0), Complex::new(5.0, 6.0)]
        );
    }

    // -- Display trait for VectorSlice --

    #[test]
    fn test_vector_slice_display() {
        let data = [1, 2, 3];
        let vslice: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 0..3);
        let display = format!("{}", vslice);
        // The exact string depends on your orientation name implementation
        assert!(display.contains("Column VectorSlice"));
        assert!(display.contains("[1, 2, 3]"));
    }

    #[test]
    fn test_vector_slice_display_complex() {
        let data = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let vslice: VectorSlice<'_, Complex<f64>, Row> = VectorSlice::from_range(&data, 0..2);
        let display = format!("{}", vslice);
        assert!(display.contains("Row VectorSlice"));
        assert!(display.contains("Complex { re: 1.0, im: 2.0 }"));
        assert!(display.contains("Complex { re: 3.0, im: 4.0 }"));
    }

    // -- Debug trait for VectorSlice --

    #[test]
    fn test_vector_slice_debug() {
        let data = [1, 2, 3];
        let vslice: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 0..3);
        let debug = format!("{:?}", vslice);
        assert!(debug.contains("VectorSlice"));
        assert!(debug.contains("orientation"));
        assert!(debug.contains("elements"));
        assert!(debug.contains("1"));
        assert!(debug.contains("2"));
        assert!(debug.contains("3"));
    }

    #[test]
    fn test_vector_slice_debug_complex() {
        let data = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let vslice: VectorSlice<'_, Complex<f64>, Row> = VectorSlice::from_range(&data, 0..2);
        let debug = format!("{:?}", vslice);
        assert!(debug.contains("VectorSlice"));
        assert!(debug.contains("orientation"));
        assert!(debug.contains("elements"));
        assert!(debug.contains("Complex { re: 1.0, im: 2.0 }"));
        assert!(debug.contains("Complex { re: 3.0, im: 4.0 }"));
    }

    // -- Hash trait for VectorSlice --

    #[test]
    fn test_vector_slice_hash_basic() {
        let data = [1, 2, 3, 4];
        let vslice1: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 1..4);
        let vslice2: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 1..4);

        let mut hasher1 = DefaultHasher::new();
        vslice1.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        vslice2.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_vector_slice_hash_different() {
        let data = [1, 2, 3, 4];
        let vslice1: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 0..3);
        let vslice2: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 1..4);

        let mut hasher1 = DefaultHasher::new();
        vslice1.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        vslice2.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_vector_slice_hash_complex() {
        let data = [Complex::new(1, 2), Complex::new(3, 4), Complex::new(5, 6)];
        let vslice1: VectorSlice<'_, Complex<i32>, Row> = VectorSlice::from_range(&data, 0..2);
        let vslice2: VectorSlice<'_, Complex<i32>, Row> = VectorSlice::from_range(&data, 0..2);

        let mut hasher1 = DefaultHasher::new();
        vslice1.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        vslice2.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        assert_eq!(hash1, hash2);
    }

    // -- Ord / PartialOrd traits for VectorSlice --

    #[test]
    fn test_vector_slice_ord_basic() {
        let data = [1, 2, 3, 4, 5];
        let vslice1: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 1..4); // [2, 3, 4]
        let vslice2: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 2..5); // [3, 4, 5]
        assert!(vslice1 < vslice2);
        assert!(vslice2 > vslice1);
        assert_eq!(vslice1, VectorSlice::from_range(&data, 1..4));
    }

    #[test]
    fn test_vector_slice_ord_equal() {
        let data = [10, 20, 30];
        let vslice1: VectorSlice<'_, i32, Row> = VectorSlice::from_range(&data, 0..3);
        let vslice2: VectorSlice<'_, i32, Row> = VectorSlice::from_range(&data, 0..3);
        assert_eq!(vslice1, vslice2);
        assert!(vslice1 <= vslice2);
        assert!(vslice1 >= vslice2);
    }

    #[test]
    fn test_vector_slice_partial_ord_f64() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let vslice1: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&data, 0..3); // [1.0, 2.0, 3.0]
        let vslice2: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&data, 1..4); // [2.0, 3.0, 4.0]
        assert!(vslice1 < vslice2);
        assert!(vslice2 > vslice1);
        assert_eq!(vslice1.partial_cmp(&vslice1), Some(std::cmp::Ordering::Equal));
    }

    // -- VectorBase trait for VectorSlice --

    #[test]
    fn test_vector_slice_as_slice_basic() {
        let data = [1, 2, 3, 4];
        let vslice: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 1..3);
        assert_eq!(vslice.as_slice(), &[2, 3]);
    }

    #[test]
    fn test_vector_slice_as_slice_full_range() {
        let data = [10, 20, 30];
        let vslice: VectorSlice<'_, i32, Row> = VectorSlice::from_range(&data, 0..data.len());
        assert_eq!(vslice.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_vector_slice_as_slice_empty() {
        let data = [1, 2, 3];
        let vslice: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 1..1);
        assert_eq!(vslice.as_slice(), &[]);
    }

    #[test]
    fn test_vector_slice_as_slice_complex() {
        let data = [num::Complex::new(1.0, 2.0), num::Complex::new(3.0, 4.0)];
        let vslice: VectorSlice<'_, num::Complex<f64>, Column> =
            VectorSlice::from_range(&data, 0..2);
        assert_eq!(vslice.as_slice(), &data);
    }

    #[test]
    fn test_vector_slice_len_and_is_empty() {
        let data = [1, 2, 3];
        let vslice: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 0..2);
        assert_eq!(vslice.len(), 2);
        assert!(!vslice.is_empty());
        let empty: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 1..1);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_vector_slice_get_and_first_last() {
        let data = [1, 2, 3];
        let vslice: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 0..3);
        assert_eq!(vslice.get(1), Some(&2));
        assert_eq!(vslice.first(), Some(&1));
        assert_eq!(vslice.last(), Some(&3));
        let empty: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 1..1);
        assert_eq!(empty.get(1), None);
        assert_eq!(empty.first(), None);
        assert_eq!(empty.last(), None);
    }

    #[test]
    fn test_vector_slice_iter_and_to_vec() {
        let data = [1, 2, 3];
        let vslice: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&data, 0..3);
        let collected: Vec<i32> = vslice.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3]);
        assert_eq!(vslice.to_vec(), vec![1, 2, 3]);
    }

    // -- VectorOps trait for VectorSlice --

    // -- translate --

    #[test]
    fn test_vector_slice_translate() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let result = vslice_a.translate(&vslice_b).unwrap();
        assert_eq!(result.as_slice(), &[5, 7, 9]);
    }

    #[test]
    fn test_vector_slice_translate_mismatched_length() {
        let a = [1, 2, 3];
        let b = [4, 5];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let result = vslice_a.translate(&vslice_b);
        assert!(result.is_err());
    }

    // -- translate_into --

    #[test]
    fn test_vector_slice_translate_into_basic() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let mut out = [0; 3];
        vslice_a.translate_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, [5, 7, 9]);
    }

    #[test]
    fn test_vector_slice_translate_into_empty() {
        let a: [i32; 0] = [];
        let b: [i32; 0] = [];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..0);
        let vslice_b = VectorSlice::from_range(&b, 0..0);
        let mut out: [i32; 0] = [];
        vslice_a.translate_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, []);
    }

    #[test]
    fn test_vector_slice_translate_into_mismatched_length() {
        let a = [1, 2, 3];
        let b = [4, 5];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let mut out = [0; 3];
        let result = vslice_a.translate_into(&vslice_b, &mut out);
        assert!(result.is_err());

        let a = [1, 2, 3];
        let b = [4, 5, 6];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let mut out = [0; 2];
        let result = vslice_a.translate_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    // -- dot --

    #[test]
    fn test_vector_slice_dot() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        let vslice_a: VectorSlice<'_, i32, Row> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let result = vslice_a.dot(&vslice_b).unwrap();
        assert_eq!(result, 1 * 4 + 2 * 5 + 3 * 6);
    }

    #[test]
    fn test_vector_slice_dot_mismatched_length() {
        let a = [1, 2, 3];
        let b = [4, 5];
        let vslice_a: VectorSlice<'_, i32, Row> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let result = vslice_a.dot(&vslice_b);
        assert!(result.is_err());
    }

    // -- dot_to_f64 --

    #[test]
    fn test_vector_slice_dot_to_f64() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let result = vslice_a.dot_to_f64(&vslice_b).unwrap();
        assert_eq!(result, 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);
    }

    #[test]
    fn test_vector_slice_dot_to_f64_mismatched_length() {
        let a = [1, 2, 3];
        let b = [4, 5];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let result = vslice_a.dot_to_f64(&vslice_b);
        assert!(result.is_err());
    }

    // -- cross --

    #[test]
    fn test_vector_slice_cross() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        let vslice_a: VectorSlice<'_, i32, Row> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let result = vslice_a.cross(&vslice_b).unwrap();
        assert_eq!(result.as_slice(), &[-3, 6, -3]);
    }

    #[test]
    fn test_vector_slice_cross_incorrect_length() {
        let a = [1, 2, 3, 4];
        let b = [4, 5, 6, 7];
        let vslice_a: VectorSlice<'_, i32, Row> = VectorSlice::from_range(&a, 0..4);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let result = vslice_a.cross(&vslice_b);
        assert!(result.is_err());
    }

    // -- cross_into --

    #[test]
    fn test_vector_slice_cross_into_basic() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        let vslice_a: VectorSlice<'_, i32, Row> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let mut out = [0; 3];
        vslice_a.cross_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, [-3, 6, -3]);
    }

    #[test]
    fn test_vector_slice_cross_into_incorrect_length() {
        let a = [1, 2, 3, 4];
        let b = [4, 5, 6, 7];
        let vslice_a: VectorSlice<'_, i32, Row> = VectorSlice::from_range(&a, 0..4);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let mut out = [0; 3];
        let result = vslice_a.cross_into(&vslice_b, &mut out);
        assert!(result.is_err());

        let a = [1, 2, 3];
        let b = [4, 5, 6];
        let vslice_a: VectorSlice<'_, i32, Row> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let mut out = [0; 2];
        let result = vslice_a.cross_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_cross_into_empty() {
        let a: [i32; 0] = [];
        let b: [i32; 0] = [];
        let vslice_a: VectorSlice<'_, i32, Row> = VectorSlice::from_range(&a, 0..0);
        let vslice_b = VectorSlice::from_range(&b, 0..0);
        let mut out: [i32; 0] = [];
        let result = vslice_a.cross_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    // -- elementwise_min --

    #[test]
    fn test_vector_slice_elementwise_min() {
        let a = [1, 5, 3];
        let b = [4, 2, 6];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let result = vslice_a.elementwise_min(&vslice_b).unwrap();
        assert_eq!(result.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_vector_slice_elementwise_min_mismatched_length() {
        let a = [1, 5, 3];
        let b = [4, 2];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let result = vslice_a.elementwise_min(&vslice_b);
        assert!(result.is_err());
    }

    // -- elementwise_min_into --

    #[test]
    fn test_vector_slice_elementwise_min_into_basic() {
        let a = [1, 5, 3];
        let b = [4, 2, 6];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let mut out = [0; 3];
        vslice_a.elementwise_min_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, [1, 2, 3]);
    }

    #[test]
    fn test_vector_slice_elementwise_min_into_equal() {
        let a = [2, 2, 2];
        let b = [2, 2, 2];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let mut out = [0; 3];
        vslice_a.elementwise_min_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, [2, 2, 2]);
    }

    #[test]
    fn test_vector_slice_elementwise_min_into_mismatched_length() {
        let a = [1, 5, 3];
        let b = [4, 2];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let mut out = [0; 3];
        let result = vslice_a.elementwise_min_into(&vslice_b, &mut out);
        assert!(result.is_err());

        let a = [1, 5, 3];
        let b = [4, 2, 6];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let mut out = [0; 2];
        let result = vslice_a.elementwise_min_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_elementwise_min_into_empty() {
        let a: [i32; 0] = [];
        let b: [i32; 0] = [];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..0);
        let vslice_b = VectorSlice::from_range(&b, 0..0);
        let mut out: [i32; 0] = [];
        vslice_a.elementwise_min_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, []);
    }

    // -- elementwise_max --

    #[test]
    fn test_vector_slice_elementwise_max() {
        let a = [1, 5, 3];
        let b = [4, 2, 6];
        let vslice_a: VectorSlice<'_, i32, Row> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let result = vslice_a.elementwise_max(&vslice_b).unwrap();
        assert_eq!(result.as_slice(), &[4, 5, 6]);
    }

    #[test]
    fn test_vector_slice_elementwise_max_mismatched_length() {
        let a = [1, 5, 3];
        let b = [4, 2];
        let vslice_a: VectorSlice<'_, i32, Row> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let result = vslice_a.elementwise_max(&vslice_b);
        assert!(result.is_err());
    }

    // -- elementwise_max_into --

    #[test]
    fn test_vector_slice_elementwise_max_into_basic() {
        let a = [1, 5, 3];
        let b = [4, 2, 6];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let mut out = [0; 3];
        vslice_a.elementwise_max_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, [4, 5, 6]);
    }

    #[test]
    fn test_vector_slice_elementwise_max_into_equal() {
        let a = [2, 2, 2];
        let b = [2, 2, 2];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let mut out = [0; 3];
        vslice_a.elementwise_max_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, [2, 2, 2]);
    }

    #[test]
    fn test_vector_slice_elementwise_max_into_mismatched_length() {
        let a = [1, 5, 3];
        let b = [4, 2];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let mut out = [0; 3];
        let result = vslice_a.elementwise_max_into(&vslice_b, &mut out);
        assert!(result.is_err());

        let a = [1, 5, 3];
        let b = [4, 2, 6];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let mut out = [0; 2];
        let result = vslice_a.elementwise_max_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_elementwise_max_into_empty() {
        let a: [i32; 0] = [];
        let b: [i32; 0] = [];
        let vslice_a: VectorSlice<'_, i32, Column> = VectorSlice::from_range(&a, 0..0);
        let vslice_b = VectorSlice::from_range(&b, 0..0);
        let mut out: [i32; 0] = [];
        vslice_a.elementwise_max_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, []);
    }

    // -- VectorOpsFloat trait for VectorSlice --

    #[test]
    fn test_vector_slice_normalize() {
        let a = [3.0, 4.0];
        let vslice: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a, 0..2);
        let result = vslice.normalize().unwrap();
        let expected = [0.6, 0.8];
        for (x, y) in result.as_slice().iter().zip(expected.iter()) {
            assert!((x - y).abs() < 1e-8);
        }
    }

    #[test]
    fn test_vector_slice_normalize_zero_vector() {
        let a = [0.0, 0.0];
        let vslice: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a, 0..2);
        let result = vslice.normalize();
        assert!(result.is_err());
    }

    // -- normalize_into --

    #[test]
    fn test_vector_slice_normalize_into() {
        let a = [3.0, 4.0];
        let vslice: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a, 0..2);
        let mut out = [0.0; 2];
        vslice.normalize_into(&mut out).unwrap();
        assert!((out[0] - 0.6).abs() < 1e-8);
        assert!((out[1] - 0.8).abs() < 1e-8);
    }

    // -- normalize_to --

    #[test]
    fn test_vector_slice_normalize_to() {
        let a = [3.0, 4.0];
        let vslice: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a, 0..2);
        let result = vslice.normalize_to(10.0).unwrap();
        let expected = [6.0, 8.0];
        for (x, y) in result.as_slice().iter().zip(expected.iter()) {
            assert!((x - y).abs() < 1e-8);
        }
    }

    // -- normalize_to_into --

    #[test]
    fn test_vector_slice_normalize_to_into_basic() {
        let a = [3.0, 4.0];
        let vslice: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a, 0..2);
        let mut out = [0.0; 2];
        vslice.normalize_to_into(10.0, &mut out).unwrap();
        // The norm is 5.0, so the normalized vector with magnitude 10.0 should be [6.0, 8.0]
        assert!((out[0] - 6.0).abs() < 1e-8);
        assert!((out[1] - 8.0).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_normalize_to_into_zero_vector() {
        let a = [0.0, 0.0];
        let vslice: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a, 0..2);
        let mut out = [0.0; 2];
        let result = vslice.normalize_to_into(10.0, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_normalize_to_into_wrong_length() {
        let a = [3.0, 4.0];
        let vslice: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a, 0..2);
        let mut out = [0.0; 1];
        let result = vslice.normalize_to_into(10.0, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_normalize_to_into_empty() {
        let a: [f64; 0] = [];
        let vslice: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a, 0..0);
        let mut out: [f64; 0] = [];
        let result = vslice.normalize_to_into(10.0, &mut out);
        assert!(result.is_err());
    }

    // -- lerp --

    #[test]
    fn test_vector_slice_lerp() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let vslice_a: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let result = vslice_a.lerp(&vslice_b, 0.5).unwrap();
        assert_eq!(result.as_slice(), &[2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_vector_slice_lerp_weight_out_of_bounds() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        let vslice_a: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        assert!(vslice_a.lerp(&vslice_b, -0.1).is_err());
        assert!(vslice_a.lerp(&vslice_b, 1.1).is_err());
    }

    // -- lerp_into --

    #[test]
    fn test_vector_slice_lerp_into_basic() {
        let a_data = [1.0, 2.0, 3.0];
        let b_data = [4.0, 5.0, 6.0];
        let vslice_a: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a_data, 0..3);
        let vslice_b: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&b_data, 0..3);
        let mut out = [0.0; 3];
        vslice_a.lerp_into(&vslice_b, 0.5, &mut out).unwrap();
        assert_eq!(out, [2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_vector_slice_lerp_into_weight_zero() {
        let a_data = [1.0, 2.0, 3.0];
        let b_data = [4.0, 5.0, 6.0];
        let vslice_a: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a_data, 0..3);
        let vslice_b: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&b_data, 0..3);
        let mut out = [0.0; 3];
        vslice_a.lerp_into(&vslice_b, 0.0, &mut out).unwrap();
        assert_eq!(out, [1.0, 2.0, 3.0]); // Should be equal to a_data
    }

    #[test]
    fn test_vector_slice_lerp_into_weight_one() {
        let a_data = [1.0, 2.0, 3.0];
        let b_data = [4.0, 5.0, 6.0];
        let vslice_a: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a_data, 0..3);
        let vslice_b: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&b_data, 0..3);
        let mut out = [0.0; 3];
        vslice_a.lerp_into(&vslice_b, 1.0, &mut out).unwrap();
        assert_eq!(out, [4.0, 5.0, 6.0]); // Should be equal to b_data
    }

    #[test]
    fn test_vector_slice_lerp_into_mismatched_length_end() {
        let a_data = [1.0, 2.0, 3.0];
        let b_data = [4.0, 5.0];
        let vslice_a: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a_data, 0..3);
        let vslice_b: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&b_data, 0..2);
        let mut out = [0.0; 3];
        let result = vslice_a.lerp_into(&vslice_b, 0.5, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_lerp_into_mismatched_length_out() {
        let a_data = [1.0, 2.0, 3.0];
        let b_data = [4.0, 5.0, 6.0];
        let vslice_a: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a_data, 0..3);
        let vslice_b: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&b_data, 0..3);
        let mut out = [0.0; 2]; // out buffer is shorter
        let result = vslice_a.lerp_into(&vslice_b, 0.5, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_lerp_into_weight_out_of_bounds_low() {
        let a_data = [1.0, 2.0];
        let b_data = [3.0, 4.0];
        let vslice_a: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a_data, 0..2);
        let vslice_b: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&b_data, 0..2);
        let mut out = [0.0; 2];
        let result = vslice_a.lerp_into(&vslice_b, -0.1, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_lerp_into_weight_out_of_bounds_high() {
        let a_data = [1.0, 2.0];
        let b_data = [3.0, 4.0];
        let vslice_a: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a_data, 0..2);
        let vslice_b: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&b_data, 0..2);
        let mut out = [0.0; 2];
        let result = vslice_a.lerp_into(&vslice_b, 1.1, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_lerp_into_empty() {
        let a_data: [f64; 0] = [];
        let b_data: [f64; 0] = [];
        let vslice_a: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a_data, 0..0);
        let vslice_b: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&b_data, 0..0);
        let mut out: [f64; 0] = [];
        vslice_a.lerp_into(&vslice_b, 0.5, &mut out).unwrap();
        assert_eq!(out, [] as [f64; 0]);
    }

    // -- midpoint --

    #[test]
    fn test_vector_slice_midpoint() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let vslice_a: VectorSlice<'_, f64, Row> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let result = vslice_a.midpoint(&vslice_b).unwrap();
        assert_eq!(result.as_slice(), &[2.5, 3.5, 4.5]);
    }

    // -- midpoint_into --

    #[test]
    fn test_vector_slice_midpoint_into_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let vslice_a: VectorSlice<'_, f64, Row> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let mut out = [0.0; 3];
        vslice_a.midpoint_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, [2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_vector_slice_midpoint_into_weight_matches_midpoint() {
        let a = [10.0, 20.0];
        let b = [30.0, 40.0];
        let vslice_a: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let mut out = [0.0; 2];
        vslice_a.midpoint_into(&vslice_b, &mut out).unwrap();
        // Should match midpoint formula
        assert!((out[0] - 20.0).abs() < 1e-8);
        assert!((out[1] - 30.0).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_midpoint_into_mismatched_length_end() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0];
        let vslice_a: VectorSlice<'_, f64, Row> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let mut out = [0.0; 3];
        let result = vslice_a.midpoint_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_midpoint_into_mismatched_length_out() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let vslice_a: VectorSlice<'_, f64, Row> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let mut out = [0.0; 2];
        let result = vslice_a.midpoint_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_midpoint_into_empty() {
        let a: [f64; 0] = [];
        let b: [f64; 0] = [];
        let vslice_a: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a, 0..0);
        let vslice_b = VectorSlice::from_range(&b, 0..0);
        let mut out: [f64; 0] = [];
        vslice_a.midpoint_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, []);
    }

    // -- distance --

    #[test]
    fn test_vector_slice_distance() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 3.0];
        let vslice_a: VectorSlice<'_, f64, Row> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let result = vslice_a.distance(&vslice_b).unwrap();
        assert!((result - 5.0).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_manhattan_distance() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 3.0];
        let vslice_a: VectorSlice<'_, f64, Row> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let result = vslice_a.manhattan_distance(&vslice_b).unwrap();
        assert!((result - 7.0).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_chebyshev_distance() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 3.0];
        let vslice_a: VectorSlice<'_, f64, Row> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let result = vslice_a.chebyshev_distance(&vslice_b).unwrap();
        assert!((result - 4.0).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_minkowski_distance() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 3.0];
        let vslice_a: VectorSlice<'_, f64, Row> = VectorSlice::from_range(&a, 0..3);
        let vslice_b = VectorSlice::from_range(&b, 0..3);
        let result = vslice_a.minkowski_distance(&vslice_b, 3.0).unwrap();
        assert!((result - 4.497941445275415).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_angle_with() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let vslice_a: VectorSlice<'_, f64, Row> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let result = vslice_a.angle_with(&vslice_b).unwrap();
        assert!((result - std::f64::consts::FRAC_PI_2).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_project_onto() {
        let a = [3.0, 4.0];
        let b = [6.0, 8.0];
        let vslice_a: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let result = vslice_a.project_onto(&vslice_b).unwrap();
        let expected = [3.0, 4.0];
        for (x, y) in result.as_slice().iter().zip(expected.iter()) {
            assert!((x - y).abs() < 1e-8);
        }
    }

    #[test]
    fn test_vector_slice_cosine_similarity() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let vslice_a: VectorSlice<'_, f64, Row> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let result = vslice_a.cosine_similarity(&vslice_b).unwrap();
        assert!((result - 0.0).abs() < 1e-8);
    }

    // -- project_onto_into --

    #[test]
    fn test_vector_slice_project_onto_into_basic() {
        let a = [3.0, 4.0];
        let b = [6.0, 8.0];
        let vslice_a: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let mut out = [0.0; 2];
        vslice_a.project_onto_into(&vslice_b, &mut out).unwrap();
        // a is already parallel to b, so projection should be a
        assert!((out[0] - 3.0).abs() < 1e-8);
        assert!((out[1] - 4.0).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_project_onto_into_parallel() {
        let a = [2.0, 4.0];
        let b = [1.0, 2.0];
        let vslice_a: VectorSlice<'_, f64, Row> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let mut out = [0.0; 2];
        vslice_a.project_onto_into(&vslice_b, &mut out).unwrap();
        // a is parallel to b, so projection should be a
        assert!((out[0] - 2.0).abs() < 1e-8);
        assert!((out[1] - 4.0).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_project_onto_into_orthogonal() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let vslice_a: VectorSlice<'_, f64, Row> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let mut out = [99.0, 99.0];
        vslice_a.project_onto_into(&vslice_b, &mut out).unwrap();
        // a is orthogonal to b, so projection should be [0, 0]
        assert!((out[0]).abs() < 1e-8);
        assert!((out[1]).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_project_onto_into_identical() {
        let a = [5.0, 5.0];
        let b = [5.0, 5.0];
        let vslice_a: VectorSlice<'_, f64, Row> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let mut out = [0.0, 0.0];
        vslice_a.project_onto_into(&vslice_b, &mut out).unwrap();
        assert!((out[0] - 5.0).abs() < 1e-8);
        assert!((out[1] - 5.0).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_project_onto_into_zero_vector() {
        let a = [1.0, 2.0];
        let b = [0.0, 0.0];
        let vslice_a: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let mut out = [0.0, 0.0];
        let result = vslice_a.project_onto_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_project_onto_into_mismatched_length_other() {
        let a = [1.0, 2.0];
        let b = [3.0];
        let vslice_a: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..1);
        let mut out = [0.0, 0.0];
        let result = vslice_a.project_onto_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_project_onto_into_mismatched_length_out() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        let vslice_a: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let mut out = [0.0; 1];
        let result = vslice_a.project_onto_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_project_onto_into_empty() {
        let a: [f64; 0] = [];
        let b: [f64; 0] = [];
        let vslice_a: VectorSlice<'_, f64, Column> = VectorSlice::from_range(&a, 0..0);
        let vslice_b = VectorSlice::from_range(&b, 0..0);
        let mut out: [f64; 0] = [];
        let result = vslice_a.project_onto_into(&vslice_b, &mut out);
        assert!(result.is_err()); // zero vector error
    }

    // -- VectorOpsComplex for VectorSlice --

    // -- normalize --

    #[test]
    fn test_vector_slice_complex_normalize() {
        use num::Complex;
        let a = [Complex::new(3.0, 4.0), Complex::new(0.0, 0.0)];
        let vslice: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..2);
        // The norm is sqrt(|3+4i|^2 + |0|^2) = sqrt(25) = 5
        let result = vslice.normalize().unwrap();
        let expected = [Complex::new(3.0 / 5.0, 4.0 / 5.0), Complex::new(0.0, 0.0)];
        for (x, y) in result.as_slice().iter().zip(expected.iter()) {
            assert!((x.re - y.re).abs() < 1e-8);
            assert!((x.im - y.im).abs() < 1e-8);
        }
    }

    #[test]
    fn test_vector_slice_complex_normalize_zero_vector() {
        use num::Complex;
        let a = [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let vslice: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..2);
        let result = vslice.normalize();
        assert!(result.is_err());
    }

    // -- normalize_to --

    #[test]
    fn test_vector_slice_complex_normalize_to() {
        use num::Complex;
        let a = [Complex::new(3.0, 4.0), Complex::new(0.0, 0.0)];
        let vslice: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..2);
        // The norm is 5, so scaling to magnitude 10 multiplies by 2
        let result = vslice.normalize_to(10.0).unwrap();
        let expected = [Complex::new(6.0, 8.0), Complex::new(0.0, 0.0)];
        for (x, y) in result.as_slice().iter().zip(expected.iter()) {
            assert!((x.re - y.re).abs() < 1e-8);
            assert!((x.im - y.im).abs() < 1e-8);
        }
    }

    #[test]
    fn test_vector_slice_complex_normalize_to_zero_vector() {
        use num::Complex;
        let a = [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let vslice: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..2);
        let result = vslice.normalize_to(10.0);
        assert!(result.is_err());
    }

    // -- dot --

    #[test]
    fn test_vector_slice_complex_dot_basic() {
        use num::Complex;
        let a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        // Hermitian dot: conj(a0)*b0 + conj(a1)*b1
        let expected = a[0].conj() * b[0] + a[1].conj() * b[1];
        let result = VectorOpsComplex::dot(&vslice_a, &vslice_b).unwrap();
        assert!((result.re - expected.re).abs() < 1e-12);
        assert!((result.im - expected.im).abs() < 1e-12);
    }

    #[test]
    fn test_vector_slice_complex_dot_negative_values() {
        use num::Complex;
        let a = [Complex::new(-1.0, -2.0), Complex::new(-3.0, -4.0)];
        let b = [Complex::new(2.0, 1.0), Complex::new(4.0, 3.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Row> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let expected = a[0].conj() * b[0] + a[1].conj() * b[1];
        let result = VectorOpsComplex::dot(&vslice_a, &vslice_b).unwrap();
        assert!((result.re - expected.re).abs() < 1e-12);
        assert!((result.im - expected.im).abs() < 1e-12);
    }

    #[test]
    fn test_vector_slice_complex_dot_zero_vector() {
        use num::Complex;
        let a = [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let b = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let result = VectorOpsComplex::dot(&vslice_a, &vslice_b).unwrap();
        assert!((result.re).abs() < 1e-12);
        assert!((result.im).abs() < 1e-12);
    }

    #[test]
    fn test_vector_slice_complex_dot_empty() {
        use num::Complex;
        let a: [Complex<f64>; 0] = [];
        let b: [Complex<f64>; 0] = [];
        let vslice_a: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..0);
        let vslice_b = VectorSlice::from_range(&b, 0..0);
        let result = VectorOpsComplex::dot(&vslice_a, &vslice_b).unwrap();
        assert!((result.re).abs() < 1e-12);
        assert!((result.im).abs() < 1e-12);
    }

    #[test]
    fn test_vector_slice_complex_dot_mismatched_length() {
        use num::Complex;
        let a = [Complex::new(1.0, 2.0)];
        let b = [Complex::new(3.0, 4.0), Complex::new(5.0, 6.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..1);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let result = VectorOpsComplex::dot(&vslice_a, &vslice_b);
        assert!(result.is_err());
    }

    // -- lerp --

    #[test]
    fn test_vector_slice_complex_lerp() {
        use num::Complex;
        let a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        // Lerp with weight 0.25
        let result = vslice_a.lerp(&vslice_b, 0.25).unwrap();
        let expected = [
            Complex::new(1.0 + 0.25 * (5.0 - 1.0), 2.0 + 0.25 * (6.0 - 2.0)),
            Complex::new(3.0 + 0.25 * (7.0 - 3.0), 4.0 + 0.25 * (8.0 - 4.0)),
        ];
        for (x, y) in result.as_slice().iter().zip(expected.iter()) {
            assert!((x.re - y.re).abs() < 1e-8);
            assert!((x.im - y.im).abs() < 1e-8);
        }
    }

    #[test]
    fn test_vector_slice_complex_lerp_weight_out_of_bounds() {
        use num::Complex;
        let a = [Complex::new(1.0, 2.0)];
        let b = [Complex::new(3.0, 4.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..1);
        let vslice_b = VectorSlice::from_range(&b, 0..1);
        assert!(vslice_a.lerp(&vslice_b, -0.1).is_err());
        assert!(vslice_a.lerp(&vslice_b, 1.1).is_err());
    }

    // -- midpoint --

    #[test]
    fn test_vector_slice_complex_midpoint() {
        use num::Complex;
        let a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Row> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let result = vslice_a.midpoint(&vslice_b).unwrap();
        let expected = [
            Complex::new((1.0 + 5.0) / 2.0, (2.0 + 6.0) / 2.0),
            Complex::new((3.0 + 7.0) / 2.0, (4.0 + 8.0) / 2.0),
        ];
        for (x, y) in result.as_slice().iter().zip(expected.iter()) {
            assert!((x.re - y.re).abs() < 1e-8);
            assert!((x.im - y.im).abs() < 1e-8);
        }
    }

    // -- distance --

    #[test]
    fn test_vector_slice_complex_distance() {
        use num::Complex;
        let a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        // Euclidean distance: sqrt(sum_i |a[i] - b[i]|^2)
        let d0 = (a[0] - b[0]).norm_sqr();
        let d1 = (a[1] - b[1]).norm_sqr();
        let expected = (d0 + d1).sqrt();
        let dist = vslice_a.distance(&vslice_b).unwrap();
        assert!((dist - expected).abs() < 1e-12);
    }

    // -- manhattan_distance --

    #[test]
    fn test_vector_slice_complex_manhattan_distance() {
        use num::Complex;
        let a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Row> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        // Manhattan distance: sum_i |a[i] - b[i]|
        let d0 = (a[0] - b[0]).norm();
        let d1 = (a[1] - b[1]).norm();
        let expected = d0 + d1;
        let dist = vslice_a.manhattan_distance(&vslice_b).unwrap();
        assert!((dist - expected).abs() < 1e-12);
    }

    // -- chebyshev_distance --

    #[test]
    fn test_vector_slice_complex_chebyshev_distance() {
        use num::Complex;
        let a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        // Chebyshev distance: max_i |a[i] - b[i]|
        let d0 = (a[0] - b[0]).norm();
        let d1 = (a[1] - b[1]).norm();
        let expected = d0.max(d1);
        let dist = vslice_a.chebyshev_distance(&vslice_b).unwrap();
        assert!((dist - expected).abs() < 1e-12);
    }

    // -- minkowski_distance --

    #[test]
    fn test_vector_slice_complex_minkowski_distance() {
        use num::Complex;
        let a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Row> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let p = 3.0;
        // Minkowski distance: (|a[0]-b[0]|^p + |a[1]-b[1]|^p)^(1/p)
        let d0 = (a[0] - b[0]).norm().powf(p);
        let d1 = (a[1] - b[1]).norm().powf(p);
        let expected = (d0 + d1).powf(1.0 / p);
        let dist = vslice_a.minkowski_distance(&vslice_b, p).unwrap();
        assert!((dist - expected).abs() < 1e-12);
    }

    // -- project_onto --

    #[test]
    fn test_vector_slice_complex_project_onto_basic() {
        use num::Complex;
        let a = [Complex::new(3.0, 4.0), Complex::new(0.0, 0.0)];
        let b = [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        // Project a onto b: should be [3.0 - 4.0i, 0.0]
        let proj = vslice_a.project_onto(&vslice_b).unwrap();
        assert!((proj.as_slice()[0] - Complex::new(3.0, -4.0)).norm() < 1e-12);
        assert!((proj.as_slice()[1] - Complex::new(0.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_vector_slice_complex_project_onto_parallel() {
        use num::Complex;
        let a = [Complex::new(2.0, 2.0), Complex::new(4.0, 4.0)];
        let b = [Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Row> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let proj = vslice_a.project_onto(&vslice_b).unwrap();
        assert!((proj.as_slice()[0] - Complex::new(2.0, 2.0)).norm() < 1e-12);
        assert!((proj.as_slice()[1] - Complex::new(4.0, 4.0)).norm() < 1e-12);
    }

    #[test]
    fn test_vector_slice_complex_project_onto_orthogonal() {
        use num::Complex;
        let a = [Complex::new(0.0, 1.0), Complex::new(0.0, 0.0)];
        let b = [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let proj = vslice_a.project_onto(&vslice_b).unwrap();
        assert!((proj.as_slice()[0] - Complex::new(0.0, -1.0)).norm() < 1e-12);
        assert!((proj.as_slice()[1] - Complex::new(0.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_vector_slice_complex_project_onto_identical() {
        use num::Complex;
        let a = [Complex::new(5.0, 5.0), Complex::new(5.0, 5.0)];
        let b = [Complex::new(5.0, 5.0), Complex::new(5.0, 5.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Row> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let proj = vslice_a.project_onto(&vslice_b).unwrap();
        assert!((proj.as_slice()[0] - Complex::new(5.0, 5.0)).norm() < 1e-12);
        assert!((proj.as_slice()[1] - Complex::new(5.0, 5.0)).norm() < 1e-12);
    }

    #[test]
    fn test_vector_slice_complex_project_onto_zero_vector() {
        use num::Complex;
        let a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let b = [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let result = vslice_a.project_onto(&vslice_b);
        assert!(result.is_err());
    }

    // -- cosine_similarity --

    #[test]
    fn test_vector_slice_complex_cosine_similarity_parallel() {
        use num::Complex;
        let a = [Complex::new(1.0, 2.0), Complex::new(2.0, 4.0)];
        let b = [Complex::new(2.0, 4.0), Complex::new(4.0, 8.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let cos_sim = vslice_a.cosine_similarity(&vslice_b).unwrap();
        assert!((cos_sim - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_vector_slice_complex_cosine_similarity_orthogonal() {
        use num::Complex;
        let a = [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        let b = [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..2);
        let vslice_b = VectorSlice::from_range(&b, 0..2);
        let cos_sim = vslice_a.cosine_similarity(&vslice_b).unwrap();
        assert!((cos_sim - Complex::new(0.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_vector_slice_complex_cosine_similarity_opposite() {
        use num::Complex;
        let a = [Complex::new(1.0, 0.0)];
        let b = [Complex::new(-1.0, 0.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..1);
        let vslice_b = VectorSlice::from_range(&b, 0..1);
        let cos_sim = vslice_a.cosine_similarity(&vslice_b).unwrap();
        assert!((cos_sim + Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_vector_slice_complex_cosine_similarity_identical() {
        use num::Complex;
        let a = [Complex::new(3.0, 4.0)];
        let b = [Complex::new(3.0, 4.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..1);
        let vslice_b = VectorSlice::from_range(&b, 0..1);
        let cos_sim = vslice_a.cosine_similarity(&vslice_b).unwrap();
        assert!((cos_sim - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_vector_slice_complex_cosine_similarity_arbitrary() {
        use num::Complex;
        let a = [Complex::new(1.0, 2.0)];
        let b = [Complex::new(2.0, 1.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..1);
        let vslice_b = VectorSlice::from_range(&b, 0..1);
        let cos_sim = vslice_a.cosine_similarity(&vslice_b).unwrap();
        assert!(cos_sim.norm() <= 1.0 + 1e-12);
    }

    #[test]
    fn test_vector_slice_complex_cosine_similarity_zero_vector() {
        use num::Complex;
        let a = [Complex::new(0.0, 0.0)];
        let b = [Complex::new(1.0, 2.0)];
        let vslice_a: VectorSlice<'_, Complex<f64>, Column> = VectorSlice::from_range(&a, 0..1);
        let vslice_b = VectorSlice::from_range(&b, 0..1);
        let result = vslice_a.cosine_similarity(&vslice_b);
        assert!(result.is_err());
    }

    // /////////////////////////////////
    // ================================
    //
    // VectorSliceMut type
    //
    // ================================
    // /////////////////////////////////

    // -- new --
    #[test]
    fn test_vector_slice_mut_new() {
        let mut data = [1, 2, 3, 4, 5];
        let vslice: VectorSliceMut<'_, i32, Column> = VectorSliceMut::new(&mut data);
        assert_eq!(vslice.elements, &mut [1, 2, 3, 4, 5]);
        // Mutate through the slice
        vslice.elements[0] = 10;
        assert_eq!(vslice.elements[0], 10);
        assert_eq!(data[0], 10);
    }

    #[test]
    fn test_vector_slice_mut_new_complex() {
        let mut data = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0), Complex::new(5.0, 6.0)];
        let vslice: VectorSliceMut<'_, Complex<f64>, Column> = VectorSliceMut::new(&mut data);
        assert_eq!(
            vslice.elements,
            &mut [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0), Complex::new(5.0, 6.0),]
        );
        // Mutate through the slice
        vslice.elements[0] = Complex::new(7.0, 8.0);
        assert_eq!(vslice.elements[0], Complex::new(7.0, 8.0));
    }

    // -- from_range --
    #[test]
    fn test_vector_slice_mut_from_range_middle() {
        let mut data = [10, 20, 30, 40, 50];
        {
            let vslice: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut data, 1..4);
            assert_eq!(vslice.elements, &mut [20, 30, 40]);
            vslice.elements[1] = 99;
        }
        assert_eq!(data, [10, 20, 99, 40, 50]);
    }

    #[test]
    fn test_vector_slice_mut_from_range_full() {
        let mut data = [7, 8, 9];
        {
            let vslice: VectorSliceMut<'_, i32, Column> =
                VectorSliceMut::from_range(&mut data, 0..3);
            assert_eq!(vslice.elements, &mut [7, 8, 9]);
            vslice.elements[2] = 42;
        }
        assert_eq!(data, [7, 8, 42]);
    }

    #[test]
    fn test_vector_slice_mut_from_range_empty() {
        let mut data = [1, 2, 3];
        let vslice: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut data, 1..1);
        assert_eq!(vslice.elements, &mut []);
    }

    #[test]
    fn test_vector_slice_mut_from_range_middle_complex() {
        let mut data = [
            Complex::new(10.0, 1.0),
            Complex::new(20.0, 2.0),
            Complex::new(30.0, 3.0),
            Complex::new(40.0, 4.0),
            Complex::new(50.0, 5.0),
        ];
        {
            let vslice: VectorSliceMut<'_, Complex<f64>, Row> =
                VectorSliceMut::from_range(&mut data, 1..4);
            assert_eq!(
                vslice.elements,
                &mut [Complex::new(20.0, 2.0), Complex::new(30.0, 3.0), Complex::new(40.0, 4.0),]
            );
            vslice.elements[2] = Complex::new(99.0, 99.0);
        }
        assert_eq!(
            data,
            [
                Complex::new(10.0, 1.0),
                Complex::new(20.0, 2.0),
                Complex::new(30.0, 3.0),
                Complex::new(99.0, 99.0),
                Complex::new(50.0, 5.0),
            ]
        );
    }

    #[test]
    fn test_vector_slice_mut_from_range_full_complex() {
        let mut data = [Complex::new(7.0, 0.0), Complex::new(8.0, 1.0), Complex::new(9.0, 2.0)];
        {
            let vslice: VectorSliceMut<'_, Complex<f64>, Column> =
                VectorSliceMut::from_range(&mut data, 0..3);
            assert_eq!(
                vslice.elements,
                &mut [Complex::new(7.0, 0.0), Complex::new(8.0, 1.0), Complex::new(9.0, 2.0),]
            );
            vslice.elements[1] = Complex::new(42.0, 24.0);
        }
        assert_eq!(
            data,
            [Complex::new(7.0, 0.0), Complex::new(42.0, 24.0), Complex::new(9.0, 2.0),]
        );
    }

    #[test]
    fn test_vector_slice_mut_from_range_empty_complex() {
        let mut data = [Complex::new(1.0, 1.0), Complex::new(2.0, 2.0), Complex::new(3.0, 3.0)];
        let vslice: VectorSliceMut<'_, Complex<f64>, Row> =
            VectorSliceMut::from_range(&mut data, 1..1);
        assert_eq!(vslice.elements, &mut []);
    }

    // -- Deref trait for VectorSliceMut --
    #[test]
    fn test_vector_slice_mut_deref_access() {
        let mut data = [1, 2, 3, 4, 5];
        let vslice: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut data, 1..4);
        // Deref allows direct indexing
        assert_eq!(vslice[0], 2);
        assert_eq!(vslice[1], 3);
        assert_eq!(vslice[2], 4);
        // Deref allows using slice methods
        assert_eq!(vslice.len(), 3);
        assert!(vslice.contains(&3));
        assert_eq!(vslice.iter().sum::<i32>(), 9);
    }

    #[test]
    fn test_vector_slice_mut_deref_complex() {
        let mut data = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0), Complex::new(5.0, 6.0)];
        let mut vslice: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut data, 0..2);
        // Deref allows direct indexing
        assert_eq!(vslice[0], Complex::new(1.0, 2.0));
        assert_eq!(vslice[1], Complex::new(3.0, 4.0));
        // DerefMut allows mutation
        vslice[1] = Complex::new(7.0, 8.0);
        assert_eq!(vslice[1], Complex::new(7.0, 8.0));
    }

    // -- DerefMut trait for VectorSliceMut --
    #[test]
    fn test_vector_slice_mut_deref_mut_access() {
        let mut fv: FlexVector<i32, Column> = FlexVector::from([10, 20, 30, 40, 50]);
        {
            let mut vslice: VectorSliceMut<'_, i32, Column> =
                VectorSliceMut::from_range(&mut fv, 1..4);
            // DerefMut allows mutation through indexing
            vslice[0] = 100;
            vslice[2] = 400;
            // DerefMut allows using mutable slice methods
            vslice.reverse();
        }
        // After mutation, check the underlying data
        assert_eq!(fv.as_slice(), &[10, 400, 30, 100, 50]);
    }

    // -- AsMut trait for VectorSliceMut --
    #[test]
    fn test_vector_slice_mut_as_mut_basic() {
        let mut data = [1, 2, 3, 4, 5];
        let mut vslice: VectorSliceMut<'_, i32, Column> =
            VectorSliceMut::from_range(&mut data, 2..5);
        let slice_mut: &mut [i32] = vslice.as_mut();
        slice_mut[0] = 10;
        slice_mut[2] = 50;
        assert_eq!(slice_mut, &[10, 4, 50]);
        // Changes are reflected in the original data
        assert_eq!(data, [1, 2, 10, 4, 50]);
    }

    #[test]
    fn test_vector_slice_mut_as_mut_complex() {
        let mut data = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0), Complex::new(5.0, 6.0)];
        let mut vslice: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut data, 1..3);
        let slice_mut: &mut [Complex<f64>] = vslice.as_mut();
        slice_mut[1] = Complex::new(9.0, 9.0);
        assert_eq!(slice_mut, &[Complex::new(3.0, 4.0), Complex::new(9.0, 9.0)]);
        // Changes are reflected in the original data
        assert_eq!(data, [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0), Complex::new(9.0, 9.0)]);
    }

    #[test]
    fn test_vector_slice_mut_as_mut_with_std_function() {
        let mut data = [10, 20, 30, 40];
        let mut vslice: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut data, 1..4);
        // Use a standard library function that takes AsMut<[i32]>
        vslice.as_mut().reverse();
        assert_eq!(vslice.as_mut(), &[40, 30, 20]);
        // Changes are reflected in the original data
        assert_eq!(data, [10, 40, 30, 20]);
    }

    // -- IntoIterator trait for VectorSliceMut --
    #[test]
    fn test_vector_slice_mut_into_iter_basic() {
        let mut data = [1, 2, 3, 4];
        let vslice: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut data, 1..4);
        let collected: Vec<i32> = vslice
            .into_iter()
            .map(|x| {
                *x += 10;
                *x
            })
            .collect();
        assert_eq!(collected, vec![12, 13, 14]);
        // The original data is also updated
        assert_eq!(data, [1, 12, 13, 14]);
    }

    #[test]
    fn test_vector_slice_mut_into_iter_ref() {
        let mut data = [10, 20, 30];
        let mut vslice: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut data, 0..2);
        {
            for x in &mut vslice {
                *x *= 2;
            }
        }
        // The original data is updated
        assert_eq!(data, [20, 40, 30]);
    }

    #[test]
    fn test_vector_slice_mut_into_iter_complex() {
        let mut data = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0), Complex::new(5.0, 6.0)];
        let vslice: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut data, 0..3);
        for x in vslice.into_iter() {
            x.im += 1.0;
        }
        assert_eq!(data, [Complex::new(1.0, 3.0), Complex::new(3.0, 5.0), Complex::new(5.0, 7.0)]);
    }

    // -- Display trait for VectorSliceMut --

    #[test]
    fn test_vector_slice_mut_display() {
        let mut data = [10, 20, 30];
        let vslice: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut data, 0..3);
        let display = format!("{}", vslice);
        assert!(display.contains("Column VectorSliceMut"));
        assert!(display.contains("[10, 20, 30]"));
    }

    #[test]
    fn test_vector_slice_mut_display_complex() {
        let mut data = [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        let vslice: VectorSliceMut<'_, Complex<f64>, Row> =
            VectorSliceMut::from_range(&mut data, 0..2);
        let display = format!("{}", vslice);
        assert!(display.contains("Row VectorSliceMut"));
        assert!(display.contains("Complex { re: 5.0, im: 6.0 }"));
        assert!(display.contains("Complex { re: 7.0, im: 8.0 }"));
    }

    // -- Debug trait for VectorSliceMut --

    #[test]
    fn test_vector_slice_mut_debug() {
        let mut data = [10, 20, 30];
        let vslice: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut data, 0..3);
        let debug = format!("{:?}", vslice);
        assert!(debug.contains("VectorSliceMut"));
        assert!(debug.contains("orientation"));
        assert!(debug.contains("elements"));
        assert!(debug.contains("10"));
        assert!(debug.contains("20"));
        assert!(debug.contains("30"));
    }

    #[test]
    fn test_vector_slice_mut_debug_complex() {
        let mut data = [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        let vslice: VectorSliceMut<'_, Complex<f64>, Row> =
            VectorSliceMut::from_range(&mut data, 0..2);
        let debug = format!("{:?}", vslice);
        assert!(debug.contains("VectorSliceMut"));
        assert!(debug.contains("orientation"));
        assert!(debug.contains("elements"));
        assert!(debug.contains("Complex { re: 5.0, im: 6.0 }"));
        assert!(debug.contains("Complex { re: 7.0, im: 8.0 }"));
    }

    // -- VectorBase trait for VectorSliceMut --

    #[test]
    fn test_vector_slice_mut_as_slice_basic() {
        let mut data = [1, 2, 3, 4];
        let vslice: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut data, 1..3);
        assert_eq!(vslice.as_slice(), &[2, 3]);
    }

    #[test]
    fn test_vector_slice_mut_as_slice_full_range() {
        let mut data = [10, 20, 30];
        let length = data.len();
        let vslice: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut data, 0..length);
        assert_eq!(vslice.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_vector_slice_mut_as_slice_empty() {
        let mut data = [1, 2, 3];
        let vslice: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut data, 1..1);
        assert_eq!(vslice.as_slice(), &[]);
    }

    #[test]
    fn test_vector_slice_mut_as_slice_complex() {
        let mut data = [num::Complex::new(1.0, 2.0), num::Complex::new(3.0, 4.0)];
        let vslice: VectorSliceMut<'_, num::Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut data, 0..2);
        assert_eq!(vslice.as_slice(), &[num::Complex::new(1.0, 2.0), num::Complex::new(3.0, 4.0)]);
    }

    #[test]
    fn test_vector_slice_mut_len_and_is_empty() {
        let mut data = [1, 2, 3];
        let vslice: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut data, 0..2);
        assert_eq!(vslice.len(), 2);
        assert!(!vslice.is_empty());
        let empty: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut data, 1..1);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    // -- VectorBaseMut trait for VectorSliceMut --

    #[test]
    fn test_vector_slice_mut_as_mut_slice_basic() {
        let mut data = [1, 2, 3, 4];
        let mut vslice: VectorSliceMut<'_, i32, Column> =
            VectorSliceMut::from_range(&mut data, 1..3);
        assert_eq!(vslice.as_mut_slice(), &mut [2, 3]);
        // Mutate through as_mut_slice
        vslice.as_mut_slice()[0] = 20;
        vslice.as_mut_slice()[1] = 30;
        assert_eq!(vslice.as_mut_slice(), &mut [20, 30]);
        // Changes are reflected in the original data
        assert_eq!(data, [1, 20, 30, 4]);
    }

    #[test]
    fn test_vector_slice_mut_as_mut_slice_full_range() {
        let mut data = [10, 20, 30];
        let length = data.len();
        let mut vslice: VectorSliceMut<'_, i32, Row> =
            VectorSliceMut::from_range(&mut data, 0..length);
        assert_eq!(vslice.as_mut_slice(), &mut [10, 20, 30]);
        vslice.as_mut_slice()[2] = 99;
        assert_eq!(data, [10, 20, 99]);
    }

    #[test]
    fn test_vector_slice_mut_as_mut_slice_empty() {
        let mut data = [1, 2, 3];
        let mut vslice: VectorSliceMut<'_, i32, Column> =
            VectorSliceMut::from_range(&mut data, 1..1);
        assert_eq!(vslice.as_mut_slice(), &mut []);
    }

    #[test]
    fn test_vector_slice_mut_as_mut_slice_complex() {
        let mut data = [num::Complex::new(1.0, 2.0), num::Complex::new(3.0, 4.0)];
        let mut vslice: VectorSliceMut<'_, num::Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut data, 0..2);
        assert_eq!(
            vslice.as_mut_slice(),
            &mut [num::Complex::new(1.0, 2.0), num::Complex::new(3.0, 4.0)]
        );
        // Mutate through as_mut_slice
        vslice.as_mut_slice()[1] = num::Complex::new(9.0, 9.0);
        assert_eq!(data, [num::Complex::new(1.0, 2.0), num::Complex::new(9.0, 9.0)]);
    }

    // -- VectorOps trait for VectorSliceMut --

    // -- translate --

    #[test]
    fn test_vector_slice_mut_translate() {
        let mut a = [1, 2, 3];
        let mut b = [4, 5, 6];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut b, 0..3);
        let result = vslice_a.translate(&vslice_b).unwrap();
        assert_eq!(result.as_slice(), &[5, 7, 9]);
    }

    #[test]
    fn test_vector_slice_mut_translate_mismatched_length() {
        let mut a = [1, 2, 3];
        let mut b = [4, 5];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut b, 0..2);
        let result = vslice_a.translate(&vslice_b);
        assert!(result.is_err());
    }

    // -- translate_into --

    #[test]
    fn test_vector_slice_mut_translate_into_basic() {
        let mut a = [1, 2, 3];
        let mut b = [4, 5, 6];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let mut out = [0; 3];
        vslice_a.translate_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, [5, 7, 9]);
    }

    #[test]
    fn test_vector_slice_mut_translate_into_empty() {
        let mut a: [i32; 0] = [];
        let mut b: [i32; 0] = [];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..0);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..0);
        let mut out: [i32; 0] = [];
        vslice_a.translate_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, []);
    }

    #[test]
    fn test_vector_slice_mut_translate_into_mismatched_length() {
        let mut a = [1, 2, 3];
        let mut b = [4, 5];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let mut out = [0; 3];
        let result = vslice_a.translate_into(&vslice_b, &mut out);
        assert!(result.is_err());

        let mut a = [1, 2, 3];
        let mut b = [4, 5, 6];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let mut out = [0; 2];
        let result = vslice_a.translate_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    // -- dot --

    #[test]
    fn test_vector_slice_mut_dot() {
        let mut a = [1, 2, 3];
        let mut b = [4, 5, 6];
        let vslice_a: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut b, 0..3);
        let result = vslice_a.dot(&vslice_b).unwrap();
        assert_eq!(result, 1 * 4 + 2 * 5 + 3 * 6);
    }

    #[test]
    fn test_vector_slice_mut_dot_mismatched_length() {
        let mut a = [1, 2, 3];
        let mut b = [4, 5];
        let vslice_a: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut b, 0..2);
        let result = vslice_a.dot(&vslice_b);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_dot_to_f64() {
        let mut a = [1, 2, 3];
        let mut b = [4, 5, 6];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut b, 0..3);
        let result = vslice_a.dot_to_f64(&vslice_b).unwrap();
        assert_eq!(result, 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);
    }

    #[test]
    fn test_vector_slice_mut_dot_to_f64_mismatched_length() {
        let mut a = [1, 2, 3];
        let mut b = [4, 5];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut b, 0..2);
        let result = vslice_a.dot_to_f64(&vslice_b);
        assert!(result.is_err());
    }

    // -- cross --

    #[test]
    fn test_vector_slice_mut_cross() {
        let mut a = [1, 2, 3];
        let mut b = [4, 5, 6];
        let vslice_a: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut b, 0..3);
        let result = vslice_a.cross(&vslice_b).unwrap();
        assert_eq!(result.as_slice(), &[-3, 6, -3]);
    }

    #[test]
    fn test_vector_slice_mut_cross_incorrect_length() {
        let mut a = [1, 2, 3, 4];
        let mut b = [4, 5, 6, 7];
        let vslice_a: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut a, 0..4);
        let vslice_b: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut b, 0..3);
        let result = vslice_a.cross(&vslice_b);
        assert!(result.is_err());
    }

    // -- cross_into --

    #[test]
    fn test_vector_slice_mut_cross_into_basic() {
        let mut a = [1, 2, 3];
        let mut b = [4, 5, 6];
        let vslice_a: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut b, 0..3);
        let mut out = [0; 3];
        vslice_a.cross_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, [-3, 6, -3]);
    }

    #[test]
    fn test_vector_slice_mut_cross_into_incorrect_length() {
        let mut a = [1, 2, 3, 4];
        let mut b = [4, 5, 6, 7];
        let vslice_a: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut a, 0..4);
        let vslice_b: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut b, 0..3);
        let mut out = [0; 3];
        let result = vslice_a.cross_into(&vslice_b, &mut out);
        assert!(result.is_err());

        let mut a = [1, 2, 3];
        let mut b = [4, 5, 6];
        let vslice_a: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut b, 0..3);
        let mut out = [0; 2];
        let result = vslice_a.cross_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_cross_into_empty() {
        let mut a: [i32; 0] = [];
        let mut b: [i32; 0] = [];
        let vslice_a: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut a, 0..0);
        let vslice_b: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut b, 0..0);
        let mut out: [i32; 0] = [];
        let result = vslice_a.cross_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    // -- elementwise_min --

    #[test]
    fn test_vector_slice_mut_elementwise_min() {
        let mut a = [1, 5, 3];
        let mut b = [4, 2, 6];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut b, 0..3);
        let result = vslice_a.elementwise_min(&vslice_b).unwrap();
        assert_eq!(result.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_vector_slice_mut_elementwise_min_mismatched_length() {
        let mut a = [1, 5, 3];
        let mut b = [4, 2];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut b, 0..2);
        let result = vslice_a.elementwise_min(&vslice_b);
        assert!(result.is_err());
    }

    // -- elementwise_min_into --

    #[test]
    fn test_vector_slice_mut_elementwise_min_into_basic() {
        let mut a = [1, 5, 3];
        let mut b = [4, 2, 6];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let mut out = [0; 3];
        vslice_a.elementwise_min_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, [1, 2, 3]);
    }

    #[test]
    fn test_vector_slice_mut_elementwise_min_into_equal() {
        let mut a = [2, 2, 2];
        let mut b = [2, 2, 2];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let mut out = [0; 3];
        vslice_a.elementwise_min_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, [2, 2, 2]);
    }

    #[test]
    fn test_vector_slice_mut_elementwise_min_into_mismatched_length() {
        let mut a = [1, 5, 3];
        let mut b = [4, 2];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let mut out = [0; 3];
        let result = vslice_a.elementwise_min_into(&vslice_b, &mut out);
        assert!(result.is_err());

        let mut a = [1, 5, 3];
        let mut b = [4, 2, 6];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let mut out = [0; 2];
        let result = vslice_a.elementwise_min_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_elementwise_min_into_empty() {
        let mut a: [i32; 0] = [];
        let mut b: [i32; 0] = [];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..0);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..0);
        let mut out: [i32; 0] = [];
        vslice_a.elementwise_min_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, []);
    }

    // -- elementwise_max --

    #[test]
    fn test_vector_slice_mut_elementwise_max() {
        let mut a = [1, 5, 3];
        let mut b = [4, 2, 6];
        let vslice_a: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut b, 0..3);
        let result = vslice_a.elementwise_max(&vslice_b).unwrap();
        assert_eq!(result.as_slice(), &[4, 5, 6]);
    }

    #[test]
    fn test_vector_slice_mut_elementwise_max_mismatched_length() {
        let mut a = [1, 5, 3];
        let mut b = [4, 2];
        let vslice_a: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut b, 0..2);
        let result = vslice_a.elementwise_max(&vslice_b);
        assert!(result.is_err());
    }

    // -- elementwise_max_into --

    #[test]
    fn test_vector_slice_mut_elementwise_max_into_basic() {
        let mut a = [1, 5, 3];
        let mut b = [4, 2, 6];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let mut out = [0; 3];
        vslice_a.elementwise_max_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, [4, 5, 6]);
    }

    #[test]
    fn test_vector_slice_mut_elementwise_max_into_equal() {
        let mut a = [2, 2, 2];
        let mut b = [2, 2, 2];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let mut out = [0; 3];
        vslice_a.elementwise_max_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, [2, 2, 2]);
    }

    #[test]
    fn test_vector_slice_mut_elementwise_max_into_mismatched_length() {
        let mut a = [1, 5, 3];
        let mut b = [4, 2];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let mut out = [0; 3];
        let result = vslice_a.elementwise_max_into(&vslice_b, &mut out);
        assert!(result.is_err());

        let mut a = [1, 5, 3];
        let mut b = [4, 2, 6];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let mut out = [0; 2];
        let result = vslice_a.elementwise_max_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_elementwise_max_into_empty() {
        let mut a: [i32; 0] = [];
        let mut b: [i32; 0] = [];
        let vslice_a: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..0);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..0);
        let mut out: [i32; 0] = [];
        vslice_a.elementwise_max_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, []);
    }

    // -- VectorOpsMut trait for VectorSliceMut --

    #[test]
    fn test_vector_slice_mut_mut_translate() {
        let mut a = [1, 2, 3];
        let mut b = [4, 5, 6];
        let mut vslice_a: VectorSliceMut<'_, i32, Column> =
            VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut b, 0..3);
        vslice_a.mut_translate(&vslice_b).unwrap();
        assert_eq!(vslice_a.as_slice(), &[5, 7, 9]);
        assert_eq!(a, [5, 7, 9]);
    }

    #[test]
    fn test_vector_slice_mut_mut_translate_mismatched_length() {
        let mut a = [1, 2, 3];
        let mut b = [4, 5];
        let mut vslice_a: VectorSliceMut<'_, i32, Column> =
            VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut b, 0..2);
        let result = vslice_a.mut_translate(&vslice_b);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_mut_scale() {
        let mut a = [1, 2, 3];
        let mut vslice: VectorSliceMut<'_, i32, Row> = VectorSliceMut::from_range(&mut a, 0..3);
        vslice.mut_scale(10);
        assert_eq!(vslice.as_slice(), &[10, 20, 30]);
        assert_eq!(a, [10, 20, 30]);
    }

    #[test]
    fn test_vector_slice_mut_mut_negate() {
        let mut a = [1, -2, 3];
        let mut vslice: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        vslice.mut_negate();
        assert_eq!(vslice.as_slice(), &[-1, 2, -3]);
        assert_eq!(a, [-1, 2, -3]);
    }

    #[test]
    fn test_vector_slice_mut_mut_zero() {
        let mut a = [1, -2, 3];
        let mut vslice: VectorSliceMut<'_, i32, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        vslice.mut_zero();
        assert_eq!(vslice.as_slice(), [0, 0, 0]);
        assert_eq!(a, [0, 0, 0]);
    }

    // -- VectorOpsFloat trait for VectorSliceMut --

    #[test]
    fn test_vector_slice_mut_normalize() {
        let mut a = [3.0, 4.0];
        let vslice: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        let result = vslice.normalize().unwrap();
        let expected = [0.6, 0.8];
        for (x, y) in result.as_slice().iter().zip(expected.iter()) {
            assert!((x - y).abs() < 1e-8);
        }
    }

    #[test]
    fn test_vector_slice_mut_normalize_zero_vector() {
        let mut a = [0.0, 0.0];
        let vslice: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        let result = vslice.normalize();
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_normalize_into() {
        let mut a = [3.0, 4.0];
        let vslice: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        let mut out: [f64; 2] = [0.0; 2];
        let result = vslice.normalize_into(&mut out);
        assert!(result.is_ok());
        for (x, y) in out.iter().zip([0.6, 0.8].iter()) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn test_vector_slice_mut_normalize_to() {
        let mut a = [3.0, 4.0];
        let vslice: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        let result = vslice.normalize_to(10.0).unwrap();
        let expected = [6.0, 8.0];
        for (x, y) in result.as_slice().iter().zip(expected.iter()) {
            assert!((x - y).abs() < 1e-8);
        }
    }

    // -- normalize_to_into --

    #[test]
    fn test_vector_slice_mut_normalize_to_into_basic() {
        let mut a = [3.0, 4.0];
        let vslice: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        let mut out = [0.0; 2];
        vslice.normalize_to_into(10.0, &mut out).unwrap();
        // The norm is 5.0, so the normalized vector with magnitude 10.0 should be [6.0, 8.0]
        assert!((out[0] - 6.0).abs() < 1e-8);
        assert!((out[1] - 8.0).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_mut_normalize_to_into_zero_vector() {
        let mut a = [0.0, 0.0];
        let vslice: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        let mut out = [0.0; 2];
        let result = vslice.normalize_to_into(10.0, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_normalize_to_into_wrong_length() {
        let mut a = [3.0, 4.0];
        let vslice: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        let mut out = [0.0; 1];
        let result = vslice.normalize_to_into(10.0, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_normalize_to_into_empty() {
        let mut a: [f64; 0] = [];
        let vslice: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..0);
        let mut out: [f64; 0] = [];
        let result = vslice.normalize_to_into(10.0, &mut out);
        assert!(result.is_err());
    }

    // -- lerp --

    #[test]
    fn test_vector_slice_mut_lerp() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [4.0, 5.0, 6.0];
        let vslice_a: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let result = vslice_a.lerp(&vslice_b, 0.5).unwrap();
        assert_eq!(result.as_slice(), &[2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_vector_slice_mut_lerp_weight_out_of_bounds() {
        let mut a = [1.0, 2.0];
        let mut b = [3.0, 4.0];
        let vslice_a: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        assert!(vslice_a.lerp(&vslice_b, -0.1).is_err());
        assert!(vslice_a.lerp(&vslice_b, 1.1).is_err());
    }

    // -- lerp_into --

    #[test]
    fn test_vector_slice_mut_lerp_into_basic() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [4.0, 5.0, 6.0];
        let vslice_a: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let mut out = [0.0; 3];
        vslice_a.lerp_into(&vslice_b, 0.5, &mut out).unwrap();
        assert_eq!(out, [2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_vector_slice_mut_lerp_into_weight_zero() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [4.0, 5.0, 6.0];
        let vslice_a: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let mut out = [0.0; 3];
        vslice_a.lerp_into(&vslice_b, 0.0, &mut out).unwrap();
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vector_slice_mut_lerp_into_weight_one() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [4.0, 5.0, 6.0];
        let vslice_a: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let mut out = [0.0; 3];
        vslice_a.lerp_into(&vslice_b, 1.0, &mut out).unwrap();
        assert_eq!(out, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_vector_slice_mut_lerp_into_mismatched_length_end() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [4.0, 5.0];
        let vslice_a: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let mut out = [0.0; 3];
        let result = vslice_a.lerp_into(&vslice_b, 0.5, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_lerp_into_mismatched_length_out() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [4.0, 5.0, 6.0];
        let vslice_a: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let mut out = [0.0; 2];
        let result = vslice_a.lerp_into(&vslice_b, 0.5, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_lerp_into_weight_out_of_bounds_low() {
        let mut a = [1.0, 2.0];
        let mut b = [3.0, 4.0];
        let vslice_a: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let mut out = [0.0; 2];
        let result = vslice_a.lerp_into(&vslice_b, -0.1, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_lerp_into_weight_out_of_bounds_high() {
        let mut a = [1.0, 2.0];
        let mut b = [3.0, 4.0];
        let vslice_a: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let mut out = [0.0; 2];
        let result = vslice_a.lerp_into(&vslice_b, 1.1, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_lerp_into_empty() {
        let mut a: [f64; 0] = [];
        let mut b: [f64; 0] = [];
        let vslice_a: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..0);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..0);
        let mut out: [f64; 0] = [];
        vslice_a.lerp_into(&vslice_b, 0.5, &mut out).unwrap();
        assert_eq!(out, []);
    }

    // -- midpoint --

    #[test]
    fn test_vector_slice_mut_midpoint() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [4.0, 5.0, 6.0];
        let vslice_a: VectorSliceMut<'_, f64, Row> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let result = vslice_a.midpoint(&vslice_b).unwrap();
        assert_eq!(result.as_slice(), &[2.5, 3.5, 4.5]);
    }

    // -- midpoint_into --

    #[test]
    fn test_vector_slice_mut_midpoint_into_basic() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [4.0, 5.0, 6.0];
        let vslice_a: VectorSliceMut<'_, f64, Row> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let mut out = [0.0; 3];
        vslice_a.midpoint_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, [2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_vector_slice_mut_midpoint_into_weight_matches_midpoint() {
        let mut a = [10.0, 20.0];
        let mut b = [30.0, 40.0];
        let vslice_a: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let mut out = [0.0; 2];
        vslice_a.midpoint_into(&vslice_b, &mut out).unwrap();
        assert!((out[0] - 20.0).abs() < 1e-8);
        assert!((out[1] - 30.0).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_mut_midpoint_into_mismatched_length_end() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [4.0, 5.0];
        let vslice_a: VectorSliceMut<'_, f64, Row> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let mut out = [0.0; 3];
        let result = vslice_a.midpoint_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_midpoint_into_mismatched_length_out() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [4.0, 5.0, 6.0];
        let vslice_a: VectorSliceMut<'_, f64, Row> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let mut out = [0.0; 2];
        let result = vslice_a.midpoint_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_midpoint_into_empty() {
        let mut a: [f64; 0] = [];
        let mut b: [f64; 0] = [];
        let vslice_a: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..0);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..0);
        let mut out: [f64; 0] = [];
        vslice_a.midpoint_into(&vslice_b, &mut out).unwrap();
        assert_eq!(out, []);
    }

    // -- distance --

    #[test]
    fn test_vector_slice_mut_distance() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [4.0, 6.0, 3.0];
        let vslice_a: VectorSliceMut<'_, f64, Row> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let result = vslice_a.distance(&vslice_b).unwrap();
        assert!((result - 5.0).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_mut_manhattan_distance() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [4.0, 6.0, 3.0];
        let vslice_a: VectorSliceMut<'_, f64, Row> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let result = vslice_a.manhattan_distance(&vslice_b).unwrap();
        assert!((result - 7.0).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_mut_chebyshev_distance() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [4.0, 6.0, 3.0];
        let vslice_a: VectorSliceMut<'_, f64, Row> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let result = vslice_a.chebyshev_distance(&vslice_b).unwrap();
        assert!((result - 4.0).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_mut_minkowski_distance() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [4.0, 6.0, 3.0];
        let vslice_a: VectorSliceMut<'_, f64, Row> = VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        let result = vslice_a.minkowski_distance(&vslice_b, 3.0).unwrap();
        assert!((result - 4.497941445275415).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_mut_angle_with() {
        let mut a = [1.0, 0.0];
        let mut b = [0.0, 1.0];
        let vslice_a: VectorSliceMut<'_, f64, Row> = VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let result = vslice_a.angle_with(&vslice_b).unwrap();
        assert!((result - std::f64::consts::FRAC_PI_2).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_mut_project_onto() {
        let mut a = [3.0, 4.0];
        let mut b = [6.0, 8.0];
        let vslice_a: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let result = vslice_a.project_onto(&vslice_b).unwrap();
        let expected = [3.0, 4.0];
        for (x, y) in result.as_slice().iter().zip(expected.iter()) {
            assert!((x - y).abs() < 1e-8);
        }
    }

    #[test]
    fn test_vector_slice_mut_cosine_similarity() {
        let mut a = [1.0, 0.0];
        let mut b = [0.0, 1.0];
        let vslice_a: VectorSliceMut<'_, f64, Row> = VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let result = vslice_a.cosine_similarity(&vslice_b).unwrap();
        assert!((result - 0.0).abs() < 1e-8);
    }

    // -- project_onto_into --

    #[test]
    fn test_vector_slice_mut_project_onto_into_basic() {
        let mut a = [3.0, 4.0];
        let mut b = [6.0, 8.0];
        let vslice_a: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let mut out = [0.0; 2];
        vslice_a.project_onto_into(&vslice_b, &mut out).unwrap();
        assert!((out[0] - 3.0).abs() < 1e-8);
        assert!((out[1] - 4.0).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_mut_project_onto_into_parallel() {
        let mut a = [2.0, 4.0];
        let mut b = [1.0, 2.0];
        let vslice_a: VectorSliceMut<'_, f64, Row> = VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let mut out = [0.0; 2];
        vslice_a.project_onto_into(&vslice_b, &mut out).unwrap();
        assert!((out[0] - 2.0).abs() < 1e-8);
        assert!((out[1] - 4.0).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_mut_project_onto_into_orthogonal() {
        let mut a = [1.0, 0.0];
        let mut b = [0.0, 1.0];
        let vslice_a: VectorSliceMut<'_, f64, Row> = VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let mut out = [99.0, 99.0];
        vslice_a.project_onto_into(&vslice_b, &mut out).unwrap();
        assert!((out[0]).abs() < 1e-8);
        assert!((out[1]).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_mut_project_onto_into_identical() {
        let mut a = [5.0, 5.0];
        let mut b = [5.0, 5.0];
        let vslice_a: VectorSliceMut<'_, f64, Row> = VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let mut out = [0.0, 0.0];
        vslice_a.project_onto_into(&vslice_b, &mut out).unwrap();
        assert!((out[0] - 5.0).abs() < 1e-8);
        assert!((out[1] - 5.0).abs() < 1e-8);
    }

    #[test]
    fn test_vector_slice_mut_project_onto_into_zero_vector() {
        let mut a = [1.0, 2.0];
        let mut b = [0.0, 0.0];
        let vslice_a: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let mut out = [0.0, 0.0];
        let result = vslice_a.project_onto_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_project_onto_into_mismatched_length_other() {
        let mut a = [1.0, 2.0];
        let mut b = [3.0];
        let vslice_a: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..1);
        let mut out = [0.0, 0.0];
        let result = vslice_a.project_onto_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_project_onto_into_mismatched_length_out() {
        let mut a = [1.0, 2.0];
        let mut b = [3.0, 4.0];
        let vslice_a: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let mut out = [0.0; 1];
        let result = vslice_a.project_onto_into(&vslice_b, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_project_onto_into_empty() {
        let mut a: [f64; 0] = [];
        let mut b: [f64; 0] = [];
        let vslice_a: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..0);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..0);
        let mut out: [f64; 0] = [];
        let result = vslice_a.project_onto_into(&vslice_b, &mut out);
        assert!(result.is_err()); // zero vector error
    }

    // -- VectorOpsFloatMut trait for VectorSliceMut --

    #[test]
    fn test_vector_slice_mut_mut_normalize() {
        let mut a = [3.0, 4.0];
        let mut vslice: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        vslice.mut_normalize().unwrap();
        let expected = [0.6, 0.8];
        for (x, y) in vslice.as_slice().iter().zip(expected.iter()) {
            assert!((x - y).abs() < 1e-8);
        }
    }

    #[test]
    fn test_vector_slice_mut_mut_normalize_zero_vector() {
        let mut a = [0.0, 0.0];
        let mut vslice: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        let result = vslice.mut_normalize();
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_mut_normalize_to() {
        let mut a = [3.0, 4.0];
        let mut vslice: VectorSliceMut<'_, f64, Column> = VectorSliceMut::from_range(&mut a, 0..2);
        vslice.mut_normalize_to(10.0).unwrap();
        let expected = [6.0, 8.0];
        for (x, y) in vslice.as_slice().iter().zip(expected.iter()) {
            assert!((x - y).abs() < 1e-8);
        }
    }

    #[test]
    fn test_vector_slice_mut_mut_lerp() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [4.0, 5.0, 6.0];
        let mut vslice_a: VectorSliceMut<'_, f64, Column> =
            VectorSliceMut::from_range(&mut a, 0..3);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..3);
        vslice_a.mut_lerp(&vslice_b, 0.5).unwrap();
        assert_eq!(vslice_a.as_slice(), &[2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_vector_slice_mut_mut_lerp_weight_out_of_bounds() {
        let mut a = [1.0, 2.0];
        let mut b = [3.0, 4.0];
        let mut vslice_a: VectorSliceMut<'_, f64, Column> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        assert!(vslice_a.mut_lerp(&vslice_b, -0.1).is_err());
        assert!(vslice_a.mut_lerp(&vslice_b, 1.1).is_err());
    }

    // -- VectorOpsComplex trait for VectorSliceMut --

    // -- normalize --

    #[test]
    fn test_vector_slice_mut_complex_normalize() {
        use num::Complex;
        let mut a = [Complex::new(3.0, 4.0), Complex::new(0.0, 0.0)];
        let vslice: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..2);
        // The norm is sqrt(|3+4i|^2 + |0|^2) = sqrt(25) = 5
        let result = vslice.normalize().unwrap();
        let expected = [Complex::new(3.0 / 5.0, 4.0 / 5.0), Complex::new(0.0, 0.0)];
        for (x, y) in result.as_slice().iter().zip(expected.iter()) {
            assert!((x.re - y.re).abs() < 1e-8);
            assert!((x.im - y.im).abs() < 1e-8);
        }
    }

    #[test]
    fn test_vector_slice_mut_complex_normalize_zero_vector() {
        use num::Complex;
        let mut a = [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let vslice: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let result = vslice.normalize();
        assert!(result.is_err());
    }

    // -- normalize_to --

    #[test]
    fn test_vector_slice_mut_complex_normalize_to() {
        use num::Complex;
        let mut a = [Complex::new(3.0, 4.0), Complex::new(0.0, 0.0)];
        let vslice: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..2);
        // The norm is 5, so scaling to magnitude 10 multiplies by 2
        let result = vslice.normalize_to(10.0).unwrap();
        let expected = [Complex::new(6.0, 8.0), Complex::new(0.0, 0.0)];
        for (x, y) in result.as_slice().iter().zip(expected.iter()) {
            assert!((x.re - y.re).abs() < 1e-8);
            assert!((x.im - y.im).abs() < 1e-8);
        }
    }

    #[test]
    fn test_vector_slice_mut_complex_normalize_to_zero_vector() {
        use num::Complex;
        let mut a = [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let vslice: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let result = vslice.normalize_to(10.0);
        assert!(result.is_err());
    }

    // -- dot --

    #[test]
    fn test_vector_slice_mut_complex_dot_basic() {
        use num::Complex;
        let mut a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let mut b = [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        // Compute expected before mutable borrow
        let expected = a[0].conj() * b[0] + a[1].conj() * b[1];
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let result = VectorOpsComplex::dot(&vslice_a, &vslice_b).unwrap();
        assert!((result.re - expected.re).abs() < 1e-12);
        assert!((result.im - expected.im).abs() < 1e-12);
    }

    #[test]
    fn test_vector_slice_mut_complex_dot_mismatched_length() {
        use num::Complex;
        let mut a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let mut b = [Complex::new(5.0, 6.0)];
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..1);
        let result = VectorOpsComplex::dot(&vslice_a, &vslice_b);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_slice_mut_complex_dot_zero() {
        use num::Complex;
        let mut a = [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let mut b = [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let result = VectorOpsComplex::dot(&vslice_a, &vslice_b).unwrap();
        assert!((result.re).abs() < 1e-12);
        assert!((result.im).abs() < 1e-12);
    }

    #[test]
    fn test_vector_slice_mut_complex_dot_empty() {
        use num::Complex;
        let mut a: [Complex<f64>; 0] = [];
        let mut b: [Complex<f64>; 0] = [];
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..0);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..0);
        let result = VectorOpsComplex::dot(&vslice_a, &vslice_b).unwrap();
        assert!((result.re).abs() < 1e-12);
        assert!((result.im).abs() < 1e-12);
    }

    // -- lerp --

    #[test]
    fn test_vector_slice_mut_complex_lerp() {
        use num::Complex;
        let mut a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let mut b = [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        // Lerp with weight 0.25
        let result = vslice_a.lerp(&vslice_b, 0.25).unwrap();
        let expected = [
            Complex::new(1.0 + 0.25 * (5.0 - 1.0), 2.0 + 0.25 * (6.0 - 2.0)),
            Complex::new(3.0 + 0.25 * (7.0 - 3.0), 4.0 + 0.25 * (8.0 - 4.0)),
        ];
        for (x, y) in result.as_slice().iter().zip(expected.iter()) {
            assert!((x.re - y.re).abs() < 1e-8);
            assert!((x.im - y.im).abs() < 1e-8);
        }
    }

    #[test]
    fn test_vector_slice_mut_complex_lerp_weight_out_of_bounds() {
        use num::Complex;
        let mut a = [Complex::new(1.0, 2.0)];
        let mut b = [Complex::new(3.0, 4.0)];
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..1);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..1);
        assert!(vslice_a.lerp(&vslice_b, -0.1).is_err());
        assert!(vslice_a.lerp(&vslice_b, 1.1).is_err());
    }

    // -- midpoint --

    #[test]
    fn test_vector_slice_mut_complex_midpoint() {
        use num::Complex;
        let mut a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let mut b = [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Row> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let result = vslice_a.midpoint(&vslice_b).unwrap();
        let expected = [
            Complex::new((1.0 + 5.0) / 2.0, (2.0 + 6.0) / 2.0),
            Complex::new((3.0 + 7.0) / 2.0, (4.0 + 8.0) / 2.0),
        ];
        for (x, y) in result.as_slice().iter().zip(expected.iter()) {
            assert!((x.re - y.re).abs() < 1e-8);
            assert!((x.im - y.im).abs() < 1e-8);
        }
    }

    // -- distance --

    #[test]
    fn test_vector_slice_mut_complex_distance() {
        use num::Complex;
        let mut a = [Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        let mut b = [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        // Euclidean distance: sqrt(sum_i |a[i] - b[i]|^2)
        let d0 = (a[0] - b[0]).norm_sqr();
        let d1 = (a[1] - b[1]).norm_sqr();
        let expected = (d0 + d1).sqrt();
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let dist = vslice_a.distance(&vslice_b).unwrap();
        assert!((dist - expected).abs() < 1e-12);
    }

    // -- manhattan_distance --

    #[test]
    fn test_vector_slice_mut_complex_manhattan_distance() {
        use num::Complex;
        let mut a = [Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        let mut b = [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        // Manhattan distance: sum_i |a[i] - b[i]|
        let d0 = (a[0] - b[0]).norm();
        let d1 = (a[1] - b[1]).norm();
        let expected = d0 + d1;
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Row> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let dist = vslice_a.manhattan_distance(&vslice_b).unwrap();
        assert!((dist - expected).abs() < 1e-12);
    }

    // -- chebyshev_distance --

    #[test]
    fn test_vector_slice_mut_complex_chebyshev_distance() {
        use num::Complex;
        let mut a = [Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        let mut b = [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        // Chebyshev distance: max_i |a[i] - b[i]|
        let d0 = (a[0] - b[0]).norm();
        let d1 = (a[1] - b[1]).norm();
        let expected = d0.max(d1);
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let dist = vslice_a.chebyshev_distance(&vslice_b).unwrap();
        assert!((dist - expected).abs() < 1e-12);
    }

    // -- minkowski_distance --

    #[test]
    fn test_vector_slice_mut_complex_minkowski_distance() {
        use num::Complex;
        let mut a = [Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        let mut b = [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        // Minkowski distance: (|a[0]-b[0]|^p + |a[1]-b[1]|^p)^(1/p)
        let p = 3.0;
        let d0 = (a[0] - b[0]).norm().powf(p);
        let d1 = (a[1] - b[1]).norm().powf(p);
        let expected = (d0 + d1).powf(1.0 / p);
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Row> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let dist = vslice_a.minkowski_distance(&vslice_b, p).unwrap();
        assert!((dist - expected).abs() < 1e-12);
    }

    // -- project_onto --

    #[test]
    fn test_vector_slice_mut_complex_project_onto_basic() {
        use num::Complex;
        let mut a = [Complex::new(3.0, 4.0), Complex::new(0.0, 0.0)];
        let mut b = [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        // Project a onto b: should be [3.0 - 4.0i, 0.0]
        let proj = vslice_a.project_onto(&vslice_b).unwrap();
        assert!((proj.as_slice()[0] - Complex::new(3.0, -4.0)).norm() < 1e-12);
        assert!((proj.as_slice()[1] - Complex::new(0.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_vector_slice_mut_complex_project_onto_parallel() {
        use num::Complex;
        let mut a = [Complex::new(2.0, 2.0), Complex::new(4.0, 4.0)];
        let mut b = [Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)];
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Row> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let proj = vslice_a.project_onto(&vslice_b).unwrap();
        assert!((proj.as_slice()[0] - Complex::new(2.0, 2.0)).norm() < 1e-12);
        assert!((proj.as_slice()[1] - Complex::new(4.0, 4.0)).norm() < 1e-12);
    }

    #[test]
    fn test_vector_slice_mut_complex_project_onto_orthogonal() {
        use num::Complex;
        let mut a = [Complex::new(0.0, 1.0), Complex::new(0.0, 0.0)];
        let mut b = [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let proj = vslice_a.project_onto(&vslice_b).unwrap();
        assert!((proj.as_slice()[0] - Complex::new(0.0, -1.0)).norm() < 1e-12);
        assert!((proj.as_slice()[1] - Complex::new(0.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_vector_slice_mut_complex_project_onto_identical() {
        use num::Complex;
        let mut a = [Complex::new(5.0, 5.0), Complex::new(5.0, 5.0)];
        let mut b = [Complex::new(5.0, 5.0), Complex::new(5.0, 5.0)];
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Row> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let proj = vslice_a.project_onto(&vslice_b).unwrap();
        assert!((proj.as_slice()[0] - Complex::new(5.0, 5.0)).norm() < 1e-12);
        assert!((proj.as_slice()[1] - Complex::new(5.0, 5.0)).norm() < 1e-12);
    }

    #[test]
    fn test_vector_slice_mut_complex_project_onto_zero_vector() {
        use num::Complex;
        let mut a = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let mut b = [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let result = vslice_a.project_onto(&vslice_b);
        assert!(result.is_err());
    }

    // -- cosine_similarity --

    #[test]
    fn test_vector_slice_mut_complex_cosine_similarity_parallel() {
        use num::Complex;
        let mut a = [Complex::new(1.0, 2.0), Complex::new(2.0, 4.0)];
        let mut b = [Complex::new(2.0, 4.0), Complex::new(4.0, 8.0)];
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let cos_sim = vslice_a.cosine_similarity(&vslice_b).unwrap();
        assert!((cos_sim - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_vector_slice_mut_complex_cosine_similarity_orthogonal() {
        use num::Complex;
        let mut a = [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        let mut b = [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..2);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..2);
        let cos_sim = vslice_a.cosine_similarity(&vslice_b).unwrap();
        assert!((cos_sim - Complex::new(0.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_vector_slice_mut_complex_cosine_similarity_opposite() {
        use num::Complex;
        let mut a = [Complex::new(1.0, 0.0)];
        let mut b = [Complex::new(-1.0, 0.0)];
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..1);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..1);
        let cos_sim = vslice_a.cosine_similarity(&vslice_b).unwrap();
        assert!((cos_sim + Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_vector_slice_mut_complex_cosine_similarity_identical() {
        use num::Complex;
        let mut a = [Complex::new(3.0, 4.0)];
        let mut b = [Complex::new(3.0, 4.0)];
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..1);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..1);
        let cos_sim = vslice_a.cosine_similarity(&vslice_b).unwrap();
        assert!((cos_sim - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_vector_slice_mut_complex_cosine_similarity_arbitrary() {
        use num::Complex;
        let mut a = [Complex::new(1.0, 2.0)];
        let mut b = [Complex::new(2.0, 1.0)];
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..1);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..1);
        let cos_sim = vslice_a.cosine_similarity(&vslice_b).unwrap();
        assert!(cos_sim.norm() <= 1.0 + 1e-12);
    }

    #[test]
    fn test_vector_slice_mut_complex_cosine_similarity_zero_vector() {
        use num::Complex;
        let mut a = [Complex::new(0.0, 0.0)];
        let mut b = [Complex::new(1.0, 2.0)];
        let vslice_a: VectorSliceMut<'_, Complex<f64>, Column> =
            VectorSliceMut::from_range(&mut a, 0..1);
        let vslice_b = VectorSliceMut::from_range(&mut b, 0..1);
        let result = vslice_a.cosine_similarity(&vslice_b);
        assert!(result.is_err());
    }
}
