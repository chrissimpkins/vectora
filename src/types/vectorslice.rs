//! VectorSlice types.

use crate::errors::VectorError;
use crate::types::flexvector::FlexVector;
use crate::types::orientation::Column;
use crate::types::traits::{
    VectorBase, VectorBaseMut, VectorOps, VectorOpsMut, VectorOrientationName,
};
use crate::types::utils::{
    cross_impl, cross_into_impl, dot_impl, dot_to_f64_impl, elementwise_max_impl,
    elementwise_max_into_impl, elementwise_min_impl, elementwise_min_into_impl, mut_translate_impl,
    translate_impl,
};

use std::fmt;
use std::marker::PhantomData;

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
    T: Clone,
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
        T: PartialOrd + Clone,
    {
        self.check_same_length_and_raise(other)?;
        Ok(FlexVector::from_vec(elementwise_min_impl(self.as_slice(), other.as_slice())))
    }

    #[inline]
    fn elementwise_min_into(&self, other: &Self, out: &mut [T]) -> Result<(), VectorError>
    where
        T: PartialOrd + Clone,
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
        T: PartialOrd + Clone,
    {
        self.check_same_length_and_raise(other)?;
        Ok(FlexVector::from_vec(elementwise_max_impl(self.as_slice(), other.as_slice())))
    }

    #[inline]
    fn elementwise_max_into(&self, other: &Self, out: &mut [T]) -> Result<(), VectorError>
    where
        T: PartialOrd + Clone,
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
    T: Clone,
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
        T: PartialOrd + Clone,
    {
        self.check_same_length_and_raise(other)?;
        Ok(FlexVector::from_vec(elementwise_min_impl(self.as_slice(), other.as_slice())))
    }

    #[inline]
    fn elementwise_min_into(&self, other: &Self, out: &mut [T]) -> Result<(), VectorError>
    where
        T: PartialOrd + Clone,
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
        T: PartialOrd + Clone,
    {
        self.check_same_length_and_raise(other)?;
        Ok(FlexVector::from_vec(elementwise_max_impl(self.as_slice(), other.as_slice())))
    }

    #[inline]
    fn elementwise_max_into(&self, other: &Self, out: &mut [T]) -> Result<(), VectorError>
    where
        T: PartialOrd + Clone,
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
    T: Clone,
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
}
