//! VectorSlice types.

use crate::types::orientation::Column;
use std::marker::PhantomData;

// /////////////////////////////////
// ================================
//
// VectorSlice type
//
// ================================
// /////////////////////////////////

/// ...
#[derive(Clone, Debug, PartialEq, Eq)]
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

// /////////////////////////////////
// ================================
//
// VectorSliceMut type
//
// ================================
// /////////////////////////////////

/// ...
#[derive(Debug)]
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
        self.elements.into_iter()
    }
}

impl<'a, T, O> IntoIterator for &'a mut VectorSliceMut<'a, T, O> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{FlexVector, VectorBase};
    use crate::types::orientation::{Column, Row};
    use num::Complex;

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

    // -- Deref trait --
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

    // -- AsRef trait --

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
}
