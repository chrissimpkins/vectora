//! Vector types.

// use std::cmp::PartialOrd;
// use std::convert::From;
use std::{
    borrow::{Borrow, BorrowMut},
    iter::IntoIterator,
    ops::{Index, IndexMut},
    slice::SliceIndex,
};

use approx::{relative_eq, RelativeEq};
use num::Num;

// ================================
//
// Type aliases for 2D and 3D types
//
// ================================
// Two-dimensional
/// Type alias for a generic 2D vector
pub type Vector2d<T> = Vector<T, 2>;
// signed integers
/// Type alias for a 2D i8 integer vector.
pub type Vector2dI8 = Vector<i8, 2>;
/// Type alias for a 2D i16 integer vector.
pub type Vector2dI16 = Vector<i16, 2>;
/// Type alias for a 2D i32 integer vector.
pub type Vector2dI32 = Vector<i32, 2>;
/// Type alias for a 2D i64 integer vector.
pub type Vector2dI64 = Vector<i64, 2>;
/// Type alias for a 2D i128 integer vector.
pub type Vector2dI128 = Vector<i128, 2>;
// unsigned integers
/// Type alias for a 2D u8 integer vector.
pub type Vector2dU8 = Vector<u8, 2>;
/// Type alias for a 2D u16 integer vector.
pub type Vector2dU16 = Vector<u16, 2>;
/// Type alias for a 2D u32 integer vector.
pub type Vector2dU32 = Vector<u32, 2>;
/// Type alias for a 2D u64 integer vector.
pub type Vector2dU64 = Vector<u64, 2>;
/// Type alias for a 2D u128 integer vector.
pub type Vector2dU128 = Vector<u128, 2>;
// floats
/// Type alias for a 2D f32 float vector.
pub type Vector2dF32 = Vector<f32, 2>;
/// Type alias for a 2D f64 float vector.
pub type Vector2dF64 = Vector<f64, 2>;
// architecture-specific
/// Type alias for a 2D isize integer vector.
pub type Vector2dIsize = Vector<isize, 2>;
/// Type alias for a 2D usize integer vector.
pub type Vector2dUsize = Vector<usize, 2>;

// Three dimensional
/// Type alias for a generic 3D vector.
pub type Vector3d<T> = Vector<T, 3>;
// signed integers
/// Type alias for a 3D i8 integer vector.
pub type Vector3dI8 = Vector<i8, 3>;
/// Type alias for a 3D i16 integer vector.
pub type Vector3dI16 = Vector<i16, 3>;
/// Type alias for a 3D i32 integer vector.
pub type Vector3dI32 = Vector<i32, 3>;
/// Type alias for a 3D i64 integer vector.
pub type Vector3dI64 = Vector<i64, 3>;
/// Type alias for a 3D i128 integer vector.
pub type Vector3dI128 = Vector<i128, 3>;
// unsigned integers
/// Type alias for a 3D u8 integer vector.
pub type Vector3dU8 = Vector<u8, 3>;
/// Type alias for a 3D u16 integer vector.
pub type Vector3dU16 = Vector<u16, 3>;
/// Type alias for a 3D u16 integer vector.
pub type Vector3dU32 = Vector<u32, 3>;
/// Type alias for a 3D u32 integer vector.
pub type Vector3dU64 = Vector<u64, 3>;
/// Type alias for a 3D u64 integer vector.
pub type Vector3dU128 = Vector<u128, 3>;
// floats
/// Type alias for a 3D f32 float vector.
pub type Vector3dF32 = Vector<f32, 3>;
/// Type alias for a 3D f64 float vector.
pub type Vector3dF64 = Vector<f64, 3>;
// architecture-specific
/// Type alias for a 3D isize integer vector.
pub type Vector3dIsize = Vector<isize, 3>;
/// Type alias for a 3D usize integer vector.
pub type Vector3dUsize = Vector<usize, 3>;

/// A generic vector type that holds N-dimensional `coord` data.
#[derive(Copy, Clone, Debug)]
pub struct Vector<T, const N: usize>
where
    T: Num + Copy,
{
    /// N-dimensional vector coordinate values.
    pub coord: [T; N],
}

// ================================
//
// Instantiation method impl
//
// ================================
impl<T, const N: usize> Vector<T, N>
where
    T: Num + Copy + Default,
{
    /// Returns a new [`Vector`] filled with default values.
    ///
    /// # Examples
    ///
    /// ## Turbofish syntax
    ///
    /// Use turbofish syntax to define the numeric type of the vector
    /// coordinate values and number of vector dimensions:
    ///
    /// ```
    /// # use vectora::types::vector::*;
    /// let vec_2d_u32 = Vector::<u32, 2>::new();
    /// let vec_3d_f64 = Vector::<f64, 3>::new();
    /// ```
    ///
    /// ## Type alias syntax
    ///
    /// Simplify instantiation with one of the defined type aliases:
    ///
    /// ```
    /// # use vectora::types::vector::*;
    /// let vec_2d_u32_1 = Vector2d::<u32>::new();
    /// let vec_2d_u32_2 = Vector2dU32::new();
    ///
    /// let vec_3d_f64_1 = Vector3d::<f64>::new();
    /// let vec_3d_f64_2 = Vector3dF64::new();
    /// ```
    pub fn new() -> Self {
        Self { coord: [T::default(); N] }
    }
}

impl<T, const N: usize> Default for Vector<T, N>
where
    T: Num + Copy + Default,
{
    /// Returns a new [`Vector`] filled with default values.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v1 = Vector::<u32, 2>::default();
    /// let v2 = Vector::<f64, 3>::default();
    /// ```
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Vector<T, N>
where
    T: Num + Copy,
{
    /// Returns a new [`Vector`] defined with values in `array`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v1 = Vector::<u32, 2>::from_array([1, 2]);
    /// let v2 = Vector::<f64, 3>::from_array([3.0, 4.0, 5.0]);
    /// ```
    pub fn from_array(array: [T; N]) -> Self {
        Self { coord: array }
    }

    /// Returns a reference to a [`Vector`] index value or range,
    /// or `None` if the index is out of bounds.
    ///
    /// This method provides safe, bounds checked immutable access to coordinate
    /// values.
    ///
    /// # Examples
    ///
    /// ## With Index Values
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<u32, 3>::from_array([1, 2, 3]);
    /// assert_eq!(v.get(0), Some(&1));
    /// assert_eq!(v.get(1), Some(&2));
    /// assert_eq!(v.get(3), None);
    /// ```
    ///
    /// ## With Index Ranges
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<u32, 5>::from_array([1, 2, 3, 4, 5]);
    /// assert_eq!(v.get(0..3).unwrap(), [1, 2, 3]);
    /// assert_eq!(v.get(3..).unwrap(), [4, 5]);
    /// assert_eq!(v.get(..2).unwrap(), [1, 2]);
    /// assert_eq!(v.get(..).unwrap(), [1, 2, 3, 4, 5]);
    /// assert_eq!(v.get(2..10), None);
    /// ```
    #[inline]
    pub fn get<I>(&self, index: I) -> Option<&I::Output>
    where
        I: SliceIndex<[T]>,
    {
        self.coord.get(index)
    }

    /// Returns a mutable reference to a [`Vector`] index value or range,
    /// or `None` if the index is out of bounds.
    ///
    /// This method provides safe, bounds checked mutable access to coordinate
    /// values.
    ///
    /// # Examples
    ///
    /// ## With Index Values
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<u32, 3>::from_array([1, 2, 3]);
    ///
    /// let x = v.get_mut(0).unwrap();
    /// assert_eq!(x, &mut 1);
    /// *x = 10;
    /// assert_eq!(v[0], 10);
    /// assert_eq!(v[1], 2);
    /// assert_eq!(v[2], 3);
    /// ```
    ///
    /// ## With Index Ranges
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<u32, 3>::from_array([1, 2, 3]);
    ///
    /// let x = v.get_mut(0..2).unwrap();
    /// assert_eq!(x, &mut [1, 2]);
    /// x[0] = 5;
    /// x[1] = 6;
    /// assert_eq!(v[0], 5);
    /// assert_eq!(v[1], 6);
    /// assert_eq!(v[2], 3);
    /// ```
    #[inline]
    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut I::Output>
    where
        I: SliceIndex<[T]>,
    {
        self.coord.get_mut(index)
    }

    /// Returns an iterator over immutable [`Vector`] coordinate references.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = &Vector::<u32, 3>::from_array([1, 2, 3]);
    /// let mut v_iter = v.iter();
    /// assert_eq!(v_iter.next(), Some(&1));
    /// assert_eq!(v_iter.next(), Some(&2));
    /// assert_eq!(v_iter.next(), Some(&3));
    /// assert_eq!(v_iter.next(), None);
    /// ```
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.coord.iter()
    }

    /// Returns an iterator over mutable [`Vector`] coordinate references.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<u32, 3>::from_array([1, 2, 3]);
    /// for x in v.iter_mut() {
    ///     *x += 3;
    /// }
    ///
    /// assert_eq!(v[0], 4);
    /// assert_eq!(v[1], 5);
    /// assert_eq!(v[2], 6);
    /// ```
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.coord.iter_mut()
    }

    // ================================
    //
    // Private methods
    //
    // ================================
    #[inline]
    fn partial_eq_int(&self, other: &Self) -> bool {
        self.coord == other.coord
    }

    #[inline]
    fn partial_eq_float(&self, other: &Self) -> bool
    where
        T: Num + Copy + RelativeEq<T>,
    {
        for (i, item) in self.coord.iter().enumerate() {
            if !relative_eq!(*item, other[i]) {
                return false;
            }
        }

        true
    }
}

// ================================
//
// Index trait impl
//
// ================================
impl<T, const N: usize> Index<usize> for Vector<T, N>
where
    T: Num + Copy,
{
    type Output = T;
    /// Returns [`Vector`] values by zero-based index.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<u32, 3>::from_array([1, 2, 3]);
    /// assert_eq!(v[0], 1);
    /// assert_eq!(v[1], 2);
    /// assert_eq!(v[2], 3);
    /// ```
    ///
    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.coord[i]
    }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N>
where
    T: Num + Copy,
{
    /// Returns mutable [`Vector`] values by zero-based index.
    ///
    /// Supports coordinate value assignment by index.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<u32, 3>::from_array([1, 2, 3]);
    /// assert_eq!(v[0], 1);
    /// assert_eq!(v[1], 2);
    /// assert_eq!(v[2], 3);
    /// v[0] = 5;
    /// v[1] = 6;
    /// assert_eq!(v[0], 5);
    /// assert_eq!(v[1], 6);
    /// assert_eq!(v[2], 3);
    /// ```
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.coord[i]
    }
}

// ================================
//
// Iter / IntoIterator trait impl
//
// ================================

impl<T, const N: usize> IntoIterator for Vector<T, N>
where
    T: Num + Copy,
{
    type Item = T;
    type IntoIter = std::array::IntoIter<Self::Item, N>;

    /// Creates a consuming iterator that iterates over coordinates by value.
    fn into_iter(self) -> Self::IntoIter {
        self.coord.into_iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a Vector<T, N>
where
    T: Num + Copy,
{
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    /// Creates an iterator over immutable coordinate references.
    fn into_iter(self) -> Self::IntoIter {
        self.coord.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut Vector<T, N>
where
    T: Num + Copy,
{
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    /// Creates an iterator over mutable coordinate references.
    fn into_iter(self) -> Self::IntoIter {
        self.coord.iter_mut()
    }
}

// ================================
//
// PartialEq trait impl
//
// ================================

/// PartialEq trait implementation for [`Vector`] with integer coordinate types.
///
/// These comparisons establish the symmetry and transitivity relationships
/// required for the partial equivalence relation definition.
///
/// /// Note:
///
/// - Negative zero to positive zero comparisons are considered equal.
macro_rules! impl_vector_int_partialeq_from {
    ($IntTyp: ty, $doc: expr) => {
        impl<const N: usize> PartialEq<Vector<$IntTyp, N>> for Vector<$IntTyp, N> {
            #[doc = $doc]
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.partial_eq_int(other)
            }
        }
    };
    ($IntTyp: ty) => {
        impl_vector_int_partialeq_from!(
            $IntTyp,
            concat!("PartialEq trait implementation for `Vector<", stringify!($IntTyp), ",N>`")
        );
    };
}

impl_vector_int_partialeq_from!(usize);
impl_vector_int_partialeq_from!(u8);
impl_vector_int_partialeq_from!(u16);
impl_vector_int_partialeq_from!(u32);
impl_vector_int_partialeq_from!(u64);
impl_vector_int_partialeq_from!(u128);
impl_vector_int_partialeq_from!(isize);
impl_vector_int_partialeq_from!(i8);
impl_vector_int_partialeq_from!(i16);
impl_vector_int_partialeq_from!(i32);
impl_vector_int_partialeq_from!(i64);
impl_vector_int_partialeq_from!(i128);

/// PartialEq trait implementation for [`Vector`] with float coordinate types.
///
/// These comparisons establish the symmetry and transitivity relationships
/// required for the partial equivalence relation definition for floating point
/// types.  
///
/// Note:
///
/// - Negative zero to positive zero comparisons are considered equal.
/// - Positive infinity to positive infinity comparisons are considered equal.
/// - Negative infinity to negative infinity comparisons are considered equal.
/// - NaN comparisons are considered not equal.
///
/// This approach uses the approx library relative epsilon float equality testing implementation.
/// This equivalence relation implementation is based on the approach described in
/// [Comparing Floating Point Numbers, 2012 Edition](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
macro_rules! impl_vector_float_partialeq_from {
    ($FloatTyp: ty, $doc: expr) => {
        impl<const N: usize> PartialEq<Vector<$FloatTyp, N>> for Vector<$FloatTyp, N> {
            #[doc = $doc]
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.partial_eq_float(other)
            }
        }
    };
    ($FloatTyp: ty) => {
        impl_vector_float_partialeq_from!(
            $FloatTyp,
            concat!("PartialEq trait implementation for `Vector<", stringify!($FloatTyp), ",N>`")
        );
    };
}

impl_vector_float_partialeq_from!(f32);
impl_vector_float_partialeq_from!(f64);

// ================================
//
// AsRef / AsMut trait impl
//
// ================================
impl<T, const N: usize> AsRef<Vector<T, N>> for Vector<T, N>
where
    T: Num + Copy,
{
    fn as_ref(&self) -> &Vector<T, N> {
        self
    }
}

impl<T, const N: usize> AsRef<[T]> for Vector<T, N>
where
    T: Num + Copy,
{
    fn as_ref(&self) -> &[T] {
        &self.coord
    }
}

impl<T, const N: usize> AsMut<Vector<T, N>> for Vector<T, N>
where
    T: Num + Copy,
{
    fn as_mut(&mut self) -> &mut Vector<T, N> {
        self
    }
}

impl<T, const N: usize> AsMut<[T]> for Vector<T, N>
where
    T: Num + Copy,
{
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.coord
    }
}

// ================================
//
// Borrow trait impl
//
// ================================
impl<T, const N: usize> Borrow<[T]> for Vector<T, N>
where
    T: Num + Copy,
{
    fn borrow(&self) -> &[T] {
        &self.coord[..]
    }
}

impl<T, const N: usize> BorrowMut<[T]> for Vector<T, N>
where
    T: Num + Copy,
{
    fn borrow_mut(&mut self) -> &mut [T] {
        &mut self.coord[..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use approx::{assert_relative_eq, assert_relative_ne};
    #[allow(unused_imports)]
    use pretty_assertions::{assert_eq, assert_ne};

    // ================================
    //
    // Instantiation tests
    //
    // ================================

    #[test]
    fn vector_instantiation_new_i8() {
        // Two dimension
        let mut tests = vec![];
        let v1 = Vector::<i8, 2>::new();
        let v2 = Vector2d::<i8>::new();
        let v3 = Vector2dI8::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as i8);
            assert_eq!(v[1], 0 as i8);
            assert_eq!(v.coord.len(), 2);
        }

        // Three dimension
        let mut tests = vec![];
        let v1 = Vector::<i8, 3>::new();
        let v2 = Vector3d::<i8>::new();
        let v3 = Vector3dI8::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as i8);
            assert_eq!(v[1], 0 as i8);
            assert_eq!(v[2], 0 as i8);
            assert_eq!(v.coord.len(), 3);
        }
    }

    #[test]
    fn vector_instantiation_new_i16() {
        // Two dimension
        let mut tests = vec![];
        let v1 = Vector::<i16, 2>::new();
        let v2 = Vector2d::<i16>::new();
        let v3 = Vector2dI16::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as i16);
            assert_eq!(v[1], 0 as i16);
            assert_eq!(v.coord.len(), 2);
        }

        // Three dimension
        let mut tests = vec![];
        let v1 = Vector::<i16, 3>::new();
        let v2 = Vector3d::<i16>::new();
        let v3 = Vector3dI16::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as i16);
            assert_eq!(v[1], 0 as i16);
            assert_eq!(v[2], 0 as i16);
            assert_eq!(v.coord.len(), 3);
        }
    }

    #[test]
    fn vector_instantiation_new_i32() {
        // Two dimension
        let mut tests = vec![];
        let v1 = Vector::<i32, 2>::new();
        let v2 = Vector2d::<i32>::new();
        let v3 = Vector2dI32::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as i32);
            assert_eq!(v[1], 0 as i32);
            assert_eq!(v.coord.len(), 2);
        }

        // Three dimension
        let mut tests = vec![];
        let v1 = Vector::<i32, 3>::new();
        let v2 = Vector3d::<i32>::new();
        let v3 = Vector3dI32::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as i32);
            assert_eq!(v[1], 0 as i32);
            assert_eq!(v[2], 0 as i32);
            assert_eq!(v.coord.len(), 3);
        }
    }

    #[test]
    fn vector_instantiation_new_i64() {
        // Two dimension
        let mut tests = vec![];
        let v1 = Vector::<i64, 2>::new();
        let v2 = Vector2d::<i64>::new();
        let v3 = Vector2dI64::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as i64);
            assert_eq!(v[1], 0 as i64);
            assert_eq!(v.coord.len(), 2);
        }

        // Three dimension
        let mut tests = vec![];
        let v1 = Vector::<i64, 3>::new();
        let v2 = Vector3d::<i64>::new();
        let v3 = Vector3dI64::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as i64);
            assert_eq!(v[1], 0 as i64);
            assert_eq!(v[2], 0 as i64);
            assert_eq!(v.coord.len(), 3);
        }
    }

    #[test]
    fn vector_instantiation_new_i128() {
        // Two dimension
        let mut tests = vec![];
        let v1 = Vector::<i128, 2>::new();
        let v2 = Vector2d::<i128>::new();
        let v3 = Vector2dI128::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as i128);
            assert_eq!(v[1], 0 as i128);
            assert_eq!(v.coord.len(), 2);
        }

        // Three dimension
        let mut tests = vec![];
        let v1 = Vector::<i128, 3>::new();
        let v2 = Vector3d::<i128>::new();
        let v3 = Vector3dI128::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as i128);
            assert_eq!(v[1], 0 as i128);
            assert_eq!(v[2], 0 as i128);
            assert_eq!(v.coord.len(), 3);
        }
    }

    #[test]
    fn vector_instantiation_new_u8() {
        // Two dimension
        let mut tests = vec![];
        let v1 = Vector::<u8, 2>::new();
        let v2 = Vector2d::<u8>::new();
        let v3 = Vector2dU8::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as u8);
            assert_eq!(v[1], 0 as u8);
            assert_eq!(v.coord.len(), 2);
        }

        // Three dimension
        let mut tests = vec![];
        let v1 = Vector::<u8, 3>::new();
        let v2 = Vector3d::<u8>::new();
        let v3 = Vector3dU8::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as u8);
            assert_eq!(v[1], 0 as u8);
            assert_eq!(v[2], 0 as u8);
            assert_eq!(v.coord.len(), 3);
        }
    }

    #[test]
    fn vector_instantiation_new_u16() {
        // Two dimension
        let mut tests = vec![];
        let v1 = Vector::<u16, 2>::new();
        let v2 = Vector2d::<u16>::new();
        let v3 = Vector2dU16::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as u16);
            assert_eq!(v[1], 0 as u16);
            assert_eq!(v.coord.len(), 2);
        }

        // Three dimension
        let mut tests = vec![];
        let v1 = Vector::<u16, 3>::new();
        let v2 = Vector3d::<u16>::new();
        let v3 = Vector3dU16::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as u16);
            assert_eq!(v[1], 0 as u16);
            assert_eq!(v[2], 0 as u16);
            assert_eq!(v.coord.len(), 3);
        }
    }

    #[test]
    fn vector_instantiation_new_u32() {
        // Two dimension
        let mut tests = vec![];
        let v1 = Vector::<u32, 2>::new();
        let v2 = Vector2d::<u32>::new();
        let v3 = Vector2dU32::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as u32);
            assert_eq!(v[1], 0 as u32);
            assert_eq!(v.coord.len(), 2);
        }

        // Three dimension
        let mut tests = vec![];
        let v1 = Vector::<u32, 3>::new();
        let v2 = Vector3d::<u32>::new();
        let v3 = Vector3dU32::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as u32);
            assert_eq!(v[1], 0 as u32);
            assert_eq!(v[2], 0 as u32);
            assert_eq!(v.coord.len(), 3);
        }
    }

    #[test]
    fn vector_instantiation_new_u64() {
        // Two dimension
        let mut tests = vec![];
        let v1 = Vector::<u64, 2>::new();
        let v2 = Vector2d::<u64>::new();
        let v3 = Vector2dU64::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as u64);
            assert_eq!(v[1], 0 as u64);
            assert_eq!(v.coord.len(), 2);
        }

        // Three dimension
        let mut tests = vec![];
        let v1 = Vector::<u64, 3>::new();
        let v2 = Vector3d::<u64>::new();
        let v3 = Vector3dU64::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as u64);
            assert_eq!(v[1], 0 as u64);
            assert_eq!(v[2], 0 as u64);
            assert_eq!(v.coord.len(), 3);
        }
    }

    #[test]
    fn vector_instantiation_new_u128() {
        // Two dimension
        let mut tests = vec![];
        let v1 = Vector::<u128, 2>::new();
        let v2 = Vector2d::<u128>::new();
        let v3 = Vector2dU128::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as u128);
            assert_eq!(v[1], 0 as u128);
            assert_eq!(v.coord.len(), 2);
        }

        // Three dimension
        let mut tests = vec![];
        let v1 = Vector::<u128, 3>::new();
        let v2 = Vector3d::<u128>::new();
        let v3 = Vector3dU128::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as u128);
            assert_eq!(v[1], 0 as u128);
            assert_eq!(v[2], 0 as u128);
            assert_eq!(v.coord.len(), 3);
        }
    }

    #[test]
    fn vector_instantiation_new_f32() {
        // Two dimension
        let mut tests = vec![];
        let v1 = Vector::<f32, 2>::new();
        let v2 = Vector2d::<f32>::new();
        let v3 = Vector2dF32::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_relative_eq!(v[0], 0.0 as f32);
            assert_relative_eq!(v[1], 0.0 as f32);
            assert_eq!(v.coord.len(), 2);
        }

        // Three dimension
        let mut tests = vec![];
        let v1 = Vector::<f32, 3>::new();
        let v2 = Vector3d::<f32>::new();
        let v3 = Vector3dF32::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_relative_eq!(v[0], 0.0 as f32);
            assert_relative_eq!(v[1], 0.0 as f32);
            assert_relative_eq!(v[2], 0.0 as f32);
            assert_eq!(v.coord.len(), 3);
        }
    }

    #[test]
    fn vector_instantiation_new_f64() {
        // Two dimension
        let mut tests = vec![];
        let v1 = Vector::<f64, 2>::new();
        let v2 = Vector2d::<f64>::new();
        let v3 = Vector2dF64::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_relative_eq!(v[0], 0.0 as f64);
            assert_relative_eq!(v[1], 0.0 as f64);
            assert_eq!(v.coord.len(), 2);
        }

        // Three dimension
        let mut tests = vec![];
        let v1 = Vector::<f64, 3>::new();
        let v2 = Vector3d::<f64>::new();
        let v3 = Vector3dF64::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_relative_eq!(v[0], 0.0 as f64);
            assert_relative_eq!(v[1], 0.0 as f64);
            assert_relative_eq!(v[2], 0.0 as f64);
            assert_eq!(v.coord.len(), 3);
        }
    }

    #[test]
    fn vector_instantiation_new_usize() {
        // Two dimension
        let mut tests = vec![];
        let v1 = Vector::<usize, 2>::new();
        let v2 = Vector2d::<usize>::new();
        let v3 = Vector2dUsize::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as usize);
            assert_eq!(v[1], 0 as usize);
            assert_eq!(v.coord.len(), 2);
        }

        // Three dimension
        let mut tests = vec![];
        let v1 = Vector::<usize, 3>::new();
        let v2 = Vector3d::<usize>::new();
        let v3 = Vector3dUsize::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as usize);
            assert_eq!(v[1], 0 as usize);
            assert_eq!(v[2], 0 as usize);
            assert_eq!(v.coord.len(), 3);
        }
    }

    #[test]
    fn vector_instantiation_new_isize() {
        // Two dimension
        let mut tests = vec![];
        let v1 = Vector::<isize, 2>::new();
        let v2 = Vector2d::<isize>::new();
        let v3 = Vector2dIsize::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as isize);
            assert_eq!(v[1], 0 as isize);
            assert_eq!(v.coord.len(), 2);
        }

        // Three dimension
        let mut tests = vec![];
        let v1 = Vector::<isize, 3>::new();
        let v2 = Vector3d::<isize>::new();
        let v3 = Vector3dIsize::new();
        tests.push(v1);
        tests.push(v2);
        tests.push(v3);
        for v in tests {
            assert_eq!(v[0], 0 as isize);
            assert_eq!(v[1], 0 as isize);
            assert_eq!(v[2], 0 as isize);
            assert_eq!(v.coord.len(), 3);
        }
    }

    #[test]
    fn vector_instantiation_default_u32() {
        // Two dimension
        let v = Vector::<u32, 2>::default();
        assert_eq!(v[0], 0 as u32);
        assert_eq!(v[1], 0 as u32);
        assert_eq!(v.coord.len(), 2);

        // Three dimension
        let v = Vector::<u32, 3>::default();
        assert_eq!(v[0], 0 as u32);
        assert_eq!(v[1], 0 as u32);
        assert_eq!(v[2], 0 as u32);
        assert_eq!(v.coord.len(), 3);
    }

    #[test]
    fn vector_instantiation_default_f64() {
        // Two dimension
        let v = Vector::<f64, 2>::default();
        assert_relative_eq!(v[0], 0.0 as f64);
        assert_relative_eq!(v[1], 0.0 as f64);
        assert_eq!(v.coord.len(), 2);

        // Three dimension
        let v = Vector::<f64, 3>::default();
        assert_relative_eq!(v[0], 0.0 as f64);
        assert_relative_eq!(v[1], 0.0 as f64);
        assert_relative_eq!(v[2], 0.0 as f64);
        assert_eq!(v.coord.len(), 3);
    }

    #[test]
    fn vector_instantiation_from_array() {
        // Two dimension
        let v1 = Vector::<u32, 2>::from_array([1, 2]);
        let v2 = Vector::<f64, 2>::from_array([1.0, 2.0]);
        assert_eq!(v1[0], 1);
        assert_eq!(v1[1], 2);
        assert_eq!(v1.coord.len(), 2);

        assert_relative_eq!(v2[0], 1.0 as f64);
        assert_relative_eq!(v2[1], 2.0 as f64);
        assert_eq!(v2.coord.len(), 2);

        // Three dimension
        let v1 = Vector::<u32, 3>::from_array([1, 2, 3]);
        let v2 = Vector::<f64, 3>::from_array([1.0, 2.0, 3.0]);
        assert_eq!(v1[0], 1);
        assert_eq!(v1[1], 2);
        assert_eq!(v1[2], 3);
        assert_eq!(v1.coord.len(), 3);

        assert_relative_eq!(v2[0], 1.0 as f64);
        assert_relative_eq!(v2[1], 2.0 as f64);
        assert_relative_eq!(v2[2], 3.0 as f64);
        assert_eq!(v2.coord.len(), 3);
    }

    // ================================
    //
    // Indexing tests
    //
    // ================================
    #[test]
    fn vector_index_access() {
        let v1 = Vector::<u32, 2>::from_array([1, 2]);
        let v2 = Vector::<f64, 2>::from_array([1.0, 2.0]);
        assert_eq!(v1[0], 1);
        assert_eq!(v1[1], 2);
        assert_relative_eq!(v2[0], 1.0);
        assert_relative_eq!(v2[1], 2.0);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn vector_index_access_out_of_bounds() {
        let v = Vector::<u32, 10>::new();
        v[10];
    }

    #[test]
    fn vector_index_assignment() {
        let mut v1 = Vector::<u32, 2>::from_array([1, 2]);
        let mut v2 = Vector::<f64, 2>::from_array([1.0, 2.0]);
        v1[0] = 5;
        v1[1] = 6;
        v2[0] = 5.0;
        v2[1] = 6.0;
        assert_eq!(v1[0], 5);
        assert_eq!(v1[1], 6);
        assert_relative_eq!(v2[0], 5.0);
        assert_relative_eq!(v2[1], 6.0);
    }

    // ================================
    //
    // get method
    //
    // ================================
    #[test]
    fn vector_method_get_with_value() {
        let v1 = Vector::<u32, 2>::from_array([1, 2]);
        let v2 = Vector::<f64, 2>::from_array([1.0, 2.0]);
        assert_eq!(v1.get(0).unwrap(), &1);
        assert_eq!(v1.get(1).unwrap(), &2);
        assert_eq!(v1.get(2), None);
        assert_relative_eq!(v2.get(0).unwrap(), &1.0);
        assert_relative_eq!(v2.get(1).unwrap(), &2.0);
        assert_eq!(v2.get(2), None);
    }

    #[test]
    fn vector_method_get_with_range() {
        let v1 = Vector::<u32, 5>::from_array([1, 2, 3, 4, 5]);
        let v2 = Vector::<f64, 5>::from_array([1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(v1.get(0..2).unwrap(), &[1, 2]);
        assert_eq!(v1.get(..2).unwrap(), &[1, 2]);
        assert_eq!(v1.get(2..).unwrap(), &[3, 4, 5]);
        assert_eq!(v1.get(..).unwrap(), &[1, 2, 3, 4, 5]);
        assert_eq!(v1.get(4..8), None);

        assert_eq!(v2.get(0..2).unwrap(), &[1.0, 2.0]);
        assert_eq!(v2.get(..2).unwrap(), &[1.0, 2.0]);
        assert_eq!(v2.get(2..).unwrap(), &[3.0, 4.0, 5.0]);
        assert_eq!(v2.get(..).unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(v2.get(4..8), None);
    }

    #[test]
    fn vector_method_get_mut_with_value() {
        let mut v1 = Vector::<u32, 2>::from_array([1, 2]);
        let mut v2 = Vector::<f64, 2>::from_array([1.0, 2.0]);

        let x1 = v1.get_mut(0).unwrap();
        assert_eq!(*x1, 1);
        *x1 = 10;
        assert_eq!(v1.coord[0], 10);
        assert_eq!(v1.coord[1], 2);

        let x2 = v2.get_mut(0).unwrap();
        assert_relative_eq!(*x2, 1.0);
        *x2 = 10.0;
        assert_relative_eq!(v2.coord[0], 10.0);
        assert_relative_eq!(v2.coord[1], 2.0);

        let mut v3 = Vector::<u32, 1>::default();
        let mut v4 = Vector::<f64, 1>::default();
        assert_eq!(v3.get_mut(10), None);
        assert_eq!(v4.get_mut(10), None);
    }

    #[test]
    fn vector_method_get_mut_with_range() {
        let mut v1 = Vector::<u32, 3>::from_array([1, 2, 3]);
        let mut v2 = Vector::<f64, 3>::from_array([1.0, 2.0, 3.0]);

        let r1 = v1.get_mut(0..2).unwrap();
        assert_eq!(*r1, [1, 2]);
        r1[0] = 5;
        r1[1] = 6;
        assert_eq!(v1.coord, [5, 6, 3]);

        let r2 = v2.get_mut(0..2).unwrap();
        assert_eq!(r2.len(), 2);
        assert_relative_eq!(r2[0], 1.0);
        assert_relative_eq!(r2[1], 2.0);
        r2[0] = 5.0;
        r2[1] = 6.0;
        assert_eq!(v2.coord.len(), 3);
        assert_relative_eq!(v2.coord[0], 5.0);
        assert_relative_eq!(v2.coord[1], 6.0);
        assert_relative_eq!(v2.coord[2], 3.0);
    }

    // ================================
    //
    // iterator traits and method impl
    //
    // ================================
    #[test]
    fn vector_trait_intoiterator_ref() {
        let v1 = Vector::<u32, 3>::from_array([1, 2, 3]);
        let v2 = Vector::<f64, 3>::from_array([1.0, 2.0, 3.0]);

        let mut i: usize = 0;
        for x in &v1 {
            match i {
                0 => assert_eq!(x, &1),
                1 => assert_eq!(x, &2),
                2 => assert_eq!(x, &3),
                _ => unimplemented!(),
            }
            i += 1;
        }

        let mut i: usize = 0;
        for x in &v2 {
            match i {
                0 => assert_relative_eq!(x, &1.0),
                1 => assert_relative_eq!(x, &2.0),
                2 => assert_relative_eq!(x, &3.0),
                _ => unimplemented!(),
            }
            i += 1;
        }
    }

    #[test]
    fn vector_trait_intoiterator_mut_ref() {
        let mut v1 = Vector::<u32, 3>::from_array([1, 2, 3]);
        let mut v2 = Vector::<f64, 3>::from_array([1.0, 2.0, 3.0]);

        let mut i: usize = 0;
        for x in &mut v1 {
            match i {
                0 => assert_eq!(x, &mut 1),
                1 => assert_eq!(x, &mut 2),
                2 => assert_eq!(x, &mut 3),
                _ => unimplemented!(),
            }
            i += 1;
        }

        let mut i: usize = 0;
        for x in &mut v2 {
            match i {
                0 => assert_relative_eq!(x, &mut 1.0),
                1 => assert_relative_eq!(x, &mut 2.0),
                2 => assert_relative_eq!(x, &mut 3.0),
                _ => unimplemented!(),
            }
            i += 1;
        }
    }

    #[test]
    fn vector_trait_intoiterator_owned() {
        let v1 = Vector::<u32, 3>::from_array([1, 2, 3]);
        let v2 = Vector::<f64, 3>::from_array([1.0, 2.0, 3.0]);

        let mut i: usize = 0;
        for x in v1 {
            match i {
                0 => assert_eq!(x, 1),
                1 => assert_eq!(x, 2),
                2 => assert_eq!(x, 3),
                _ => unimplemented!(),
            }
            i += 1;
        }

        let mut i: usize = 0;
        for x in v2 {
            match i {
                0 => assert_relative_eq!(x, 1.0),
                1 => assert_relative_eq!(x, 2.0),
                2 => assert_relative_eq!(x, 3.0),
                _ => unimplemented!(),
            }
            i += 1;
        }
    }

    #[test]
    fn vector_method_iter() {
        let v1 = Vector::<u32, 3>::from_array([1, 2, 3]);
        let v2 = Vector::<f64, 3>::from_array([1.0, 2.0, 3.0]);
        let mut v1_iter = v1.iter();
        let mut v2_iter = v2.iter();

        assert_eq!(v1_iter.next().unwrap(), &1);
        assert_eq!(v1_iter.next().unwrap(), &2);
        assert_eq!(v1_iter.next().unwrap(), &3);
        assert_eq!(v1_iter.next(), None);

        assert_relative_eq!(v2_iter.next().unwrap(), &1.0);
        assert_relative_eq!(v2_iter.next().unwrap(), &2.0);
        assert_relative_eq!(v2_iter.next().unwrap(), &3.0);
        assert_eq!(v2_iter.next(), None);
    }

    // ================================
    //
    // PartialEq trait impl
    //
    // ================================

    #[test]
    fn vector_trait_partial_eq_i8() {
        let v1 = Vector::<i8, 3>::from_array([-1, 2, 3]);
        let v2 = Vector::<i8, 3>::from_array([-1, 2, 3]);
        let v_eq = Vector::<i8, 3>::from_array([-1, 2, 3]);
        let v_diff = Vector::<i8, 3>::from_array([-1, 2, 4]);

        let v_zero = Vector::<i8, 3>::from_array([0, 0, 0]);
        let v_zero_neg = Vector::<i8, 3>::from_array([-0, -0, -0]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
        assert!(v_zero == v_zero_neg);
    }

    #[test]
    fn vector_trait_partial_eq_i16() {
        let v1 = Vector::<i16, 3>::from_array([-1, 2, 3]);
        let v2 = Vector::<i16, 3>::from_array([-1, 2, 3]);
        let v_eq = Vector::<i16, 3>::from_array([-1, 2, 3]);
        let v_diff = Vector::<i16, 3>::from_array([-1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_i32() {
        let v1 = Vector::<i32, 3>::from_array([-1, 2, 3]);
        let v2 = Vector::<i32, 3>::from_array([-1, 2, 3]);
        let v_eq = Vector::<i32, 3>::from_array([-1, 2, 3]);
        let v_diff = Vector::<i32, 3>::from_array([-1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_i64() {
        let v1 = Vector::<i64, 3>::from_array([-1, 2, 3]);
        let v2 = Vector::<i64, 3>::from_array([-1, 2, 3]);
        let v_eq = Vector::<i64, 3>::from_array([-1, 2, 3]);
        let v_diff = Vector::<i64, 3>::from_array([-1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_i128() {
        let v1 = Vector::<i128, 3>::from_array([-1, 2, 3]);
        let v2 = Vector::<i128, 3>::from_array([-1, 2, 3]);
        let v_eq = Vector::<i128, 3>::from_array([-1, 2, 3]);
        let v_diff = Vector::<i128, 3>::from_array([-1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_u8() {
        let v1 = Vector::<u8, 3>::from_array([1, 2, 3]);
        let v2 = Vector::<u8, 3>::from_array([1, 2, 3]);
        let v_eq = Vector::<u8, 3>::from_array([1, 2, 3]);
        let v_diff = Vector::<u8, 3>::from_array([1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_u16() {
        let v1 = Vector::<u16, 3>::from_array([1, 2, 3]);
        let v2 = Vector::<u16, 3>::from_array([1, 2, 3]);
        let v_eq = Vector::<u16, 3>::from_array([1, 2, 3]);
        let v_diff = Vector::<u16, 3>::from_array([1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_u32() {
        let v1 = Vector::<u32, 3>::from_array([1, 2, 3]);
        let v2 = Vector::<u32, 3>::from_array([1, 2, 3]);
        let v_eq = Vector::<u32, 3>::from_array([1, 2, 3]);
        let v_diff = Vector::<u32, 3>::from_array([1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_u64() {
        let v1 = Vector::<u64, 3>::from_array([1, 2, 3]);
        let v2 = Vector::<u64, 3>::from_array([1, 2, 3]);
        let v_eq = Vector::<u64, 3>::from_array([1, 2, 3]);
        let v_diff = Vector::<u64, 3>::from_array([1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_u128() {
        let v1 = Vector::<u128, 3>::from_array([1, 2, 3]);
        let v2 = Vector::<u128, 3>::from_array([1, 2, 3]);
        let v_eq = Vector::<u128, 3>::from_array([1, 2, 3]);
        let v_diff = Vector::<u128, 3>::from_array([1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_isize() {
        let v1 = Vector::<isize, 3>::from_array([1, 2, 3]);
        let v2 = Vector::<isize, 3>::from_array([1, 2, 3]);
        let v_eq = Vector::<isize, 3>::from_array([1, 2, 3]);
        let v_diff = Vector::<isize, 3>::from_array([1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_usize() {
        let v1 = Vector::<usize, 3>::from_array([1, 2, 3]);
        let v2 = Vector::<usize, 3>::from_array([1, 2, 3]);
        let v_eq = Vector::<usize, 3>::from_array([1, 2, 3]);
        let v_diff = Vector::<usize, 3>::from_array([1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_f32() {
        let v1 = Vector::<f32, 3>::from_array([-1.1, 2.2, 3.3]);
        let v2 = Vector::<f32, 3>::from_array([-1.1, 2.2, 3.3]);
        let v_eq = Vector::<f32, 3>::from_array([-1.1, 2.2, 3.3]);
        let v_diff = Vector::<f32, 3>::from_array([-1.1, 2.2, 4.4]);
        let v_close = Vector::<f32, 3>::from_array([-1.1 + (f32::EPSILON * 2.), 2.2, 3.3]);

        let v_zero = Vector::<f32, 3>::from_array([0.0, 0.0, 0.0]);
        let v_zero_eq = Vector::<f32, 3>::from_array([0.0, 0.0, 0.0]);
        let v_zero_neg_eq = Vector::<f32, 3>::from_array([-0.0, -0.0, -0.0]);

        let v_nan = Vector::<f32, 3>::from_array([f32::NAN, 0.0, 0.0]);
        let v_nan_diff = Vector::<f32, 3>::from_array([f32::NAN, 0.0, 0.0]);

        let v_inf_pos = Vector::<f32, 3>::from_array([f32::INFINITY, 0.0, 0.0]);
        let v_inf_pos_eq = Vector::<f32, 3>::from_array([f32::INFINITY, 0.0, 0.0]);
        let v_inf_neg = Vector::<f32, 3>::from_array([f32::NEG_INFINITY, 0.0, 0.0]);
        let v_inf_neg_eq = Vector::<f32, 3>::from_array([f32::NEG_INFINITY, 0.0, 0.0]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
        assert!(v1 != v_close);
        assert!(v_zero == v_zero_eq);
        assert!(v_zero == v_zero_neg_eq); // zero and neg zero are defined as equivalent
        assert!(v_nan != v_nan_diff); // NaN comparisons are defined as different
        assert!(v_inf_pos == v_inf_pos_eq); // postive infinity comparisons are defined as equivalent
        assert!(v_inf_neg == v_inf_neg_eq); // negative infinity comparisons are defined as equivalent
    }

    #[test]
    fn vector_trait_partial_eq_f64() {
        let v1 = Vector::<f64, 3>::from_array([-1.1, 2.2, 3.3]);
        let v2 = Vector::<f64, 3>::from_array([-1.1, 2.2, 3.3]);
        let v_eq = Vector::<f64, 3>::from_array([-1.1, 2.2, 3.3]);
        let v_diff = Vector::<f64, 3>::from_array([-1.1, 2.2, 4.4]);
        let v_close = Vector::<f64, 3>::from_array([-1.1 + (f64::EPSILON * 2.), 2.2, 3.3]);

        let v_zero = Vector::<f64, 3>::from_array([0.0, 0.0, 0.0]);
        let v_zero_eq = Vector::<f64, 3>::from_array([0.0, 0.0, 0.0]);
        let v_zero_neg_eq = Vector::<f64, 3>::from_array([-0.0, -0.0, -0.0]);

        let v_nan = Vector::<f64, 3>::from_array([f64::NAN, 0.0, 0.0]);
        let v_nan_diff = Vector::<f64, 3>::from_array([f64::NAN, 0.0, 0.0]);

        let v_inf_pos = Vector::<f64, 3>::from_array([f64::INFINITY, 0.0, 0.0]);
        let v_inf_pos_eq = Vector::<f64, 3>::from_array([f64::INFINITY, 0.0, 0.0]);
        let v_inf_neg = Vector::<f64, 3>::from_array([f64::NEG_INFINITY, 0.0, 0.0]);
        let v_inf_neg_eq = Vector::<f64, 3>::from_array([f64::NEG_INFINITY, 0.0, 0.0]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
        assert!(v1 != v_close);
        assert!(v_zero == v_zero_eq);
        assert!(v_zero == v_zero_neg_eq); // zero and neg zero are defined as equivalent
        assert!(v_nan != v_nan_diff); // NaN comparisons are defined as different
        assert!(v_inf_pos == v_inf_pos_eq); // postive infinity comparisons are defined as equivalent
        assert!(v_inf_neg == v_inf_neg_eq); // negative infinity comparisons are defined as equivalent
    }
}
