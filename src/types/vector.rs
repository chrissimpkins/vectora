//! Vector types.

// use std::cmp::PartialOrd;
use std::{
    borrow::{Borrow, BorrowMut},
    iter::{IntoIterator, Sum},
    ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, Neg, Sub},
    slice::SliceIndex,
};

use approx::{relative_eq, RelativeEq};
use num::{Float, Num};

use crate::errors::VectorError;

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

/// A generic, fixed length vector type that supports computation with N-dimensional
/// scalar data.
#[derive(Copy, Clone, Debug)]
pub struct Vector<T, const N: usize>
where
    T: Num + Copy + Sync + Send,
{
    /// N-dimensional vector component values.
    pub components: [T; N],
}

// ================================
//
// Instantiation method impl
//
// ================================
impl<T, const N: usize> Vector<T, N>
where
    T: Num + Copy + Default + Sync + Send,
{
    /// Returns a new [`Vector`] filled with default scalar component values.
    ///
    /// # Examples
    ///
    /// ## Turbofish syntax
    ///
    /// Use turbofish syntax to define the numeric type of the vector
    /// component scalar values and number of vector dimensions:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let vec_2d_i32 = Vector::<i32, 2>::new();
    /// assert_eq!(vec_2d_i32.len(), 2);
    /// assert_eq!(vec_2d_i32[0], i32::default());
    /// assert_eq!(vec_2d_i32[1], i32::default());
    ///
    /// let vec_3d_f64 = Vector::<f64, 3>::new();
    /// assert_eq!(vec_3d_f64.len(), 3);
    /// assert_eq!(vec_3d_f64[0], f64::default());
    /// assert_eq!(vec_3d_f64[1], f64::default());
    /// assert_eq!(vec_3d_f64[2], f64::default());
    /// ```
    ///
    /// ## Type alias syntax
    ///
    /// Simplify instantiation with one of the defined type aliases:
    ///
    /// ```
    /// # use vectora::types::vector::*;
    /// let vec_2d_i32_1 = Vector2d::<i32>::new();
    /// let vec_2d_i32_2 = Vector2dI32::new();
    ///
    /// let vec_3d_f64_1 = Vector3d::<f64>::new();
    /// let vec_3d_f64_2 = Vector3dF64::new();
    /// ```
    ///
    /// ## With type inference
    ///
    /// There is no requirement for additional type data when the type can be inferrred
    /// by the compiler:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v: Vector<u32, 3> = Vector::new();
    /// ```
    pub fn new() -> Self {
        Self { components: [T::default(); N] }
    }

    /// Returns a new [`Vector`] filled with scalar component values defined as zero.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ## Integer types
    ///
    /// ```
    /// # use vectora::types::vector::*;
    /// let vi: Vector<i32, 3> = Vector::zero();
    /// assert_eq!(vi.len(), 3);
    /// assert_eq!(vi[0], 0 as i32);
    /// assert_eq!(vi[1], 0 as i32);
    /// assert_eq!(vi[2], 0 as i32);
    /// ```
    ///
    /// ## Float types
    ///
    /// ```
    /// # use vectora::types::vector::*;
    /// let vf: Vector<f64, 3> = Vector::zero();
    /// assert_eq!(vf.len(), 3);
    /// assert_eq!(vf[0], 0.0 as f64);
    /// assert_eq!(vf[1], 0.0 as f64);
    /// assert_eq!(vf[2], 0.0 as f64);
    /// ```
    pub fn zero() -> Self {
        Self { components: [T::zero(); N] }
    }
}

impl<T, const N: usize> Default for Vector<T, N>
where
    T: Num + Copy + Default + Sync + Send,
{
    /// Returns a new [`Vector`] filled with default scalar values for each component.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v1 = Vector::<i32, 2>::default();
    /// let v2 = Vector::<f64, 3>::default();
    /// ```
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    /// Returns a reference to a [`Vector`] index value or range,
    /// or `None` if the index is out of bounds.
    ///
    /// This method provides safe, bounds checked immutable access to scalar component
    /// values.
    ///
    /// # Examples
    ///
    /// ## With Index Values
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// assert_eq!(v.get(0), Some(&1));
    /// assert_eq!(v.get(1), Some(&2));
    /// assert_eq!(v.get(3), None);
    /// ```
    ///
    /// ## With Index Ranges
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 5>::from(&[1, 2, 3, 4, 5]);
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
        self.components.get(index)
    }

    /// Returns a mutable reference to a [`Vector`] index value or range,
    /// or `None` if the index is out of bounds.
    ///
    /// This method provides safe, bounds checked mutable access to scalar
    /// component values.
    ///
    /// # Examples
    ///
    /// ## With Index Values
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from(&[1, 2, 3]);
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
    /// let mut v = Vector::<i32, 3>::from(&[1, 2, 3]);
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
        self.components.get_mut(index)
    }

    /// Returns an iterator over immutable [`Vector`] scalar component references.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = &Vector::<i32, 3>::from(&[1, 2, 3]);
    /// let mut v_iter = v.iter();
    /// assert_eq!(v_iter.next(), Some(&1));
    /// assert_eq!(v_iter.next(), Some(&2));
    /// assert_eq!(v_iter.next(), Some(&3));
    /// assert_eq!(v_iter.next(), None);
    /// ```
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.components.iter()
    }

    /// Returns an iterator over mutable [`Vector`] scalar component references.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from(&[1, 2, 3]);
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
        self.components.iter_mut()
    }

    /// Returns a [`slice`] representation of the [`Vector`] scalar components.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// let x: &[i32] = v.as_slice();
    /// assert_eq!(x, &[1, 2, 3]);
    /// ```
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.components[..]
    }

    /// Returns a mutable [`slice`] representation of the [`Vector`] scalar components.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// let mut x: &mut [i32] = v.as_mut_slice();
    /// assert_eq!(x, &[1, 2, 3]);
    /// x[0] = 10;
    /// assert_eq!(x, &[10,2,3]);
    /// # assert_eq!(v[0], 10);
    /// ```
    ///
    /// Note: The edit above **changes** the [`Vector`] data:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// # let mut v = Vector::<u32, 3>::from(&[1, 2, 3]);
    /// # let mut x: &mut [u32] = v.as_mut_slice();
    /// # assert_eq!(x, &[1, 2, 3]);
    /// # x[0] = 10;
    /// # assert_eq!(x, &[10,2,3]);
    /// assert_eq!(v[0], 10);
    /// ```
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.components[..]
    }

    /// Returns an [`array`] reference representation of the [`Vector`] scalar components.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// let x: &[i32;3] = v.as_array();
    /// assert_eq!(x, &[1, 2, 3]);
    /// ```
    #[inline]
    pub fn as_array(&self) -> &[T; N] {
        &self.components
    }

    /// Returns a mutable [`array`] reference representation of the [`Vector`] scalar components.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// let mut x: &mut [i32;3] = v.as_mut_array();
    /// assert_eq!(x, &[1, 2, 3]);
    /// x[0] = 10;
    /// assert_eq!(x, &[10,2,3]);
    /// # assert_eq!(v[0], 10);
    /// ```
    ///
    /// Note: The edit above **changes** the [`Vector`] data:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// # let mut v = Vector::<u32, 3>::from(&[1, 2, 3]);
    /// # let mut x: &mut [u32;3] = v.as_mut_array();
    /// # assert_eq!(x, &[1, 2, 3]);
    /// # x[0] = 10;
    /// # assert_eq!(x, &[10,2,3]);
    /// assert_eq!(v[0], 10);
    /// ```
    #[inline]
    pub fn as_mut_array(&mut self) -> &mut [T; N] {
        &mut self.components
    }

    /// Returns a new, allocated [`array`] representation of the [`Vector`] scalar components
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// let mut x: [i32; 3] = v.to_array();
    /// assert_eq!(x, [1,2,3]);
    /// x[0] = 10;
    /// assert_eq!(x, [10, 2, 3]);
    /// # assert_eq!(v[0], 1);
    /// ```
    ///
    /// Note: The edit above returns a new, owned array and
    /// **does not change** the [`Vector`] data:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// # let v = Vector::<u32, 3>::from(&[1, 2, 3]);
    /// # let mut x: [u32; 3] = v.to_array();
    /// # assert_eq!(x, [1,2,3]);
    /// # x[0] = 10;
    /// # assert_eq!(x, [10, 2, 3]);
    /// assert_eq!(v[0], 1);
    /// ```
    #[inline]
    pub fn to_array(&self) -> [T; N] {
        self.components
    }

    /// Returns a new, allocated [`Vec`] representation of the [`Vector`] scalar components.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// let mut x: Vec<i32> = v.to_vec();
    /// assert_eq!(x, Vec::from([1,2,3]));
    /// x[0] = 10;
    /// assert_eq!(x, Vec::from([10, 2, 3]));
    /// # assert_eq!(v[0], 1);
    /// ```
    ///
    /// Note: The edit above returns a new, owned [`Vec`] and
    /// **does not change** the [`Vector`] data:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// # let v = Vector::<u32, 3>::from(&[1, 2, 3]);
    /// # let mut x: Vec<u32> = v.to_vec();
    /// # assert_eq!(x, Vec::from([1,2,3]));
    /// # x[0] = 10;
    /// # assert_eq!(x, Vec::from([10, 2, 3]));
    /// assert_eq!(v[0], 1);
    /// ```
    #[inline]
    pub fn to_vec(&self) -> Vec<T> {
        Vec::from(self.components)
    }

    /// Returns the [`Vector`] component number.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v2d: Vector<i32, 2> = Vector::new();
    /// assert_eq!(v2d.len(), 2);
    ///
    /// let v3d: Vector<f64, 3> = Vector::new();
    /// assert_eq!(v3d.len(), 3);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Returns `true` if the [`Vector`] contains no items and `false` otherwise.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Vector addition with mutation of the calling [`Vector`].
    ///
    /// Returns a mutable reference to the calling [`Vector`].
    ///
    /// Vector addition with the [`+` operator overload](#impl-Add<Vector<T%2C%20N>>) allocates a new [`Vector`].  This method
    /// supports in-place vector addition mutation of the calling [`Vector`] with data in the
    /// parameter [`Vector`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// let other = Vector::<i32, 3>::from(&[4, 5, 6]);
    /// v.mut_add(&other);
    ///
    /// assert_eq!(v[0], 5);
    /// assert_eq!(v[1], 7);
    /// assert_eq!(v[2], 9);
    /// ```
    #[inline]
    pub fn mut_add(&mut self, rhs: &Vector<T, N>) -> &mut Self {
        for _ in self.components.iter_mut().zip(rhs).map(|(a, b)| *a = *a + *b) {}

        self
    }

    /// Vector subtraction with mutation of the calling [`Vector`].
    ///
    /// Returns a mutable reference to the calling [`Vector`].
    ///
    /// Vector subtraction with the [`-` operator overload](#impl-Sub<Vector<T%2C%20N>>) allocates a new [`Vector`].  This method
    /// supports in-place vector subtraction mutation of the calling [`Vector`] with data in the
    /// parameter [`Vector`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// let other = Vector::<i32, 3>::from(&[4, 5, 6]);
    /// v.mut_sub(&other);
    ///
    /// assert_eq!(v[0], -3);
    /// assert_eq!(v[1], -3);
    /// assert_eq!(v[2], -3);
    /// ```
    #[inline]
    pub fn mut_sub(&mut self, rhs: &Vector<T, N>) -> &mut Self {
        for _ in self.components.iter_mut().zip(rhs.iter()).map(|(a, b)| *a = *a - *b) {}

        self
    }

    /// Scalar multiplication with mutation of the calling [`Vector`].
    ///
    /// Returns a mutable reference to the calling [`Vector`].
    ///
    /// Scalar multiplication with the [`*` operator overload](#impl-Mul<T>) allocates a
    /// new [`Vector`].  This method supports in-place scalar multiplication mutation of the calling
    /// [`Vector`] with a scalar parameter value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// v.mut_mul(4);
    ///
    /// assert_eq!(v[0], 4);
    /// assert_eq!(v[1], 8);
    /// assert_eq!(v[2], 12);
    /// ```
    #[inline]
    pub fn mut_mul(&mut self, scale: T) -> &mut Self {
        for _ in self.components.iter_mut().map(|a| *a = *a * scale) {}

        self
    }

    /// Returns the dot product of two vectors.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v1: Vector<i32, 3> = Vector::from([1, 3, -5]);
    /// let v2: Vector<i32, 3> = Vector::from([4, -2, -1]);
    /// let x1 = v1 * 3;
    /// let x2 = v2 * 6;
    /// assert_eq!(v1.dot(&v2), 3);
    /// assert_eq!(v2.dot(&v1), 3);
    /// assert_eq!(-v1.dot(&-v2), 3);
    /// assert_eq!(x1.dot(&x2), (3 * 6) * v1.dot(&v2));
    /// ```
    #[inline]
    pub fn dot(&self, other: &Vector<T, N>) -> T
    where
        T: Num + Copy + std::iter::Sum<T> + Sync + Send,
    {
        self.components.iter().zip(other.components.iter()).map(|(a, b)| *a * *b).sum()
    }

    /// Returns the displacement [`Vector`] from a parameter [`Vector`] to the calling [`Vector`].
    ///
    /// Note: This is an alias for the [`Vector::sub`] vector subtraction
    /// method and the operation can be performed with the overloaded `-` operator.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let to = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// let from = Vector::<i32, 3>::from(&[2, 4, 6]);
    /// let v_d = to.displacement(&from);
    ///
    /// assert_eq!(v_d[0], -1);
    /// assert_eq!(v_d[1], -2);
    /// assert_eq!(v_d[2], -3);
    /// ```
    pub fn displacement(&self, other: &Vector<T, N>) -> Self {
        self.sub(*other)
    }

    /// Returns a [`Vector`] with the same magnitude and opposite direction for
    /// a non-zero [`Vector`].
    ///
    /// If the caller is a zero [`Vector`], the method returns a zero [`Vector`].
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// let v_o = v.opposite();
    ///
    /// assert_eq!(v_o[0], -1);
    /// assert_eq!(v_o[1], -2);
    /// assert_eq!(v_o[2], -3);
    /// assert_eq!(v + v_o, Vector::zero());
    /// ```
    ///
    /// The zero vector case:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v_zero = Vector::<i32, 3>::zero();
    /// let v_zero_o = v_zero.opposite();
    /// assert_eq!(v_zero_o, v_zero);
    /// ```
    pub fn opposite(&self) -> Self {
        self.mul(T::zero() - T::one())
    }

    /// Returns a [`Vector`] that is scaled by a given scalar parameter value.
    ///
    /// Note: This is an alias for the [`Vector::mul`] scalar multiplication
    /// method and the operation can be performed with the overloaded `*` operator.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// let v_s = v.scale(10);
    ///
    /// assert_eq!(v_s[0], 10);
    /// assert_eq!(v_s[1], 20);
    /// assert_eq!(v_s[2], 30);
    /// ```
    #[inline]
    pub fn scale(&self, scale: T) -> Self {
        self.mul(scale)
    }

    /// Returns a translated [`Vector`] with displacement defined by a
    /// translation [`Vector`] parameter.
    ///
    /// Note: This is an alias for the [`Vector::add`] vector addition method
    /// and the operation can be performed with the overloaded `+` operator.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// let translation_vec = Vector::<i32, 3>::from(&[4, 5, 6]);
    /// let v_t = v.translate(&translation_vec);
    ///
    /// assert_eq!(v_t[0], 5);
    /// assert_eq!(v_t[1], 7);
    /// assert_eq!(v_t[2], 9);
    /// ```
    #[inline]
    pub fn translate(&self, translation_vector: &Vector<T, N>) -> Self {
        self.add(*translation_vector)
    }

    // ================================
    //
    // Private methods
    //
    // ================================
    #[inline]
    fn partial_eq_int(&self, other: &Self) -> bool {
        self.components == other.components
    }

    #[inline]
    fn partial_eq_float(&self, other: &Self) -> bool
    where
        T: Num + Copy + RelativeEq<T>,
    {
        for (i, item) in self.components.iter().enumerate() {
            if !relative_eq!(*item, other[i]) {
                return false;
            }
        }

        true
    }
}

// ================================
//
// Numeric type specific methods
//
// ================================
impl<T, const N: usize> Vector<T, N>
where
    T: Float + Copy + Sync + Send + Sum,
{
    /// Returns the magnitude of the displacement vector between the
    /// calling float [`Vector`] and a float [`Vector`] parameter.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    ///# use vectora::types::vector::Vector;
    /// use approx::assert_relative_eq;
    ///
    /// let v1: Vector<f64, 2> = Vector::from([2.0, 2.0]);
    /// let v2: Vector<f64, 2> = Vector::from([4.0, 4.0]);
    ///
    /// assert_relative_eq!(v1.distance(&v2), 8.0_f64.sqrt());
    /// assert_relative_eq!(v1.distance(&v1), 0.0_f64);
    /// ```
    ///
    /// Note: This method can be used with a **lossless** integer
    /// to float [`Vector`] type cast:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// use approx::assert_relative_eq;
    ///
    /// let v1: Vector<i32, 2> = Vector::from([2, 2]);
    /// let v1_f: Vector<f64, 2> = v1.into();
    /// ```
    ///
    /// Lossy integer to float type casts are **not** supported (e.g.,
    /// `i64` to `f64`).
    pub fn distance(&self, other: &Vector<T, N>) -> T {
        (*self - *other).magnitude()
    }

    /// Returns the vector magnitude for a float [`Vector`].
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// use approx::assert_relative_eq;
    ///
    /// let v1: Vector<f64, 2> = Vector::from([2.8, 2.6]);
    /// let v2: Vector<f64, 2> = Vector::from([-2.8, -2.6]);
    /// let v_zero: Vector<f64, 2> = Vector::zero();
    ///
    /// assert_relative_eq!(v1.magnitude(), 3.82099463490856);
    /// assert_relative_eq!(v2.magnitude(), 3.82099463490856);
    /// assert_relative_eq!(v_zero.magnitude(), 0.0);
    /// ```
    ///
    /// Note: This method can be used with a **lossless** integer
    /// to float [`Vector`] type cast:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// use approx::assert_relative_eq;
    ///
    /// let v1: Vector<i32, 2> = Vector::from([2, 2]);
    /// let v1_f: Vector<f64, 2> = v1.into();
    /// ```
    ///
    /// Lossy integer to float type casts are **not** supported (e.g.,
    /// `i64` to `f64`).
    pub fn magnitude(&self) -> T {
        let x: T = self.components.iter().map(|a| *a * *a).sum();
        x.sqrt()
    }

    /// Returns a new, normalized float [`Vector`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// use approx::assert_relative_eq;
    ///
    /// let v: Vector<f64, 2> = Vector::from([25.123, 30.456]);
    /// assert_relative_eq!(v.normalize().magnitude(), 1.0);
    /// assert_relative_eq!(v[0], 25.123);
    /// assert_relative_eq!(v[1], 30.456);
    /// ```
    ///
    /// Note: This method can be used with a **lossless** integer
    /// to float [`Vector`] type cast:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// use approx::assert_relative_eq;
    ///
    /// let v1: Vector<i32, 2> = Vector::from([2, 2]);
    /// let v1_f: Vector<f64, 2> = v1.into();
    /// ```
    ///
    /// Lossy integer to float type casts are **not** supported (e.g.,
    /// `i64` to `f64`).
    pub fn normalize(&self) -> Self
    where
        T: Float + Copy + Sync + Send + Sum,
    {
        let mut new_components = [T::zero(); N];
        let this_magnitude = self.magnitude();
        for _ in
            self.components.iter().enumerate().map(|(i, a)| new_components[i] = *a / this_magnitude)
        {
        }
        Self { components: new_components }
    }

    /// Normalizes a float [`Vector`] in place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// use approx::assert_relative_eq;
    ///
    /// let mut v: Vector<f64, 2> = Vector::from([25.123, 30.456]);
    /// assert_relative_eq!(v.mut_normalize().magnitude(), 1.0);
    /// assert_relative_eq!(v[0], 0.6363347262144607);
    /// assert_relative_eq!(v[1], 0.7714130645857428);
    /// ```
    ///
    /// Note: This method can be used with a **lossless** integer
    /// to float [`Vector`] type cast:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// use approx::assert_relative_eq;
    ///
    /// let v1: Vector<i32, 2> = Vector::from([2, 2]);
    /// let v1_f: Vector<f64, 2> = v1.into();
    /// ```
    ///
    /// Lossy integer to float type casts are **not** supported (e.g.,
    /// `i64` to `f64`).
    pub fn mut_normalize(&mut self) -> Self
    where
        T: Float + Copy + Sync + Send + Sum,
    {
        let this_magnitude = self.magnitude();
        for _ in self.components.iter_mut().map(|a| *a = *a / this_magnitude) {}
        *self
    }
}

// ================================
//
// Index / IndexMut trait impl
//
// ================================
impl<T, const N: usize> Index<usize> for Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    type Output = T;
    /// Returns [`Vector`] scalar component values by zero-based index.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// assert_eq!(v[0], 1);
    /// assert_eq!(v[1], 2);
    /// assert_eq!(v[2], 3);
    /// ```
    ///
    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.components[i]
    }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    /// Returns mutable [`Vector`] scalar component values by zero-based index.
    ///
    /// Supports scalar component value assignment by index.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from(&[1, 2, 3]);
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
        &mut self.components[i]
    }
}

// ================================
//
// Iter / IntoIterator trait impl
//
// ================================

impl<T, const N: usize> IntoIterator for Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    type Item = T;
    type IntoIter = std::array::IntoIter<Self::Item, N>;

    /// Creates a consuming iterator that iterates over scalar components by value.
    fn into_iter(self) -> Self::IntoIter {
        self.components.into_iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    /// Creates an iterator over immutable scalar component references.
    fn into_iter(self) -> Self::IntoIter {
        self.components.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    /// Creates an iterator over mutable scalar component references.
    fn into_iter(self) -> Self::IntoIter {
        self.components.iter_mut()
    }
}

// ================================
//
// FromIterator trait impl
//
// ================================

impl<T, const N: usize> FromIterator<T> for Vector<T, N>
where
    T: Num + Copy + Default + Sync + Send,
{
    /// FromIterator trait implementation with support for `collect`.
    ///
    /// # Important
    ///
    /// This implementation is designed to be permissive across iterables
    /// with lengths that differ from the requested [`Vector`] length. The
    /// approaches to underflow and overflow are:
    ///
    /// - On underflow: take all items in the iterator and fill subsequent
    /// undefined components with the default value for the numeric type
    /// (`T::default`)
    /// - On overflow: truncate data after the first N items in the iterator
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v: Vector<i32, 3> = [1, 2, 3].into_iter().collect();
    /// assert_eq!(v.len(), 3);
    /// assert_eq!(v[0], 1 as i32);
    /// assert_eq!(v[1], 2 as i32);
    /// assert_eq!(v[2], 3 as i32);
    /// ```
    ///
    /// ## Overflow Example
    ///
    /// Three dimensional data used to instantiate a two dimensional
    /// [`Vector`] results in truncation.
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v: Vector<i32, 2> = [1, 2, 3].into_iter().collect();
    /// assert_eq!(v.len(), 2);
    /// assert_eq!(v[0], 1 as i32);
    /// assert_eq!(v[1], 2 as i32);
    /// ```
    ///
    /// ## Underflow Example
    ///
    /// Two dimensional data used to instantiate a three dimensional
    /// [`Vector`] results in a default numeric type value fill for
    /// the undefined final component.
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v: Vector<i32, 3> = [1, 2].into_iter().collect();
    /// assert_eq!(v.len(), 3);
    /// assert_eq!(v[0], 1 as i32);
    /// assert_eq!(v[1], 2 as i32);
    /// assert_eq!(v[2], 0 as i32);
    /// ```
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Vector<T, N> {
        let mut newvec = Vector::<T, N>::new();
        let mut it = iter.into_iter();
        // n.b. *Truncation on overflows*
        // We take a maximum of N items from the iterator.
        // This results in truncation on overflow.
        for i in 0..N {
            // n.b. *Zero value fills on underflows*
            // no need to edit values here because
            // the type was instantiated with default
            // type-secific zero values
            if let Some(c) = it.next() {
                newvec[i] = c
            }
        }

        newvec
    }
}

// ================================
//
// PartialEq trait impl
//
// ================================

/// PartialEq trait implementation for [`Vector`] with integer component types.
///
/// These comparisons establish the symmetry and transitivity relationships
/// required for the partial equivalence relation definition with integer types.
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

/// PartialEq trait implementation for [`Vector`] with float component types.
///
/// These comparisons establish the symmetry and transitivity relationships
/// required for the partial equivalence relation definition with floating point
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
    T: Num + Copy + Sync + Send,
{
    fn as_ref(&self) -> &Vector<T, N> {
        self
    }
}

impl<T, const N: usize> AsRef<[T]> for Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    fn as_ref(&self) -> &[T] {
        &self.components
    }
}

impl<T, const N: usize> AsMut<Vector<T, N>> for Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    fn as_mut(&mut self) -> &mut Vector<T, N> {
        self
    }
}

impl<T, const N: usize> AsMut<[T]> for Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.components
    }
}

// ================================
//
// Borrow trait impl
//
// ================================
impl<T, const N: usize> Borrow<[T]> for Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    fn borrow(&self) -> &[T] {
        &self.components
    }
}

impl<T, const N: usize> BorrowMut<[T]> for Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    fn borrow_mut(&mut self) -> &mut [T] {
        &mut self.components
    }
}

// ================================
//
// Deref / DerefMut trait impl
//
// ================================
impl<T, const N: usize> Deref for Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.components
    }
}

impl<T, const N: usize> DerefMut for Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.components
    }
}

// ================================
//
// From trait impl
//
// ================================
impl<T, const N: usize> From<[T; N]> for Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    /// Returns a new [`Vector`] as defined by an [`array`] parameter.
    ///
    /// Note: The [`Vector`] dimension size is defined by the fixed [`array`]
    /// size.
    #[inline]
    fn from(t_n_array: [T; N]) -> Vector<T, N> {
        Vector { components: t_n_array }
    }
}

impl<T, const N: usize> From<&[T; N]> for Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    /// Returns a new [`Vector`] as defined by an [`array`] reference parameter.
    ///
    /// Note: The [`Vector`] dimension size is defined by the fixed [`array`]
    /// size.
    #[inline]
    fn from(t_n_array: &[T; N]) -> Vector<T, N> {
        Vector { components: *t_n_array }
    }
}

impl<T, const N: usize> TryFrom<Vec<T>> for Vector<T, N>
where
    T: Num + Copy + Default + Sync + Send + std::fmt::Debug,
{
    type Error = VectorError;
    /// Returns a new [`Vector`] as defined by the [`Vec`] parameter.
    ///
    /// [`Vec`] lengths are not known at compile time. Bounds checks are used in
    /// this approach.  This is slower than instantiation from arrays and will fail
    /// with overflows and underflows.
    ///
    /// # Errors
    ///
    /// Raises [`VectorError::TryFromVecError`] when the [`Vec`] parameter length
    /// does not equal the expected [`Vector`] component length.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let _: Vector<i32, 3> = Vector::try_from(Vec::from([1, 2, 3])).unwrap();
    /// let _: Vector<f64, 2> = Vector::try_from(Vec::from([1.0, 2.0])).unwrap();
    /// ```
    ///
    /// Callers should confirm that the length of the [`Vec`] is
    /// the same as the number of requested [`Vector`] dimensions.  The following
    /// code raises [`VectorError::TryFromVecError`] on an attempt to make a
    /// three dimensional [`Vector`] with two dimensional data:
    ///
    /// ```
    ///# use vectora::types::vector::Vector;
    /// let v = vec![1 as i32, 2 as i32];
    /// let e = Vector::<i32, 3>::try_from(v);
    /// assert!(e.is_err());
    /// ```
    #[inline]
    fn try_from(t_vec: Vec<T>) -> Result<Vector<T, N>, VectorError> {
        if t_vec.len() != N {
            return Err(VectorError::TryFromVecError(format!(
                "expected Vec with {} items, but received Vec with {} items",
                N,
                t_vec.len()
            )));
        }
        match t_vec.try_into() {
            Ok(s) => Ok(Self { components: s }),
            Err(err) => Err(VectorError::TryFromVecError(format!(
                "failed to cast Vec to Vector type: {:?}",
                err
            ))),
        }
    }
}

impl<T, const N: usize> TryFrom<&Vec<T>> for Vector<T, N>
where
    T: Num + Copy + Default + Sync + Send + std::fmt::Debug,
{
    type Error = VectorError;
    /// Returns a new [`Vector`] as defined by the [`Vec`] reference parameter.
    ///
    /// [`Vec`] lengths are not known at compile time. Bounds checks are used in
    /// this approach.  This is slower than instantiation from arrays and will fail
    /// with overflows and underflows.
    ///
    /// # Errors
    ///
    /// Raises [`VectorError::TryFromVecError`] when the [`Vec`] parameter length
    /// does not equal the expected [`Vector`] component length.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let _: Vector<i32, 3> = Vector::try_from(&Vec::from([1, 2, 3])).unwrap();
    /// let _: Vector<f64, 2> = Vector::try_from(&Vec::from([1.0, 2.0])).unwrap();
    /// ```
    ///
    /// Callers should confirm that the length of the [`Vec`] is
    /// the same as the number of requested [`Vector`] dimensions.  The following
    /// code raises [`VectorError::TryFromVecError`] on an attempt to make a
    /// three dimensional [`Vector`] with two dimensional data:
    ///
    /// ```
    ///# use vectora::types::vector::Vector;
    /// let v = vec![1 as i32, 2 as i32];
    /// let e = Vector::<i32, 3>::try_from(&v);
    /// assert!(e.is_err());
    /// ```
    #[inline]
    fn try_from(t_vec: &Vec<T>) -> Result<Vector<T, N>, VectorError> {
        if t_vec.len() != N {
            return Err(VectorError::TryFromVecError(format!(
                "expected Vec with {} items, but received Vec with {} items",
                N,
                t_vec.len()
            )));
        }

        match t_vec.clone().try_into() {
            Ok(s) => Ok(Self { components: s }),
            Err(err) => Err(VectorError::TryFromVecError(format!(
                "failed to cast Vec to Vector type: {:?}",
                err
            ))),
        }
    }
}

impl<T, const N: usize> TryFrom<&[T]> for Vector<T, N>
where
    T: Num + Copy + Default + Sync + Send,
{
    type Error = VectorError;
    /// Returns a new [`Vector`] as defined by a [`slice`] parameter.
    ///
    /// # Errors
    ///
    /// Raises [`VectorError::TryFromSliceError`] when the [`slice`] parameter length
    /// is not equal to the requested [`Vector`] component length.
    ///
    /// # Examples
    ///
    /// ## From [`array`] slice
    ///
    /// ```
    ///# use vectora::types::vector::Vector;
    /// let _: Vector<i32, 3> = Vector::try_from(&[1, 2, 3][..]).unwrap();
    /// let _: Vector<f64, 2> = Vector::try_from(&[1.0, 2.0][..]).unwrap();
    /// ```
    ///
    /// ## From [`Vec`] slice
    ///
    /// ```
    ///# use vectora::types::vector::Vector;
    /// let _: Vector<i32, 3> = Vector::try_from(Vec::from([1, 2, 3]).as_slice()).unwrap();
    /// let _: Vector<f64, 2> = Vector::try_from(Vec::from([1.0, 2.0]).as_slice()).unwrap();
    /// ```
    #[inline]
    fn try_from(t_slice: &[T]) -> Result<Vector<T, N>, VectorError> {
        // n.b. if the length of the slice is less than the required number
        // of Vector components, then we raise an error because this does
        // not meet the requirements of the calling code.
        if t_slice.len() != N {
            return Err(VectorError::TryFromSliceError(format!(
                "expected slice with {} items, but received slice with {} items",
                N,
                t_slice.len()
            )));
        }

        match t_slice.try_into() {
            Ok(s) => Ok(Self { components: s }),
            Err(err) => Err(VectorError::TryFromSliceError(format!(
                "failed to cast slice to Vector type: {}",
                err.to_string()
            ))),
        }
    }
}

/// Returns a new [`Vector`] with lossless [`Vector`] scalar component numeric type
/// cast support.
macro_rules! impl_vector_from_vector {
    ($Small: ty, $Large: ty, $doc: expr) => {
        impl<const N: usize> From<Vector<$Small, N>> for Vector<$Large, N> {
            #[doc = $doc]
            #[inline]
            fn from(small: Vector<$Small, N>) -> Vector<$Large, N> {
                let mut new_components: [$Large; N] = [0 as $Large; N];
                let mut i = 0;
                for c in &small.components {
                    new_components[i] = *c as $Large;
                    i += 1;
                }
                Vector { components: new_components }
            }
        }
    };
    ($Small: ty, $Large: ty) => {
        impl_vector_from_vector!(
            $Small,
            $Large,
            concat!(
                "Converts [`",
                stringify!($Small),
                "`] scalar components to [`",
                stringify!($Large),
                "`] losslessly."
            )
        );
    };
}

// Unsigned to Unsigned
impl_vector_from_vector!(u8, u16);
impl_vector_from_vector!(u8, u32);
impl_vector_from_vector!(u8, u64);
impl_vector_from_vector!(u8, u128);
impl_vector_from_vector!(u8, usize);
impl_vector_from_vector!(u16, u32);
impl_vector_from_vector!(u16, u64);
impl_vector_from_vector!(u16, u128);
impl_vector_from_vector!(u32, u64);
impl_vector_from_vector!(u32, u128);
impl_vector_from_vector!(u64, u128);

// Signed to Signed
impl_vector_from_vector!(i8, i16);
impl_vector_from_vector!(i8, i32);
impl_vector_from_vector!(i8, i64);
impl_vector_from_vector!(i8, i128);
impl_vector_from_vector!(i8, isize);
impl_vector_from_vector!(i16, i32);
impl_vector_from_vector!(i16, i64);
impl_vector_from_vector!(i16, i128);
impl_vector_from_vector!(i32, i64);
impl_vector_from_vector!(i32, i128);
impl_vector_from_vector!(i64, i128);

// Unsigned to Signed
impl_vector_from_vector!(u8, i16);
impl_vector_from_vector!(u8, i32);
impl_vector_from_vector!(u8, i64);
impl_vector_from_vector!(u8, i128);
impl_vector_from_vector!(u16, i32);
impl_vector_from_vector!(u16, i64);
impl_vector_from_vector!(u16, i128);
impl_vector_from_vector!(u32, i64);
impl_vector_from_vector!(u32, i128);
impl_vector_from_vector!(u64, i128);

// Signed to Float
impl_vector_from_vector!(i8, f32);
impl_vector_from_vector!(i8, f64);
impl_vector_from_vector!(i16, f32);
impl_vector_from_vector!(i16, f64);
impl_vector_from_vector!(i32, f64);

// Unsigned to Float
impl_vector_from_vector!(u8, f32);
impl_vector_from_vector!(u8, f64);
impl_vector_from_vector!(u16, f32);
impl_vector_from_vector!(u16, f64);
impl_vector_from_vector!(u32, f64);

// ================================
//
// Operator overloads
//
// ================================

// Unary

impl<T, const N: usize> Neg for Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    type Output = Self;

    /// Unary negation operator overload implementation.
    fn neg(self) -> Self::Output {
        let new_components = &mut [T::zero(); N];
        for (i, x) in new_components.iter_mut().enumerate() {
            *x = T::zero() - self[i];
        }
        Self { components: *new_components }
    }
}

// Binary

impl<T, const N: usize> Add for Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    type Output = Self;

    /// Binary add operator overload implemenatation for vector addition.
    fn add(self, rhs: Self) -> Self::Output {
        let new_components = &mut [T::zero(); N];
        for (i, x) in new_components.iter_mut().enumerate() {
            *x = self[i] + rhs[i];
        }
        Self { components: *new_components }
    }
}

impl<T, const N: usize> Sub for Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    type Output = Self;

    /// Binary subtraction operator overload implementation for vector substration.
    fn sub(self, rhs: Self) -> Self::Output {
        let new_components = &mut [T::zero(); N];
        for (i, x) in new_components.iter_mut().enumerate() {
            *x = self[i] - rhs[i];
        }
        Self { components: *new_components }
    }
}

impl<T, const N: usize> Mul<T> for Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    type Output = Self;

    /// Binary multiplication operator overload implementation for vector
    /// scalar multiplication.
    fn mul(self, rhs: T) -> Self::Output {
        let new_components = &mut [T::zero(); N];
        for (i, x) in new_components.iter_mut().enumerate() {
            *x = self[i] * rhs;
        }
        Self { components: *new_components }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use approx::{assert_relative_eq, assert_relative_ne};
    #[allow(unused_imports)]
    use pretty_assertions::{assert_eq, assert_ne};

    // =======================================
    //
    // Instantiation associated function tests
    //
    // =======================================

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
            assert_eq!(v.components.len(), 2);
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
            assert_eq!(v.components.len(), 3);
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
            assert_eq!(v.components.len(), 2);
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
            assert_eq!(v.components.len(), 3);
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
            assert_eq!(v.components.len(), 2);
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
            assert_eq!(v.components.len(), 3);
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
            assert_eq!(v.components.len(), 2);
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
            assert_eq!(v.components.len(), 3);
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
            assert_eq!(v.components.len(), 2);
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
            assert_eq!(v.components.len(), 3);
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
            assert_eq!(v.components.len(), 2);
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
            assert_eq!(v.components.len(), 3);
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
            assert_eq!(v.components.len(), 2);
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
            assert_eq!(v.components.len(), 3);
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
            assert_eq!(v.components.len(), 2);
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
            assert_eq!(v.components.len(), 3);
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
            assert_eq!(v.components.len(), 2);
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
            assert_eq!(v.components.len(), 3);
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
            assert_eq!(v.components.len(), 2);
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
            assert_eq!(v.components.len(), 3);
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
            assert_eq!(v.components.len(), 2);
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
            assert_eq!(v.components.len(), 3);
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
            assert_eq!(v.components.len(), 2);
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
            assert_eq!(v.components.len(), 3);
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
            assert_eq!(v.components.len(), 2);
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
            assert_eq!(v.components.len(), 3);
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
            assert_eq!(v.components.len(), 2);
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
            assert_eq!(v.components.len(), 3);
        }
    }

    #[test]
    fn vector_instantiation_default_u32() {
        // Two dimension
        let v = Vector::<u32, 2>::default();
        assert_eq!(v[0], 0 as u32);
        assert_eq!(v[1], 0 as u32);
        assert_eq!(v.components.len(), 2);

        // Three dimension
        let v = Vector::<u32, 3>::default();
        assert_eq!(v[0], 0 as u32);
        assert_eq!(v[1], 0 as u32);
        assert_eq!(v[2], 0 as u32);
        assert_eq!(v.components.len(), 3);
    }

    #[test]
    fn vector_instantiation_default_f64() {
        // Two dimension
        let v = Vector::<f64, 2>::default();
        assert_relative_eq!(v[0], 0.0 as f64);
        assert_relative_eq!(v[1], 0.0 as f64);
        assert_eq!(v.components.len(), 2);

        // Three dimension
        let v = Vector::<f64, 3>::default();
        assert_relative_eq!(v[0], 0.0 as f64);
        assert_relative_eq!(v[1], 0.0 as f64);
        assert_relative_eq!(v[2], 0.0 as f64);
        assert_eq!(v.components.len(), 3);
    }

    #[test]
    fn vector_instantiation_from_array() {
        // Two dimension
        let v1 = Vector::<u32, 2>::from(&[1, 2]);
        let v2 = Vector::<f64, 2>::from(&[1.0, 2.0]);
        assert_eq!(v1[0], 1);
        assert_eq!(v1[1], 2);
        assert_eq!(v1.components.len(), 2);

        assert_relative_eq!(v2[0], 1.0 as f64);
        assert_relative_eq!(v2[1], 2.0 as f64);
        assert_eq!(v2.components.len(), 2);

        // Three dimension
        let v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);
        assert_eq!(v1[0], 1);
        assert_eq!(v1[1], 2);
        assert_eq!(v1[2], 3);
        assert_eq!(v1.components.len(), 3);

        assert_relative_eq!(v2[0], 1.0 as f64);
        assert_relative_eq!(v2[1], 2.0 as f64);
        assert_relative_eq!(v2[2], 3.0 as f64);
        assert_eq!(v2.components.len(), 3);
    }

    // ================================
    //
    // get method tests
    //
    // ================================
    #[test]
    fn vector_method_get_with_value() {
        let v1 = Vector::<u32, 2>::from(&[1, 2]);
        let v2 = Vector::<f64, 2>::from(&[1.0, 2.0]);
        assert_eq!(v1.get(0).unwrap(), &1);
        assert_eq!(v1.get(1).unwrap(), &2);
        assert_eq!(v1.get(2), None);
        assert_relative_eq!(v2.get(0).unwrap(), &1.0);
        assert_relative_eq!(v2.get(1).unwrap(), &2.0);
        assert_eq!(v2.get(2), None);
    }

    #[test]
    fn vector_method_get_with_range() {
        let v1 = Vector::<u32, 5>::from(&[1, 2, 3, 4, 5]);
        let v2 = Vector::<f64, 5>::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);

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
        let mut v1 = Vector::<u32, 2>::from(&[1, 2]);
        let mut v2 = Vector::<f64, 2>::from(&[1.0, 2.0]);

        let x1 = v1.get_mut(0).unwrap();
        assert_eq!(*x1, 1);
        *x1 = 10;
        assert_eq!(v1.components[0], 10);
        assert_eq!(v1.components[1], 2);

        let x2 = v2.get_mut(0).unwrap();
        assert_relative_eq!(*x2, 1.0);
        *x2 = 10.0;
        assert_relative_eq!(v2.components[0], 10.0);
        assert_relative_eq!(v2.components[1], 2.0);

        let mut v3 = Vector::<u32, 1>::default();
        let mut v4 = Vector::<f64, 1>::default();
        assert_eq!(v3.get_mut(10), None);
        assert_eq!(v4.get_mut(10), None);
    }

    #[test]
    fn vector_method_get_mut_with_range() {
        let mut v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let mut v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);

        let r1 = v1.get_mut(0..2).unwrap();
        assert_eq!(*r1, [1, 2]);
        r1[0] = 5;
        r1[1] = 6;
        assert_eq!(v1.components, [5, 6, 3]);

        let r2 = v2.get_mut(0..2).unwrap();
        assert_eq!(r2.len(), 2);
        assert_relative_eq!(r2[0], 1.0);
        assert_relative_eq!(r2[1], 2.0);
        r2[0] = 5.0;
        r2[1] = 6.0;
        assert_eq!(v2.components.len(), 3);
        assert_relative_eq!(v2.components[0], 5.0);
        assert_relative_eq!(v2.components[1], 6.0);
        assert_relative_eq!(v2.components[2], 3.0);
    }

    // ================================
    //
    // as_* method tests
    //
    // ================================
    #[test]
    fn vector_method_as_slice() {
        let v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);
        let _: &[u32] = v1.as_slice();
        let _: &[f64] = v2.as_slice();
    }

    #[test]
    fn vector_method_as_mut_slice() {
        let mut v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let mut v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);
        let _: &mut [u32] = v1.as_mut_slice();
        let _: &mut [f64] = v2.as_mut_slice();
    }

    #[test]
    fn vector_method_as_array() {
        let v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);
        let _: &[u32; 3] = v1.as_array();
        let _: &[f64; 3] = v2.as_array();
    }

    #[test]
    fn vector_method_as_mut_array() {
        let mut v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let mut v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);
        let _: &mut [u32; 3] = v1.as_mut_array();
        let _: &mut [f64; 3] = v2.as_mut_array();
    }

    // ================================
    //
    // to_* method tests
    //
    // ================================
    #[test]
    fn vector_method_to_array() {
        let v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);
        let _: [u32; 3] = v1.to_array();
        let _: [f64; 3] = v2.to_array();
    }

    #[test]
    fn vector_method_to_vec() {
        let v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);
        let _: Vec<u32> = v1.to_vec();
        let _: Vec<f64> = v2.to_vec();
    }

    // ================================
    //
    // len and is_empty method tests
    //
    // ================================
    #[test]
    fn vector_method_len_is_empty() {
        let v3 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let v_empty = Vector::<u32, 0>::from(&[]);
        assert_eq!(v3.len(), 3);
        assert_eq!(v_empty.len(), 0);
        assert!(v_empty.is_empty());
    }

    // ================================
    //
    // dot method tests
    //
    // ================================
    #[test]
    fn vector_method_dot() {
        let v1: Vector<i32, 3> = Vector::from([1, 3, -5]);
        let v2: Vector<i32, 3> = Vector::from([4, -2, -1]);
        let x1 = v1 * 3;
        let x2 = v2 * 6;
        assert_eq!(v1.dot(&v2), 3);
        assert_eq!(v2.dot(&v1), 3);
        assert_eq!(-v1.dot(&-v2), 3);
        assert_eq!(x1.dot(&x2), (3 * 6) * v1.dot(&v2));

        let v1: Vector<f64, 3> = Vector::from([1.0, 3.0, -5.0]);
        let v2: Vector<f64, 3> = Vector::from([4.0, -2.0, -1.0]);
        let x1 = v1 * 3.0;
        let x2 = v2 * 6.0;
        assert_relative_eq!(v1.dot(&v2), 3.0);
        assert_relative_eq!(v2.dot(&v1), 3.0);
        assert_relative_eq!(-v1.dot(&-v2), 3.0);
        assert_relative_eq!(x1.dot(&x2), (3.0 * 6.0) * v1.dot(&v2));
    }

    // ================================
    //
    // magnitude method tests
    //
    // ================================

    #[test]
    fn vector_method_magnitude_int() {
        let v1: Vector<i32, 2> = Vector::from([2, 2]);
        let v2: Vector<i32, 2> = Vector::from([-2, -2]);
        let v1_f: Vector<f64, 2> = v1.into();
        let v2_f: Vector<f64, 2> = v2.into();
        assert_relative_eq!(v1_f.magnitude(), 2.8284271247461903);
        assert_relative_eq!(v2_f.magnitude(), 2.8284271247461903);
    }

    #[test]
    fn vector_method_magnitude_float() {
        let v1: Vector<f64, 2> = Vector::from([2.8, 2.6]);
        let v2: Vector<f64, 2> = Vector::from([-2.8, -2.6]);

        assert_relative_eq!(v1.magnitude(), 3.82099463490856);
        assert_relative_eq!(v2.magnitude(), 3.82099463490856);
    }

    // ================================
    //
    // normalize method tests
    //
    // ================================

    #[test]
    fn vector_method_normalize() {
        let v1: Vector<f64, 2> = Vector::from([25.123, 30.456]);
        let v2: Vector<f64, 2> = Vector::from([-25.123, -30.456]);
        let mut v3: Vector<f64, 2> = Vector::from([25.123, 30.456]);
        let mut v4: Vector<f64, 2> = Vector::from([-25.123, -30.456]);
        assert_relative_eq!(v1.normalize().magnitude(), 1.0);
        assert_relative_eq!(v2.normalize().magnitude(), 1.0);
        // normalize does not mutate the calling Vector
        assert_relative_eq!(v1[0], 25.123);
        assert_relative_eq!(v1[1], 30.456);

        assert_relative_eq!(v3.mut_normalize().magnitude(), 1.0);
        assert_relative_eq!(v4.mut_normalize().magnitude(), 1.0);
        // mut_normalize does mutate the calling Vector
        assert_relative_eq!(v3[0], 0.6363347262144607);
        assert_relative_eq!(v3[1], 0.7714130645857428);

        assert_eq!((v1.normalize() * v1.magnitude()).magnitude(), v1.magnitude());
    }

    // ================================
    //
    // Index / IndexMut trait tests
    //
    // ================================
    #[test]
    fn vector_trait_index_access() {
        let v1 = Vector::<u32, 2>::from(&[1, 2]);
        let v2 = Vector::<f64, 2>::from(&[1.0, 2.0]);
        assert_eq!(v1[0], 1);
        assert_eq!(v1[1], 2);
        assert_relative_eq!(v2[0], 1.0);
        assert_relative_eq!(v2[1], 2.0);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn vector_trait_index_access_out_of_bounds() {
        let v = Vector::<u32, 10>::new();
        v[10];
    }

    #[test]
    fn vector_trait_index_assignment() {
        let mut v1 = Vector::<u32, 2>::from(&[1, 2]);
        let mut v2 = Vector::<f64, 2>::from(&[1.0, 2.0]);
        v1[0] = 5;
        v1[1] = 6;
        v2[0] = 5.0;
        v2[1] = 6.0;
        assert_eq!(v1[0], 5);
        assert_eq!(v1[1], 6);
        assert_relative_eq!(v2[0], 5.0);
        assert_relative_eq!(v2[1], 6.0);
    }

    // ===================================
    //
    // IntoIterator traits tests
    //
    // ===================================
    #[test]
    fn vector_trait_intoiterator_ref() {
        let v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);

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
        let mut v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let mut v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);

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
        let v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);

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
        let v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);
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

    // ==================================
    //
    // FromIterator trait / collect tests
    //
    // ==================================

    #[test]
    fn vector_trait_fromiterator_collect() {
        let v1: Vector<i32, 3> = [1, 2, 3].into_iter().collect();
        assert_eq!(v1.components.len(), 3);
        assert_eq!(v1[0], 1 as i32);
        assert_eq!(v1[1], 2 as i32);
        assert_eq!(v1[2], 3 as i32);

        // overflow test
        // should truncate at the Vector length for overflows
        let v2: Vector<i32, 2> = [1, 2, 3].into_iter().collect();
        assert_eq!(v2.components.len(), 2);
        assert_eq!(v2[0], 1 as i32);
        assert_eq!(v2[1], 2 as i32);

        // underflow test
        // zero value fills for underflows
        let v3: Vector<i32, 5> = [1, 2, 3].into_iter().collect();
        assert_eq!(v3.components.len(), 5);
        assert_eq!(v3[0], 1 as i32);
        assert_eq!(v3[1], 2 as i32);
        assert_eq!(v3[2], 3 as i32);
        assert_eq!(v3[3], 0 as i32);
        assert_eq!(v3[4], 0 as i32);

        // test with Vector as the iterable
        let v4: Vector<i32, 3> = Vector::from([1, 2, 3]).into_iter().map(|x| x * 2).collect();
        assert_eq!(v4.components.len(), 3);
        assert_eq!(v4[0], 2);
        assert_eq!(v4[1], 4);
        assert_eq!(v4[2], 6);

        // empty iterable test
        let v5: Vector<i32, 3> = [].into_iter().collect();
        assert_eq!(v5.components.len(), 3);
        assert_eq!(v5[0], 0 as i32);
        assert_eq!(v5[1], 0 as i32);
        assert_eq!(v5[2], 0 as i32);
    }

    // ================================
    //
    // PartialEq trait tests
    //
    // ================================

    #[test]
    fn vector_trait_partial_eq_i8() {
        let v1 = Vector::<i8, 3>::from(&[-1, 2, 3]);
        let v2 = Vector::<i8, 3>::from(&[-1, 2, 3]);
        let v_eq = Vector::<i8, 3>::from(&[-1, 2, 3]);
        let v_diff = Vector::<i8, 3>::from(&[-1, 2, 4]);

        let v_zero = Vector::<i8, 3>::zero();
        let v_zero_neg = Vector::<i8, 3>::from(&[-0, -0, -0]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
        assert!(v_zero == v_zero);
        assert!(v_zero == v_zero_neg);
    }

    #[test]
    fn vector_trait_partial_eq_i16() {
        let v1 = Vector::<i16, 3>::from(&[-1, 2, 3]);
        let v2 = Vector::<i16, 3>::from(&[-1, 2, 3]);
        let v_eq = Vector::<i16, 3>::from(&[-1, 2, 3]);
        let v_diff = Vector::<i16, 3>::from(&[-1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_i32() {
        let v1 = Vector::<i32, 3>::from(&[-1, 2, 3]);
        let v2 = Vector::<i32, 3>::from(&[-1, 2, 3]);
        let v_eq = Vector::<i32, 3>::from(&[-1, 2, 3]);
        let v_diff = Vector::<i32, 3>::from(&[-1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_i64() {
        let v1 = Vector::<i64, 3>::from(&[-1, 2, 3]);
        let v2 = Vector::<i64, 3>::from(&[-1, 2, 3]);
        let v_eq = Vector::<i64, 3>::from(&[-1, 2, 3]);
        let v_diff = Vector::<i64, 3>::from(&[-1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_i128() {
        let v1 = Vector::<i128, 3>::from(&[-1, 2, 3]);
        let v2 = Vector::<i128, 3>::from(&[-1, 2, 3]);
        let v_eq = Vector::<i128, 3>::from(&[-1, 2, 3]);
        let v_diff = Vector::<i128, 3>::from(&[-1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_u8() {
        let v1 = Vector::<u8, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<u8, 3>::from(&[1, 2, 3]);
        let v_eq = Vector::<u8, 3>::from(&[1, 2, 3]);
        let v_diff = Vector::<u8, 3>::from(&[1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_u16() {
        let v1 = Vector::<u16, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<u16, 3>::from(&[1, 2, 3]);
        let v_eq = Vector::<u16, 3>::from(&[1, 2, 3]);
        let v_diff = Vector::<u16, 3>::from(&[1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_u32() {
        let v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let v_eq = Vector::<u32, 3>::from(&[1, 2, 3]);
        let v_diff = Vector::<u32, 3>::from(&[1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_u64() {
        let v1 = Vector::<u64, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<u64, 3>::from(&[1, 2, 3]);
        let v_eq = Vector::<u64, 3>::from(&[1, 2, 3]);
        let v_diff = Vector::<u64, 3>::from(&[1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_u128() {
        let v1 = Vector::<u128, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<u128, 3>::from(&[1, 2, 3]);
        let v_eq = Vector::<u128, 3>::from(&[1, 2, 3]);
        let v_diff = Vector::<u128, 3>::from(&[1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_isize() {
        let v1 = Vector::<isize, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<isize, 3>::from(&[1, 2, 3]);
        let v_eq = Vector::<isize, 3>::from(&[1, 2, 3]);
        let v_diff = Vector::<isize, 3>::from(&[1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_usize() {
        let v1 = Vector::<usize, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<usize, 3>::from(&[1, 2, 3]);
        let v_eq = Vector::<usize, 3>::from(&[1, 2, 3]);
        let v_diff = Vector::<usize, 3>::from(&[1, 2, 4]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_f32() {
        let v1 = Vector::<f32, 3>::from(&[-1.1, 2.2, 3.3]);
        let v2 = Vector::<f32, 3>::from(&[-1.1, 2.2, 3.3]);
        let v_eq = Vector::<f32, 3>::from(&[-1.1, 2.2, 3.3]);
        let v_diff = Vector::<f32, 3>::from(&[-1.1, 2.2, 4.4]);
        let v_close = Vector::<f32, 3>::from(&[-1.1 + (f32::EPSILON * 2.), 2.2, 3.3]);

        let v_zero = Vector::<f32, 3>::zero();
        let v_zero_eq = Vector::<f32, 3>::zero();
        let v_zero_neg_eq = Vector::<f32, 3>::from(&[-0.0, -0.0, -0.0]);

        let v_nan = Vector::<f32, 3>::from(&[f32::NAN, 0.0, 0.0]);
        let v_nan_diff = Vector::<f32, 3>::from(&[f32::NAN, 0.0, 0.0]);

        let v_inf_pos = Vector::<f32, 3>::from(&[f32::INFINITY, 0.0, 0.0]);
        let v_inf_pos_eq = Vector::<f32, 3>::from(&[f32::INFINITY, 0.0, 0.0]);
        let v_inf_neg = Vector::<f32, 3>::from(&[f32::NEG_INFINITY, 0.0, 0.0]);
        let v_inf_neg_eq = Vector::<f32, 3>::from(&[f32::NEG_INFINITY, 0.0, 0.0]);

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
        let v1 = Vector::<f64, 3>::from(&[-1.1, 2.2, 3.3]);
        let v2 = Vector::<f64, 3>::from(&[-1.1, 2.2, 3.3]);
        let v_eq = Vector::<f64, 3>::from(&[-1.1, 2.2, 3.3]);
        let v_diff = Vector::<f64, 3>::from(&[-1.1, 2.2, 4.4]);
        let v_close = Vector::<f64, 3>::from(&[-1.1 + (f64::EPSILON * 2.), 2.2, 3.3]);

        let v_zero = Vector::<f64, 3>::zero();
        let v_zero_eq = Vector::<f64, 3>::zero();
        let v_zero_neg_eq = Vector::<f64, 3>::from(&[-0.0, -0.0, -0.0]);

        let v_nan = Vector::<f64, 3>::from(&[f64::NAN, 0.0, 0.0]);
        let v_nan_diff = Vector::<f64, 3>::from(&[f64::NAN, 0.0, 0.0]);

        let v_inf_pos = Vector::<f64, 3>::from(&[f64::INFINITY, 0.0, 0.0]);
        let v_inf_pos_eq = Vector::<f64, 3>::from(&[f64::INFINITY, 0.0, 0.0]);
        let v_inf_neg = Vector::<f64, 3>::from(&[f64::NEG_INFINITY, 0.0, 0.0]);
        let v_inf_neg_eq = Vector::<f64, 3>::from(&[f64::NEG_INFINITY, 0.0, 0.0]);

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

    // ================================
    //
    // AsRef / AsMutRef trait tests
    //
    // ================================
    #[test]
    fn vector_trait_as_ref() {
        let v = Vector::<u32, 3>::from(&[1, 2, 3]);
        let test_vector: &Vector<u32, 3> = v.as_ref();
        let test_slice: &[u32] = v.as_ref();

        assert_eq!(test_vector[0], 1);
        assert_eq!(test_vector[1], 2);
        assert_eq!(test_vector[2], 3);

        assert_eq!(test_slice[0], 1);
        assert_eq!(test_slice[1], 2);
        assert_eq!(test_slice[2], 3);
    }

    #[test]
    fn vector_trait_as_mut() {
        let mut v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let mut v2 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let test_vector: &mut Vector<u32, 3> = v1.as_mut();
        let test_slice: &mut [u32] = v2.as_mut();

        test_vector[0] = 10;
        test_slice[0] = 10;

        assert_eq!(test_vector.components.len(), 3);
        assert_eq!(test_vector[0], 10);
        assert_eq!(test_vector[1], 2);
        assert_eq!(test_vector[2], 3);
    }

    // ================================
    //
    // Borrow / BorrowMut trait tests
    //
    // ================================
    #[test]
    fn vector_trait_borrow() {
        let v = Vector::<u32, 3>::from(&[1, 2, 3]);
        let test_slice: &[u32] = v.borrow();

        assert_eq!(test_slice, [1, 2, 3]);
    }

    #[test]
    fn vector_trait_borrow_mut() {
        let mut v = Vector::<u32, 3>::from(&[1, 2, 3]);
        let test_slice: &mut [u32] = v.borrow_mut();

        test_slice[0] = 10;

        assert_eq!(test_slice, [10, 2, 3]);
    }

    // ================================
    //
    // Deref / DerefMut trait tests
    //
    // ================================
    #[test]
    fn vector_trait_deref() {
        let v = Vector::<u32, 3>::from(&[1, 2, 3]);
        let test_slice: &[u32] = &v;

        assert_eq!(test_slice, [1, 2, 3]);
    }

    #[test]
    fn vector_trait_deref_mut() {
        let mut v = Vector::<u32, 3>::from(&[1, 2, 3]);
        let test_slice: &mut [u32] = &mut v;

        assert_eq!(test_slice, [1, 2, 3]);
    }

    // ================================
    //
    // From trait tests
    //
    // ================================
    #[test]
    fn vector_trait_from_into_uint_to_uint() {
        let v_u8 = Vector::<u8, 3>::from(&[1, 2, 3]);
        let v_u16 = Vector::<u16, 3>::new();
        let v_u32 = Vector::<u32, 3>::new();
        let v_u64 = Vector::<u64, 3>::new();

        let v_new_16: Vector<u16, 3> = Vector::<u16, 3>::from(v_u8);
        let _: Vector<u16, 3> = v_u8.into();
        assert_eq!(v_new_16.components.len(), 3);
        assert_eq!(v_new_16[0], 1 as u16);
        assert_eq!(v_new_16[1], 2 as u16);
        assert_eq!(v_new_16[2], 3 as u16);

        let _: Vector<u32, 3> = Vector::<u32, 3>::from(v_u8);
        let _: Vector<u32, 3> = v_u8.into();

        let _: Vector<u64, 3> = Vector::<u64, 3>::from(v_u8);
        let _: Vector<u64, 3> = v_u8.into();

        let _: Vector<u128, 3> = Vector::<u128, 3>::from(v_u8);
        let _: Vector<u128, 3> = v_u8.into();

        let _: Vector<u32, 3> = Vector::<u32, 3>::from(v_u16);
        let _: Vector<u32, 3> = v_u16.into();

        let _: Vector<u64, 3> = Vector::<u64, 3>::from(v_u16);
        let _: Vector<u64, 3> = v_u16.into();

        let _: Vector<u128, 3> = Vector::<u128, 3>::from(v_u16);
        let _: Vector<u128, 3> = v_u16.into();

        let _: Vector<u64, 3> = Vector::<u64, 3>::from(v_u32);
        let _: Vector<u64, 3> = v_u32.into();

        let _: Vector<u128, 3> = Vector::<u128, 3>::from(v_u32);
        let _: Vector<u128, 3> = v_u32.into();

        let _: Vector<u128, 3> = Vector::<u128, 3>::from(v_u64);
        let _: Vector<u128, 3> = v_u64.into();
    }

    #[test]
    fn vector_trait_from_into_iint_to_iint() {
        let v_i8 = Vector::<i8, 3>::from(&[1, 2, 3]);
        let v_i16 = Vector::<i16, 3>::new();
        let v_i32 = Vector::<i32, 3>::new();
        let v_i64 = Vector::<i64, 3>::new();

        let v_new_16: Vector<i16, 3> = Vector::<i16, 3>::from(v_i8);
        let _: Vector<i16, 3> = v_i8.into();
        assert_eq!(v_new_16.components.len(), 3);
        assert_eq!(v_new_16[0], 1 as i16);
        assert_eq!(v_new_16[1], 2 as i16);
        assert_eq!(v_new_16[2], 3 as i16);

        let _: Vector<i32, 3> = Vector::<i32, 3>::from(v_i8);
        let _: Vector<i32, 3> = v_i8.into();

        let _: Vector<i64, 3> = Vector::<i64, 3>::from(v_i8);
        let _: Vector<i64, 3> = v_i8.into();

        let _: Vector<i128, 3> = Vector::<i128, 3>::from(v_i8);
        let _: Vector<i128, 3> = v_i8.into();

        let _: Vector<i32, 3> = Vector::<i32, 3>::from(v_i16);
        let _: Vector<i32, 3> = v_i16.into();

        let _: Vector<i64, 3> = Vector::<i64, 3>::from(v_i16);
        let _: Vector<i64, 3> = v_i16.into();

        let _: Vector<i128, 3> = Vector::<i128, 3>::from(v_i16);
        let _: Vector<i128, 3> = v_i16.into();

        let _: Vector<i64, 3> = Vector::<i64, 3>::from(v_i32);
        let _: Vector<i64, 3> = v_i32.into();

        let _: Vector<i128, 3> = Vector::<i128, 3>::from(v_i32);
        let _: Vector<i128, 3> = v_i32.into();

        let _: Vector<i128, 3> = Vector::<i128, 3>::from(v_i64);
        let _: Vector<i128, 3> = v_i64.into();
    }

    #[test]
    fn vector_trait_from_into_uint_to_iint() {
        let v_u8 = Vector::<u8, 3>::from(&[1, 2, 3]);
        let v_u16 = Vector::<u16, 3>::new();
        let v_u32 = Vector::<u32, 3>::new();
        let v_u64 = Vector::<u64, 3>::new();

        let v_new_16: Vector<i16, 3> = Vector::<i16, 3>::from(v_u8);
        let _: Vector<i16, 3> = v_u8.into();
        assert_eq!(v_new_16.components.len(), 3);
        assert_eq!(v_new_16[0], 1 as i16);
        assert_eq!(v_new_16[1], 2 as i16);
        assert_eq!(v_new_16[2], 3 as i16);

        let _: Vector<i32, 3> = Vector::<i32, 3>::from(v_u8);
        let _: Vector<i32, 3> = v_u8.into();

        let _: Vector<i64, 3> = Vector::<i64, 3>::from(v_u8);
        let _: Vector<i64, 3> = v_u8.into();

        let _: Vector<i64, 3> = Vector::<i64, 3>::from(v_u8);
        let _: Vector<i64, 3> = v_u8.into();

        let _: Vector<i128, 3> = Vector::<i128, 3>::from(v_u8);
        let _: Vector<i128, 3> = v_u8.into();

        let _: Vector<i32, 3> = Vector::<i32, 3>::from(v_u16);
        let _: Vector<i32, 3> = v_u16.into();

        let _: Vector<i64, 3> = Vector::<i64, 3>::from(v_u16);
        let _: Vector<i64, 3> = v_u16.into();

        let _: Vector<i128, 3> = Vector::<i128, 3>::from(v_u16);
        let _: Vector<i128, 3> = v_u16.into();

        let _: Vector<i64, 3> = Vector::<i64, 3>::from(v_u32);
        let _: Vector<i64, 3> = v_u32.into();

        let _: Vector<i128, 3> = Vector::<i128, 3>::from(v_u32);
        let _: Vector<i128, 3> = v_u32.into();

        let _: Vector<i128, 3> = Vector::<i128, 3>::from(v_u64);
        let _: Vector<i128, 3> = v_u64.into();
    }

    #[test]
    fn vector_trait_from_into_uint_to_float() {
        let v_u8 = Vector::<u8, 3>::from(&[1, 2, 3]);
        let v_u16 = Vector::<u16, 3>::new();
        let v_u32 = Vector::<u32, 3>::new();

        let v_new_32: Vector<f32, 3> = Vector::<f32, 3>::from(v_u8);
        let _: Vector<f32, 3> = v_u8.into();
        assert_eq!(v_new_32.components.len(), 3);
        assert_relative_eq!(v_new_32[0], 1.0 as f32);
        assert_relative_eq!(v_new_32[1], 2.0 as f32);
        assert_relative_eq!(v_new_32[2], 3.0 as f32);

        let _: Vector<f64, 3> = Vector::<f64, 3>::from(v_u8);
        let _: Vector<f64, 3> = v_u8.into();

        let _: Vector<f32, 3> = Vector::<f32, 3>::from(v_u16);
        let _: Vector<f32, 3> = v_u16.into();

        let _: Vector<f64, 3> = Vector::<f64, 3>::from(v_u16);
        let _: Vector<f64, 3> = v_u16.into();

        let _: Vector<f64, 3> = Vector::<f64, 3>::from(v_u32);
        let _: Vector<f64, 3> = v_u32.into();
    }

    #[test]
    fn vector_trait_from_into_iint_to_float() {
        let v_i8 = Vector::<i8, 3>::from(&[1, 2, 3]);
        let v_i16 = Vector::<i16, 3>::new();
        let v_i32 = Vector::<i32, 3>::new();

        let v_new_32: Vector<f32, 3> = Vector::<f32, 3>::from(v_i8);
        let _: Vector<f32, 3> = v_i8.into();
        assert_eq!(v_new_32.components.len(), 3);
        assert_relative_eq!(v_new_32[0], 1.0 as f32);
        assert_relative_eq!(v_new_32[1], 2.0 as f32);
        assert_relative_eq!(v_new_32[2], 3.0 as f32);

        let _: Vector<f64, 3> = Vector::<f64, 3>::from(v_i8);
        let _: Vector<f64, 3> = v_i8.into();

        let _: Vector<f32, 3> = Vector::<f32, 3>::from(v_i16);
        let _: Vector<f32, 3> = v_i16.into();

        let _: Vector<f64, 3> = Vector::<f64, 3>::from(v_i16);
        let _: Vector<f64, 3> = v_i16.into();

        let _: Vector<f64, 3> = Vector::<f64, 3>::from(v_i32);
        let _: Vector<f64, 3> = v_i32.into();
    }

    #[test]
    fn vector_trait_from_into_array_to_vector() {
        let a = [1, 2, 3];
        let a_slice = &a[..];
        // from / into with array
        let va = Vector::<i32, 3>::from(a);
        let va2: Vector<i32, 3> = Vector::from(&a);
        let va3: Vector<i32, 3> = a.into();
        assert_eq!(va.components.len(), 3);
        assert_eq!(va[0], 1 as i32);
        assert_eq!(va[1], 2 as i32);
        assert_eq!(va[2], 3 as i32);
        assert_eq!(va, va2);
        assert_eq!(va, va3);
        // from / into with array slice
        let vas: Vector<i32, 3> = Vector::<i32, 3>::try_from(a_slice).unwrap();
        let vas2: Vector<i32, 3> = Vector::try_from(&a[..]).unwrap();
        let vas3: Vector<i32, 3> = a_slice.try_into().unwrap();
        assert_eq!(vas.components.len(), 3);
        assert_eq!(vas[0], 1 as i32);
        assert_eq!(vas[1], 2 as i32);
        assert_eq!(vas[2], 3 as i32);
        assert_eq!(vas, vas2);
        assert_eq!(vas, vas3);
    }

    #[test]
    fn vector_trait_from_into_vec_to_vector() {
        let v = Vec::from([1, 2, 3]);
        let v_slice = &v[..];
        let v_to_owned = Vec::from([1, 2, 3]);
        // from / into with Vec
        let vv = Vector::<i32, 3>::try_from(&v).unwrap();
        let vv2: Vector<i32, 3> = (&v).try_into().unwrap();
        let vv3: Vector<i32, 3> = Vector::<i32, 3>::try_from(v_to_owned).unwrap();
        assert_eq!(vv.components.len(), 3);
        assert_eq!(vv[0], 1 as i32);
        assert_eq!(vv[1], 2 as i32);
        assert_eq!(vv[2], 3 as i32);
        assert_eq!(vv, vv2);
        assert_eq!(vv, vv3);
        // from / into with Vector slice
        let vvs: Vector<i32, 3> = Vector::<i32, 3>::try_from(v_slice).unwrap();
        let vvs2: Vector<i32, 3> = Vector::try_from(v.as_slice()).unwrap();
        let vvs3: Vector<i32, 3> = v_slice.try_into().unwrap();
        assert_eq!(vvs.components.len(), 3);
        assert_eq!(vvs[0], 1 as i32);
        assert_eq!(vvs[1], 2 as i32);
        assert_eq!(vvs[2], 3 as i32);
        assert_eq!(vvs, vvs2);
        assert_eq!(vvs, vvs3);
    }

    #[test]
    fn vector_trait_from_into_slice_err() {
        let v = Vec::from([1, 2, 3]);
        let v_slice = &v[..];
        let e1 = Vector::<i32, 2>::try_from(v_slice);
        let e2 = Vector::<i32, 4>::try_from(v_slice);
        assert!(matches!(e1, Err(VectorError::TryFromSliceError(_))));
        assert!(matches!(e2, Err(VectorError::TryFromSliceError(_))));
    }

    // ================================
    //
    // Operator overloads
    //
    // ================================

    #[test]
    fn vector_trait_neg_unary() {
        let v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let v2: Vector<i32, 3> = Vector::from([-1, -2, -3]);
        let v3: Vector<f64, 3> = Vector::from([1.0, 2.0, 3.0]);
        let v4: Vector<f64, 3> = Vector::from([-1.0, -2.0, -3.0]);

        assert_eq!(-v1, -v1);
        assert_eq!(-v1, v2);
        assert_eq!(-v2, v1);
        assert_eq!(-v3, -v3);
        assert_eq!(-v3, v4);
        assert_eq!(-v4, v3);
    }

    #[test]
    fn vector_trait_add() {
        let v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let v2: Vector<i32, 3> = Vector::from([4, 5, 6]);
        let v3: Vector<i32, 3> = Vector::from([-2, -3, -4]);
        let v_zero: Vector<i32, 3> = Vector::zero();

        assert_eq!(v1 + v2, Vector::<i32, 3>::from([5, 7, 9]));
        assert_eq!(v2 + v3, Vector::<i32, 3>::from([2, 2, 2]));
        assert_eq!(v1 + v2 + v3, Vector::<i32, 3>::from([3, 4, 5]));
        assert_eq!(v1 + v_zero, v1);
        assert_eq!(v_zero + v1, v1);
        assert_eq!(v1 + v2, v2 + v1);
        assert_eq!((v1 + v2) + v3, v1 + (v2 + v3));
        assert_eq!(v1 + (-v1), v_zero);
        assert_eq!((v1 + v2) * 10, (v1 * 10) + (v2 * 10));

        let v1: Vector<f64, 3> = Vector::from([1.0, 2.0, 3.0]);
        let v2: Vector<f64, 3> = Vector::from([4.0, 5.0, 6.0]);
        let v3: Vector<f64, 3> = Vector::from([-2.0, -3.0, -4.0]);
        let v_zero: Vector<f64, 3> = Vector::zero();

        assert_eq!(v1 + v2, Vector::<f64, 3>::from([5.0, 7.0, 9.0]));
        assert_eq!(v2 + v3, Vector::<f64, 3>::from([2.0, 2.0, 2.0]));
        assert_eq!(v1 + v2 + v3, Vector::<f64, 3>::from([3.0, 4.0, 5.0]));
        assert_eq!(v1 + v_zero, v1);
        assert_eq!(v_zero + v1, v1);
        assert_eq!(-v_zero + v1, v1);
        assert_eq!(v1 + v2, v2 + v1);
        assert_eq!((v1 + v2) + v3, v1 + (v2 + v3));
        assert_eq!(v1 + (-v1), v_zero);
        assert_eq!((v1 + v2) * 10.0, (v1 * 10.0) + (v2 * 10.0));
    }

    #[test]
    fn vector_method_mut_add() {
        let mut v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let mut v2: Vector<i32, 3> = Vector::from([4, 5, 6]);
        let v3: Vector<i32, 3> = Vector::from([-2, -3, -4]);

        v1.mut_add(&v2);
        v2.mut_add(&v3);

        assert_eq!(v1, Vector::<i32, 3>::from([5, 7, 9]));
        assert_eq!(v2, Vector::<i32, 3>::from([2, 2, 2]));

        // reset
        let mut v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let v2: Vector<i32, 3> = Vector::from([4, 5, 6]);
        let v3: Vector<i32, 3> = Vector::from([-2, -3, -4]);

        v1.mut_add(&v2).mut_add(&v3);
        assert_eq!(v1, Vector::<i32, 3>::from([3, 4, 5]));

        // reset
        let mut v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let v_zero: Vector<i32, 3> = Vector::zero();

        v1.mut_add(&v_zero);

        assert_eq!(v1, Vector::<i32, 3>::from([1, 2, 3]));

        // reset
        let v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let mut v_zero: Vector<i32, 3> = Vector::zero();

        v_zero.mut_add(&v1);

        assert_eq!(v_zero, v1);

        // reset
        let mut v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let v2: Vector<i32, 3> = Vector::from([4, 5, 6]);
        let v3: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let mut v4: Vector<i32, 3> = Vector::from([4, 5, 6]);

        v1.mut_add(&v2);
        v4.mut_add(&v3);

        assert_eq!(v1, v4);

        // reset
        let mut v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let v2: Vector<i32, 3> = Vector::from([4, 5, 6]);
        let v3: Vector<i32, 3> = Vector::from([7, 8, 9]);
        let v4: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let v5: Vector<i32, 3> = Vector::from([4, 5, 6]);
        let mut v6: Vector<i32, 3> = Vector::from([7, 8, 9]);

        assert_eq!(v1.mut_add(&v2).mut_add(&v3), v6.mut_add(&v5).mut_add(&v4));
        assert_eq!(v1, Vector::from([12, 15, 18]));
        assert_eq!(v6, Vector::from([12, 15, 18]));

        // reset
        let mut v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
        v1.mut_add(&v1.neg());
        assert_eq!(v1, Vector::<i32, 3>::zero());

        // reset
        let mut v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let v2: Vector<i32, 3> = Vector::from([4, 5, 6]);
        let v3: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let v4: Vector<i32, 3> = Vector::from([4, 5, 6]);
        v1.mut_add(&v2);
        assert_eq!(v1 * 10, *(v3 * 10).mut_add(&(v4 * 10)));
    }

    #[test]
    #[should_panic(expected = "attempt to add with overflow")]
    fn vector_trait_add_panics_on_overflow() {
        let v1: Vector<u8, 3> = Vector::from([u8::MAX, 2, 3]);
        let v2: Vector<u8, 3> = Vector::from([1, 1, 1]);
        let _ = v1 + v2;
    }

    #[test]
    #[should_panic(expected = "attempt to add with overflow")]
    fn vector_method_mut_add_panics_on_overflow() {
        let mut v1: Vector<u8, 3> = Vector::from([u8::MAX, 2, 3]);
        let v2: Vector<u8, 3> = Vector::from([1, 1, 1]);
        v1.mut_add(&v2);
    }

    #[test]
    fn vector_trait_sub() {
        let v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let v2: Vector<i32, 3> = Vector::from([4, 5, 6]);
        let v3: Vector<i32, 3> = Vector::from([-2, -3, -4]);
        let v_zero: Vector<i32, 3> = Vector::zero();

        assert_eq!(v1 - v2, Vector::<i32, 3>::from([-3, -3, -3]));
        assert_eq!(v2 - v1, Vector::<i32, 3>::from([3, 3, 3]));
        assert_eq!(v1 - v2 - v3, Vector::<i32, 3>::from([-1, 0, 1]));
        assert_eq!((v1 - v2) - v3, v1 - v2 - v3);
        assert_eq!(v1 - (v2 - v3), Vector::<i32, 3>::from([-5, -6, -7]));
        assert_eq!(v1 - v_zero, v1);
        assert_eq!(v1 - (-v_zero), v1);
        assert_eq!(v_zero - v1, -v1);
        assert_eq!(-v_zero - v1, -v1);

        let v1: Vector<f64, 3> = Vector::from([1.0, 2.0, 3.0]);
        let v2: Vector<f64, 3> = Vector::from([4.0, 5.0, 6.0]);
        let v3: Vector<f64, 3> = Vector::from([-2.0, -3.0, -4.0]);
        let v_zero: Vector<f64, 3> = Vector::zero();

        assert_eq!(v1 - v2, Vector::<f64, 3>::from([-3.0, -3.0, -3.0]));
        assert_eq!(v2 - v1, Vector::<f64, 3>::from([3.0, 3.0, 3.0]));
        assert_eq!(v1 - v2 - v3, Vector::<f64, 3>::from([-1.0, 0.0, 1.0]));
        assert_eq!((v1 - v2) - v3, v1 - v2 - v3);
        assert_eq!(v1 - (v2 - v3), Vector::<f64, 3>::from([-5.0, -6.0, -7.0]));
        assert_eq!(v1 - v_zero, v1);
        assert_eq!(v1 - (-v_zero), v1);
        assert_eq!(v_zero - v1, -v1);
        assert_eq!(-v_zero - v1, -v1);
    }

    #[test]
    fn vector_method_mut_sub() {
        let mut v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let v2: Vector<i32, 3> = Vector::from([4, 5, 6]);

        v1.mut_sub(&v2);

        assert_eq!(v1, Vector::<i32, 3>::from([-3, -3, -3]));

        // reset
        let v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let mut v2: Vector<i32, 3> = Vector::from([4, 5, 6]);

        v2.mut_sub(&v1);

        assert_eq!(v2, Vector::<i32, 3>::from([3, 3, 3]));

        // reset
        let mut v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let v2: Vector<i32, 3> = Vector::from([4, 5, 6]);
        let v3: Vector<i32, 3> = Vector::from([-2, -3, -4]);

        v1.mut_sub(&v2).mut_sub(&v3);

        assert_eq!(v1, Vector::<i32, 3>::from([-1, 0, 1]));

        // reset
        let mut v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let v_zero: Vector<i32, 3> = Vector::zero();

        v1.mut_sub(&v_zero);

        assert_eq!(v1, Vector::<i32, 3>::from([1, 2, 3]));

        // reset
        let mut v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let v_zero: Vector<i32, 3> = Vector::zero();

        v1.mut_sub(&v_zero.neg());

        assert_eq!(v1, Vector::<i32, 3>::from([1, 2, 3]));

        // reset
        let v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
        let mut v_zero: Vector<i32, 3> = Vector::zero();

        v_zero.mut_sub(&v1);

        assert_eq!(v_zero, Vector::<i32, 3>::from([-1, -2, -3]));

        // reset
        let v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
        // note unary negation of the vector definition
        let mut v_zero: Vector<i32, 3> = -Vector::zero();

        v_zero.mut_sub(&v1);

        assert_eq!(v_zero, Vector::<i32, 3>::from([-1, -2, -3]));
    }

    #[test]
    #[should_panic(expected = "attempt to subtract with overflow")]
    fn vector_trait_sub_panics_on_overflow() {
        let v1: Vector<u32, 3> = Vector::from([1, 2, 3]);
        let v2: Vector<u32, 3> = Vector::from([4, 5, 6]);
        let _ = v1 - v2;
    }

    #[test]
    #[should_panic(expected = "attempt to subtract with overflow")]
    fn vector_method_mut_sub_panics_on_overflow() {
        let mut v1: Vector<u32, 3> = Vector::from([1, 2, 3]);
        let v2: Vector<u32, 3> = Vector::from([4, 5, 6]);
        let _ = v1.mut_sub(&v2);
    }

    #[test]
    fn vector_trait_mul() {
        let v1: Vector<i32, 3> = Vector::from([1, 1, 1]);
        assert_eq!(v1 * 10, Vector::from([10, 10, 10]));
        assert_eq!(v1 * -10, Vector::from([-10, -10, -10]));
        assert_eq!(v1 * 0, Vector::from([0, 0, 0]));
        assert_eq!(v1 * -0, Vector::from([0, 0, 0]));
        assert_eq!(v1 * 10 * 5, Vector::from([50, 50, 50]));
        assert_eq!(v1 * 10 * -5, Vector::from([-50, -50, -50]));
        assert_eq!((v1 * 10) * 5, v1 * (10 * 5));
        assert_eq!(v1 * 1, v1);
        assert_eq!(v1 * -1, -v1);
        assert_eq!(v1 * 10 + v1 * 5, v1 * (10 + 5));

        let v1: Vector<f64, 3> = Vector::from([1.0, 1.0, 1.0]);
        assert_eq!(v1 * 10.0, Vector::from([10.0, 10.0, 10.0]));
        assert_eq!(v1 * -10.0, Vector::from([-10.0, -10.0, -10.0]));
        assert_eq!(v1 * 0.0, Vector::from([0.0, 0.0, 0.0]));
        assert_eq!(v1 * -0.0, Vector::from([0.0, 0.0, 0.0]));
        assert_eq!(v1 * 10.0 * 5.0, Vector::from([50.0, 50.0, 50.0]));
        assert_eq!(v1 * 10.0 * -5.0, Vector::from([-50.0, -50.0, -50.0]));
        assert_eq!((v1 * 10.0) * 5.0, v1 * (10.0 * 5.0));
        assert_eq!(v1 * 1.0, v1);
        assert_eq!(v1 * -1.0, -v1);
        assert_eq!(v1 * 10.0 + v1 * 5.0, v1 * (10.0 + 5.0));
    }

    #[test]
    fn vector_method_mut_mul() {
        let mut v1: Vector<i32, 3> = Vector::from([1, 1, 1]);
        v1.mut_mul(10);
        assert_eq!(v1, Vector::from([10, 10, 10]));

        // reset
        let mut v1: Vector<i32, 3> = Vector::from([1, 1, 1]);
        v1.mut_mul(-10);
        assert_eq!(v1, Vector::from([-10, -10, -10]));

        // reset
        let mut v1: Vector<i32, 3> = Vector::from([1, 1, 1]);
        v1.mut_mul(0);
        assert_eq!(v1, Vector::zero());

        // reset
        let mut v1: Vector<i32, 3> = Vector::from([1, 1, 1]);
        v1.mut_mul(-0);
        assert_eq!(v1, Vector::zero());

        // reset
        let mut v1: Vector<i32, 3> = Vector::from([1, 1, 1]);
        v1.mut_mul(10).mut_mul(5);
        assert_eq!(v1, Vector::from([50, 50, 50]));

        // reset
        let mut v1: Vector<i32, 3> = Vector::from([1, 1, 1]);
        v1.mut_mul(10).mut_mul(-5);
        assert_eq!(v1, Vector::from([-50, -50, -50]));

        // reset
        let mut v1: Vector<i32, 3> = Vector::from([1, 1, 1]);
        let mut v2: Vector<i32, 3> = Vector::from([1, 1, 1]);
        v1.mut_mul(10).mut_mul(5);
        v2.mut_mul(10 * 5);
        assert_eq!(v1, v2);

        // reset
        let mut v1: Vector<i32, 3> = Vector::from([1, 1, 1]);
        let v2: Vector<i32, 3> = Vector::from([1, 1, 1]);

        v1.mut_mul(1);

        assert_eq!(v1, v2);

        v1.mut_mul(-1);

        assert_eq!(v1, -v2);
    }

    #[test]
    #[should_panic(expected = "attempt to multiply with overflow")]
    fn vector_trait_mul_overflow_panic() {
        let v1: Vector<u8, 2> = Vector::from([2, 2]);
        let _ = v1 * u8::MAX;
    }

    #[test]
    #[should_panic(expected = "attempt to multiply with overflow")]
    fn vector_method_mut_mul_overflow_panic() {
        let mut v1: Vector<u8, 2> = Vector::from([2, 2]);
        let _ = v1.mut_mul(u8::MAX);
    }

    #[test]
    fn vector_multi_overloaded_operator_precedence() {
        let v1: Vector<i32, 3> = Vector::from([1, 1, 1]);
        let v2: Vector<i32, 3> = Vector::from([-2, -2, -2]);
        let v_zero: Vector<i32, 3> = Vector::zero();
        assert_eq!(v1 + -v2 * 10, Vector::<i32, 3>::from([21, 21, 21]));
        assert_eq!((v1 + -v2) * 10, Vector::<i32, 3>::from([30, 30, 30]));
        assert_eq!(v1 - -v2 * 10, Vector::<i32, 3>::from([-19, -19, -19]));
        assert_eq!((v1 - -v2) * 10, Vector::<i32, 3>::from([-10, -10, -10]));
        assert_eq!(v1 + v2 * 0, v1);
        assert_eq!((v1 + v2) * 0, v_zero);

        let v1: Vector<f64, 3> = Vector::from([1.0, 1.0, 1.0]);
        let v2: Vector<f64, 3> = Vector::from([-2.0, -2.0, -2.0]);
        let v_zero: Vector<f64, 3> = Vector::zero();
        assert_eq!(v1 + -v2 * 10.0, Vector::<f64, 3>::from([21.0, 21.0, 21.0]));
        assert_eq!((v1 + -v2) * 10.0, Vector::<f64, 3>::from([30.0, 30.0, 30.0]));
        assert_eq!(v1 - -v2 * 10.0, Vector::<f64, 3>::from([-19.0, -19.0, -19.0]));
        assert_eq!((v1 - -v2) * 10.0, Vector::<f64, 3>::from([-10.0, -10.0, -10.0]));
        assert_eq!(v1 + v2 * 0.0, v1);
        assert_eq!((v1 + v2) * 0.0, v_zero);
    }
}
