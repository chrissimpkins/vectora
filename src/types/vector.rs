//! Vector types.

use std::{
    borrow::{Borrow, BorrowMut},
    iter::{IntoIterator, Sum},
    ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, Neg, Sub},
    slice::SliceIndex,
};

use approx::{AbsDiffEq, Relative, RelativeEq, UlpsEq};
use num::{Complex, Float, Num};

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

/// A generic, fixed length, ordered vector type that supports
/// computation with N-dimensional scalar data.
#[derive(Copy, Clone, Debug)]
pub struct Vector<T, const N: usize>
where
    T: Num + Copy + Sync + Send,
{
    /// N-dimensional vector data component values.
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
    /// Returns a new [`Vector`] initialized with default numeric
    /// type scalar values.
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
    ///
    /// assert_eq!(vec_2d_i32.len(), 2);
    /// assert_eq!(vec_2d_i32[0], i32::default());
    /// assert_eq!(vec_2d_i32[1], i32::default());
    ///
    /// let vec_3d_f64 = Vector::<f64, 3>::new();
    ///
    /// assert_eq!(vec_3d_f64.len(), 3);
    /// assert_eq!(vec_3d_f64[0], f64::default());
    /// assert_eq!(vec_3d_f64[1], f64::default());
    /// assert_eq!(vec_3d_f64[2], f64::default());
    /// ```
    ///
    /// ## Type alias syntax
    ///
    /// Simplify instantiation with one of the defined 2D or 3D type aliases:
    ///
    /// ```
    /// # use vectora::Vector;
    /// use vectora::types::vector::{Vector2d, Vector2dI32, Vector3d, Vector3dF64};
    ///
    /// let vec_2d_i32_1 = Vector2d::<i32>::new();
    /// let vec_2d_i32_2 = Vector2dI32::new();
    ///
    /// let vec_3d_f64_1 = Vector3d::<f64>::new();
    /// let vec_3d_f64_2 = Vector3dF64::new();
    /// ```
    ///
    /// ## With type inference
    ///
    /// There is no requirement for additional data when the type can be inferrred
    /// by the compiler:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v: Vector<u32, 3> = Vector::new();
    /// ```
    pub fn new() -> Self {
        Self { components: [T::default(); N] }
    }

    /// Returns a new [`Vector`] initialized with scalar values
    /// defined as zero for the corresponding numeric type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ## Integer types
    ///
    /// ```
    /// # use vectora::types::vector::*;
    /// let v: Vector<i32, 3> = Vector::zero();
    ///
    /// assert_eq!(v.len(), 3);
    /// assert_eq!(v[0], 0 as i32);
    /// assert_eq!(v[1], 0 as i32);
    /// assert_eq!(v[2], 0 as i32);
    /// ```
    ///
    /// ## Float types
    ///
    /// ```
    /// # use vectora::types::vector::*;
    /// let v: Vector<f64, 3> = Vector::zero();
    ///
    /// assert_eq!(v.len(), 3);
    /// assert_eq!(v[0], 0.0 as f64);
    /// assert_eq!(v[1], 0.0 as f64);
    /// assert_eq!(v[2], 0.0 as f64);
    /// ```
    pub fn zero() -> Self {
        Self { components: [T::zero(); N] }
    }
}

impl<T, const N: usize> Default for Vector<T, N>
where
    T: Num + Copy + Default + Sync + Send,
{
    /// Returns a new [`Vector`] initialized with default
    /// numeric type scalar values.
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
    /// This method provides safe, bounds checked immutable access to scalar data
    /// references.
    ///
    /// # Examples
    ///
    /// ## With Index Values
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 3>::from([1, 2, 3]);
    ///
    /// assert_eq!(v.get(0), Some(&1));
    /// assert_eq!(v.get(1), Some(&2));
    /// assert_eq!(v.get(2), Some(&3));
    /// assert_eq!(v.get(10), None);
    /// ```
    ///
    /// ## With Index Ranges
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 5>::from([1, 2, 3, 4, 5]);
    ///
    /// assert_eq!(v.get(0..3).unwrap(), [1, 2, 3]);
    /// assert_eq!(v.get(3..).unwrap(), [4, 5]);
    /// assert_eq!(v.get(..2).unwrap(), [1, 2]);
    /// assert_eq!(v.get(..).unwrap(), [1, 2, 3, 4, 5]);
    /// assert_eq!(v.get(2..10), None);
    /// ```
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
    /// data references.
    ///
    /// # Examples
    ///
    /// ## With Index Values
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from([1, 2, 3]);
    ///
    /// assert_eq!(v.get_mut(0), Some(&mut 1));
    /// assert_eq!(v.get_mut(1), Some(&mut 2));
    /// assert_eq!(v.get_mut(2), Some(&mut 3));
    /// assert_eq!(v.get_mut(10), None);
    ///
    /// let x = v.get_mut(0).unwrap();
    ///
    /// assert_eq!(x, &mut 1);
    ///
    /// *x = 10;
    ///
    /// assert_eq!(v[0], 10);
    /// assert_eq!(v[1], 2);
    /// assert_eq!(v[2], 3);
    /// ```
    ///
    /// ## With Index Ranges
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from([1, 2, 3]);
    ///
    /// assert_eq!(v.get_mut(0..3).unwrap(), [1, 2, 3]);
    /// assert_eq!(v.get_mut(2..10), None);
    ///
    /// let r = v.get_mut(0..2).unwrap();
    ///
    /// assert_eq!(r, &mut [1, 2]);
    ///
    /// r[0] = 5;
    /// r[1] = 6;
    ///
    /// assert_eq!(v[0], 5);
    /// assert_eq!(v[1], 6);
    /// assert_eq!(v[2], 3);
    /// ```
    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut I::Output>
    where
        I: SliceIndex<[T]>,
    {
        self.components.get_mut(index)
    }

    /// Returns an iterator over immutable [`Vector`] scalar data references.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = &Vector::<i32, 3>::from([1, 2, 3]);
    /// let mut v_iter = v.iter();
    ///
    /// assert_eq!(v_iter.next(), Some(&1));
    /// assert_eq!(v_iter.next(), Some(&2));
    /// assert_eq!(v_iter.next(), Some(&3));
    /// assert_eq!(v_iter.next(), None);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.components.iter()
    }

    /// Returns an iterator over mutable [`Vector`] scalar data references.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from([1, 2, 3]);
    ///
    /// for x in v.iter_mut() {
    ///     *x += 3;
    /// }
    ///
    /// assert_eq!(v[0], 4);
    /// assert_eq!(v[1], 5);
    /// assert_eq!(v[2], 6);
    /// ```
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.components.iter_mut()
    }

    /// Returns a [`slice`] representation of the [`Vector`] scalar data.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 3>::from([1, 2, 3]);
    /// let x: &[i32] = v.as_slice();
    ///
    /// assert_eq!(x, &[1, 2, 3][..]);
    /// ```
    pub fn as_slice(&self) -> &[T] {
        &self.components[..]
    }

    /// Returns a mutable [`slice`] representation of the [`Vector`] scalar data.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from([1, 2, 3]);
    /// let mut x: &mut [i32] = v.as_mut_slice();
    ///
    /// assert_eq!(x, &[1, 2, 3][..]);
    ///
    /// x[0] = 10;
    ///
    /// assert_eq!(x, &[10,2,3][..]);
    /// ```
    ///
    /// Note: The assignment above **changes** the [`Vector`] data:
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
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.components[..]
    }

    /// Returns an [`array`] reference representation of the [`Vector`] scalar data.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 3>::from([1, 2, 3]);
    /// let x: &[i32;3] = v.as_array();
    ///
    /// assert_eq!(x, &[1, 2, 3]);
    /// ```
    pub fn as_array(&self) -> &[T; N] {
        &self.components
    }

    /// Returns a mutable [`array`] reference representation of the [`Vector`] scalar data.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from([1, 2, 3]);
    /// let mut x: &mut [i32;3] = v.as_mut_array();
    ///
    /// assert_eq!(x, &[1, 2, 3]);
    ///
    /// x[0] = 10;
    ///
    /// assert_eq!(x, &[10,2,3]);
    /// ```
    ///
    /// Note: The assignment above **changes** the [`Vector`] data:
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
    pub fn as_mut_array(&mut self) -> &mut [T; N] {
        &mut self.components
    }

    /// Returns a new [`Vector`] with an **unchecked** numeric type cast as defined by
    /// the return type in a closure parameter.
    ///
    /// # Safety
    ///
    /// While this method does not use `unsafe` blocks of code
    /// and returned data *should* be sound, it can lead to information loss
    /// and incorrect programs if return values are unexpected
    /// (e.g., saturating overflows and underflows,
    /// `NAN` can cast to zero integer values).
    ///
    /// Please understand the nuances of your type cast and use caution.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<f32, 3>::from([1.12, 2.50, 3.99]);
    ///
    /// // Float to int type with default trunc
    /// let i64_trunc_v = v.to_num_cast(|x| x as i64);
    /// assert_eq!(i64_trunc_v, Vector::<i64, 3>::from([1, 2, 3]));
    ///
    /// // Float to int type with round
    /// let i64_round_v = v.to_num_cast(|x| x.round() as i64);
    /// assert_eq!(i64_round_v, Vector::<i64, 3>::from([1, 3, 4]));
    ///
    /// // Float to int type with ceil
    /// let i64_ceil_v = v.to_num_cast(|x| x.ceil() as i64);
    /// assert_eq!(i64_ceil_v, Vector::<i64, 3>::from([2, 3, 4]));
    ///
    /// // Float to int type with floor
    /// let i64_floor_v = v.to_num_cast(|x| x.floor() as i64);
    /// assert_eq!(i64_floor_v, Vector::<i64, 3>::from([1, 2, 3]));
    /// ```
    pub fn to_num_cast<U, V>(&self, closur: U) -> Vector<V, N>
    where
        U: Fn(T) -> V,
        V: Num + Copy + Sync + Send,
    {
        let mut new_components: [V; N] = [V::zero(); N];
        self.components.iter().enumerate().for_each(|(i, x)| new_components[i] = closur(*x));
        Vector { components: new_components }
    }

    /// Returns a new, allocated [`array`] representation of the [`Vector`] scalar data.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// let mut x: [i32; 3] = v.to_array();
    ///
    /// assert_eq!(x, [1, 2, 3]);
    ///
    /// x[0] = 10;
    ///
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
    pub fn to_array(&self) -> [T; N] {
        self.components
    }

    /// Returns a new, allocated [`Vec`] representation of the [`Vector`] scalar data.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// let mut x: Vec<i32> = v.to_vec();
    ///
    /// assert_eq!(x, Vec::from([1, 2, 3]));
    ///
    /// x[0] = 10;
    ///
    /// assert_eq!(x, Vec::from([10, 2, 3]));
    /// # assert_eq!(v[0], 1);
    /// ```
    ///
    /// Note: The assignment above returns a new, owned [`Vec`] with
    /// copied data and **does not change** the [`Vector`] data:
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
    pub fn to_vec(&self) -> Vec<T> {
        Vec::from(self.components)
    }

    /// Returns the length of the [`Vector`] scalar data.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v2d: Vector<i32, 2> = Vector::new();
    ///
    /// assert_eq!(v2d.len(), 2);
    ///
    /// let v3d: Vector<f64, 3> = Vector::new();
    ///
    /// assert_eq!(v3d.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Returns `true` if the [`Vector`] contains no items and `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Vector addition with mutation of the calling [`Vector`].
    ///
    /// Returns a mutable reference to the [`Vector`].
    ///
    /// Vector addition with the [`+` operator overload](#impl-Add<Vector<T%2C%20N>>) allocates a new [`Vector`].  This method
    /// is an alternative that supports in-place vector addition mutation of the calling [`Vector`] with data in the
    /// parameter [`Vector`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// let other = Vector::<i32, 3>::from(&[4, 5, 6]);
    ///
    /// v.mut_add(&other);
    ///
    /// assert_eq!(v[0], 5);
    /// assert_eq!(v[1], 7);
    /// assert_eq!(v[2], 9);
    /// ```
    pub fn mut_add(&mut self, rhs: &Vector<T, N>) -> &mut Self {
        self.components.iter_mut().zip(rhs).for_each(|(a, b)| *a = *a + *b);

        self
    }

    /// Vector subtraction with mutation of the calling [`Vector`].
    ///
    /// Returns a mutable reference to the [`Vector`].
    ///
    /// Vector subtraction with the [`-` operator overload](#impl-Sub<Vector<T%2C%20N>>) allocates a new [`Vector`].  This method
    /// is an alternative that supports in-place vector subtraction mutation of the calling [`Vector`] with data in the
    /// parameter [`Vector`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from(&[1, 2, 3]);
    /// let other = Vector::<i32, 3>::from(&[4, 5, 6]);
    ///
    /// v.mut_sub(&other);
    ///
    /// assert_eq!(v[0], -3);
    /// assert_eq!(v[1], -3);
    /// assert_eq!(v[2], -3);
    /// ```
    pub fn mut_sub(&mut self, rhs: &Vector<T, N>) -> &mut Self {
        self.components.iter_mut().zip(rhs.iter()).for_each(|(a, b)| *a = *a - *b);

        self
    }

    /// Scalar multiplication with mutation of the calling [`Vector`].
    ///
    /// Returns a mutable reference to the [`Vector`].
    ///
    /// Scalar multiplication with the [`*` operator overload](#impl-Mul<T>) allocates a
    /// new [`Vector`].  This method is an alternative that supports in-place scalar
    /// multiplication mutation of the calling [`Vector`] with a scalar parameter value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from(&[1, 2, 3]);
    ///
    /// v.mut_mul(4);
    ///
    /// assert_eq!(v[0], 4);
    /// assert_eq!(v[1], 8);
    /// assert_eq!(v[2], 12);
    /// ```
    pub fn mut_mul(&mut self, scale: T) -> &mut Self {
        self.components.iter_mut().for_each(|a| *a = *a * scale);

        self
    }

    /// Returns the dot product of two real number [`Vector`] types.
    ///
    /// The return value is a scalar with the [`Vector`] numeric
    /// type.
    ///
    /// Note: This method is not intended for use with [`Vector`]
    /// **of** [`num::complex::Complex`] number types.
    ///
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v1: Vector<i32, 3> = Vector::from([1, 3, -5]);
    /// let v2: Vector<i32, 3> = Vector::from([4, -2, -1]);
    ///
    /// let x1 = v1 * 3;
    /// let x2 = v2 * 6;
    ///
    /// assert_eq!(v1.dot(&v2), 3);
    /// assert_eq!(v2.dot(&v1), 3);
    /// assert_eq!(-v1.dot(&-v2), 3);
    /// assert_eq!(x1.dot(&x2), (3 * 6) * v1.dot(&v2));
    /// ```
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
    /// let to = Vector::<i32, 3>::from([1, 2, 3]);
    /// let from = Vector::<i32, 3>::from([2, 4, 6]);
    ///
    /// let v = to.displacement(&from);
    ///
    /// assert_eq!(v[0], -1);
    /// assert_eq!(v[1], -2);
    /// assert_eq!(v[2], -3);
    /// ```
    pub fn displacement(&self, from: &Vector<T, N>) -> Self {
        self.sub(*from)
    }

    /// Returns a [`Vector`] with the same magnitude and opposite direction for
    /// a non-zero [`Vector`].
    ///
    /// This operation does not change zero vectors.
    ///
    /// Note: This is an alias for the unary [`Vector::neg`] operation
    /// and can be performed with the overloaded unary `-` operator.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 3>::from([1, 2, 3]);
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
    ///
    /// assert_eq!(v_zero_o, v_zero);
    /// ```
    pub fn opposite(&self) -> Self {
        -*self
    }

    /// Mutates a non-zero [`Vector`] in place to one with the same magnitude and opposite direction.
    ///
    /// This operation returns a zero vector when the calling vector is a zero vector.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from([1, 2, 3]);
    ///
    /// v.mut_opposite();
    ///
    /// assert_eq!(v[0], -1);
    /// assert_eq!(v[1], -2);
    /// assert_eq!(v[2], -3);
    /// ```
    ///
    /// The zero vector case:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v_zero = Vector::<i32, 3>::zero();
    ///
    /// v_zero.mut_opposite();
    ///
    /// assert_eq!(v_zero[0], 0);
    /// assert_eq!(v_zero[1], 0);
    /// assert_eq!(v_zero[2], 0);
    /// ```
    pub fn mut_opposite(&mut self) -> &mut Self {
        self.components.iter_mut().for_each(|a| *a = T::zero() - *a);
        self
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
    /// let v = Vector::<i32, 3>::from([1, 2, 3]);
    ///
    /// let v_s = v.scale(10);
    ///
    /// assert_eq!(v_s[0], 10);
    /// assert_eq!(v_s[1], 20);
    /// assert_eq!(v_s[2], 30);
    /// ```
    pub fn scale(&self, scale: T) -> Self {
        self.mul(scale)
    }

    /// Scales a [`Vector`] in place by a given scalar parameter value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from([1, 2, 3]);
    ///
    /// v.mut_scale(10);
    ///
    /// assert_eq!(v[0], 10);
    /// assert_eq!(v[1], 20);
    /// assert_eq!(v[2], 30);
    /// ```
    pub fn mut_scale(&mut self, scale: T) -> &mut Self {
        self.mut_mul(scale)
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
    /// let v = Vector::<i32, 3>::from([1, 2, 3]);
    /// let translation_vec = Vector::<i32, 3>::from([4, 5, 6]);
    ///
    /// let v_t = v.translate(&translation_vec);
    ///
    /// assert_eq!(v_t[0], 5);
    /// assert_eq!(v_t[1], 7);
    /// assert_eq!(v_t[2], 9);
    /// ```
    pub fn translate(&self, translation_vector: &Vector<T, N>) -> Self {
        self.add(*translation_vector)
    }

    /// Translates a [`Vector`] in place with displacement defined by a
    /// translation [`Vector`] parameter.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from([1, 2, 3]);
    /// let translation_vec = Vector::<i32, 3>::from([4, 5, 6]);
    ///
    /// v.mut_translate(&translation_vec);
    ///
    /// assert_eq!(v[0], 5);
    /// assert_eq!(v[1], 7);
    /// assert_eq!(v[2], 9);
    /// ```
    pub fn mut_translate(&mut self, translation_vector: &Vector<T, N>) -> &mut Self {
        self.mut_add(translation_vector)
    }

    /// Returns a new [`Vector`] with scalar data that are modified
    /// according to the definition in a closure parameter.
    ///
    /// Note: the closure must return items of the same numeric
    /// type as the [`Vector`] numeric type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 3>::from([1, 2, 3]);
    /// let square = |x: i32| { x.pow(2) };
    ///
    /// let squared_v = v.map_closure(square);
    ///
    /// assert_eq!(squared_v, Vector::from([1, 4, 9]));
    /// ```
    pub fn map_closure<U>(&self, closur: U) -> Self
    where
        U: Fn(T) -> T,
    {
        let mut new_components: [T; N] = [T::zero(); N];
        self.components.iter().enumerate().for_each(|(i, x)| new_components[i] = closur(*x));
        Self { components: new_components }
    }

    /// Mutates the [`Vector`] data in place according to the
    /// definition in a closure parameter.
    ///
    /// Note: the closure must return items of the same numeric
    /// type as the [`Vector`] numeric type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from([1, 2, 3]);
    /// let square = |x: i32| { x.pow(2) };
    ///
    /// v.mut_map_closure(square);
    ///
    /// assert_eq!(v, Vector::from([1, 4, 9]));
    /// ```
    pub fn mut_map_closure<U>(&mut self, mut closur: U) -> &mut Self
    where
        U: FnMut(T) -> T,
    {
        self.components.iter_mut().for_each(|x| *x = closur(*x));
        self
    }

    /// Returns a new [`Vector`] with data that are modified
    /// according to the definition in a function parameter.
    ///
    /// Note: the function must return items of the same numeric
    /// type as the [`Vector`] numeric type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// fn square(x: i32) -> i32 {
    ///     x.pow(2)
    /// }
    ///
    /// let mut v = Vector::<i32, 3>::from([1, 2, 3]);
    ///
    /// let squared_v = v.map_fn(square);
    ///
    /// assert_eq!(squared_v, Vector::from([1, 4, 9]));
    /// ```
    pub fn map_fn(&self, func: fn(T) -> T) -> Self {
        let mut new_components: [T; N] = [T::zero(); N];
        self.components.iter().enumerate().for_each(|(i, x)| new_components[i] = func(*x));
        Self { components: new_components }
    }

    /// Mutates the [`Vector`] data in place according to
    /// the definition in a function parameter.
    ///
    /// Note: the function must return items of the same numeric
    /// type as the [`Vector`] numeric type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// fn square(x: i32) -> i32 {
    ///     x.pow(2)
    /// }
    ///
    /// let mut v = Vector::<i32, 3>::from([1, 2, 3]);
    ///
    /// v.mut_map_fn(square);
    ///
    /// assert_eq!(v, Vector::from([1, 4, 9]));
    /// ```
    pub fn mut_map_fn(&mut self, func: fn(T) -> T) -> &mut Self {
        self.components.iter_mut().for_each(|x| *x = func(*x));
        self
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
    /// calling real, floating point number [`Vector`] and a real, floating
    /// point number [`Vector`] parameter.
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
    /// Note: This method can be used on data in integer [`Vector`] types
    /// with **lossless** integer to float [`Vector`] type casts:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let vi: Vector<i32, 2> = Vector::from([2, 2]);
    /// let vf: Vector<f64, 2> = vi.into();
    /// ```
    pub fn distance(&self, other: &Vector<T, N>) -> T {
        (*self - *other).magnitude()
    }

    /// Returns the linear interpolant between a real, floating point number [`Vector`] and a
    /// parameter [`Vector`] given a parametric line equation `weight` parameter.
    ///
    /// Calculated with the parametric line equation `(1 - t)A + tB`
    /// where `A` is the start vector, `B` is the end vector, and `t` is the weight.
    ///
    /// The `weight` parameter must be in the closed interval [0.0, 1.0].
    ///
    /// # Errors
    ///
    /// This method does not support extrapolation.  [`VectorError::ValueError`]
    /// is raised if the `weight` parameter is not in the closed interval [0.0, 1.0].
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v1: Vector<f64, 2> = Vector::from([0.0, 0.0]);
    /// let v2: Vector<f64, 2> = Vector::from([10.0, 10.0]);
    ///
    /// assert_eq!(v1.lerp(&v2, 0.0).unwrap(), Vector::from([0.0, 0.0]));
    /// assert_eq!(v1.lerp(&v2, 0.25).unwrap(), Vector::from([2.5, 2.5]));
    /// assert_eq!(v1.lerp(&v2, 0.5).unwrap(), Vector::from([5.0, 5.0]));
    /// assert_eq!(v1.lerp(&v2, 0.75).unwrap(), Vector::from([7.5, 7.5]));
    /// assert_eq!(v1.lerp(&v2, 1.0).unwrap(), Vector::from([10.0, 10.0]));
    /// ```
    ///
    /// Note: This method can be used on data in integer [`Vector`] types
    /// with **lossless** integer to float [`Vector`] type casts:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let vi: Vector<i32, 2> = Vector::from([2, 2]);
    /// let vf: Vector<f64, 2> = vi.into();
    /// ```
    pub fn lerp(&self, end: &Vector<T, N>, weight: T) -> Result<Self, VectorError>
    where
        T: std::fmt::Debug,
    {
        // weight bounds check
        if weight > T::one() || weight < T::zero() {
            return Err(VectorError::ValueError(format!(
                "invalid interpolation weight request. The weight must be in the closed interval [0,1]. Received '{:?}'",
                weight
            )));
        }

        // weight bounds check for NaN
        if weight.is_nan() {
            return Err(VectorError::ValueError(
                "invalid interpolation weight request. The weight must be in the closed interval [0,1]. Received NaN.".to_string(),
            ));
        }
        // if the vectors are the same, always return the start vector (== end vector)
        // no calculation is required
        if self.components == end.components {
            return Ok(Self { components: self.components });
        }

        Ok(self.lerp_impl(end, weight))
    }

    /// Returns the midpoint between a real, floating point number [`Vector`] and a parameter [`Vector`].
    ///
    /// This vector is defined as the [linear interpolant](#method.lerp) with `weight` = 0.5.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    ///
    /// let v1: Vector<f64, 2> = Vector::from([0.0, 0.0]);
    /// let v2: Vector<f64, 2> = Vector::from([10.0, 10.0]);
    ///
    /// let mid = v1.midpoint(&v2);
    ///
    /// assert_eq!(mid, Vector::from([5.0, 5.0]));
    /// ```
    ///
    /// Note: This method can be used on data in integer [`Vector`] types
    /// with **lossless** integer to float [`Vector`] type casts:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let vi: Vector<i32, 2> = Vector::from([2, 2]);
    /// let vf: Vector<f64, 2> = vi.into();
    /// ```
    pub fn midpoint(&self, end: &Vector<T, N>) -> Self
    where
        T: std::fmt::Debug,
    {
        // we don't need the bounds checks on weight because it has
        // an explicit, valid definition here.  Ok to use `lerp_impl`
        // directly.
        self.lerp_impl(end, num::cast(0.5).unwrap())
    }

    /// Returns the vector magnitude for a real, floating point number [`Vector`].
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
    /// Note: This method can be used on data in integer [`Vector`] types
    /// with **lossless** integer to float [`Vector`] type casts:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let vi: Vector<i32, 2> = Vector::from([2, 2]);
    /// let vf: Vector<f64, 2> = vi.into();
    /// ```
    pub fn magnitude(&self) -> T {
        let x: T = self.components.iter().map(|a| *a * *a).sum();
        x.sqrt()
    }

    /// Returns a new, normalized unit [`Vector`] from a real,
    /// floating point number calling [`Vector`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// use approx::assert_relative_eq;
    ///
    /// let v: Vector<f64, 2> = Vector::from([25.123, 30.456]);
    ///
    /// assert_relative_eq!(v.normalize().magnitude(), 1.0);
    /// assert_relative_eq!(v[0], 25.123);
    /// assert_relative_eq!(v[1], 30.456);
    /// ```
    ///
    /// Note: This method can be used on data in integer [`Vector`] types
    /// with **lossless** integer to float [`Vector`] type casts:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let vi: Vector<i32, 2> = Vector::from([2, 2]);
    /// let vf: Vector<f64, 2> = vi.into();
    /// ```
    pub fn normalize(&self) -> Self
    where
        T: Float + Copy + Sync + Send + Sum,
    {
        let mut new_components = [T::zero(); N];
        let this_magnitude = self.magnitude();

        self.components
            .iter()
            .enumerate()
            .for_each(|(i, a)| new_components[i] = *a / this_magnitude);

        Self { components: new_components }
    }

    /// Normalizes a real, floating point number [`Vector`] to a unit [`Vector`] in place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// use approx::assert_relative_eq;
    ///
    /// let mut v: Vector<f64, 2> = Vector::from([25.123, 30.456]);
    ///
    /// v.mut_normalize();
    ///
    /// assert_relative_eq!(v.magnitude(), 1.0);
    /// assert_relative_eq!(v[0], 0.6363347262144607);
    /// assert_relative_eq!(v[1], 0.7714130645857428);
    /// ```
    ///
    /// Note: This method can be used on data in integer [`Vector`] types
    /// with **lossless** integer to float [`Vector`] type casts:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let vi: Vector<i32, 2> = Vector::from([2, 2]);
    /// let vf: Vector<f64, 2> = vi.into();
    /// ```
    pub fn mut_normalize(&mut self) -> &mut Self
    where
        T: Float + Copy + Sync + Send + Sum,
    {
        let this_magnitude = self.magnitude();
        self.components.iter_mut().for_each(|a| *a = *a / this_magnitude);
        self
    }

    // ================================
    //
    // Private methods
    //
    // ================================

    // Vector linear interpolation implementation
    fn lerp_impl(&self, end: &Vector<T, N>, weight: T) -> Self {
        let mut new_components = [T::zero(); N];
        self.components
            .iter()
            .zip(end.components.iter())
            .map(|(a, b)| ((T::one() - weight) * *a) + (weight * *b))
            .enumerate()
            .for_each(|(i, a)| new_components[i] = a);

        Self { components: new_components }
    }
}

// ================================
//
// Index / IndexMut trait impl
//
// ================================
impl<I, T, const N: usize> Index<I> for Vector<T, N>
where
    I: SliceIndex<[T]>,
    T: Num + Copy + Sync + Send,
{
    type Output = I::Output;
    /// Returns [`Vector`] scalar data values by zero-based index.
    ///
    /// # Examples
    ///
    /// Indexing:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 3>::from([1, 2, 3]);
    ///
    /// assert_eq!(v[0], 1);
    /// assert_eq!(v[1], 2);
    /// assert_eq!(v[2], 3);
    /// ```
    ///
    /// Slicing:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let v = Vector::<i32, 3>::from([1, 2, 3]);
    ///
    /// let v_slice = &v[..];
    ///
    /// assert_eq!(v_slice, [1, 2, 3]);
    /// ```
    ///
    fn index(&self, i: I) -> &Self::Output {
        &self.components[i]
    }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N>
where
    T: Num + Copy + Sync + Send,
{
    /// Returns mutable [`Vector`] scalar data values by zero-based index.
    ///
    /// Supports scalar value assignment by index.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use vectora::types::vector::Vector;
    /// let mut v = Vector::<i32, 3>::from(&[1, 2, 3]);
    ///
    /// assert_eq!(v[0], 1);
    /// assert_eq!(v[1], 2);
    /// assert_eq!(v[2], 3);
    ///
    /// v[0] = 5;
    /// v[1] = 6;
    ///
    /// assert_eq!(v[0], 5);
    /// assert_eq!(v[1], 6);
    /// assert_eq!(v[2], 3);
    /// ```
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

    /// Creates a consuming iterator that iterates over scalar data by value.
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

    /// Creates an iterator over immutable scalar data references.
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

    /// Creates an iterator over mutable scalar data references.
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
    /// undefined data components with the default value for the numeric type
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
    ///
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
    ///
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
    ///
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

/// PartialEq trait implementation for [`Vector`] with integer data types.
///
/// These comparisons establish the symmetry and transitivity relationships
/// required for the partial equivalence relation definition with integer types.
///
/// Note:
///
/// - Negative zero to positive zero comparisons are considered equal.
macro_rules! impl_vector_int_partialeq_from {
    ($IntTyp: ty, $doc: expr) => {
        impl<const N: usize> PartialEq<Vector<$IntTyp, N>> for Vector<$IntTyp, N> {
            #[doc = $doc]
            fn eq(&self, other: &Self) -> bool {
                self.components == other.components
            }
        }
    };
    ($IntTyp: ty) => {
        impl_vector_int_partialeq_from!(
            $IntTyp,
            concat!("PartialEq trait implementation for `Vector<", stringify!($IntTyp), ", N>`")
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
/// This approach uses the default approx crate relative epsilon float equality testing implementation.
/// This equivalence relation implementation is based on the approach described in
/// [Comparing Floating Point Numbers, 2012 Edition](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
macro_rules! impl_vector_float_partialeq_from {
    ($FloatTyp: ty, $doc: expr) => {
        impl<const N: usize> PartialEq<Vector<$FloatTyp, N>> for Vector<$FloatTyp, N> {
            #[doc = $doc]
            fn eq(&self, other: &Self) -> bool {
                for (i, item) in self.components.iter().enumerate() {
                    // uses default approx crate epsilon and max relative
                    // diff = epsilon value parameter definitions
                    if !Relative::default().eq(item, &other[i]) {
                        return false;
                    }
                }

                true
            }
        }
    };
    ($FloatTyp: ty) => {
        impl_vector_float_partialeq_from!(
            $FloatTyp,
            concat!("PartialEq trait implementation for `Vector<", stringify!($FloatTyp), ", N>`")
        );
    };
}

impl_vector_float_partialeq_from!(f32);
impl_vector_float_partialeq_from!(f64);

/// PartialEq trait implementation for [`Vector`] of [`num:Complex`] filled with
/// integer real and imaginary part data.
///
/// These comparisons establish the symmetry and transitivity relationships
/// required for the partial equivalence relation definition with complex numbers
/// with integer real and imaginary part types.
///
/// Note:
///
/// - Negative zero to positive zero comparisons are considered equal.
macro_rules! impl_vector_complex_int_partialeq_from {
    ($IntTyp: ty, $doc: expr) => {
        impl<const N: usize> PartialEq<Vector<Complex<$IntTyp>, N>>
            for Vector<Complex<$IntTyp>, N>
        {
            #[doc = $doc]
            fn eq(&self, other: &Self) -> bool {
                self.components == other.components
            }
        }
    };
    ($IntTyp: ty) => {
        impl_vector_complex_int_partialeq_from!(
            $IntTyp,
            concat!(
                "PartialEq trait implementation for `Vector<Complex<",
                stringify!($IntTyp),
                ">, N>`"
            )
        );
    };
}

impl_vector_complex_int_partialeq_from!(usize);
impl_vector_complex_int_partialeq_from!(u8);
impl_vector_complex_int_partialeq_from!(u16);
impl_vector_complex_int_partialeq_from!(u32);
impl_vector_complex_int_partialeq_from!(u64);
impl_vector_complex_int_partialeq_from!(u128);
impl_vector_complex_int_partialeq_from!(isize);
impl_vector_complex_int_partialeq_from!(i8);
impl_vector_complex_int_partialeq_from!(i16);
impl_vector_complex_int_partialeq_from!(i32);
impl_vector_complex_int_partialeq_from!(i64);
impl_vector_complex_int_partialeq_from!(i128);

/// PartialEq trait implementation for [`Vector`] of [`Complex`] numbers with
/// floating point real and imaginary parts.
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
/// This approach uses the default approx crate relative epsilon float equality testing implementation.
/// This equivalence relation implementation is based on the approach described in
/// [Comparing Floating Point Numbers, 2012 Edition](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
macro_rules! impl_vector_complex_float_partialeq_from {
    ($FloatTyp: ty, $doc: expr) => {
        impl<const N: usize> PartialEq<Vector<Complex<$FloatTyp>, N>>
            for Vector<Complex<$FloatTyp>, N>
        {
            #[doc = $doc]
            fn eq(&self, other: &Self) -> bool {
                for (i, item) in self.components.iter().enumerate() {
                    // uses default approx crate epsilon and max relative
                    // diff = epsilon value parameter definitions
                    if !Relative::default().eq(&item.re, &other[i].re)
                        || !Relative::default().eq(&item.im, &other[i].im)
                    {
                        return false;
                    }
                }

                true
            }
        }
    };
    ($FloatTyp: ty) => {
        impl_vector_complex_float_partialeq_from!(
            $FloatTyp,
            concat!(
                "PartialEq trait implementation for `Vector<Complex<",
                stringify!($FloatTyp),
                ">, N>`"
            )
        );
    };
}

impl_vector_complex_float_partialeq_from!(f32);
impl_vector_complex_float_partialeq_from!(f64);

// ======================================================
//
// approx crate float approximate equivalence trait impl
// AbsDiffEq, RelativeEq, UlpsEq traits
//
// ======================================================

// approx::AbsDiffEq trait impl for floating point data types
macro_rules! impl_vector_float_absdiffeq_from {
    ($FloatTyp: ty, $doc: expr) => {
        impl<const N: usize> AbsDiffEq for Vector<$FloatTyp, N> {
            type Epsilon = $FloatTyp;

            fn default_epsilon() -> $FloatTyp {
                <$FloatTyp>::default_epsilon()
            }

            fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                for (i, item) in self.components.iter().enumerate() {
                    if !<$FloatTyp>::abs_diff_eq(item, &other[i], epsilon) {
                        return false;
                    }
                }
                true
            }
        }
    };
    ($FloatTyp: ty) => {
        impl_vector_float_absdiffeq_from!(
            $FloatTyp,
            concat!(
                "approx::AbsDiffEq trait implementation for `Vector<",
                stringify!($FloatTyp),
                ", N>`"
            )
        );
    };
}

impl_vector_float_absdiffeq_from!(f32);
impl_vector_float_absdiffeq_from!(f64);

// approx::AbsDiffEq trait impl for complex numbers with floating point
// real and imaginary part data types
macro_rules! impl_vector_complex_float_absdiffeq_from {
    ($FloatTyp: ty, $doc: expr) => {
        impl<const N: usize> AbsDiffEq for Vector<Complex<$FloatTyp>, N> {
            type Epsilon = $FloatTyp;

            fn default_epsilon() -> $FloatTyp {
                <$FloatTyp>::default_epsilon()
            }

            fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                for (i, item) in self.components.iter().enumerate() {
                    // compare the real and imaginary parts for eq
                    if !<$FloatTyp>::abs_diff_eq(&item.re, &other[i].re, epsilon)
                        || !<$FloatTyp>::abs_diff_eq(&item.im, &other[i].im, epsilon)
                    {
                        return false;
                    }
                }
                true
            }
        }
    };
    ($FloatTyp: ty) => {
        impl_vector_complex_float_absdiffeq_from!(
            $FloatTyp,
            concat!(
                "approx::AbsDiffEq trait implementation for `Vector<Complex<",
                stringify!($FloatTyp),
                ">, N>`"
            )
        );
    };
}

impl_vector_complex_float_absdiffeq_from!(f32);
impl_vector_complex_float_absdiffeq_from!(f64);

// approx::RelativeEq trait impl for floating point data types
macro_rules! impl_vector_float_relativeeq_from {
    ($FloatTyp: ty, $doc: expr) => {
        impl<const N: usize> RelativeEq for Vector<$FloatTyp, N> {
            fn default_max_relative() -> $FloatTyp {
                <$FloatTyp>::default_max_relative()
            }

            fn relative_eq(
                &self,
                other: &Self,
                epsilon: Self::Epsilon,
                max_relative: Self::Epsilon,
            ) -> bool {
                for (i, item) in self.components.iter().enumerate() {
                    if !<$FloatTyp>::relative_eq(item, &other[i], epsilon, max_relative) {
                        return false;
                    }
                }
                true
            }
        }
    };
    ($FloatTyp: ty) => {
        impl_vector_float_relativeeq_from!(
            $FloatTyp,
            concat!(
                "approx::RelativeEq trait implementation for `Vector<",
                stringify!($FloatTyp),
                ", N>`"
            )
        );
    };
}

impl_vector_float_relativeeq_from!(f32);
impl_vector_float_relativeeq_from!(f64);

// approx::RelativeEq trait impl for floating point data types
macro_rules! impl_vector_complex_float_relativeeq_from {
    ($FloatTyp: ty, $doc: expr) => {
        impl<const N: usize> RelativeEq for Vector<Complex<$FloatTyp>, N> {
            fn default_max_relative() -> $FloatTyp {
                <$FloatTyp>::default_max_relative()
            }

            fn relative_eq(
                &self,
                other: &Self,
                epsilon: Self::Epsilon,
                max_relative: Self::Epsilon,
            ) -> bool {
                for (i, item) in self.components.iter().enumerate() {
                    // compare the real and imaginary parts for eq
                    if !<$FloatTyp>::relative_eq(&item.re, &other[i].re, epsilon, max_relative)
                        || !<$FloatTyp>::relative_eq(&item.im, &other[i].im, epsilon, max_relative)
                    {
                        return false;
                    }
                }
                true
            }
        }
    };
    ($FloatTyp: ty) => {
        impl_vector_complex_float_relativeeq_from!(
            $FloatTyp,
            concat!(
                "approx::RelativeEq trait implementation for `Vector<Complex<",
                stringify!($FloatTyp),
                ">, N>`"
            )
        );
    };
}

impl_vector_complex_float_relativeeq_from!(f32);
impl_vector_complex_float_relativeeq_from!(f64);

// approx::UlpsEq trait impl
macro_rules! impl_vector_float_ulpseq_from {
    ($FloatTyp: ty, $doc: expr) => {
        impl<const N: usize> UlpsEq for Vector<$FloatTyp, N> {
            fn default_max_ulps() -> u32 {
                <$FloatTyp>::default_max_ulps()
            }

            fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                for (i, item) in self.components.iter().enumerate() {
                    if !<$FloatTyp>::ulps_eq(item, &other[i], epsilon, max_ulps) {
                        return false;
                    }
                }
                true
            }
        }
    };
    ($FloatTyp: ty) => {
        impl_vector_float_ulpseq_from!(
            $FloatTyp,
            concat!(
                "approx::UlpsEq trait implementation for `Vector<",
                stringify!($FloatTyp),
                ", N>`"
            )
        );
    };
}

impl_vector_float_ulpseq_from!(f32);
impl_vector_float_ulpseq_from!(f64);

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
    /// [`Vec`] lengths are not fixed and may not be known at compile time. Bounds
    /// checks are used in this approach.  This is slower than instantiation from
    /// arrays and will fail with overflows and underflows.
    ///
    /// # Errors
    ///
    /// Raises [`VectorError::TryFromVecError`] when the [`Vec`] parameter length
    /// does not equal the expected [`Vector`] data component length.
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
    /// the same as the number of requested [`Vector`] data components.  The following
    /// code raises [`VectorError::TryFromVecError`] on an attempt to make a
    /// three dimensional [`Vector`] with two dimensional data:
    ///
    /// ```
    ///# use vectora::types::vector::Vector;
    /// let v = vec![1 as i32, 2 as i32];
    /// let e = Vector::<i32, 3>::try_from(v);
    ///
    /// assert!(e.is_err());
    /// ```
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
    /// [`Vec`] lengths are not fixed and may not be known at compile time. Bounds
    /// checks are used in this approach.  This is slower than instantiation from
    /// arrays and will fail with overflows and underflows.
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
    /// the same as the number of requested [`Vector`] data dimensions.  The following
    /// code raises [`VectorError::TryFromVecError`] on an attempt to make a
    /// three dimensional [`Vector`] with two dimensional data:
    ///
    /// ```
    ///# use vectora::types::vector::Vector;
    /// let v = vec![1 as i32, 2 as i32];
    /// let e = Vector::<i32, 3>::try_from(&v);
    ///
    /// assert!(e.is_err());
    /// ```
    fn try_from(t_vec: &Vec<T>) -> Result<Vector<T, N>, VectorError> {
        if t_vec.len() != N {
            return Err(VectorError::TryFromVecError(format!(
                "expected Vec with {} items, but received Vec with {} items",
                N,
                t_vec.len()
            )));
        }

        Self::try_from(&t_vec[..])
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
    /// is not equal to the requested [`Vector`] data component length.
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
    ///
    /// Callers should confirm that the length of the [`slice`] is
    /// the same as the number of requested [`Vector`] data dimensions.  The following
    /// code raises [`VectorError::TryFromSliceError`] on an attempt to make a
    /// three dimensional [`Vector`] with two dimensional data:
    ///
    /// ```
    ///# use vectora::types::vector::Vector;
    /// let v = vec![1 as i32, 2 as i32];
    /// let s = &v[..];
    /// let e = Vector::<i32, 3>::try_from(s);
    ///
    /// assert!(e.is_err());
    /// ```
    fn try_from(t_slice: &[T]) -> Result<Vector<T, N>, VectorError> {
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
                err
            ))),
        }
    }
}

/// Returns a new [`Vector`] with lossless [`Vector`] scalar numeric type data
/// cast support.
macro_rules! impl_vector_from_vector {
    ($Small: ty, $Large: ty, $doc: expr) => {
        impl<const N: usize> From<Vector<$Small, N>> for Vector<$Large, N> {
            #[doc = $doc]
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
    use num::complex::Complex;
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
    fn vector_instantiation_complex_i32() {
        let c1 = Complex::new(10_i32, 20_i32);
        let c2 = Complex::new(3_i32, -4_i32);
        let v: Vector<Complex<i32>, 2> = Vector::from([c1, c2]);
        assert_eq!(v[0].re, 10_i32);
        assert_eq!(v[0].im, 20_i32);
        assert_eq!(v[1].re, 3_i32);
        assert_eq!(v[1].im, -4_i32);
    }

    #[test]
    fn vector_instantiation_complex_f64() {
        let c1 = Complex::new(10.0_f64, 20.0_f64);
        let c2 = Complex::new(3.1_f64, -4.2_f64);
        let v: Vector<Complex<f64>, 2> = Vector::from([c1, c2]);
        assert_relative_eq!(v[0].re, 10.0_f64);
        assert_relative_eq!(v[0].im, 20.0_f64);
        assert_relative_eq!(v[1].re, 3.1_f64);
        assert_relative_eq!(v[1].im, -4.2_f64);
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
    fn vector_instantiation_complex_i32_default() {
        let v = Vector::<Complex<i32>, 2>::default();
        assert_eq!(v[0].re, 0_i32);
        assert_eq!(v[0].im, 0_i32);
    }

    #[test]
    fn vector_instantiation_complex_f64_default() {
        let v = Vector::<Complex<f64>, 2>::default();
        assert_relative_eq!(v[0].re, 0.0_f64);
        assert_relative_eq!(v[0].im, 0.0_f64);
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
    // concurrency support
    //
    // ================================

    #[test]
    fn vector_send_sync_concurrency_int() {
        let v = Vector::<i32, 2>::from([1, 2]);

        let handle = std::thread::spawn(move || {
            println!("{:?}", v);
        });

        handle.join().unwrap();
    }

    #[test]
    fn vector_send_sync_concurrency_float() {
        let v = Vector::<f64, 2>::from([1.0, 2.0]);

        let handle = std::thread::spawn(move || {
            println!("{:?}", v);
        });

        handle.join().unwrap();
    }

    #[test]
    fn vector_send_sync_concurrency_complex_float() {
        let v = Vector::<Complex<f64>, 2>::from([Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);

        let handle = std::thread::spawn(move || {
            println!("{:?}", v);
        });

        handle.join().unwrap();
    }

    // ================================
    //
    // get method tests
    //
    // ================================
    #[test]
    fn vector_method_get_with_value() {
        let v1 = Vector::<u32, 2>::from([1, 2]);
        let v2 = Vector::<f64, 2>::from([1.0, 2.0]);
        assert_eq!(v1.get(0).unwrap(), &1);
        assert_eq!(v1.get(1).unwrap(), &2);
        assert_eq!(v1.get(2), None);
        assert_relative_eq!(v2.get(0).unwrap(), &1.0);
        assert_relative_eq!(v2.get(1).unwrap(), &2.0);
        assert_eq!(v2.get(2), None);
    }

    #[test]
    fn vector_method_get_with_value_complex() {
        let v1 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);
        let v2 = Vector::<Complex<f64>, 2>::from([Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        assert_eq!(v1.get(0).unwrap().re, 1);
        assert_eq!(v1.get(0).unwrap().im, 2);
        assert_eq!(v1.get(1).unwrap().re, 3);
        assert_eq!(v1.get(1).unwrap().im, 4);
        assert_eq!(v1.get(2), None);

        assert_relative_eq!(v2.get(0).unwrap().re, 1.0);
        assert_relative_eq!(v2.get(0).unwrap().im, 2.0);
        assert_relative_eq!(v2.get(1).unwrap().re, 3.0);
        assert_relative_eq!(v2.get(1).unwrap().im, 4.0);
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
    fn vector_method_get_with_range_complex() {
        let v1 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);
        let v2 = Vector::<Complex<f64>, 2>::from([Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        assert_eq!(v1.get(0..2).unwrap(), [Complex::new(1_i32, 2_i32), Complex::new(3_i32, 4_i32)]);
        assert_eq!(v1.get(..2).unwrap(), [Complex::new(1_i32, 2_i32), Complex::new(3_i32, 4_i32)]);
        assert_eq!(v1.get(1..).unwrap(), [Complex::new(3_i32, 4_i32)]);
        assert_eq!(v1.get(..).unwrap(), [Complex::new(1_i32, 2_i32), Complex::new(3_i32, 4_i32)]);
        assert_eq!(v1.get(4..8), None);

        assert_eq!(
            v2.get(0..2).unwrap(),
            [Complex::new(1.0_f64, 2.0_f64), Complex::new(3.0_f64, 4.0_f64)]
        );
        assert_eq!(
            v2.get(..2).unwrap(),
            [Complex::new(1.0_f64, 2.0_f64), Complex::new(3.0_f64, 4.0_f64)]
        );
        assert_eq!(v2.get(1..).unwrap(), [Complex::new(3.0_f64, 4.0_f64)]);
        assert_eq!(
            v2.get(..).unwrap(),
            [Complex::new(1.0_f64, 2.0_f64), Complex::new(3.0_f64, 4.0_f64)]
        );
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
    fn vector_method_get_mut_with_value_complex() {
        let mut v1 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);
        let mut v2 =
            Vector::<Complex<f64>, 2>::from([Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);

        let c1 = v1.get_mut(0).unwrap();
        assert_eq!(c1.re, 1);
        assert_eq!(c1.im, 2);
        c1.re = 5;
        c1.im = 6;
        assert_eq!(v1[0], Complex::new(5, 6));
        assert_eq!(v1[1], Complex::new(3, 4));

        let c2 = v2.get_mut(0).unwrap();
        assert_relative_eq!(c2.re, 1.0);
        assert_relative_eq!(c2.im, 2.0);
        c2.re = 5.0;
        c2.im = 6.0;
        assert_relative_eq!(v2[0].re, 5.0);
        assert_relative_eq!(v2[0].im, 6.0);
        assert_relative_eq!(v2[1].re, 3.0);
        assert_relative_eq!(v2[1].im, 4.0);
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

    #[test]
    fn vector_method_get_mut_with_range_complex() {
        let mut v1 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);
        let mut v2 =
            Vector::<Complex<f64>, 2>::from([Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);

        let r1 = v1.get_mut(0..2).unwrap();
        assert_eq!(r1, [Complex::new(1, 2), Complex::new(3, 4)]);
        r1[0].re = 5;
        r1[0].im = 6;
        assert_eq!(v1.components, [Complex::new(5, 6), Complex::new(3, 4)]);

        let r2 = v2.get_mut(0..2).unwrap();
        assert_relative_eq!(r2[0].re, 1.0);
        assert_relative_eq!(r2[0].im, 2.0);
        assert_relative_eq!(r2[1].re, 3.0);
        assert_relative_eq!(r2[1].im, 4.0);
        r2[0].re = 5.0;
        r2[0].im = 6.0;
        assert_relative_eq!(v2[0].re, 5.0);
        assert_relative_eq!(v2[0].im, 6.0);
        assert_relative_eq!(v2[1].re, 3.0);
        assert_relative_eq!(v2[1].im, 4.0);
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
        let v3 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);
        let v4 = Vector::<Complex<f64>, 2>::from([Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let _: &[u32] = v1.as_slice();
        let _: &[f64] = v2.as_slice();
        let _: &[Complex<i32>] = v3.as_slice();
        let _: &[Complex<f64>] = v4.as_slice();
    }

    #[test]
    fn vector_method_as_mut_slice() {
        let mut v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let mut v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);
        let mut v3 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);
        let mut v4 =
            Vector::<Complex<f64>, 2>::from([Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let _: &mut [u32] = v1.as_mut_slice();
        let _: &mut [f64] = v2.as_mut_slice();
        let _: &mut [Complex<i32>] = v3.as_mut_slice();
        let _: &mut [Complex<f64>] = v4.as_mut_slice();
    }

    #[test]
    fn vector_method_as_array() {
        let v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);
        let v3 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);
        let v4 = Vector::<Complex<f64>, 2>::from([Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let _: &[u32; 3] = v1.as_array();
        let _: &[f64; 3] = v2.as_array();
        let _: &[Complex<i32>; 2] = v3.as_array();
        let _: &[Complex<f64>; 2] = v4.as_array();
    }

    #[test]
    fn vector_method_as_mut_array() {
        let mut v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let mut v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);
        let mut v3 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);
        let mut v4 =
            Vector::<Complex<f64>, 2>::from([Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let _: &mut [u32; 3] = v1.as_mut_array();
        let _: &mut [f64; 3] = v2.as_mut_array();
        let _: &mut [Complex<i32>; 2] = v3.as_mut_array();
        let _: &mut [Complex<f64>; 2] = v4.as_mut_array();
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
        let v3 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);
        let v4 = Vector::<Complex<f64>, 2>::from([Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let _: [u32; 3] = v1.to_array();
        let _: [f64; 3] = v2.to_array();
        let _: [Complex<i32>; 2] = v3.to_array();
        let _: [Complex<f64>; 2] = v4.to_array();
    }

    #[test]
    fn vector_method_to_vec() {
        let v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);
        let v3 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);
        let v4 = Vector::<Complex<f64>, 2>::from([Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let _: Vec<u32> = v1.to_vec();
        let _: Vec<f64> = v2.to_vec();
        let _: Vec<Complex<i32>> = v3.to_vec();
        let _: Vec<Complex<f64>> = v4.to_vec();
    }

    #[test]
    fn vector_to_num_cast() {
        // ~~~~~~~~~~~~~~~~~~
        // Float to int tests
        // ~~~~~~~~~~~~~~~~~~
        let v = Vector::<f32, 3>::from([1.12, 2.50, 3.99]);

        // Float to int type with default trunc
        let i64_trunc_v = v.to_num_cast(|x| x as i64);
        assert_eq!(i64_trunc_v, Vector::<i64, 3>::from([1, 2, 3]));

        // Float to int type with round
        let i64_round_v = v.to_num_cast(|x| x.round() as i64);
        assert_eq!(i64_round_v, Vector::<i64, 3>::from([1, 3, 4]));

        // Float to int type with ceil
        let i64_ceil_v = v.to_num_cast(|x| x.ceil() as i64);
        assert_eq!(i64_ceil_v, Vector::<i64, 3>::from([2, 3, 4]));

        // Float to int type with floor
        let i64_floor_v = v.to_num_cast(|x| x.floor() as i64);
        assert_eq!(i64_floor_v, Vector::<i64, 3>::from([1, 2, 3]));

        // ~~~~~~~~~~~~~~~~~~
        // Int to float tests
        // ~~~~~~~~~~~~~~~~~~
        let v = Vector::<i32, 3>::from([1, 2, 3]);

        let f64_v = v.to_num_cast(|x| x as f64);
        assert_eq!(f64_v, Vector::<f64, 3>::from([1.0, 2.0, 3.0]));

        let v = Vector::<i16, 3>::from([1, 2, 3]);

        let f32_v = v.to_num_cast(|x| x as f32);
        assert_eq!(f32_v, Vector::<f32, 3>::from([1.0, 2.0, 3.0]));
    }

    #[test]
    fn vector_to_num_cast_complex() {
        let v = Vector::<Complex<i32>, 2>::from([Complex::new(1, 0), Complex::new(3, 4)]);

        let v_i64: Vector<Complex<i64>, 2> =
            v.to_num_cast(|x| Complex { re: x.re as i64, im: x.im as i64 });

        assert!(v_i64.len() == 2);
        assert_eq!(v_i64[0].re, 1_i64);
        assert_eq!(v_i64[0].im, 0_i64);
        assert_eq!(v_i64[1].re, 3_i64);
        assert_eq!(v_i64[1].im, 4_i64);
    }

    #[test]
    fn vector_to_num_cast_safety() {
        let v = Vector::<f64, 2>::from([f64::MAX, 1.00]);
        let i8_v = v.to_num_cast(|x| x as i8);

        // Rust 1.45+ overflows are handled with a saturating cast.
        assert_eq!(i8_v[0], f64::MAX as i8);
        assert_eq!(i8_v[0], i8::MAX);
        assert_eq!(i8_v[0], 127);

        let v = Vector::<i64, 2>::from([i64::MIN, 1]);
        let u8_v_2 = v.to_num_cast(|x| x as u8);

        // signed to unsigned can change the value and eliminate the unary neg num sign
        // underflows saturate too
        assert_eq!(u8_v_2[0], i64::MIN as u8);
        assert_eq!(u8_v_2[0], u8::MIN);
        assert_eq!(u8_v_2[0], 0);

        let v = Vector::<f64, 2>::from([f64::NAN, 1.00]);
        let u8_v_3 = v.to_num_cast(|x| x as u8);

        // NAN casts to zero value
        assert_eq!(u8_v_3[0], f64::NAN as u8);
        assert_eq!(u8_v_3[0], 0);
    }

    // ================================
    //
    // len and is_empty method tests
    //
    // ================================
    #[test]
    fn vector_method_len_is_empty() {
        let v = Vector::<u32, 3>::from(&[1, 2, 3]);
        let v_empty = Vector::<u32, 0>::from(&[]);
        assert_eq!(v.len(), 3);
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
        // Note: this method *should* support valid dot products
        // with 2-length Vector *as* complex numbers, where item
        // 0 is the real part and item 1 is the imaginary part;
        // however it is not appropriate for use with Vector
        // of complex number types (i.e. each item in the Vector
        // is a complex number).
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
    // lerp method tests
    //
    // ================================
    #[test]
    fn vector_method_lerp() {
        let v1: Vector<f64, 2> = Vector::from([0.0, 0.0]);
        let v2: Vector<f64, 2> = Vector::from([10.0, 10.0]);

        // the value at weight 0 is the start vector
        assert_eq!(v1.lerp(&v2, 0.0).unwrap(), v1);
        // values move from start to end
        assert_eq!(v1.lerp(&v2, 0.25).unwrap(), Vector::from([2.5, 2.5]));
        assert_eq!(v1.lerp(&v2, 0.5).unwrap(), Vector::from([5.0, 5.0]));
        assert_eq!(v1.lerp(&v2, 0.75).unwrap(), Vector::from([7.5, 7.5]));
        // the value at weight 1 is the end vector
        assert_eq!(v1.lerp(&v2, 1.0).unwrap(), v2);

        // if start == end, the value at any weight will always be start (equivalent to end)
        assert_eq!(v2.lerp(&v2, 0.0).unwrap(), Vector::from([10.0, 10.0]));
        assert_eq!(v2.lerp(&v2, 0.1).unwrap(), Vector::from([10.0, 10.0]));
        assert_eq!(v2.lerp(&v2, 0.2).unwrap(), Vector::from([10.0, 10.0]));
        assert_eq!(v2.lerp(&v2, 0.3).unwrap(), Vector::from([10.0, 10.0]));
        assert_eq!(v2.lerp(&v2, 0.4).unwrap(), Vector::from([10.0, 10.0]));
        assert_eq!(v2.lerp(&v2, 0.5).unwrap(), Vector::from([10.0, 10.0]));
        assert_eq!(v2.lerp(&v2, 0.6).unwrap(), Vector::from([10.0, 10.0]));
        assert_eq!(v2.lerp(&v2, 0.7).unwrap(), Vector::from([10.0, 10.0]));
        assert_eq!(v2.lerp(&v2, 0.8).unwrap(), Vector::from([10.0, 10.0]));
        assert_eq!(v2.lerp(&v2, 1.0).unwrap(), Vector::from([10.0, 10.0]));

        let v1: Vector<f64, 2> = Vector::from([0.0, 0.0]);
        let v2: Vector<f64, 2> = Vector::from([-10.0, -10.0]);

        // the value at weight 0 is the start vector
        assert_eq!(v1.lerp(&v2, 0.0).unwrap(), v1);
        // values move from start to end
        assert_eq!(v1.lerp(&v2, 0.25).unwrap(), Vector::from([-2.5, -2.5]));
        assert_eq!(v1.lerp(&v2, 0.5).unwrap(), Vector::from([-5.0, -5.0]));
        assert_eq!(v1.lerp(&v2, 0.75).unwrap(), Vector::from([-7.5, -7.5]));
        // value at weight 1 is the end vector
        assert_eq!(v1.lerp(&v2, 1.0).unwrap(), v2);

        // if start == end, the value at any weight will always be start (equivalent to end)
        assert_eq!(v2.lerp(&v2, 0.0).unwrap(), Vector::from([-10.0, -10.0]));
        assert_eq!(v2.lerp(&v2, 0.1).unwrap(), Vector::from([-10.0, -10.0]));
        assert_eq!(v2.lerp(&v2, 0.2).unwrap(), Vector::from([-10.0, -10.0]));
        assert_eq!(v2.lerp(&v2, 0.3).unwrap(), Vector::from([-10.0, -10.0]));
        assert_eq!(v2.lerp(&v2, 0.4).unwrap(), Vector::from([-10.0, -10.0]));
        assert_eq!(v2.lerp(&v2, 0.5).unwrap(), Vector::from([-10.0, -10.0]));
        assert_eq!(v2.lerp(&v2, 0.6).unwrap(), Vector::from([-10.0, -10.0]));
        assert_eq!(v2.lerp(&v2, 0.7).unwrap(), Vector::from([-10.0, -10.0]));
        assert_eq!(v2.lerp(&v2, 0.8).unwrap(), Vector::from([-10.0, -10.0]));
        assert_eq!(v2.lerp(&v2, 1.0).unwrap(), Vector::from([-10.0, -10.0]));

        // higher dimension tests
        let v1: Vector<f64, 3> = Vector::from([-10.0, -10.0, -10.0]);
        let v2: Vector<f64, 3> = Vector::from([10.0, 10.0, 10.0]);

        assert_eq!(v1.lerp(&v2, 0.0).unwrap(), v1);
        assert_eq!(v1.lerp(&v2, 0.25).unwrap(), Vector::from([-5.0, -5.0, -5.0]));
        assert_eq!(v1.lerp(&v2, 0.5).unwrap(), Vector::from([0.0, 0.0, 0.0]));
        assert_eq!(v1.lerp(&v2, 0.75).unwrap(), Vector::from([5.0, 5.0, 5.0]));
        assert_eq!(v1.lerp(&v2, 1.0).unwrap(), v2);

        // NaN tests
        let v1: Vector<f64, 2> = Vector::from([0.0, 0.0]);
        let v2: Vector<f64, 2> = Vector::from([f64::NAN, 10.0]);
        let v3: Vector<f64, 2> = Vector::from([f64::NAN, f64::NAN]);

        // interpolation with NaN does not fail and interpolated value
        // is always NaN
        let v_res_1 = v1.lerp(&v2, 0.5).unwrap();
        assert!(v_res_1[0].is_nan());
        assert_relative_eq!(v_res_1[1], 5.0);

        let v_res_2 = v2.lerp(&v3, 0.5).unwrap();
        assert!(v_res_2[0].is_nan());
        assert!(v_res_2[1].is_nan());
    }

    #[test]
    fn vector_method_lerp_bounds_checks() {
        let v1: Vector<f64, 2> = Vector::from([0.0, 0.0]);
        let v2: Vector<f64, 2> = Vector::from([10.0, 10.0]);

        let err1 = v1.lerp(&v2, -0.01);
        let err2 = v1.lerp(&v2, 1.01);

        assert!(err1.is_err());
        assert!(matches!(err1, Err(VectorError::ValueError(_))));
        assert!(err2.is_err());
        assert!(matches!(err2, Err(VectorError::ValueError(_))));
    }

    #[test]
    fn vector_method_lerp_bounds_check_nan() {
        let v1: Vector<f64, 2> = Vector::from([0.0, 0.0]);
        let v2: Vector<f64, 2> = Vector::from([10.0, 10.0]);

        let v_res = v1.lerp(&v2, f64::NAN);
        assert!(v_res.is_err());
        assert!(matches!(v_res, Err(VectorError::ValueError(_))));
    }

    // ================================
    //
    // midpoint method tests
    //
    // ================================

    #[test]
    fn vector_method_midpoint() {
        let v1: Vector<f64, 2> = Vector::from([0.0, 0.0]);
        let v2: Vector<f64, 2> = Vector::from([10.0, 10.0]);

        assert_eq!(v1.midpoint(&v2), Vector::from([5.0, 5.0]));

        let v1: Vector<f64, 2> = Vector::from([0.0, 0.0]);
        let v2: Vector<f64, 2> = Vector::from([-10.0, -10.0]);

        assert_eq!(v1.midpoint(&v2), Vector::from([-5.0, -5.0]));

        let v1: Vector<f64, 3> = Vector::from([-10.0, -10.0, -10.0]);
        let v2: Vector<f64, 3> = Vector::from([10.0, 10.0, 10.0]);

        assert_eq!(v1.midpoint(&v2), Vector::zero());

        // NaN tests
        let v1: Vector<f64, 2> = Vector::from([0.0, 0.0]);
        let v2: Vector<f64, 2> = Vector::from([f64::NAN, 10.0]);
        let v3: Vector<f64, 2> = Vector::from([f64::NAN, f64::NAN]);

        // interpolation with NaN does not fail and midpoint value
        // is always NaN
        let v_res_1 = v1.midpoint(&v2);
        assert!(v_res_1[0].is_nan());
        assert_relative_eq!(v_res_1[1], 5.0);

        let v_res_2 = v2.midpoint(&v3);
        assert!(v_res_2[0].is_nan());
        assert!(v_res_2[1].is_nan());
    }

    // ===================================
    //
    // map_closure & map_func method tests
    //
    // ===================================

    #[test]
    fn vector_method_map_closure() {
        let v1 = Vector::<i32, 2>::from([-1, 2]);
        let v2 = Vector::<f64, 2>::from([-1.0, 2.0]);
        let v3 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);
        let v4 = Vector::<Complex<f64>, 2>::from([Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);

        let square_int = |x: i32| x.pow(2);
        let square_float = |x: f64| x.powi(2);
        let square_complex_i32 = |x: Complex<i32>| x.powi(2);
        let square_complex_f64 = |x: Complex<f64>| x.powi(2);

        let v1_squared = v1.map_closure(square_int);
        let v2_squared = v2.map_closure(square_float);
        let v3_squared = v3.map_closure(square_complex_i32);
        let v4_squared = v4.map_closure(square_complex_f64);

        assert_eq!(v1_squared, Vector::<i32, 2>::from([1, 4]));
        assert_eq!(v2_squared, Vector::<f64, 2>::from([1.0, 4.0]));

        assert_eq!(v1, Vector::<i32, 2>::from([-1, 2]));
        assert_eq!(v2, Vector::<f64, 2>::from([-1.0, 2.0]));

        // Expacted values for complex multiplication
        // real part = (ac - bd)
        // imaginary part = (ad + bc)
        assert_eq!(v3_squared[0].re, -3);
        assert_eq!(v3_squared[0].im, 4);
        assert_eq!(v3_squared[1].re, -7);
        assert_eq!(v3_squared[1].im, 24);

        assert_relative_eq!(v4_squared[0].re, -3.0);
        assert_relative_eq!(v4_squared[0].im, 4.0);
        assert_relative_eq!(v4_squared[1].re, -7.0);
        assert_relative_eq!(v4_squared[1].im, 24.0);
    }

    #[test]
    fn vector_method_mut_map_closure() {
        let mut v1 = Vector::<i32, 2>::from([-1, 2]);
        let mut v2 = Vector::<f64, 2>::from([-1.0, 2.0]);
        let mut v3 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);
        let mut v4 =
            Vector::<Complex<f64>, 2>::from([Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);

        let square_int = |x: i32| x.pow(2);
        let square_float = |x: f64| x.powi(2);
        let square_complex_i32 = |x: Complex<i32>| x.powi(2);
        let square_complex_f64 = |x: Complex<f64>| x.powi(2);

        v1.mut_map_closure(square_int);
        v2.mut_map_closure(square_float);
        v3.mut_map_closure(square_complex_i32);
        v4.mut_map_closure(square_complex_f64);

        assert_eq!(v1, Vector::<i32, 2>::from([1, 4]));
        assert_eq!(v2, Vector::<f64, 2>::from([1.0, 4.0]));

        // Expacted values for complex multiplication
        // real part = (ac - bd)
        // imaginary part = (ad + bc)
        assert_eq!(v3[0].re, -3);
        assert_eq!(v3[0].im, 4);
        assert_eq!(v3[1].re, -7);
        assert_eq!(v3[1].im, 24);

        assert_relative_eq!(v4[0].re, -3.0);
        assert_relative_eq!(v4[0].im, 4.0);
        assert_relative_eq!(v4[1].re, -7.0);
        assert_relative_eq!(v4[1].im, 24.0);
    }

    #[test]
    fn vector_method_map_fn() {
        let v1 = Vector::<i32, 2>::from([-1, 2]);
        let v2 = Vector::<f64, 2>::from([-1.0, 2.0]);
        let v3 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);
        let v4 = Vector::<Complex<f64>, 2>::from([Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);

        fn square_int(x: i32) -> i32 {
            x.pow(2)
        }

        fn square_float(x: f64) -> f64 {
            x.powi(2)
        }

        fn square_complex_int(x: Complex<i32>) -> Complex<i32> {
            x.powi(2)
        }

        fn square_complex_float(x: Complex<f64>) -> Complex<f64> {
            x.powi(2)
        }

        let v1_squared = v1.map_fn(square_int);
        let v2_squared = v2.map_fn(square_float);
        let v3_squared = v3.map_fn(square_complex_int);
        let v4_squared = v4.map_fn(square_complex_float);

        assert_eq!(v1_squared, Vector::<i32, 2>::from([1, 4]));
        assert_eq!(v2_squared, Vector::<f64, 2>::from([1.0, 4.0]));

        assert_eq!(v1, Vector::<i32, 2>::from([-1, 2]));
        assert_eq!(v2, Vector::<f64, 2>::from([-1.0, 2.0]));

        // Expacted values for complex multiplication
        // real part = (ac - bd)
        // imaginary part = (ad + bc)
        assert_eq!(v3_squared[0].re, -3);
        assert_eq!(v3_squared[0].im, 4);
        assert_eq!(v3_squared[1].re, -7);
        assert_eq!(v3_squared[1].im, 24);

        assert_relative_eq!(v4_squared[0].re, -3.0);
        assert_relative_eq!(v4_squared[0].im, 4.0);
        assert_relative_eq!(v4_squared[1].re, -7.0);
        assert_relative_eq!(v4_squared[1].im, 24.0);
    }

    #[test]
    fn vector_method_mut_map_fn() {
        let mut v1 = Vector::<i32, 2>::from([-1, 2]);
        let mut v2 = Vector::<f64, 2>::from([-1.0, 2.0]);
        let mut v3 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);
        let mut v4 =
            Vector::<Complex<f64>, 2>::from([Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);

        fn square_int(x: i32) -> i32 {
            x.pow(2)
        }

        fn square_float(x: f64) -> f64 {
            x.powi(2)
        }

        fn square_complex_int(x: Complex<i32>) -> Complex<i32> {
            x.powi(2)
        }

        fn square_complex_float(x: Complex<f64>) -> Complex<f64> {
            x.powi(2)
        }

        v1.mut_map_fn(square_int);
        v2.mut_map_fn(square_float);
        v3.mut_map_fn(square_complex_int);
        v4.mut_map_fn(square_complex_float);

        assert_eq!(v1, Vector::<i32, 2>::from([1, 4]));
        assert_eq!(v2, Vector::<f64, 2>::from([1.0, 4.0]));

        // Expacted values for complex multiplication
        // real part = (ac - bd)
        // imaginary part = (ad + bc)
        assert_eq!(v3[0].re, -3);
        assert_eq!(v3[0].im, 4);
        assert_eq!(v3[1].re, -7);
        assert_eq!(v3[1].im, 24);

        assert_relative_eq!(v4[0].re, -3.0);
        assert_relative_eq!(v4[0].im, 4.0);
        assert_relative_eq!(v4[1].re, -7.0);
        assert_relative_eq!(v4[1].im, 24.0);
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
        let v3 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);
        assert_eq!(v1[0], 1);
        assert_eq!(v1[1], 2);
        assert_relative_eq!(v2[0], 1.0);
        assert_relative_eq!(v2[1], 2.0);
        assert_eq!(v3[0].re, 1);
        assert_eq!(v3[0].im, 2);
        assert_eq!(v3[1].re, 3);
        assert_eq!(v3[1].im, 4);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn vector_trait_index_access_out_of_bounds() {
        let v = Vector::<u32, 10>::new();
        v[10];
    }

    #[test]
    fn vector_trait_index_slicing() {
        let v = Vector::<i32, 3>::from(&[1, 2, 3]);
        let v_slice = &v[..];

        assert_eq!(v_slice, [1, 2, 3]);

        let v_slice = &v[0..2];

        assert_eq!(v_slice, [1, 2]);
    }

    #[test]
    #[should_panic(expected = "range end index 11 out of range for slice of length 10")]
    fn vector_trait_index_slice_access_out_of_bounds() {
        let v = Vector::<u32, 10>::new();
        let _ = &v[0..11];
    }

    #[test]
    fn vector_trait_index_assignment() {
        let mut v1 = Vector::<u32, 2>::from(&[1, 2]);
        let mut v2 = Vector::<f64, 2>::from(&[1.0, 2.0]);
        let mut v3 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);
        v1[0] = 5;
        v1[1] = 6;
        v2[0] = 5.0;
        v2[1] = 6.0;
        v3[0].re = 4;
        v3[0].im = 5;
        assert_eq!(v1[0], 5);
        assert_eq!(v1[1], 6);
        assert_relative_eq!(v2[0], 5.0);
        assert_relative_eq!(v2[1], 6.0);
        assert_eq!(v3[0].re, 4);
        assert_eq!(v3[0].im, 5);
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
        let v3 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);

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

        let mut i: usize = 0;
        for x in &v3 {
            match i {
                0 => assert_eq!(x, &Complex::new(1, 2)),
                1 => assert_eq!(x, &Complex::new(3, 4)),
                _ => unimplemented!(),
            }
            i += 1;
        }
    }

    #[test]
    fn vector_trait_intoiterator_mut_ref() {
        let mut v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let mut v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);
        let mut v3 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);

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

        let mut i: usize = 0;
        for x in &mut v3 {
            match i {
                0 => assert_eq!(x, &mut Complex::new(1, 2)),
                1 => assert_eq!(x, &mut Complex::new(3, 4)),
                _ => unimplemented!(),
            }
            i += 1;
        }
    }

    #[test]
    fn vector_trait_intoiterator_owned() {
        let v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);
        let v3 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);

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

        let mut i: usize = 0;
        for x in v3 {
            match i {
                0 => assert_eq!(x, Complex::new(1, 2)),
                1 => assert_eq!(x, Complex::new(3, 4)),
                _ => unimplemented!(),
            }
            i += 1;
        }
    }

    #[test]
    fn vector_method_iter() {
        let v1 = Vector::<u32, 3>::from(&[1, 2, 3]);
        let v2 = Vector::<f64, 3>::from(&[1.0, 2.0, 3.0]);
        let v3 = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);

        let mut v1_iter = v1.iter();
        let mut v2_iter = v2.iter();
        let mut v3_iter = v3.iter();

        assert_eq!(v1_iter.next().unwrap(), &1);
        assert_eq!(v1_iter.next().unwrap(), &2);
        assert_eq!(v1_iter.next().unwrap(), &3);
        assert_eq!(v1_iter.next(), None);

        assert_relative_eq!(v2_iter.next().unwrap(), &1.0);
        assert_relative_eq!(v2_iter.next().unwrap(), &2.0);
        assert_relative_eq!(v2_iter.next().unwrap(), &3.0);
        assert_eq!(v2_iter.next(), None);

        assert_eq!(v3_iter.next().unwrap(), &Complex::new(1, 2));
        assert_eq!(v3_iter.next().unwrap(), &Complex::new(3, 4));
        assert_eq!(v3_iter.next(), None);
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

        // empty iterable tests
        let v5: Vector<i32, 3> = [].into_iter().collect();
        assert_eq!(v5.components.len(), 3);
        assert_eq!(v5[0], 0 as i32);
        assert_eq!(v5[1], 0 as i32);
        assert_eq!(v5[2], 0 as i32);

        let v6: Vector<Complex<i32>, 2> = [].into_iter().collect();
        assert_eq!(v6.components.len(), 2);
        assert_eq!(v6[0], Complex::new(0 as i32, 0 as i32));
        assert_eq!(v6[1], Complex::new(0 as i32, 0 as i32));
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

    #[test]
    fn vector_trait_partial_eq_complex_i32() {
        // Note: Complex types with integer real and imaginary parts are only tested with i32 data types
        let v1 = Vector::<Complex<i32>, 2>::from([Complex::new(1, -2), Complex::new(3, 4)]);
        let v2 = Vector::<Complex<i32>, 2>::from([Complex::new(1, -2), Complex::new(3, 4)]);
        let v_eq = Vector::<Complex<i32>, 2>::from([Complex::new(1, -2), Complex::new(3, 4)]);
        let v_diff = Vector::<Complex<i32>, 2>::from([Complex::new(1, 2), Complex::new(3, 4)]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff);
    }

    #[test]
    fn vector_trait_partial_eq_complex_f64() {
        let v1 = Vector::<Complex<f64>, 2>::from([Complex::new(-1.1, 2.2), Complex::new(2.2, 3.3)]);
        let v2 = Vector::<Complex<f64>, 2>::from([Complex::new(-1.1, 2.2), Complex::new(2.2, 3.3)]);
        let v_eq =
            Vector::<Complex<f64>, 2>::from([Complex::new(-1.1, 2.2), Complex::new(2.2, 3.3)]);
        let v_diff_re =
            Vector::<Complex<f64>, 2>::from([Complex::new(-1.1, 2.2), Complex::new(3.3, 3.3)]);
        let v_diff_im =
            Vector::<Complex<f64>, 2>::from([Complex::new(-1.1, 2.2), Complex::new(2.2, 4.4)]);
        let v_diff_re_im =
            Vector::<Complex<f64>, 2>::from([Complex::new(-1.1, 2.2), Complex::new(3.3, 4.4)]);

        let v_close = Vector::<Complex<f64>, 2>::from([
            Complex::new(-1.1 + (f64::EPSILON * 2.), 2.2),
            Complex::new(2.2, 3.3),
        ]);

        let v_zero = Vector::<Complex<f64>, 2>::zero();
        let v_zero_eq = Vector::<Complex<f64>, 2>::zero();
        let v_zero_neg_eq =
            Vector::<Complex<f64>, 2>::from([Complex::new(-0.0, -0.0), Complex::new(-0.0, -0.0)]);

        let v_nan =
            Vector::<Complex<f64>, 2>::from([Complex::new(f64::NAN, 1.0), Complex::new(2.0, 3.0)]);
        let v_nan_diff =
            Vector::<Complex<f64>, 2>::from([Complex::new(f64::NAN, 1.0), Complex::new(2.0, 3.0)]);

        let v_inf_pos = Vector::<Complex<f64>, 2>::from([
            Complex::new(f64::INFINITY, 0.0),
            Complex::new(3.0, 4.0),
        ]);
        let v_inf_pos_eq = Vector::<Complex<f64>, 2>::from([
            Complex::new(f64::INFINITY, 0.0),
            Complex::new(3.0, 4.0),
        ]);
        let v_inf_neg = Vector::<Complex<f64>, 2>::from([
            Complex::new(f64::NEG_INFINITY, 0.0),
            Complex::new(3.0, 4.0),
        ]);
        let v_inf_neg_eq = Vector::<Complex<f64>, 2>::from([
            Complex::new(f64::NEG_INFINITY, 0.0),
            Complex::new(3.0, 4.0),
        ]);

        assert!(v1 == v_eq);
        assert!(v_eq == v1); // symmetry
        assert!(v2 == v_eq);
        assert!(v1 == v2); // transitivity
        assert!(v1 != v_diff_re);
        assert!(v1 != v_diff_im);
        assert!(v1 != v_diff_re_im);
        assert!(v1 != v_close);
        assert!(v_zero == v_zero_eq);
        assert!(v_zero == v_zero_neg_eq); // zero and neg zero are defined as equivalent
        assert!(v_nan != v_nan_diff); // NaN comparisons are defined as different
        assert!(v_inf_pos == v_inf_pos_eq); // postive infinity comparisons are defined as equivalent
        assert!(v_inf_neg == v_inf_neg_eq); // negative infinity comparisons are defined as equivalent
    }

    // ======================================================
    //
    // approx crate float approximate equivalence trait tests
    // AbsDiffEq, RelativeEq, UlpsEq traits
    //
    // ======================================================

    #[test]
    fn vector_trait_absdiffeq_f64() {
        let v1 = Vector::<f64, 3>::from([-1.1, 2.2, 3.3]);
        let v2 = Vector::<f64, 3>::from([-0.1 - 1.0, 2.2, 3.3]);
        let v_eq = Vector::<f64, 3>::from([-1.1, 2.2, 3.3]);
        let v_diff = Vector::<f64, 3>::from([-1.1, 2.2, 4.4]);
        let v_close = Vector::<f64, 3>::from([-1.1 + (f64::EPSILON), 2.2, 3.3]);
        let v_not_close_enough = Vector::<f64, 3>::from([-1.1 + (f64::EPSILON * 2.), 2.2, 3.3]);

        let v_zero = Vector::<f64, 3>::zero();
        let v_zero_eq = Vector::<f64, 3>::zero();
        let v_zero_neg_eq = Vector::<f64, 3>::from([-0.0, -0.0, -0.0]);
        let v_zero_close = Vector::<f64, 3>::from([0.0 + f64::EPSILON, 0.0, 0.0]);

        let v_nan = Vector::<f64, 3>::from([f64::NAN, 0.0, 0.0]);
        let v_nan_diff = Vector::<f64, 3>::from([f64::NAN, 0.0, 0.0]);

        let v_inf_pos = Vector::<f64, 3>::from([f64::INFINITY, 0.0, 0.0]);
        let v_inf_pos_eq = Vector::<f64, 3>::from([f64::INFINITY, 0.0, 0.0]);
        let v_inf_neg = Vector::<f64, 3>::from([f64::NEG_INFINITY, 0.0, 0.0]);
        let v_inf_neg_eq = Vector::<f64, 3>::from([f64::NEG_INFINITY, 0.0, 0.0]);

        assert!(v1.abs_diff_eq(&v_eq, f64::default_epsilon()));
        assert!(v_eq.abs_diff_eq(&v1, f64::default_epsilon()));
        assert!(v2.abs_diff_eq(&v_eq, f64::default_epsilon()));
        assert!(v1.abs_diff_eq(&v2, f64::default_epsilon()));
        assert!(!v1.abs_diff_eq(&v_diff, f64::default_epsilon()));
        assert!(v1.abs_diff_eq(&v_close, f64::default_epsilon()));
        // when difference is epsilon multipled by a factor of 2,
        // we are outside of the absolute diff bounds
        assert!(!v1.abs_diff_eq(&v_not_close_enough, f64::default_epsilon()));
        // unless we adjust the acceptable epsilon threshold, now we consider it eq
        assert!(v1.abs_diff_eq(&v_not_close_enough, f64::default_epsilon() * 2.));

        assert!(v_zero.abs_diff_eq(&v_zero_eq, f64::default_epsilon()));
        assert!(v_zero.abs_diff_eq(&v_zero_neg_eq, f64::default_epsilon()));
        // we are within epsilon of zero, consider this eq
        assert!(v_zero.abs_diff_eq(&v_zero_close, f64::default_epsilon()));
        // but is defined as ne when we decrease epsilon value
        assert!(!v_zero.abs_diff_eq(&v_zero_close, f64::default_epsilon() / 2.));

        assert!(!v_nan.abs_diff_eq(&v_nan_diff, f64::default_epsilon()));

        // note: different result than with relative eq comparison when we test
        // infinite values
        assert!(!v_inf_pos.abs_diff_eq(&v_inf_pos_eq, f64::default_epsilon()));
        assert!(!v_inf_neg.abs_diff_eq(&v_inf_neg_eq, f64::default_epsilon()));
        assert!(!v_inf_pos.abs_diff_eq(&v_inf_neg, f64::default_epsilon()));
    }

    #[test]
    fn vector_trait_absdiffeq_complex_f64() {
        let v1 = Vector::<Complex<f64>, 2>::from([Complex::new(-1.1, 2.2), Complex::new(3.3, 4.4)]);
        let v2 = Vector::<Complex<f64>, 2>::from([Complex::new(-1.1, 2.2), Complex::new(3.3, 4.4)]);
        let v_eq =
            Vector::<Complex<f64>, 2>::from([Complex::new(-1.1, 2.2), Complex::new(3.3, 4.4)]);
        let v_diff_re =
            Vector::<Complex<f64>, 2>::from([Complex::new(-1.3, 2.2), Complex::new(3.3, 4.4)]);
        let v_diff_im =
            Vector::<Complex<f64>, 2>::from([Complex::new(-1.1, 2.4), Complex::new(3.3, 4.4)]);
        let v_close = Vector::<Complex<f64>, 2>::from([
            Complex::new(-1.1 + (f64::EPSILON), 2.2),
            Complex::new(3.3, 4.4),
        ]);
        let v_not_close_enough_re = Vector::<Complex<f64>, 2>::from([
            Complex::new(-1.1 + (f64::EPSILON * 2.), 2.2),
            Complex::new(3.3, 4.4),
        ]);

        let v_not_close_enough_im = Vector::<Complex<f64>, 2>::from([
            Complex::new(-1.1, 2.2 + (f64::EPSILON * 2.)),
            Complex::new(3.3, 4.4),
        ]);

        let v_zero = Vector::<Complex<f64>, 2>::zero();
        let v_zero_eq = Vector::<Complex<f64>, 2>::zero();
        let v_zero_neg_eq =
            Vector::<Complex<f64>, 2>::from([Complex::new(-0.0, -0.0), Complex::new(-0.0, -0.0)]);
        let v_zero_close = Vector::<Complex<f64>, 2>::from([
            Complex::new(0.0 + f64::EPSILON, -0.0),
            Complex::new(-0.0, -0.0),
        ]);

        let v_nan =
            Vector::<Complex<f64>, 2>::from([Complex::new(f64::NAN, 0.0), Complex::new(1.0, 2.0)]);
        let v_nan_diff =
            Vector::<Complex<f64>, 2>::from([Complex::new(f64::NAN, 0.0), Complex::new(1.0, 2.0)]);

        let v_inf_pos = Vector::<Complex<f64>, 2>::from([
            Complex::new(f64::INFINITY, 0.0),
            Complex::new(0.0, f64::INFINITY),
        ]);
        let v_inf_pos_eq = Vector::<Complex<f64>, 2>::from([
            Complex::new(f64::INFINITY, 0.0),
            Complex::new(0.0, f64::INFINITY),
        ]);
        let v_inf_neg = Vector::<Complex<f64>, 2>::from([
            Complex::new(f64::NEG_INFINITY, 0.0),
            Complex::new(0.0, f64::NEG_INFINITY),
        ]);
        let v_inf_neg_eq = Vector::<Complex<f64>, 2>::from([
            Complex::new(f64::NEG_INFINITY, 0.0),
            Complex::new(0.0, f64::NEG_INFINITY),
        ]);

        assert!(v1.abs_diff_eq(&v_eq, f64::default_epsilon()));
        assert!(v_eq.abs_diff_eq(&v1, f64::default_epsilon()));
        assert!(v2.abs_diff_eq(&v_eq, f64::default_epsilon()));
        assert!(v1.abs_diff_eq(&v2, f64::default_epsilon()));
        assert!(!v1.abs_diff_eq(&v_diff_re, f64::default_epsilon()));
        assert!(!v1.abs_diff_eq(&v_diff_im, f64::default_epsilon()));
        assert!(v1.abs_diff_eq(&v_close, f64::default_epsilon()));
        // when difference is epsilon multipled by a factor of 2,
        // we are outside of the absolute diff bounds
        assert!(!v1.abs_diff_eq(&v_not_close_enough_re, f64::default_epsilon()));
        assert!(!v1.abs_diff_eq(&v_not_close_enough_im, f64::default_epsilon()));
        // unless we adjust the acceptable epsilon threshold, now we consider it eq
        assert!(v1.abs_diff_eq(&v_not_close_enough_re, f64::default_epsilon() * 2.));
        assert!(v1.abs_diff_eq(&v_not_close_enough_im, f64::default_epsilon() * 2.));

        assert!(v_zero.abs_diff_eq(&v_zero_eq, f64::default_epsilon()));
        assert!(v_zero.abs_diff_eq(&v_zero_neg_eq, f64::default_epsilon()));
        // we are within epsilon of zero, consider this eq
        assert!(v_zero.abs_diff_eq(&v_zero_close, f64::default_epsilon()));
        // but is defined as ne when we decrease epsilon value
        assert!(!v_zero.abs_diff_eq(&v_zero_close, f64::default_epsilon() / 2.));

        assert!(!v_nan.abs_diff_eq(&v_nan_diff, f64::default_epsilon()));

        // note: different result than with relative eq comparison when we test
        // infinite values
        assert!(!v_inf_pos.abs_diff_eq(&v_inf_pos_eq, f64::default_epsilon()));
        assert!(!v_inf_neg.abs_diff_eq(&v_inf_neg_eq, f64::default_epsilon()));
        assert!(!v_inf_pos.abs_diff_eq(&v_inf_neg, f64::default_epsilon()));
    }

    #[test]
    fn vector_trait_relativeeq_f64() {
        let v1 = Vector::<f64, 3>::from([-1.1, 2.2, 3.3]);
        let v2 = Vector::<f64, 3>::from([-0.1 - 1.0, 2.2, 3.3]);
        let v_eq = Vector::<f64, 3>::from([-1.1, 2.2, 3.3]);
        let v_diff = Vector::<f64, 3>::from([-1.1, 2.2, 4.4]);
        let v_close = Vector::<f64, 3>::from([-1.1 + (f64::EPSILON), 2.2, 3.3]);
        let v_not_close_enough = Vector::<f64, 3>::from([-1.1 + (f64::EPSILON * 2.), 2.2, 3.3]);

        let v_zero = Vector::<f64, 3>::zero();
        let v_zero_eq = Vector::<f64, 3>::zero();
        let v_zero_neg_eq = Vector::<f64, 3>::from([-0.0, -0.0, -0.0]);
        let v_zero_close = Vector::<f64, 3>::from([0.0 + f64::EPSILON, 0.0, 0.0]);

        let v_nan = Vector::<f64, 3>::from([f64::NAN, 0.0, 0.0]);
        let v_nan_diff = Vector::<f64, 3>::from([f64::NAN, 0.0, 0.0]);

        let v_inf_pos = Vector::<f64, 3>::from([f64::INFINITY, 0.0, 0.0]);
        let v_inf_pos_eq = Vector::<f64, 3>::from([f64::INFINITY, 0.0, 0.0]);
        let v_inf_neg = Vector::<f64, 3>::from([f64::NEG_INFINITY, 0.0, 0.0]);
        let v_inf_neg_eq = Vector::<f64, 3>::from([f64::NEG_INFINITY, 0.0, 0.0]);

        assert!(v1.relative_eq(&v_eq, f64::default_epsilon(), f64::default_epsilon()));
        assert!(v_eq.relative_eq(&v1, f64::default_epsilon(), f64::default_epsilon()));
        assert!(v2.relative_eq(&v_eq, f64::default_epsilon(), f64::default_epsilon()));
        assert!(v1.relative_eq(&v2, f64::default_epsilon(), f64::default_epsilon()));
        assert!(!v1.relative_eq(&v_diff, f64::default_epsilon(), f64::default_epsilon()));
        assert!(v1.relative_eq(&v_close, f64::default_epsilon(), f64::default_epsilon()));
        // when difference is epsilon multiplied by a factor of 2,
        // we are outside of the relative diff bounds
        assert!(!v1.relative_eq(
            &v_not_close_enough,
            f64::default_epsilon(),
            f64::default_epsilon()
        ));
        // unless we adjust the acceptable max relative diff
        assert!(v1.relative_eq(
            &v_not_close_enough,
            f64::default_epsilon(),
            f64::default_epsilon() * 2.
        ));

        // near zero use the absolute diff vs. epsilon comparisons
        assert!(v_zero.relative_eq(&v_zero_eq, f64::default_epsilon(), f64::default_epsilon()));
        assert!(v_zero.relative_eq(&v_zero_neg_eq, f64::default_epsilon(), f64::default_epsilon()));
        // considered eq when within epsilon of zero
        assert!(v_zero.relative_eq(&v_zero_close, f64::default_epsilon(), f64::default_epsilon()));
        // considered ne when we decrease the epsilon definition below the diff from zero
        assert!(!v_zero.relative_eq(
            &v_zero_close,
            f64::default_epsilon() / 2.,
            f64::default_epsilon()
        ));

        assert!(!v_nan.relative_eq(&v_nan_diff, f64::default_epsilon(), f64::default_epsilon()));

        assert!(v_inf_pos.relative_eq(
            &v_inf_pos_eq,
            f64::default_epsilon(),
            f64::default_epsilon()
        ));
        assert!(v_inf_neg.relative_eq(
            &v_inf_neg_eq,
            f64::default_epsilon(),
            f64::default_epsilon()
        ));
        assert!(!v_inf_pos.relative_eq(&v_inf_neg, f64::default_epsilon(), f64::default_epsilon()));
    }

    #[test]
    fn vector_trait_relativeeq_complex_f64() {
        let v1 = Vector::<Complex<f64>, 2>::from([Complex::new(-1.1, 1.1), Complex::new(3.3, 4.4)]);
        let v2 = Vector::<Complex<f64>, 2>::from([Complex::new(-1.1, 1.1), Complex::new(3.3, 4.4)]);
        let v_eq =
            Vector::<Complex<f64>, 2>::from([Complex::new(-1.1, 1.1), Complex::new(3.3, 4.4)]);
        let v_diff_re =
            Vector::<Complex<f64>, 2>::from([Complex::new(-1.3, 1.1), Complex::new(3.3, 4.4)]);
        let v_diff_im =
            Vector::<Complex<f64>, 2>::from([Complex::new(-1.1, 2.4), Complex::new(3.3, 4.4)]);
        let v_close = Vector::<Complex<f64>, 2>::from([
            Complex::new(-1.1 + (f64::EPSILON), 1.1),
            Complex::new(3.3, 4.4),
        ]);
        let v_not_close_enough_re = Vector::<Complex<f64>, 2>::from([
            Complex::new(-1.1 + (f64::EPSILON * 2.), 1.1),
            Complex::new(3.3, 4.4),
        ]);

        let v_not_close_enough_im = Vector::<Complex<f64>, 2>::from([
            Complex::new(-1.1, 1.1 - (f64::EPSILON * 2.)),
            Complex::new(3.3, 4.4),
        ]);

        let v_zero = Vector::<Complex<f64>, 2>::zero();
        let v_zero_eq = Vector::<Complex<f64>, 2>::zero();
        let v_zero_neg_eq =
            Vector::<Complex<f64>, 2>::from([Complex::new(-0.0, -0.0), Complex::new(-0.0, -0.0)]);
        let v_zero_close = Vector::<Complex<f64>, 2>::from([
            Complex::new(0.0 + f64::EPSILON, -0.0),
            Complex::new(-0.0, -0.0),
        ]);

        let v_nan =
            Vector::<Complex<f64>, 2>::from([Complex::new(f64::NAN, 0.0), Complex::new(1.0, 2.0)]);
        let v_nan_diff =
            Vector::<Complex<f64>, 2>::from([Complex::new(f64::NAN, 0.0), Complex::new(1.0, 2.0)]);

        let v_inf_pos = Vector::<Complex<f64>, 2>::from([
            Complex::new(f64::INFINITY, 0.0),
            Complex::new(0.0, f64::INFINITY),
        ]);
        let v_inf_pos_eq = Vector::<Complex<f64>, 2>::from([
            Complex::new(f64::INFINITY, 0.0),
            Complex::new(0.0, f64::INFINITY),
        ]);
        let v_inf_neg = Vector::<Complex<f64>, 2>::from([
            Complex::new(f64::NEG_INFINITY, 0.0),
            Complex::new(0.0, f64::NEG_INFINITY),
        ]);
        let v_inf_neg_eq = Vector::<Complex<f64>, 2>::from([
            Complex::new(f64::NEG_INFINITY, 0.0),
            Complex::new(0.0, f64::NEG_INFINITY),
        ]);

        assert!(v1.relative_eq(&v_eq, f64::default_epsilon(), f64::default_epsilon()));
        assert!(v_eq.relative_eq(&v1, f64::default_epsilon(), f64::default_epsilon()));
        assert!(v2.relative_eq(&v_eq, f64::default_epsilon(), f64::default_epsilon()));
        assert!(v1.relative_eq(&v2, f64::default_epsilon(), f64::default_epsilon()));
        assert!(!v1.relative_eq(&v_diff_re, f64::default_epsilon(), f64::default_epsilon()));
        assert!(!v1.relative_eq(&v_diff_im, f64::default_epsilon(), f64::default_epsilon()));
        assert!(v1.relative_eq(&v_close, f64::default_epsilon(), f64::default_epsilon()));
        // when difference is epsilon multiplied by a factor of 2,
        // we are outside of the relative diff bounds
        assert!(!v1.relative_eq(
            &v_not_close_enough_re,
            f64::default_epsilon(),
            f64::default_epsilon()
        ));
        assert!(!v1.relative_eq(
            &v_not_close_enough_im,
            f64::default_epsilon(),
            f64::default_epsilon()
        ));
        // unless we adjust the acceptable max relative diff
        assert!(v1.relative_eq(
            &v_not_close_enough_re,
            f64::default_epsilon(),
            f64::default_epsilon() * 2.
        ));
        assert!(v1.relative_eq(
            &v_not_close_enough_im,
            f64::default_epsilon(),
            f64::default_epsilon() * 2.
        ));

        // near zero use the absolute diff vs. epsilon comparisons
        assert!(v_zero.relative_eq(&v_zero_eq, f64::default_epsilon(), f64::default_epsilon()));
        assert!(v_zero.relative_eq(&v_zero_neg_eq, f64::default_epsilon(), f64::default_epsilon()));
        // considered eq when within epsilon of zero
        assert!(v_zero.relative_eq(&v_zero_close, f64::default_epsilon(), f64::default_epsilon()));
        // considered ne when we decrease the epsilon definition below the diff from zero
        assert!(!v_zero.relative_eq(
            &v_zero_close,
            f64::default_epsilon() / 2.,
            f64::default_epsilon()
        ));

        assert!(!v_nan.relative_eq(&v_nan_diff, f64::default_epsilon(), f64::default_epsilon()));

        assert!(v_inf_pos.relative_eq(
            &v_inf_pos_eq,
            f64::default_epsilon(),
            f64::default_epsilon()
        ));
        assert!(v_inf_neg.relative_eq(
            &v_inf_neg_eq,
            f64::default_epsilon(),
            f64::default_epsilon()
        ));
        assert!(!v_inf_pos.relative_eq(&v_inf_neg, f64::default_epsilon(), f64::default_epsilon()));
    }

    #[test]
    fn vector_trait_ulpseq_f64() {
        let v1 = Vector::<f64, 3>::from([-1.1, 2.2, 3.3]);
        let v2 = Vector::<f64, 3>::from([-0.1 - 1.0, 2.2, 3.3]);
        let v_eq = Vector::<f64, 3>::from([-1.1, 2.2, 3.3]);
        let v_diff = Vector::<f64, 3>::from([-1.1, 2.2, 4.4]);
        let v_close = Vector::<f64, 3>::from([-1.1 + (f64::EPSILON), 2.2, 3.3]);
        let v_not_close_enough = Vector::<f64, 3>::from([-1.1 + (f64::EPSILON * 2.), 2.2, 3.3]);

        let v_zero = Vector::<f64, 3>::zero();
        let v_zero_eq = Vector::<f64, 3>::zero();
        let v_zero_neg_eq = Vector::<f64, 3>::from([-0.0, -0.0, -0.0]);
        let v_zero_close = Vector::<f64, 3>::from([0.0 + f64::EPSILON, 0.0, 0.0]);

        let v_nan = Vector::<f64, 3>::from([f64::NAN, 0.0, 0.0]);
        let v_nan_diff = Vector::<f64, 3>::from([f64::NAN, 0.0, 0.0]);

        let v_inf_pos = Vector::<f64, 3>::from([f64::INFINITY, 0.0, 0.0]);
        let v_inf_pos_eq = Vector::<f64, 3>::from([f64::INFINITY, 0.0, 0.0]);
        let v_inf_neg = Vector::<f64, 3>::from([f64::NEG_INFINITY, 0.0, 0.0]);
        let v_inf_neg_eq = Vector::<f64, 3>::from([f64::NEG_INFINITY, 0.0, 0.0]);

        assert!(v1.ulps_eq(&v_eq, f64::default_epsilon(), 1));
        assert!(v_eq.ulps_eq(&v1, f64::default_epsilon(), 1));
        assert!(v2.ulps_eq(&v_eq, f64::default_epsilon(), 1));
        assert!(v1.ulps_eq(&v2, f64::default_epsilon(), 1));
        assert!(!v1.ulps_eq(&v_diff, f64::default_epsilon(), 1));
        assert!(v1.ulps_eq(&v_close, f64::default_epsilon(), 1));
        // when difference is epsilon multipled by a factor of 2,
        // we are outside of the diff bounds defined by max ulps = 1
        assert!(!v1.ulps_eq(&v_not_close_enough, f64::default_epsilon(), 1));
        // unless we adjust the acceptable ulps threshold
        assert!(v1.ulps_eq(&v_not_close_enough, f64::default_epsilon(), 2));

        // near zero, use the epsilon value and absolute diff vs. epsilon comparisons
        assert!(v_zero.ulps_eq(&v_zero_eq, f64::default_epsilon(), 1));
        assert!(v_zero.ulps_eq(&v_zero_neg_eq, f64::default_epsilon(), 1));
        // we are within epsilon of zero, consider this eq
        assert!(v_zero.ulps_eq(&v_zero_close, f64::default_epsilon(), 1));
        // but is defined as ne when we decrease epsilon value
        assert!(!v_zero.ulps_eq(&v_zero_close, f64::default_epsilon() / 2., 1));

        assert!(!v_nan.ulps_eq(&v_nan_diff, f64::default_epsilon(), 1));

        assert!(v_inf_pos.ulps_eq(&v_inf_pos_eq, f64::default_epsilon(), 1));
        assert!(v_inf_neg.ulps_eq(&v_inf_neg_eq, f64::default_epsilon(), 1));
        assert!(!v_inf_pos.ulps_eq(&v_inf_neg, f64::default_epsilon(), 1));
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
