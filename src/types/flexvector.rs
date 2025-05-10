//! FlexVector type.

use crate::{
    impl_vector_binop, impl_vector_binop_assign, impl_vector_scalar_div_op,
    impl_vector_scalar_div_op_assign, impl_vector_scalar_op, impl_vector_scalar_op_assign,
    impl_vector_unary_op, types::traits::VectorBase, types::traits::VectorOps,
    types::traits::VectorOpsComplex, types::traits::VectorOpsFloat,
};

use crate::errors::VectorError;

use num::Complex;
use num::Num;

/// A dynamic, heap-allocated vector type for n-dimensional real and complex scalar data.
///
/// The length of the vector is determined at runtime and stored on the heap.
/// This type is analogous to `Vector<T, N>` but supports dynamic sizing.
#[derive(Clone, Debug)]
pub struct FlexVector<T>
where
    T: Num + Clone + Sync + Send,
{
    /// Ordered n-dimensional scalar values.
    pub components: Vec<T>,
}

// ================================
//
// Constructors
//
// ================================
impl<T> FlexVector<T>
where
    T: Num + Clone + Default + Sync + Send,
{
    /// Creates a new, empty FlexVector.
    #[inline]
    pub fn new() -> Self {
        Self { components: Vec::new() }
    }

    /// Creates a new FlexVector with a pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self { components: Vec::with_capacity(capacity) }
    }

    /// Returns a new FlexVector of the given length, filled with zeros.
    #[inline]
    pub fn zero(len: usize) -> Self
    where
        T: num::Zero,
    {
        Self { components: vec![T::zero(); len] }
    }

    /// Returns a new FlexVector of the given length, filled with ones.
    #[inline]
    pub fn one(len: usize) -> Self
    where
        T: num::One,
    {
        Self { components: vec![T::one(); len] }
    }

    /// Returns a new FlexVector of the given length, filled with the given value.
    #[inline]
    pub fn filled(len: usize, value: T) -> Self {
        Self { components: vec![value; len] }
    }

    /// Creates a new FlexVector from a slice.
    #[inline]
    pub fn from_slice(slice: &[T]) -> Self {
        Self { components: slice.to_vec() }
    }

    /// Creates a FlexVector from a Vec.
    #[inline]
    pub fn from_vec(vec: Vec<T>) -> Self {
        Self { components: vec }
    }
}

// ================================
//
// Default trait impl
//
// ================================
impl<T> Default for FlexVector<T>
where
    T: num::Num + Clone + Default + Sync + Send,
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ================================
//
// Display trait impl
//
// ================================
impl<T> std::fmt::Display for FlexVector<T>
where
    T: num::Num + Clone + Sync + Send + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.components)
    }
}

// ================================
//
// VectorBase trait impl
//
// ================================
impl<T> VectorBase<T> for FlexVector<T>
where
    T: num::Num + Clone + Sync + Send,
{
    /// Returns an immutable slice of the FlexVector's components.
    #[inline]
    fn as_slice(&self) -> &[T] {
        &self.components
    }

    /// Returns a mutable slice of the FlexVector's components.
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.components[..]
    }
}

// ================================
//
// VectorOps trait impl
//
// ================================
// TODO: add tests
impl<T> VectorOps<T> for FlexVector<T>
where
    T: num::Num + Clone + Sync + Send,
{
    type Output = Vec<T>;

    #[inline]
    fn translate(&self, other: &Self) -> Self::Output
    where
        T: num::Num + Clone,
    {
        self.components.iter().zip(&other.components).map(|(a, b)| a.clone() + b.clone()).collect()
    }

    #[inline]
    fn scale(&self, scalar: T) -> Self::Output
    where
        T: num::Num + Copy,
        Self::Output: std::iter::FromIterator<T>,
    {
        self.as_slice().iter().map(|a| *a * scalar).collect()
    }

    #[inline]
    fn negate(&self) -> Self::Output
    where
        T: std::ops::Neg<Output = T> + Clone,
        Self::Output: std::iter::FromIterator<T>,
    {
        self.as_slice().iter().map(|a| -a.clone()).collect()
    }
}

// ================================
//
// VectorOpsFloat trait impl
//
// ================================
// TODO: add tests
impl<T> VectorOpsFloat<T> for FlexVector<T>
where
    T: num::Float + Clone + PartialOrd + std::iter::Sum<T> + Sync + Send,
{
    type Output = Vec<T>;

    #[inline]
    fn normalize(&self) -> Result<Self::Output, VectorError> {
        let n = self.norm();
        if n == T::zero() {
            return Err(VectorError::ZeroVectorError("Cannot normalize a zero vector".to_string()));
        }
        Ok(self.components.iter().map(|a| *a / n).collect())
    }

    /// Returns a new vector with the same direction and the given magnitude.
    #[inline]
    fn normalize_to(&self, magnitude: T) -> Result<Self::Output, VectorError>
    where
        T: num::Float + Clone + std::iter::Sum<T>,
        Self::Output: std::iter::FromIterator<T>,
    {
        let n = self.norm();
        if n == T::zero() {
            return Err(VectorError::ZeroVectorError("Cannot normalize a zero vector".to_string()));
        }
        let scale = magnitude / n;
        Ok(self.as_slice().iter().map(|a| *a * scale).collect())
    }

    #[inline]
    fn lerp(&self, end: &Self, weight: T) -> Result<Self::Output, VectorError>
    where
        T: num::Float + Clone + PartialOrd,
    {
        if weight < T::zero() || weight > T::one() {
            return Err(VectorError::OutOfRangeError("weight must be in [0, 1]".to_string()));
        }
        let w = weight;
        let one_minus_w = T::one() - w;
        Ok(self
            .components
            .iter()
            .zip(&end.components)
            .map(|(a, b)| one_minus_w * *a + w * *b)
            .collect())
    }
}

// ================================
//
// VectorOpsComplexFloat trait impl
//
// ================================
// TODO: add tests
impl<N> VectorOpsComplex<N> for FlexVector<Complex<N>>
where
    N: num::Float + Clone + PartialOrd + std::iter::Sum<N> + Sync + Send,
{
    type Output = Vec<Complex<N>>;

    #[inline]
    fn normalize(&self) -> Result<Self::Output, VectorError>
    where
        N: num::Float + Clone + std::iter::Sum<N>,
        Self::Output: std::iter::FromIterator<Complex<N>>,
    {
        let n = self.norm();
        if n == N::zero() {
            return Err(VectorError::ZeroVectorError("Cannot normalize a zero vector".to_string()));
        }
        Ok(self.components.iter().map(|a| *a / n).collect())
    }

    #[inline]
    fn normalize_to(&self, magnitude: N) -> Result<Self::Output, VectorError>
    where
        N: num::Float + Clone + std::iter::Sum<N>,
        Self::Output: std::iter::FromIterator<Complex<N>>,
    {
        let n = self.norm();
        if n == N::zero() {
            return Err(VectorError::ZeroVectorError("Cannot normalize a zero vector".to_string()));
        }
        let scale = magnitude / n;
        Ok(self.as_slice().iter().map(|a| *a * scale).collect())
    }

    #[inline]
    fn lerp(&self, end: &Self, weight: N) -> Result<Self::Output, VectorError>
    where
        N: num::Float + Clone + PartialOrd,
    {
        if weight < N::zero() || weight > N::one() {
            return Err(VectorError::OutOfRangeError("weight must be in [0, 1]".to_string()));
        }
        let w = Complex::new(weight, N::zero());
        let one_minus_w = Complex::new(N::one() - weight, N::zero());
        Ok(self
            .components
            .iter()
            .zip(&end.components)
            .map(|(a, b)| one_minus_w * *a + w * *b)
            .collect())
    }
}

// ================================
//
// Methods
//
// ================================
impl<T> FlexVector<T>
where
    T: num::Num + Clone + Sync + Send,
{
    /// Adds an element to the end of the vector.
    #[inline]
    pub fn push(&mut self, value: T) {
        self.components.push(value);
    }

    /// Removes the last element and returns it, or None if empty.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.components.pop()
    }

    /// Inserts an element at position index, shifting all elements after it.
    #[inline]
    pub fn insert(&mut self, index: usize, value: T) {
        self.components.insert(index, value);
    }

    /// Removes and returns the element at position index.
    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        self.components.remove(index)
    }

    /// Resizes the vector in-place.
    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T) {
        self.components.resize(new_len, value);
    }

    /// Clears the vector, removing all values.
    #[inline]
    pub fn clear(&mut self) {
        self.components.clear();
    }

    /// Returns a mutable reference to a FlexVector index value or range,
    /// or `None` if the index is out of bounds.
    #[inline]
    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut I::Output>
    where
        I: std::slice::SliceIndex<[T]>,
    {
        self.components.get_mut(index)
    }

    /// Returns an iterator over mutable references to the elements.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.components.iter_mut()
    }

    /// Returns a new FlexVector with each element mapped to a new value using the provided closure or function.
    #[inline]
    pub fn map<U, F>(&self, mut f: F) -> FlexVector<U>
    where
        F: FnMut(T) -> U,
        U: num::Num + Clone + Sync + Send,
    {
        let new_components = self.components.iter().cloned().map(&mut f).collect();
        FlexVector { components: new_components }
    }

    /// Applies a closure or function to each element, modifying them in place.
    #[inline]
    pub fn mut_map<F>(&mut self, mut f: F)
    where
        F: FnMut(T) -> T,
    {
        for x in self.components.iter_mut() {
            // Use clone since T may not be Copy
            *x = f(x.clone());
        }
    }

    /// Cosine similarity between self and other.
    #[inline]
    pub fn cosine_similarity(&self, other: &Self) -> T
    where
        T: num::Float + Clone + std::iter::Sum<T>,
    {
        let dot = self.dot(other);
        let norm_self = self.norm();
        let norm_other = other.norm();
        if norm_self == T::zero() || norm_other == T::zero() {
            T::zero()
        } else {
            dot / (norm_self * norm_other)
        }
    }
}

// ================================
//
// Operator overload trait impl
//
// ================================
impl_vector_unary_op!(FlexVector, Neg, neg, -);

impl_vector_binop!(FlexVector, Add, add, +);
impl_vector_binop!(FlexVector, Sub, sub, -);
impl_vector_binop!(FlexVector, Mul, mul, *);

impl_vector_binop_assign!(FlexVector, AddAssign, add_assign, +);
impl_vector_binop_assign!(FlexVector, SubAssign, sub_assign, -);
impl_vector_binop_assign!(FlexVector, MulAssign, mul_assign, *);

impl_vector_scalar_op!(FlexVector, Mul, mul, *);
impl_vector_scalar_op_assign!(FlexVector, MulAssign, mul_assign, *);

impl_vector_scalar_div_op!(FlexVector);
impl_vector_scalar_div_op_assign!(FlexVector);

#[cfg(test)]
mod tests {
    use super::*;
    use num::complex::ComplexFloat;
    use num::Complex;

    // ================================
    //
    // Constructor tests
    //
    // ================================
    #[test]
    fn test_new() {
        let v = FlexVector::<i32>::new();
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
    }

    #[test]
    fn test_with_capacity() {
        let v = FlexVector::<i32>::with_capacity(10);
        assert_eq!(v.len(), 0);
        assert!(v.components.capacity() >= 10);
    }

    #[test]
    fn test_with_capacity_large() {
        let v = FlexVector::<i32>::with_capacity(1000);
        assert_eq!(v.len(), 0);
        assert!(v.components.capacity() >= 1000);
    }

    #[test]
    fn test_zero() {
        let v = FlexVector::<i32>::zero(5);
        assert_eq!(v.len(), 5);
        assert!(v.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_zero_f64() {
        let v = FlexVector::<f64>::zero(3);
        assert_eq!(v.len(), 3);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_zero_complex() {
        let v = FlexVector::<Complex<f64>>::zero(2);
        assert_eq!(v.len(), 2);
        assert!(v.iter().all(|x| x.re == 0.0 && x.im == 0.0));
    }

    #[test]
    fn test_one() {
        let v = FlexVector::<i32>::one(4);
        assert_eq!(v.len(), 4);
        assert!(v.iter().all(|&x| x == 1));
    }

    #[test]
    fn test_one_f64() {
        let v = FlexVector::<f64>::one(2);
        assert_eq!(v.len(), 2);
        assert!(v.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_one_complex() {
        let v = FlexVector::<Complex<f64>>::one(2);
        assert_eq!(v.len(), 2);
        assert!(v.iter().all(|x| x.re == 1.0 && x.im == 0.0));
    }

    #[test]
    fn test_filled() {
        let v = FlexVector::<i32>::filled(3, 42);
        assert_eq!(v.len(), 3);
        assert!(v.iter().all(|&x| x == 42));
    }

    #[test]
    fn test_filled_f64() {
        let v = FlexVector::<f64>::filled(3, -1.5);
        assert_eq!(v.len(), 3);
        assert!(v.iter().all(|&x| x == -1.5));
    }

    #[test]
    fn test_filled_f32_fractional() {
        let v = FlexVector::<f32>::filled(2, 2.25);
        assert_eq!(v.len(), 2);
        assert!(v.iter().all(|&x| x == 2.25));
    }

    #[test]
    fn test_filled_nan() {
        let v = FlexVector::<f64>::filled(2, f64::NAN);
        assert_eq!(v.len(), 2);
        assert!(v.iter().all(|&x| x.is_nan()));
    }

    #[test]
    fn test_filled_infinity() {
        let v = FlexVector::<f64>::filled(2, f64::INFINITY);
        assert_eq!(v.len(), 2);
        assert!(v.iter().all(|&x| x.is_infinite() && x.is_sign_positive()));
    }

    #[test]
    fn test_filled_complex() {
        let value = Complex::new(2.0, 3.0);
        let v = FlexVector::<Complex<f64>>::filled(2, value);
        assert_eq!(v.len(), 2);
        assert!(v.iter().all(|x| *x == value));
    }

    #[test]
    fn test_from_slice() {
        let data = [1, 2, 3, 4];
        let v = FlexVector::<i32>::from_slice(&data);
        assert_eq!(v.len(), 4);
        assert_eq!(v.as_slice(), &data);
    }

    #[test]
    fn test_from_slice_immutability() {
        let mut data = [1, 2, 3];
        let v = FlexVector::<i32>::from_slice(&data);
        data[0] = 99;
        assert_eq!(v.as_slice()[0], 1);
    }

    #[test]
    fn test_from_vec() {
        let data = vec![5, 6, 7];
        let v = FlexVector::<i32>::from_vec(data.clone());
        assert_eq!(v.len(), 3);
        assert_eq!(v.as_slice(), &data[..]);
    }

    #[test]
    fn test_zero_length_constructors() {
        assert_eq!(FlexVector::<i32>::zero(0).len(), 0);
        assert_eq!(FlexVector::<i32>::one(0).len(), 0);
        assert_eq!(FlexVector::<i32>::filled(0, 123).len(), 0);
        assert_eq!(FlexVector::<i32>::from_slice(&[]).len(), 0);
        assert_eq!(FlexVector::<i32>::from_vec(vec![]).len(), 0);
    }

    // ================================
    //
    // Default trait tests
    //
    // ================================
    #[test]
    fn test_default_i32() {
        let v = FlexVector::<i32>::default();
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
    }

    #[test]
    fn test_default_f64() {
        let v = FlexVector::<f64>::default();
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
    }

    #[test]
    fn test_default_complex() {
        use num::Complex;
        let v = FlexVector::<Complex<f64>>::default();
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
    }

    // ================================
    //
    // Display trait tests
    //
    // ================================
    #[test]
    fn test_display_i32() {
        let v = FlexVector::from_vec(vec![1, 2, 3]);
        assert_eq!(format!("{}", v), "[1, 2, 3]");
    }

    #[test]
    fn test_display_f64() {
        let v = FlexVector::from_vec(vec![-1.0, 2.0, 3.0]);
        assert_eq!(format!("{}", v), "[-1.0, 2.0, 3.0]");
    }

    #[test]
    fn test_display_complex() {
        use num::Complex;
        let v = FlexVector::from_vec(vec![Complex::new(0, -1), Complex::new(3, 4)]);
        assert_eq!(format!("{}", v), "[Complex { re: 0, im: -1 }, Complex { re: 3, im: 4 }]");
    }

    #[test]
    fn test_display_empty() {
        let v = FlexVector::<i32>::new();
        assert_eq!(format!("{}", v), "[]");
    }

    // ================================
    //
    // VectorBase trait method tests
    //
    // ================================
    // --- as_slice ---
    #[test]
    fn test_as_slice() {
        let v = FlexVector::from_vec(vec![1, 2, 3]);
        let slice = v.as_slice();
        assert_eq!(slice, &[1, 2, 3]);
    }

    #[test]
    fn test_as_slice_f64() {
        let v = FlexVector::from_vec(vec![1.1, 2.2, 3.3]);
        let slice = v.as_slice();
        assert_eq!(slice, &[1.1, 2.2, 3.3]);
    }

    #[test]
    fn test_as_slice_complex() {
        let v = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let slice = v.as_slice();
        assert_eq!(slice, &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),]);
    }

    // --- len ---
    #[test]
    fn test_len() {
        let v = FlexVector::from_vec(vec![10, 20, 30, 40]);
        assert_eq!(v.len(), 4);
        let empty = FlexVector::<i32>::new();
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_len_f64() {
        let v = FlexVector::from_vec(vec![10.0, 20.0]);
        assert_eq!(v.len(), 2);
        let empty = FlexVector::<f64>::new();
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_len_complex() {
        let v = FlexVector::from_vec(vec![Complex::new(1.0, 0.0)]);
        assert_eq!(v.len(), 1);
        let empty = FlexVector::<Complex<f64>>::new();
        assert_eq!(empty.len(), 0);
    }

    // --- is_empty ---
    #[test]
    fn test_is_empty() {
        let v = FlexVector::from_vec(vec![1]);
        assert!(!v.is_empty());
        let empty = FlexVector::<i32>::new();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_is_empty_f64() {
        let v = FlexVector::from_vec(vec![1.0]);
        assert!(!v.is_empty());
        let empty = FlexVector::<f64>::new();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_is_empty_complex() {
        let v = FlexVector::from_vec(vec![Complex::new(0.0, 1.0)]);
        assert!(!v.is_empty());
        let empty = FlexVector::<Complex<f64>>::new();
        assert!(empty.is_empty());
    }

    // --- get ---
    #[test]
    fn test_get() {
        let v = FlexVector::from_vec(vec![10, 20, 30]);
        assert_eq!(v.get(0), Some(&10));
        assert_eq!(v.get(2), Some(&30));
        assert_eq!(v.get(3), None);
    }

    #[test]
    fn test_get_f64() {
        let v = FlexVector::from_vec(vec![10.5, 20.5, 30.5]);
        assert_eq!(v.get(0), Some(&10.5));
        assert_eq!(v.get(2), Some(&30.5));
        assert_eq!(v.get(3), None);
    }

    #[test]
    fn test_get_complex() {
        let v = FlexVector::from_vec(vec![Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)]);
        assert_eq!(v.get(0), Some(&Complex::new(1.0, 1.0)));
        assert_eq!(v.get(1), Some(&Complex::new(2.0, 2.0)));
        assert_eq!(v.get(2), None);
    }

    // --- first ---
    #[test]
    fn test_first() {
        let v = FlexVector::from_vec(vec![5, 6, 7]);
        assert_eq!(v.first(), Some(&5));
        let empty = FlexVector::<i32>::new();
        assert_eq!(empty.first(), None);
    }

    #[test]
    fn test_first_f64() {
        let v = FlexVector::from_vec(vec![5.5, 6.5]);
        assert_eq!(v.first(), Some(&5.5));
        let empty = FlexVector::<f64>::new();
        assert_eq!(empty.first(), None);
    }

    #[test]
    fn test_first_complex() {
        let v = FlexVector::from_vec(vec![Complex::new(7.0, 8.0), Complex::new(9.0, 10.0)]);
        assert_eq!(v.first(), Some(&Complex::new(7.0, 8.0)));
        let empty = FlexVector::<Complex<f64>>::new();
        assert_eq!(empty.first(), None);
    }

    // --- last ---
    #[test]
    fn test_last() {
        let v = FlexVector::from_vec(vec![1, 2, 3]);
        assert_eq!(v.last(), Some(&3));
        let empty = FlexVector::<i32>::new();
        assert_eq!(empty.last(), None);
    }

    #[test]
    fn test_last_f64() {
        let v = FlexVector::from_vec(vec![1.5, 2.5, 3.5]);
        assert_eq!(v.last(), Some(&3.5));
        let empty = FlexVector::<f64>::new();
        assert_eq!(empty.last(), None);
    }

    #[test]
    fn test_last_complex() {
        let v = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        assert_eq!(v.last(), Some(&Complex::new(3.0, 4.0)));
        let empty = FlexVector::<Complex<f64>>::new();
        assert_eq!(empty.last(), None);
    }

    // --- iter ---
    #[test]
    fn test_iter() {
        let v = FlexVector::from_vec(vec![1, 2, 3]);
        let mut iter = v.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_f64() {
        let v = FlexVector::from_vec(vec![1.1, 2.2, 3.3]);
        let collected: Vec<_> = v.iter().copied().collect();
        assert_eq!(collected, vec![1.1, 2.2, 3.3]);
    }

    #[test]
    fn test_iter_complex() {
        let v = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let collected: Vec<_> = v.iter().cloned().collect();
        assert_eq!(collected, vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),]);
    }

    // --- iter_rev ---
    #[test]
    fn test_iter_rev() {
        let v = FlexVector::from_vec(vec![1, 2, 3]);
        let collected: Vec<_> = v.iter_rev().copied().collect();
        assert_eq!(collected, vec![3, 2, 1]);
    }

    #[test]
    fn test_iter_rev_f64() {
        let v = FlexVector::from_vec(vec![1.1, 2.2, 3.3]);
        let collected: Vec<_> = v.iter_rev().copied().collect();
        assert_eq!(collected, vec![3.3, 2.2, 1.1]);
    }

    #[test]
    fn test_iter_rev_complex() {
        let v = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let collected: Vec<_> = v.iter_rev().cloned().collect();
        assert_eq!(collected, vec![Complex::new(3.0, 4.0), Complex::new(1.0, 2.0),]);
    }

    // --- enumerate ---
    #[test]
    fn test_enumerate() {
        let v = FlexVector::from_vec(vec![10, 20, 30]);
        let pairs: Vec<_> = v.enumerate().collect();
        assert_eq!(pairs, vec![(0, &10), (1, &20), (2, &30)]);
    }

    #[test]
    fn test_enumerate_f64() {
        let v = FlexVector::from_vec(vec![1.5, 2.5]);
        let pairs: Vec<_> = v.enumerate().collect();
        assert_eq!(pairs, vec![(0, &1.5), (1, &2.5)]);
    }

    #[test]
    fn test_enumerate_complex() {
        let v = FlexVector::from_vec(vec![Complex::new(0.0, 1.0), Complex::new(2.0, 3.0)]);
        let pairs: Vec<_> = v.enumerate().collect();
        assert_eq!(pairs, vec![(0, &Complex::new(0.0, 1.0)), (1, &Complex::new(2.0, 3.0)),]);
    }

    // --- to_vec ---
    #[test]
    fn test_to_vec() {
        let v = FlexVector::from_vec(vec![1, 2, 3]);
        let vec_copy = v.to_vec();
        assert_eq!(vec_copy, vec![1, 2, 3]);
    }

    #[test]
    fn test_to_vec_f64() {
        let v = FlexVector::from_vec(vec![1.5, 2.5]);
        let vec_copy = v.to_vec();
        assert_eq!(vec_copy, vec![1.5, 2.5]);
    }

    #[test]
    fn test_to_vec_complex() {
        let v = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let vec_copy = v.to_vec();
        assert_eq!(vec_copy, vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),]);
    }

    // --- pretty ---
    #[test]
    fn test_pretty() {
        let v = FlexVector::from_vec(vec![1, 2, 3]);
        let pretty = v.pretty();
        assert!(pretty.contains("1"));
        assert!(pretty.contains("2"));
        assert!(pretty.contains("3"));
        assert!(pretty.starts_with("["));
    }

    #[test]
    fn test_pretty_f64() {
        let v = FlexVector::from_vec(vec![1.5, 2.5]);
        let pretty = v.pretty();
        assert!(pretty.contains("1.5"));
        assert!(pretty.contains("2.5"));
        assert!(pretty.starts_with("["));
    }

    #[test]
    fn test_pretty_complex() {
        let v = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let pretty = v.pretty();
        assert!(pretty.contains("1.0"));
        assert!(pretty.contains("2.0"));
        assert!(pretty.contains("3.0"));
        assert!(pretty.contains("4.0"));
        assert!(pretty.starts_with("["));
    }

    // --- contains ---
    #[test]
    fn test_contains() {
        let v = FlexVector::from_vec(vec![1, 2, 3]);
        assert!(v.contains(&2));
        assert!(!v.contains(&4));
    }

    #[test]
    fn test_contains_f64() {
        let v = FlexVector::from_vec(vec![1.1, 2.2, 3.3]);
        assert!(v.contains(&2.2));
        assert!(!v.contains(&4.4));
    }

    #[test]
    fn test_contains_complex() {
        let v = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        assert!(v.contains(&Complex::new(1.0, 2.0)));
        assert!(!v.contains(&Complex::new(0.0, 0.0)));
    }

    // --- starts_with ---
    #[test]
    fn test_starts_with() {
        let v = FlexVector::from_vec(vec![1, 2, 3]);
        assert!(v.starts_with(&[1, 2]));
        assert!(!v.starts_with(&[2, 3]));
    }

    #[test]
    fn test_starts_with_f64() {
        let v = FlexVector::from_vec(vec![1.1, 2.2, 3.3]);
        assert!(v.starts_with(&[1.1, 2.2]));
        assert!(!v.starts_with(&[2.2, 3.3]));
    }

    #[test]
    fn test_starts_with_complex() {
        let v = FlexVector::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ]);
        assert!(v.starts_with(&[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]));
        assert!(!v.starts_with(&[Complex::new(3.0, 4.0)]));
    }

    // --- ends_with ---
    #[test]
    fn test_ends_with() {
        let v = FlexVector::from_vec(vec![1, 2, 3]);
        assert!(v.ends_with(&[2, 3]));
        assert!(!v.ends_with(&[1, 2]));
    }

    #[test]
    fn test_ends_with_f64() {
        let v = FlexVector::from_vec(vec![1.1, 2.2, 3.3]);
        assert!(v.ends_with(&[2.2, 3.3]));
        assert!(!v.ends_with(&[1.1, 2.2]));
    }

    #[test]
    fn test_ends_with_complex() {
        let v = FlexVector::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ]);
        assert!(v.ends_with(&[Complex::new(3.0, 4.0), Complex::new(5.0, 6.0)]));
        assert!(!v.ends_with(&[Complex::new(1.0, 2.0)]));
    }

    // --- position ---
    #[test]
    fn test_position() {
        let v = FlexVector::from_vec(vec![10, 20, 30]);
        assert_eq!(v.position(|&x| x == 20), Some(1));
        assert_eq!(v.position(|&x| x == 99), None);
    }

    #[test]
    fn test_position_f64() {
        let v = FlexVector::from_vec(vec![10.0, 20.0, 30.0]);
        assert_eq!(v.position(|&x| x == 20.0), Some(1));
        assert_eq!(v.position(|&x| x == 99.0), None);
    }

    #[test]
    fn test_position_complex() {
        let v = FlexVector::from_vec(vec![
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 2.0),
            Complex::new(3.0, 3.0),
        ]);
        assert_eq!(v.position(|x| *x == Complex::new(2.0, 2.0)), Some(1));
        assert_eq!(v.position(|x| *x == Complex::new(9.0, 9.0)), None);
    }

    // --- rposition ---
    #[test]
    fn test_rposition() {
        let v = FlexVector::from_vec(vec![1, 2, 3, 2]);
        assert_eq!(v.rposition(|&x| x == 2), Some(3));
        assert_eq!(v.rposition(|&x| x == 99), None);
    }

    #[test]
    fn test_rposition_f64() {
        let v = FlexVector::from_vec(vec![1.0, 2.0, 3.0, 2.0]);
        assert_eq!(v.rposition(|&x| x == 2.0), Some(3));
        assert_eq!(v.rposition(|&x| x == 99.0), None);
    }

    #[test]
    fn test_rposition_complex() {
        let v = FlexVector::from_vec(vec![
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 2.0),
            Complex::new(3.0, 3.0),
            Complex::new(2.0, 2.0),
        ]);
        assert_eq!(v.rposition(|x| *x == Complex::new(2.0, 2.0)), Some(3));
        assert_eq!(v.rposition(|x| *x == Complex::new(9.0, 9.0)), None);
    }

    // --- windows ---
    #[test]
    fn test_windows() {
        let v = FlexVector::from_vec(vec![1, 2, 3, 4]);
        let windows: Vec<_> = v.windows(2).collect();
        assert_eq!(windows, vec![&[1, 2][..], &[2, 3][..], &[3, 4][..]]);
    }

    #[test]
    fn test_windows_f64() {
        let v = FlexVector::from_vec(vec![1.0, 2.0, 3.0]);
        let windows: Vec<_> = v.windows(2).collect();
        assert_eq!(windows, vec![&[1.0, 2.0][..], &[2.0, 3.0][..]]);
    }

    #[test]
    fn test_windows_complex() {
        let v = FlexVector::from_vec(vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        ]);
        let windows: Vec<_> = v.windows(2).collect();
        assert_eq!(
            windows,
            vec![
                &[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)][..],
                &[Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)][..]
            ]
        );
    }

    // --- chunks ---
    #[test]
    fn test_chunks() {
        let v = FlexVector::from_vec(vec![1, 2, 3, 4, 5]);
        let chunks: Vec<_> = v.chunks(2).collect();
        assert_eq!(chunks, vec![&[1, 2][..], &[3, 4][..], &[5][..]]);
    }

    #[test]
    fn test_chunks_f64() {
        let v = FlexVector::from_vec(vec![1.0, 2.0, 3.0]);
        let chunks: Vec<_> = v.chunks(2).collect();
        assert_eq!(chunks, vec![&[1.0, 2.0][..], &[3.0][..]]);
    }

    #[test]
    fn test_chunks_complex() {
        let v = FlexVector::from_vec(vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        ]);
        let chunks: Vec<_> = v.chunks(2).collect();
        assert_eq!(
            chunks,
            vec![
                &[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)][..],
                &[Complex::new(3.0, 0.0)][..]
            ]
        );
    }

    // --- split_at ---
    #[test]
    fn test_split_at() {
        let v = FlexVector::from_vec(vec![1, 2, 3, 4]);
        let (left, right) = v.split_at(2);
        assert_eq!(left, &[1, 2]);
        assert_eq!(right, &[3, 4]);
    }

    #[test]
    fn test_split_at_f64() {
        let v = FlexVector::from_vec(vec![1.0, 2.0, 3.0]);
        let (left, right) = v.split_at(1);
        assert_eq!(left, &[1.0]);
        assert_eq!(right, &[2.0, 3.0]);
    }

    #[test]
    fn test_split_at_complex() {
        let v = FlexVector::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);
        let (left, right) = v.split_at(1);
        assert_eq!(left, &[Complex::new(1.0, 0.0)]);
        assert_eq!(right, &[Complex::new(2.0, 0.0)]);
    }

    // --- split ---
    #[test]
    fn test_split() {
        let v = FlexVector::from_vec(vec![1, 2, 0, 3, 0, 4]);
        let splits: Vec<_> = v.split(|&x| x == 0).collect();
        assert_eq!(splits, vec![&[1, 2][..], &[3][..], &[4][..]]);
    }

    #[test]
    fn test_split_f64() {
        let v = FlexVector::from_vec(vec![1.0, 0.0, 2.0, 0.0, 3.0]);
        let splits: Vec<_> = v.split(|&x| x == 0.0).collect();
        assert_eq!(splits, vec![&[1.0][..], &[2.0][..], &[3.0][..]]);
    }

    #[test]
    fn test_split_complex() {
        let v = FlexVector::from_vec(vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.0, 0.0),
        ]);
        let splits: Vec<_> = v.split(|x| *x == Complex::new(0.0, 0.0)).collect();
        assert_eq!(splits, vec![&[Complex::new(1.0, 0.0)][..], &[Complex::new(2.0, 0.0)][..]]);
    }

    // --- splitn ---
    #[test]
    fn test_splitn() {
        let v = FlexVector::from_vec(vec![1, 0, 2, 0, 3]);
        let splits: Vec<_> = v.splitn(2, |&x| x == 0).collect();
        assert_eq!(splits, vec![&[1][..], &[2, 0, 3][..]]);
    }

    #[test]
    fn test_splitn_f64() {
        let v = FlexVector::from_vec(vec![1.0, 0.0, 2.0, 0.0, 3.0]);
        let splits: Vec<_> = v.splitn(2, |&x| x == 0.0).collect();
        assert_eq!(splits, vec![&[1.0][..], &[2.0, 0.0, 3.0][..]]);
    }

    #[test]
    fn test_splitn_complex() {
        let v = FlexVector::from_vec(vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(3.0, 0.0),
        ]);
        let splits: Vec<_> = v.splitn(2, |x| *x == Complex::new(0.0, 0.0)).collect();
        assert_eq!(
            splits,
            vec![
                &[Complex::new(1.0, 0.0)][..],
                &[Complex::new(2.0, 0.0), Complex::new(0.0, 0.0), Complex::new(3.0, 0.0)][..]
            ]
        );
    }

    // --- rsplit ---
    #[test]
    fn test_rsplit() {
        let v = FlexVector::from_vec(vec![1, 0, 2, 0, 3]);
        let splits: Vec<_> = v.rsplit(|&x| x == 0).collect();
        assert_eq!(splits, vec![&[3][..], &[2][..], &[1][..]]);
    }

    #[test]
    fn test_rsplit_f64() {
        let v = FlexVector::from_vec(vec![1.0, 0.0, 2.0, 0.0, 3.0]);
        let splits: Vec<_> = v.rsplit(|&x| x == 0.0).collect();
        assert_eq!(splits, vec![&[3.0][..], &[2.0][..], &[1.0][..]]);
    }

    #[test]
    fn test_rsplit_complex() {
        let v = FlexVector::from_vec(vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(3.0, 0.0),
        ]);
        let splits: Vec<_> = v.rsplit(|x| *x == Complex::new(0.0, 0.0)).collect();
        assert_eq!(
            splits,
            vec![
                &[Complex::new(3.0, 0.0)][..],
                &[Complex::new(2.0, 0.0)][..],
                &[Complex::new(1.0, 0.0)][..]
            ]
        );
    }

    // --- rsplitn ---
    #[test]
    fn test_rsplitn() {
        let v = FlexVector::from_vec(vec![1, 0, 2, 0, 3]);
        let splits: Vec<_> = v.rsplitn(2, |&x| x == 0).collect();
        assert_eq!(splits, vec![&[3][..], &[1, 0, 2][..]]);
    }

    #[test]
    fn test_rsplitn_f64() {
        let v = FlexVector::from_vec(vec![1.0, 0.0, 2.0, 0.0, 3.0]);
        let splits: Vec<_> = v.rsplitn(2, |&x| x == 0.0).collect();
        assert_eq!(splits, vec![&[3.0][..], &[1.0, 0.0, 2.0][..]]);
    }

    #[test]
    fn test_rsplitn_complex() {
        let v = FlexVector::from_vec(vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(3.0, 0.0),
        ]);
        let splits: Vec<_> = v.rsplitn(2, |x| *x == Complex::new(0.0, 0.0)).collect();
        assert_eq!(
            splits,
            vec![
                &[Complex::new(3.0, 0.0)][..],
                &[Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(2.0, 0.0)][..]
            ]
        );
    }

    // --- push ---
    #[test]
    fn test_push() {
        let mut v = FlexVector::new();
        v.push(1);
        v.push(2);
        assert_eq!(v.as_slice(), &[1, 2]);
    }

    #[test]
    fn test_push_f64() {
        let mut v = FlexVector::new();
        v.push(1.1);
        v.push(2.2);
        assert_eq!(v.as_slice(), &[1.1, 2.2]);
    }

    #[test]
    fn test_push_complex() {
        let mut v = FlexVector::new();
        v.push(Complex::new(1.0, 2.0));
        v.push(Complex::new(3.0, 4.0));
        assert_eq!(v.as_slice(), &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
    }

    // --- pop ---
    #[test]
    fn test_pop() {
        let mut v = FlexVector::from_vec(vec![1, 2]);
        assert_eq!(v.pop(), Some(2));
        assert_eq!(v.pop(), Some(1));
        assert_eq!(v.pop(), None);
    }

    #[test]
    fn test_pop_f64() {
        let mut v = FlexVector::from_vec(vec![1.1, 2.2]);
        assert_eq!(v.pop(), Some(2.2));
        assert_eq!(v.pop(), Some(1.1));
        assert_eq!(v.pop(), None);
    }

    #[test]
    fn test_pop_complex() {
        let mut v = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        assert_eq!(v.pop(), Some(Complex::new(3.0, 4.0)));
        assert_eq!(v.pop(), Some(Complex::new(1.0, 2.0)));
        assert_eq!(v.pop(), None);
    }

    // --- insert ---
    #[test]
    fn test_insert() {
        let mut v = FlexVector::from_vec(vec![1, 3]);
        v.insert(1, 2);
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_insert_f64() {
        let mut v = FlexVector::from_vec(vec![1.1, 3.3]);
        v.insert(1, 2.2);
        assert_eq!(v.as_slice(), &[1.1, 2.2, 3.3]);
    }

    #[test]
    fn test_insert_complex() {
        let mut v = FlexVector::from_vec(vec![Complex::new(1.0, 1.0), Complex::new(3.0, 3.0)]);
        v.insert(1, Complex::new(2.0, 2.0));
        assert_eq!(
            v.as_slice(),
            &[Complex::new(1.0, 1.0), Complex::new(2.0, 2.0), Complex::new(3.0, 3.0)]
        );
    }

    // --- remove ---
    #[test]
    fn test_remove() {
        let mut v = FlexVector::from_vec(vec![1, 2, 3]);
        assert_eq!(v.remove(1), 2);
        assert_eq!(v.as_slice(), &[1, 3]);
    }

    #[test]
    fn test_remove_f64() {
        let mut v = FlexVector::from_vec(vec![1.1, 2.2, 3.3]);
        assert_eq!(v.remove(1), 2.2);
        assert_eq!(v.as_slice(), &[1.1, 3.3]);
    }

    #[test]
    fn test_remove_complex() {
        let mut v = FlexVector::from_vec(vec![
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 2.0),
            Complex::new(3.0, 3.0),
        ]);
        assert_eq!(v.remove(1), Complex::new(2.0, 2.0));
        assert_eq!(v.as_slice(), &[Complex::new(1.0, 1.0), Complex::new(3.0, 3.0)]);
    }

    // --- resize ---
    #[test]
    fn test_resize() {
        let mut v = FlexVector::from_vec(vec![1, 2]);
        v.resize(4, 0);
        assert_eq!(v.as_slice(), &[1, 2, 0, 0]);
        v.resize(2, 0);
        assert_eq!(v.as_slice(), &[1, 2]);
    }

    #[test]
    fn test_resize_f64() {
        let mut v = FlexVector::from_vec(vec![1.1, 2.2]);
        v.resize(4, 0.0);
        assert_eq!(v.as_slice(), &[1.1, 2.2, 0.0, 0.0]);
        v.resize(1, 0.0);
        assert_eq!(v.as_slice(), &[1.1]);
    }

    #[test]
    fn test_resize_complex() {
        let mut v = FlexVector::from_vec(vec![Complex::new(1.0, 1.0)]);
        v.resize(3, Complex::new(0.0, 0.0));
        assert_eq!(
            v.as_slice(),
            &[Complex::new(1.0, 1.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)]
        );
        v.resize(1, Complex::new(0.0, 0.0));
        assert_eq!(v.as_slice(), &[Complex::new(1.0, 1.0)]);
    }

    // --- clear ---
    #[test]
    fn test_clear() {
        let mut v = FlexVector::from_vec(vec![1, 2, 3]);
        assert!(!v.is_empty());
        v.clear();
        assert!(v.is_empty());
    }

    #[test]
    fn test_clear_f64() {
        let mut v = FlexVector::from_vec(vec![1.1, 2.2, 3.3]);
        assert!(!v.is_empty());
        v.clear();
        assert!(v.is_empty());
    }

    #[test]
    fn test_clear_complex() {
        let mut v = FlexVector::from_vec(vec![Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)]);
        assert!(!v.is_empty());
        v.clear();
        assert!(v.is_empty());
    }

    // --- get_mut ---
    #[test]
    fn test_get_mut_single() {
        let mut v = FlexVector::from_vec(vec![10, 20, 30]);
        if let Some(x) = v.get_mut(1) {
            *x = 99;
        }
        assert_eq!(v.as_slice(), &[10, 99, 30]);
    }

    #[test]
    fn test_get_mut_out_of_bounds() {
        let mut v = FlexVector::from_vec(vec![1, 2, 3]);
        assert!(v.get_mut(10).is_none());
    }

    #[test]
    fn test_get_mut_range() {
        let mut v = FlexVector::from_vec(vec![1, 2, 3, 4]);
        if let Some(slice) = v.get_mut(1..3) {
            slice[0] = 20;
            slice[1] = 30;
        }
        assert_eq!(v.as_slice(), &[1, 20, 30, 4]);
    }

    #[test]
    fn test_get_mut_full_range() {
        let mut v = FlexVector::from_vec(vec![5, 6, 7]);
        if let Some(slice) = v.get_mut(..) {
            for x in slice {
                *x *= 2;
            }
        }
        assert_eq!(v.as_slice(), &[10, 12, 14]);
    }

    // --- iter_mut ---
    #[test]
    fn test_iter_mut_i32() {
        let mut v = FlexVector::from_vec(vec![1, 2, 3]);
        for x in v.iter_mut() {
            *x *= 2;
        }
        assert_eq!(v.as_slice(), &[2, 4, 6]);
    }

    #[test]
    fn test_iter_mut_f64() {
        let mut v = FlexVector::from_vec(vec![1.5, -2.0, 0.0]);
        for x in v.iter_mut() {
            *x += 1.0;
        }
        assert_eq!(v.as_slice(), &[2.5, -1.0, 1.0]);
    }

    #[test]
    fn test_iter_mut_complex() {
        use num::Complex;
        let mut v = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(-3.0, 4.0)]);
        for x in v.iter_mut() {
            x.re += 1.0;
            x.im *= 2.0;
        }
        assert_eq!(v.as_slice(), &[Complex::new(2.0, 4.0), Complex::new(-2.0, 8.0)]);
    }

    #[test]
    fn test_iter_mut_empty() {
        let mut v = FlexVector::<i32>::new();
        let mut count = 0;
        for _ in v.iter_mut() {
            count += 1;
        }
        assert_eq!(count, 0);
    }

    // --- as_mut_slice ---
    #[test]
    fn test_as_mut_slice_i32() {
        let mut v = FlexVector::from_vec(vec![1, 2, 3]);
        let slice = v.as_mut_slice();
        slice[0] = 10;
        slice[2] = 30;
        assert_eq!(v.as_slice(), &[10, 2, 30]);
    }

    #[test]
    fn test_as_mut_slice_f64() {
        let mut v = FlexVector::from_vec(vec![1.1, 2.2, 3.3]);
        let slice = v.as_mut_slice();
        slice[1] = 9.9;
        assert_eq!(v.as_slice(), &[1.1, 9.9, 3.3]);
    }

    #[test]
    fn test_as_mut_slice_complex() {
        use num::Complex;
        let mut v = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let slice = v.as_mut_slice();
        slice[0].re = 10.0;
        slice[1].im = 40.0;
        assert_eq!(v.as_slice(), &[Complex::new(10.0, 2.0), Complex::new(3.0, 40.0)]);
    }

    #[test]
    fn test_as_mut_slice_empty() {
        let mut v = FlexVector::<i32>::new();
        let slice = v.as_mut_slice();
        assert_eq!(slice.len(), 0);
    }

    // --- map ---
    #[test]
    fn test_map_i32() {
        let v = FlexVector::from_vec(vec![1, 2, 3]);
        let squared = v.map(|x| x * x);
        assert_eq!(squared.as_slice(), &[1, 4, 9]);
        // original unchanged
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_map_f64() {
        let v: FlexVector<f64> = FlexVector::from_vec(vec![1.5, -2.0, 0.0]);
        let abs = v.map(|x| x.abs());
        assert_eq!(abs.as_slice(), &[1.5, 2.0, 0.0]);
    }

    #[test]
    fn test_map_complex() {
        use num::Complex;
        let v = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(-3.0, 4.0)]);
        let conj = v.map(|x| x.conj());
        assert_eq!(conj.as_slice(), &[Complex::new(1.0, -2.0), Complex::new(-3.0, -4.0)]);
    }

    #[test]
    fn test_map_empty() {
        let v = FlexVector::<i32>::new();
        let mapped = v.map(|x| x + 1);
        assert!(mapped.is_empty());
    }

    // used for function pointer test below
    fn square(x: i32) -> i32 {
        x * x
    }

    #[test]
    fn test_map_with_fn_pointer() {
        let v = FlexVector::from_vec(vec![1, 2, 3]);
        let squared = v.map(square);
        assert_eq!(squared.as_slice(), &[1, 4, 9]);
    }

    // --- mut_map ---
    #[test]
    fn test_mut_map_i32() {
        let mut v = FlexVector::from_vec(vec![1, 2, 3]);
        v.mut_map(|x| x * 10);
        assert_eq!(v.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_mut_map_f64() {
        let mut v = FlexVector::from_vec(vec![1.5, -2.0, 0.0]);
        v.mut_map(|x| x + 1.0);
        assert_eq!(v.as_slice(), &[2.5, -1.0, 1.0]);
    }

    #[test]
    fn test_mut_map_complex() {
        use num::Complex;
        let mut v = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(-3.0, 4.0)]);
        v.mut_map(|x| Complex::new(x.re + 1.0, x.im * 2.0));
        assert_eq!(v.as_slice(), &[Complex::new(2.0, 4.0), Complex::new(-2.0, 8.0)]);
    }

    #[test]
    fn test_mut_map_empty() {
        let mut v = FlexVector::<i32>::new();
        v.mut_map(|x| x + 1);
        assert!(v.is_empty());
    }

    // used for function pointer test below
    fn double(x: i32) -> i32 {
        x * 2
    }

    #[test]
    fn test_mut_map_with_fn_pointer() {
        let mut v = FlexVector::from_vec(vec![1, 2, 3]);
        v.mut_map(double);
        assert_eq!(v.as_slice(), &[2, 4, 6]);
    }

    // --- cosine_similarity ---
    #[test]
    fn test_cosine_similarity_parallel() {
        let v1 = FlexVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![2.0, 4.0, 6.0]);
        let cos_sim = v1.cosine_similarity(&v2);
        assert!((cos_sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let v1 = FlexVector::from_vec(vec![1.0, 0.0]);
        let v2 = FlexVector::from_vec(vec![0.0, 1.0]);
        let cos_sim = v1.cosine_similarity(&v2);
        assert!((cos_sim - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let v1 = FlexVector::from_vec(vec![1.0, 0.0]);
        let v2 = FlexVector::from_vec(vec![-1.0, 0.0]);
        let cos_sim = v1.cosine_similarity(&v2);
        assert!((cos_sim + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let v1 = FlexVector::from_vec(vec![0.0, 0.0]);
        let v2 = FlexVector::from_vec(vec![1.0, 2.0]);
        let cos_sim = v1.cosine_similarity(&v2);
        assert_eq!(cos_sim, 0.0);
    }

    // ================================
    //
    // Unary Negation trait tests
    //
    // ================================
    #[test]
    fn test_neg() {
        let v = FlexVector::from_vec(vec![1, -2, 3]);
        let neg_v = -v;
        assert_eq!(neg_v.as_slice(), &[-1, 2, -3]);
    }

    #[test]
    fn test_neg_f64() {
        let v = FlexVector::from_vec(vec![1.5, -2.5, 0.0]);
        let neg_v = -v;
        assert_eq!(neg_v.as_slice(), &[-1.5, 2.5, -0.0]);
    }

    #[test]
    fn test_neg_complex() {
        let v = FlexVector::from_vec(vec![Complex::new(1.0, -2.0), Complex::new(-3.0, 4.0)]);
        let neg_v = -v;
        assert_eq!(neg_v.as_slice(), &[Complex::new(-1.0, 2.0), Complex::new(3.0, -4.0)]);
    }

    #[test]
    fn test_neg_nan() {
        let v = FlexVector::from_vec(vec![f64::NAN, -f64::NAN]);
        let neg_v = -v;
        // Negating NaN is still NaN, but sign bit may flip
        assert!(neg_v.as_slice()[0].is_nan());
        assert!(neg_v.as_slice()[1].is_nan());
        // Optionally check sign bit if desired
        assert_eq!(neg_v.as_slice()[0].is_sign_negative(), true);
        assert_eq!(neg_v.as_slice()[1].is_sign_positive(), true);
    }

    #[test]
    fn test_neg_infinity() {
        let v = FlexVector::from_vec(vec![f64::INFINITY, f64::NEG_INFINITY]);
        let neg_v = -v;
        assert_eq!(neg_v.as_slice(), &[-f64::INFINITY, f64::INFINITY]);
    }

    #[test]
    fn test_add() {
        let v1 = FlexVector::from_vec(vec![1, 2, 3]);
        let v2 = FlexVector::from_vec(vec![4, 5, 6]);
        let sum = v1 + v2;
        assert_eq!(sum.as_slice(), &[5, 7, 9]);
    }

    #[test]
    fn test_add_f64() {
        let v1 = FlexVector::from_vec(vec![1.0, 2.5]);
        let v2 = FlexVector::from_vec(vec![3.0, 4.5]);
        let sum = v1 + v2;
        assert_eq!(sum.as_slice(), &[4.0, 7.0]);
    }

    #[test]
    fn test_add_complex() {
        let v1 = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let v2 = FlexVector::from_vec(vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)]);
        let sum = v1 + v2;
        assert_eq!(sum.as_slice(), &[Complex::new(6.0, 8.0), Complex::new(10.0, 12.0)]);
    }

    #[test]
    fn test_add_nan_infinity() {
        let v1 = FlexVector::from_vec(vec![f64::NAN, f64::INFINITY, 1.0]);
        let v2 = FlexVector::from_vec(vec![1.0, 2.0, f64::INFINITY]);
        let sum = v1 + v2;
        assert!(sum.as_slice()[0].is_nan());
        assert_eq!(sum.as_slice()[1], f64::INFINITY);
        assert_eq!(sum.as_slice()[2], f64::INFINITY);
    }

    #[test]
    fn test_sub() {
        let v1 = FlexVector::from_vec(vec![10, 20, 30]);
        let v2 = FlexVector::from_vec(vec![1, 2, 3]);
        let diff = v1 - v2;
        assert_eq!(diff.as_slice(), &[9, 18, 27]);
    }

    #[test]
    fn test_sub_f64() {
        let v1 = FlexVector::from_vec(vec![5.5, 2.0]);
        let v2 = FlexVector::from_vec(vec![1.5, 1.0]);
        let diff = v1 - v2;
        assert_eq!(diff.as_slice(), &[4.0, 1.0]);
    }

    #[test]
    fn test_sub_complex() {
        let v1 = FlexVector::from_vec(vec![Complex::new(5.0, 7.0), Complex::new(3.0, 4.0)]);
        let v2 = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(2.0, 1.0)]);
        let diff = v1 - v2;
        assert_eq!(diff.as_slice(), &[Complex::new(4.0, 5.0), Complex::new(1.0, 3.0)]);
    }

    #[test]
    fn test_sub_nan_infinity() {
        let v1 = FlexVector::from_vec(vec![f64::NAN, f64::INFINITY, 5.0]);
        let v2 = FlexVector::from_vec(vec![2.0, f64::INFINITY, f64::NAN]);
        let diff = v1 - v2;
        assert!(diff.as_slice()[0].is_nan());
        assert!(diff.as_slice()[1].is_nan()); // inf - inf = NaN
        assert!(diff.as_slice()[2].is_nan());
    }

    #[test]
    fn test_mul() {
        let v1 = FlexVector::from_vec(vec![2, 3, 4]);
        let v2 = FlexVector::from_vec(vec![5, 6, 7]);
        let prod = v1 * v2;
        assert_eq!(prod.as_slice(), &[10, 18, 28]);
    }

    #[test]
    fn test_mul_f64() {
        let v1 = FlexVector::from_vec(vec![1.5, 2.0]);
        let v2 = FlexVector::from_vec(vec![2.0, 3.0]);
        let prod = v1 * v2;
        assert_eq!(prod.as_slice(), &[3.0, 6.0]);
    }

    #[test]
    fn test_mul_complex() {
        let v1 = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let v2 = FlexVector::from_vec(vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)]);
        let prod = v1 * v2;
        assert_eq!(
            prod.as_slice(),
            &[
                Complex::new(-7.0, 16.0), // (1+2i)*(5+6i) = (1*5 - 2*6) + (1*6 + 2*5)i = (5-12)+(6+10)i = -7+16i
                Complex::new(-11.0, 52.0) // (3+4i)*(7+8i) = (3*7-4*8)+(3*8+4*7)i = (21-32)+(24+28)i = -11+52i
            ]
        );
    }

    #[test]
    fn test_mul_nan_infinity() {
        let v1 = FlexVector::from_vec(vec![f64::NAN, f64::INFINITY, 2.0]);
        let v2 = FlexVector::from_vec(vec![3.0, 2.0, f64::INFINITY]);
        let prod = v1 * v2;
        assert!(prod.as_slice()[0].is_nan());
        assert_eq!(prod.as_slice()[1], f64::INFINITY);
        assert_eq!(prod.as_slice()[2], f64::INFINITY);
    }

    #[test]
    fn test_add_assign() {
        let mut v1 = FlexVector::from_vec(vec![1, 2, 3]);
        let v2 = FlexVector::from_vec(vec![4, 5, 6]);
        v1 += v2;
        assert_eq!(v1.as_slice(), &[5, 7, 9]);
    }

    #[test]
    fn test_add_assign_f64() {
        let mut v1 = FlexVector::from_vec(vec![1.0, 2.5]);
        let v2 = FlexVector::from_vec(vec![3.0, 4.5]);
        v1 += v2;
        assert_eq!(v1.as_slice(), &[4.0, 7.0]);
    }

    #[test]
    fn test_add_assign_complex() {
        let mut v1 = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let v2 = FlexVector::from_vec(vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)]);
        v1 += v2;
        assert_eq!(v1.as_slice(), &[Complex::new(6.0, 8.0), Complex::new(10.0, 12.0)]);
    }

    #[test]
    fn test_sub_assign() {
        let mut v1 = FlexVector::from_vec(vec![10, 20, 30]);
        let v2 = FlexVector::from_vec(vec![1, 2, 3]);
        v1 -= v2;
        assert_eq!(v1.as_slice(), &[9, 18, 27]);
    }

    #[test]
    fn test_sub_assign_f64() {
        let mut v1 = FlexVector::from_vec(vec![5.5, 2.0]);
        let v2 = FlexVector::from_vec(vec![1.5, 1.0]);
        v1 -= v2;
        assert_eq!(v1.as_slice(), &[4.0, 1.0]);
    }

    #[test]
    fn test_sub_assign_complex() {
        let mut v1 = FlexVector::from_vec(vec![Complex::new(5.0, 7.0), Complex::new(3.0, 4.0)]);
        let v2 = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(2.0, 1.0)]);
        v1 -= v2;
        assert_eq!(v1.as_slice(), &[Complex::new(4.0, 5.0), Complex::new(1.0, 3.0)]);
    }

    #[test]
    fn test_mul_assign() {
        let mut v1 = FlexVector::from_vec(vec![2, 3, 4]);
        let v2 = FlexVector::from_vec(vec![5, 6, 7]);
        v1 *= v2;
        assert_eq!(v1.as_slice(), &[10, 18, 28]);
    }

    #[test]
    fn test_mul_assign_f64() {
        let mut v1 = FlexVector::from_vec(vec![1.5, 2.0]);
        let v2 = FlexVector::from_vec(vec![2.0, 3.0]);
        v1 *= v2;
        assert_eq!(v1.as_slice(), &[3.0, 6.0]);
    }

    #[test]
    fn test_mul_assign_complex() {
        let mut v1 = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let v2 = FlexVector::from_vec(vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)]);
        v1 *= v2;
        assert_eq!(v1.as_slice(), &[Complex::new(-7.0, 16.0), Complex::new(-11.0, 52.0)]);
    }

    #[test]
    fn test_scalar_mul() {
        let v = FlexVector::from_vec(vec![2, -3, 4]);
        let prod = v.clone() * 3;
        assert_eq!(prod.as_slice(), &[6, -9, 12]);
    }

    #[test]
    fn test_scalar_mul_f64() {
        let v = FlexVector::from_vec(vec![1.5, -2.0, 0.0]);
        let prod = v.clone() * 2.0;
        assert_eq!(prod.as_slice(), &[3.0, -4.0, 0.0]);
    }

    #[test]
    fn test_scalar_mul_complex() {
        use num::Complex;
        let v = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(-3.0, 4.0)]);
        let scalar = Complex::new(2.0, 0.0);
        let prod = v.clone() * scalar;
        assert_eq!(prod.as_slice(), &[Complex::new(2.0, 4.0), Complex::new(-6.0, 8.0)]);
    }

    #[test]
    fn test_scalar_mul_assign() {
        let mut v = FlexVector::from_vec(vec![2, -3, 4]);
        v *= 3;
        assert_eq!(v.as_slice(), &[6, -9, 12]);
    }

    #[test]
    fn test_scalar_mul_assign_f64() {
        let mut v = FlexVector::from_vec(vec![1.5, -2.0, 0.0]);
        v *= 2.0;
        assert_eq!(v.as_slice(), &[3.0, -4.0, 0.0]);
    }

    #[test]
    fn test_scalar_mul_assign_complex() {
        use num::Complex;
        let mut v = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(-3.0, 4.0)]);
        let scalar = Complex::new(2.0, 0.0);
        v *= scalar;
        assert_eq!(v.as_slice(), &[Complex::new(2.0, 4.0), Complex::new(-6.0, 8.0)]);
    }

    #[test]
    fn test_scalar_div_f64() {
        let v: FlexVector<f64> = FlexVector::from_vec(vec![2.0, -4.0, 8.0]);
        let result = v.clone() / 2.0;
        assert_eq!(result.as_slice(), &[1.0, -2.0, 4.0]);
    }

    #[test]
    fn test_scalar_div_assign_f64() {
        let mut v: FlexVector<f64> = FlexVector::from_vec(vec![2.0, -4.0, 8.0]);
        v /= 2.0;
        assert_eq!(v.as_slice(), &[1.0, -2.0, 4.0]);
    }

    #[test]
    fn test_scalar_div_complex_by_f64() {
        use num::Complex;
        let v: FlexVector<Complex<f64>> =
            FlexVector::from_vec(vec![Complex::new(2.0, 4.0), Complex::new(-6.0, 8.0)]);
        let result = v.clone() / 2.0;
        assert_eq!(result.as_slice(), &[Complex::new(1.0, 2.0), Complex::new(-3.0, 4.0)]);
    }

    #[test]
    fn test_scalar_div_assign_complex_by_f64() {
        use num::Complex;
        let mut v: FlexVector<Complex<f64>> =
            FlexVector::from_vec(vec![Complex::new(2.0, 4.0), Complex::new(-6.0, 8.0)]);
        v /= 2.0;
        assert_eq!(v.as_slice(), &[Complex::new(1.0, 2.0), Complex::new(-3.0, 4.0)]);
    }

    #[test]
    fn test_scalar_div_complex_by_complex() {
        use num::Complex;
        let v: FlexVector<Complex<f64>> =
            FlexVector::from_vec(vec![Complex::new(2.0, 4.0), Complex::new(-6.0, 8.0)]);
        let divisor = Complex::new(2.0, 0.0);
        let result = v.clone() / divisor;
        assert_eq!(result.as_slice(), &[Complex::new(1.0, 2.0), Complex::new(-3.0, 4.0)]);
    }

    #[test]
    fn test_scalar_div_assign_complex_by_complex() {
        use num::Complex;
        let mut v: FlexVector<Complex<f64>> =
            FlexVector::from_vec(vec![Complex::new(2.0, 4.0), Complex::new(-6.0, 8.0)]);
        let divisor = Complex::new(2.0, 0.0);
        v /= divisor;
        assert_eq!(v.as_slice(), &[Complex::new(1.0, 2.0), Complex::new(-3.0, 4.0)]);
    }

    #[test]
    fn test_scalar_div_nan_infinity() {
        let v = FlexVector::from_vec(vec![f64::NAN, f64::INFINITY, 4.0]);
        let result: FlexVector<f64> = v.clone() / 2.0;
        assert!(result.as_slice()[0].is_nan());
        assert_eq!(result.as_slice()[1], f64::INFINITY);
        assert_eq!(result.as_slice()[2], 2.0);
    }

    #[test]
    fn test_scalar_div_assign_nan_infinity() {
        let mut v: FlexVector<f64> = FlexVector::from_vec(vec![f64::NAN, f64::INFINITY, 4.0]);
        v /= 2.0;
        assert!(v.as_slice()[0].is_nan());
        assert_eq!(v.as_slice()[1], f64::INFINITY);
        assert_eq!(v.as_slice()[2], 2.0);
    }

    // additional math operator overload edge cases

    #[test]
    fn test_add_overflow_i32() {
        let v1 = FlexVector::from_vec(vec![i32::MAX]);
        let v2 = FlexVector::from_vec(vec![1]);
        if cfg!(debug_assertions) {
            // In debug mode, should panic on overflow
            let result = std::panic::catch_unwind(|| {
                let _ = v1.clone() + v2.clone();
            });
            assert!(result.is_err(), "Should panic on overflow in debug mode");
        } else {
            // In release mode, should wrap
            let sum = v1 + v2;
            assert_eq!(sum.as_slice()[0], i32::MIN);
        }
    }

    #[test]
    fn test_mul_overflow_i32() {
        let v1 = FlexVector::from_vec(vec![i32::MAX]);
        let v2 = FlexVector::from_vec(vec![2]);
        if cfg!(debug_assertions) {
            // In debug mode, should panic on overflow
            let result = std::panic::catch_unwind(|| {
                let _ = v1.clone() * v2.clone();
            });
            assert!(result.is_err(), "Should panic on overflow in debug mode");
        } else {
            // In release mode, should wrap
            let prod = v1 * v2;
            assert_eq!(prod.as_slice()[0], -2);
        }
    }

    #[test]
    fn test_divide_by_zero_f64() {
        let v: FlexVector<f64> = FlexVector::from_vec(vec![1.0, -2.0, 0.0]);
        let result = v.clone() / 0.0;
        assert_eq!(result.as_slice()[0], f64::INFINITY);
        assert_eq!(result.as_slice()[1], f64::NEG_INFINITY);
        assert!(result.as_slice()[2].is_nan());
    }

    #[test]
    fn test_neg_zero_f64() {
        let v = FlexVector::from_vec(vec![0.0, -0.0]);
        let neg_v = -v;
        assert_eq!(neg_v.as_slice()[0], -0.0);
        assert_eq!(neg_v.as_slice()[1], 0.0);
    }

    #[test]
    fn test_complex_div_by_zero() {
        use num::Complex;
        let v: FlexVector<Complex<f64>> = FlexVector::from_vec(vec![Complex::new(1.0, 1.0)]);
        let result = v.clone() / Complex::new(0.0, 0.0);
        assert!(result.as_slice()[0].re.is_nan());
        assert!(result.as_slice()[0].im.is_nan());
    }

    #[test]
    fn test_empty_vector_ops() {
        let v1: FlexVector<i32> = FlexVector::new();
        let v2: FlexVector<i32> = FlexVector::new();
        let sum = v1.clone() + v2.clone();
        assert!(sum.is_empty());
        let prod = v1 * v2;
        assert!(prod.is_empty());
    }

    #[test]
    #[should_panic]
    fn test_mismatched_length_add() {
        let v1 = FlexVector::from_vec(vec![1, 2]);
        let v2 = FlexVector::from_vec(vec![1, 2, 3]);
        let _ = v1 + v2; // should panic
    }
}
