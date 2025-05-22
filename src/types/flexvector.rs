//! FlexVector type.

use std::any::TypeId;
use std::borrow::{Borrow, BorrowMut, Cow};
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::{
    Deref, DerefMut, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive,
};

use crate::{
    impl_vector_binop, impl_vector_binop_assign, impl_vector_binop_div,
    impl_vector_binop_div_assign, impl_vector_scalar_div_op, impl_vector_scalar_div_op_assign,
    impl_vector_scalar_op, impl_vector_scalar_op_assign, impl_vector_unary_op,
    types::orientation::Column, types::orientation::Row, types::orientation::VectorOrientation,
    types::traits::Transposable, types::traits::VectorBase, types::traits::VectorHasOrientation,
    types::traits::VectorOps, types::traits::VectorOpsComplex, types::traits::VectorOpsFloat,
    types::traits::VectorOrientationName,
};

use crate::types::utils::{
    angle_with_impl, chebyshev_distance_complex_impl, chebyshev_distance_impl,
    cosine_similarity_complex_impl, cosine_similarity_impl, cross_impl, distance_complex_impl,
    distance_impl, dot_impl, dot_to_f64_impl, hermitian_dot_impl, lerp_impl,
    manhattan_distance_complex_impl, manhattan_distance_impl, minkowski_distance_complex_impl,
    minkowski_distance_impl, mut_lerp_impl, mut_normalize_impl, mut_normalize_to_impl,
    mut_translate_impl, normalize_impl, normalize_to_impl, project_onto_impl, translate_impl,
};

use crate::errors::VectorError;

use num::{Complex, Zero};

/// A dynamic, heap-allocated vector type for n-dimensional real and complex scalar data.
///
/// The length of the vector is determined at runtime and stored on the heap.
/// This type is analogous to `Vector<T, N>` but supports dynamic sizing.
#[derive(Clone)]
pub struct FlexVector<T, O = Column> {
    /// ...
    pub components: Vec<T>,
    _orientation: PhantomData<O>,
}

/// ...
pub type FVector<T> = FlexVector<T, Column>;
/// ...
pub type ColFVector<T> = FlexVector<T, Column>;
/// ...
pub type RowFVector<T> = FlexVector<T, Row>;

// ================================
//
// Constructors
//
// ================================
impl<T, O> FlexVector<T, O> {
    /// Creates a new, empty FlexVector.
    #[inline]
    pub fn new() -> Self {
        Self { components: Vec::new(), _orientation: PhantomData }
    }

    /// Creates a new FlexVector with a pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self { components: Vec::with_capacity(capacity), _orientation: PhantomData }
    }

    /// Returns a new FlexVector of the given length, filled with zeros.
    #[inline]
    pub fn zero(len: usize) -> Self
    where
        T: num::Zero + Clone,
    {
        Self { components: vec![T::zero(); len], _orientation: PhantomData }
    }

    /// Returns a new FlexVector of the given length, filled with ones.
    #[inline]
    pub fn one(len: usize) -> Self
    where
        T: num::One + Clone,
    {
        Self { components: vec![T::one(); len], _orientation: PhantomData }
    }

    /// Returns a new FlexVector of the given length, filled with the given value.
    #[inline]
    pub fn filled(len: usize, value: T) -> Self
    where
        T: Clone,
    {
        Self { components: vec![value; len], _orientation: PhantomData }
    }

    /// Creates a new FlexVector from a slice.
    #[inline]
    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Clone,
    {
        Self { components: slice.to_vec(), _orientation: PhantomData }
    }

    /// Creates a FlexVector from a Vec.
    #[inline]
    pub fn from_vec(vec: Vec<T>) -> Self {
        Self { components: vec, _orientation: PhantomData }
    }

    /// Creates a FlexVector from a Cow<T>.
    #[inline]
    pub fn from_cow<'a>(data: impl Into<Cow<'a, [T]>>) -> Self
    where
        T: Clone + 'a,
    {
        Self::from(data.into())
    }

    /// Fallible error-propagating construction from an iterator of Results.
    #[inline]
    pub fn try_from_iter<I, E>(iter: I) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<T, E>>,
    {
        let components: Result<Vec<T>, E> = iter.into_iter().collect();
        components.map(|vec| FlexVector { components: vec, _orientation: PhantomData })
    }

    /// Creates a new [`FlexVector`] by calling the provided function or closure for each index.
    ///
    /// # Example
    /// ```
    /// use vectora::prelude::*;
    ///
    /// let v = FlexVector::<usize>::from_fn(4, |i| i * i);
    /// assert_eq!(v.as_slice(), &[0, 1, 4, 9]);
    /// ```
    #[inline]
    pub fn from_fn<F>(len: usize, f: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        let components = (0..len).map(f).collect();
        FlexVector { components, _orientation: PhantomData }
    }

    /// Fallibly creates a new [`FlexVector`] by calling the provided function or closure for each index.
    ///
    /// Returns the first error encountered, or a [`FlexVector`] of all results if successful.
    ///
    /// # Example
    /// ```
    /// use vectora::prelude::*;
    ///
    /// let v = FlexVector::<i32>::try_from_fn(3, |i| if i == 1 { Err("fail") } else { Ok(i as i32) });
    /// assert!(v.is_err());
    /// ```
    #[inline]
    pub fn try_from_fn<F, E>(len: usize, mut f: F) -> Result<Self, E>
    where
        F: FnMut(usize) -> Result<T, E>,
    {
        let mut components = Vec::with_capacity(len);
        for i in 0..len {
            components.push(f(i)?);
        }
        Ok(FlexVector { components, _orientation: PhantomData })
    }

    /// Returns a new FlexVector by repeating the pattern until length `len` is reached.
    /// Returns an error if `pattern` is empty and `len` > 0.
    #[inline]
    pub fn repeat_pattern(pattern: &[T], len: usize) -> Result<Self, VectorError>
    where
        T: Clone,
    {
        if pattern.is_empty() && len > 0 {
            return Err(VectorError::ValueError(
                "pattern must not be empty if len > 0".to_string(),
            ));
        }
        let components = pattern.iter().cloned().cycle().take(len).collect();
        Ok(FlexVector { components, _orientation: PhantomData })
    }
}

// ================================
//
// Default trait impl
//
// ================================
impl<T, O> Default for FlexVector<T, O> {
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
impl<T, O> std::fmt::Display for FlexVector<T, O>
where
    T: std::fmt::Debug,
    O: VectorOrientationName + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} FlexVector {:?}", O::orientation_name(), self.components)
    }
}

// ================================
//
// Debug trait impl
//
// ================================
impl<T, O> std::fmt::Debug for FlexVector<T, O>
where
    T: std::fmt::Debug,
    O: VectorOrientationName + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlexVector")
            .field("orientation", &O::orientation_name())
            .field("components", &self.components)
            .finish()
    }
}

// ================================
//
// FromIterator trait impl
//
// ================================
impl<T, O> FromIterator<T> for FlexVector<T, O> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        FlexVector { components: iter.into_iter().collect(), _orientation: PhantomData }
    }
}

// ================================
//
// Deref/DerefMut trait impl
//
// ================================
impl<T, O> Deref for FlexVector<T, O> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.components
    }
}

impl<T, O> DerefMut for FlexVector<T, O> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.components
    }
}

// ================================
//
// AsRef/AsMut trait impl
//
// ================================
impl<T, O> AsRef<[T]> for FlexVector<T, O> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.components
    }
}
impl<T, O> AsMut<[T]> for FlexVector<T, O> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.components
    }
}

// ================================
//
// Borrow/BorrowMut trait impl
//
// ================================
impl<T, O> Borrow<[T]> for FlexVector<T, O> {
    #[inline]
    fn borrow(&self) -> &[T] {
        &self.components
    }
}
impl<T, O> BorrowMut<[T]> for FlexVector<T, O> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        &mut self.components
    }
}

// ================================
//
// IntoIterator trait impl
//
// ================================
impl<T, O> IntoIterator for FlexVector<T, O> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.components.into_iter()
    }
}
impl<'a, T, O> IntoIterator for &'a FlexVector<T, O> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.components.iter()
    }
}
impl<'a, T, O> IntoIterator for &'a mut FlexVector<T, O> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.components.iter_mut()
    }
}

// ================================
//
// PartialEq/Eq trait impl
//
// ================================
impl<T, O> PartialEq for FlexVector<T, O>
where
    T: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.components == other.components
    }
}
impl<T, O> Eq for FlexVector<T, O> where T: Eq {}

// ================================
//
// PartialOrd/Ord trait impl
//
// ================================
impl<T, O> PartialOrd for FlexVector<T, O>
where
    T: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.components.partial_cmp(&other.components)
    }
}
impl<T, O> Ord for FlexVector<T, O>
where
    T: Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.components.cmp(&other.components)
    }
}

// ================================
//
// Hash trait impl
//
// ================================
impl<T, O> std::hash::Hash for FlexVector<T, O>
where
    T: std::hash::Hash,
{
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.components.hash(state)
    }
}

// ================================
//
// From trait impl
//
// ================================
impl<T, O> From<Vec<T>> for FlexVector<T, O> {
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        FlexVector { components: vec, _orientation: PhantomData }
    }
}
impl<T, O> From<&[T]> for FlexVector<T, O>
where
    T: Clone,
{
    #[inline]
    fn from(slice: &[T]) -> Self {
        FlexVector { components: slice.to_vec(), _orientation: PhantomData }
    }
}

impl<T, O> From<Cow<'_, [T]>> for FlexVector<T, O>
where
    T: Clone,
    [T]: ToOwned<Owned = Vec<T>>,
{
    #[inline]
    fn from(cow: Cow<'_, [T]>) -> Self {
        match cow {
            Cow::Borrowed(slice) => FlexVector::from_slice(slice),
            Cow::Owned(vec) => FlexVector::from_vec(vec),
        }
    }
}

impl<T, O> From<&Cow<'_, [T]>> for FlexVector<T, O>
where
    T: Clone,
{
    #[inline]
    fn from(cow: &Cow<'_, [T]>) -> Self {
        match cow {
            Cow::Borrowed(slice) => FlexVector::from_slice(slice),
            Cow::Owned(vec) => FlexVector::from_slice(vec),
        }
    }
}

impl<T> From<FlexVector<T, Column>> for FlexVector<T, Row> {
    #[inline]
    fn from(v: FlexVector<T, Column>) -> Self {
        FlexVector { components: v.components, _orientation: PhantomData }
    }
}
impl<T> From<FlexVector<T, Row>> for FlexVector<T, Column> {
    #[inline]
    fn from(v: FlexVector<T, Row>) -> Self {
        FlexVector { components: v.components, _orientation: PhantomData }
    }
}

// ================================
//
// Index trait impl
//
// ================================
impl<T, O> Index<usize> for FlexVector<T, O> {
    type Output = T;

    #[inline]
    fn index(&self, idx: usize) -> &Self::Output {
        &self.components[idx]
    }
}

impl<T, O> Index<Range<usize>> for FlexVector<T, O> {
    type Output = [T];

    #[inline]
    fn index(&self, range: Range<usize>) -> &Self::Output {
        &self.components[range]
    }
}

impl<T, O> Index<RangeFrom<usize>> for FlexVector<T, O> {
    type Output = [T];

    #[inline]
    fn index(&self, range: RangeFrom<usize>) -> &Self::Output {
        &self.components[range]
    }
}

impl<T, O> Index<RangeTo<usize>> for FlexVector<T, O> {
    type Output = [T];

    #[inline]
    fn index(&self, range: RangeTo<usize>) -> &Self::Output {
        &self.components[range]
    }
}

impl<T, O> Index<RangeFull> for FlexVector<T, O> {
    type Output = [T];

    #[inline]
    fn index(&self, range: RangeFull) -> &Self::Output {
        &self.components[range]
    }
}

impl<T, O> Index<RangeInclusive<usize>> for FlexVector<T, O> {
    type Output = [T];

    #[inline]
    fn index(&self, range: RangeInclusive<usize>) -> &Self::Output {
        &self.components[range]
    }
}

impl<T, O> Index<RangeToInclusive<usize>> for FlexVector<T, O> {
    type Output = [T];

    #[inline]
    fn index(&self, range: RangeToInclusive<usize>) -> &Self::Output {
        &self.components[range]
    }
}

// ================================
//
// IndexMut trait impl
//
// ================================
impl<T, O> IndexMut<usize> for FlexVector<T, O> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.components[idx]
    }
}

impl<T, O> IndexMut<Range<usize>> for FlexVector<T, O> {
    #[inline]
    fn index_mut(&mut self, range: Range<usize>) -> &mut Self::Output {
        &mut self.components[range]
    }
}

impl<T, O> IndexMut<RangeFrom<usize>> for FlexVector<T, O> {
    #[inline]
    fn index_mut(&mut self, range: RangeFrom<usize>) -> &mut Self::Output {
        &mut self.components[range]
    }
}

impl<T, O> IndexMut<RangeTo<usize>> for FlexVector<T, O> {
    #[inline]
    fn index_mut(&mut self, range: RangeTo<usize>) -> &mut Self::Output {
        &mut self.components[range]
    }
}

impl<T, O> IndexMut<RangeFull> for FlexVector<T, O> {
    #[inline]
    fn index_mut(&mut self, range: RangeFull) -> &mut Self::Output {
        &mut self.components[range]
    }
}

impl<T, O> IndexMut<RangeInclusive<usize>> for FlexVector<T, O> {
    #[inline]
    fn index_mut(&mut self, range: RangeInclusive<usize>) -> &mut Self::Output {
        &mut self.components[range]
    }
}

impl<T, O> IndexMut<RangeToInclusive<usize>> for FlexVector<T, O> {
    #[inline]
    fn index_mut(&mut self, range: RangeToInclusive<usize>) -> &mut Self::Output {
        &mut self.components[range]
    }
}

// ================================
//
// Extend trait impl
//
// ================================
impl<T, O> Extend<T> for FlexVector<T, O> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.components.extend(iter)
    }
}

// ================================
//
// VectorBase trait impl
//
// ================================
impl<T, O> VectorBase<T> for FlexVector<T, O> {
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
// Transposable trait impl
//
// ================================

impl<T> Transposable for FlexVector<T, Row> {
    type Transposed = FlexVector<T, Column>;

    #[inline]
    fn transpose(self) -> Self::Transposed {
        FlexVector { components: self.components, _orientation: PhantomData }
    }
}

impl<T> Transposable for FlexVector<T, Column> {
    type Transposed = FlexVector<T, Row>;

    #[inline]
    fn transpose(self) -> Self::Transposed {
        FlexVector { components: self.components, _orientation: PhantomData }
    }
}

// ================================
//
// VectorHasOrientation trait impl
//
// ================================

impl<T, O> VectorHasOrientation for FlexVector<T, O>
where
    O: 'static,
{
    #[inline]
    fn orientation(&self) -> VectorOrientation {
        if self.is_column() {
            VectorOrientation::Column
        } else {
            VectorOrientation::Row
        }
    }
}

// ================================
//
// VectorOps trait impl
//
// ================================
impl<T, O> VectorOps<T> for FlexVector<T, O>
where
    T: Clone,
{
    type Output = Self;

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
    fn mut_translate(&mut self, other: &Self) -> Result<(), VectorError>
    where
        T: num::Num + Copy,
    {
        self.check_same_length_and_raise(other)?;
        mut_translate_impl(self.as_mut_slice(), other.as_slice());
        Ok(())
    }

    #[inline]
    fn scale(&self, scalar: T) -> Self::Output
    where
        T: num::Num + Clone,
        Self::Output: std::iter::FromIterator<T>,
    {
        self.as_slice().iter().map(|a| a.clone() * scalar.clone()).collect()
    }

    #[inline]
    fn negate(&self) -> Self::Output
    where
        T: std::ops::Neg<Output = T> + Clone,
        Self::Output: std::iter::FromIterator<T>,
    {
        self.as_slice().iter().map(|a| -(a.clone())).collect()
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

    /// Cross product (only for 3D vectors).
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

    /// Element-wise min
    #[inline]
    fn elementwise_min(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: PartialOrd + Clone,
    {
        self.check_same_length_and_raise(other)?;
        let components = self
            .as_slice()
            .iter()
            .zip(other.as_slice())
            .map(|(a, b)| if a < b { a.clone() } else { b.clone() })
            .collect();
        Ok(FlexVector { components, _orientation: PhantomData })
    }

    /// Element-wise max
    #[inline]
    fn elementwise_max(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: PartialOrd + Clone,
    {
        self.check_same_length_and_raise(other)?;
        let components = self
            .as_slice()
            .iter()
            .zip(other.as_slice())
            .map(|(a, b)| if a > b { a.clone() } else { b.clone() })
            .collect();
        Ok(FlexVector { components, _orientation: PhantomData })
    }
}

// ================================
//
// VectorOpsFloat trait impl
//
// ================================
impl<T, O> VectorOpsFloat<T> for FlexVector<T, O>
where
    T: num::Float + Clone + std::iter::Sum<T>,
{
    type Output = Self;

    #[inline]
    fn normalize(&self) -> Result<Self::Output, VectorError>
    where
        T: Copy + PartialEq + std::ops::Div<T, Output = T> + num::Zero,
    {
        normalize_impl(self.as_slice(), self.norm())
    }

    #[inline]
    fn mut_normalize(&mut self) -> Result<(), VectorError>
    where
        T: Copy + PartialEq + std::ops::Div<T, Output = T> + num::Zero,
    {
        let norm = self.norm();
        mut_normalize_impl(self.as_mut_slice(), norm)
    }

    /// Returns a new vector with the same direction and the given magnitude.
    #[inline]
    fn normalize_to(&self, magnitude: T) -> Result<Self::Output, VectorError>
    where
        T: Copy
            + PartialEq
            + std::ops::Div<T, Output = T>
            + std::ops::Mul<T, Output = T>
            + num::Zero,
        Self::Output: std::iter::FromIterator<T>,
    {
        normalize_to_impl(self.as_slice(), self.norm(), magnitude)
    }

    #[inline]
    fn mut_normalize_to(&mut self, magnitude: T) -> Result<(), VectorError>
    where
        T: Copy
            + PartialEq
            + std::ops::Div<T, Output = T>
            + std::ops::Mul<T, Output = T>
            + num::Zero,
    {
        let n = self.norm();
        mut_normalize_to_impl(self.as_mut_slice(), n, magnitude)
    }

    #[inline]
    fn lerp(&self, end: &Self, weight: T) -> Result<Self::Output, VectorError>
    where
        T: num::Float + Copy,
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
    fn mut_lerp(&mut self, end: &Self, weight: T) -> Result<(), VectorError>
    where
        T: num::Float + Copy + PartialOrd,
    {
        self.check_same_length_and_raise(end)?;
        if weight < T::zero() || weight > T::one() {
            return Err(VectorError::OutOfRangeError("weight must be in [0, 1]".to_string()));
        }
        mut_lerp_impl(self.as_mut_slice(), end.as_slice(), weight);
        Ok(())
    }

    #[inline]
    fn midpoint(&self, other: &Self) -> Result<Self::Output, VectorError>
    where
        T: num::Float + Clone,
    {
        self.check_same_length_and_raise(other)?;
        let mut out = FlexVector::zero(self.len());
        lerp_impl(self.as_slice(), other.as_slice(), T::from(0.5).unwrap(), out.as_mut_slice());
        Ok(out)
    }

    #[inline]
    fn distance(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + Clone + std::iter::Sum<T>,
    {
        self.check_same_length_and_raise(other)?;
        Ok(distance_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn manhattan_distance(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + Clone + std::iter::Sum<T>,
    {
        self.check_same_length_and_raise(other)?;
        Ok(manhattan_distance_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn chebyshev_distance(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + Clone + PartialOrd,
    {
        self.check_same_length_and_raise(other)?;
        Ok(chebyshev_distance_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn minkowski_distance(&self, other: &Self, p: T) -> Result<T, VectorError>
    where
        T: num::Float + Clone + std::iter::Sum<T>,
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
        T: num::Float + Clone + std::iter::Sum<T>,
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
        T: num::Float + Clone + std::iter::Sum<T>,
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
    fn cosine_similarity(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + Clone + std::iter::Sum<T> + std::ops::Div<Output = T>,
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

// ================================
//
// VectorOpsComplex trait impl
//
// ================================
// TODO: add tests
impl<N, O> VectorOpsComplex<N> for FlexVector<Complex<N>, O>
where
    N: num::Float + Clone + std::iter::Sum<N>,
{
    type Output = Self;

    #[inline]
    fn normalize(&self) -> Result<Self::Output, VectorError>
    where
        Complex<N>: Copy + PartialEq + std::ops::Div<Complex<N>, Output = Complex<N>>,
        Self::Output: std::iter::FromIterator<Complex<N>>,
    {
        normalize_impl(self.as_slice(), Complex::new(self.norm(), N::zero()))
    }

    #[inline]
    fn mut_normalize(&mut self) -> Result<(), VectorError>
    where
        Complex<N>: Copy + PartialEq + std::ops::Div<Complex<N>, Output = Complex<N>> + num::Zero,
    {
        let norm = self.norm();
        mut_normalize_impl(self.as_mut_slice(), Complex::new(norm, N::zero()))
    }

    #[inline]
    fn normalize_to(&self, magnitude: N) -> Result<Self::Output, VectorError>
    where
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
    fn mut_normalize_to(&mut self, magnitude: N) -> Result<(), VectorError>
    where
        Complex<N>: Copy
            + PartialEq
            + std::ops::Div<Complex<N>, Output = Complex<N>>
            + std::ops::Mul<Complex<N>, Output = Complex<N>>
            + num::Zero,
    {
        let n = self.norm();
        mut_normalize_to_impl(
            self.as_mut_slice(),
            Complex::new(n, N::zero()),
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
        N: num::Float + Clone + PartialOrd,
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
    fn mut_lerp(&mut self, end: &Self, weight: N) -> Result<(), VectorError>
    where
        N: num::Float + Copy + PartialOrd,
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
        mut_lerp_impl(self.as_mut_slice(), end.as_slice(), w);
        Ok(())
    }

    #[inline]
    fn midpoint(&self, end: &Self) -> Result<Self::Output, VectorError>
    where
        N: num::Float + Clone,
    {
        self.check_same_length_and_raise(end)?;
        self.lerp(end, num::cast(0.5).unwrap())
    }

    #[inline]
    fn distance(&self, other: &Self) -> Result<N, VectorError>
    where
        N: num::Float + Clone + std::iter::Sum<N>,
    {
        self.check_same_length_and_raise(other)?;
        Ok(distance_complex_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn manhattan_distance(&self, other: &Self) -> Result<N, VectorError>
    where
        N: num::Float + Clone + std::iter::Sum<N>,
    {
        self.check_same_length_and_raise(other)?;
        Ok(manhattan_distance_complex_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn chebyshev_distance(&self, other: &Self) -> Result<N, VectorError>
    where
        N: num::Float + Clone + PartialOrd,
    {
        self.check_same_length_and_raise(other)?;
        Ok(chebyshev_distance_complex_impl(self.as_slice(), other.as_slice()))
    }

    #[inline]
    fn minkowski_distance(&self, other: &Self, p: N) -> Result<N, VectorError>
    where
        N: num::Float + Clone + std::iter::Sum<N>,
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
        N: num::Float + Clone + std::iter::Sum<N> + std::ops::Neg<Output = N>,
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
        N: num::Float + Clone + std::iter::Sum<N> + std::ops::Neg<Output = N>,
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
impl<T, O> FlexVector<T, O> {
    /// ...
    #[inline]
    pub fn is_row(&self) -> bool
    where
        O: 'static,
    {
        TypeId::of::<O>() == TypeId::of::<Row>()
    }

    /// ...
    #[inline]
    pub fn is_column(&self) -> bool
    where
        O: 'static,
    {
        TypeId::of::<O>() == TypeId::of::<Column>()
    }

    /// ...
    #[inline]
    pub fn as_row(&self) -> FlexVector<T, Row>
    where
        T: Clone,
    {
        FlexVector { components: self.components.clone(), _orientation: PhantomData }
    }

    /// ...
    #[inline]
    pub fn as_column(&self) -> FlexVector<T, Column>
    where
        T: Clone,
    {
        FlexVector { components: self.components.clone(), _orientation: PhantomData }
    }

    /// Consumes self and returns a Row-oriented FlexVector.
    #[inline]
    pub fn into_row(self) -> FlexVector<T, crate::types::orientation::Row> {
        FlexVector { components: self.components, _orientation: std::marker::PhantomData }
    }

    /// Consumes self and returns a Column-oriented FlexVector.
    #[inline]
    pub fn into_column(self) -> FlexVector<T, crate::types::orientation::Column> {
        FlexVector { components: self.components, _orientation: std::marker::PhantomData }
    }

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
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
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
    pub fn map<U, F>(&self, mut f: F) -> FlexVector<U, O>
    where
        F: FnMut(T) -> U,
        T: Clone,
    {
        let components = self.components.iter().cloned().map(&mut f).collect();
        FlexVector { components, _orientation: PhantomData }
    }

    /// Applies a closure or function to each element, modifying them in place.
    #[inline]
    pub fn mut_map<F>(&mut self, mut f: F)
    where
        F: FnMut(T) -> T,
        T: Clone,
    {
        for x in self.components.iter_mut() {
            *x = f(x.clone());
        }
    }

    /// Returns a new FlexVector by mapping each element to an iterator and flattening the result.
    #[inline]
    pub fn flat_map<U, F, I>(self, mut f: F) -> FlexVector<U, O>
    where
        F: FnMut(T) -> I,
        I: IntoIterator<Item = U>,
    {
        let components = self.components.into_iter().flat_map(&mut f).collect();
        FlexVector { components, _orientation: PhantomData }
    }

    /// Returns a new FlexVector containing only the elements that satisfy the predicate.
    #[inline]
    pub fn filter<F>(&self, mut predicate: F) -> Self
    where
        F: FnMut(&T) -> bool,
        T: Clone,
    {
        let components = self.components.iter().filter(|&x| predicate(x)).cloned().collect();
        FlexVector { components, _orientation: PhantomData }
    }

    /// Reduces the elements to a single value using the provided closure, or returns None if empty.
    #[inline]
    pub fn reduce<F>(&self, mut f: F) -> Option<T>
    where
        F: FnMut(T, T) -> T,
        T: Clone,
    {
        let mut iter = self.components.iter().cloned();
        let first = iter.next()?;
        Some(iter.fold(first, &mut f))
    }

    /// Folds the elements using an initial value and a closure.
    #[inline]
    pub fn fold<B, F>(&self, init: B, mut f: F) -> B
    where
        F: FnMut(B, &T) -> B,
    {
        self.components.iter().fold(init, &mut f)
    }

    /// Zips two FlexVectors into a FlexVector of pairs.
    #[inline]
    pub fn zip<U>(self, other: FlexVector<U>) -> FlexVector<(T, U), O> {
        let len = self.len().min(other.len());
        let components = self.components.into_iter().zip(other.components).take(len).collect();
        FlexVector { components, _orientation: PhantomData }
    }

    /// Zips two FlexVectors with a function, producing a FlexVector of the function's output.
    #[inline]
    pub fn zip_with<U, R, F>(self, other: FlexVector<U>, mut f: F) -> FlexVector<R, O>
    where
        F: FnMut(T, U) -> R,
    {
        let len = self.len().min(other.len());
        let components = self
            .components
            .into_iter()
            .zip(other.components)
            .map(|(a, b)| f(a, b))
            .take(len)
            .collect();
        FlexVector { components, _orientation: PhantomData }
    }

    /// Returns a new FlexVector containing every `step`th element, starting from the first.
    /// Returns an error if step == 0.
    #[inline]
    pub fn step_by(&self, step: usize) -> Result<Self, VectorError>
    where
        T: Clone,
    {
        if step == 0 {
            return Err(VectorError::ValueError("step must be non-zero".to_string()));
        }
        let components = self.components.iter().step_by(step).cloned().collect();
        Ok(FlexVector { components, _orientation: PhantomData })
    }

    /// Returns a boolean vector mask where each element is the result of applying the predicate to the corresponding element.
    #[inline]
    pub fn create_mask<F>(&self, predicate: F) -> FlexVector<bool, O>
    where
        F: FnMut(&T) -> bool,
    {
        self.components.iter().map(predicate).collect()
    }

    /// Returns a new FlexVector containing elements where the corresponding mask value is true.
    #[inline]
    pub fn filter_by_mask(&self, mask: &[bool]) -> Result<Self, VectorError>
    where
        T: Clone,
    {
        if self.len() != mask.len() {
            return Err(VectorError::MismatchedLengthError(
                "Mask length must match vector length".to_string(),
            ));
        }
        let components = self
            .components
            .iter()
            .zip(mask)
            .filter_map(|(x, &m)| if m { Some(x.clone()) } else { None })
            .collect();
        Ok(FlexVector { components, _orientation: PhantomData })
    }

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

impl<T, O> FlexVector<&T, O>
where
    T: Clone,
{
    /// Returns a FlexVector of owned values by cloning each referenced element.
    #[inline]
    pub fn cloned(&self) -> FlexVector<T, O> {
        FlexVector::from_vec(self.components.iter().map(|&x| x.clone()).collect())
    }
}

impl<T, V, O> FlexVector<V, O>
where
    V: IntoIterator<Item = T>,
{
    /// Flattens a FlexVector of iterables into a single FlexVector by concatenating all elements.
    #[inline]
    pub fn flatten(self) -> FlexVector<T, O> {
        let components = self.components.into_iter().flatten().collect();
        FlexVector { components, _orientation: PhantomData }
    }
}

impl<'a, V, T, O> FlexVector<V, O>
where
    V: IntoIterator<Item = &'a T>,
    T: Clone + 'a,
{
    /// Flattens a FlexVector of iterables into a single FlexVector by concatenating all elements as cloned elements.
    #[inline]
    pub fn flatten_cloned(self) -> FlexVector<T, O> {
        let components = self.components.into_iter().flat_map(|v| v.into_iter().cloned()).collect();
        FlexVector { components, _orientation: PhantomData }
    }
}

// ================================
//
// Operator overload trait impl
//
// ================================
impl_vector_unary_op!(FlexVector, Neg, neg, -);

impl_vector_binop!(check_len, FlexVector, Add, add, +);
impl_vector_binop!(check_len, FlexVector, Sub, sub, -);
impl_vector_binop!(check_len, FlexVector, Mul, mul, *);
impl_vector_binop_div!(check_len, FlexVector);

impl_vector_binop_assign!(check_len, FlexVector, AddAssign, add_assign, +);
impl_vector_binop_assign!(check_len, FlexVector, SubAssign, sub_assign, -);
impl_vector_binop_assign!(check_len, FlexVector, MulAssign, mul_assign, *);
impl_vector_binop_div_assign!(check_len, FlexVector);

impl_vector_scalar_op!(FlexVector, Mul, mul, *);
impl_vector_scalar_op_assign!(FlexVector, MulAssign, mul_assign, *);

impl_vector_scalar_div_op!(FlexVector);
impl_vector_scalar_div_op_assign!(FlexVector);

#[cfg(test)]
mod tests {
    use super::*;
    use num::complex::ComplexFloat;
    use num::Complex;
    use std::borrow::Cow;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // ================================
    //
    // Test utility functions
    //
    // ================================
    fn double(x: i32) -> i32 {
        x * 2
    }

    fn square(x: i32) -> i32 {
        x * x
    }

    fn square_usize(x: usize) -> usize {
        x * x
    }

    // ================================
    //
    // Constructor tests
    //
    // ================================
    #[test]
    fn test_new_default() {
        let v = FlexVector::<i32>::new();
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
    }

    #[test]
    fn test_new_explicit_column() {
        let v: FlexVector<i32, Column> = FlexVector::new();
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
    }

    #[test]
    fn test_new_explicit_row() {
        let v: FlexVector<i32, Row> = FlexVector::new();
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
    fn test_from_cow_i32_constructor() {
        use std::borrow::Cow;
        // Borrowed
        let slice: &[i32] = &[10, 20, 30];
        let fv = FlexVector::<i32>::from_cow(Cow::Borrowed(slice));
        assert_eq!(fv.as_slice(), slice);

        // Owned
        let vec = vec![40, 50, 60];
        let fv = FlexVector::<i32>::from_cow(Cow::Owned(vec.clone()));
        assert_eq!(fv.as_slice(), &vec[..]);

        // Direct slice
        let fv = FlexVector::<i32>::from_cow(slice);
        assert_eq!(fv.as_slice(), slice);

        // Direct Vec
        let fv = FlexVector::<i32>::from_cow(vec.clone());
        assert_eq!(fv.as_slice(), &vec[..]);
    }

    #[test]
    fn test_from_cow_f64_constructor() {
        use std::borrow::Cow;
        // Borrowed
        let slice: &[f64] = &[1.5, 2.5, 3.5];
        let fv = FlexVector::<f64>::from_cow(Cow::Borrowed(slice));
        assert_eq!(fv.as_slice(), slice);

        // Owned
        let vec = vec![4.5, 5.5, 6.5];
        let fv = FlexVector::<f64>::from_cow(Cow::Owned(vec.clone()));
        assert_eq!(fv.as_slice(), &vec[..]);

        // Direct slice
        let fv = FlexVector::<f64>::from_cow(slice);
        assert_eq!(fv.as_slice(), slice);

        // Direct Vec
        let fv = FlexVector::<f64>::from_cow(vec.clone());
        assert_eq!(fv.as_slice(), &vec[..]);
    }

    #[test]
    fn test_from_cow_complex_f64_constructor() {
        use std::borrow::Cow;
        // Borrowed
        let slice: &[Complex<f64>] = &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let fv = FlexVector::<Complex<f64>>::from_cow(Cow::Borrowed(slice));
        assert_eq!(fv.as_slice(), slice);

        // Owned
        let vec = vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        let fv = FlexVector::<Complex<f64>>::from_cow(Cow::Owned(vec.clone()));
        assert_eq!(fv.as_slice(), &vec[..]);

        // Direct slice
        let fv = FlexVector::<Complex<f64>>::from_cow(slice);
        assert_eq!(fv.as_slice(), slice);

        // Direct Vec
        let fv = FlexVector::<Complex<f64>>::from_cow(vec.clone());
        assert_eq!(fv.as_slice(), &vec[..]);
    }

    #[test]
    fn test_from_fn_i32() {
        let v = FVector::from_fn(5, |i| i as i32 * 2);
        assert_eq!(v.as_slice(), &[0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_from_fn_f64() {
        let v = FVector::from_fn(4, |i| (i as f64).powi(2));
        assert_eq!(v.as_slice(), &[0.0, 1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_from_fn_complex() {
        let v = FVector::from_fn(3, |i| Complex::new(i as f64, -(i as f64)));
        assert_eq!(
            v.as_slice(),
            &[Complex::new(0.0, 0.0), Complex::new(1.0, -1.0), Complex::new(2.0, -2.0)]
        );
    }

    #[test]
    fn test_from_fn_zero_length() {
        let v: FlexVector<i32> = FVector::from_fn(0, |_| 42);
        assert!(v.is_empty());
    }

    #[test]
    fn test_from_fn_with_fn_pointer() {
        let v = FVector::from_fn(4, square_usize);
        assert_eq!(v.as_slice(), &[0, 1, 4, 9]);
    }

    #[test]
    fn test_try_from_fn_success() {
        let v = FlexVector::<i32>::try_from_fn(4, |i| Ok::<i32, ()>(i as i32 * 2)).unwrap();
        assert_eq!(v.as_slice(), &[0, 2, 4, 6]);
    }

    #[test]
    fn test_try_from_fn_error() {
        let result =
            FlexVector::<i32>::try_from_fn(5, |i| if i == 3 { Err("fail") } else { Ok(i as i32) });
        assert_eq!(result.unwrap_err(), "fail");
    }

    #[test]
    fn test_try_from_fn_empty() {
        let v = FlexVector::<i32>::try_from_fn(0, |_| Ok::<i32, ()>(42)).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn test_try_from_fn_with_fn_pointer() {
        fn fallible_square(i: usize) -> Result<i32, ()> {
            Ok((i * i) as i32)
        }
        let v = FlexVector::<i32>::try_from_fn(3, fallible_square).unwrap();
        assert_eq!(v.as_slice(), &[0, 1, 4]);
    }

    #[test]
    fn test_repeat_pattern_i32_basic() {
        let pattern = [1, 2, 3];
        let v = FVector::repeat_pattern(&pattern, 8).unwrap();
        assert_eq!(v.as_slice(), &[1, 2, 3, 1, 2, 3, 1, 2]);
    }

    #[test]
    fn test_repeat_pattern_i32_exact_multiple() {
        let pattern = [4, 5];
        let v = FVector::repeat_pattern(&pattern, 6).unwrap();
        assert_eq!(v.as_slice(), &[4, 5, 4, 5, 4, 5]);
    }

    #[test]
    fn test_repeat_pattern_i32_len_zero() {
        let pattern = [1, 2, 3];
        let v = FlexVector::<i32>::repeat_pattern(&pattern, 0).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn test_repeat_pattern_i32_pattern_empty_len_zero() {
        let pattern: [i32; 0] = [];
        let v = FVector::repeat_pattern(&pattern, 0).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn test_repeat_pattern_i32_pattern_empty_len_nonzero() {
        let pattern: [i32; 0] = [];
        let result = FVector::repeat_pattern(&pattern, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_repeat_pattern_f64_basic() {
        let pattern = [1.5, 2.5];
        let v = FVector::repeat_pattern(&pattern, 5).unwrap();
        assert_eq!(v.as_slice(), &[1.5, 2.5, 1.5, 2.5, 1.5]);
    }

    #[test]
    fn test_repeat_pattern_f64_empty_pattern_len_zero() {
        let pattern: [f64; 0] = [];
        let v = FVector::repeat_pattern(&pattern, 0).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn test_repeat_pattern_f64_empty_pattern_len_nonzero() {
        let pattern: [f64; 0] = [];
        let result = FVector::repeat_pattern(&pattern, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_repeat_pattern_complex_f64_basic() {
        use num::Complex;
        let pattern = [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let v = FVector::repeat_pattern(&pattern, 5).unwrap();
        assert_eq!(
            v.as_slice(),
            &[
                Complex::new(1.0, 2.0),
                Complex::new(3.0, 4.0),
                Complex::new(1.0, 2.0),
                Complex::new(3.0, 4.0),
                Complex::new(1.0, 2.0)
            ]
        );
    }

    #[test]
    fn test_repeat_pattern_complex_f64_empty_pattern_len_zero() {
        use num::Complex;
        let pattern: [Complex<f64>; 0] = [];
        let v = FVector::repeat_pattern(&pattern, 0).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn test_repeat_pattern_complex_f64_empty_pattern_len_nonzero() {
        use num::Complex;
        let pattern: [Complex<f64>; 0] = [];
        let result = FVector::repeat_pattern(&pattern, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_i32() {
        let v_row = FlexVector::<i32, Row>::from_vec(vec![1, 2, 3]);
        let v_col = v_row.transpose();
        assert_eq!(v_col, FlexVector::<i32, Column>::from_vec(vec![1, 2, 3]));
        assert!(v_col.is_column());
        assert!(!v_col.is_row());

        let v_row2 = v_col.transpose();
        assert_eq!(v_row2, FlexVector::<i32, Row>::from_vec(vec![1, 2, 3]));
        assert!(v_row2.is_row());
        assert!(!v_row2.is_column());
    }

    #[test]
    fn test_transpose_f64() {
        let v_row = FlexVector::<f64, Row>::from_vec(vec![1.5, -2.0, 0.0]);
        let v_col = v_row.transpose();
        assert_eq!(v_col, FlexVector::<f64, Column>::from_vec(vec![1.5, -2.0, 0.0]));
        assert!(v_col.is_column());
        assert!(!v_col.is_row());

        let v_row2 = v_col.transpose();
        assert_eq!(v_row2, FlexVector::<f64, Row>::from_vec(vec![1.5, -2.0, 0.0]));
        assert!(v_row2.is_row());
        assert!(!v_row2.is_column());
    }

    #[test]
    fn test_transpose_complex_f64() {
        use num::Complex;
        let v_row = FlexVector::<Complex<f64>, Row>::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
        ]);
        let v_col = v_row.transpose();
        assert_eq!(
            v_col,
            FlexVector::<Complex<f64>, Column>::from_vec(vec![
                Complex::new(1.0, 2.0),
                Complex::new(3.0, 4.0)
            ])
        );
        assert!(v_col.is_column());
        assert!(!v_col.is_row());

        let v_row2 = v_col.transpose();
        assert_eq!(
            v_row2,
            FlexVector::<Complex<f64>, Row>::from_vec(vec![
                Complex::new(1.0, 2.0),
                Complex::new(3.0, 4.0)
            ])
        );
        assert!(v_row2.is_row());
        assert!(!v_row2.is_column());
    }

    #[test]
    fn test_orientation_i32_column_and_row() {
        let v_col = FlexVector::<i32, Column>::from_vec(vec![1, 2, 3]);
        let v_row = FlexVector::<i32, Row>::from_vec(vec![4, 5, 6]);
        assert_eq!(v_col.orientation(), VectorOrientation::Column);
        assert_eq!(v_row.orientation(), VectorOrientation::Row);
    }

    #[test]
    fn test_orientation_f64_column_and_row() {
        let v_col = FlexVector::<f64, Column>::from_vec(vec![1.1, 2.2]);
        let v_row = FlexVector::<f64, Row>::from_vec(vec![3.3, 4.4]);
        assert_eq!(v_col.orientation(), VectorOrientation::Column);
        assert_eq!(v_row.orientation(), VectorOrientation::Row);
    }

    #[test]
    fn test_orientation_complex_f64_column_and_row() {
        use num::Complex;
        let v_col = FlexVector::<Complex<f64>, Column>::from_vec(vec![Complex::new(1.0, 2.0)]);
        let v_row = FlexVector::<Complex<f64>, Row>::from_vec(vec![Complex::new(3.0, 4.0)]);
        assert_eq!(v_col.orientation(), VectorOrientation::Column);
        assert_eq!(v_row.orientation(), VectorOrientation::Row);
    }

    #[test]
    fn test_orientation_empty_vectors() {
        let v_col: FlexVector<i32, Column> = FlexVector::new();
        let v_row: FlexVector<i32, Row> = FlexVector::new();
        assert_eq!(v_col.orientation(), VectorOrientation::Column);
        assert_eq!(v_row.orientation(), VectorOrientation::Row);
    }

    #[test]
    fn test_orientation_default_type() {
        let v_default: FlexVector<i32> = FlexVector::new();
        // Default orientation is Column
        assert_eq!(v_default.orientation(), VectorOrientation::Column);
    }

    #[test]
    fn test_cloned_basic() {
        let a = 1;
        let b = 2;
        let c = 3;
        let refs = FVector::from_vec(vec![&a, &b, &c]);
        let owned = refs.cloned();
        assert_eq!(owned.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_cloned_empty() {
        let refs: FlexVector<&i32> = FlexVector::new();
        let owned = refs.cloned();
        assert!(owned.is_empty());
    }

    #[test]
    fn test_cloned_complex() {
        use num::Complex;
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let refs = FVector::from_vec(vec![&a, &b]);
        let owned = refs.cloned();
        assert_eq!(owned.as_slice(), &[a, b]);
    }

    #[test]
    fn test_flatten_flexvector_of_flexvector() {
        let row1 = FVector::from_vec(vec![1, 2]);
        let row2 = FVector::from_vec(vec![3, 4, 5]);
        let nested = FVector::from_vec(vec![row1, row2]);
        let flat = nested.flatten();
        assert_eq!(flat.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_flatten_flexvector_of_vec() {
        let nested = FVector::from_vec(vec![vec![10, 20], vec![30]]);
        let flat = nested.flatten();
        assert_eq!(flat.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_flatten_flexvector_of_array() {
        let nested = FVector::from_vec(vec![[1, 2], [3, 4]]);
        let flat = nested.flatten();
        assert_eq!(flat.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_flatten_flexvector_of_slice_refs() {
        let a = [7, 8];
        let b = [9];
        let nested = FVector::from_vec(vec![&a[..], &b[..]]);
        let flat = nested.flatten();
        assert_eq!(flat.as_slice(), &[&7, &8, &9]);
    }

    #[test]
    fn test_flatten_empty_outer() {
        let nested: FlexVector<Vec<i32>> = FlexVector::new();
        let flat = nested.flatten();
        assert!(flat.is_empty());
    }

    #[test]
    fn test_flatten_with_empty_inner() {
        let nested = FVector::from_vec(vec![vec![], vec![1, 2], vec![]]);
        let flat = nested.flatten();
        assert_eq!(flat.as_slice(), &[1, 2]);
    }

    #[test]
    fn test_flatten_cloned_flexvector_of_slice_refs() {
        let a = [8, 9];
        let b = [10];
        let nested = FVector::from_vec(vec![&a[..], &b[..]]);
        let flat = nested.flatten_cloned();
        let expected: Vec<i32> = vec![8, 9, 10];
        assert_eq!(flat.as_slice(), expected.as_slice());
        let _: &[i32] = flat.as_slice(); // type check: &[i32]
    }

    #[test]
    fn test_flatten_cloned_flexvector_of_refs() {
        let x = 42;
        let y = 43;
        let nested = FVector::from_vec(vec![vec![&x, &y], vec![&x]]);
        let flat = nested.flatten_cloned();
        assert_eq!(flat.as_slice(), &[42, 43, 42]);
        let _: &[i32] = flat.as_slice(); // type check: &[i32]
    }

    #[test]
    fn test_flatten_cloned_empty_outer() {
        let nested: FlexVector<Vec<&i32>> = FlexVector::new();
        let flat = nested.flatten_cloned();
        assert!(flat.is_empty());
    }

    #[test]
    fn test_flatten_cloned_with_empty_inner() {
        let nested = FVector::from_vec(vec![vec![], vec![&1, &2], vec![]]);
        let flat = nested.flatten_cloned();
        assert_eq!(flat.as_slice(), &[1, 2]);
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
    fn test_display_row_and_column_i32() {
        let v_col = FlexVector::<i32, Column>::from_vec(vec![1, 2, 3]);
        let v_row = FlexVector::<i32, Row>::from_vec(vec![4, 5, 6]);
        assert_eq!(format!("{}", v_col), "Column FlexVector [1, 2, 3]");
        assert_eq!(format!("{}", v_row), "Row FlexVector [4, 5, 6]");
    }

    #[test]
    fn test_display_row_and_column_f64() {
        let v_col = FlexVector::<f64, Column>::from_vec(vec![1.1, 2.2]);
        let v_row = FlexVector::<f64, Row>::from_vec(vec![3.3, 4.4]);
        assert_eq!(format!("{}", v_col), "Column FlexVector [1.1, 2.2]");
        assert_eq!(format!("{}", v_row), "Row FlexVector [3.3, 4.4]");
    }

    #[test]
    fn test_display_row_and_column_complex_f64() {
        use num::Complex;
        let v_col = FlexVector::<Complex<f64>, Column>::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
        ]);
        let v_row = FlexVector::<Complex<f64>, Row>::from_vec(vec![
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ]);
        assert_eq!(
            format!("{}", v_col),
            "Column FlexVector [Complex { re: 1.0, im: 2.0 }, Complex { re: 3.0, im: 4.0 }]"
        );
        assert_eq!(
            format!("{}", v_row),
            "Row FlexVector [Complex { re: 5.0, im: 6.0 }, Complex { re: 7.0, im: 8.0 }]"
        );
    }

    #[test]
    fn test_display_empty() {
        let v = FlexVector::<i32>::new();
        assert_eq!(format!("{}", v), "Column FlexVector []");
        let v_col = FlexVector::<i32, Column>::new();
        assert_eq!(format!("{}", v_col), "Column FlexVector []");
        let v_col = FlexVector::<i32, Row>::new();
        assert_eq!(format!("{}", v_col), "Row FlexVector []");
    }

    // ================================
    //
    // Debug trait tests
    //
    // ================================

    #[test]
    fn test_debug_row_and_column_i32() {
        let v_col = FlexVector::<i32, Column>::from_vec(vec![1, 2, 3]);
        let v_row = FlexVector::<i32, Row>::from_vec(vec![4, 5, 6]);
        let debug_col = format!("{:?}", v_col);
        let debug_row = format!("{:?}", v_row);
        assert!(debug_col.contains("FlexVector"));
        assert!(debug_col.contains("orientation"));
        assert!(debug_col.contains("Column"));
        assert!(debug_col.contains("components"));
        assert!(debug_col.contains("1"));
        assert!(debug_col.contains("2"));
        assert!(debug_col.contains("3"));
        assert!(debug_row.contains("FlexVector"));
        assert!(debug_row.contains("orientation"));
        assert!(debug_row.contains("Row"));
        assert!(debug_row.contains("components"));
        assert!(debug_row.contains("4"));
        assert!(debug_row.contains("5"));
        assert!(debug_row.contains("6"));
    }

    #[test]
    fn test_debug_row_and_column_f64() {
        let v_col = FlexVector::<f64, Column>::from_vec(vec![1.1, 2.2]);
        let v_row = FlexVector::<f64, Row>::from_vec(vec![3.3, 4.4]);
        let debug_col = format!("{:?}", v_col);
        let debug_row = format!("{:?}", v_row);
        assert!(debug_col.contains("FlexVector"));
        assert!(debug_col.contains("orientation"));
        assert!(debug_col.contains("Column"));
        assert!(debug_col.contains("components"));
        assert!(debug_col.contains("1.1"));
        assert!(debug_col.contains("2.2"));
        assert!(debug_row.contains("FlexVector"));
        assert!(debug_row.contains("orientation"));
        assert!(debug_row.contains("Row"));
        assert!(debug_row.contains("components"));
        assert!(debug_row.contains("3.3"));
        assert!(debug_row.contains("4.4"));
    }

    #[test]
    fn test_debug_row_and_column_complex_f64() {
        let v_col = FlexVector::<Complex<f64>, Column>::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
        ]);
        let v_row = FlexVector::<Complex<f64>, Row>::from_vec(vec![
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ]);
        let debug_col = format!("{:?}", v_col);
        let debug_row = format!("{:?}", v_row);
        assert!(debug_col.contains("FlexVector"));
        assert!(debug_col.contains("orientation"));
        assert!(debug_col.contains("Column"));
        assert!(debug_col.contains("components"));
        assert!(debug_col.contains("1.0"));
        assert!(debug_col.contains("2.0"));
        assert!(debug_row.contains("FlexVector"));
        assert!(debug_row.contains("orientation"));
        assert!(debug_row.contains("Row"));
        assert!(debug_row.contains("components"));
        assert!(debug_row.contains("5.0"));
        assert!(debug_row.contains("6.0"));
    }

    #[test]
    fn test_debug_empty_i32() {
        let v: FlexVector<i32> = FlexVector::new();
        let debug = format!("{:?}", v);
        assert!(debug.contains("FlexVector"));
        assert!(debug.contains("orientation"));
        assert!(debug.contains("Column")); // Default orientation is Column
        assert!(debug.contains("components"));
        assert!(debug.contains("[]"));
    }

    // ================================
    //
    // FromIterator trait tests
    //
    // ================================
    #[test]
    fn test_from_iter_i32() {
        let v: FlexVector<i32> = (1..4).collect();
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_from_iter_f64() {
        let v: FlexVector<f64> = vec![1.1, 2.2, 3.3].into_iter().collect();
        assert_eq!(v.as_slice(), &[1.1, 2.2, 3.3]);
    }

    #[test]
    fn test_from_iter_complex() {
        let data = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let v: FlexVector<Complex<f64>> = data.clone().into_iter().collect();
        assert_eq!(v.as_slice(), &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
    }

    #[test]
    fn test_from_iter_empty() {
        let v: FlexVector<i32> = Vec::<i32>::new().into_iter().collect();
        assert!(v.is_empty());
    }

    // ================================
    //
    // try_from_iter method tests
    //
    // ================================

    #[test]
    fn test_try_from_iter_success() {
        let data: Vec<Result<i32, ()>> = vec![Ok(1i32), Ok(2i32), Ok(3i32)];
        let fv = FlexVector::<i32>::try_from_iter(data).unwrap();
        assert_eq!(fv.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_try_from_iter_parse_success() {
        let data = vec!["1", "2", "3"];
        let fv = FlexVector::<i32>::try_from_iter(data.iter().map(|s| s.parse())).unwrap();
        assert_eq!(fv.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_try_from_iter_parse_error() {
        let data = vec!["1", "oops", "3"];
        let result = FlexVector::<i32>::try_from_iter(data.iter().map(|s| s.parse::<i32>()));
        assert!(result.is_err());
    }

    #[test]
    fn test_try_from_iter_custom_error() {
        #[derive(Debug, PartialEq)]
        enum MyError {
            Fail,
        }
        let data = vec![Ok(1), Err(MyError::Fail), Ok(3)];
        let result = FlexVector::<i32>::try_from_iter(data);
        assert_eq!(result.unwrap_err(), MyError::Fail);
    }

    #[test]
    fn test_try_from_iter_empty() {
        let data: Vec<Result<i32, &str>> = vec![];
        let fv = FlexVector::<i32>::try_from_iter(data).unwrap();
        assert!(fv.is_empty());
    }

    // ================================
    //
    // Deref/DerefMut trait tests
    //
    // ================================
    #[test]
    fn test_deref_access_slice_methods_i32() {
        let v = FVector::from_vec(vec![3, 1, 2]);
        // Use sort (not implemented in FlexVector directly)
        let mut sorted = v.clone();
        sorted.sort();
        assert_eq!(sorted.as_slice(), &[1, 2, 3]);
        // Use binary_search (slice method, requires sorted)
        assert_eq!(sorted.binary_search(&2), Ok(1));
    }

    #[test]
    fn test_deref_access_slice_methods_f64() {
        let v = FVector::from_vec(vec![3.5, 1.5, 2.5]);
        // Use sort_by
        let mut sorted = v.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted.as_slice(), &[1.5, 2.5, 3.5]);
        // Use rchunks
        let chunks: Vec<_> = v.rchunks(2).collect();
        assert_eq!(chunks, vec![&[1.5, 2.5][..], &[3.5][..]]);
    }

    #[test]
    fn test_deref_access_slice_methods_complex() {
        use num::Complex;
        let v = FVector::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ]);
        // Use rotate_left
        let mut rotated = v.clone();
        rotated.rotate_left(1);
        assert_eq!(
            rotated.as_slice(),
            &[Complex::new(3.0, 4.0), Complex::new(5.0, 6.0), Complex::new(1.0, 2.0)]
        );
        // Use fill
        let mut filled = v.clone();
        filled.fill(Complex::new(9.0, 9.0));
        assert_eq!(filled.as_slice(), &[Complex::new(9.0, 9.0); 3]);
    }

    #[test]
    fn test_deref_mut_i32() {
        let mut v = FVector::from_vec(vec![10, 20, 30]);
        // Mutate via indexing
        v[1] = 99;
        assert_eq!(v.as_slice(), &[10, 99, 30]);
        // Use reverse (slice method)
        v.reverse();
        assert_eq!(v.as_slice(), &[30, 99, 10]);
    }

    #[test]
    fn test_deref_mut_f64() {
        let mut v = FVector::from_vec(vec![1.5, 2.5, 3.5]);
        // Mutate via indexing
        v[0] = -1.5;
        assert_eq!(v.as_slice(), &[-1.5, 2.5, 3.5]);
        // Use fill (slice method)
        v.fill(0.0);
        assert_eq!(v.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_deref_mut_complex_f64() {
        use num::Complex;
        let mut v = FVector::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ]);
        // Mutate via indexing
        v[2] = Complex::new(9.0, 9.0);
        assert_eq!(v[2], Complex::new(9.0, 9.0));
        // Use rotate_right (slice method)
        v.rotate_right(1);
        assert_eq!(
            v.as_slice(),
            &[Complex::new(9.0, 9.0), Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]
        );
    }

    // ================================
    //
    // AsRef/AsMut trait tests
    //
    // ================================
    #[test]
    fn test_asref_i32() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let slice: &[i32] = v.as_ref();
        assert_eq!(slice, &[1, 2, 3]);
    }

    #[test]
    fn test_asref_f64() {
        let v = FVector::from_vec(vec![1.1, 2.2, 3.3]);
        let slice: &[f64] = v.as_ref();
        assert_eq!(slice, &[1.1, 2.2, 3.3]);
    }

    #[test]
    fn test_asref_complex_f64() {
        use num::Complex;
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let slice: &[Complex<f64>] = v.as_ref();
        assert_eq!(slice, &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
    }

    #[test]
    fn test_asmut_i32() {
        let mut v = FVector::from_vec(vec![1, 2, 3]);
        let slice: &mut [i32] = v.as_mut();
        slice[0] = 10;
        slice[2] = 30;
        assert_eq!(v.as_slice(), &[10, 2, 30]);
    }

    #[test]
    fn test_asmut_f64() {
        let mut v = FVector::from_vec(vec![1.1, 2.2, 3.3]);
        let slice: &mut [f64] = v.as_mut();
        slice[1] = 9.9;
        assert_eq!(v.as_slice(), &[1.1, 9.9, 3.3]);
    }

    #[test]
    fn test_asmut_complex_f64() {
        use num::Complex;
        let mut v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let slice: &mut [Complex<f64>] = v.as_mut();
        slice[0].re = 10.0;
        slice[1].im = 40.0;
        assert_eq!(v.as_slice(), &[Complex::new(10.0, 2.0), Complex::new(3.0, 40.0)]);
    }

    // ================================
    //
    // Borrow/BorrowMut trait method tests
    //
    // ================================

    #[test]
    fn test_borrow_i32() {
        use std::borrow::Borrow;
        let v = FVector::from_vec(vec![1, 2, 3]);
        let slice: &[i32] = v.borrow();
        assert_eq!(slice, &[1, 2, 3]);
    }

    #[test]
    fn test_borrow_f64() {
        use std::borrow::Borrow;
        let v = FVector::from_vec(vec![1.1, 2.2, 3.3]);
        let slice: &[f64] = v.borrow();
        assert_eq!(slice, &[1.1, 2.2, 3.3]);
    }

    #[test]
    fn test_borrow_complex_f64() {
        use num::Complex;
        use std::borrow::Borrow;
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let slice: &[Complex<f64>] = v.borrow();
        assert_eq!(slice, &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
    }

    #[test]
    fn test_borrow_mut_i32() {
        use std::borrow::BorrowMut;
        let mut v = FVector::from_vec(vec![1, 2, 3]);
        {
            let slice: &mut [i32] = v.borrow_mut();
            slice[0] = 10;
            slice[2] = 30;
        }
        assert_eq!(v.as_slice(), &[10, 2, 30]);
    }

    #[test]
    fn test_borrow_mut_f64() {
        use std::borrow::BorrowMut;
        let mut v = FVector::from_vec(vec![1.1, 2.2, 3.3]);
        {
            let slice: &mut [f64] = v.borrow_mut();
            slice[1] = 9.9;
        }
        assert_eq!(v.as_slice(), &[1.1, 9.9, 3.3]);
    }

    #[test]
    fn test_borrow_mut_complex_f64() {
        use num::Complex;
        use std::borrow::BorrowMut;
        let mut v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        {
            let slice: &mut [Complex<f64>] = v.borrow_mut();
            slice[0].re = 10.0;
            slice[1].im = 40.0;
        }
        assert_eq!(v.as_slice(), &[Complex::new(10.0, 2.0), Complex::new(3.0, 40.0)]);
    }

    // ================================
    //
    // IntoIterator trait method tests
    //
    // ================================
    #[test]
    fn test_into_iter_i32() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let collected: Vec<_> = v.into_iter().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn test_into_iter_f64() {
        let v = FVector::from_vec(vec![1.1, 2.2, 3.3]);
        let collected: Vec<_> = v.into_iter().collect();
        assert_eq!(collected, vec![1.1, 2.2, 3.3]);
    }

    #[test]
    fn test_into_iter_complex_f64() {
        use num::Complex;
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let collected: Vec<_> = v.into_iter().collect();
        assert_eq!(collected, vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
    }

    #[test]
    fn test_iter_ref_i32() {
        let v = FVector::from_vec(vec![10, 20, 30]);
        let collected: Vec<_> = (&v).into_iter().copied().collect();
        assert_eq!(collected, vec![10, 20, 30]);
    }

    #[test]
    fn test_iter_ref_f64() {
        let v = FVector::from_vec(vec![1.5, 2.5, 3.5]);
        let collected: Vec<_> = (&v).into_iter().copied().collect();
        assert_eq!(collected, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_iter_ref_complex_f64() {
        use num::Complex;
        let v = FVector::from_vec(vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)]);
        let collected: Vec<_> = (&v).into_iter().cloned().collect();
        assert_eq!(collected, vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)]);
    }

    #[test]
    fn test_iter_mutable_i32() {
        let mut v = FVector::from_vec(vec![1, 2, 3]);
        for x in &mut v {
            *x *= 10;
        }
        assert_eq!(v.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_iter_mutable_f64() {
        let mut v = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        for x in &mut v {
            *x += 0.5;
        }
        assert_eq!(v.as_slice(), &[1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_iter_mutable_complex_f64() {
        use num::Complex;
        let mut v = FVector::from_vec(vec![Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)]);
        for x in &mut v {
            x.re *= 2.0;
            x.im *= 3.0;
        }
        assert_eq!(v.as_slice(), &[Complex::new(2.0, 3.0), Complex::new(4.0, 6.0)]);
    }

    // ================================
    //
    // PartialEq/Eq trait method tests
    //
    // ================================
    #[test]
    fn test_partial_eq_i32() {
        let v1 = FVector::from_vec(vec![1, 2, 3]);
        let v2 = FVector::from_vec(vec![1, 2, 3]);
        let v3 = FVector::from_vec(vec![3, 2, 1]);
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_partial_eq_f64() {
        let v1 = FVector::from_vec(vec![1.1, 2.2, 3.3]);
        let v2 = FVector::from_vec(vec![1.1, 2.2, 3.3]);
        let v3 = FVector::from_vec(vec![3.3, 2.2, 1.1]);
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_partial_eq_complex_f64() {
        use num::Complex;
        let v1 = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let v2 = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let v3 = FVector::from_vec(vec![Complex::new(4.0, 3.0), Complex::new(2.0, 1.0)]);
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_partial_eq_empty() {
        let v1: FlexVector<i32> = FlexVector::new();
        let v2: FlexVector<i32> = FlexVector::new();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_partial_eq_different_lengths() {
        let v1 = FVector::from_vec(vec![1, 2]);
        let v2 = FVector::from_vec(vec![1, 2, 3]);
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_partial_eq_f64_nan() {
        let v1 = FVector::from_vec(vec![f64::NAN, 1.0]);
        let v2 = FVector::from_vec(vec![f64::NAN, 1.0]);
        // NaN != NaN, so these should not be equal
        assert_ne!(v1, v2);

        let v3 = FVector::from_vec(vec![f64::NAN, 1.0]);
        let v4 = FVector::from_vec(vec![f64::NAN, 2.0]);
        assert_ne!(v3, v4);
    }

    #[test]
    fn test_partial_eq_f64_zero_negzero() {
        let v1 = FVector::from_vec(vec![0.0, -0.0]);
        let v2 = FVector::from_vec(vec![0.0, -0.0]);
        let v3 = FVector::from_vec(vec![-0.0, 0.0]);
        // 0.0 == -0.0 in Rust
        assert_eq!(v1, v2);
        assert_eq!(v1, v3);
    }

    #[test]
    fn test_partial_eq_f64_infinity() {
        let v1 = FVector::from_vec(vec![f64::INFINITY, f64::NEG_INFINITY]);
        let v2 = FVector::from_vec(vec![f64::INFINITY, f64::NEG_INFINITY]);
        let v3 = FVector::from_vec(vec![f64::NEG_INFINITY, f64::INFINITY]);
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_partial_eq_complex_nan() {
        use num::Complex;
        let nan = f64::NAN;
        let v1 = FVector::from_vec(vec![Complex::new(nan, 1.0)]);
        let v2 = FVector::from_vec(vec![Complex::new(nan, 1.0)]);
        // Complex::new(NaN, 1.0) != Complex::new(NaN, 1.0)
        assert_ne!(v1, v2);

        let v3 = FVector::from_vec(vec![Complex::new(1.0, nan)]);
        let v4 = FVector::from_vec(vec![Complex::new(1.0, nan)]);
        assert_ne!(v3, v4);
    }

    #[test]
    fn test_partial_eq_complex_zero_negzero() {
        use num::Complex;
        let v1 = FVector::from_vec(vec![Complex::new(0.0, -0.0)]);
        let v2 = FVector::from_vec(vec![Complex::new(-0.0, 0.0)]);
        // 0.0 == -0.0 for both real and imaginary parts
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_partial_eq_complex_infinity() {
        use num::Complex;
        let v1 = FVector::from_vec(vec![Complex::new(f64::INFINITY, 1.0)]);
        let v2 = FVector::from_vec(vec![Complex::new(f64::INFINITY, 1.0)]);
        let v3 = FVector::from_vec(vec![Complex::new(1.0, f64::INFINITY)]);
        let v4 = FVector::from_vec(vec![Complex::new(1.0, f64::INFINITY)]);
        assert_eq!(v1, v2);
        assert_eq!(v3, v4);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_eq_trait_i32() {
        let v1 = FVector::from_vec(vec![5, 6, 7]);
        let v2 = FVector::from_vec(vec![5, 6, 7]);
        assert!(v1.eq(&v2));
    }

    #[test]
    fn test_eq_trait_f64() {
        let v1 = FVector::from_vec(vec![0.0, -0.0]);
        let v2 = FVector::from_vec(vec![0.0, -0.0]);
        assert!(v1.eq(&v2));
    }

    #[test]
    fn test_eq_trait_complex_f64() {
        use num::Complex;
        let v1 = FVector::from_vec(vec![Complex::new(0.0, 1.0)]);
        let v2 = FVector::from_vec(vec![Complex::new(0.0, 1.0)]);
        assert!(v1.eq(&v2));
    }

    // ================================
    //
    // PartialOrd/Ord trait tests
    //
    // ================================
    #[test]
    fn test_partial_ord_i32() {
        let v1 = FVector::from_vec(vec![1, 2, 3]);
        let v2 = FVector::from_vec(vec![1, 2, 4]);
        let v3 = FVector::from_vec(vec![1, 2, 3]);
        assert!(v1 < v2);
        assert!(v2 > v1);
        assert!(v1 <= v3);
        assert!(v1 >= v3);
        assert_eq!(v1.partial_cmp(&v2), Some(std::cmp::Ordering::Less));
        assert_eq!(v2.partial_cmp(&v1), Some(std::cmp::Ordering::Greater));
        assert_eq!(v1.partial_cmp(&v3), Some(std::cmp::Ordering::Equal));
    }

    #[test]
    fn test_ord_i32() {
        let v1 = FVector::from_vec(vec![1, 2, 3]);
        let v2 = FVector::from_vec(vec![1, 2, 4]);
        let v3 = FVector::from_vec(vec![1, 2, 3]);
        assert_eq!(v1.cmp(&v2), std::cmp::Ordering::Less);
        assert_eq!(v2.cmp(&v1), std::cmp::Ordering::Greater);
        assert_eq!(v1.cmp(&v3), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_partial_ord_f64() {
        let v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FVector::from_vec(vec![1.0, 2.0, 4.0]);
        let v3 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(v1.partial_cmp(&v2), Some(std::cmp::Ordering::Less));
        assert_eq!(v2.partial_cmp(&v1), Some(std::cmp::Ordering::Greater));
        assert_eq!(v1.partial_cmp(&v3), Some(std::cmp::Ordering::Equal));
    }

    #[test]
    fn test_partial_ord_f64_nan() {
        let v1 = FVector::from_vec(vec![1.0, f64::NAN]);
        let v2 = FVector::from_vec(vec![1.0, 2.0]);
        // Comparison with NaN yields None
        assert_eq!(v1.partial_cmp(&v2), None);
        assert_eq!(v2.partial_cmp(&v1), None);
    }

    #[test]
    fn test_partial_ord_f64_infinity() {
        let v1 = FVector::from_vec(vec![1.0, f64::INFINITY]);
        let v2 = FVector::from_vec(vec![1.0, f64::NEG_INFINITY]);
        let v3 = FVector::from_vec(vec![1.0, f64::INFINITY]);
        let v4 = FVector::from_vec(vec![1.0, 1.0]);
        // INFINITY > NEG_INFINITY
        assert_eq!(v1.partial_cmp(&v2), Some(std::cmp::Ordering::Greater));
        assert_eq!(v2.partial_cmp(&v1), Some(std::cmp::Ordering::Less));
        // INFINITY == INFINITY
        assert_eq!(v1.partial_cmp(&v3), Some(std::cmp::Ordering::Equal));
        // INFINITY > 1.0
        assert_eq!(v1.partial_cmp(&v4), Some(std::cmp::Ordering::Greater));
        assert_eq!(v4.partial_cmp(&v1), Some(std::cmp::Ordering::Less));
    }

    // ================================
    //
    // Hash trait method tests
    //
    // ================================
    #[test]
    fn test_hash_i32() {
        let v1 = FVector::from_vec(vec![1, 2, 3]);
        let v2 = FVector::from_vec(vec![1, 2, 3]);
        let v3 = FVector::from_vec(vec![3, 2, 1]);

        let mut hasher1 = DefaultHasher::new();
        v1.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        v2.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        let mut hasher3 = DefaultHasher::new();
        v3.hash(&mut hasher3);
        let hash3 = hasher3.finish();

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_hash_complex_i32() {
        use num::Complex;
        let v1 = FVector::from_vec(vec![Complex::new(1, 2), Complex::new(3, 4)]);
        let v2 = FVector::from_vec(vec![Complex::new(1, 2), Complex::new(3, 4)]);
        let v3 = FVector::from_vec(vec![Complex::new(4, 3), Complex::new(2, 1)]);

        let mut hasher1 = DefaultHasher::new();
        v1.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        v2.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        let mut hasher3 = DefaultHasher::new();
        v3.hash(&mut hasher3);
        let hash3 = hasher3.finish();

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_hash_empty() {
        let v1: FlexVector<i32> = FlexVector::new();
        let v2: FlexVector<i32> = FlexVector::new();

        let mut hasher1 = DefaultHasher::new();
        v1.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        v2.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        assert_eq!(hash1, hash2);
    }

    // ================================
    //
    // From trait tests
    //
    // ================================
    #[test]
    fn test_from_vec_i32() {
        let vec = vec![1, 2, 3];
        let fv: FlexVector<i32> = FlexVector::from(vec.clone());
        let fv_col: FlexVector<i32, Column> = vec.clone().into();
        let fv_row: FlexVector<i32, Row> = vec.clone().into();
        assert_eq!(fv.as_slice(), &vec[..]);
        assert!(fv_col.is_column());
        assert!(fv_row.is_row());
    }

    #[test]
    fn test_from_vec_f64() {
        let vec = vec![1.1, 2.2, 3.3];
        let fv: FlexVector<f64> = FlexVector::from(vec.clone());
        let fv_col: FlexVector<f64, Column> = vec.clone().into();
        let fv_row: FlexVector<f64, Row> = vec.clone().into();
        assert_eq!(fv.as_slice(), &vec[..]);
        assert!(fv_col.is_column());
        assert!(fv_row.is_row());
    }

    #[test]
    fn test_from_vec_complex_f64() {
        use num::Complex;
        let vec = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let fv: FlexVector<Complex<f64>> = FlexVector::from(vec.clone());
        let fv_col: FlexVector<Complex<f64>, Column> = vec.clone().into();
        let fv_row: FlexVector<Complex<f64>, Row> = vec.clone().into();
        assert_eq!(fv.as_slice(), &vec[..]);
        assert!(fv_col.is_column());
        assert!(fv_row.is_row());
    }

    #[test]
    fn test_from_slice_i32() {
        let slice: &[i32] = &[4, 5, 6];
        let fv = FVector::from(slice);
        let fv_col: FlexVector<i32, Column> = slice.into();
        let fv_row: FlexVector<i32, Row> = slice.into();
        assert_eq!(fv.as_slice(), slice);
        assert!(fv_col.is_column());
        assert!(fv_row.is_row());
    }

    #[test]
    fn test_from_slice_f64() {
        let slice: &[f64] = &[4.4, 5.5, 6.6];
        let fv = FVector::from(slice);
        let fv_col: FlexVector<f64, Column> = slice.into();
        let fv_row: FlexVector<f64, Row> = slice.into();
        assert_eq!(fv.as_slice(), slice);
        assert!(fv_col.is_column());
        assert!(fv_row.is_row());
    }

    #[test]
    fn test_from_slice_complex_f64() {
        use num::Complex;
        let slice: &[Complex<f64>] = &[Complex::new(7.0, 8.0), Complex::new(9.0, 10.0)];
        let fv = FVector::from(slice);
        let fv_col: FlexVector<Complex<f64>, Column> = slice.into();
        let fv_row: FlexVector<Complex<f64>, Row> = slice.into();
        assert_eq!(fv.as_slice(), slice);
        assert!(fv_col.is_column());
        assert!(fv_row.is_row());
    }

    #[test]
    fn test_from_cow_i32() {
        // Borrowed
        let slice: &[i32] = &[1, 2, 3];
        let cow = Cow::Borrowed(slice);
        let fv: FlexVector<i32> = FlexVector::from(cow.clone());
        assert_eq!(fv.as_slice(), slice);

        // Owned
        let owned_vec = vec![4, 5, 6];
        let cow: Cow<'_, [i32]> = Cow::Owned(owned_vec.clone());
        let fv: FlexVector<i32> = FlexVector::from(cow.clone());
        assert_eq!(fv.as_slice(), &owned_vec[..]);

        // From &Cow
        let fv: FlexVector<i32> = FlexVector::from(&cow);
        assert_eq!(fv.as_slice(), &owned_vec[..]);
    }

    #[test]
    fn test_from_cow_f64() {
        // Borrowed
        let slice: &[f64] = &[1.1, 2.2, 3.3];
        let cow = Cow::Borrowed(slice);
        let fv: FlexVector<f64> = FlexVector::from(cow.clone());
        assert_eq!(fv.as_slice(), slice);

        // Owned
        let owned_vec = vec![4.4, 5.5, 6.6];
        let cow: Cow<'_, [f64]> = Cow::Owned(owned_vec.clone());
        let fv: FlexVector<f64> = FlexVector::from(cow.clone());
        assert_eq!(fv.as_slice(), &owned_vec[..]);

        // From &Cow
        let fv: FlexVector<f64> = FlexVector::from(&cow);
        assert_eq!(fv.as_slice(), &owned_vec[..]);
    }

    #[test]
    fn test_from_cow_complex_f64() {
        use num::Complex;
        // Borrowed
        let slice: &[Complex<f64>] = &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let cow = Cow::Borrowed(slice);
        let fv: FlexVector<Complex<f64>> = FlexVector::from(cow.clone());
        assert_eq!(fv.as_slice(), slice);

        // Owned
        let owned_vec = vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)];
        let cow: Cow<'_, [Complex<f64>]> = Cow::Owned(owned_vec.clone());
        let fv: FlexVector<Complex<f64>> = FlexVector::from(cow.clone());
        assert_eq!(fv.as_slice(), &owned_vec[..]);

        // From &Cow
        let fv: FlexVector<Complex<f64>> = FlexVector::from(&cow);
        assert_eq!(fv.as_slice(), &owned_vec[..]);
    }

    #[test]
    fn test_from_column_to_row() {
        let col = FlexVector::<i32, Column>::from_vec(vec![1, 2, 3]);
        let row: FlexVector<i32, Row> = col.into();
        assert_eq!(row.as_slice(), &[1, 2, 3]);
        assert!(row.is_row());
        assert!(!row.is_column());
    }

    #[test]
    fn test_from_row_to_column() {
        let row = FlexVector::<i32, Row>::from_vec(vec![4, 5, 6]);
        let col: FlexVector<i32, Column> = row.into();
        assert_eq!(col.as_slice(), &[4, 5, 6]);
        assert!(col.is_column());
        assert!(!col.is_row());
    }

    #[test]
    fn test_from_column_to_row_f64() {
        let col = FlexVector::<f64, Column>::from_vec(vec![1.1, 2.2]);
        let row: FlexVector<f64, Row> = col.into();
        assert_eq!(row.as_slice(), &[1.1, 2.2]);
        assert!(row.is_row());
        assert!(!row.is_column());
    }

    #[test]
    fn test_from_row_to_column_f64() {
        let row = FlexVector::<f64, Row>::from_vec(vec![3.3, 4.4]);
        let col: FlexVector<f64, Column> = row.into();
        assert_eq!(col.as_slice(), &[3.3, 4.4]);
        assert!(col.is_column());
        assert!(!col.is_row());
    }

    #[test]
    fn test_from_column_to_row_complex() {
        use num::Complex;
        let col = FlexVector::<Complex<f64>, Column>::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
        ]);
        let row: FlexVector<Complex<f64>, Row> = col.into();
        assert_eq!(row.as_slice(), &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        assert!(row.is_row());
        assert!(!row.is_column());
    }

    #[test]
    fn test_from_row_to_column_complex() {
        use num::Complex;
        let row = FlexVector::<Complex<f64>, Row>::from_vec(vec![
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ]);
        let col: FlexVector<Complex<f64>, Column> = row.into();
        assert_eq!(col.as_slice(), &[Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)]);
        assert!(col.is_column());
        assert!(!col.is_row());
    }

    // ================================
    //
    // Index trait tests
    //
    // ================================

    #[test]
    fn test_index_usize() {
        let v = FVector::from_vec(vec![10, 20, 30, 40]);
        assert_eq!(v[0], 10);
        assert_eq!(v[1], 20);
        assert_eq!(v[3], 40);
    }

    #[test]
    #[should_panic]
    fn test_index_usize_out_of_bounds() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let _ = v[10];
    }

    #[test]
    fn test_index_range() {
        let v = FVector::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(&v[1..4], &[2, 3, 4]);
        assert_eq!(&v[0..2], &[1, 2]);
        assert_eq!(&v[2..5], &[3, 4, 5]);
    }

    #[test]
    fn test_index_range_from() {
        let v = FVector::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(&v[2..], &[3, 4, 5]);
        assert_eq!(&v[0..], &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_index_range_to() {
        let v = FVector::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(&v[..3], &[1, 2, 3]);
        assert_eq!(&v[..1], &[1]);
    }

    #[test]
    fn test_index_range_full() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        assert_eq!(&v[..], &[1, 2, 3]);
    }

    #[test]
    fn test_index_range_inclusive() {
        let v = FVector::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(&v[1..=3], &[2, 3, 4]);
        assert_eq!(&v[0..=4], &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_index_range_to_inclusive() {
        let v = FVector::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(&v[..=2], &[1, 2, 3]);
        assert_eq!(&v[..=0], &[1]);
    }

    #[test]
    #[should_panic]
    fn test_index_range_out_of_bounds() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let _ = &v[2..5];
    }

    #[test]
    #[should_panic]
    fn test_index_range_inclusive_out_of_bounds() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let _ = &v[1..=5];
    }

    #[test]
    #[should_panic]
    fn test_index_range_to_inclusive_out_of_bounds() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let _ = &v[..=5];
    }

    // ================================
    //
    // IndexMut trait tests
    //
    // ================================

    #[test]
    fn test_index_mut_usize() {
        let mut v = FVector::from_vec(vec![1, 2, 3]);
        v[0] = 10;
        v[2] = 30;
        assert_eq!(v.as_slice(), &[10, 2, 30]);
    }

    #[test]
    #[should_panic]
    fn test_index_mut_usize_out_of_bounds() {
        let mut v = FVector::from_vec(vec![1, 2, 3]);
        v[10] = 99;
    }

    #[test]
    fn test_index_mut_range() {
        let mut v = FVector::from_vec(vec![1, 2, 3, 4, 5]);
        v[1..4].copy_from_slice(&[20, 30, 40]);
        assert_eq!(v.as_slice(), &[1, 20, 30, 40, 5]);
    }

    #[test]
    fn test_index_mut_range_from() {
        let mut v = FVector::from_vec(vec![1, 2, 3, 4, 5]);
        v[2..].copy_from_slice(&[99, 100, 101]);
        assert_eq!(v.as_slice(), &[1, 2, 99, 100, 101]);
    }

    #[test]
    fn test_index_mut_range_to() {
        let mut v = FVector::from_vec(vec![1, 2, 3, 4, 5]);
        v[..3].copy_from_slice(&[7, 8, 9]);
        assert_eq!(v.as_slice(), &[7, 8, 9, 4, 5]);
    }

    #[test]
    fn test_index_mut_range_full() {
        let mut v = FVector::from_vec(vec![1, 2, 3]);
        v[..].copy_from_slice(&[10, 20, 30]);
        assert_eq!(v.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_index_mut_range_inclusive() {
        let mut v = FVector::from_vec(vec![1, 2, 3, 4, 5]);
        v[1..=3].copy_from_slice(&[21, 31, 41]);
        assert_eq!(v.as_slice(), &[1, 21, 31, 41, 5]);
    }

    #[test]
    fn test_index_mut_range_to_inclusive() {
        let mut v = FVector::from_vec(vec![1, 2, 3, 4, 5]);
        v[..=2].copy_from_slice(&[100, 200, 300]);
        assert_eq!(v.as_slice(), &[100, 200, 300, 4, 5]);
    }

    #[test]
    #[should_panic]
    fn test_index_mut_range_out_of_bounds() {
        let mut v = FVector::from_vec(vec![1, 2, 3]);
        v[2..5].copy_from_slice(&[9, 9, 9]);
    }

    #[test]
    #[should_panic]
    fn test_index_mut_range_inclusive_out_of_bounds() {
        let mut v = FVector::from_vec(vec![1, 2, 3]);
        v[1..=5].copy_from_slice(&[9, 9, 9, 9, 9]);
    }

    #[test]
    #[should_panic]
    fn test_index_mut_range_to_inclusive_out_of_bounds() {
        let mut v = FVector::from_vec(vec![1, 2, 3]);
        v[..=5].copy_from_slice(&[9, 9, 9, 9, 9, 9]);
    }

    // ================================
    //
    // Extend trait tests
    //
    // ================================
    #[test]
    fn test_extend_i32() {
        let mut v = FVector::from_vec(vec![1, 2]);
        v.extend(vec![3, 4]);
        assert_eq!(v.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_extend_f64() {
        let mut v = FVector::from_vec(vec![1.1, 2.2]);
        v.extend(vec![3.3, 4.4]);
        assert_eq!(v.as_slice(), &[1.1, 2.2, 3.3, 4.4]);
    }

    #[test]
    fn test_extend_complex_f64() {
        use num::Complex;
        let mut v = FVector::from_vec(vec![Complex::new(1.0, 2.0)]);
        v.extend(vec![Complex::new(3.0, 4.0), Complex::new(5.0, 6.0)]);
        assert_eq!(
            v.as_slice(),
            &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0), Complex::new(5.0, 6.0)]
        );
    }

    #[test]
    fn test_extend_empty() {
        let mut v = FlexVector::<i32>::new();
        v.extend(vec![7, 8]);
        assert_eq!(v.as_slice(), &[7, 8]);
    }

    #[test]
    fn test_extend_with_empty() {
        let mut v = FVector::from_vec(vec![1, 2]);
        v.extend(Vec::<i32>::new());
        assert_eq!(v.as_slice(), &[1, 2]);
    }

    // ================================
    //
    // VectorBase trait method tests
    //
    // ================================
    // --- as_slice ---
    #[test]
    fn test_as_slice() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let slice = v.as_slice();
        assert_eq!(slice, &[1, 2, 3]);
    }

    #[test]
    fn test_as_slice_f64() {
        let v = FVector::from_vec(vec![1.1, 2.2, 3.3]);
        let slice = v.as_slice();
        assert_eq!(slice, &[1.1, 2.2, 3.3]);
    }

    #[test]
    fn test_as_slice_complex() {
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let slice = v.as_slice();
        assert_eq!(slice, &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),]);
    }

    // --- len ---
    #[test]
    fn test_len() {
        let v = FVector::from_vec(vec![10, 20, 30, 40]);
        assert_eq!(v.len(), 4);
        let empty = FlexVector::<i32>::new();
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_len_f64() {
        let v = FVector::from_vec(vec![10.0, 20.0]);
        assert_eq!(v.len(), 2);
        let empty = FlexVector::<f64>::new();
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_len_complex() {
        let v = FVector::from_vec(vec![Complex::new(1.0, 0.0)]);
        assert_eq!(v.len(), 1);
        let empty = FlexVector::<Complex<f64>>::new();
        assert_eq!(empty.len(), 0);
    }

    // --- is_empty ---
    #[test]
    fn test_is_empty() {
        let v = FVector::from_vec(vec![1]);
        assert!(!v.is_empty());
        let empty = FlexVector::<i32>::new();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_is_empty_f64() {
        let v = FVector::from_vec(vec![1.0]);
        assert!(!v.is_empty());
        let empty = FlexVector::<f64>::new();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_is_empty_complex() {
        let v = FVector::from_vec(vec![Complex::new(0.0, 1.0)]);
        assert!(!v.is_empty());
        let empty = FlexVector::<Complex<f64>>::new();
        assert!(empty.is_empty());
    }

    // --- get ---
    #[test]
    fn test_get() {
        let v = FVector::from_vec(vec![10, 20, 30]);
        assert_eq!(v.get(0), Some(&10));
        assert_eq!(v.get(2), Some(&30));
        assert_eq!(v.get(3), None);
    }

    #[test]
    fn test_get_f64() {
        let v = FVector::from_vec(vec![10.5, 20.5, 30.5]);
        assert_eq!(v.get(0), Some(&10.5));
        assert_eq!(v.get(2), Some(&30.5));
        assert_eq!(v.get(3), None);
    }

    #[test]
    fn test_get_complex() {
        let v = FVector::from_vec(vec![Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)]);
        assert_eq!(v.get(0), Some(&Complex::new(1.0, 1.0)));
        assert_eq!(v.get(1), Some(&Complex::new(2.0, 2.0)));
        assert_eq!(v.get(2), None);
    }

    // --- first ---
    #[test]
    fn test_first() {
        let v = FVector::from_vec(vec![5, 6, 7]);
        assert_eq!(v.first(), Some(&5));
        let empty = FlexVector::<i32>::new();
        assert_eq!(empty.first(), None);
    }

    #[test]
    fn test_first_f64() {
        let v = FVector::from_vec(vec![5.5, 6.5]);
        assert_eq!(v.first(), Some(&5.5));
        let empty = FlexVector::<f64>::new();
        assert_eq!(empty.first(), None);
    }

    #[test]
    fn test_first_complex() {
        let v = FVector::from_vec(vec![Complex::new(7.0, 8.0), Complex::new(9.0, 10.0)]);
        assert_eq!(v.first(), Some(&Complex::new(7.0, 8.0)));
        let empty = FlexVector::<Complex<f64>>::new();
        assert_eq!(empty.first(), None);
    }

    // --- last ---
    #[test]
    fn test_last() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        assert_eq!(v.last(), Some(&3));
        let empty = FlexVector::<i32>::new();
        assert_eq!(empty.last(), None);
    }

    #[test]
    fn test_last_f64() {
        let v = FVector::from_vec(vec![1.5, 2.5, 3.5]);
        assert_eq!(v.last(), Some(&3.5));
        let empty = FlexVector::<f64>::new();
        assert_eq!(empty.last(), None);
    }

    #[test]
    fn test_last_complex() {
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        assert_eq!(v.last(), Some(&Complex::new(3.0, 4.0)));
        let empty = FlexVector::<Complex<f64>>::new();
        assert_eq!(empty.last(), None);
    }

    // --- iter ---
    #[test]
    fn test_iter() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let mut iter = v.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_f64() {
        let v = FVector::from_vec(vec![1.1, 2.2, 3.3]);
        let collected: Vec<_> = v.iter().copied().collect();
        assert_eq!(collected, vec![1.1, 2.2, 3.3]);
    }

    #[test]
    fn test_iter_complex() {
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let collected: Vec<_> = v.iter().cloned().collect();
        assert_eq!(collected, vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),]);
    }

    // --- iter_rev ---
    #[test]
    fn test_iter_rev() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let collected: Vec<_> = v.iter_rev().copied().collect();
        assert_eq!(collected, vec![3, 2, 1]);
    }

    #[test]
    fn test_iter_rev_f64() {
        let v = FVector::from_vec(vec![1.1, 2.2, 3.3]);
        let collected: Vec<_> = v.iter_rev().copied().collect();
        assert_eq!(collected, vec![3.3, 2.2, 1.1]);
    }

    #[test]
    fn test_iter_rev_complex() {
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let collected: Vec<_> = v.iter_rev().cloned().collect();
        assert_eq!(collected, vec![Complex::new(3.0, 4.0), Complex::new(1.0, 2.0),]);
    }

    // --- enumerate ---
    #[test]
    fn test_enumerate() {
        let v = FVector::from_vec(vec![10, 20, 30]);
        let pairs: Vec<_> = v.enumerate().collect();
        assert_eq!(pairs, vec![(0, &10), (1, &20), (2, &30)]);
    }

    #[test]
    fn test_enumerate_f64() {
        let v = FVector::from_vec(vec![1.5, 2.5]);
        let pairs: Vec<_> = v.enumerate().collect();
        assert_eq!(pairs, vec![(0, &1.5), (1, &2.5)]);
    }

    #[test]
    fn test_enumerate_complex() {
        let v = FVector::from_vec(vec![Complex::new(0.0, 1.0), Complex::new(2.0, 3.0)]);
        let pairs: Vec<_> = v.enumerate().collect();
        assert_eq!(pairs, vec![(0, &Complex::new(0.0, 1.0)), (1, &Complex::new(2.0, 3.0)),]);
    }

    // --- to_vec ---
    #[test]
    fn test_to_vec() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let vec_copy = v.to_vec();
        assert_eq!(vec_copy, vec![1, 2, 3]);
    }

    #[test]
    fn test_to_vec_f64() {
        let v = FVector::from_vec(vec![1.5, 2.5]);
        let vec_copy = v.to_vec();
        assert_eq!(vec_copy, vec![1.5, 2.5]);
    }

    #[test]
    fn test_to_vec_complex() {
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let vec_copy = v.to_vec();
        assert_eq!(vec_copy, vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),]);
    }

    // --- as_cow ---

    #[test]
    fn test_as_cow_i32() {
        let v = FlexVector::<i32>::from(vec![1, 2, 3]);
        let cow = v.as_cow();
        assert_eq!(&*cow, &[1, 2, 3]);
        // Should be borrowed, not owned
        assert!(matches!(cow, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn test_as_cow_f64() {
        let v = FlexVector::<f64>::from(vec![1.1, 2.2, 3.3]);
        let cow = v.as_cow();
        assert_eq!(&*cow, &[1.1, 2.2, 3.3]);
        assert!(matches!(cow, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn test_as_cow_complex_f64() {
        use num::Complex;
        let v =
            FlexVector::<Complex<f64>>::from(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let cow = v.as_cow();
        assert_eq!(&*cow, &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        assert!(matches!(cow, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn test_as_cow_empty() {
        let v = FlexVector::<i32>::new();
        let cow = v.as_cow();
        assert_eq!(&*cow, &[]);
        assert!(matches!(cow, std::borrow::Cow::Borrowed(_)));
    }

    // --- pretty ---
    #[test]
    fn test_pretty() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let pretty = v.pretty();
        assert!(pretty.contains("1"));
        assert!(pretty.contains("2"));
        assert!(pretty.contains("3"));
        assert!(pretty.starts_with("["));
    }

    #[test]
    fn test_pretty_f64() {
        let v = FVector::from_vec(vec![1.5, 2.5]);
        let pretty = v.pretty();
        assert!(pretty.contains("1.5"));
        assert!(pretty.contains("2.5"));
        assert!(pretty.starts_with("["));
    }

    #[test]
    fn test_pretty_complex() {
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
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
        let v = FVector::from_vec(vec![1, 2, 3]);
        assert!(v.contains(&2));
        assert!(!v.contains(&4));
    }

    #[test]
    fn test_contains_f64() {
        let v = FVector::from_vec(vec![1.1, 2.2, 3.3]);
        assert!(v.contains(&2.2));
        assert!(!v.contains(&4.4));
    }

    #[test]
    fn test_contains_complex() {
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        assert!(v.contains(&Complex::new(1.0, 2.0)));
        assert!(!v.contains(&Complex::new(0.0, 0.0)));
    }

    // --- starts_with ---
    #[test]
    fn test_starts_with() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        assert!(v.starts_with(&[1, 2]));
        assert!(!v.starts_with(&[2, 3]));
    }

    #[test]
    fn test_starts_with_f64() {
        let v = FVector::from_vec(vec![1.1, 2.2, 3.3]);
        assert!(v.starts_with(&[1.1, 2.2]));
        assert!(!v.starts_with(&[2.2, 3.3]));
    }

    #[test]
    fn test_starts_with_complex() {
        let v = FVector::from_vec(vec![
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
        let v = FVector::from_vec(vec![1, 2, 3]);
        assert!(v.ends_with(&[2, 3]));
        assert!(!v.ends_with(&[1, 2]));
    }

    #[test]
    fn test_ends_with_f64() {
        let v = FVector::from_vec(vec![1.1, 2.2, 3.3]);
        assert!(v.ends_with(&[2.2, 3.3]));
        assert!(!v.ends_with(&[1.1, 2.2]));
    }

    #[test]
    fn test_ends_with_complex() {
        let v = FVector::from_vec(vec![
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
        let v = FVector::from_vec(vec![10, 20, 30]);
        assert_eq!(v.position(|&x| x == 20), Some(1));
        assert_eq!(v.position(|&x| x == 99), None);
    }

    #[test]
    fn test_position_f64() {
        let v = FVector::from_vec(vec![10.0, 20.0, 30.0]);
        assert_eq!(v.position(|&x| x == 20.0), Some(1));
        assert_eq!(v.position(|&x| x == 99.0), None);
    }

    #[test]
    fn test_position_complex() {
        let v = FVector::from_vec(vec![
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
        let v = FVector::from_vec(vec![1, 2, 3, 2]);
        assert_eq!(v.rposition(|&x| x == 2), Some(3));
        assert_eq!(v.rposition(|&x| x == 99), None);
    }

    #[test]
    fn test_rposition_f64() {
        let v = FVector::from_vec(vec![1.0, 2.0, 3.0, 2.0]);
        assert_eq!(v.rposition(|&x| x == 2.0), Some(3));
        assert_eq!(v.rposition(|&x| x == 99.0), None);
    }

    #[test]
    fn test_rposition_complex() {
        let v = FVector::from_vec(vec![
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
        let v = FVector::from_vec(vec![1, 2, 3, 4]);
        let windows: Vec<_> = v.windows(2).collect();
        assert_eq!(windows, vec![&[1, 2][..], &[2, 3][..], &[3, 4][..]]);
    }

    #[test]
    fn test_windows_f64() {
        let v = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let windows: Vec<_> = v.windows(2).collect();
        assert_eq!(windows, vec![&[1.0, 2.0][..], &[2.0, 3.0][..]]);
    }

    #[test]
    fn test_windows_complex() {
        let v = FVector::from_vec(vec![
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
        let v = FVector::from_vec(vec![1, 2, 3, 4, 5]);
        let chunks: Vec<_> = v.chunks(2).collect();
        assert_eq!(chunks, vec![&[1, 2][..], &[3, 4][..], &[5][..]]);
    }

    #[test]
    fn test_chunks_f64() {
        let v = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let chunks: Vec<_> = v.chunks(2).collect();
        assert_eq!(chunks, vec![&[1.0, 2.0][..], &[3.0][..]]);
    }

    #[test]
    fn test_chunks_complex() {
        let v = FVector::from_vec(vec![
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
        let v = FVector::from_vec(vec![1, 2, 3, 4]);
        let (left, right) = v.split_at(2);
        assert_eq!(left, &[1, 2]);
        assert_eq!(right, &[3, 4]);
    }

    #[test]
    fn test_split_at_f64() {
        let v = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let (left, right) = v.split_at(1);
        assert_eq!(left, &[1.0]);
        assert_eq!(right, &[2.0, 3.0]);
    }

    #[test]
    fn test_split_at_complex() {
        let v = FVector::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);
        let (left, right) = v.split_at(1);
        assert_eq!(left, &[Complex::new(1.0, 0.0)]);
        assert_eq!(right, &[Complex::new(2.0, 0.0)]);
    }

    // --- split ---
    #[test]
    fn test_split() {
        let v = FVector::from_vec(vec![1, 2, 0, 3, 0, 4]);
        let splits: Vec<_> = v.split(|&x| x == 0).collect();
        assert_eq!(splits, vec![&[1, 2][..], &[3][..], &[4][..]]);
    }

    #[test]
    fn test_split_f64() {
        let v = FVector::from_vec(vec![1.0, 0.0, 2.0, 0.0, 3.0]);
        let splits: Vec<_> = v.split(|&x| x == 0.0).collect();
        assert_eq!(splits, vec![&[1.0][..], &[2.0][..], &[3.0][..]]);
    }

    #[test]
    fn test_split_complex() {
        let v = FVector::from_vec(vec![
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
        let v = FVector::from_vec(vec![1, 0, 2, 0, 3]);
        let splits: Vec<_> = v.splitn(2, |&x| x == 0).collect();
        assert_eq!(splits, vec![&[1][..], &[2, 0, 3][..]]);
    }

    #[test]
    fn test_splitn_f64() {
        let v = FVector::from_vec(vec![1.0, 0.0, 2.0, 0.0, 3.0]);
        let splits: Vec<_> = v.splitn(2, |&x| x == 0.0).collect();
        assert_eq!(splits, vec![&[1.0][..], &[2.0, 0.0, 3.0][..]]);
    }

    #[test]
    fn test_splitn_complex() {
        let v = FVector::from_vec(vec![
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
        let v = FVector::from_vec(vec![1, 0, 2, 0, 3]);
        let splits: Vec<_> = v.rsplit(|&x| x == 0).collect();
        assert_eq!(splits, vec![&[3][..], &[2][..], &[1][..]]);
    }

    #[test]
    fn test_rsplit_f64() {
        let v = FVector::from_vec(vec![1.0, 0.0, 2.0, 0.0, 3.0]);
        let splits: Vec<_> = v.rsplit(|&x| x == 0.0).collect();
        assert_eq!(splits, vec![&[3.0][..], &[2.0][..], &[1.0][..]]);
    }

    #[test]
    fn test_rsplit_complex() {
        let v = FVector::from_vec(vec![
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
        let v = FVector::from_vec(vec![1, 0, 2, 0, 3]);
        let splits: Vec<_> = v.rsplitn(2, |&x| x == 0).collect();
        assert_eq!(splits, vec![&[3][..], &[1, 0, 2][..]]);
    }

    #[test]
    fn test_rsplitn_f64() {
        let v = FVector::from_vec(vec![1.0, 0.0, 2.0, 0.0, 3.0]);
        let splits: Vec<_> = v.rsplitn(2, |&x| x == 0.0).collect();
        assert_eq!(splits, vec![&[3.0][..], &[1.0, 0.0, 2.0][..]]);
    }

    #[test]
    fn test_rsplitn_complex() {
        let v = FVector::from_vec(vec![
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

    // -- is_row + is_column --

    #[test]
    fn test_is_row_and_is_column_i32() {
        let v_col = FlexVector::<i32, Column>::from_vec(vec![1, 2, 3]);
        let v_row = FlexVector::<i32, Row>::from_vec(vec![1, 2, 3]);
        assert!(v_col.is_column());
        assert!(!v_col.is_row());
        assert!(v_row.is_row());
        assert!(!v_row.is_column());
    }

    #[test]
    fn test_is_row_and_is_column_f64() {
        let v_col = FlexVector::<f64, Column>::from_vec(vec![1.0, 2.0]);
        let v_row = FlexVector::<f64, Row>::from_vec(vec![1.0, 2.0]);
        assert!(v_col.is_column());
        assert!(!v_col.is_row());
        assert!(v_row.is_row());
        assert!(!v_row.is_column());
    }

    #[test]
    fn test_is_row_and_is_column_complex_f64() {
        use num::Complex;
        let v_col = FlexVector::<Complex<f64>, Column>::from_vec(vec![Complex::new(1.0, 2.0)]);
        let v_row = FlexVector::<Complex<f64>, Row>::from_vec(vec![Complex::new(1.0, 2.0)]);
        assert!(v_col.is_column());
        assert!(!v_col.is_row());
        assert!(v_row.is_row());
        assert!(!v_row.is_column());
    }

    // -- as_row + as_column --

    #[test]
    fn test_as_row_and_as_column_i32() {
        let v_col = FlexVector::<i32, Column>::from_vec(vec![1, 2, 3]);
        let v_row = v_col.as_row();
        assert_eq!(v_row.as_slice(), &[1, 2, 3]);
        assert!(v_row.is_row());
        assert!(!v_row.is_column());
        assert!(v_col.is_column());

        let v_col2 = v_row.as_column();
        assert_eq!(v_col2.as_slice(), &[1, 2, 3]);
        assert!(v_col2.is_column());
        assert!(!v_col2.is_row());
        assert!(v_row.is_row());
    }

    #[test]
    fn test_as_row_and_as_column_f64() {
        let v_col = FlexVector::<f64, Column>::from_vec(vec![1.1, 2.2]);
        let v_row = v_col.as_row();
        assert_eq!(v_row.as_slice(), &[1.1, 2.2]);
        assert!(v_row.is_row());
        assert!(!v_row.is_column());
        assert!(v_col.is_column());

        let v_col2 = v_row.as_column();
        assert_eq!(v_col2.as_slice(), &[1.1, 2.2]);
        assert!(v_col2.is_column());
        assert!(!v_col2.is_row());
        assert!(v_row.is_row());
    }

    #[test]
    fn test_as_row_and_as_column_complex_f64() {
        let v_col = FlexVector::<Complex<f64>, Column>::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
        ]);
        let v_row = v_col.as_row();
        assert_eq!(v_row.as_slice(), &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        assert!(v_row.is_row());
        assert!(!v_row.is_column());
        assert!(v_col.is_column());

        let v_col2 = v_row.as_column();
        assert_eq!(v_col2.as_slice(), &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        assert!(v_col2.is_column());
        assert!(!v_col2.is_row());
        assert!(v_row.is_row());
    }

    // -- into_row + into_column --

    #[test]
    fn test_into_row_and_into_column_i32() {
        let v_col = FlexVector::<i32, Column>::from_vec(vec![1, 2, 3]);
        let v_row = v_col.into_row(); // move occurs here
        assert_eq!(v_row.as_slice(), &[1, 2, 3]);
        assert!(v_row.is_row());
        assert!(!v_row.is_column());

        let v_col2 = v_row.into_column(); // move occurs here
        assert_eq!(v_col2.as_slice(), &[1, 2, 3]);
        assert!(v_col2.is_column());
        assert!(!v_col2.is_row());
    }

    #[test]
    fn test_into_row_and_into_column_f64() {
        let v_col = FlexVector::<f64, Column>::from_vec(vec![1.1, 2.2]);
        let v_row = v_col.into_row(); // move occurs here
        assert_eq!(v_row.as_slice(), &[1.1, 2.2]);
        assert!(v_row.is_row());
        assert!(!v_row.is_column());

        let v_col2 = v_row.into_column(); // move occurs here
        assert_eq!(v_col2.as_slice(), &[1.1, 2.2]);
        assert!(v_col2.is_column());
        assert!(!v_col2.is_row());
    }

    #[test]
    fn test_into_row_and_into_column_complex_f64() {
        use num::Complex;
        let v_col = FlexVector::<Complex<f64>, Column>::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
        ]);
        let v_row = v_col.into_row(); // move occurs here
        assert_eq!(v_row.as_slice(), &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        assert!(v_row.is_row());
        assert!(!v_row.is_column());

        let v_col2 = v_row.into_column(); // move occurs here
        assert_eq!(v_col2.as_slice(), &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        assert!(v_col2.is_column());
        assert!(!v_col2.is_row());
    }

    // --- push ---
    #[test]
    fn test_push() {
        let mut v = FVector::new();
        v.push(1);
        v.push(2);
        assert_eq!(v.as_slice(), &[1, 2]);
    }

    #[test]
    fn test_push_f64() {
        let mut v = FVector::new();
        v.push(1.1);
        v.push(2.2);
        assert_eq!(v.as_slice(), &[1.1, 2.2]);
    }

    #[test]
    fn test_push_complex() {
        let mut v = FVector::new();
        v.push(Complex::new(1.0, 2.0));
        v.push(Complex::new(3.0, 4.0));
        assert_eq!(v.as_slice(), &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
    }

    // --- pop ---
    #[test]
    fn test_pop() {
        let mut v = FVector::from_vec(vec![1, 2]);
        assert_eq!(v.pop(), Some(2));
        assert_eq!(v.pop(), Some(1));
        assert_eq!(v.pop(), None);
    }

    #[test]
    fn test_pop_f64() {
        let mut v = FVector::from_vec(vec![1.1, 2.2]);
        assert_eq!(v.pop(), Some(2.2));
        assert_eq!(v.pop(), Some(1.1));
        assert_eq!(v.pop(), None);
    }

    #[test]
    fn test_pop_complex() {
        let mut v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        assert_eq!(v.pop(), Some(Complex::new(3.0, 4.0)));
        assert_eq!(v.pop(), Some(Complex::new(1.0, 2.0)));
        assert_eq!(v.pop(), None);
    }

    // --- insert ---
    #[test]
    fn test_insert() {
        let mut v = FVector::from_vec(vec![1, 3]);
        v.insert(1, 2);
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_insert_f64() {
        let mut v = FVector::from_vec(vec![1.1, 3.3]);
        v.insert(1, 2.2);
        assert_eq!(v.as_slice(), &[1.1, 2.2, 3.3]);
    }

    #[test]
    fn test_insert_complex() {
        let mut v = FVector::from_vec(vec![Complex::new(1.0, 1.0), Complex::new(3.0, 3.0)]);
        v.insert(1, Complex::new(2.0, 2.0));
        assert_eq!(
            v.as_slice(),
            &[Complex::new(1.0, 1.0), Complex::new(2.0, 2.0), Complex::new(3.0, 3.0)]
        );
    }

    // --- remove ---
    #[test]
    fn test_remove() {
        let mut v = FVector::from_vec(vec![1, 2, 3]);
        assert_eq!(v.remove(1), 2);
        assert_eq!(v.as_slice(), &[1, 3]);
    }

    #[test]
    fn test_remove_f64() {
        let mut v = FVector::from_vec(vec![1.1, 2.2, 3.3]);
        assert_eq!(v.remove(1), 2.2);
        assert_eq!(v.as_slice(), &[1.1, 3.3]);
    }

    #[test]
    fn test_remove_complex() {
        let mut v = FVector::from_vec(vec![
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
        let mut v = FVector::from_vec(vec![1, 2]);
        v.resize(4, 0);
        assert_eq!(v.as_slice(), &[1, 2, 0, 0]);
        v.resize(2, 0);
        assert_eq!(v.as_slice(), &[1, 2]);
    }

    #[test]
    fn test_resize_f64() {
        let mut v = FVector::from_vec(vec![1.1, 2.2]);
        v.resize(4, 0.0);
        assert_eq!(v.as_slice(), &[1.1, 2.2, 0.0, 0.0]);
        v.resize(1, 0.0);
        assert_eq!(v.as_slice(), &[1.1]);
    }

    #[test]
    fn test_resize_complex() {
        let mut v = FVector::from_vec(vec![Complex::new(1.0, 1.0)]);
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
        let mut v = FVector::from_vec(vec![1, 2, 3]);
        assert!(!v.is_empty());
        v.clear();
        assert!(v.is_empty());
    }

    #[test]
    fn test_clear_f64() {
        let mut v = FVector::from_vec(vec![1.1, 2.2, 3.3]);
        assert!(!v.is_empty());
        v.clear();
        assert!(v.is_empty());
    }

    #[test]
    fn test_clear_complex() {
        let mut v = FVector::from_vec(vec![Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)]);
        assert!(!v.is_empty());
        v.clear();
        assert!(v.is_empty());
    }

    // --- get_mut ---
    #[test]
    fn test_get_mut_single() {
        let mut v = FVector::from_vec(vec![10, 20, 30]);
        if let Some(x) = v.get_mut(1) {
            *x = 99;
        }
        assert_eq!(v.as_slice(), &[10, 99, 30]);
    }

    #[test]
    fn test_get_mut_out_of_bounds() {
        let mut v = FVector::from_vec(vec![1, 2, 3]);
        assert!(v.get_mut(10).is_none());
    }

    #[test]
    fn test_get_mut_range() {
        let mut v = FVector::from_vec(vec![1, 2, 3, 4]);
        if let Some(slice) = v.get_mut(1..3) {
            slice[0] = 20;
            slice[1] = 30;
        }
        assert_eq!(v.as_slice(), &[1, 20, 30, 4]);
    }

    #[test]
    fn test_get_mut_full_range() {
        let mut v = FVector::from_vec(vec![5, 6, 7]);
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
        let mut v = FVector::from_vec(vec![1, 2, 3]);
        for x in v.iter_mut() {
            *x *= 2;
        }
        assert_eq!(v.as_slice(), &[2, 4, 6]);
    }

    #[test]
    fn test_iter_mut_f64() {
        let mut v = FVector::from_vec(vec![1.5, -2.0, 0.0]);
        for x in v.iter_mut() {
            *x += 1.0;
        }
        assert_eq!(v.as_slice(), &[2.5, -1.0, 1.0]);
    }

    #[test]
    fn test_iter_mut_complex() {
        use num::Complex;
        let mut v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(-3.0, 4.0)]);
        for x in v.iter_mut() {
            x.re += 1.0;
            x.im *= 2.0;
        }
        assert_eq!(v.as_slice(), &[Complex::new(2.0, 4.0), Complex::new(-2.0, 8.0)]);
    }

    #[test]
    fn test_iter_mut_empty() {
        let mut v = FVector::<i32>::new();
        let mut count = 0;
        for _ in v.iter_mut() {
            count += 1;
        }
        assert_eq!(count, 0);
    }

    // --- as_mut_slice ---
    #[test]
    fn test_as_mut_slice_i32() {
        let mut v = FVector::from_vec(vec![1, 2, 3]);
        let slice = v.as_mut_slice();
        slice[0] = 10;
        slice[2] = 30;
        assert_eq!(v.as_slice(), &[10, 2, 30]);
    }

    #[test]
    fn test_as_mut_slice_f64() {
        let mut v = FVector::from_vec(vec![1.1, 2.2, 3.3]);
        let slice = v.as_mut_slice();
        slice[1] = 9.9;
        assert_eq!(v.as_slice(), &[1.1, 9.9, 3.3]);
    }

    #[test]
    fn test_as_mut_slice_complex() {
        use num::Complex;
        let mut v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let slice = v.as_mut_slice();
        slice[0].re = 10.0;
        slice[1].im = 40.0;
        assert_eq!(v.as_slice(), &[Complex::new(10.0, 2.0), Complex::new(3.0, 40.0)]);
    }

    #[test]
    fn test_as_mut_slice_empty() {
        let mut v = FVector::<i32>::new();
        let slice = v.as_mut_slice();
        assert_eq!(slice.len(), 0);
    }

    // --- map ---
    #[test]
    fn test_map_i32() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let squared = v.map(|x| x * x);
        assert_eq!(squared.as_slice(), &[1, 4, 9]);
        // original unchanged
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_map_f64() {
        let v: FlexVector<f64> = FVector::from_vec(vec![1.5, -2.0, 0.0]);
        let abs = v.map(|x| x.abs());
        assert_eq!(abs.as_slice(), &[1.5, 2.0, 0.0]);
    }

    #[test]
    fn test_map_complex() {
        use num::Complex;
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(-3.0, 4.0)]);
        let conj = v.map(|x| x.conj());
        assert_eq!(conj.as_slice(), &[Complex::new(1.0, -2.0), Complex::new(-3.0, -4.0)]);
    }

    #[test]
    fn test_map_empty() {
        let v = FlexVector::<i32>::new();
        let mapped = v.map(|x| x + 1);
        assert!(mapped.is_empty());
    }

    #[test]
    fn test_map_with_fn_pointer() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let squared = v.map(square);
        assert_eq!(squared.as_slice(), &[1, 4, 9]);
    }

    // --- mut_map ---
    #[test]
    fn test_mut_map_i32() {
        let mut v = FVector::from_vec(vec![1, 2, 3]);
        v.mut_map(|x| x * 10);
        assert_eq!(v.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_mut_map_f64() {
        let mut v = FVector::from_vec(vec![1.5, -2.0, 0.0]);
        v.mut_map(|x| x + 1.0);
        assert_eq!(v.as_slice(), &[2.5, -1.0, 1.0]);
    }

    #[test]
    fn test_mut_map_complex() {
        let mut v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(-3.0, 4.0)]);
        v.mut_map(|x| Complex::new(x.re + 1.0, x.im * 2.0));
        assert_eq!(v.as_slice(), &[Complex::new(2.0, 4.0), Complex::new(-2.0, 8.0)]);
    }

    #[test]
    fn test_mut_map_empty() {
        let mut v = FlexVector::<i32>::new();
        v.mut_map(|x| x + 1);
        assert!(v.is_empty());
    }

    #[test]
    fn test_mut_map_with_fn_pointer() {
        let mut v = FVector::from_vec(vec![1, 2, 3]);
        v.mut_map(double);
        assert_eq!(v.as_slice(), &[2, 4, 6]);
    }

    // --- flat_map ---
    #[test]
    fn test_flat_map_i32_basic() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let flat = v.flat_map(|x| vec![x, x * 10]);
        assert_eq!(flat.as_slice(), &[1, 10, 2, 20, 3, 30]);
    }

    #[test]
    fn test_flat_map_i32_empty_inner() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let flat = v.flat_map(|x| if x % 2 == 0 { vec![] } else { vec![x] });
        assert_eq!(flat.as_slice(), &[1, 3]);
    }

    #[test]
    fn test_flat_map_i32_option() {
        let v = FVector::from_vec(vec![1, 2, 3, 4]);
        let flat = v.flat_map(|x| if x % 2 == 0 { Some(x) } else { None });
        assert_eq!(flat.as_slice(), &[2, 4]);
    }

    #[test]
    fn test_flat_map_f64_basic() {
        let v = FVector::from_vec(vec![1.0, 2.0]);
        let flat = v.flat_map(|x| vec![x, x + 0.5]);
        assert_eq!(flat.as_slice(), &[1.0, 1.5, 2.0, 2.5]);
    }

    #[test]
    fn test_flat_map_f64_empty() {
        let v: FlexVector<f64> = FlexVector::new();
        let flat = v.flat_map(|x| vec![x, x + 1.0]);
        assert!(flat.is_empty());
    }

    #[test]
    fn test_flat_map_f64_nan_infinity() {
        let v = FVector::from_vec(vec![f64::NAN, f64::INFINITY, 1.0]);
        let flat = v.flat_map(|x| vec![x]);
        assert!(flat.as_slice()[0].is_nan());
        assert_eq!(flat.as_slice()[1], f64::INFINITY);
        assert_eq!(flat.as_slice()[2], 1.0);
    }

    #[test]
    fn test_flat_map_complex_f64_basic() {
        use num::Complex;
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let flat = v.flat_map(|z| vec![z, z.conj()]);
        assert_eq!(
            flat.as_slice(),
            &[
                Complex::new(1.0, 2.0),
                Complex::new(1.0, -2.0),
                Complex::new(3.0, 4.0),
                Complex::new(3.0, -4.0)
            ]
        );
    }

    #[test]
    fn test_flat_map_complex_f64_empty_inner() {
        use num::Complex;
        let v = FVector::from_vec(vec![Complex::new(0.0, 0.0), Complex::new(1.0, 1.0)]);
        let flat = v.flat_map(|z| if z == Complex::new(0.0, 0.0) { vec![] } else { vec![z] });
        assert_eq!(flat.as_slice(), &[Complex::new(1.0, 1.0)]);
    }

    #[test]
    fn test_flat_map_complex_f64_option() {
        use num::Complex;
        let v = FVector::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]);
        let flat = v.flat_map(|z| if z.im == 0.0 { Some(z) } else { None });
        assert_eq!(flat.as_slice(), &[Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]);
    }

    #[test]
    fn test_flat_map_empty_outer() {
        let v: FlexVector<i32> = FlexVector::new();
        let flat = v.flat_map(|x| vec![x]);
        assert!(flat.is_empty());
    }

    #[test]
    fn test_flat_map_all_empty_inner() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let flat = v.flat_map(|_| Vec::<i32>::new());
        assert!(flat.is_empty());
    }

    // -- filter --

    #[test]
    fn test_filter_i32_basic() {
        let v = FVector::from_vec(vec![1, 2, 3, 4, 5]);
        let filtered = v.filter(|&x| x % 2 == 0);
        assert_eq!(filtered.as_slice(), &[2, 4]);
    }

    #[test]
    fn test_filter_i32_all_false() {
        let v = FVector::from_vec(vec![1, 3, 5]);
        let filtered = v.filter(|&x| x > 10);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_i32_all_true() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let filtered = v.filter(|_| true);
        assert_eq!(filtered.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_filter_f64_positive() {
        let v = FVector::from_vec(vec![-1.5, 0.0, 2.5, 3.3]);
        let filtered = v.filter(|&x| x > 0.0);
        assert_eq!(filtered.as_slice(), &[2.5, 3.3]);
    }

    #[test]
    fn test_filter_f64_nan() {
        let v = FVector::from_vec(vec![1.0, f64::NAN, 2.0]);
        let filtered = v.filter(|&x| x.is_nan());
        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].is_nan());
    }

    #[test]
    fn test_filter_complex_f64_real_positive() {
        use num::Complex;
        let v = FVector::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(-3.0, 4.0),
            Complex::new(0.0, 0.0),
        ]);
        let filtered = v.filter(|z| z.re > 0.0);
        assert_eq!(filtered.as_slice(), &[Complex::new(1.0, 2.0)]);
    }

    #[test]
    fn test_filter_complex_f64_imag_nonzero() {
        use num::Complex;
        let v = FVector::from_vec(vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 2.0),
            Complex::new(3.0, 0.0),
        ]);
        let filtered = v.filter(|z| z.im != 0.0);
        assert_eq!(filtered.as_slice(), &[Complex::new(0.0, 2.0)]);
    }

    #[test]
    fn test_filter_empty() {
        let v: FlexVector<i32> = FVector::new();
        let filtered = v.filter(|&x| x > 0);
        assert!(filtered.is_empty());
    }

    // -- reduce --

    #[test]
    fn test_reduce_i32_sum() {
        let v = FVector::from_vec(vec![1, 2, 3, 4]);
        let sum = v.reduce(|a, b| a + b);
        assert_eq!(sum, Some(10));
    }

    #[test]
    fn test_reduce_i32_product() {
        let v = FVector::from_vec(vec![2, 3, 4]);
        let product = v.reduce(|a, b| a * b);
        assert_eq!(product, Some(24));
    }

    #[test]
    fn test_reduce_i32_empty() {
        let v: FlexVector<i32> = FVector::new();
        let result = v.reduce(|a, b| a + b);
        assert_eq!(result, None);
    }

    #[test]
    fn test_reduce_f64_sum() {
        let v = FVector::from_vec(vec![1.5, 2.5, 3.0]);
        let sum = v.reduce(|a, b| a + b);
        assert_eq!(sum, Some(7.0));
    }

    #[test]
    fn test_reduce_f64_product() {
        let v = FVector::from_vec(vec![1.5, 2.0, 3.0]);
        let product = v.reduce(|a, b| a * b);
        assert!((product.unwrap() - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_reduce_f64_empty() {
        let v: FlexVector<f64> = FVector::new();
        let result = v.reduce(|a, b| a + b);
        assert_eq!(result, None);
    }

    #[test]
    fn test_reduce_complex_f64_sum() {
        use num::Complex;
        let v = FVector::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ]);
        let sum = v.reduce(|a, b| a + b);
        assert_eq!(sum, Some(Complex::new(9.0, 12.0)));
    }

    #[test]
    fn test_reduce_complex_f64_product() {
        use num::Complex;
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let product = v.reduce(|a, b| a * b);
        assert_eq!(product, Some(Complex::new(-5.0, 10.0))); // (1+2i)*(3+4i) = -5+10i
    }

    #[test]
    fn test_reduce_complex_f64_empty() {
        use num::Complex;
        let v: FlexVector<Complex<f64>> = FVector::new();
        let result = v.reduce(|a, b| a + b);
        assert_eq!(result, None);
    }

    // -- fold --

    #[test]
    fn test_fold_i32_sum() {
        let v = FVector::from_vec(vec![1, 2, 3, 4]);
        let sum = v.fold(0, |acc, x| acc + x);
        assert_eq!(sum, 10);
    }

    #[test]
    fn test_fold_i32_product() {
        let v = FVector::from_vec(vec![2, 3, 4]);
        let product = v.fold(1, |acc, x| acc * x);
        assert_eq!(product, 24);
    }

    #[test]
    fn test_fold_i32_empty() {
        let v: FlexVector<i32> = FVector::new();
        let sum = v.fold(0, |acc, x| acc + x);
        assert_eq!(sum, 0);
    }

    #[test]
    fn test_fold_f64_sum() {
        let v = FVector::from_vec(vec![1.5, 2.5, 3.0]);
        let sum = v.fold(0.0, |acc, x| acc + x);
        assert!((sum - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_fold_f64_product() {
        let v = FVector::from_vec(vec![1.5, 2.0, 3.0]);
        let product = v.fold(1.0, |acc, x| acc * x);
        assert!((product - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_fold_f64_empty() {
        let v: FlexVector<f64> = FVector::new();
        let sum = v.fold(0.0, |acc, x| acc + x);
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_fold_complex_f64_sum() {
        use num::Complex;
        let v = FVector::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ]);
        let sum = v.fold(Complex::new(0.0, 0.0), |acc, x| acc + x);
        assert_eq!(sum, Complex::new(9.0, 12.0));
    }

    #[test]
    fn test_fold_complex_f64_product() {
        use num::Complex;
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let product = v.fold(Complex::new(1.0, 0.0), |acc, x| acc * x);
        assert_eq!(product, Complex::new(-5.0, 10.0)); // (1+0i)*(1+2i)*(3+4i) = (1+2i)*(3+4i) = -5+10i
    }

    #[test]
    fn test_fold_complex_f64_empty() {
        use num::Complex;
        let v: FlexVector<Complex<f64>> = FVector::new();
        let sum = v.fold(Complex::new(0.0, 0.0), |acc, x| acc + x);
        assert_eq!(sum, Complex::new(0.0, 0.0));
    }

    // --- zip ---
    #[test]
    fn test_zip_i32() {
        let v1 = FVector::from_vec(vec![1, 2, 3]);
        let v2 = FlexVector::from_vec(vec![4, 5, 6]);
        let zipped = v1.zip(v2);
        assert_eq!(zipped.as_slice(), &[(1, 4), (2, 5), (3, 6)]);
    }

    #[test]
    fn test_zip_f64() {
        let v1 = FVector::from_vec(vec![1.1, 2.2, 3.3]);
        let v2 = FVector::from_vec(vec![4.4, 5.5, 6.6]);
        let zipped = v1.zip(v2);
        assert_eq!(zipped.as_slice(), &[(1.1, 4.4), (2.2, 5.5), (3.3, 6.6)]);
    }

    #[test]
    fn test_zip_complex_f64() {
        use num::Complex;
        let v1 = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let v2 = FVector::from_vec(vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)]);
        let zipped = v1.zip(v2);
        assert_eq!(
            zipped.as_slice(),
            &[
                (Complex::new(1.0, 2.0), Complex::new(5.0, 6.0)),
                (Complex::new(3.0, 4.0), Complex::new(7.0, 8.0))
            ]
        );
    }

    #[test]
    fn test_zip_different_lengths() {
        let v1 = FVector::from_vec(vec![1, 2]);
        let v2 = FVector::from_vec(vec![3, 4, 5]);
        let zipped = v1.zip(v2);
        assert_eq!(zipped.as_slice(), &[(1, 3), (2, 4)]);
    }

    #[test]
    fn test_zip_empty() {
        let v1: FlexVector<i32> = FVector::new();
        let v2: FlexVector<i32> = FVector::new();
        let zipped = v1.zip(v2);
        assert!(zipped.is_empty());
    }

    // --- zip_with ---
    #[test]
    fn test_zip_with_i32() {
        let v1 = FVector::from_vec(vec![1, 2, 3]);
        let v2 = FVector::from_vec(vec![4, 5, 6]);
        let summed = v1.zip_with(v2, |a, b| a + b);
        assert_eq!(summed.as_slice(), &[5, 7, 9]);
    }

    #[test]
    fn test_zip_with_f64() {
        let v1 = FVector::from_vec(vec![1.5, 2.5, 3.5]);
        let v2 = FVector::from_vec(vec![4.5, 5.5, 6.5]);
        let prod = v1.zip_with(v2, |a, b| a * b);
        assert_eq!(prod.as_slice(), &[6.75, 13.75, 22.75]);
    }

    #[test]
    fn test_zip_with_complex_f64() {
        use num::Complex;
        let v1 = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let v2 = FVector::from_vec(vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)]);
        let sum = v1.zip_with(v2, |a, b| a + b);
        assert_eq!(sum.as_slice(), &[Complex::new(6.0, 8.0), Complex::new(10.0, 12.0)]);
    }

    #[test]
    fn test_zip_with_different_lengths() {
        let v1 = FVector::from_vec(vec![1, 2]);
        let v2 = FVector::from_vec(vec![10, 20, 30]);
        let zipped = v1.zip_with(v2, |a, b| a + b);
        assert_eq!(zipped.as_slice(), &[11, 22]);
    }

    #[test]
    fn test_zip_with_empty() {
        let v1: FlexVector<i32> = FlexVector::new();
        let v2: FlexVector<i32> = FlexVector::new();
        let zipped = v1.zip_with(v2, |a, b| a + b);
        assert!(zipped.is_empty());
    }

    // --- step_by ---

    #[test]
    fn test_step_by_i32_basic() {
        let v = FVector::from_vec(vec![1, 2, 3, 4, 5, 6]);
        let stepped = v.step_by(2).unwrap();
        assert_eq!(stepped.as_slice(), &[1, 3, 5]);
    }

    #[test]
    fn test_step_by_i32_step_one() {
        let v = FVector::from_vec(vec![10, 20, 30]);
        let stepped = v.step_by(1).unwrap();
        assert_eq!(stepped.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_step_by_i32_step_equals_len() {
        let v = FVector::from_vec(vec![7, 8, 9]);
        let stepped = v.step_by(3).unwrap();
        assert_eq!(stepped.as_slice(), &[7]);
    }

    #[test]
    fn test_step_by_i32_step_greater_than_len() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let stepped = v.step_by(5).unwrap();
        assert_eq!(stepped.as_slice(), &[1]);
    }

    #[test]
    fn test_step_by_i32_empty() {
        let v: FlexVector<i32> = FlexVector::new();
        let stepped = v.step_by(2).unwrap();
        assert!(stepped.is_empty());
    }

    #[test]
    fn test_step_by_i32_zero_step() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let result = v.step_by(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_step_by_f64_basic() {
        let v = FVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let stepped = v.step_by(2).unwrap();
        assert_eq!(stepped.as_slice(), &[1.0, 3.0]);
    }

    #[test]
    fn test_step_by_f64_empty() {
        let v: FlexVector<f64> = FlexVector::new();
        let stepped = v.step_by(3).unwrap();
        assert!(stepped.is_empty());
    }

    #[test]
    fn test_step_by_f64_zero_step() {
        let v = FVector::from_vec(vec![1.1, 2.2]);
        let result = v.step_by(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_step_by_complex_f64_basic() {
        use num::Complex;
        let v = FVector::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ]);
        let stepped = v.step_by(2).unwrap();
        assert_eq!(stepped.as_slice(), &[Complex::new(1.0, 2.0), Complex::new(5.0, 6.0)]);
    }

    #[test]
    fn test_step_by_complex_f64_empty() {
        use num::Complex;
        let v: FlexVector<Complex<f64>> = FlexVector::new();
        let stepped = v.step_by(2).unwrap();
        assert!(stepped.is_empty());
    }

    #[test]
    fn test_step_by_complex_f64_zero_step() {
        use num::Complex;
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0)]);
        let result = v.step_by(0);
        assert!(result.is_err());
    }

    // -- create_mask --

    #[test]
    fn test_create_mask_i32_greater_than() {
        let v = FVector::from_vec(vec![1, 5, 3, 7]);
        let mask = v.create_mask(|&x| x > 3);
        assert_eq!(mask.as_slice(), &[false, true, false, true]);
    }

    #[test]
    fn test_create_mask_i32_even() {
        let v = FVector::from_vec(vec![2, 3, 4, 5]);
        let mask = v.create_mask(|&x| x % 2 == 0);
        assert_eq!(mask.as_slice(), &[true, false, true, false]);
    }

    #[test]
    fn test_create_mask_f64_positive() {
        let v = FVector::from_vec(vec![-1.0, 0.0, 2.5, -3.3]);
        let mask = v.create_mask(|&x| x > 0.0);
        assert_eq!(mask.as_slice(), &[false, false, true, false]);
    }

    #[test]
    fn test_create_mask_f64_nan() {
        let v = FVector::from_vec(vec![1.0, f64::NAN, 2.0]);
        let mask = v.create_mask(|&x| x.is_nan());
        assert_eq!(mask.as_slice(), &[false, true, false]);
    }

    #[test]
    fn test_create_mask_complex_f64_real_positive() {
        use num::Complex;
        let v = FVector::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(-3.0, 4.0),
            Complex::new(0.0, 0.0),
        ]);
        let mask = v.create_mask(|z| z.re > 0.0);
        assert_eq!(mask.as_slice(), &[true, false, false]);
    }

    #[test]
    fn test_create_mask_complex_f64_imag_nonzero() {
        use num::Complex;
        let v = FVector::from_vec(vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 2.0),
            Complex::new(3.0, 0.0),
        ]);
        let mask = v.create_mask(|z| z.im != 0.0);
        assert_eq!(mask.as_slice(), &[false, true, false]);
    }

    #[test]
    fn test_create_mask_empty() {
        let v: FlexVector<i32> = FVector::new();
        let mask = v.create_mask(|&x| x > 0);
        assert!(mask.is_empty());
    }

    // -- filter_by_mask --

    #[test]
    fn test_filter_by_mask_i32_basic() {
        let v = FVector::from_vec(vec![10, 20, 30, 40]);
        let mask = vec![false, true, false, true];
        let filtered = v.filter_by_mask(&mask).unwrap();
        assert_eq!(filtered.as_slice(), &[20, 40]);
    }

    #[test]
    fn test_filter_by_mask_i32_all_true() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let mask = vec![true, true, true];
        let filtered = v.filter_by_mask(&mask).unwrap();
        assert_eq!(filtered.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_filter_by_mask_i32_all_false() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let mask = vec![false, false, false];
        let filtered = v.filter_by_mask(&mask).unwrap();
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_by_mask_i32_empty() {
        let v: FlexVector<i32> = FlexVector::new();
        let mask: Vec<bool> = vec![];
        let filtered = v.filter_by_mask(&mask).unwrap();
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_by_mask_i32_mismatched_length() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let mask = vec![true, false];
        let result = v.filter_by_mask(&mask);
        assert!(result.is_err());
    }

    #[test]
    fn test_filter_by_mask_f64_basic() {
        let v = FVector::from_vec(vec![1.1, 2.2, 3.3, 4.4]);
        let mask = vec![true, false, true, false];
        let filtered = v.filter_by_mask(&mask).unwrap();
        assert_eq!(filtered.as_slice(), &[1.1, 3.3]);
    }

    #[test]
    fn test_filter_by_mask_f64_all_true() {
        let v = FVector::from_vec(vec![1.1, 2.2]);
        let mask = vec![true, true];
        let filtered = v.filter_by_mask(&mask).unwrap();
        assert_eq!(filtered.as_slice(), &[1.1, 2.2]);
    }

    #[test]
    fn test_filter_by_mask_f64_all_false() {
        let v = FVector::from_vec(vec![1.1, 2.2]);
        let mask = vec![false, false];
        let filtered = v.filter_by_mask(&mask).unwrap();
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_by_mask_f64_empty() {
        let v: FlexVector<f64> = FlexVector::new();
        let mask: Vec<bool> = vec![];
        let filtered = v.filter_by_mask(&mask).unwrap();
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_by_mask_f64_mismatched_length() {
        let v = FVector::from_vec(vec![1.1, 2.2]);
        let mask = vec![true];
        let result = v.filter_by_mask(&mask);
        assert!(result.is_err());
    }

    #[test]
    fn test_filter_by_mask_complex_f64_basic() {
        use num::Complex;
        let v = FVector::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ]);
        let mask = vec![false, true, true];
        let filtered = v.filter_by_mask(&mask).unwrap();
        assert_eq!(filtered.as_slice(), &[Complex::new(3.0, 4.0), Complex::new(5.0, 6.0)]);
    }

    #[test]
    fn test_filter_by_mask_complex_f64_all_true() {
        use num::Complex;
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let mask = vec![true, true];
        let filtered = v.filter_by_mask(&mask).unwrap();
        assert_eq!(filtered.as_slice(), &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
    }

    #[test]
    fn test_filter_by_mask_complex_f64_all_false() {
        use num::Complex;
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let mask = vec![false, false];
        let filtered = v.filter_by_mask(&mask).unwrap();
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_by_mask_complex_f64_empty() {
        use num::Complex;
        let v: FlexVector<Complex<f64>> = FlexVector::new();
        let mask: Vec<bool> = vec![];
        let filtered = v.filter_by_mask(&mask).unwrap();
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_by_mask_complex_f64_mismatched_length() {
        use num::Complex;
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0)]);
        let mask = vec![true, false];
        let result = v.filter_by_mask(&mask);
        assert!(result.is_err());
    }

    // ================================
    //
    // VectorOps trait tests
    //
    // ================================

    // -- translate --
    #[test]
    fn test_translate_i32() {
        let v1 = FVector::from_vec(vec![1, 2, 3]);
        let v2 = FlexVector::from_vec(vec![4, 5, 6]);
        let result = v1.translate(&v2).unwrap();
        assert_eq!(result.as_slice(), &[5, 7, 9]);
    }

    #[test]
    fn test_translate_f64() {
        let v1 = FVector::from_vec(vec![1.5, 2.5, 3.5]);
        let v2 = FlexVector::from_vec(vec![0.5, 1.5, 2.5]);
        let result = v1.translate(&v2).unwrap();
        assert_eq!(result.as_slice(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_translate_complex_f64() {
        let v1 = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let v2 = FlexVector::from_vec(vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)]);
        let result = v1.translate(&v2).unwrap();
        assert_eq!(result.as_slice(), &[Complex::new(6.0, 8.0), Complex::new(10.0, 12.0)]);
    }

    #[test]
    fn test_translate_mismatched_lengths() {
        let v1 = FVector::from_vec(vec![1, 2]);
        let v2 = FlexVector::from_vec(vec![4, 5, 6]);
        let result = v1.translate(&v2);
        assert!(result.is_err());
    }

    // -- mut_translate --
    #[test]
    fn test_mut_translate_i32() {
        let mut v1 = FVector::from_vec(vec![1, 2, 3]);
        let v2 = FlexVector::from_vec(vec![4, 5, 6]);
        v1.mut_translate(&v2).unwrap();
        assert_eq!(v1.as_slice(), &[5, 7, 9]);
    }

    #[test]
    fn test_mut_translate_f64() {
        let mut v1 = FVector::from_vec(vec![1.5, 2.5, 3.5]);
        let v2 = FlexVector::from_vec(vec![0.5, 1.5, 2.5]);
        v1.mut_translate(&v2).unwrap();
        assert_eq!(v1.as_slice(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_mut_translate_complex_f64() {
        let mut v1 = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let v2 = FlexVector::from_vec(vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)]);
        v1.mut_translate(&v2).unwrap();
        assert_eq!(v1.as_slice(), &[Complex::new(6.0, 8.0), Complex::new(10.0, 12.0)]);
    }

    #[test]
    fn test_mut_translate_mismatched_lengths() {
        let mut v1 = FVector::from_vec(vec![1, 2]);
        let v2 = FlexVector::from_vec(vec![4, 5, 6]);
        let result = v1.mut_translate(&v2);
        assert!(result.is_err());
    }

    // -- scale --
    #[test]
    fn test_scale_i32() {
        let v = FVector::from_vec(vec![1, 2, 3]);
        let scaled = v.scale(10);
        assert_eq!(scaled.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_scale_f64() {
        let v = FVector::from_vec(vec![1.5, -2.0, 0.0]);
        let scaled = v.scale(2.0);
        assert_eq!(scaled.as_slice(), &[3.0, -4.0, 0.0]);
    }

    #[test]
    fn test_scale_complex_f64() {
        use num::Complex;
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(-3.0, 4.0)]);
        let scalar = Complex::new(2.0, 0.0);
        let scaled = v.scale(scalar);
        assert_eq!(scaled.as_slice(), &[Complex::new(2.0, 4.0), Complex::new(-6.0, 8.0)]);
    }

    // -- mut_scale --
    #[test]
    fn test_mut_scale_i32() {
        let mut v = FVector::from_vec(vec![1, 2, 3]);
        v.mut_scale(10);
        assert_eq!(v.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_mut_scale_f64() {
        let mut v = FVector::from_vec(vec![1.5, -2.0, 0.0]);
        v.mut_scale(2.0);
        assert_eq!(v.as_slice(), &[3.0, -4.0, 0.0]);
    }

    #[test]
    fn test_mut_scale_complex_f64() {
        let mut v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(-3.0, 4.0)]);
        let scalar = Complex::new(2.0, 0.0);
        v.mut_scale(scalar);
        assert_eq!(v.as_slice(), &[Complex::new(2.0, 4.0), Complex::new(-6.0, 8.0)]);
    }

    // -- negate --
    #[test]
    fn test_negate_i32() {
        let v = FVector::from_vec(vec![1, -2, 3]);
        let neg = v.negate();
        assert_eq!(neg.as_slice(), &[-1, 2, -3]);
        // original unchanged
        assert_eq!(v.as_slice(), &[1, -2, 3]);
    }

    #[test]
    fn test_negate_f64() {
        let v = FVector::from_vec(vec![1.5, -2.5, 0.0]);
        let neg = v.negate();
        assert_eq!(neg.as_slice(), &[-1.5, 2.5, -0.0]);
        assert_eq!(v.as_slice(), &[1.5, -2.5, 0.0]);
    }

    #[test]
    fn test_negate_complex_f64() {
        use num::Complex;
        let v = FVector::from_vec(vec![Complex::new(1.0, -2.0), Complex::new(-3.0, 4.0)]);
        let neg = v.negate();
        assert_eq!(neg.as_slice(), &[Complex::new(-1.0, 2.0), Complex::new(3.0, -4.0)]);
        assert_eq!(v.as_slice(), &[Complex::new(1.0, -2.0), Complex::new(-3.0, 4.0)]);
    }

    // -- mut_negate --
    #[test]
    fn test_mut_negate_i32() {
        let mut v = FVector::from_vec(vec![1, -2, 3]);
        v.mut_negate();
        assert_eq!(v.as_slice(), &[-1, 2, -3]);
    }

    #[test]
    fn test_mut_negate_f64() {
        let mut v = FVector::from_vec(vec![1.5, -2.5, 0.0]);
        v.mut_negate();
        assert_eq!(v.as_slice(), &[-1.5, 2.5, -0.0]);
    }

    #[test]
    fn test_mut_negate_complex_f64() {
        use num::Complex;
        let mut v = FVector::from_vec(vec![Complex::new(1.0, -2.0), Complex::new(-3.0, 4.0)]);
        v.mut_negate();
        assert_eq!(v.as_slice(), &[Complex::new(-1.0, 2.0), Complex::new(3.0, -4.0)]);
    }

    // -- mut_zero --
    #[test]
    fn test_mut_zero_i32() {
        let mut v = FVector::from_vec(vec![1, -2, 3]);
        v.mut_zero();
        assert_eq!(v.as_slice(), &[0, 0, 0]);
    }

    #[test]
    fn test_mut_zero_f64() {
        let mut v = FVector::from_vec(vec![1.5, -2.5, 0.0]);
        v.mut_zero();
        assert_eq!(v.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mut_zero_complex_f64() {
        use num::Complex;
        let mut v = FVector::from_vec(vec![Complex::new(1.0, -2.0), Complex::new(-3.0, 4.0)]);
        v.mut_zero();
        assert_eq!(v.as_slice(), &[Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)]);
    }

    // -- dot --
    #[test]
    fn test_dot_i32() {
        let v1 = FVector::from_vec(vec![1, 2, 3]);
        let v2 = FlexVector::from_vec(vec![4, 5, 6]);
        let dot = v1.dot(&v2).unwrap();
        assert_eq!(dot, 1 * 4 + 2 * 5 + 3 * 6); // 32
    }

    #[test]
    fn test_dot_f64() {
        let v1 = FVector::from_vec(vec![1.5, 2.0, -3.0]);
        let v2 = FlexVector::from_vec(vec![2.0, 0.5, 4.0]);
        let dot = v1.dot(&v2).unwrap();
        assert!((dot - (1.5 * 2.0 + 2.0 * 0.5 + -3.0 * 4.0)).abs() < 1e-12);
    }

    #[test]
    fn test_dot_mismatched_lengths() {
        let v1 = FVector::from_vec(vec![1, 2]);
        let v2 = FlexVector::from_vec(vec![4, 5, 6]);
        let dot = v1.dot(&v2);
        assert!(dot.is_err());
    }

    #[test]
    fn test_dot_empty() {
        let v1: FlexVector<i32> = FlexVector::from_vec(vec![]);
        let v2 = FlexVector::from_vec(vec![]);
        let dot = v1.dot(&v2).unwrap();
        assert_eq!(dot, 0);
    }

    // complex number dot product tested in VectorOpsComplex trait impl testing section below

    // -- dot_to_f64 --
    #[test]
    fn test_dot_to_f64_i32() {
        let v1 = FVector::from_vec(vec![1, 2, 3]);
        let v2 = FlexVector::from_vec(vec![4, 5, 6]);
        let dot = v1.dot_to_f64(&v2).unwrap();
        assert!((dot - 32.0).abs() < 1e-12);
    }

    #[test]
    fn test_dot_to_f64_f64() {
        let v1 = FVector::from_vec(vec![1.5, 2.0, -3.0]);
        let v2 = FlexVector::from_vec(vec![2.0, 0.5, 4.0]);
        let dot = v1.dot_to_f64(&v2).unwrap();
        assert!((dot - (1.5 * 2.0 + 2.0 * 0.5 + -3.0 * 4.0)).abs() < 1e-12);
    }

    #[test]
    fn test_dot_to_f64_mismatched_lengths() {
        let v1 = FVector::from_vec(vec![1, 2]);
        let v2 = FlexVector::from_vec(vec![4, 5, 6]);
        let dot = v1.dot(&v2);
        assert!(dot.is_err());
    }

    // complex number dot_to_f64 tested in VectorOpsComplex trait impl testing section below

    // -- cross --
    #[test]
    fn test_cross_i32() {
        let v1 = FVector::from_vec(vec![1, 2, 3]);
        let v2 = FlexVector::from_vec(vec![4, 5, 6]);
        let cross = v1.cross(&v2).unwrap();
        // [2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4] = [12-15, 12-6, 5-8] = [-3, 6, -3]
        assert_eq!(cross.as_slice(), &[-3, 6, -3]);
    }

    #[test]
    fn test_cross_f64() {
        let v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![4.0, 5.0, 6.0]);
        let cross = v1.cross(&v2).unwrap();
        assert_eq!(cross.as_slice(), &[-3.0, 6.0, -3.0]);
    }

    // intentionally skipping complex number cross product testing
    // due to lack of universally agreed upon definition in the
    // complex vector space.

    #[test]
    fn test_cross_wrong_length_1() {
        let v1 = FVector::from_vec(vec![1, 2]);
        let v2 = FlexVector::from_vec(vec![3, 4, 5]);
        let result = v1.cross(&v2);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_wrong_length_2() {
        let v1 = FVector::from_vec(vec![1, 2, 3]);
        let v2 = FlexVector::from_vec(vec![3, 4]);
        let result = v1.cross(&v2);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_wrong_length_3() {
        let v1 = FVector::from_vec(vec![1, 2]);
        let v2 = FlexVector::from_vec(vec![3, 4]);
        let result = v1.cross(&v2);
        assert!(result.is_err());
    }

    // -- sum --
    #[test]
    fn test_sum_i32() {
        let v = FVector::from_vec(vec![1, 2, 3, 4]);
        let s = v.sum();
        assert_eq!(s, 10);
    }

    #[test]
    fn test_sum_f64() {
        let v = FVector::from_vec(vec![1.5, -2.5, 3.0]);
        let s = v.sum();
        assert!((s - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_complex_f64() {
        use num::Complex;
        let v = FVector::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(-3.0, 4.0),
            Complex::new(5.0, -6.0),
        ]);
        let s = v.sum();
        assert_eq!(s, Complex::new(3.0, 0.0));
    }

    // -- product --
    #[test]
    fn test_product_i32() {
        let v = FVector::from_vec(vec![2, 3, 4]);
        let p = v.product();
        assert_eq!(p, 24);
    }

    #[test]
    fn test_product_f64() {
        let v = FVector::from_vec(vec![1.5, -2.0, 3.0]);
        let p = v.product();
        assert!((p - (1.5 * -2.0 * 3.0)).abs() < 1e-12);
    }

    #[test]
    fn test_product_complex_f64() {
        use num::Complex;
        let v = FVector::from_vec(vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, -1.0),
            Complex::new(2.0, 0.5),
        ]);
        let p = v.product();
        let expected = Complex::new(1.0, 2.0) * Complex::new(3.0, -1.0) * Complex::new(2.0, 0.5);
        assert!((p - expected).norm() < 1e-12);
    }

    // -- minimum --
    #[test]
    fn test_minimum_i32() {
        let v = FVector::from_vec(vec![3, 1, 4, 2]);
        assert_eq!(v.minimum(), Some(1));
    }

    #[test]
    fn test_minimum_f64_basic() {
        let v = FVector::from_vec(vec![1.5, -2.0, 3.0]);
        assert_eq!(v.minimum(), Some(-2.0));
    }

    #[test]
    fn test_minimum_f64_all_positive() {
        let v = FVector::from_vec(vec![2.0, 4.0, 1.0, 3.0]);
        assert_eq!(v.minimum(), Some(1.0));
    }

    #[test]
    fn test_minimum_f64_all_negative() {
        let v = FVector::from_vec(vec![-1.0, -2.0, -3.0]);
        assert_eq!(v.minimum(), Some(-3.0));
    }

    #[test]
    fn test_minimum_f64_single_element() {
        let v = FVector::from_vec(vec![42.0]);
        assert_eq!(v.minimum(), Some(42.0));
    }

    #[test]
    fn test_minimum_f64_empty() {
        let v: FlexVector<f64> = FlexVector::new();
        assert_eq!(v.minimum(), None);
    }

    #[test]
    fn test_minimum_f64_with_nan() {
        let v = FVector::from_vec(vec![1.0, f64::NAN, 2.0]);
        // The result is not guaranteed to be meaningful if NaN is present,
        // but it should return Some value (could be NaN or a number).
        assert!(v.minimum().is_some());
    }

    #[test]
    fn test_minimum_empty() {
        let v: FlexVector<i32> = FlexVector::new();
        assert_eq!(v.minimum(), None);
    }

    // -- elementwise_min --

    #[test]
    fn test_elementwise_min_i32() {
        let v1 = FVector::from_vec(vec![1, 5, 3, 7]);
        let v2 = FlexVector::from_vec(vec![2, 4, 6, 0]);
        let min = v1.elementwise_min(&v2).unwrap();
        assert_eq!(min.as_slice(), &[1, 4, 3, 0]);
    }

    #[test]
    fn test_elementwise_min_f64() {
        let v1 = FVector::from_vec(vec![1.5, -2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![2.5, -3.0, 2.0]);
        let min = v1.elementwise_min(&v2).unwrap();
        assert_eq!(min.as_slice(), &[1.5, -3.0, 2.0]);
    }

    #[test]
    fn test_elementwise_min_empty() {
        let v1: FlexVector<i32> = FlexVector::new();
        let v2: FlexVector<i32> = FlexVector::new();
        let min = v1.elementwise_min(&v2).unwrap();
        assert!(min.is_empty());
    }

    #[test]
    fn test_elementwise_min_mismatched_length() {
        let v1 = FVector::from_vec(vec![1, 2]);
        let v2 = FlexVector::from_vec(vec![3, 4, 5]);
        let result = v1.elementwise_min(&v2);
        assert!(result.is_err());
    }

    #[test]
    fn test_elementwise_min_f64_with_nan() {
        let v1 = FVector::from_vec(vec![1.0, f64::NAN, 3.0]);
        let v2 = FlexVector::from_vec(vec![2.0, 2.0, f64::NAN]);
        let min = v1.elementwise_min(&v2).unwrap();
        assert_eq!(min[0], 1.0); // min(1.0, 2.0) = 1.0
        assert_eq!(min[1], 2.0); // min(NaN, 2.0) = 2.0 (since NaN < 2.0 is false)
        assert!(min[2].is_nan()); // min(3.0, NaN) = NaN (since 3.0 < NaN is false, returns NaN)
    }

    #[test]
    fn test_elementwise_min_f64_both_nan() {
        let v1 = FVector::from_vec(vec![f64::NAN]);
        let v2 = FlexVector::from_vec(vec![f64::NAN]);
        let min = v1.elementwise_min(&v2).unwrap();
        assert!(min[0].is_nan());
    }

    // -- maximum --
    #[test]
    fn test_maximum_i32() {
        let v = FVector::from_vec(vec![3, 1, 4, 2]);
        assert_eq!(v.maximum(), Some(4));
    }

    #[test]
    fn test_maximum_f64_basic() {
        let v = FVector::from_vec(vec![1.5, -2.0, 3.0]);
        assert_eq!(v.maximum(), Some(3.0));
    }

    #[test]
    fn test_maximum_f64_all_positive() {
        let v = FVector::from_vec(vec![2.0, 4.0, 1.0, 3.0]);
        assert_eq!(v.maximum(), Some(4.0));
    }

    #[test]
    fn test_maximum_f64_all_negative() {
        let v = FVector::from_vec(vec![-1.0, -2.0, -3.0]);
        assert_eq!(v.maximum(), Some(-1.0));
    }

    #[test]
    fn test_maximum_f64_single_element() {
        let v = FVector::from_vec(vec![42.0]);
        assert_eq!(v.maximum(), Some(42.0));
    }

    #[test]
    fn test_maximum_f64_empty() {
        let v: FlexVector<f64> = FlexVector::new();
        assert_eq!(v.maximum(), None);
    }

    #[test]
    fn test_maximum_f64_with_nan() {
        let v = FVector::from_vec(vec![1.0, f64::NAN, 2.0]);
        // The result is not guaranteed to be meaningful if NaN is present,
        // but it should return Some value (could be NaN or a number).
        assert!(v.maximum().is_some());
    }

    // -- elementwise_max --

    #[test]
    fn test_elementwise_max_i32() {
        let v1 = FVector::from_vec(vec![1, 5, 3, 7]);
        let v2 = FlexVector::from_vec(vec![2, 4, 6, 0]);
        let max = v1.elementwise_max(&v2).unwrap();
        assert_eq!(max.as_slice(), &[2, 5, 6, 7]);
    }

    #[test]
    fn test_elementwise_max_f64() {
        let v1 = FVector::from_vec(vec![1.5, -2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![2.5, -3.0, 2.0]);
        let max = v1.elementwise_max(&v2).unwrap();
        assert_eq!(max.as_slice(), &[2.5, -2.0, 3.0]);
    }

    #[test]
    fn test_elementwise_max_empty() {
        let v1: FlexVector<i32> = FlexVector::new();
        let v2: FlexVector<i32> = FlexVector::new();
        let max = v1.elementwise_max(&v2).unwrap();
        assert!(max.is_empty());
    }

    #[test]
    fn test_elementwise_max_mismatched_length() {
        let v1 = FVector::from_vec(vec![1, 2, 3]);
        let v2 = FlexVector::from_vec(vec![4, 5]);
        let result = v1.elementwise_max(&v2);
        assert!(result.is_err());
    }

    #[test]
    fn test_elementwise_max_f64_with_nan() {
        let v1 = FVector::from_vec(vec![1.0, f64::NAN, 3.0]);
        let v2 = FlexVector::from_vec(vec![2.0, 2.0, f64::NAN]);
        let max = v1.elementwise_max(&v2).unwrap();
        assert_eq!(max[0], 2.0); // max(1.0, 2.0) = 2.0
        assert_eq!(max[1], 2.0); // max(NaN, 2.0) = 2.0 (since NaN > 2.0 is false)
        assert!(max[2].is_nan()); // max(3.0, NaN) = NaN (since 3.0 > NaN is false, returns NaN)
    }

    #[test]
    fn test_elementwise_max_f64_both_nan() {
        let v1 = FVector::from_vec(vec![f64::NAN]);
        let v2 = FlexVector::from_vec(vec![f64::NAN]);
        let max = v1.elementwise_max(&v2).unwrap();
        assert!(max[0].is_nan());
    }

    // -- elementwise_clamp --

    #[test]
    fn test_elementwise_clamp_i32_basic() {
        let v = FVector::from_vec(vec![-5, 0, 5, 10, 15]);
        let clamped = v.elementwise_clamp(0, 10);
        assert_eq!(clamped.as_slice(), &[0, 0, 5, 10, 10]);
    }

    #[test]
    fn test_elementwise_clamp_i32_all_below() {
        let v = FVector::from_vec(vec![-3, -2, -1]);
        let clamped = v.elementwise_clamp(0, 5);
        assert_eq!(clamped.as_slice(), &[0, 0, 0]);
    }

    #[test]
    fn test_elementwise_clamp_i32_all_above() {
        let v = FVector::from_vec(vec![11, 12, 13]);
        let clamped = v.elementwise_clamp(0, 10);
        assert_eq!(clamped.as_slice(), &[10, 10, 10]);
    }

    #[test]
    fn test_elementwise_clamp_i32_empty() {
        let v: FlexVector<i32> = FlexVector::new();
        let clamped = v.elementwise_clamp(0, 10);
        assert!(clamped.is_empty());
    }

    #[test]
    fn test_elementwise_clamp_f64_basic() {
        let v = FVector::from_vec(vec![-2.5, 0.0, 3.5, 7.2, 12.0]);
        let clamped = v.elementwise_clamp(0.0, 10.0);
        assert_eq!(clamped.as_slice(), &[0.0, 0.0, 3.5, 7.2, 10.0]);
    }

    #[test]
    fn test_elementwise_clamp_f64_all_below() {
        let v = FVector::from_vec(vec![-1.1, -2.2]);
        let clamped = v.elementwise_clamp(0.0, 5.0);
        assert_eq!(clamped.as_slice(), &[0.0, 0.0]);
    }

    #[test]
    fn test_elementwise_clamp_f64_all_above() {
        let v = FVector::from_vec(vec![11.1, 12.2]);
        let clamped = v.elementwise_clamp(0.0, 10.0);
        assert_eq!(clamped.as_slice(), &[10.0, 10.0]);
    }

    #[test]
    fn test_elementwise_clamp_f64_with_nan() {
        let v = FVector::from_vec(vec![1.0, f64::NAN, 5.0]);
        let clamped = v.elementwise_clamp(0.0, 4.0);
        assert_eq!(clamped[0], 1.0);
        assert!(clamped[1].is_nan());
        assert_eq!(clamped[2], 4.0);
    }

    #[test]
    fn test_elementwise_clamp_f64_empty() {
        let v: FlexVector<f64> = FlexVector::new();
        let clamped = v.elementwise_clamp(0.0, 10.0);
        assert!(clamped.is_empty());
    }

    // -- l1_norm --
    #[test]
    fn test_l1_norm_i32() {
        let v = FVector::from_vec(vec![1, -2, 3]);
        let norm = v.l1_norm();
        assert_eq!(norm, 6);
    }

    #[test]
    fn test_l1_norm_f64() {
        let v = FVector::from_vec(vec![1.5, -2.5, 3.0]);
        let norm = v.l1_norm();
        assert!((norm - 7.0).abs() < 1e-12);
    }

    // l1_norm testing of complex number types is in the VectorOpsComplex trait impl tests below

    // -- linf_norm --
    #[test]
    fn test_linf_norm_i32() {
        let v = FVector::from_vec(vec![1, -5, 3, 2]);
        let norm = v.linf_norm();
        assert_eq!(norm, 5);
    }

    #[test]
    fn test_linf_norm_f64() {
        let v = FVector::from_vec(vec![1.5, -2.5, 3.0, -7.2]);
        let norm = v.linf_norm();
        assert!((norm - 7.2).abs() < 1e-12);
    }

    // ================================
    //
    // VectorOpsFloat trait tests
    //
    // ================================

    // -- normalize --
    #[test]
    fn test_normalize_f64() {
        let v = FVector::from_vec(vec![3.0, 4.0]);
        let normalized = v.normalize().unwrap();
        // The norm is 5.0, so the normalized vector should be [0.6, 0.8]
        assert!((normalized.as_slice()[0] - 0.6).abs() < 1e-12);
        assert!((normalized.as_slice()[1] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_normalize_f64_zero_vector() {
        let v = FVector::from_vec(vec![0.0, 0.0]);
        let result = v.normalize();
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_f64_negative_values() {
        let v = FVector::from_vec(vec![-3.0, -4.0]);
        let normalized = v.normalize().unwrap();
        // The norm is 5.0, so the normalized vector should be [-0.6, -0.8]
        assert!((normalized.as_slice()[0] + 0.6).abs() < 1e-12);
        assert!((normalized.as_slice()[1] + 0.8).abs() < 1e-12);
    }

    // -- mut_normalize --
    #[test]
    fn test_mut_normalize_f64() {
        let mut v = FVector::from_vec(vec![3.0, 4.0]);
        v.mut_normalize().unwrap();
        // The norm is 5.0, so the normalized vector should be [0.6, 0.8]
        assert!((v.as_slice()[0] - 0.6).abs() < 1e-12);
        assert!((v.as_slice()[1] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_mut_normalize_f64_zero_vector() {
        let mut v = FVector::from_vec(vec![0.0, 0.0]);
        let result = v.mut_normalize();
        assert!(result.is_err());
    }

    #[test]
    fn test_mut_normalize_f64_negative_values() {
        let mut v = FVector::from_vec(vec![-3.0, -4.0]);
        v.mut_normalize().unwrap();
        // The norm is 5.0, so the normalized vector should be [-0.6, -0.8]
        assert!((v.as_slice()[0] + 0.6).abs() < 1e-12);
        assert!((v.as_slice()[1] + 0.8).abs() < 1e-12);
    }

    // -- normalize_to --
    #[test]
    fn test_normalize_to_f64() {
        let v = FVector::from_vec(vec![3.0, 4.0]);
        let normalized = v.normalize_to(10.0).unwrap();
        // The original norm is 5.0, so the normalized vector should be [6.0, 8.0]
        assert!((normalized.as_slice()[0] - 6.0).abs() < 1e-12);
        assert!((normalized.as_slice()[1] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_normalize_to_f64_zero_vector() {
        let v = FVector::from_vec(vec![0.0, 0.0]);
        let result = v.normalize_to(1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_to_f64_negative_values() {
        let v = FVector::from_vec(vec![-3.0, -4.0]);
        let normalized = v.normalize_to(5.0).unwrap();
        // The original norm is 5.0, so the normalized vector should be [-3.0, -4.0]
        assert!((normalized.as_slice()[0] + 3.0).abs() < 1e-12);
        assert!((normalized.as_slice()[1] + 4.0).abs() < 1e-12);
    }

    // -- mut_normalize_to --
    #[test]
    fn test_mut_normalize_to_f64() {
        let mut v = FVector::from_vec(vec![3.0, 4.0]);
        v.mut_normalize_to(10.0).unwrap();
        // The original norm is 5.0, so the normalized vector should be [6.0, 8.0]
        assert!((v.as_slice()[0] - 6.0).abs() < 1e-12);
        assert!((v.as_slice()[1] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_mut_normalize_to_f64_zero_vector() {
        let mut v = FVector::from_vec(vec![0.0, 0.0]);
        let result = v.mut_normalize_to(1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_mut_normalize_to_f64_negative_values() {
        let mut v = FVector::from_vec(vec![-3.0, -4.0]);
        v.mut_normalize_to(5.0).unwrap();
        // The original norm is 5.0, so the normalized vector should be [-3.0, -4.0]
        assert!((v.as_slice()[0] + 3.0).abs() < 1e-12);
        assert!((v.as_slice()[1] + 4.0).abs() < 1e-12);
    }

    // -- lerp --
    #[test]
    fn test_lerp_f64_weight_zero() {
        let v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![4.0, 5.0, 6.0]);
        let result = v1.lerp(&v2, 0.0).unwrap();
        // Should be equal to v1
        assert!((result.as_slice()[0] - 1.0).abs() < 1e-12);
        assert!((result.as_slice()[1] - 2.0).abs() < 1e-12);
        assert!((result.as_slice()[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_lerp_f64_weight_one() {
        let v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![4.0, 5.0, 6.0]);
        let result = v1.lerp(&v2, 1.0).unwrap();
        // Should be equal to v2
        assert!((result.as_slice()[0] - 4.0).abs() < 1e-12);
        assert!((result.as_slice()[1] - 5.0).abs() < 1e-12);
        assert!((result.as_slice()[2] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_lerp_f64_weight_half() {
        let v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![4.0, 5.0, 6.0]);
        let result = v1.lerp(&v2, 0.5).unwrap();
        // Should be the midpoint
        assert!((result.as_slice()[0] - 2.5).abs() < 1e-12);
        assert!((result.as_slice()[1] - 3.5).abs() < 1e-12);
        assert!((result.as_slice()[2] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn test_lerp_f64_weight_out_of_bounds() {
        let v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![4.0, 5.0, 6.0]);
        let result_low = v1.lerp(&v2, -0.1);
        let result_high = v1.lerp(&v2, 1.1);
        assert!(result_low.is_err());
        assert!(result_high.is_err());
    }

    // -- mut_lerp --
    #[test]
    fn test_mut_lerp_f64_weight_zero() {
        let mut v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![4.0, 5.0, 6.0]);
        v1.mut_lerp(&v2, 0.0).unwrap();
        // Should be equal to original v1
        assert!((v1.as_slice()[0] - 1.0).abs() < 1e-12);
        assert!((v1.as_slice()[1] - 2.0).abs() < 1e-12);
        assert!((v1.as_slice()[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_mut_lerp_f64_weight_one() {
        let mut v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![4.0, 5.0, 6.0]);
        v1.mut_lerp(&v2, 1.0).unwrap();
        // Should be equal to v2
        assert!((v1.as_slice()[0] - 4.0).abs() < 1e-12);
        assert!((v1.as_slice()[1] - 5.0).abs() < 1e-12);
        assert!((v1.as_slice()[2] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_mut_lerp_f64_weight_half() {
        let mut v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![4.0, 5.0, 6.0]);
        v1.mut_lerp(&v2, 0.5).unwrap();
        // Should be the midpoint
        assert!((v1.as_slice()[0] - 2.5).abs() < 1e-12);
        assert!((v1.as_slice()[1] - 3.5).abs() < 1e-12);
        assert!((v1.as_slice()[2] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn test_mut_lerp_f64_weight_out_of_bounds() {
        let mut v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![4.0, 5.0, 6.0]);
        let result_low = v1.mut_lerp(&v2, -0.1);
        let result_high = v1.mut_lerp(&v2, 1.1);
        assert!(result_low.is_err());
        assert!(result_high.is_err());
    }

    // -- midpoint --
    #[test]
    fn test_midpoint_f64() {
        let v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![4.0, 5.0, 6.0]);
        let midpoint = v1.midpoint(&v2).unwrap();
        // Should be the average of each component
        assert!((midpoint.as_slice()[0] - 2.5).abs() < 1e-12);
        assert!((midpoint.as_slice()[1] - 3.5).abs() < 1e-12);
        assert!((midpoint.as_slice()[2] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn test_midpoint_f64_negative_values() {
        let v1 = FVector::from_vec(vec![-1.0, -2.0, -3.0]);
        let v2 = FlexVector::from_vec(vec![1.0, 2.0, 3.0]);
        let midpoint = v1.midpoint(&v2).unwrap();
        // Should be [0.0, 0.0, 0.0]
        assert!((midpoint.as_slice()[0]).abs() < 1e-12);
        assert!((midpoint.as_slice()[1]).abs() < 1e-12);
        assert!((midpoint.as_slice()[2]).abs() < 1e-12);
    }

    #[test]
    fn test_midpoint_f64_mismatched_length() {
        let v1 = FVector::from_vec(vec![1.0, 2.0]);
        let v2 = FlexVector::from_vec(vec![3.0, 4.0, 5.0]);
        let result = v1.midpoint(&v2);
        assert!(result.is_err());
    }

    // -- distance --
    #[test]
    fn test_distance_f64_basic() {
        let v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![4.0, 6.0, 8.0]);
        let dist = v1.distance(&v2).unwrap();
        // sqrt((1-4)^2 + (2-6)^2 + (3-8)^2) = sqrt(9 + 16 + 25) = sqrt(50)
        assert!((dist - 50f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_distance_f64_zero() {
        let v1 = FVector::from_vec(vec![0.0, 0.0, 0.0]);
        let v2 = FlexVector::from_vec(vec![0.0, 0.0, 0.0]);
        let dist = v1.distance(&v2).unwrap();
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_distance_f64_negative_values() {
        let v1 = FVector::from_vec(vec![-1.0, -2.0, -3.0]);
        let v2 = FlexVector::from_vec(vec![1.0, 2.0, 3.0]);
        let dist = v1.distance(&v2).unwrap();
        // sqrt(((-1)-1)^2 + ((-2)-2)^2 + ((-3)-3)^2) = sqrt(4 + 16 + 36) = sqrt(56)
        assert!((dist - 56f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_distance_f64_mismatched_length() {
        let v1 = FVector::from_vec(vec![1.0, 2.0]);
        let v2 = FlexVector::from_vec(vec![3.0, 4.0, 5.0]);
        let result = v1.distance(&v2);
        assert!(result.is_err());
    }

    // -- manhattan_distance --
    #[test]
    fn test_manhattan_distance_f64_basic() {
        let v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![4.0, 6.0, 8.0]);
        let dist = v1.manhattan_distance(&v2).unwrap();
        // |1-4| + |2-6| + |3-8| = 3 + 4 + 5 = 12
        assert!((dist - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_manhattan_distance_f64_zero() {
        let v1 = FVector::from_vec(vec![0.0, 0.0, 0.0]);
        let v2 = FlexVector::from_vec(vec![0.0, 0.0, 0.0]);
        let dist = v1.manhattan_distance(&v2).unwrap();
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_manhattan_distance_f64_negative_values() {
        let v1 = FVector::from_vec(vec![-1.0, -2.0, -3.0]);
        let v2 = FlexVector::from_vec(vec![1.0, 2.0, 3.0]);
        let dist = v1.manhattan_distance(&v2).unwrap();
        // |(-1)-1| + |(-2)-2| + |(-3)-3| = 2 + 4 + 6 = 12
        assert!((dist - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_manhattan_distance_f64_mismatched_length() {
        let v1 = FVector::from_vec(vec![1.0, 2.0]);
        let v2 = FlexVector::from_vec(vec![3.0, 4.0, 5.0]);
        let result = v1.manhattan_distance(&v2);
        assert!(result.is_err());
    }

    // -- chebyshev_distance --
    #[test]
    fn test_chebyshev_distance_f64_basic() {
        let v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![4.0, 6.0, 8.0]);
        let dist = v1.chebyshev_distance(&v2).unwrap();
        // max(|1-4|, |2-6|, |3-8|) = max(3, 4, 5) = 5
        assert!((dist - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_chebyshev_distance_f64_zero() {
        let v1 = FVector::from_vec(vec![0.0, 0.0, 0.0]);
        let v2 = FlexVector::from_vec(vec![0.0, 0.0, 0.0]);
        let dist = v1.chebyshev_distance(&v2).unwrap();
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_chebyshev_distance_f64_negative_values() {
        let v1 = FVector::from_vec(vec![-1.0, -2.0, -3.0]);
        let v2 = FlexVector::from_vec(vec![1.0, 2.0, 3.0]);
        let dist = v1.chebyshev_distance(&v2).unwrap();
        // max(|-1-1|, |-2-2|, |-3-3|) = max(2, 4, 6) = 6
        assert!((dist - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_chebyshev_distance_f64_mismatched_length() {
        let v1 = FVector::from_vec(vec![1.0, 2.0]);
        let v2 = FlexVector::from_vec(vec![3.0, 4.0, 5.0]);
        let result = v1.chebyshev_distance(&v2);
        assert!(result.is_err());
    }

    // -- minkowski_distance --
    #[test]
    fn test_minkowski_distance_f64_basic() {
        let v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![4.0, 6.0, 8.0]);
        let dist = v1.minkowski_distance(&v2, 3.0).unwrap();
        // ((|1-4|^3 + |2-6|^3 + |3-8|^3))^(1/3) = (27 + 64 + 125)^(1/3) = (216)^(1/3) = 6
        assert!((dist - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_minkowski_distance_f64_p1() {
        let v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![4.0, 6.0, 8.0]);
        let dist = v1.minkowski_distance(&v2, 1.0).unwrap();
        // Should match manhattan distance: 3 + 4 + 5 = 12
        assert!((dist - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_minkowski_distance_f64_p2() {
        let v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![4.0, 6.0, 8.0]);
        let dist = v1.minkowski_distance(&v2, 2.0).unwrap();
        // Should match euclidean distance: sqrt(9 + 16 + 25) = sqrt(50)
        assert!((dist - 50f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_minkowski_distance_f64_empty() {
        let v1: FlexVector<f64> = FlexVector::new();
        let v2: FlexVector<f64> = FlexVector::new();
        let dist = v1.minkowski_distance(&v2, 2.0).unwrap();
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_minkowski_distance_f64_identical() {
        let v1 = FVector::from_vec(vec![1.23, 4.56, 7.89]);
        let v2 = FlexVector::from_vec(vec![1.23, 4.56, 7.89]);
        let dist = v1.minkowski_distance(&v2, 2.0).unwrap();
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_minkowski_distance_f64_partial() {
        let v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![4.0, 6.0]);
        let result = v1.minkowski_distance(&v2, 2.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_minkowski_distance_f64_invalid_p() {
        let v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![4.0, 6.0, 8.0]);
        let result = v1.minkowski_distance(&v2, 0.5);
        assert!(result.is_err());
    }

    // -- norm --
    #[test]
    fn test_norm_f64_basic() {
        let v = FVector::from_vec(vec![3.0, 4.0]);
        let norm = v.norm();
        // sqrt(3^2 + 4^2) = 5
        assert!((norm - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_norm_f64_zero() {
        let v = FVector::from_vec(vec![0.0, 0.0, 0.0]);
        let norm = v.norm();
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_norm_f64_single_element() {
        let v = FVector::from_vec(vec![7.0]);
        let norm = v.norm();
        assert_eq!(norm, 7.0);
    }

    #[test]
    fn test_norm_f64_negative_values() {
        let v = FVector::from_vec(vec![-3.0, -4.0]);
        let norm = v.norm();
        // sqrt((-3)^2 + (-4)^2) = 5
        assert!((norm - 5.0).abs() < 1e-12);
    }

    // -- magnitude --
    #[test]
    fn test_magnitude_f64_basic() {
        let v = FVector::from_vec(vec![3.0, 4.0]);
        let mag = v.magnitude();
        // sqrt(3^2 + 4^2) = 5
        assert!((mag - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_magnitude_f64_zero() {
        let v = FVector::from_vec(vec![0.0, 0.0, 0.0]);
        let mag = v.magnitude();
        assert_eq!(mag, 0.0);
    }

    #[test]
    fn test_magnitude_f64_single_element() {
        let v = FVector::from_vec(vec![7.0]);
        let mag = v.magnitude();
        assert_eq!(mag, 7.0);
    }

    #[test]
    fn test_magnitude_f64_negative_values() {
        let v = FVector::from_vec(vec![-3.0, -4.0]);
        let mag = v.magnitude();
        // sqrt((-3)^2 + (-4)^2) = 5
        assert!((mag - 5.0).abs() < 1e-12);
    }

    // -- lp_norm --
    #[test]
    fn test_lp_norm_f64_p1() {
        let v = FVector::from_vec(vec![1.0, -2.0, 3.0]);
        let norm = v.lp_norm(1.0).unwrap();
        // L1 norm: |1| + |−2| + |3| = 1 + 2 + 3 = 6
        assert!((norm - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_lp_norm_f64_p2() {
        let v = FVector::from_vec(vec![3.0, 4.0]);
        let norm = v.lp_norm(2.0).unwrap();
        // L2 norm: sqrt(3^2 + 4^2) = 5
        assert!((norm - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_lp_norm_f64_p3() {
        let v = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let norm = v.lp_norm(3.0).unwrap();
        // (|1|^3 + |2|^3 + |3|^3)^(1/3) = (1 + 8 + 27)^(1/3) = 36^(1/3)
        assert!((norm - 36f64.powf(1.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn test_lp_norm_f64_zero() {
        let v = FVector::from_vec(vec![0.0, 0.0, 0.0]);
        let norm = v.lp_norm(2.0).unwrap();
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_lp_norm_f64_invalid_p() {
        let v = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let result = v.lp_norm(0.5);
        assert!(result.is_err());
    }

    // -- angle_with --
    #[test]
    fn test_angle_with_f64_orthogonal() {
        let v1 = FVector::from_vec(vec![1.0, 0.0]);
        let v2 = FlexVector::from_vec(vec![0.0, 1.0]);
        let angle = v1.angle_with(&v2).unwrap();
        // Orthogonal vectors: angle should be pi/2
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn test_angle_with_f64_parallel() {
        let v1 = FVector::from_vec(vec![1.0, 2.0]);
        let v2 = FlexVector::from_vec(vec![2.0, 4.0]);
        let angle = v1.angle_with(&v2).unwrap();
        // Parallel vectors: angle should be 0 (allow for floating-point error)
        assert!(angle.abs() < 1e-7, "angle was {}", angle);
    }

    #[test]
    fn test_angle_with_f64_opposite() {
        let v1 = FVector::from_vec(vec![1.0, 0.0]);
        let v2 = FlexVector::from_vec(vec![-1.0, 0.0]);
        let angle = v1.angle_with(&v2).unwrap();
        // Opposite vectors: angle should be pi
        assert!((angle - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn test_angle_with_f64_identical() {
        let v1 = FVector::from_vec(vec![3.0, 4.0]);
        let v2 = FlexVector::from_vec(vec![3.0, 4.0]);
        let angle = v1.angle_with(&v2).unwrap();
        // Identical vectors: angle should be 0
        assert!(angle.abs() < 1e-12);
    }

    #[test]
    fn test_angle_with_f64_arbitrary() {
        let v1 = FVector::from_vec(vec![1.0, 2.0]);
        let v2 = FlexVector::from_vec(vec![2.0, 1.0]);
        let angle = v1.angle_with(&v2).unwrap();
        // Check that the angle is between 0 and pi
        assert!(angle > 0.0 && angle < std::f64::consts::PI);
    }

    #[test]
    fn test_angle_with_f64_zero_vector() {
        let v1 = FVector::from_vec(vec![0.0, 0.0]);
        let v2 = FlexVector::from_vec(vec![1.0, 2.0]);
        let result = v1.angle_with(&v2);
        assert!(result.is_err());
    }

    #[test]
    fn test_angle_with_f64_mismatched_length() {
        let v1 = FVector::from_vec(vec![1.0, 2.0]);
        let v2 = FlexVector::from_vec(vec![3.0, 4.0, 5.0]);
        let result = v1.angle_with(&v2);
        assert!(result.is_err());
    }

    // -- project_onto --
    #[test]
    fn test_project_onto_f64_basic() {
        let v1 = FVector::from_vec(vec![3.0, 4.0]);
        let v2 = FlexVector::from_vec(vec![1.0, 0.0]);
        let proj = v1.project_onto(&v2).unwrap();
        // Projection of [3,4] onto [1,0] is [3,0]
        assert!((proj.as_slice()[0] - 3.0).abs() < 1e-12);
        assert!((proj.as_slice()[1] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_project_onto_f64_parallel() {
        let v1 = FVector::from_vec(vec![2.0, 4.0]);
        let v2 = FlexVector::from_vec(vec![1.0, 2.0]);
        let proj = v1.project_onto(&v2).unwrap();
        // v1 is parallel to v2, so projection should be v1 itself
        assert!((proj.as_slice()[0] - 2.0).abs() < 1e-12);
        assert!((proj.as_slice()[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_project_onto_f64_orthogonal() {
        let v1 = FVector::from_vec(vec![0.0, 1.0]);
        let v2 = FlexVector::from_vec(vec![1.0, 0.0]);
        let proj = v1.project_onto(&v2).unwrap();
        // Orthogonal vectors: projection should be [0,0]
        assert!((proj.as_slice()[0] - 0.0).abs() < 1e-12);
        assert!((proj.as_slice()[1] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_project_onto_f64_identical() {
        let v1 = FVector::from_vec(vec![5.0, 5.0]);
        let v2 = FlexVector::from_vec(vec![5.0, 5.0]);
        let proj = v1.project_onto(&v2).unwrap();
        // Should be v1 itself
        assert!((proj.as_slice()[0] - 5.0).abs() < 1e-12);
        assert!((proj.as_slice()[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_project_onto_f64_zero_vector() {
        let v1 = FVector::from_vec(vec![1.0, 2.0]);
        let v2 = FlexVector::from_vec(vec![0.0, 0.0]);
        let result = v1.project_onto(&v2);
        assert!(result.is_err());
    }

    #[test]
    fn test_project_onto_f64_mismatched_length() {
        let v1 = FVector::from_vec(vec![1.0, 2.0]);
        let v2 = FlexVector::from_vec(vec![3.0, 4.0, 5.0]);
        let result = v1.project_onto(&v2);
        assert!(result.is_err());
    }

    // --- cosine_similarity ---
    #[test]
    fn test_cosine_similarity_f64_parallel() {
        let v1 = FVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = FlexVector::from_vec(vec![2.0, 4.0, 6.0]);
        let cos_sim = v1.cosine_similarity(&v2).unwrap();
        assert!((cos_sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_f64_orthogonal() {
        let v1 = FVector::from_vec(vec![1.0, 0.0]);
        let v2 = FlexVector::from_vec(vec![0.0, 1.0]);
        let cos_sim = v1.cosine_similarity(&v2).unwrap();
        assert!((cos_sim - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_f64_opposite() {
        let v1 = FVector::from_vec(vec![1.0, 0.0]);
        let v2 = FlexVector::from_vec(vec![-1.0, 0.0]);
        let cos_sim = v1.cosine_similarity(&v2).unwrap();
        assert!((cos_sim + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_f64_zero_vector() {
        let v1 = FVector::from_vec(vec![0.0, 0.0]);
        let v2 = FlexVector::from_vec(vec![1.0, 2.0]);
        let cos_sim = v1.cosine_similarity(&v2);
        assert!(cos_sim.is_err());
    }

    // ================================
    //
    // VectorOpsComplex trait tests
    //
    // ================================

    // -- normalize --
    #[test]
    fn test_normalize_complex_f64_basic() {
        let v = FVector::from_vec(vec![Complex::new(3.0, 4.0)]);
        let normalized = v.normalize().unwrap();
        // The norm is 5.0, so the normalized vector should be [0.6 + 0.8i]
        assert!((normalized.as_slice()[0].re - 0.6).abs() < 1e-12);
        assert!((normalized.as_slice()[0].im - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_normalize_complex_f64_multiple_elements() {
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let norm = ((1.0 * 1.0 + 2.0 * 2.0) + (3.0 * 3.0 + 4.0 * 4.0)).sqrt();
        let normalized = v.normalize().unwrap();
        assert!((normalized.as_slice()[0].re - 1.0 / norm).abs() < 1e-12);
        assert!((normalized.as_slice()[0].im - 2.0 / norm).abs() < 1e-12);
        assert!((normalized.as_slice()[1].re - 3.0 / norm).abs() < 1e-12);
        assert!((normalized.as_slice()[1].im - 4.0 / norm).abs() < 1e-12);
    }

    #[test]
    fn test_normalize_complex_f64_zero_vector() {
        let v = FVector::from_vec(vec![Complex::new(0.0, 0.0)]);
        let result = v.normalize();
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_complex_f64_empty() {
        let v: FlexVector<Complex<f64>> = FlexVector::new();
        let normalized = v.normalize();
        assert!(normalized.is_err());
    }

    //TODO: continue tests for all methods in VectorOpsComplex trait

    // ================================
    //
    // Operator overload trait tests
    //
    // ================================

    #[test]
    fn test_neg() {
        let v = FVector::from_vec(vec![1, -2, 3]);
        let neg_v = -v;
        assert_eq!(neg_v.as_slice(), &[-1, 2, -3]);
    }

    #[test]
    fn test_neg_f64() {
        let v = FVector::from_vec(vec![1.5, -2.5, 0.0]);
        let neg_v = -v;
        assert_eq!(neg_v.as_slice(), &[-1.5, 2.5, -0.0]);
    }

    #[test]
    fn test_neg_complex() {
        let v = FVector::from_vec(vec![Complex::new(1.0, -2.0), Complex::new(-3.0, 4.0)]);
        let neg_v = -v;
        assert_eq!(neg_v.as_slice(), &[Complex::new(-1.0, 2.0), Complex::new(3.0, -4.0)]);
    }

    #[test]
    fn test_neg_nan() {
        let v = FVector::from_vec(vec![f64::NAN, -f64::NAN]);
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
        let v = FVector::from_vec(vec![f64::INFINITY, f64::NEG_INFINITY]);
        let neg_v = -v;
        assert_eq!(neg_v.as_slice(), &[-f64::INFINITY, f64::INFINITY]);
    }

    #[test]
    fn test_add() {
        let v1 = FVector::from_vec(vec![1, 2, 3]);
        let v2 = FlexVector::from_vec(vec![4, 5, 6]);
        let sum = v1 + v2;
        assert_eq!(sum.as_slice(), &[5, 7, 9]);
    }

    #[test]
    fn test_add_f64() {
        let v1 = FVector::from_vec(vec![1.0, 2.5]);
        let v2 = FlexVector::from_vec(vec![3.0, 4.5]);
        let sum = v1 + v2;
        assert_eq!(sum.as_slice(), &[4.0, 7.0]);
    }

    #[test]
    fn test_add_complex() {
        let v1 = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let v2 = FlexVector::from_vec(vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)]);
        let sum = v1 + v2;
        assert_eq!(sum.as_slice(), &[Complex::new(6.0, 8.0), Complex::new(10.0, 12.0)]);
    }

    #[test]
    fn test_add_nan_infinity() {
        let v1 = FVector::from_vec(vec![f64::NAN, f64::INFINITY, 1.0]);
        let v2 = FlexVector::from_vec(vec![1.0, 2.0, f64::INFINITY]);
        let sum = v1 + v2;
        assert!(sum.as_slice()[0].is_nan());
        assert_eq!(sum.as_slice()[1], f64::INFINITY);
        assert_eq!(sum.as_slice()[2], f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "Vector length mismatch")]
    fn test_add_panic_on_mismatched_length() {
        let v1 = FVector::from_vec(vec![1, 2]);
        let v2 = FlexVector::from_vec(vec![1, 2, 3]);
        let _ = v1 + v2;
    }

    #[test]
    fn test_sub() {
        let v1 = FVector::from_vec(vec![10, 20, 30]);
        let v2 = FlexVector::from_vec(vec![1, 2, 3]);
        let diff = v1 - v2;
        assert_eq!(diff.as_slice(), &[9, 18, 27]);
    }

    #[test]
    fn test_sub_f64() {
        let v1 = FVector::from_vec(vec![5.5, 2.0]);
        let v2 = FlexVector::from_vec(vec![1.5, 1.0]);
        let diff = v1 - v2;
        assert_eq!(diff.as_slice(), &[4.0, 1.0]);
    }

    #[test]
    fn test_sub_complex() {
        let v1 = FVector::from_vec(vec![Complex::new(5.0, 7.0), Complex::new(3.0, 4.0)]);
        let v2 = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(2.0, 1.0)]);
        let diff = v1 - v2;
        assert_eq!(diff.as_slice(), &[Complex::new(4.0, 5.0), Complex::new(1.0, 3.0)]);
    }

    #[test]
    fn test_sub_nan_infinity() {
        let v1 = FVector::from_vec(vec![f64::NAN, f64::INFINITY, 5.0]);
        let v2 = FlexVector::from_vec(vec![2.0, f64::INFINITY, f64::NAN]);
        let diff = v1 - v2;
        assert!(diff.as_slice()[0].is_nan());
        assert!(diff.as_slice()[1].is_nan()); // inf - inf = NaN
        assert!(diff.as_slice()[2].is_nan());
    }

    #[test]
    #[should_panic(expected = "Vector length mismatch")]
    fn test_sub_panic_on_mismatched_length() {
        let v1 = FVector::from_vec(vec![1, 2]);
        let v2 = FlexVector::from_vec(vec![1, 2, 3]);
        let _ = v1 - v2;
    }

    #[test]
    fn test_mul() {
        let v1 = FVector::from_vec(vec![2, 3, 4]);
        let v2 = FlexVector::from_vec(vec![5, 6, 7]);
        let prod = v1 * v2;
        assert_eq!(prod.as_slice(), &[10, 18, 28]);
    }

    #[test]
    fn test_mul_f64() {
        let v1 = FVector::from_vec(vec![1.5, 2.0]);
        let v2 = FlexVector::from_vec(vec![2.0, 3.0]);
        let prod = v1 * v2;
        assert_eq!(prod.as_slice(), &[3.0, 6.0]);
    }

    #[test]
    fn test_mul_complex() {
        let v1 = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
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
        let v1 = FVector::from_vec(vec![f64::NAN, f64::INFINITY, 2.0]);
        let v2 = FlexVector::from_vec(vec![3.0, 2.0, f64::INFINITY]);
        let prod = v1 * v2;
        assert!(prod.as_slice()[0].is_nan());
        assert_eq!(prod.as_slice()[1], f64::INFINITY);
        assert_eq!(prod.as_slice()[2], f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "Vector length mismatch")]
    fn test_mul_panic_on_mismatched_length() {
        let v1 = FVector::from_vec(vec![1, 2]);
        let v2 = FlexVector::from_vec(vec![1, 2, 3]);
        let _ = v1 * v2;
    }

    #[test]
    fn test_elem_div_f32() {
        let v1 = FVector::from_vec(vec![2.0f32, 4.0, 8.0]);
        let v2 = FlexVector::from_vec(vec![1.0f32, 2.0, 4.0]);
        let result = v1 / v2;
        assert_eq!(result.as_slice(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_elem_div_f64() {
        let v1 = FVector::from_vec(vec![2.0f64, 4.0, 8.0]);
        let v2 = FlexVector::from_vec(vec![1.0f64, 2.0, 4.0]);
        let result = v1 / v2;
        assert_eq!(result.as_slice(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_elem_div_complex_f32() {
        let v1 = FVector::from_vec(vec![Complex::new(2.0f32, 2.0), Complex::new(4.0, 0.0)]);
        let v2 = FlexVector::from_vec(vec![Complex::new(1.0f32, 1.0), Complex::new(2.0, 0.0)]);
        let result = v1 / v2;
        assert!((result[0] - Complex::new(2.0, 0.0)).norm() < 1e-6);
        assert!((result[1] - Complex::new(2.0, 0.0)).norm() < 1e-6);
    }

    #[test]
    fn test_elem_div_complex_f64() {
        let v1 = FVector::from_vec(vec![Complex::new(2.0f64, 2.0), Complex::new(4.0, 0.0)]);
        let v2 = FlexVector::from_vec(vec![Complex::new(1.0f64, 1.0), Complex::new(2.0, 0.0)]);
        let result = v1 / v2;
        assert!((result[0] - Complex::new(2.0, 0.0)).norm() < 1e-12);
        assert!((result[1] - Complex::new(2.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "Vector length mismatch")]
    fn test_elem_div_panic_on_mismatched_length() {
        let v1 = FVector::from_vec(vec![1.0f64, 2.0]);
        let v2 = FlexVector::from_vec(vec![1.0f64, 2.0, 3.0]);
        let _ = v1 / v2;
    }

    #[test]
    fn test_add_assign() {
        let mut v1 = FVector::from_vec(vec![1, 2, 3]);
        let v2 = FlexVector::from_vec(vec![4, 5, 6]);
        v1 += v2;
        assert_eq!(v1.as_slice(), &[5, 7, 9]);
    }

    #[test]
    fn test_add_assign_f64() {
        let mut v1 = FVector::from_vec(vec![1.0, 2.5]);
        let v2 = FlexVector::from_vec(vec![3.0, 4.5]);
        v1 += v2;
        assert_eq!(v1.as_slice(), &[4.0, 7.0]);
    }

    #[test]
    fn test_add_assign_complex() {
        let mut v1 = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let v2 = FlexVector::from_vec(vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)]);
        v1 += v2;
        assert_eq!(v1.as_slice(), &[Complex::new(6.0, 8.0), Complex::new(10.0, 12.0)]);
    }

    #[test]
    #[should_panic(expected = "Vector length mismatch")]
    fn test_add_assign_panic_on_mismatched_length() {
        let mut v1 = FVector::from_vec(vec![1, 2]);
        let v2 = FlexVector::from_vec(vec![1, 2, 3]);
        v1 += v2;
    }

    #[test]
    fn test_sub_assign() {
        let mut v1 = FVector::from_vec(vec![10, 20, 30]);
        let v2 = FlexVector::from_vec(vec![1, 2, 3]);
        v1 -= v2;
        assert_eq!(v1.as_slice(), &[9, 18, 27]);
    }

    #[test]
    fn test_sub_assign_f64() {
        let mut v1 = FVector::from_vec(vec![5.5, 2.0]);
        let v2 = FlexVector::from_vec(vec![1.5, 1.0]);
        v1 -= v2;
        assert_eq!(v1.as_slice(), &[4.0, 1.0]);
    }

    #[test]
    fn test_sub_assign_complex() {
        let mut v1 = FVector::from_vec(vec![Complex::new(5.0, 7.0), Complex::new(3.0, 4.0)]);
        let v2 = FlexVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(2.0, 1.0)]);
        v1 -= v2;
        assert_eq!(v1.as_slice(), &[Complex::new(4.0, 5.0), Complex::new(1.0, 3.0)]);
    }

    #[test]
    #[should_panic(expected = "Vector length mismatch")]
    fn test_sub_assign_panic_on_mismatched_length() {
        let mut v1 = FVector::from_vec(vec![1, 2]);
        let v2 = FlexVector::from_vec(vec![1, 2, 3]);
        v1 -= v2;
    }

    #[test]
    fn test_mul_assign() {
        let mut v1 = FVector::from_vec(vec![2, 3, 4]);
        let v2 = FlexVector::from_vec(vec![5, 6, 7]);
        v1 *= v2;
        assert_eq!(v1.as_slice(), &[10, 18, 28]);
    }

    #[test]
    fn test_mul_assign_f64() {
        let mut v1 = FVector::from_vec(vec![1.5, 2.0]);
        let v2 = FlexVector::from_vec(vec![2.0, 3.0]);
        v1 *= v2;
        assert_eq!(v1.as_slice(), &[3.0, 6.0]);
    }

    #[test]
    fn test_mul_assign_complex() {
        let mut v1 = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let v2 = FlexVector::from_vec(vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)]);
        v1 *= v2;
        assert_eq!(v1.as_slice(), &[Complex::new(-7.0, 16.0), Complex::new(-11.0, 52.0)]);
    }

    #[test]
    #[should_panic(expected = "Vector length mismatch")]
    fn test_mul_assign_panic_on_mismatched_length() {
        let mut v1 = FVector::from_vec(vec![1, 2]);
        let v2 = FlexVector::from_vec(vec![1, 2, 3]);
        v1 *= v2;
    }

    #[test]
    fn test_elem_div_assign_f32() {
        let mut v1 = FVector::from_vec(vec![2.0f32, 4.0, 8.0]);
        let v2 = FlexVector::from_vec(vec![1.0f32, 2.0, 4.0]);
        v1 /= v2;
        assert_eq!(v1.as_slice(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_elem_div_assign_f64() {
        let mut v1 = FVector::from_vec(vec![2.0f64, 4.0, 8.0]);
        let v2 = FlexVector::from_vec(vec![1.0f64, 2.0, 4.0]);
        v1 /= v2;
        assert_eq!(v1.as_slice(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_elem_div_assign_complex_f32() {
        let mut v1 = FVector::from_vec(vec![Complex::new(2.0f32, 2.0), Complex::new(4.0, 0.0)]);
        let v2 = FlexVector::from_vec(vec![Complex::new(1.0f32, 1.0), Complex::new(2.0, 0.0)]);
        v1 /= v2;
        assert!((v1[0] - Complex::new(2.0, 0.0)).norm() < 1e-6);
        assert!((v1[1] - Complex::new(2.0, 0.0)).norm() < 1e-6);
    }

    #[test]
    fn test_elem_div_assign_complex_f64() {
        let mut v1 = FVector::from_vec(vec![Complex::new(2.0f64, 2.0), Complex::new(4.0, 0.0)]);
        let v2 = FlexVector::from_vec(vec![Complex::new(1.0f64, 1.0), Complex::new(2.0, 0.0)]);
        v1 /= v2;
        assert!((v1[0] - Complex::new(2.0, 0.0)).norm() < 1e-12);
        assert!((v1[1] - Complex::new(2.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "Vector length mismatch")]
    fn test_elem_div_assign_panic_on_mismatched_length() {
        let mut v1 = FVector::from_vec(vec![1.0f64, 2.0]);
        let v2 = FlexVector::from_vec(vec![1.0f64, 2.0, 3.0]);
        v1 /= v2;
    }

    #[test]
    fn test_scalar_mul() {
        let v = FVector::from_vec(vec![2, -3, 4]);
        let prod = v.clone() * 3;
        assert_eq!(prod.as_slice(), &[6, -9, 12]);
    }

    #[test]
    fn test_scalar_mul_f64() {
        let v = FVector::from_vec(vec![1.5, -2.0, 0.0]);
        let prod = v.clone() * 2.0;
        assert_eq!(prod.as_slice(), &[3.0, -4.0, 0.0]);
    }

    #[test]
    fn test_scalar_mul_complex() {
        use num::Complex;
        let v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(-3.0, 4.0)]);
        let scalar = Complex::new(2.0, 0.0);
        let prod = v.clone() * scalar;
        assert_eq!(prod.as_slice(), &[Complex::new(2.0, 4.0), Complex::new(-6.0, 8.0)]);
    }

    #[test]
    fn test_scalar_mul_assign() {
        let mut v = FVector::from_vec(vec![2, -3, 4]);
        v *= 3;
        assert_eq!(v.as_slice(), &[6, -9, 12]);
    }

    #[test]
    fn test_scalar_mul_assign_f64() {
        let mut v = FVector::from_vec(vec![1.5, -2.0, 0.0]);
        v *= 2.0;
        assert_eq!(v.as_slice(), &[3.0, -4.0, 0.0]);
    }

    #[test]
    fn test_scalar_mul_assign_complex() {
        use num::Complex;
        let mut v = FVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(-3.0, 4.0)]);
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
            FVector::from_vec(vec![Complex::new(2.0, 4.0), Complex::new(-6.0, 8.0)]);
        let divisor = Complex::new(2.0, 0.0);
        v /= divisor;
        assert_eq!(v.as_slice(), &[Complex::new(1.0, 2.0), Complex::new(-3.0, 4.0)]);
    }

    #[test]
    fn test_scalar_div_nan_infinity() {
        let v = FVector::from_vec(vec![f64::NAN, f64::INFINITY, 4.0]);
        let result: FlexVector<f64> = v.clone() / 2.0;
        assert!(result.as_slice()[0].is_nan());
        assert_eq!(result.as_slice()[1], f64::INFINITY);
        assert_eq!(result.as_slice()[2], 2.0);
    }

    #[test]
    fn test_scalar_div_assign_nan_infinity() {
        let mut v: FlexVector<f64> = FVector::from_vec(vec![f64::NAN, f64::INFINITY, 4.0]);
        v /= 2.0;
        assert!(v.as_slice()[0].is_nan());
        assert_eq!(v.as_slice()[1], f64::INFINITY);
        assert_eq!(v.as_slice()[2], 2.0);
    }

    // additional math operator overload edge cases

    #[test]
    fn test_add_overflow_i32() {
        let v1 = FVector::from_vec(vec![i32::MAX]);
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
        let v1 = FVector::from_vec(vec![i32::MAX]);
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
        let v: FlexVector<f64> = FVector::from_vec(vec![1.0, -2.0, 0.0]);
        let result = v.clone() / 0.0;
        assert_eq!(result.as_slice()[0], f64::INFINITY);
        assert_eq!(result.as_slice()[1], f64::NEG_INFINITY);
        assert!(result.as_slice()[2].is_nan());
    }

    #[test]
    fn test_neg_zero_f64() {
        let v = FVector::from_vec(vec![0.0, -0.0]);
        let neg_v = -v;
        assert_eq!(neg_v.as_slice()[0], -0.0);
        assert_eq!(neg_v.as_slice()[1], 0.0);
    }

    #[test]
    fn test_complex_div_by_zero() {
        use num::Complex;
        let v: FlexVector<Complex<f64>> = FVector::from_vec(vec![Complex::new(1.0, 1.0)]);
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
}
