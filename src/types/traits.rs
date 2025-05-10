//! Traits.

use num::Complex;

pub trait VectorBase<T> {
    // --- Core accessors ---
    /// ...
    fn as_slice(&self) -> &[T];

    /// ...
    fn as_mut_slice(&mut self) -> &mut [T];

    /// ...
    fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// ...
    fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    // --- Element access ---
    /// ...
    fn get(&self, index: usize) -> Option<&T> {
        self.as_slice().get(index)
    }

    /// ...
    fn first(&self) -> Option<&T> {
        self.as_slice().first()
    }

    /// ...
    fn last(&self) -> Option<&T> {
        self.as_slice().last()
    }

    // --- Iteration ---
    /// ...
    fn iter(&self) -> std::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// ...
    fn iter_rev(&self) -> std::iter::Rev<std::slice::Iter<'_, T>> {
        self.as_slice().iter().rev()
    }

    /// ...
    fn enumerate(&self) -> std::iter::Enumerate<std::slice::Iter<'_, T>> {
        self.iter().enumerate()
    }

    // --- Conversion ---
    /// ...
    fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.as_slice().to_vec()
    }

    /// ...
    fn pretty(&self) -> String
    where
        T: std::fmt::Debug,
    {
        format!("{:#?}", self.as_slice())
    }

    // --- Search/containment ---
    /// ...
    fn contains(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        self.as_slice().contains(x)
    }

    /// ...
    fn starts_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        self.as_slice().starts_with(needle)
    }

    /// ...
    fn ends_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        self.as_slice().ends_with(needle)
    }

    /// ...
    fn position<P>(&self, predicate: P) -> Option<usize>
    where
        P: FnMut(&T) -> bool,
    {
        self.as_slice().iter().position(predicate)
    }

    /// ...
    fn rposition<P>(&self, predicate: P) -> Option<usize>
    where
        P: FnMut(&T) -> bool,
    {
        self.as_slice().iter().rposition(predicate)
    }

    // --- Slicing/chunking ---
    /// ...
    fn windows(&self, size: usize) -> std::slice::Windows<'_, T> {
        self.as_slice().windows(size)
    }

    /// ...
    fn chunks(&self, size: usize) -> std::slice::Chunks<'_, T> {
        self.as_slice().chunks(size)
    }

    /// ...
    fn split_at(&self, mid: usize) -> (&[T], &[T]) {
        self.as_slice().split_at(mid)
    }

    /// ...
    fn split<F>(&self, pred: F) -> std::slice::Split<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        self.as_slice().split(pred)
    }

    /// ...
    fn splitn<F>(&self, n: usize, pred: F) -> std::slice::SplitN<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        self.as_slice().splitn(n, pred)
    }

    /// ...
    fn rsplit<F>(&self, pred: F) -> std::slice::RSplit<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        self.as_slice().rsplit(pred)
    }

    /// ...
    fn rsplitn<F>(&self, n: usize, pred: F) -> std::slice::RSplitN<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        self.as_slice().rsplitn(n, pred)
    }

    // --- Pointer access ---
    /// ...
    fn as_ptr(&self) -> *const T {
        self.as_slice().as_ptr()
    }
}

pub trait VectorOps<T>: VectorBase<T> {
    type Output;

    /// ...
    fn translate(&self, other: &Self) -> Self::Output
    where
        T: num::Num + Copy;

    /// ...
    fn mut_translate(&mut self, other: &Self)
    where
        T: num::Num + Copy,
    {
        for (a, b) in self.as_mut_slice().iter_mut().zip(other.as_slice()) {
            *a = *a + *b;
        }
    }

    /// ...
    fn dot(&self, other: &Self) -> T
    where
        T: num::Num + Copy + std::iter::Sum<T>,
    {
        self.as_slice().iter().zip(other.as_slice()).map(|(a, b)| *a * *b).sum()
    }

    /// ...
    fn sum(&self) -> T
    where
        T: num::Num + Copy + std::iter::Sum<T>,
    {
        self.as_slice().iter().copied().sum()
    }

    /// ...
    fn product(&self) -> T
    where
        T: num::Num + Copy + std::iter::Product<T>,
    {
        self.as_slice().iter().copied().product()
    }

    /// ...
    fn min(&self) -> Option<T>
    where
        T: Ord + Copy,
    {
        self.as_slice().iter().copied().min()
    }

    /// ...
    fn max(&self) -> Option<T>
    where
        T: Ord + Copy,
    {
        self.as_slice().iter().copied().max()
    }
}

pub trait VectorOpsFloat<T>: VectorBase<T> {
    type Output;

    /// ...
    fn normalize(&self) -> Self::Output
    where
        T: num::Float + Clone + std::iter::Sum<T>,
        Self::Output: std::iter::FromIterator<T>;

    /// ...
    fn mut_normalize(&mut self)
    where
        T: num::Float + Copy + std::iter::Sum<T>;

    /// Linear interpolation between self and end by weight in [0, 1].
    fn lerp(&self, end: &Self, weight: T) -> Self::Output
    where
        T: num::Float + Clone + PartialOrd;

    /// Midpoint
    fn midpoint(&self, end: &Self) -> Self::Output
    where
        T: num::Float + Clone,
    {
        self.lerp(end, num::cast(0.5).unwrap())
    }

    /// Euclidean distance between self and other.
    fn distance(&self, other: &Self) -> T
    where
        T: num::Float + Clone + std::iter::Sum<T>,
    {
        self.as_slice()
            .iter()
            .zip(other.as_slice())
            .map(|(a, b)| (*a - *b).powi(2))
            .sum::<T>()
            .sqrt()
    }

    /// Euclidean norm (magnitude) of the vector.
    fn norm(&self) -> T
    where
        T: num::Float + Clone + std::iter::Sum<T>,
    {
        self.as_slice().iter().map(|a| (*a).powi(2)).sum::<T>().sqrt()
    }

    /// Alias for norm (magnitude).
    fn magnitude(&self) -> T
    where
        T: num::Float + Clone + std::iter::Sum<T>,
    {
        self.norm()
    }
}

pub trait VectorOpsComplexFloat<F>: VectorBase<Complex<F>> {
    type Output;

    /// ...
    fn normalize(&self) -> Self::Output
    where
        F: num::Float + Clone + std::iter::Sum<F>,
        Self::Output: std::iter::FromIterator<Complex<F>>;

    /// ...
    fn mut_normalize(&mut self)
    where
        F: num::Float + Copy + std::iter::Sum<F>;

    /// Linear interpolation between self and end by real weight in [0, 1].
    fn lerp(&self, end: &Self, weight: F) -> Self::Output
    where
        F: num::Float + Clone + PartialOrd;

    /// Midpoint
    fn midpoint(&self, end: &Self) -> Self::Output
    where
        F: num::Float + Clone,
    {
        self.lerp(end, num::cast(0.5).unwrap())
    }

    /// Euclidean distance (L2 norm) between self and other (returns real).
    fn distance(&self, other: &Self) -> F
    where
        F: num::Float + Clone + std::iter::Sum<F>,
    {
        self.as_slice()
            .iter()
            .zip(other.as_slice())
            .map(|(a, b)| {
                let diff = *a - *b;
                diff.norm_sqr()
            })
            .sum::<F>()
            .sqrt()
    }

    /// Euclidean norm (magnitude) of the vector (returns real).
    fn norm(&self) -> F
    where
        F: num::Float + Clone + std::iter::Sum<F>,
    {
        self.as_slice().iter().map(|a| a.norm_sqr()).sum::<F>().sqrt()
    }

    /// Alias for norm (magnitude).
    fn magnitude(&self) -> F
    where
        F: num::Float + Clone + std::iter::Sum<F>,
    {
        self.norm()
    }
}
