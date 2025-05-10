//! Traits.

use num::Complex;

use crate::errors::VectorError;

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

pub trait VectorOps<T>: VectorBase<T> {
    type Output;

    /// ...
    fn translate(&self, other: &Self) -> Self::Output
    where
        T: num::Num + Copy;

    /// ...
    #[inline]
    fn mut_translate(&mut self, other: &Self)
    where
        T: num::Num + Copy,
    {
        for (a, b) in self.as_mut_slice().iter_mut().zip(other.as_slice()) {
            *a = *a + *b;
        }
    }

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
    #[inline]
    fn dot(&self, other: &Self) -> T
    where
        T: num::Num + Copy + std::iter::Sum<T>,
    {
        self.as_slice().iter().zip(other.as_slice()).map(|(a, b)| *a * *b).sum()
    }

    /// Dot product as f64 (for integer and float types).
    #[inline]
    fn dot_to_f64(&self, other: &Self) -> f64
    where
        T: num::ToPrimitive,
    {
        self.as_slice()
            .iter()
            .zip(other.as_slice())
            .map(|(a, b)| a.to_f64().unwrap() * b.to_f64().unwrap())
            .sum()
    }

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
    fn min(&self) -> Option<T>
    where
        T: Ord + Copy,
    {
        self.as_slice().iter().copied().min()
    }

    /// ...
    #[inline]
    fn max(&self) -> Option<T>
    where
        T: Ord + Copy,
    {
        self.as_slice().iter().copied().max()
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

pub trait VectorOpsFloat<T>: VectorBase<T> {
    type Output;

    /// ...
    fn normalize(&self) -> Result<Self::Output, VectorError>
    where
        T: num::Float + Clone + std::iter::Sum<T>,
        Self::Output: std::iter::FromIterator<T>;

    /// ...
    #[inline]
    fn mut_normalize(&mut self) -> Result<(), VectorError>
    where
        T: num::Float + Copy + std::iter::Sum<T>,
    {
        let n = self.norm();
        if n == T::zero() {
            return Err(VectorError::ZeroVectorError("Cannot normalize a zero vector".to_string()));
        }
        for a in self.as_mut_slice().iter_mut() {
            *a = *a / n;
        }
        Ok(())
    }

    /// Returns a new vector with the same direction and the given magnitude.
    fn normalize_to(&self, magnitude: T) -> Result<Self::Output, VectorError>
    where
        T: num::Float + Clone + std::iter::Sum<T>,
        Self::Output: std::iter::FromIterator<T>;

    /// ...
    #[inline]
    fn mut_normalize_to(&mut self, magnitude: T) -> Result<(), VectorError>
    where
        T: num::Float + Copy + std::iter::Sum<T>,
    {
        let n = self.norm();
        if n == T::zero() {
            return Err(VectorError::ZeroVectorError("Cannot normalize a zero vector".to_string()));
        }
        let scale = magnitude / n;
        for a in self.as_mut_slice().iter_mut() {
            *a = *a * scale;
        }
        Ok(())
    }

    /// Linear interpolation between self and end by weight in [0, 1].
    fn lerp(&self, end: &Self, weight: T) -> Result<Self::Output, VectorError>
    where
        T: num::Float + Clone + PartialOrd;

    /// In-place linear interpolation between self and end by weight in [0, 1].
    #[inline]
    fn mut_lerp(&mut self, end: &Self, weight: T) -> Result<(), VectorError>
    where
        T: num::Float + Copy + PartialOrd,
    {
        if weight < T::zero() || weight > T::one() {
            return Err(VectorError::OutOfRangeError("weight must be in [0, 1]".to_string()));
        }
        let w = weight;
        let one_minus_w = T::one() - w;
        for (a, b) in self.as_mut_slice().iter_mut().zip(end.as_slice()) {
            *a = one_minus_w * *a + w * *b;
        }
        Ok(())
    }

    /// Midpoint
    #[inline]
    fn midpoint(&self, end: &Self) -> Self::Output
    where
        T: num::Float + Clone,
    {
        // OK to unwrap because by definition it uses an in-range weight
        self.lerp(end, num::cast(0.5).unwrap()).unwrap()
    }

    /// Euclidean distance between self and other.
    #[inline]
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

    /// Manhattan (L1) distance between self and other.
    #[inline]
    fn manhattan_distance(&self, other: &Self) -> T
    where
        T: num::Float + Clone + std::iter::Sum<T>,
    {
        self.as_slice().iter().zip(other.as_slice()).map(|(a, b)| (*a - *b).abs()).sum()
    }

    /// Chebyshev (L∞) distance between self and other.
    #[inline]
    fn chebyshev_distance(&self, other: &Self) -> T
    where
        T: num::Float + Clone + PartialOrd,
    {
        self.as_slice()
            .iter()
            .zip(other.as_slice())
            .map(|(a, b)| (*a - *b).abs())
            .fold(T::zero(), |acc, x| acc.max(x))
    }

    /// Minkowski (Lp) distance between self and other.
    #[inline]
    fn minkowski_distance(&self, other: &Self, p: T) -> Result<T, VectorError>
    where
        T: num::Float + Clone + std::iter::Sum<T>,
    {
        if p < T::one() {
            return Err(VectorError::OutOfRangeError("p must be >= 1".to_string()));
        }
        Ok(self
            .as_slice()
            .iter()
            .zip(other.as_slice())
            .map(|(a, b)| (*a - *b).abs().powf(p))
            .sum::<T>()
            .powf(T::one() / p))
    }

    /// Euclidean norm (magnitude) of the vector.
    #[inline]
    fn norm(&self) -> T
    where
        T: num::Float + Clone + std::iter::Sum<T>,
    {
        self.as_slice().iter().map(|a| (*a).powi(2)).sum::<T>().sqrt()
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

    /// Alias for norm (magnitude).
    #[inline]
    fn magnitude(&self) -> T
    where
        T: num::Float + Clone + std::iter::Sum<T>,
    {
        self.norm()
    }

    fn angle_with(&self, other: &Self) -> Result<T, VectorError>
    where
        T: num::Float + Clone + std::iter::Sum<T>;
}

pub trait VectorOpsComplex<N>: VectorBase<Complex<N>> {
    type Output;

    /// ...
    fn normalize(&self) -> Result<Self::Output, VectorError>
    where
        N: num::Float + Clone + std::iter::Sum<N>,
        Self::Output: std::iter::FromIterator<Complex<N>>;

    /// ...
    #[inline]
    fn mut_normalize(&mut self) -> Result<(), VectorError>
    where
        N: num::Float + Copy + std::iter::Sum<N>,
    {
        let n = self.norm();
        if n == N::zero() {
            return Err(VectorError::ZeroVectorError("Cannot normalize a zero vector".to_string()));
        }
        for a in self.as_mut_slice().iter_mut() {
            *a = *a / n;
        }
        Ok(())
    }

    /// Returns a new vector with the same direction and the given magnitude (real).
    fn normalize_to(&self, magnitude: N) -> Result<Self::Output, VectorError>
    where
        N: num::Float + Clone + std::iter::Sum<N>,
        Self::Output: std::iter::FromIterator<Complex<N>>;

    /// Scales the complex vector in place to the given (real) magnitude.
    #[inline]
    fn mut_normalize_to(&mut self, magnitude: N) -> Result<(), VectorError>
    where
        N: num::Float + Copy + std::iter::Sum<N>,
    {
        let n = self.norm();
        if n == N::zero() {
            return Err(VectorError::ZeroVectorError("Cannot normalize a zero vector".to_string()));
        }
        let scale = magnitude / n;
        for a in self.as_mut_slice().iter_mut() {
            *a = *a * scale;
        }
        Ok(())
    }

    /// Hermitian dot product: for all complex types
    #[inline]
    fn dot(&self, other: &Self) -> Complex<N>
    where
        N: num::Num + Copy + std::iter::Sum<N> + std::ops::Neg<Output = N>,
    {
        self.as_slice().iter().zip(other.as_slice()).map(|(a, b)| *a * b.conj()).sum()
    }

    /// Hermitian dot product as f64 (sums real part): for all complex types.
    #[inline]
    fn dot_to_f64(&self, other: &Self) -> f64
    where
        N: num::Num + num::ToPrimitive + Copy + std::ops::Neg<Output = N>,
    {
        self.as_slice()
            .iter()
            .zip(other.as_slice())
            .map(|(a, b)| {
                let prod = *a * b.conj();
                prod.re.to_f64().unwrap()
            })
            .sum()
    }

    /// Linear interpolation between self and end by real weight in [0, 1].
    fn lerp(&self, end: &Self, weight: N) -> Result<Self::Output, VectorError>
    where
        N: num::Float + Clone + PartialOrd;

    /// In-place linear interpolation between self and end by real weight in [0, 1].
    #[inline]
    fn mut_lerp(&mut self, end: &Self, weight: N) -> Result<(), VectorError>
    where
        N: num::Float + Copy + PartialOrd,
    {
        if weight < N::zero() || weight > N::one() {
            return Err(VectorError::OutOfRangeError("weight must be in [0, 1]".to_string()));
        }
        let w = Complex::new(weight, N::zero());
        let one_minus_w = Complex::new(N::one() - weight, N::zero());
        for (a, b) in self.as_mut_slice().iter_mut().zip(end.as_slice()) {
            *a = one_minus_w * *a + w * *b;
        }
        Ok(())
    }

    /// Midpoint
    #[inline]
    fn midpoint(&self, end: &Self) -> Self::Output
    where
        N: num::Float + Clone,
    {
        // OK to unwrap because by definition it uses an in-range weight
        self.lerp(end, num::cast(0.5).unwrap()).unwrap()
    }

    /// Euclidean distance (L2 norm) between self and other (returns real).
    #[inline]
    fn distance(&self, other: &Self) -> N
    where
        N: num::Float + Clone + std::iter::Sum<N>,
    {
        self.as_slice()
            .iter()
            .zip(other.as_slice())
            .map(|(a, b)| {
                let diff = *a - *b;
                diff.norm_sqr()
            })
            .sum::<N>()
            .sqrt()
    }

    /// Manhattan (L1) distance between self and other (sum of magnitudes of differences).
    #[inline]
    fn manhattan_distance(&self, other: &Self) -> N
    where
        N: num::Float + Clone + std::iter::Sum<N>,
    {
        self.as_slice().iter().zip(other.as_slice()).map(|(a, b)| (*a - *b).norm()).sum()
    }

    /// Chebyshev (L∞) distance between self and other (maximum magnitude of differences).
    #[inline]
    fn chebyshev_distance(&self, other: &Self) -> N
    where
        N: num::Float + Clone + PartialOrd,
    {
        self.as_slice()
            .iter()
            .zip(other.as_slice())
            .map(|(a, b)| (*a - *b).norm())
            .fold(N::zero(), |acc, x| acc.max(x))
    }

    /// Minkowski (Lp) distance between self and other.
    #[inline]
    fn minkowski_distance(&self, other: &Self, p: N) -> Result<N, VectorError>
    where
        N: num::Float + Clone + std::iter::Sum<N>,
    {
        if p < N::one() {
            return Err(VectorError::OutOfRangeError("p must be >= 1".to_string()));
        }
        Ok(self
            .as_slice()
            .iter()
            .zip(other.as_slice())
            .map(|(a, b)| (*a - *b).norm().powf(p))
            .sum::<N>()
            .powf(N::one() / p))
    }

    /// Euclidean norm (magnitude) of the vector (returns real).
    #[inline]
    fn norm(&self) -> N
    where
        N: num::Float + Clone + std::iter::Sum<N>,
    {
        self.as_slice().iter().map(|a| a.norm_sqr()).sum::<N>().sqrt()
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

    /// Alias for norm (magnitude).
    #[inline]
    fn magnitude(&self) -> N
    where
        N: num::Float + Clone + std::iter::Sum<N>,
    {
        self.norm()
    }
}
