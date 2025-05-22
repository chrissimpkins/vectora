//! Macros.

/// Creates a [`Vector`] with the given elements or repeated value.
///
/// - `vector![x, y, z]` creates a Vector from the elements.
/// - `vector![elem; n]` creates a Vector of length `n` with all elements set to `elem`.
///
/// # Examples
///
/// ```
/// use vectora::vector;
/// use num::Complex;
///
/// let v = vector![1, 2, 3];
/// let v_f64 = vector![1.0, 2.0, 3.0];
/// let v_repeat = vector![0; 4];
/// let v_complex = vector![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
/// ```
#[macro_export]
macro_rules! vector {
    ($elem:expr; $n:expr) => (
        $crate::types::vector::Vector::from([$elem; $n])
    );
    ($($x:expr),+ $(,)?) => (
        $crate::types::vector::Vector::from([$($x),+])
    );
}

/// Converts a collection (like a `Vec` or slice) into a [`Vector`] of matching length, returning a `Result`.
///
/// - `try_vector!(data)` tries to create a Vector from a runtime collection, returning an error if the length does not match.
///
/// # Examples
///
/// ```
/// use vectora::prelude::*;
/// use num::Complex;
///
/// let v: Vector<i32, 3> = try_vector!(vec![1, 2, 3]).unwrap();
/// let v_f64: Vector<f64, 3> = try_vector!([1.0, 2.0, 3.0]).unwrap();
/// let v_complex: Vector<Complex<f64>, 2> = try_vector!(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]).unwrap();
/// ```
///
/// /// Error example:
///
/// ```
/// use vectora::prelude::*;
/// // This will return an Err because the length does not match the Vector's size.
/// let result: Result<Vector<i32, 3>, _> = try_vector!(vec![1, 2]); // Suppose you want Vector<i32, 3>
/// assert!(result.is_err());
#[macro_export]
macro_rules! try_vector {
    ($elem:expr) => {
        $crate::types::vector::Vector::try_from($elem)
    };
}

/// Creates a [`FlexVector`] with the list of given elements or repeated value, optionally specifying
/// vector row or column orientation.
///
/// - `fv![x, y, z]` creates a default (column) FlexVector.
/// - `fv![Column; x, y, z]` creates a Column FlexVector.
/// - `fv![Row; x, y, z]` creates a Row FlexVector.
/// - `fv![elem; n]` creates a default (column) FlexVector of length `n` with all elements set to `elem`.
/// - `fv![Column; elem; n]` creates a Column FlexVector of length `n` with all elements set to `elem`.
/// - `fv![Row; elem; n]` creates a Row FlexVector of length `n` with all elements set to `elem`.
///
/// # Examples
/// ```
/// use vectora::prelude::*;
///
/// let fv_default: FlexVector<i32, Column> = fv![1, 2, 3];
/// let fv_col = fv![Column; 1, 2, 3];
/// let fv_row = fv![Row; 1, 2, 3];
/// let fv_default_repeat: FlexVector<i32, Column> = fv![0; 4];
/// let fv_col_repeat = fv![Column; 0; 4];
/// let fv_row_repeat = fv![Row; 0; 4];
/// ```
#[macro_export]
macro_rules! fv {
    // Row vector, repeated element: fv![Row; elem; n]
    (Row; $elem:expr; $n:expr) => {
        $crate::types::flexvector::FlexVector::<_, $crate::types::orientation::Row>::from_vec(vec![$elem; $n])
    };
    // Column vector, repeated element: fv![Column; elem; n]
    (Column; $elem:expr; $n:expr) => {
        $crate::types::flexvector::FlexVector::<_, $crate::types::orientation::Column>::from_vec(vec![$elem; $n])
    };
    // Default (column) vector, repeated element: fv![elem; n]
    ($elem:expr; $n:expr) => {
        $crate::types::flexvector::FlexVector::from_vec(vec![$elem; $n])
    };
    // Row vector, list: fv![Row; x, y, z, ...]
    (Row; $($x:expr),+ $(,)?) => {
        $crate::types::flexvector::FlexVector::<_, $crate::types::orientation::Row>::from_vec(vec![$($x),+])
    };
    // Column vector, list: fv![Column; x, y, z, ...]
    (Column; $($x:expr),+ $(,)?) => {
        $crate::types::flexvector::FlexVector::<_, $crate::types::orientation::Column>::from_vec(vec![$($x),+])
    };
    // Default (column) vector, list: fv![x, y, z, ...]
    ($($x:expr),+ $(,)?) => {
        $crate::types::flexvector::FlexVector::from_vec(vec![$($x),+])
    };
}

/// Creates a [`FlexVector`] from a collection (slice, Vec, or Cow), optionally specifying row or column orientation.
///
/// - `fv_from![data]` creates a default (column) FlexVector from any collection accepted by `from_cow`.
/// - `fv_from![Row; data]` creates a Row FlexVector from a collection.
/// - `fv_from![Column; data]` creates a Column FlexVector from a collection.
///
/// # Examples
/// ```
/// use vectora::prelude::*;
/// use std::borrow::Cow;
///
/// let slice: &[i32] = &[1, 2, 3];
/// let fv: FlexVector<i32, Column> = fv_from![slice];
///
/// let vec = vec![4, 5, 6];
/// let fv_col = fv_from![Column; vec];
///
/// let cow: Cow<[i32]> = Cow::Borrowed(&[7, 8, 9]);
/// let fv_row = fv_from![Row; cow];
/// ```
#[macro_export]
macro_rules! fv_from {
    (Row; $data:expr) => {
        $crate::types::flexvector::FlexVector::<_, $crate::types::orientation::Row>::from_cow($data)
    };
    (Column; $data:expr) => {
        $crate::types::flexvector::FlexVector::<_, $crate::types::orientation::Column>::from_cow(
            $data,
        )
    };
    ($data:expr) => {
        $crate::types::flexvector::FlexVector::from_cow($data)
    };
}

/// Creates a [`FlexVector`] from any iterable (such as an iterator, range, Vec, or array),
/// optionally specifying row or column orientation.
///
/// - `fv_iter![data]` creates a default (column) FlexVector from any iterable.
/// - `fv_iter![Row; data]` creates a Row FlexVector from any iterable.
/// - `fv_iter![Column; data]` creates a Column FlexVector from any iterable.
///
/// # Examples
/// ```
/// use vectora::prelude::*;
///
/// // From a range
/// let fv: FlexVector<i32, Column> = fv_iter![0..3];
/// assert_eq!(fv.as_slice(), &[0, 1, 2]);
///
/// // From a Vec
/// let v = vec![10, 20, 30];
/// let fv_col = fv_iter![Column; v.clone()];
/// assert_eq!(fv_col.as_slice(), &[10, 20, 30]);
///
/// // From an iterator
/// let fv_row = fv_iter![Row; (1..=3).map(|x| x * 2)];
/// assert_eq!(fv_row.as_slice(), &[2, 4, 6]);
/// ```
#[macro_export]
macro_rules! fv_iter {
    (Row; $data:expr) => {
        $crate::types::flexvector::FlexVector::<_, $crate::types::orientation::Row>::from_iter(
            $data,
        )
    };
    (Column; $data:expr) => {
        $crate::types::flexvector::FlexVector::<_, $crate::types::orientation::Column>::from_iter(
            $data,
        )
    };
    ($data:expr) => {
        $crate::types::flexvector::FlexVector::from_iter($data)
    };
}

/// Fallibly constructs a [`FlexVector`] from an iterator of `Result<T, E>`, propagating the first error,
/// optionally specifying vector row or column orientation.
///
/// - `try_fv_iter!(iter)` collects all `Ok(T)` values into a default (column) FlexVector, or returns the first `Err(E)` encountered.
/// - `try_fv_iter!(Column; iter)` collects into a Column FlexVector.
/// - `try_fv_iter!(Row; iter)` collects into a Row FlexVector.
///
/// # Examples
/// ```
/// use vectora::prelude::*;
///
/// let data = vec!["1", "2", "oops"];
/// let fv: Result<FlexVector<_, Column>, _> = try_fv_iter!(data.iter().map(|s| s.parse::<i32>()));
/// assert!(fv.is_err());
///
/// let data = vec!["1", "2", "3"];
/// let fv = try_fv_iter!(Row; data.iter().map(|s| s.parse::<i32>())).unwrap();
/// assert_eq!(fv.as_slice(), &[1, 2, 3]);
/// ```
#[macro_export]
macro_rules! try_fv_iter {
    // Row vector: try_fv!(Row; iter)
    (Row; $iter:expr) => {
        $crate::types::flexvector::FlexVector::<_, $crate::types::orientation::Row>::try_from_iter($iter)
    };
    // Column vector: try_fv!(Column; iter)
    (Column; $iter:expr) => {
        $crate::types::flexvector::FlexVector::<_, $crate::types::orientation::Column>::try_from_iter($iter)
    };
    // Default (column) vector: try_fv!(iter)
    ($iter:expr) => {
        $crate::types::flexvector::FlexVector::try_from_iter($iter)
    };
}

#[macro_export]
macro_rules! impl_vector_unary_op {
    ($VectorType:ident, $trait:ident, $method:ident, $op:tt) => {
        impl<T, O> std::ops::$trait for $VectorType<T, O>
        where
            T: num::Num + Clone + std::ops::Neg<Output = T>,
        {
            type Output = Self;
            #[inline]
            fn $method(self) -> Self {
                let components = self.components.into_iter().map(|a| $op a).collect();
                Self { components, _orientation: PhantomData }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vector_binop {
    // With length check (for FlexVector)
    (check_len, $VectorType:ident, $trait:ident, $method:ident, $op:tt) => {
        impl<T, O> std::ops::$trait for $VectorType<T, O>
        where
            T: num::Num + Clone,
        {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self {
                assert_eq!(self.len(), rhs.len(), "Vector length mismatch");
                let components = self.components.into_iter()
                    .zip(rhs.components)
                    .map(|(a, b)| a $op b)
                    .collect();
                Self { components, _orientation: PhantomData }
            }
        }
    };
    // Without length check (for Vector)
    (no_check_len, $VectorType:ident, $trait:ident, $method:ident, $op:tt) => {
        impl<T, O> std::ops::$trait for $VectorType<T, O>
        where
            T: num::Num + Clone,
        {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self {
                let components = self.components.into_iter()
                    .zip(rhs.components)
                    .map(|(a, b)| a $op b)
                    .collect();
                Self { components, _orientation: PhantomData }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vector_binop_div {
    // With length check (for FlexVector)
    (check_len, $VectorType:ident) => {
        // f32
        impl<O> std::ops::Div for $VectorType<f32, O> {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self {
                assert_eq!(self.len(), rhs.len(), "Vector length mismatch");
                let components =
                    self.components.into_iter().zip(rhs.components).map(|(a, b)| a / b).collect();
                Self { components, _orientation: PhantomData }
            }
        }
        // f64
        impl<O> std::ops::Div for $VectorType<f64, O> {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self {
                assert_eq!(self.len(), rhs.len(), "Vector length mismatch");
                let components =
                    self.components.into_iter().zip(rhs.components).map(|(a, b)| a / b).collect();
                Self { components, _orientation: PhantomData }
            }
        }
        // Complex<f32>
        impl<O> std::ops::Div for $VectorType<num::Complex<f32>, O> {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self {
                assert_eq!(self.len(), rhs.len(), "Vector length mismatch");
                let components =
                    self.components.into_iter().zip(rhs.components).map(|(a, b)| a / b).collect();
                Self { components, _orientation: PhantomData }
            }
        }
        // Complex<f64>
        impl<O> std::ops::Div for $VectorType<num::Complex<f64>, O> {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self {
                assert_eq!(self.len(), rhs.len(), "Vector length mismatch");
                let components =
                    self.components.into_iter().zip(rhs.components).map(|(a, b)| a / b).collect();
                Self { components, _orientation: PhantomData }
            }
        }
    };
    // Without length check (for Vector)
    (no_check_len, $VectorType:ident) => {
        // f32
        impl<O> std::ops::Div for $VectorType<f32, O> {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self {
                let components =
                    self.components.into_iter().zip(rhs.components).map(|(a, b)| a / b).collect();
                Self { components, _orientation: PhantomData }
            }
        }
        // f64
        impl<O> std::ops::Div for $VectorType<f64, O> {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self {
                let components =
                    self.components.into_iter().zip(rhs.components).map(|(a, b)| a / b).collect();
                Self { components, _orientation: PhantomData }
            }
        }
        // Complex<f32>
        impl<O> std::ops::Div for $VectorType<num::Complex<f32>, O> {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self {
                let components =
                    self.components.into_iter().zip(rhs.components).map(|(a, b)| a / b).collect();
                Self { components, _orientation: PhantomData }
            }
        }
        // Complex<f64>
        impl<O> std::ops::Div for $VectorType<num::Complex<f64>, O> {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self {
                let components =
                    self.components.into_iter().zip(rhs.components).map(|(a, b)| a / b).collect();
                Self { components, _orientation: PhantomData }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vector_binop_assign {
    // With length check (for FlexVector)
    (check_len, $VectorType:ident, $trait:ident, $method:ident, $op:tt) => {
        impl<T, O> std::ops::$trait for $VectorType<T, O>
        where
            T: num::Num + Clone,
        {
            #[inline]
            fn $method(&mut self, rhs: Self) {
                assert_eq!(self.len(), rhs.len(), "Vector length mismatch");
                for (a, b) in self.components.iter_mut().zip(rhs.components) {
                    *a = a.clone() $op b;
                }
            }
        }
    };
    // Without length check (for Vector)
    (no_check_len, $VectorType:ident, $trait:ident, $method:ident, $op:tt) => {
        impl<T, O> std::ops::$trait for $VectorType<T, O>
        where
            T: num::Num + Clone,
        {
            #[inline]
            fn $method(&mut self, rhs: Self) {
                for (a, b) in self.components.iter_mut().zip(rhs.components) {
                    *a = a.clone() $op b;
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vector_binop_div_assign {
    // With length check (for FlexVector)
    (check_len, $VectorType:ident) => {
        // f32
        impl<O> std::ops::DivAssign for $VectorType<f32, O> {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                assert_eq!(self.len(), rhs.len(), "Vector length mismatch");
                for (a, b) in self.components.iter_mut().zip(rhs.components) {
                    *a /= b;
                }
            }
        }
        // f64
        impl<O> std::ops::DivAssign for $VectorType<f64, O> {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                assert_eq!(self.len(), rhs.len(), "Vector length mismatch");
                for (a, b) in self.components.iter_mut().zip(rhs.components) {
                    *a /= b;
                }
            }
        }
        // Complex<f32>
        impl<O> std::ops::DivAssign for $VectorType<num::Complex<f32>, O> {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                assert_eq!(self.len(), rhs.len(), "Vector length mismatch");
                for (a, b) in self.components.iter_mut().zip(rhs.components) {
                    *a /= b;
                }
            }
        }
        // Complex<f64>
        impl<O> std::ops::DivAssign for $VectorType<num::Complex<f64>, O> {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                assert_eq!(self.len(), rhs.len(), "Vector length mismatch");
                for (a, b) in self.components.iter_mut().zip(rhs.components) {
                    *a /= b;
                }
            }
        }
    };
    // Without length check (for Vector)
    (no_check_len, $VectorType:ident) => {
        // f32
        impl<O> std::ops::DivAssign for $VectorType<f32, O> {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                for (a, b) in self.components.iter_mut().zip(rhs.components) {
                    *a /= b;
                }
            }
        }
        // f64
        impl<O> std::ops::DivAssign for $VectorType<f64, O> {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                for (a, b) in self.components.iter_mut().zip(rhs.components) {
                    *a /= b;
                }
            }
        }
        // Complex<f32>
        impl<O> std::ops::DivAssign for $VectorType<num::Complex<f32>, O> {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                for (a, b) in self.components.iter_mut().zip(rhs.components) {
                    *a /= b;
                }
            }
        }
        // Complex<f64>
        impl<O> std::ops::DivAssign for $VectorType<num::Complex<f64>, O> {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                for (a, b) in self.components.iter_mut().zip(rhs.components) {
                    *a /= b;
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vector_scalar_op {
    ($VectorType:ident, $trait:ident, $method:ident, $op:tt) => {
        impl<T, O> std::ops::$trait<T> for $VectorType<T, O>
        where
            T: num::Num + Clone,
        {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: T) -> Self {
                let components = self.components.into_iter().map(|a| a $op rhs.clone()).collect();
                Self { components, _orientation: PhantomData }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vector_scalar_op_assign {
    ($VectorType:ident, $trait:ident, $method:ident, $op:tt) => {
        impl<T, O> std::ops::$trait<T> for $VectorType<T, O>
        where
            T: num::Num + Clone,
        {
            #[inline]
            fn $method(&mut self, rhs: T) {
                for a in &mut self.components {
                    *a = a.clone() $op rhs.clone();
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vector_scalar_div_op {
    ($VectorType:ident) => {
        // For f32
        impl<O> std::ops::Div<f32> for $VectorType<f32, O> {
            type Output = Self;
            #[inline]
            fn div(self, rhs: f32) -> Self {
                let components = self.components.into_iter().map(|a| a / rhs).collect();
                Self { components, _orientation: PhantomData }
            }
        }
        // For f64
        impl<O> std::ops::Div<f64> for $VectorType<f64, O> {
            type Output = Self;
            #[inline]
            fn div(self, rhs: f64) -> Self {
                let components = self.components.into_iter().map(|a| a / rhs).collect();
                Self { components, _orientation: PhantomData }
            }
        }
        // For Complex<f32> / f32
        impl<O> std::ops::Div<f32> for $VectorType<num::Complex<f32>, O> {
            type Output = Self;
            #[inline]
            fn div(self, rhs: f32) -> Self {
                let components = self.components.into_iter().map(|a| a / rhs).collect();
                Self { components, _orientation: PhantomData }
            }
        }
        // For Complex<f64> / f64
        impl<O> std::ops::Div<f64> for $VectorType<num::Complex<f64>, O> {
            type Output = Self;
            #[inline]
            fn div(self, rhs: f64) -> Self {
                let components = self.components.into_iter().map(|a| a / rhs).collect();
                Self { components, _orientation: PhantomData }
            }
        }
        // For Complex<f32> / Complex<f32>
        impl<O> std::ops::Div<num::Complex<f32>> for $VectorType<num::Complex<f32>, O> {
            type Output = Self;
            #[inline]
            fn div(self, rhs: num::Complex<f32>) -> Self {
                let components = self.components.into_iter().map(|a| a / rhs).collect();
                Self { components, _orientation: PhantomData }
            }
        }
        // For Complex<f64> / Complex<f64>
        impl<O> std::ops::Div<num::Complex<f64>> for $VectorType<num::Complex<f64>, O> {
            type Output = Self;
            #[inline]
            fn div(self, rhs: num::Complex<f64>) -> Self {
                let components = self.components.into_iter().map(|a| a / rhs).collect();
                Self { components, _orientation: PhantomData }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vector_scalar_div_op_assign {
    ($VectorType:ident) => {
        // For f32
        impl<O> std::ops::DivAssign<f32> for $VectorType<f32, O> {
            #[inline]
            fn div_assign(&mut self, rhs: f32) {
                for a in &mut self.components {
                    *a = *a / rhs;
                }
            }
        }
        // For f64
        impl<O> std::ops::DivAssign<f64> for $VectorType<f64, O> {
            #[inline]
            fn div_assign(&mut self, rhs: f64) {
                for a in &mut self.components {
                    *a = *a / rhs;
                }
            }
        }
        // For Complex<f32> / f32
        impl<O> std::ops::DivAssign<f32> for $VectorType<num::Complex<f32>, O> {
            #[inline]
            fn div_assign(&mut self, rhs: f32) {
                for a in &mut self.components {
                    *a = *a / rhs;
                }
            }
        }
        // For Complex<f64> / f64
        impl<O> std::ops::DivAssign<f64> for $VectorType<num::Complex<f64>, O> {
            #[inline]
            fn div_assign(&mut self, rhs: f64) {
                for a in &mut self.components {
                    *a = *a / rhs;
                }
            }
        }
        // For Complex<f32> / Complex<f32>
        impl<O> std::ops::DivAssign<num::Complex<f32>> for $VectorType<num::Complex<f32>, O> {
            #[inline]
            fn div_assign(&mut self, rhs: num::Complex<f32>) {
                for a in &mut self.components {
                    *a = *a / rhs;
                }
            }
        }
        // For Complex<f64> / Complex<f64>
        impl<O> std::ops::DivAssign<num::Complex<f64>> for $VectorType<num::Complex<f64>, O> {
            #[inline]
            fn div_assign(&mut self, rhs: num::Complex<f64>) {
                for a in &mut self.components {
                    *a = *a / rhs;
                }
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::types::orientation::{Column, Row};
    use crate::types::traits::VectorBase;
    use crate::{FlexVector, Vector};
    #[allow(unused_imports)]
    use approx::{assert_relative_eq, assert_relative_ne};
    #[allow(unused_imports)]
    use num::complex::Complex;
    #[allow(unused_imports)]
    use pretty_assertions::{assert_eq, assert_ne};

    #[test]
    fn macro_vector_usize() {
        let v1 = vector![1 as usize, 2 as usize, 3 as usize];
        let v2: Vector<usize, 3> = vector![1, 2, 3];
        let v3: Vector<usize, 3> = vector![1; 3];

        assert_eq!(v1.len(), 3);
        assert_eq!(v1[0], 1 as usize);
        assert_eq!(v1[1], 2 as usize);
        assert_eq!(v1[2], 3 as usize);

        assert_eq!(v2.len(), 3);
        assert_eq!(v2[0], 1 as usize);
        assert_eq!(v2[1], 2 as usize);
        assert_eq!(v2[2], 3 as usize);

        assert_eq!(v3.len(), 3);
        assert_eq!(v3[0], 1 as usize);
        assert_eq!(v3[1], 1 as usize);
        assert_eq!(v3[2], 1 as usize);
    }

    #[test]
    fn macro_vector_u8() {
        let v1 = vector![1 as u8, 2 as u8, 3 as u8];
        let v2: Vector<u8, 3> = vector![1, 2, 3];
        let v3: Vector<u8, 3> = vector![1; 3];

        assert_eq!(v1.len(), 3);
        assert_eq!(v1[0], 1 as u8);
        assert_eq!(v1[1], 2 as u8);
        assert_eq!(v1[2], 3 as u8);

        assert_eq!(v2.len(), 3);
        assert_eq!(v2[0], 1 as u8);
        assert_eq!(v2[1], 2 as u8);
        assert_eq!(v2[2], 3 as u8);

        assert_eq!(v3.len(), 3);
        assert_eq!(v3[0], 1 as u8);
        assert_eq!(v3[1], 1 as u8);
        assert_eq!(v3[2], 1 as u8);
    }

    #[test]
    fn macro_vector_u16() {
        let v1 = vector![1 as u16, 2 as u16, 3 as u16];
        let v2: Vector<u16, 3> = vector![1, 2, 3];
        let v3: Vector<u16, 3> = vector![1; 3];

        assert_eq!(v1.len(), 3);
        assert_eq!(v1[0], 1 as u16);
        assert_eq!(v1[1], 2 as u16);
        assert_eq!(v1[2], 3 as u16);

        assert_eq!(v2.len(), 3);
        assert_eq!(v2[0], 1 as u16);
        assert_eq!(v2[1], 2 as u16);
        assert_eq!(v2[2], 3 as u16);

        assert_eq!(v3.len(), 3);
        assert_eq!(v3[0], 1 as u16);
        assert_eq!(v3[1], 1 as u16);
        assert_eq!(v3[2], 1 as u16);
    }

    #[test]
    fn macro_vector_u32() {
        let v1 = vector![1 as u32, 2 as u32, 3 as u32];
        let v2: Vector<u32, 3> = vector![1, 2, 3];
        let v3: Vector<u32, 3> = vector![1; 3];

        assert_eq!(v1.len(), 3);
        assert_eq!(v1[0], 1 as u32);
        assert_eq!(v1[1], 2 as u32);
        assert_eq!(v1[2], 3 as u32);

        assert_eq!(v2.len(), 3);
        assert_eq!(v2[0], 1 as u32);
        assert_eq!(v2[1], 2 as u32);
        assert_eq!(v2[2], 3 as u32);

        assert_eq!(v3.len(), 3);
        assert_eq!(v3[0], 1 as u32);
        assert_eq!(v3[1], 1 as u32);
        assert_eq!(v3[2], 1 as u32);
    }

    #[test]
    fn macro_vector_u64() {
        let v1 = vector![1 as u64, 2 as u64, 3 as u64];
        let v2: Vector<u64, 3> = vector![1, 2, 3];
        let v3: Vector<u64, 3> = vector![1; 3];

        assert_eq!(v1.len(), 3);
        assert_eq!(v1[0], 1 as u64);
        assert_eq!(v1[1], 2 as u64);
        assert_eq!(v1[2], 3 as u64);

        assert_eq!(v2.len(), 3);
        assert_eq!(v2[0], 1 as u64);
        assert_eq!(v2[1], 2 as u64);
        assert_eq!(v2[2], 3 as u64);

        assert_eq!(v3.len(), 3);
        assert_eq!(v3[0], 1 as u64);
        assert_eq!(v3[1], 1 as u64);
        assert_eq!(v3[2], 1 as u64);
    }

    #[test]
    fn macro_vector_u128() {
        let v1 = vector![1 as u128, 2 as u128, 3 as u128];
        let v2: Vector<u128, 3> = vector![1, 2, 3];
        let v3: Vector<u128, 3> = vector![1; 3];

        assert_eq!(v1.len(), 3);
        assert_eq!(v1[0], 1 as u128);
        assert_eq!(v1[1], 2 as u128);
        assert_eq!(v1[2], 3 as u128);

        assert_eq!(v2.len(), 3);
        assert_eq!(v2[0], 1 as u128);
        assert_eq!(v2[1], 2 as u128);
        assert_eq!(v2[2], 3 as u128);

        assert_eq!(v3.len(), 3);
        assert_eq!(v3[0], 1 as u128);
        assert_eq!(v3[1], 1 as u128);
        assert_eq!(v3[2], 1 as u128);
    }

    #[test]
    fn macro_vector_isize() {
        let v1 = vector![1 as isize, 2 as isize, 3 as isize];
        let v2: Vector<isize, 3> = vector![1, 2, 3];
        let v3: Vector<isize, 3> = vector![1; 3];

        assert_eq!(v1.len(), 3);
        assert_eq!(v1[0], 1 as isize);
        assert_eq!(v1[1], 2 as isize);
        assert_eq!(v1[2], 3 as isize);

        assert_eq!(v2.len(), 3);
        assert_eq!(v2[0], 1 as isize);
        assert_eq!(v2[1], 2 as isize);
        assert_eq!(v2[2], 3 as isize);

        assert_eq!(v3.len(), 3);
        assert_eq!(v3[0], 1 as isize);
        assert_eq!(v3[1], 1 as isize);
        assert_eq!(v3[2], 1 as isize);
    }

    #[test]
    fn macro_vector_i8() {
        let v1 = vector![1 as i8, 2 as i8, 3 as i8];
        let v2: Vector<i8, 3> = vector![1, 2, 3];
        let v3: Vector<i8, 3> = vector![1; 3];

        assert_eq!(v1.len(), 3);
        assert_eq!(v1[0], 1 as i8);
        assert_eq!(v1[1], 2 as i8);
        assert_eq!(v1[2], 3 as i8);

        assert_eq!(v2.len(), 3);
        assert_eq!(v2[0], 1 as i8);
        assert_eq!(v2[1], 2 as i8);
        assert_eq!(v2[2], 3 as i8);

        assert_eq!(v3.len(), 3);
        assert_eq!(v3[0], 1 as i8);
        assert_eq!(v3[1], 1 as i8);
        assert_eq!(v3[2], 1 as i8);
    }

    #[test]
    fn macro_vector_i16() {
        let v1 = vector![1 as i16, 2 as i16, 3 as i16];
        let v2: Vector<i16, 3> = vector![1, 2, 3];
        let v3: Vector<i16, 3> = vector![1; 3];

        assert_eq!(v1.len(), 3);
        assert_eq!(v1[0], 1 as i16);
        assert_eq!(v1[1], 2 as i16);
        assert_eq!(v1[2], 3 as i16);

        assert_eq!(v2.len(), 3);
        assert_eq!(v2[0], 1 as i16);
        assert_eq!(v2[1], 2 as i16);
        assert_eq!(v2[2], 3 as i16);

        assert_eq!(v3.len(), 3);
        assert_eq!(v3[0], 1 as i16);
        assert_eq!(v3[1], 1 as i16);
        assert_eq!(v3[2], 1 as i16);
    }

    #[test]
    fn macro_vector_i32() {
        let v1 = vector![1 as i32, 2 as i32, 3 as i32];
        let v2: Vector<i32, 3> = vector![1, 2, 3];
        let v3: Vector<i32, 3> = vector![1; 3];

        assert_eq!(v1.len(), 3);
        assert_eq!(v1[0], 1 as i32);
        assert_eq!(v1[1], 2 as i32);
        assert_eq!(v1[2], 3 as i32);

        assert_eq!(v2.len(), 3);
        assert_eq!(v2[0], 1 as i32);
        assert_eq!(v2[1], 2 as i32);
        assert_eq!(v2[2], 3 as i32);

        assert_eq!(v3.len(), 3);
        assert_eq!(v3[0], 1 as i32);
        assert_eq!(v3[1], 1 as i32);
        assert_eq!(v3[2], 1 as i32);
    }

    #[test]
    fn macro_vector_i64() {
        let v1 = vector![1 as i64, 2 as i64, 3 as i64];
        let v2: Vector<i64, 3> = vector![1, 2, 3];
        let v3: Vector<i64, 3> = vector![1; 3];

        assert_eq!(v1.len(), 3);
        assert_eq!(v1[0], 1 as i64);
        assert_eq!(v1[1], 2 as i64);
        assert_eq!(v1[2], 3 as i64);

        assert_eq!(v2.len(), 3);
        assert_eq!(v2[0], 1 as i64);
        assert_eq!(v2[1], 2 as i64);
        assert_eq!(v2[2], 3 as i64);

        assert_eq!(v3.len(), 3);
        assert_eq!(v3[0], 1 as i64);
        assert_eq!(v3[1], 1 as i64);
        assert_eq!(v3[2], 1 as i64);
    }

    #[test]
    fn macro_vector_i128() {
        let v1 = vector![1 as i128, 2 as i128, 3 as i128];
        let v2: Vector<i128, 3> = vector![1, 2, 3];
        let v3: Vector<i128, 3> = vector![1; 3];

        assert_eq!(v1.len(), 3);
        assert_eq!(v1[0], 1 as i128);
        assert_eq!(v1[1], 2 as i128);
        assert_eq!(v1[2], 3 as i128);

        assert_eq!(v2.len(), 3);
        assert_eq!(v2[0], 1 as i128);
        assert_eq!(v2[1], 2 as i128);
        assert_eq!(v2[2], 3 as i128);

        assert_eq!(v3.len(), 3);
        assert_eq!(v3[0], 1 as i128);
        assert_eq!(v3[1], 1 as i128);
        assert_eq!(v3[2], 1 as i128);
    }

    #[test]
    fn macro_vector_f32() {
        let v1 = vector![1. as f32, 2. as f32, 3. as f32];
        let v2: Vector<f32, 3> = vector![1., 2., 3.];
        let v3: Vector<f32, 3> = vector![1.; 3];

        assert_eq!(v1.len(), 3);
        assert_relative_eq!(v1[0], 1.0 as f32);
        assert_relative_eq!(v1[1], 2.0 as f32);
        assert_relative_eq!(v1[2], 3.0 as f32);

        assert_eq!(v2.len(), 3);
        assert_relative_eq!(v2[0], 1.0 as f32);
        assert_relative_eq!(v2[1], 2.0 as f32);
        assert_relative_eq!(v2[2], 3.0 as f32);

        assert_eq!(v3.len(), 3);
        assert_relative_eq!(v3[0], 1.0 as f32);
        assert_relative_eq!(v3[1], 1.0 as f32);
        assert_relative_eq!(v3[2], 1.0 as f32);
    }

    #[test]
    fn macro_vector_f64() {
        let v1 = vector![1. as f64, 2. as f64, 3. as f64];
        let v2: Vector<f64, 3> = vector![1., 2., 3.];
        let v3: Vector<f64, 3> = vector![1.; 3];

        assert_eq!(v1.len(), 3);
        assert_relative_eq!(v1[0], 1.0 as f64);
        assert_relative_eq!(v1[1], 2.0 as f64);
        assert_relative_eq!(v1[2], 3.0 as f64);

        assert_eq!(v2.len(), 3);
        assert_relative_eq!(v2[0], 1.0 as f64);
        assert_relative_eq!(v2[1], 2.0 as f64);
        assert_relative_eq!(v2[2], 3.0 as f64);

        assert_eq!(v3.len(), 3);
        assert_relative_eq!(v3[0], 1.0 as f64);
        assert_relative_eq!(v3[1], 1.0 as f64);
        assert_relative_eq!(v3[2], 1.0 as f64);
    }

    #[test]
    fn macro_vector_complex_i32() {
        let v1 = vector![Complex::new(1 as i32, -2 as i32), Complex::new(-1 as i32, 2 as i32)];
        let v2: Vector<Complex<i32>, 2> = vector![Complex::new(1, -2), Complex::new(-1, 2)];
        let v3: Vector<Complex<i32>, 2> = vector![Complex::new(1, -2); 2];

        assert_eq!(v1.len(), 2);
        assert_eq!(v1[0].re, 1 as i32);
        assert_eq!(v1[0].im, -2 as i32);
        assert_eq!(v1[1].re, -1 as i32);
        assert_eq!(v1[1].im, 2 as i32);

        assert_eq!(v2.len(), 2);
        assert_eq!(v2[0].re, 1 as i32);
        assert_eq!(v2[0].im, -2 as i32);
        assert_eq!(v2[1].re, -1 as i32);
        assert_eq!(v2[1].im, 2 as i32);

        assert_eq!(v3.len(), 2);
        assert_eq!(v3[0].re, 1 as i32);
        assert_eq!(v3[0].im, -2 as i32);
        assert_eq!(v3[1].re, 1 as i32);
        assert_eq!(v3[1].im, -2 as i32);
    }

    #[test]
    fn macro_vector_complex_f64() {
        let v1 = vector![Complex::new(1. as f64, -2. as f64), Complex::new(-1. as f64, 2. as f64)];
        let v2: Vector<Complex<f64>, 2> = vector![Complex::new(1., -2.), Complex::new(-1., 2.)];
        let v3: Vector<Complex<f64>, 2> = vector![Complex::new(1., -2.); 2];

        assert_eq!(v1.len(), 2);
        assert_relative_eq!(v1[0].re, 1. as f64);
        assert_relative_eq!(v1[0].im, -2. as f64);
        assert_relative_eq!(v1[1].re, -1. as f64);
        assert_relative_eq!(v1[1].im, 2. as f64);

        assert_eq!(v2.len(), 2);
        assert_relative_eq!(v2[0].re, 1. as f64);
        assert_relative_eq!(v2[0].im, -2. as f64);
        assert_relative_eq!(v2[1].re, -1. as f64);
        assert_relative_eq!(v2[1].im, 2. as f64);

        assert_eq!(v3.len(), 2);
        assert_relative_eq!(v3[0].re, 1. as f64);
        assert_relative_eq!(v3[0].im, -2. as f64);
        assert_relative_eq!(v3[1].re, 1. as f64);
        assert_relative_eq!(v3[1].im, -2. as f64);
    }

    #[test]
    fn macro_try_vector_i32() {
        let slv = vec![1_i32, 2_i32, 3_i32];
        let v: Vector<i32, 3> = try_vector!(slv).unwrap();

        assert_eq!(v[0], 1_i32);
        assert_eq!(v[1], 2_i32);
        assert_eq!(v[2], 3_i32);

        let slv = vec![1_i32, 2_i32, 3_i32];
        let v: Vector<i32, 3> = try_vector!(&slv).unwrap();

        assert_eq!(v[0], 1_i32);
        assert_eq!(v[1], 2_i32);
        assert_eq!(v[2], 3_i32);

        let slv = vec![1_i32, 2_i32, 3_i32];
        let v: Vector<i32, 3> = try_vector!(&slv[..]).unwrap();

        assert_eq!(v[0], 1_i32);
        assert_eq!(v[1], 2_i32);
        assert_eq!(v[2], 3_i32);
    }

    #[test]
    fn macro_try_vector_f64() {
        let slv = vec![1.0_f64, 2.0_f64, 3.0_f64];
        let v: Vector<f64, 3> = try_vector!(slv).unwrap();

        assert_relative_eq!(v[0], 1.0_f64);
        assert_relative_eq!(v[1], 2.0_f64);
        assert_relative_eq!(v[2], 3.0_f64);

        let slv = vec![1.0_f64, 2.0_f64, 3.0_f64];
        let v: Vector<f64, 3> = try_vector!(&slv).unwrap();

        assert_relative_eq!(v[0], 1.0_f64);
        assert_relative_eq!(v[1], 2.0_f64);
        assert_relative_eq!(v[2], 3.0_f64);

        let slv = vec![1.0_f64, 2.0_f64, 3.0_f64];
        let v: Vector<f64, 3> = try_vector!(&slv[..]).unwrap();

        assert_relative_eq!(v[0], 1.0_f64);
        assert_relative_eq!(v[1], 2.0_f64);
        assert_relative_eq!(v[2], 3.0_f64);
    }

    #[test]
    fn macro_try_vector_complex_i32() {
        let slv = vec![Complex::new(1_i32, 2_i32), Complex::new(3_i32, 4_i32)];
        let v: Vector<Complex<i32>, 2> = try_vector!(slv).unwrap();

        assert_eq!(v[0].re, 1_i32);
        assert_eq!(v[0].im, 2_i32);
        assert_eq!(v[1].re, 3_i32);
        assert_eq!(v[1].im, 4_i32);

        let slv = vec![Complex::new(1_i32, 2_i32), Complex::new(3_i32, 4_i32)];
        let v: Vector<Complex<i32>, 2> = try_vector!(&slv).unwrap();

        assert_eq!(v[0].re, 1_i32);
        assert_eq!(v[0].im, 2_i32);
        assert_eq!(v[1].re, 3_i32);
        assert_eq!(v[1].im, 4_i32);

        let slv = vec![Complex::new(1_i32, 2_i32), Complex::new(3_i32, 4_i32)];
        let v: Vector<Complex<i32>, 2> = try_vector!(&slv[..]).unwrap();

        assert_eq!(v[0].re, 1_i32);
        assert_eq!(v[0].im, 2_i32);
        assert_eq!(v[1].re, 3_i32);
        assert_eq!(v[1].im, 4_i32);
    }

    #[test]
    fn macro_try_vector_complex_f64() {
        let slv = vec![Complex::new(1.0_f64, 2.0_f64), Complex::new(3.0_f64, 4.0_f64)];
        let v: Vector<Complex<f64>, 2> = try_vector!(slv).unwrap();

        assert_relative_eq!(v[0].re, 1.0_f64);
        assert_relative_eq!(v[0].im, 2.0_f64);
        assert_relative_eq!(v[1].re, 3.0_f64);
        assert_relative_eq!(v[1].im, 4.0_f64);

        let slv = vec![Complex::new(1.0_f64, 2.0_f64), Complex::new(3.0_f64, 4.0_f64)];
        let v: Vector<Complex<f64>, 2> = try_vector!(&slv).unwrap();

        assert_relative_eq!(v[0].re, 1.0_f64);
        assert_relative_eq!(v[0].im, 2.0_f64);
        assert_relative_eq!(v[1].re, 3.0_f64);
        assert_relative_eq!(v[1].im, 4.0_f64);

        let slv = vec![Complex::new(1.0_f64, 2.0_f64), Complex::new(3.0_f64, 4.0_f64)];
        let v: Vector<Complex<f64>, 2> = try_vector!(&slv[..]).unwrap();

        assert_relative_eq!(v[0].re, 1.0_f64);
        assert_relative_eq!(v[0].im, 2.0_f64);
        assert_relative_eq!(v[1].re, 3.0_f64);
        assert_relative_eq!(v[1].im, 4.0_f64);
    }

    // -- fv! macro --

    #[test]
    fn fv_macro_default_column_vec_i32() {
        let v = fv![1, 2, 3];
        assert_eq!(v, FlexVector::<i32, Column>::from_vec(vec![1, 2, 3]));
        let v = fv![10; 4];
        assert_eq!(v, FlexVector::<i32, Column>::from_vec(vec![10, 10, 10, 10]));
    }

    #[test]
    fn fv_macro_explicit_column_vec_i32() {
        let v = fv![Column; 1, 2, 3];
        assert_eq!(v, FlexVector::<i32, Column>::from_vec(vec![1, 2, 3]));
        let v = fv![Column; 7; 2];
        assert_eq!(v, FlexVector::<i32, Column>::from_vec(vec![7, 7]));
    }

    #[test]
    fn fv_macro_row_vec_i32() {
        let v = fv![Row; 1, 2, 3];
        assert_eq!(v, FlexVector::<i32, Row>::from_vec(vec![1, 2, 3]));
        let v = fv![Row; 5; 3];
        assert_eq!(v, FlexVector::<i32, Row>::from_vec(vec![5, 5, 5]));
    }

    #[test]
    fn fv_macro_default_column_vec_f64() {
        let v = fv![1.0_f64, 2.0, 3.0];
        assert_eq!(v, FlexVector::<f64, Column>::from_vec(vec![1.0, 2.0, 3.0]));
        let v = fv![0.5_f64; 2];
        assert_eq!(v, FlexVector::<f64, Column>::from_vec(vec![0.5, 0.5]));
    }

    #[test]
    fn fv_macro_explicit_column_vec_f64() {
        let v = fv![Column; 1.0_f64, 2.0, 3.0];
        assert_eq!(v, FlexVector::<f64, Column>::from_vec(vec![1.0, 2.0, 3.0]));
        let v = fv![Column; 2.5_f64; 3];
        assert_eq!(v, FlexVector::<f64, Column>::from_vec(vec![2.5, 2.5, 2.5]));
    }

    #[test]
    fn fv_macro_row_vec_f64() {
        let v = fv![Row; 1.0_f64, 2.0, 3.0];
        assert_eq!(v, FlexVector::<f64, Row>::from_vec(vec![1.0, 2.0, 3.0]));
        let v = fv![Row; -1.5_f64; 2];
        assert_eq!(v, FlexVector::<f64, Row>::from_vec(vec![-1.5, -1.5]));
    }

    #[test]
    fn fv_macro_default_column_vec_complex_f64() {
        let v = fv![Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        assert_eq!(
            v,
            FlexVector::<Complex<f64>, Column>::from_vec(vec![
                Complex::new(1.0, 2.0),
                Complex::new(3.0, 4.0)
            ])
        );
        let v = fv![Complex::new(0.0_f64, 1.0); 2];
        assert_eq!(
            v,
            FlexVector::<Complex<f64>, Column>::from_vec(vec![
                Complex::new(0.0, 1.0),
                Complex::new(0.0, 1.0)
            ])
        );
    }

    #[test]
    fn fv_macro_explicit_column_vec_complex_f64() {
        let v = fv![Column; Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        assert_eq!(
            v,
            FlexVector::<Complex<f64>, Column>::from_vec(vec![
                Complex::new(1.0, 2.0),
                Complex::new(3.0, 4.0)
            ])
        );
        let v = fv![Column; Complex::new(2.0_f64, -2.0); 3];
        assert_eq!(
            v,
            FlexVector::<Complex<f64>, Column>::from_vec(vec![
                Complex::new(2.0, -2.0),
                Complex::new(2.0, -2.0),
                Complex::new(2.0, -2.0)
            ])
        );
    }

    #[test]
    fn fv_macro_row_vec_complex_f64() {
        let v = fv![Row; Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        assert_eq!(
            v,
            FlexVector::<Complex<f64>, Row>::from_vec(vec![
                Complex::new(1.0, 2.0),
                Complex::new(3.0, 4.0)
            ])
        );
        let v = fv![Row; Complex::new(-1.0_f64, 0.0); 2];
        assert_eq!(
            v,
            FlexVector::<Complex<f64>, Row>::from_vec(vec![
                Complex::new(-1.0, 0.0),
                Complex::new(-1.0, 0.0)
            ])
        );
    }

    // -- fv_iter! macro --

    #[test]
    fn fv_iter_macro_default_column_vec_i32() {
        let v = fv_iter![1..=3];
        assert_eq!(v, FlexVector::<i32, Column>::from_vec(vec![1, 2, 3]));

        let arr = [10, 20, 30];
        let v = fv_iter![arr];
        assert_eq!(v, FlexVector::<i32, Column>::from_vec(vec![10, 20, 30]));

        let v = fv_iter![vec![4, 5, 6]];
        assert_eq!(v, FlexVector::<i32, Column>::from_vec(vec![4, 5, 6]));
    }

    #[test]
    fn fv_iter_macro_explicit_column_vec_i32() {
        let v = fv_iter![Column; 1..=3];
        assert_eq!(v, FlexVector::<i32, Column>::from_vec(vec![1, 2, 3]));

        let arr = [7, 8];
        let v = fv_iter![Column; arr];
        assert_eq!(v, FlexVector::<i32, Column>::from_vec(vec![7, 8]));

        let v = fv_iter![Column; vec![9, 10]];
        assert_eq!(v, FlexVector::<i32, Column>::from_vec(vec![9, 10]));
    }

    #[test]
    fn fv_iter_macro_row_vec_i32() {
        let v = fv_iter![Row; 1..=3];
        assert_eq!(v, FlexVector::<i32, Row>::from_vec(vec![1, 2, 3]));

        let arr = [11, 12];
        let v = fv_iter![Row; arr];
        assert_eq!(v, FlexVector::<i32, Row>::from_vec(vec![11, 12]));

        let v = fv_iter![Row; vec![13, 14]];
        assert_eq!(v, FlexVector::<i32, Row>::from_vec(vec![13, 14]));
    }

    #[test]
    fn fv_iter_macro_default_column_vec_f64() {
        let v = fv_iter![0..3];
        assert_eq!(v, FlexVector::<i32, Column>::from_vec(vec![0, 1, 2]));

        let v = fv_iter![vec![1.5, 2.5, 3.5]];
        assert_eq!(v, FlexVector::<f64, Column>::from_vec(vec![1.5, 2.5, 3.5]));
    }

    #[test]
    fn fv_iter_macro_explicit_column_vec_f64() {
        let v = fv_iter![Column; 0..2];
        assert_eq!(v, FlexVector::<i32, Column>::from_vec(vec![0, 1]));

        let v = fv_iter![Column; vec![4.4, 5.5]];
        assert_eq!(v, FlexVector::<f64, Column>::from_vec(vec![4.4, 5.5]));
    }

    #[test]
    fn fv_iter_macro_row_vec_f64() {
        let v = fv_iter![Row; 1..=2];
        assert_eq!(v, FlexVector::<i32, Row>::from_vec(vec![1, 2]));

        let v = fv_iter![Row; vec![-1.5, -2.5]];
        assert_eq!(v, FlexVector::<f64, Row>::from_vec(vec![-1.5, -2.5]));
    }

    #[test]
    fn fv_iter_macro_default_column_vec_complex_f64() {
        use num::Complex;
        let v = fv_iter![vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]];
        assert_eq!(
            v,
            FlexVector::<Complex<f64>, Column>::from_vec(vec![
                Complex::new(1.0, 2.0),
                Complex::new(3.0, 4.0)
            ])
        );
    }

    #[test]
    fn fv_iter_macro_explicit_column_vec_complex_f64() {
        use num::Complex;
        let v = fv_iter![Column; vec![Complex::new(2.0, -2.0), Complex::new(0.0, 1.0)]];
        assert_eq!(
            v,
            FlexVector::<Complex<f64>, Column>::from_vec(vec![
                Complex::new(2.0, -2.0),
                Complex::new(0.0, 1.0)
            ])
        );
    }

    #[test]
    fn fv_iter_macro_row_vec_complex_f64() {
        use num::Complex;
        let v = fv_iter![Row; vec![Complex::new(-1.0, 0.0), Complex::new(2.0, 2.0)]];
        assert_eq!(
            v,
            FlexVector::<Complex<f64>, Row>::from_vec(vec![
                Complex::new(-1.0, 0.0),
                Complex::new(2.0, 2.0)
            ])
        );
    }

    #[test]
    fn fv_iter_macro_empty() {
        let v: FlexVector<i32, Column> = fv_iter![std::iter::empty::<i32>()];
        assert!(v.is_empty());

        let v: FlexVector<f64, Row> = fv_iter![Row; std::iter::empty::<f64>()];
        assert!(v.is_empty());
    }

    // -- try_fv_iter! macro --

    #[test]
    fn try_fv_iter_macro_default_column_vec_i32() {
        let data: Vec<Result<i32, ()>> = vec![Ok(1), Ok(2), Ok(3)];
        let fv = try_fv_iter!(data).unwrap();
        assert_eq!(fv, FlexVector::<i32, Column>::from_vec(vec![1, 2, 3]));
    }

    #[test]
    fn try_fv_iter_macro_explicit_column_vec_i32() {
        let data: Vec<Result<i32, ()>> = vec![Ok(4), Ok(5)];
        let fv = try_fv_iter!(Column; data).unwrap();
        assert_eq!(fv, FlexVector::<i32, Column>::from_vec(vec![4, 5]));
    }

    #[test]
    fn try_fv_iter_macro_row_vec_i32() {
        let data: Vec<Result<i32, ()>> = vec![Ok(7), Ok(8), Ok(9)];
        let fv = try_fv_iter!(Row; data).unwrap();
        assert_eq!(fv, FlexVector::<i32, Row>::from_vec(vec![7, 8, 9]));
    }

    #[test]
    fn try_fv_iter_macro_default_column_vec_f64() {
        let data: Vec<Result<f64, ()>> = vec![Ok(1.5), Ok(2.5)];
        let fv = try_fv_iter!(data).unwrap();
        assert_eq!(fv, FlexVector::<f64, Column>::from_vec(vec![1.5, 2.5]));
    }

    #[test]
    fn try_fv_iter_macro_explicit_column_vec_f64() {
        let data: Vec<Result<f64, ()>> = vec![Ok(3.0), Ok(4.0), Ok(5.0)];
        let fv = try_fv_iter!(Column; data).unwrap();
        assert_eq!(fv, FlexVector::<f64, Column>::from_vec(vec![3.0, 4.0, 5.0]));
    }

    #[test]
    fn try_fv_iter_macro_row_vec_f64() {
        let data: Vec<Result<f64, ()>> = vec![Ok(-1.0), Ok(-2.0)];
        let fv = try_fv_iter!(Row; data).unwrap();
        assert_eq!(fv, FlexVector::<f64, Row>::from_vec(vec![-1.0, -2.0]));
    }

    #[test]
    fn try_fv_iter_macro_default_column_vec_complex_f64() {
        let data: Vec<Result<Complex<f64>, ()>> =
            vec![Ok(Complex::new(1.0, 2.0)), Ok(Complex::new(3.0, 4.0))];
        let fv = try_fv_iter!(data).unwrap();
        assert_eq!(
            fv,
            FlexVector::<Complex<f64>, Column>::from_vec(vec![
                Complex::new(1.0, 2.0),
                Complex::new(3.0, 4.0)
            ])
        );
    }

    #[test]
    fn try_fv_iter_macro_explicit_column_vec_complex_f64() {
        let data: Vec<Result<Complex<f64>, ()>> =
            vec![Ok(Complex::new(2.0, -2.0)), Ok(Complex::new(0.0, 1.0))];
        let fv = try_fv_iter!(Column; data).unwrap();
        assert_eq!(
            fv,
            FlexVector::<Complex<f64>, Column>::from_vec(vec![
                Complex::new(2.0, -2.0),
                Complex::new(0.0, 1.0)
            ])
        );
    }

    #[test]
    fn try_fv_iter_macro_row_vec_complex_f64() {
        let data: Vec<Result<Complex<f64>, ()>> =
            vec![Ok(Complex::new(-1.0, 0.0)), Ok(Complex::new(2.0, 2.0))];
        let fv = try_fv_iter!(Row; data).unwrap();
        assert_eq!(
            fv,
            FlexVector::<Complex<f64>, Row>::from_vec(vec![
                Complex::new(-1.0, 0.0),
                Complex::new(2.0, 2.0)
            ])
        );
    }

    #[test]
    fn try_fv_iter_macro_error_propagation() {
        let data = vec![Ok(1), Err("fail"), Ok(3)];
        let result: Result<FlexVector<i32, Column>, &str> = try_fv_iter!(data);
        assert_eq!(result.unwrap_err(), "fail");
    }

    #[test]
    fn try_fv_iter_macro_empty() {
        let data: Vec<Result<f64, ()>> = vec![];
        let fv: FlexVector<f64, Column> = try_fv_iter!(data).unwrap();
        assert!(fv.is_empty());
    }
}
