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

/// Creates a [`FlexVector`] with the given elements or repeated value.
///
/// - `flexvector![x, y, z]` creates a FlexVector from the elements.
/// - `flexvector![elem; n]` creates a FlexVector of length `n` with all elements set to `elem`.
///
/// # Examples
///
/// ```
/// use vectora::prelude::*;
/// use num::Complex;
///
/// let fv = flexvector![1, 2, 3];
/// let fv_f64 = flexvector![1.0_f64, 2.0_f64, 3.0_f64];
/// let fv_repeat = flexvector![0; 4];
/// let fv_complex = flexvector![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
/// ```
#[macro_export]
macro_rules! flexvector {
    // Repeated element syntax: flexvector![elem; n]
    ($elem:expr; $n:expr) => {
        $crate::types::flexvector::FlexVector::from_vec(vec![$elem; $n])
    };
    // List syntax: flexvector![x, y, z, ...]
    ($($x:expr),+ $(,)?) => {
        $crate::types::flexvector::FlexVector::from_vec(vec![$($x),+])
    };
}

/// Fallibly constructs a [`FlexVector`] from an iterator of `Result<T, E>`, propagating the first error.
///
/// - `try_flexvector!(iter)` collects all `Ok(T)` values into a FlexVector, or returns the first `Err(E)` encountered.
///
/// # Examples
///
/// ```
/// use vectora::prelude::*;
///
/// let data = vec!["1", "2", "oops"];
/// let fv = try_flexvector!(data.iter().map(|s| s.parse::<i32>()));
/// assert!(fv.is_err());
///
/// let data = vec!["1", "2", "3"];
/// let fv = try_flexvector!(data.iter().map(|s| s.parse::<i32>())).unwrap();
/// assert_eq!(fv.as_slice(), &[1, 2, 3]);
/// ```
#[macro_export]
macro_rules! try_flexvector {
    ($iter:expr) => {
        $crate::types::flexvector::FlexVector::try_from_iter($iter)
    };
}

#[macro_export]
macro_rules! impl_vector_unary_op {
    ($VectorType:ident, $trait:ident, $method:ident, $op:tt) => {
        impl<T> std::ops::$trait for $VectorType<T>
        where
            T: num::Num + Clone + std::ops::Neg<Output = T>,
        {
            type Output = Self;
            fn $method(self) -> Self {
                let components = self.components.into_iter().map(|a| $op a).collect();
                Self { components }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vector_binop {
    // With length check (for FlexVector)
    (check_len, $VectorType:ident, $trait:ident, $method:ident, $op:tt) => {
        impl<T> std::ops::$trait for $VectorType<T>
        where
            T: num::Num + Clone,
        {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                assert_eq!(self.len(), rhs.len(), "Vector length mismatch");
                let components = self.components.into_iter()
                    .zip(rhs.components)
                    .map(|(a, b)| a $op b)
                    .collect();
                Self { components }
            }
        }
    };
    // Without length check (for Vector)
    (no_check_len, $VectorType:ident, $trait:ident, $method:ident, $op:tt) => {
        impl<T> std::ops::$trait for $VectorType<T>
        where
            T: num::Num + Clone,
        {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                let components = self.components.into_iter()
                    .zip(rhs.components)
                    .map(|(a, b)| a $op b)
                    .collect();
                Self { components }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vector_binop_div {
    // With length check (for FlexVector)
    (check_len, $VectorType:ident) => {
        // f32
        impl std::ops::Div for $VectorType<f32> {
            type Output = Self;
            fn div(self, rhs: Self) -> Self {
                assert_eq!(self.len(), rhs.len(), "Vector length mismatch");
                let components =
                    self.components.into_iter().zip(rhs.components).map(|(a, b)| a / b).collect();
                Self { components }
            }
        }
        // f64
        impl std::ops::Div for $VectorType<f64> {
            type Output = Self;
            fn div(self, rhs: Self) -> Self {
                assert_eq!(self.len(), rhs.len(), "Vector length mismatch");
                let components =
                    self.components.into_iter().zip(rhs.components).map(|(a, b)| a / b).collect();
                Self { components }
            }
        }
        // Complex<f32>
        impl std::ops::Div for $VectorType<num::Complex<f32>> {
            type Output = Self;
            fn div(self, rhs: Self) -> Self {
                assert_eq!(self.len(), rhs.len(), "Vector length mismatch");
                let components =
                    self.components.into_iter().zip(rhs.components).map(|(a, b)| a / b).collect();
                Self { components }
            }
        }
        // Complex<f64>
        impl std::ops::Div for $VectorType<num::Complex<f64>> {
            type Output = Self;
            fn div(self, rhs: Self) -> Self {
                assert_eq!(self.len(), rhs.len(), "Vector length mismatch");
                let components =
                    self.components.into_iter().zip(rhs.components).map(|(a, b)| a / b).collect();
                Self { components }
            }
        }
    };
    // Without length check (for Vector)
    (no_check_len, $VectorType:ident) => {
        // f32
        impl std::ops::Div for $VectorType<f32> {
            type Output = Self;
            fn div(self, rhs: Self) -> Self {
                let components =
                    self.components.into_iter().zip(rhs.components).map(|(a, b)| a / b).collect();
                Self { components }
            }
        }
        // f64
        impl std::ops::Div for $VectorType<f64> {
            type Output = Self;
            fn div(self, rhs: Self) -> Self {
                let components =
                    self.components.into_iter().zip(rhs.components).map(|(a, b)| a / b).collect();
                Self { components }
            }
        }
        // Complex<f32>
        impl std::ops::Div for $VectorType<num::Complex<f32>> {
            type Output = Self;
            fn div(self, rhs: Self) -> Self {
                let components =
                    self.components.into_iter().zip(rhs.components).map(|(a, b)| a / b).collect();
                Self { components }
            }
        }
        // Complex<f64>
        impl std::ops::Div for $VectorType<num::Complex<f64>> {
            type Output = Self;
            fn div(self, rhs: Self) -> Self {
                let components =
                    self.components.into_iter().zip(rhs.components).map(|(a, b)| a / b).collect();
                Self { components }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vector_binop_assign {
    // With length check (for FlexVector)
    (check_len, $VectorType:ident, $trait:ident, $method:ident, $op:tt) => {
        impl<T> std::ops::$trait for $VectorType<T>
        where
            T: num::Num + Clone,
        {
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
        impl<T> std::ops::$trait for $VectorType<T>
        where
            T: num::Num + Clone,
        {
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
        impl std::ops::DivAssign for $VectorType<f32> {
            fn div_assign(&mut self, rhs: Self) {
                assert_eq!(self.len(), rhs.len(), "Vector length mismatch");
                for (a, b) in self.components.iter_mut().zip(rhs.components) {
                    *a /= b;
                }
            }
        }
        // f64
        impl std::ops::DivAssign for $VectorType<f64> {
            fn div_assign(&mut self, rhs: Self) {
                assert_eq!(self.len(), rhs.len(), "Vector length mismatch");
                for (a, b) in self.components.iter_mut().zip(rhs.components) {
                    *a /= b;
                }
            }
        }
        // Complex<f32>
        impl std::ops::DivAssign for $VectorType<num::Complex<f32>> {
            fn div_assign(&mut self, rhs: Self) {
                assert_eq!(self.len(), rhs.len(), "Vector length mismatch");
                for (a, b) in self.components.iter_mut().zip(rhs.components) {
                    *a /= b;
                }
            }
        }
        // Complex<f64>
        impl std::ops::DivAssign for $VectorType<num::Complex<f64>> {
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
        impl std::ops::DivAssign for $VectorType<f32> {
            fn div_assign(&mut self, rhs: Self) {
                for (a, b) in self.components.iter_mut().zip(rhs.components) {
                    *a /= b;
                }
            }
        }
        // f64
        impl std::ops::DivAssign for $VectorType<f64> {
            fn div_assign(&mut self, rhs: Self) {
                for (a, b) in self.components.iter_mut().zip(rhs.components) {
                    *a /= b;
                }
            }
        }
        // Complex<f32>
        impl std::ops::DivAssign for $VectorType<num::Complex<f32>> {
            fn div_assign(&mut self, rhs: Self) {
                for (a, b) in self.components.iter_mut().zip(rhs.components) {
                    *a /= b;
                }
            }
        }
        // Complex<f64>
        impl std::ops::DivAssign for $VectorType<num::Complex<f64>> {
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
        impl<T> std::ops::$trait<T> for $VectorType<T>
        where
            T: num::Num + Clone,
        {
            type Output = Self;
            fn $method(self, rhs: T) -> Self {
                let components = self.components.into_iter().map(|a| a $op rhs.clone()).collect();
                Self { components }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vector_scalar_op_assign {
    ($VectorType:ident, $trait:ident, $method:ident, $op:tt) => {
        impl<T> std::ops::$trait<T> for $VectorType<T>
        where
            T: num::Num + Clone,
        {
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
        impl std::ops::Div<f32> for $VectorType<f32> {
            type Output = Self;
            fn div(self, rhs: f32) -> Self {
                let components = self.components.into_iter().map(|a| a / rhs).collect();
                Self { components }
            }
        }
        // For f64
        impl std::ops::Div<f64> for $VectorType<f64> {
            type Output = Self;
            fn div(self, rhs: f64) -> Self {
                let components = self.components.into_iter().map(|a| a / rhs).collect();
                Self { components }
            }
        }
        // For Complex<f32> / f32
        impl std::ops::Div<f32> for $VectorType<num::Complex<f32>> {
            type Output = Self;
            fn div(self, rhs: f32) -> Self {
                let components = self.components.into_iter().map(|a| a / rhs).collect();
                Self { components }
            }
        }
        // For Complex<f64> / f64
        impl std::ops::Div<f64> for $VectorType<num::Complex<f64>> {
            type Output = Self;
            fn div(self, rhs: f64) -> Self {
                let components = self.components.into_iter().map(|a| a / rhs).collect();
                Self { components }
            }
        }
        // For Complex<f32> / Complex<f32>
        impl std::ops::Div<num::Complex<f32>> for $VectorType<num::Complex<f32>> {
            type Output = Self;
            fn div(self, rhs: num::Complex<f32>) -> Self {
                let components = self.components.into_iter().map(|a| a / rhs).collect();
                Self { components }
            }
        }
        // For Complex<f64> / Complex<f64>
        impl std::ops::Div<num::Complex<f64>> for $VectorType<num::Complex<f64>> {
            type Output = Self;
            fn div(self, rhs: num::Complex<f64>) -> Self {
                let components = self.components.into_iter().map(|a| a / rhs).collect();
                Self { components }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vector_scalar_div_op_assign {
    ($VectorType:ident) => {
        // For f32
        impl std::ops::DivAssign<f32> for $VectorType<f32> {
            fn div_assign(&mut self, rhs: f32) {
                for a in &mut self.components {
                    *a = *a / rhs;
                }
            }
        }
        // For f64
        impl std::ops::DivAssign<f64> for $VectorType<f64> {
            fn div_assign(&mut self, rhs: f64) {
                for a in &mut self.components {
                    *a = *a / rhs;
                }
            }
        }
        // For Complex<f32> / f32
        impl std::ops::DivAssign<f32> for $VectorType<num::Complex<f32>> {
            fn div_assign(&mut self, rhs: f32) {
                for a in &mut self.components {
                    *a = *a / rhs;
                }
            }
        }
        // For Complex<f64> / f64
        impl std::ops::DivAssign<f64> for $VectorType<num::Complex<f64>> {
            fn div_assign(&mut self, rhs: f64) {
                for a in &mut self.components {
                    *a = *a / rhs;
                }
            }
        }
        // For Complex<f32> / Complex<f32>
        impl std::ops::DivAssign<num::Complex<f32>> for $VectorType<num::Complex<f32>> {
            fn div_assign(&mut self, rhs: num::Complex<f32>) {
                for a in &mut self.components {
                    *a = *a / rhs;
                }
            }
        }
        // For Complex<f64> / Complex<f64>
        impl std::ops::DivAssign<num::Complex<f64>> for $VectorType<num::Complex<f64>> {
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
    use crate::types::traits::VectorBase;
    use crate::Vector;
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

    #[test]
    fn macro_flexvector_usize() {
        let v1 = flexvector![1_usize, 2_usize, 3_usize];
        let v2 = flexvector![1, 2, 3];
        let v3 = flexvector![1; 3];

        assert_eq!(v1.len(), 3);
        assert_eq!(v1[0], 1_usize);
        assert_eq!(v1[1], 2_usize);
        assert_eq!(v1[2], 3_usize);

        assert_eq!(v2.len(), 3);
        assert_eq!(v2[0], 1_usize);
        assert_eq!(v2[1], 2_usize);
        assert_eq!(v2[2], 3_usize);

        assert_eq!(v3.len(), 3);
        assert_eq!(v3[0], 1_usize);
        assert_eq!(v3[1], 1_usize);
        assert_eq!(v3[2], 1_usize);
    }

    #[test]
    fn macro_flexvector_f64() {
        let v1 = flexvector![1.0_f64, 2.0, 3.0];
        let v2 = flexvector![1.0, 2.0, 3.0];
        let v3 = flexvector![1.0; 3];

        assert_eq!(v1.len(), 3);
        assert_eq!(v1[0], 1.0);
        assert_eq!(v1[1], 2.0);
        assert_eq!(v1[2], 3.0);

        assert_eq!(v2.len(), 3);
        assert_eq!(v2[0], 1.0);
        assert_eq!(v2[1], 2.0);
        assert_eq!(v2[2], 3.0);

        assert_eq!(v3.len(), 3);
        assert_eq!(v3[0], 1.0);
        assert_eq!(v3[1], 1.0);
        assert_eq!(v3[2], 1.0);
    }

    #[test]
    fn macro_flexvector_complex_f64() {
        let v1 = flexvector![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let v2 = flexvector![Complex::new(1.0, 2.0); 2];

        assert_eq!(v1.len(), 2);
        assert_eq!(v1[0], Complex::new(1.0, 2.0));
        assert_eq!(v1[1], Complex::new(3.0, 4.0));

        assert_eq!(v2.len(), 2);
        assert_eq!(v2[0], Complex::new(1.0, 2.0));
        assert_eq!(v2[1], Complex::new(1.0, 2.0));
    }

    #[test]
    fn macro_try_flexvector_success() {
        let data: Vec<Result<i32, ()>> = vec![Ok(1), Ok(2), Ok(3)];
        let fv = try_flexvector!(data).unwrap();
        assert_eq!(fv.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn macro_try_flexvector_parse_success() {
        let data = vec!["1", "2", "3"];
        let fv = try_flexvector!(data.iter().map(|s| s.parse::<i32>())).unwrap();
        assert_eq!(fv.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn macro_try_flexvector_parse_error() {
        let data = vec!["1", "oops", "3"];
        let result = try_flexvector!(data.iter().map(|s| s.parse::<i32>()));
        assert!(result.is_err());
    }

    #[test]
    fn macro_try_flexvector_custom_error() {
        #[derive(Debug, PartialEq)]
        enum MyError {
            Fail,
        }
        let data = vec![Ok(1), Err(MyError::Fail), Ok(3)];
        let result = try_flexvector!(data);
        assert_eq!(result.unwrap_err(), MyError::Fail);
    }

    #[test]
    fn macro_try_flexvector_empty() {
        let data: Vec<Result<i32, &str>> = vec![];
        let fv = try_flexvector!(data).unwrap();
        assert!(fv.is_empty());
    }
}
