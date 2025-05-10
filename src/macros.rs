//! Macros.

/// Returns a [`crate::Vector`] with scalar data contents and order as defined in
/// the numeric type arguments.
///
/// This macro supports standard library [`array`]-like initialization syntax of
/// a [`crate::Vector`] with the numeric type and length defined by the macro arguments.
///
/// Import the macro before use with:
///
/// ```
/// use vectora::vector;
/// ```
///
/// # Examples
///
/// ## Integer types
///
/// ```
/// use vectora::{vector, Vector};
///
/// let v_i32_1 = vector![1_i32, 2_i32, 3_i32];
/// let v_i32_2: Vector<i32, 3> = vector![1, 2, 3];
/// let v_i32_3: Vector<i32, 3> = vector![10; 3];
///
/// assert_eq!(v_i32_1[0], 1_i32);
/// assert_eq!(v_i32_1[1], 2_i32);
/// assert_eq!(v_i32_1[2], 3_i32);
///
/// assert_eq!(v_i32_2[0], 1_i32);
/// assert_eq!(v_i32_2[1], 2_i32);
/// assert_eq!(v_i32_2[2], 3_i32);
///
/// assert_eq!(v_i32_3[0], 10_i32);
/// assert_eq!(v_i32_3[1], 10_i32);
/// assert_eq!(v_i32_3[2], 10_i32);
/// ```
///
/// ## Floating point types
///
/// ```
/// use vectora::{vector, Vector};
///
/// use approx::assert_relative_eq;
///
/// let v_f64_1 = vector![1.0_f64, 2.0_f64, 3.0_f64];
/// let v_f64_2: Vector<f64, 3> = vector![1.0, 2.0, 3.0];
/// let v_f64_3: Vector<f64, 3> = vector![10.0; 3];
///
/// assert_relative_eq!(v_f64_1[0], 1.0_f64);
/// assert_relative_eq!(v_f64_1[1], 2.0_f64);
/// assert_relative_eq!(v_f64_1[2], 3.0_f64);
///
/// assert_relative_eq!(v_f64_2[0], 1.0_f64);
/// assert_relative_eq!(v_f64_2[1], 2.0_f64);
/// assert_relative_eq!(v_f64_2[2], 3.0_f64);
///
/// assert_relative_eq!(v_f64_3[0], 10.0_f64);
/// assert_relative_eq!(v_f64_3[1], 10.0_f64);
/// assert_relative_eq!(v_f64_3[2], 10.0_f64);
/// ```
///
/// ## Complex number types
///
/// ```
/// use vectora::{vector, Vector};
///
/// use approx::assert_relative_eq;
/// use num::Complex;
///
/// let v_complex = vector![Complex::new(1.0_f64, 2.0_f64), Complex::new(-1.0_f64, -2.0_f64)];
///
/// assert_relative_eq!(v_complex[0].re, 1.0_f64);
/// assert_relative_eq!(v_complex[0].im, 2.0_f64);
/// assert_relative_eq!(v_complex[1].re, -1.0_f64);
/// assert_relative_eq!(v_complex[1].im, -2.0_f64);
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

/// Returns a [`crate::Vector`] with scalar data contents and order as defined in a supported
/// fallible numeric data collection type argument.
///
/// This macro can be used with supported types that may not have known length at compile
/// time (e.g., [`Vec`] and [`slice`]).  The macro takes a single numeric data collection
/// type argument.
///
/// Import the macro before use with:
///
/// ```
/// use vectora::try_vector;
/// ```
///
/// # Errors
///
/// The macro returns an error if the length of the argument data collection type differs from the
/// requested [`crate::Vector`] length.  Errors are also propagated from argument data types. Please review
/// the crate `TryFrom` trait implementation documentation for additional details.
///
/// ```
/// use vectora::{try_vector, Vector};
///
/// let stdlib_vec_too_long = vec![1, 2, 3, 4, 5];
/// let res: Result<Vector<i32, 3>, _> = try_vector!(stdlib_vec_too_long);
///
/// assert!(res.is_err());
/// ```
///
/// ```
/// use vectora::{try_vector, Vector};
///
/// let stdlib_vec_too_short = vec![1, 2];
/// let res: Result<Vector<i32, 3>, _> = try_vector!(stdlib_vec_too_short);
///
/// assert!(res.is_err());
/// ```
///
/// # Examples
///
/// ## Integer types
///
/// ```
/// use vectora::{try_vector, Vector};
///
/// let stdlib_vec_i32 = vec![1_i32, 2_i32, 3_i32];
/// let v_i32: Vector<i32, 3> = try_vector!(stdlib_vec_i32).unwrap();
///
/// assert_eq!(v_i32[0], 1_i32);
/// assert_eq!(v_i32[1], 2_i32);
/// assert_eq!(v_i32[2], 3_i32);
/// ```
///
/// ## Floating point types
///
/// ```
/// use vectora::{try_vector, Vector};
///
/// use approx::assert_relative_eq;
///
/// let stdlib_vec_f64 = vec![1.0_f64, 2.0_f64, 3.0_f64];
/// let v_f64: Vector<f64, 3> = try_vector!(stdlib_vec_f64).unwrap();
///
/// assert_relative_eq!(v_f64[0], 1.0_f64);
/// assert_relative_eq!(v_f64[1], 2.0_f64);
/// assert_relative_eq!(v_f64[2], 3.0_f64);
/// ```
///
/// ## Complex number types
///
/// ```
/// use vectora::{try_vector, Vector};
///
/// use approx::assert_relative_eq;
/// use num::Complex;
///
/// let stdlib_vec_complex = vec![Complex::new(1.0_f64, 2.0_f64), Complex::new(3.0_f64, 4.0_f64)];
/// let v_complex_f64: Vector<Complex<f64>, 2> = try_vector!(stdlib_vec_complex).unwrap();
///
/// assert_relative_eq!(v_complex_f64[0].re, 1.0_f64);
/// assert_relative_eq!(v_complex_f64[0].im, 2.0_f64);
/// assert_relative_eq!(v_complex_f64[1].re, 3.0_f64);
/// assert_relative_eq!(v_complex_f64[1].im, 4.0_f64);
/// ```
#[macro_export]
macro_rules! try_vector {
    ($elem:expr) => {
        $crate::types::vector::Vector::try_from($elem)
    };
}

#[macro_export]
macro_rules! impl_vector_unary_op {
    ($VectorType:ident, $trait:ident, $method:ident, $op:tt) => {
        impl<T> std::ops::$trait for $VectorType<T>
        where
            T: num::Num + Clone + Sync + Send + std::ops::Neg<Output = T>,
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
    ($VectorType:ident, $trait:ident, $method:ident, $op:tt) => {
        impl<T> std::ops::$trait for $VectorType<T>
        where
            T: num::Num + Clone + Sync + Send,
        {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                assert_eq!(self.len(), rhs.len());
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
macro_rules! impl_vector_binop_assign {
    ($VectorType:ident, $trait:ident, $method:ident, $op:tt) => {
        impl<T> std::ops::$trait for $VectorType<T>
        where
            T: num::Num + Clone + Sync + Send,
        {
            fn $method(&mut self, rhs: Self) {
                assert_eq!(self.len(), rhs.len());
                for (a, b) in self.components.iter_mut().zip(rhs.components) {
                    *a = a.clone() $op b;
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
            T: num::Num + Clone + Sync + Send,
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
            T: num::Num + Clone + Sync + Send,
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
}
