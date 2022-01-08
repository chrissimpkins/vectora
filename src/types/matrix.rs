//! Matrix types

use std::fmt::Debug;
use std::ops::{Index, IndexMut};
use std::slice::SliceIndex;

use num::Num;

use crate::errors::MatrixError;
use crate::Vector;

/// A generic M-by-N matrix type parameterized by the numeric type `T`.
#[derive(Clone, Debug)]
pub struct Matrix<T>
where
    T: Num + Copy + Sync + Send,
{
    /// Matrix column data
    pub rows: Vec<Vec<T>>,
}

impl<T> Matrix<T>
where
    T: Num + Copy + Sync + Send + Default,
{
    /// Returns a new M-by-N [`Matrix`] initialized with `T` numeric
    /// type zero scalar values, row number defined by the `rows` parameter,
    /// and column number as defined by `N`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use vectora::Matrix;
    ///
    /// let m: Matrix<i32> = Matrix::new(2, 3);
    /// assert_eq!(m.rows.len(), 2);
    ///
    /// assert_eq!(m[0].len(), 3);
    /// assert_eq!(m[0][0], 0_i32);
    /// assert_eq!(m[0][1], 0_i32);
    /// assert_eq!(m[0][2], 0_i32);
    ///
    /// assert_eq!(m[1].len(), 3);
    /// assert_eq!(m[1][0], 0_i32);
    /// assert_eq!(m[1][1], 0_i32);
    /// assert_eq!(m[1][2], 0_i32);
    /// ```
    pub fn new(rows: usize, columns: usize) -> Self {
        if rows == 0 || columns == 0 {
            Self { rows: vec![] }
        } else {
            Self { rows: vec![vec![T::zero(); columns]; rows] }
        }
    }

    /// Returns a new M-by-N [`Matrix`] initialized with default (zero) `T` numeric
    /// type scalar values, row number defined by the `rows` parameter,
    /// and column number as defined by `N`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use vectora::Matrix;
    ///
    /// let m: Matrix<i32> = Matrix::zero(2, 3);
    /// assert_eq!(m.rows.len(), 2);
    ///
    /// assert_eq!(m[0].len(), 3);
    /// assert_eq!(m[0][0], 0_i32);
    /// assert_eq!(m[0][1], 0_i32);
    /// assert_eq!(m[0][2], 0_i32);
    ///
    /// assert_eq!(m[1].len(), 3);
    /// assert_eq!(m[1][0], 0_i32);
    /// assert_eq!(m[1][1], 0_i32);
    /// assert_eq!(m[1][2], 0_i32);
    /// ```
    pub fn zero(rows: usize, columns: usize) -> Self {
        Self::new(rows, columns)
    }

    /// Returns a new M-by-N [`Matrix`] initialized with `T` numeric
    /// type scalar values as defined in the `rows` collection, row number
    /// as defined by the length of the `rows` collection, and column number
    /// as defined by the `N` length of row vector elements in the `rows` collection.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use vectora::Matrix;
    ///
    /// let rows = [vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]];
    /// let m = Matrix::from_rows(&rows);
    ///
    /// assert_eq!(m.rows.len(), 3);
    /// assert_eq!(m.rows[0].len(), 3);
    /// assert_eq!(m[0].len(), 3);
    /// assert_eq!(m.rows[1].len(), 3);
    /// assert_eq!(m[1].len(), 3);
    /// assert_eq!(m.rows[2].len(), 3);
    /// assert_eq!(m[2].len(), 3);
    ///
    /// assert_eq!(m[0][0], 0);
    /// assert_eq!(m[0][1], 1);
    /// assert_eq!(m[0][2], 2);
    ///
    /// assert_eq!(m[1][0], 3);
    /// assert_eq!(m[1][1], 4);
    /// assert_eq!(m[1][2], 5);
    ///
    /// assert_eq!(m[2][0], 6);
    /// assert_eq!(m[2][1], 7);
    /// assert_eq!(m[2][2], 8);
    /// ```
    pub fn from_rows(rows: &[Vec<T>]) -> Self {
        if rows.is_empty() {
            // Instantiate with an empty Vec when the row vector length is zero.
            Self { rows: vec![] }
        } else {
            let mut new_rows = Vec::<Vec<T>>::with_capacity(rows.len());

            for row in rows.iter() {
                new_rows.push(row.to_vec());
            }

            Self { rows: new_rows }
        }
    }

    /// Returns the M-by-N [`Matrix`] `(number of row vectors = M, number of column vectors = N)`
    /// as [`usize`] values.
    ///
    /// TODO:
    pub fn dim(&self) -> (usize, usize) {
        (self.dim_row(), self.dim_column())
    }

    /// Returns the M-by-N [`Matrix`] number of row vectors `M` as a [`usize`] value.
    ///
    ///  TODO:
    pub fn dim_row(&self) -> usize {
        if self.rows.is_empty() {
            0
        } else {
            self.rows.len()
        }
    }

    /// Returns the M-by-N [`Matrix`] number of column vectors `N` as a [`usize`] value.
    ///
    /// TODO:
    pub fn dim_column(&self) -> usize {
        if self.rows.is_empty() {
            0
        } else {
            // measured as the length of the index 0 row in a non-empty Matrix
            self.rows[0].len()
        }
    }

    /// Returns a [`Matrix`] column [`Vec`] given the zero-based column index parameter `column_index`.
    ///
    /// Returns `None` if the index is out of range or the [`Matrix`] is empty.
    ///
    /// TODO:
    pub fn get_column_vec(&self, column_index: usize) -> Option<Vec<T>> {
        if self.rows.is_empty() || self.rows[0].len() < column_index + 1 {
            return None;
        }

        let mut v = Vec::<T>::with_capacity(self.dim_column());
        for row in self.rows.iter() {
            v.push(row[column_index]);
        }

        Some(v)
    }

    /// Returns a [`Matrix`] row vector given the zero-based row index parameter `row_index`.
    ///
    /// Returns `None` if the index is out of range or the [`Matrix`] is empty.
    ///
    /// TODO:
    pub fn get_row_vec(&self, row_index: usize) -> Option<Vec<T>> {
        if self.rows.is_empty() || self.rows.len() < row_index + 1 {
            return None;
        }

        Some(self.rows[row_index].clone())
    }

    /// Returns a [`Matrix`] column [`Vector`] given the zero-based column index parameter
    /// `column_index`.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::TryFromMatrixError`] when:
    ///
    /// - the `column_index` request is out of bounds
    /// - the [`Matrix`] is empty
    ///
    /// TODO:
    pub fn get_column_vector<const N: usize>(
        &self,
        column_index: usize,
    ) -> Result<Vector<T, N>, MatrixError>
    where
        Vector<T, N>: TryFrom<Vec<T>>,
    {
        // validate the Matrix shape and request
        // the matrix dimensions must support the data request
        if self.rows.is_empty() || self.rows[0].len() < column_index + 1 {
            return Err(MatrixError::TryFromMatrixError(format!(
                "the Matrix dimensions {:?} do not support the request.",
                self.dim()
            )));
        }

        // validate the length of the column vs. the length of the requested Vector
        // the Matrix column vector length must be identical to the requested Vector length
        let col_vec = self.get_column_vec(column_index).unwrap();
        if col_vec.len() != N {
            return Err(MatrixError::TryFromMatrixError(format!(
                "the requested Matrix column vector length ({}) does not match the requested Vector length ({})",
                col_vec.len(),
                N
            )));
        }

        match Vector::<T, N>::try_from(col_vec) {
            Ok(v) => Ok(v),
            Err(_) => panic!(),
        }
    }

    /// Returns a [`Matrix`] row [`Vector`] given the zero-based row index parameter
    /// `row_index`.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::TryFromMatrixError`] when:
    ///
    /// - the `row_index` request is out of bounds
    /// - the [`Matrix`] is empty
    ///
    /// TODO:
    pub fn get_row_vector<'a, const N: usize>(
        &'a self,
        row_index: usize,
    ) -> Result<Vector<T, N>, MatrixError>
    where
        Vector<T, N>: TryFrom<&'a Vec<T>> + 'a,
    {
        // validate the Matrix shape and request
        // the matrix dimensions must support the data request
        if self.rows.is_empty() || self.rows.len() < row_index + 1 {
            return Err(MatrixError::TryFromMatrixError(format!(
                "the Matrix dimensions {:?} do not support the request.",
                self.dim()
            )));
        }

        // validate the length of the row vs. the length of the requested Vector
        // the Matrix row vector length must be identical to the requested Vector length
        let row_vec = &self.rows[row_index];
        if row_vec.len() != N {
            return Err(MatrixError::TryFromMatrixError(format!(
                "the requested Matrix row vector length ({}) does not match the requested Vector length ({})",
                row_vec.len(),
                N
            )));
        }

        match Vector::<T, N>::try_from(row_vec) {
            Ok(v) => Ok(v),
            Err(_) => panic!(),
        }
    }

    /// Returns `true` if the [`Matrix`] is empty.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Returns `Ok(())` if lhs and rhs matrix shapes validate on:
    ///
    /// - lhs number of columns == rhs number of rows
    ///
    /// This criterion must be met for matrix multiplication.
    pub fn validate_shape_lhs_n_eq_rhs_m(&self, rhs: &Matrix<T>) -> Result<(), MatrixError> {
        if self.rows[0].len() != rhs.rows.len() {
            return Err(MatrixError::InvalidMatrixShapeError(format!(
                "the lhs matrix column number ({}) must match the rhs matrix row number ({}) with this operation.",
                self.rows[0].len(),
                rhs.rows.len()
            )));
        }
        Ok(())
    }

    /// Returns `Ok(())` if lhs and rhs matrix shapes validate on:
    ///
    /// - lhs M, N dimensions are the same as rhs M, N dimensions
    ///
    /// This criterion must be met for matrix addition and subtraction.
    pub fn validate_shape_lhs_mn_eq_rhs_mn(&self, rhs: &Matrix<T>) -> Result<(), MatrixError> {
        if self.dim() != rhs.dim() {
            return Err(MatrixError::InvalidMatrixShapeError(format!(
                "the matrix dimensions must be the same with this operation. Received dimensions {:?} and {:?}",
                self.dim(),
                rhs.dim(),
            )));
        }
        Ok(())
    }

    // fn dot(&self, rhs: &Self) -> Self {
    //     //

    // }
}

// ================================
//
// Index and IndexMut trait impl
//
// ================================

impl<I, T> Index<I> for Matrix<T>
where
    I: SliceIndex<[Vec<T>]>,
    T: Num + Copy + Sync + Send,
{
    type Output = I::Output;
    /// Returns a [`Matrix`] row [`Vector`] by zero-based index.
    fn index(&self, i: I) -> &Self::Output {
        &self.rows[i]
    }
}

impl<T> IndexMut<usize> for Matrix<T>
where
    T: Num + Copy + Sync + Send,
{
    /// Returns a mutable [`Matrix`] row [`Vec`] by zero-based index.
    fn index_mut(&mut self, i: usize) -> &mut Vec<T> {
        &mut self.rows[i]
    }
}

// ================================
//
// Operator overloads
//
// ================================

// impl<T, const M: usize, const N: usize> Mul<Vector<T, M>> for Matrix<T>
// where
//     T: Num + Copy + Sync + Send + Default + Debug,
// {
//     type Output = Vector<T, N>;

//     /// Binary multiplication operator overload implementation for matrix : vector
//     /// multiplication.
//     fn mul(self, rhs: Vector<T, M>) -> Self::Output {
//         let mut a = [T::zero(); N];

//         for (i, rhs_scalar) in rhs.enumerate() {
//             let col_vec = self.get_column_vec(i + 1).unwrap();
//             for (j, x) in col_vec.iter().enumerate() {
//                 a[j] = a[j] + (*x * *rhs_scalar);
//             }
//         }

//         // return the Vector
//         Vector { components: a }
//     }
// }

// impl<T> Mul<Matrix<T>> for Matrix<T>
// where
//     T: Num + Copy + Sync + Send + Default + Debug,
// {
//     type Output = Matrix<T>;

//     /// Binary multiplication operator overload implementation for matrix : matrix
//     /// multiplication.
//     fn mul(self, rhs: Matrix<T>) -> Self::Output {
//         let mut a = [T::zero(); N];

//         for (i, rhs_scalar) in rhs.enumerate() {
//             let col_vec = self.get_column_vec(i + 1).unwrap();
//             for (j, x) in col_vec.iter().enumerate() {
//                 a[j] = a[j] + (*x * *rhs_scalar);
//             }
//         }

//         // return the Vector
//         Vector { components: a }
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{vector, Vector};
    #[allow(unused_imports)]
    use approx::{assert_relative_eq, assert_relative_ne};
    use num::complex::Complex;
    #[allow(unused_imports)]
    use pretty_assertions::{assert_eq, assert_ne};

    // ================================
    //
    // Initialization tests
    //
    // ================================

    #[test]
    fn matrix_method_new_i32() {
        let m: Matrix<i32> = Matrix::new(2, 3);

        assert_eq!(m.rows.len(), 2);

        assert_eq!(m.rows[0].len(), 3);
        assert_eq!(m.rows[0][0], 0_i32);
        assert_eq!(m.rows[0][1], 0_i32);
        assert_eq!(m.rows[0][2], 0_i32);

        assert_eq!(m.rows[1].len(), 3);
        assert_eq!(m.rows[1][0], 0_i32);
        assert_eq!(m.rows[1][1], 0_i32);
        assert_eq!(m.rows[1][2], 0_i32);

        // additional one off tests of edge cases
        // (only tested in this i32 function)
        let m: Matrix<i32> = Matrix::new(0, 3);

        assert_eq!(m.rows.len(), 0);

        let m: Matrix<i32> = Matrix::new(10, 0);

        assert_eq!(m.rows.len(), 0);
    }

    #[test]
    fn matrix_method_new_f64() {
        let m: Matrix<f64> = Matrix::new(2, 3);

        assert_eq!(m.rows.len(), 2);

        assert_eq!(m.rows[0].len(), 3);
        assert_relative_eq!(m.rows[0][0], 0.0_f64);
        assert_relative_eq!(m.rows[0][1], 0.0_f64);
        assert_relative_eq!(m.rows[0][2], 0.0_f64);

        assert_eq!(m.rows[1].len(), 3);
        assert_relative_eq!(m.rows[1][0], 0.0_f64);
        assert_relative_eq!(m.rows[1][1], 0.0_f64);
        assert_relative_eq!(m.rows[1][2], 0.0_f64);
    }

    #[test]
    fn matrix_method_new_complex_i32() {
        let m: Matrix<Complex<i32>> = Matrix::new(2, 3);

        assert_eq!(m.rows.len(), 2);

        assert_eq!(m.rows[0].len(), 3);
        assert_eq!(m.rows[0][0].re, 0_i32);
        assert_eq!(m.rows[0][0].im, 0_i32);
        assert_eq!(m.rows[0][1].re, 0_i32);
        assert_eq!(m.rows[0][1].im, 0_i32);
        assert_eq!(m.rows[0][2].re, 0_i32);
        assert_eq!(m.rows[0][2].im, 0_i32);

        assert_eq!(m.rows[1].len(), 3);
        assert_eq!(m.rows[1][0].re, 0_i32);
        assert_eq!(m.rows[1][0].im, 0_i32);
        assert_eq!(m.rows[1][1].re, 0_i32);
        assert_eq!(m.rows[1][1].im, 0_i32);
        assert_eq!(m.rows[1][2].re, 0_i32);
        assert_eq!(m.rows[1][2].im, 0_i32);
    }

    #[test]
    fn matrix_method_new_complex_f64() {
        let m: Matrix<Complex<f64>> = Matrix::new(2, 3);

        assert_eq!(m.rows.len(), 2);

        assert_eq!(m.rows[0].len(), 3);
        assert_relative_eq!(m.rows[0][0].re, 0.0_f64);
        assert_relative_eq!(m.rows[0][0].im, 0.0_f64);
        assert_relative_eq!(m.rows[0][1].re, 0.0_f64);
        assert_relative_eq!(m.rows[0][1].im, 0.0_f64);
        assert_relative_eq!(m.rows[0][2].re, 0.0_f64);
        assert_relative_eq!(m.rows[0][2].im, 0.0_f64);

        assert_eq!(m.rows[1].len(), 3);
        assert_relative_eq!(m.rows[1][0].re, 0.0_f64);
        assert_relative_eq!(m.rows[1][0].im, 0.0_f64);
        assert_relative_eq!(m.rows[1][1].re, 0.0_f64);
        assert_relative_eq!(m.rows[1][1].im, 0.0_f64);
        assert_relative_eq!(m.rows[1][2].re, 0.0_f64);
        assert_relative_eq!(m.rows[1][2].im, 0.0_f64);
    }

    // ================================
    //
    // len_* method tests
    //
    // ================================

    #[test]
    fn matrix_method_dim() {
        let m: Matrix<i32> = Matrix::new(10, 3);

        assert_eq!(m.dim(), (10, 3));

        let m: Matrix<i32> = Matrix::new(0, 3);

        assert_eq!(m.dim(), (0, 0));

        let m: Matrix<i32> = Matrix::new(10, 0);

        assert_eq!(m.dim(), (0, 0));
    }

    #[test]
    fn matrix_method_dim_column() {
        let m: Matrix<i32> = Matrix::new(10, 3);

        assert_eq!(m.dim_column(), 3);

        let m: Matrix<i32> = Matrix::new(0, 3);

        assert_eq!(m.dim_column(), 0);

        let m: Matrix<i32> = Matrix::new(10, 0);

        assert_eq!(m.dim_column(), 0);
    }

    #[test]
    fn matrix_method_dim_row() {
        let m: Matrix<i32> = Matrix::new(10, 3);

        assert_eq!(m.dim_row(), 10);

        let m: Matrix<i32> = Matrix::new(10, 0);

        assert_eq!(m.dim_row(), 0);

        let m: Matrix<i32> = Matrix::new(0, 10);

        assert_eq!(m.dim_row(), 0);
    }

    #[test]
    fn matrix_method_from_rows_i32() {
        let m: Matrix<i32> = Matrix::from_rows(&[vec![1, 2, 3], vec![4, 5, 6]]);

        assert_eq!(m.rows.len(), 2);

        assert_eq!(m.rows[0].len(), 3);
        assert_eq!(m.rows[0][0], 1_i32);
        assert_eq!(m.rows[0][1], 2_i32);
        assert_eq!(m.rows[0][2], 3_i32);

        assert_eq!(m.rows[1].len(), 3);
        assert_eq!(m.rows[1][0], 4_i32);
        assert_eq!(m.rows[1][1], 5_i32);
        assert_eq!(m.rows[1][2], 6_i32);
    }

    #[test]
    fn matrix_method_from_rows_f64() {
        let m: Matrix<f64> = Matrix::from_rows(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

        assert_eq!(m.rows.len(), 2);

        assert_eq!(m.rows[0].len(), 3);
        assert_relative_eq!(m.rows[0][0], 1.0_f64);
        assert_relative_eq!(m.rows[0][1], 2.0_f64);
        assert_relative_eq!(m.rows[0][2], 3.0_f64);

        assert_eq!(m.rows[1].len(), 3);
        assert_relative_eq!(m.rows[1][0], 4.0_f64);
        assert_relative_eq!(m.rows[1][1], 5.0_f64);
        assert_relative_eq!(m.rows[1][2], 6.0_f64);
    }

    #[test]
    fn matrix_method_from_rows_complex_i32() {
        let m: Matrix<Complex<i32>> = Matrix::from_rows(&[
            vec![Complex::new(1, 2), Complex::new(3, 4)],
            vec![Complex::new(5, 6), Complex::new(7, 8)],
        ]);

        assert_eq!(m.rows.len(), 2);

        assert_eq!(m.rows[0].len(), 2);
        assert_eq!(m.rows[0][0].re, 1_i32);
        assert_eq!(m.rows[0][0].im, 2_i32);
        assert_eq!(m.rows[0][1].re, 3_i32);
        assert_eq!(m.rows[0][1].im, 4_i32);

        assert_eq!(m.rows[1].len(), 2);
        assert_eq!(m.rows[1][0].re, 5_i32);
        assert_eq!(m.rows[1][0].im, 6_i32);
        assert_eq!(m.rows[1][1].re, 7_i32);
        assert_eq!(m.rows[1][1].im, 8_i32);
    }

    #[test]
    fn matrix_method_from_rows_complex_f64() {
        let m: Matrix<Complex<f64>> = Matrix::from_rows(&[
            vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
            vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)],
        ]);

        assert_eq!(m.rows.len(), 2);

        assert_eq!(m.rows[0].len(), 2);
        assert_relative_eq!(m.rows[0][0].re, 1.0);
        assert_relative_eq!(m.rows[0][0].im, 2.0);
        assert_relative_eq!(m.rows[0][1].re, 3.0);
        assert_relative_eq!(m.rows[0][1].im, 4.0);

        assert_eq!(m.rows[1].len(), 2);
        assert_relative_eq!(m.rows[1][0].re, 5.0);
        assert_relative_eq!(m.rows[1][0].im, 6.0);
        assert_relative_eq!(m.rows[1][1].re, 7.0);
        assert_relative_eq!(m.rows[1][1].im, 8.0);
    }

    // ================================
    //
    // get_column_vec method tests
    //
    // ================================

    #[test]
    fn matrix_method_get_column_vec() {
        let m: Matrix<i32> = Matrix::from_rows(&[vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);

        assert_eq!(m.get_column_vec(0).unwrap(), vec![1, 4, 7]);
        assert_eq!(m.get_column_vec(1).unwrap(), vec![2, 5, 8]);
        assert_eq!(m.get_column_vec(2).unwrap(), vec![3, 6, 9]);
    }

    #[test]
    fn matrix_method_get_column_vec_panics_on_column_out_of_bounds_high() {
        let m: Matrix<i32> = Matrix::from_rows(&vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);

        assert!(m.get_column_vec(3).is_none());
    }

    // ================================
    //
    // get_row_vec method tests
    //
    // ================================

    #[test]
    fn matrix_method_get_row_vec() {
        let m: Matrix<i32> = Matrix::from_rows(&[vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);

        assert_eq!(m.get_row_vec(0).unwrap(), vec![1, 2, 3]);
        assert_eq!(m.get_row_vec(1).unwrap(), vec![4, 5, 6]);
        assert_eq!(m.get_row_vec(2).unwrap(), vec![7, 8, 9]);
    }

    #[test]
    fn matrix_method_get_row_vec_panics_on_column_out_of_bounds_high() {
        let m: Matrix<i32> = Matrix::from_rows(&vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);

        assert!(m.get_row_vec(3).is_none());
    }

    // ================================
    //
    // get_column_vector method tests
    //
    // ================================

    #[test]
    fn matrix_method_get_column_vector() {
        let m: Matrix<i32> = Matrix::from_rows(&[vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);

        assert_eq!(m.get_column_vector(0).unwrap(), vector![1, 4, 7]);
        assert_eq!(m.get_column_vector(1).unwrap(), vector![2, 5, 8]);
        assert_eq!(m.get_column_vector(2).unwrap(), vector![3, 6, 9]);
    }

    #[test]
    fn matrix_method_get_column_vector_out_of_bounds_high() {
        let m: Matrix<i32> = Matrix::from_rows(&[vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);

        let res: Result<Vector<i32, 3>, MatrixError> = m.get_column_vector(3);

        assert!(res.is_err());
        assert!(matches!(res, Err(MatrixError::TryFromMatrixError(_))));
    }

    #[test]
    fn matrix_method_get_column_vector_empty_matrix() {
        let m: Matrix<i32> = Matrix::from_rows(&[]);

        let res: Result<Vector<i32, 3>, MatrixError> = m.get_column_vector(1);

        assert!(res.is_err());
        assert!(matches!(res, Err(MatrixError::TryFromMatrixError(_))));
    }

    // ================================
    //
    // get_row_vector method tests
    //
    // ================================

    #[test]
    fn matrix_method_get_row_vector() {
        let m: Matrix<i32> = Matrix::from_rows(&[vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);

        assert_eq!(m.get_row_vector(0).unwrap(), vector![1, 2, 3]);
        assert_eq!(m.get_row_vector(1).unwrap(), vector![4, 5, 6]);
        assert_eq!(m.get_row_vector(2).unwrap(), vector![7, 8, 9]);
    }

    #[test]
    fn matrix_method_get_row_vector_out_of_bounds_high() {
        let m: Matrix<i32> = Matrix::from_rows(&[vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);

        let res: Result<Vector<i32, 3>, MatrixError> = m.get_row_vector(3);

        assert!(res.is_err());
        assert!(matches!(res, Err(MatrixError::TryFromMatrixError(_))));
    }

    #[test]
    fn matrix_method_get_row_vector_empty_matrix() {
        let m: Matrix<i32> = Matrix::from_rows(&[]);

        let res: Result<Vector<i32, 3>, MatrixError> = m.get_row_vector(1);

        assert!(res.is_err());
        assert!(matches!(res, Err(MatrixError::TryFromMatrixError(_))));
    }

    // ===================================
    //
    // validate_shape method tests
    //
    // ===================================

    #[test]
    fn matrix_method_validate_shape_lhs_n_eq_rhs_m() {
        let m1: Matrix<i32> = Matrix::new(3, 2);
        let m2: Matrix<i32> = Matrix::new(2, 3);
        let m3: Matrix<i32> = Matrix::new(2, 5);
        let m4: Matrix<i32> = Matrix::new(3, 5);

        assert!(m1.validate_shape_lhs_n_eq_rhs_m(&m2).is_ok());
        assert!(m2.validate_shape_lhs_n_eq_rhs_m(&m1).is_ok());

        assert!(m1.validate_shape_lhs_n_eq_rhs_m(&m3).is_ok());
        assert!(m3.validate_shape_lhs_n_eq_rhs_m(&m1).is_err());
        assert!(matches!(
            m3.validate_shape_lhs_n_eq_rhs_m(&m1),
            Err(MatrixError::InvalidMatrixShapeError(_))
        ));

        assert!(m4.validate_shape_lhs_n_eq_rhs_m(&m1).is_err());
        assert!(matches!(
            m4.validate_shape_lhs_n_eq_rhs_m(&m1),
            Err(MatrixError::InvalidMatrixShapeError(_))
        ));
        assert!(m1.validate_shape_lhs_n_eq_rhs_m(&m4).is_err());
        assert!(matches!(
            m1.validate_shape_lhs_n_eq_rhs_m(&m4),
            Err(MatrixError::InvalidMatrixShapeError(_))
        ));
    }

    #[test]
    fn matrix_method_validate_shape_lhs_mn_eq_rhs_mn() {
        let m1: Matrix<i32> = Matrix::new(3, 2);
        let m2: Matrix<i32> = Matrix::new(3, 2);
        let m3: Matrix<i32> = Matrix::new(2, 3);
        let m4: Matrix<i32> = Matrix::new(4, 4);

        assert!(m1.validate_shape_lhs_mn_eq_rhs_mn(&m2).is_ok());

        assert!(m1.validate_shape_lhs_mn_eq_rhs_mn(&m3).is_err());
        assert!(matches!(
            m1.validate_shape_lhs_mn_eq_rhs_mn(&m3),
            Err(MatrixError::InvalidMatrixShapeError(_))
        ));
        assert!(m1.validate_shape_lhs_mn_eq_rhs_mn(&m4).is_err());
        assert!(matches!(
            m1.validate_shape_lhs_mn_eq_rhs_mn(&m4),
            Err(MatrixError::InvalidMatrixShapeError(_))
        ));
    }

    // ================================
    //
    // Index and IndexMut trait tests
    //
    // ================================

    #[test]
    fn matrix_trait_index_i32() {
        let rows = [vec![0_i32, 1, 2], vec![3, 4, 5]];
        let m = Matrix::from_rows(&rows);

        assert_eq!(m.rows.len(), 2);
        assert_eq!(m.rows[0].len(), 3);
        assert_eq!(m[0].len(), 3);
        assert_eq!(m.rows[1].len(), 3);
        assert_eq!(m[1].len(), 3);

        assert_eq!(m[0][0], 0);
        assert_eq!(m[0][1], 1);
        assert_eq!(m[0][2], 2);

        assert_eq!(m[1][0], 3);
        assert_eq!(m[1][1], 4);
        assert_eq!(m[1][2], 5);
    }

    #[test]
    fn matrix_trait_index_f64() {
        let rows = [vec![0.0_f64, 1.0, 2.0], vec![3.0, 4.0, 5.0]];
        let m = Matrix::from_rows(&rows);

        assert_eq!(m.rows.len(), 2);
        assert_eq!(m.rows[0].len(), 3);
        assert_eq!(m[0].len(), 3);
        assert_eq!(m.rows[1].len(), 3);
        assert_eq!(m[1].len(), 3);

        assert_relative_eq!(m[0][0], 0.0);
        assert_relative_eq!(m[0][1], 1.0);
        assert_relative_eq!(m[0][2], 2.0);

        assert_relative_eq!(m[1][0], 3.0);
        assert_relative_eq!(m[1][1], 4.0);
        assert_relative_eq!(m[1][2], 5.0);
    }

    #[test]
    fn matrix_trait_index_complex_i32() {
        let rows = vec![
            vec![Complex::new(0_i32, 1), Complex::new(2, 3), Complex::new(4, 5)],
            vec![Complex::new(6, 7), Complex::new(8, 9), Complex::new(10, 11)],
        ];
        let m = Matrix::from_rows(&rows);

        assert_eq!(m.rows.len(), 2);
        assert_eq!(m.rows[0].len(), 3);
        assert_eq!(m[0].len(), 3);
        assert_eq!(m.rows[1].len(), 3);
        assert_eq!(m[1].len(), 3);

        assert_eq!(m[0][0], Complex::new(0, 1));
        assert_eq!(m[0][1], Complex::new(2, 3));
        assert_eq!(m[0][2], Complex::new(4, 5));

        assert_eq!(m[1][0], Complex::new(6, 7));
        assert_eq!(m[1][1], Complex::new(8, 9));
        assert_eq!(m[1][2], Complex::new(10, 11));
    }

    #[test]
    fn matrix_trait_index_complex_f64() {
        let rows = vec![
            vec![Complex::new(0.0_f64, 1.0), Complex::new(2.0, 3.0), Complex::new(4.0, 5.0)],
            vec![Complex::new(6.0, 7.0), Complex::new(8.0, 9.0), Complex::new(10.0, 11.0)],
        ];
        let m = Matrix::from_rows(&rows);

        assert_eq!(m.rows.len(), 2);
        assert_eq!(m.rows[0].len(), 3);
        assert_eq!(m[0].len(), 3);
        assert_eq!(m.rows[1].len(), 3);
        assert_eq!(m[1].len(), 3);

        assert_eq!(m[0][0], Complex::new(0.0_f64, 1.0));
        assert_eq!(m[0][1], Complex::new(2.0_f64, 3.0));
        assert_eq!(m[0][2], Complex::new(4.0_f64, 5.0));

        assert_eq!(m[1][0], Complex::new(6.0_f64, 7.0));
        assert_eq!(m[1][1], Complex::new(8.0_f64, 9.0));
        assert_eq!(m[1][2], Complex::new(10.0_f64, 11.0));
    }

    #[test]
    fn matrix_trait_index_mut_i32() {
        let rows = [vec![0_i32, 1, 2], vec![3, 4, 5]];
        let mut m = Matrix::from_rows(&rows);

        assert_eq!(m.rows.len(), 2);
        assert_eq!(m.rows[0].len(), 3);
        assert_eq!(m[0].len(), 3);
        assert_eq!(m.rows[1].len(), 3);
        assert_eq!(m[1].len(), 3);

        // modify
        m[1] = Vec::from([6, 7, 8]);

        assert_eq!(m[0][0], 0);
        assert_eq!(m[0][1], 1);
        assert_eq!(m[0][2], 2);

        assert_eq!(m[1][0], 6);
        assert_eq!(m[1][1], 7);
        assert_eq!(m[1][2], 8);

        // modify
        m[0][0] = 3;
        m[0][1] = 4;
        m[0][2] = 5;

        assert_eq!(m[0][0], 3);
        assert_eq!(m[0][1], 4);
        assert_eq!(m[0][2], 5);

        assert_eq!(m[1][0], 6);
        assert_eq!(m[1][1], 7);
        assert_eq!(m[1][2], 8);
    }

    // TODO: add rest of tests for IndexMut beginning with f64 types

    // ================================
    //
    // Operator overload tests
    //
    // ================================

    // #[test]
    // fn matrix_trait_mul_operator_matrix_vector() {
    //     let mut m: Matrix<i32, 2> = Matrix::new(2);
    //     m.rows[0][0] = 2;
    //     m.rows[0][1] = 3;
    //     m.rows[1][0] = -1;
    //     m.rows[1][1] = 5;
    //     let v: Vector<i32, 2> = Vector::from([2, 1]);

    //     assert_eq!(m * v, Vector::from([7, 3]));
    // }
}
