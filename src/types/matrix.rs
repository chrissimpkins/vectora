//! Matrix types

use std::fmt::Debug;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};
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
    ///
    /// # Examples
    ///
    /// // TODO:
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Returns the additive inverse [`Matrix`].
    ///
    /// # Examples
    ///
    /// // TODO:
    pub fn additive_inverse(&self) -> Self {
        -self
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

    // ================================
    //
    // Private methods
    //
    // ================================

    // Returns a `<Vec<Vec<T>>` row vector data collection following matrix : matrix addition
    // of M-by-N matrices with the same row and column dimensions.
    fn impl_matrix_matrix_add(&self, rhs: &Matrix<T>) -> Vec<Vec<T>> {
        if self.dim() != rhs.dim() {
            panic!("the lhs Matrix and rhs Matrix must have the same row and column dimensions to support matrix addition.")
        }

        let mut rows_collection = Vec::with_capacity(self.rows.len());

        for (lhs_vec, rhs_vec) in self.rows.iter().zip(rhs.rows.iter()) {
            rows_collection.push(lhs_vec.iter().zip(rhs_vec).map(|(x, y)| *x + *y).collect());
        }

        rows_collection
    }

    // Returns a `<Vec<Vec<T>>` row vector data collection following matrix scalar multiplication
    fn impl_matrix_scalar_mul(&self, rhs: &T) -> Vec<Vec<T>> {
        let mut rows_collection = Vec::with_capacity(self.rows.len());

        for lhs_vec in self.rows.iter() {
            rows_collection.push(lhs_vec.iter().map(|x| *x * *rhs).collect());
        }

        rows_collection
    }

    // Returns a `<Vec<Vec<T>>` row vector data collection following matrix : matrix subtraction
    // of M-by-N matrices with the same row and column dimensions.
    fn impl_matrix_matrix_sub(&self, rhs: &Matrix<T>) -> Vec<Vec<T>> {
        if self.dim() != rhs.dim() {
            panic!("the lhs Matrix and rhs Matrix must have the same row and column dimensions to support matrix subtraction.")
        }

        let mut rows_collection = Vec::with_capacity(self.rows.len());

        for (lhs_vec, rhs_vec) in self.rows.iter().zip(rhs.rows.iter()) {
            rows_collection.push(lhs_vec.iter().zip(rhs_vec).map(|(x, y)| *x - *y).collect());
        }

        rows_collection
    }
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

// Unary

// ================================
//
// Neg trait
//
// ================================

impl<T> Neg for Matrix<T>
where
    T: Num + Copy + Default + Sync + Send,
{
    type Output = Self;

    /// Unary negation operator overload implementation.
    fn neg(self) -> Self::Output {
        let mut rows_collection = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            let v: Vec<T> = row.iter().map(|x| T::zero() - (*x)).collect();
            rows_collection.push(v);
        }

        Self { rows: rows_collection }
    }
}

impl<T> Neg for &Matrix<T>
where
    T: Num + Copy + Default + Sync + Send,
{
    type Output = Matrix<T>;

    /// Unary negation operator overload implementation.
    fn neg(self) -> Self::Output {
        let mut rows_collection = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            let v: Vec<T> = row.iter().map(|x| T::zero() - (*x)).collect();
            rows_collection.push(v);
        }

        Matrix::from_rows(&rows_collection)
    }
}

// Binary

// ================================
//
// Add trait
//
// ================================

impl<T> Add for Matrix<T>
where
    T: Num + Copy + Sync + Send + Default,
{
    type Output = Self;

    /// Binary add operator overload implementation for matrix : matrix addition with
    /// owned [`Matrix`].
    fn add(self, rhs: Self) -> Self::Output {
        Self { rows: self.impl_matrix_matrix_add(&rhs) }
    }
}

impl<T> Add for &Matrix<T>
where
    T: Num + Copy + Sync + Send + Default,
{
    type Output = Matrix<T>;

    /// Binary add operator overload implementation for matrix : matrix addition with
    /// [`Matrix`] references.
    fn add(self, rhs: Self) -> Self::Output {
        Matrix::from_rows(&self.impl_matrix_matrix_add(rhs))
    }
}

// ================================
//
// Sub trait
//
// ================================

impl<T> Sub for Matrix<T>
where
    T: Num + Copy + Sync + Send + Default,
{
    type Output = Self;

    /// Binary subtraction operator overload implementation for matrix : matrix subtraction with
    /// owned [`Matrix`].
    fn sub(self, rhs: Self) -> Self::Output {
        Self { rows: self.impl_matrix_matrix_sub(&rhs) }
    }
}

impl<T> Sub for &Matrix<T>
where
    T: Num + Copy + Sync + Send + Default,
{
    type Output = Matrix<T>;

    /// Binary subtraction operator overload implementation for matrix : matrix subtraction with
    /// [`Matrix`] references.
    fn sub(self, rhs: Self) -> Self::Output {
        Matrix::from_rows(&self.impl_matrix_matrix_sub(rhs))
    }
}

// ================================
//
// Mul trait
//
// ================================

// Scalar multiplication

impl<T> Mul<T> for Matrix<T>
where
    T: Num + Copy + Sync + Send + Default,
{
    type Output = Self;

    /// Binary multiplication operator overload implementation for matrix
    /// scalar multiplication with owned [`Matrix`].
    fn mul(self, rhs: T) -> Self::Output {
        Self { rows: self.impl_matrix_scalar_mul(&rhs) }
    }
}

impl<T> Mul<T> for &Matrix<T>
where
    T: Num + Copy + Sync + Send + Default,
{
    type Output = Matrix<T>;

    /// Binary multiplication operator overload implementation for matrix
    /// scalar multiplication with [`Matrix`] references.
    fn mul(self, rhs: T) -> Self::Output {
        Matrix::from_rows(&self.impl_matrix_scalar_mul(&rhs))
    }
}

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
    // additive_inverse method tests
    //
    // ================================

    #[test]
    fn matrix_method_additive_inverse_i32() {
        let rows = [vec![0_i32, -1, 2], vec![3, -4, 5]];
        let rows_expected_neg = [vec![0, 1, -2], vec![-3, 4, -5]];

        let m = Matrix::from_rows(&rows);

        // the method should borrow, it does not move contents
        assert_eq!(m.additive_inverse().rows, rows_expected_neg);
        // and can be used again
        let _ = m.additive_inverse();
    }

    #[test]
    fn matrix_method_additive_inverse_f64() {
        let rows = [vec![0.0_f64, -1.0, 2.0], vec![3.0_f64, -4.0, 5.0]];
        let rows_expected_neg = [vec![0.0_f64, 1.0, -2.0], vec![-3.0_f64, 4.0, -5.0]];

        let m = Matrix::from_rows(&rows);

        assert_relative_eq!(m.additive_inverse()[0][0], rows_expected_neg[0][0]);
        assert_relative_eq!(m.additive_inverse()[0][1], rows_expected_neg[0][1]);
        assert_relative_eq!(m.additive_inverse()[0][2], rows_expected_neg[0][2]);
        assert_relative_eq!(m.additive_inverse()[1][0], rows_expected_neg[1][0]);
        assert_relative_eq!(m.additive_inverse()[1][1], rows_expected_neg[1][1]);
        assert_relative_eq!(m.additive_inverse()[1][2], rows_expected_neg[1][2]);
    }

    #[test]
    fn matrix_method_additive_inverse_complex_i32() {
        let rows = [
            vec![Complex::new(0_i32, -1), Complex::new(-2_i32, 3)],
            vec![Complex::new(4_i32, -5), Complex::new(-6_i32, 7)],
        ];
        let rows_expected_neg = [
            vec![Complex::new(-0_i32, 1), Complex::new(2_i32, -3)],
            vec![Complex::new(-4_i32, 5), Complex::new(6_i32, -7)],
        ];

        let m = Matrix::from_rows(&rows);

        assert_eq!(m.additive_inverse().rows, rows_expected_neg);
    }

    #[test]
    fn matrix_method_additive_inverse_complex_f64() {
        let rows = [
            vec![Complex::new(0.0_f64, -1.0), Complex::new(-2.0_f64, 3.0)],
            vec![Complex::new(4.0_f64, -5.0), Complex::new(-6.0_f64, 7.0)],
        ];
        let rows_expected_neg = [
            vec![Complex::new(-0.0_f64, 1.0), Complex::new(2.0_f64, -3.0)],
            vec![Complex::new(-4.0_f64, 5.0), Complex::new(6.0_f64, -7.0)],
        ];

        let m = Matrix::from_rows(&rows);

        assert_relative_eq!(m.additive_inverse()[0][0].re, rows_expected_neg[0][0].re);
        assert_relative_eq!(m.additive_inverse()[0][0].im, rows_expected_neg[0][0].im);
        assert_relative_eq!(m.additive_inverse()[0][1].re, rows_expected_neg[0][1].re);
        assert_relative_eq!(m.additive_inverse()[0][1].im, rows_expected_neg[0][1].im);
        assert_relative_eq!(m.additive_inverse()[1][0].re, rows_expected_neg[1][0].re);
        assert_relative_eq!(m.additive_inverse()[1][0].im, rows_expected_neg[1][0].im);
        assert_relative_eq!(m.additive_inverse()[1][1].re, rows_expected_neg[1][1].re);
        assert_relative_eq!(m.additive_inverse()[1][1].im, rows_expected_neg[1][1].im);
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

    // ================================
    //
    // Neg trait tests
    //
    // ================================

    #[test]
    fn matrix_trait_neg_i32() {
        let rows = [vec![0_i32, -1, 2], vec![3, -4, 5]];
        let rows_expected_neg = [vec![0, 1, -2], vec![-3, 4, -5]];

        let m = Matrix::from_rows(&rows);

        // borrow does not move contents
        assert_eq!((-&m).rows, rows_expected_neg);
        // but unary neg on owned type does, `m` cannot be used again after unary neg here
        assert_eq!((-m).rows, rows_expected_neg);
    }

    #[test]
    fn matrix_trait_neg_f64() {
        let rows = [vec![0.0_f64, -1.0, 2.0], vec![3.0_f64, -4.0, 5.0]];
        let rows_expected_neg = [vec![0.0_f64, 1.0, -2.0], vec![-3.0_f64, 4.0, -5.0]];

        let m = Matrix::from_rows(&rows);

        let neg_m = -m;

        assert_relative_eq!(neg_m[0][0], rows_expected_neg[0][0]);
        assert_relative_eq!(neg_m[0][1], rows_expected_neg[0][1]);
        assert_relative_eq!(neg_m[0][2], rows_expected_neg[0][2]);
        assert_relative_eq!(neg_m[1][0], rows_expected_neg[1][0]);
        assert_relative_eq!(neg_m[1][1], rows_expected_neg[1][1]);
        assert_relative_eq!(neg_m[1][2], rows_expected_neg[1][2]);
    }

    #[test]
    fn matrix_trait_neg_complex_i32() {
        let rows = [
            vec![Complex::new(0_i32, -1), Complex::new(-2_i32, 3)],
            vec![Complex::new(4_i32, -5), Complex::new(-6_i32, 7)],
        ];
        let rows_expected_neg = [
            vec![Complex::new(-0_i32, 1), Complex::new(2_i32, -3)],
            vec![Complex::new(-4_i32, 5), Complex::new(6_i32, -7)],
        ];

        let m = Matrix::from_rows(&rows);

        assert_eq!((-m).rows, rows_expected_neg);
    }

    #[test]
    fn matrix_trait_neg_complex_f64() {
        let rows = [
            vec![Complex::new(0.0_f64, -1.0), Complex::new(-2.0_f64, 3.0)],
            vec![Complex::new(4.0_f64, -5.0), Complex::new(-6.0_f64, 7.0)],
        ];
        let rows_expected_neg = [
            vec![Complex::new(-0.0_f64, 1.0), Complex::new(2.0_f64, -3.0)],
            vec![Complex::new(-4.0_f64, 5.0), Complex::new(6.0_f64, -7.0)],
        ];

        let m = Matrix::from_rows(&rows);

        let neg_m = -m;

        assert_relative_eq!(neg_m[0][0].re, rows_expected_neg[0][0].re);
        assert_relative_eq!(neg_m[0][0].im, rows_expected_neg[0][0].im);
        assert_relative_eq!(neg_m[0][1].re, rows_expected_neg[0][1].re);
        assert_relative_eq!(neg_m[0][1].im, rows_expected_neg[0][1].im);
        assert_relative_eq!(neg_m[1][0].re, rows_expected_neg[1][0].re);
        assert_relative_eq!(neg_m[1][0].im, rows_expected_neg[1][0].im);
        assert_relative_eq!(neg_m[1][1].re, rows_expected_neg[1][1].re);
        assert_relative_eq!(neg_m[1][1].im, rows_expected_neg[1][1].im);
    }

    // ================================
    //
    // Add trait tests
    //
    // ================================

    #[test]
    fn matrix_trait_add_matrix_matrix_i32_owned() {
        let rows_1 = [vec![0_i32, -1, 2], vec![3, -4, 5]];
        let rows_2 = [vec![0_i32, 10, -8], vec![5, -12, 3]];

        let m1 = Matrix::from_rows(&rows_1);
        let m2 = Matrix::from_rows(&rows_2);

        let expected_rows_1_plus_2 = [vec![0_i32, 9, -6], vec![8, -16, 8]];

        // can only test this once because move occurs without use of references
        assert_eq!((m1 + m2).rows, expected_rows_1_plus_2);
    }

    #[test]
    fn matrix_trait_add_matrix_matrix_i32_ref() {
        let rows_1 = [vec![0_i32, -1, 2], vec![3, -4, 5]];
        let rows_1_neg = [vec![0, 1, -2], vec![-3, 4, -5]];
        let rows_2 = [vec![0_i32, 10, -8], vec![5, -12, 3]];

        let m1 = Matrix::from_rows(&rows_1);
        let m1_neg = Matrix::from_rows(&rows_1_neg);
        let m2 = Matrix::from_rows(&rows_2);

        let expected_rows_1_plus_2 = [vec![0_i32, 9, -6], vec![8_i32, -16, 8]];
        let expected_rows_1_plus_2_plus_2 = [vec![0_i32, 19, -14], vec![13_i32, -28, 11]];
        let expected_rows_zeroes = [vec![0_i32, 0_i32, 0_i32], vec![0_i32, 0_i32, 0_i32]];

        assert_eq!((&m1 + &m2).rows, expected_rows_1_plus_2);
        assert_eq!((&m2 + &m1).rows, expected_rows_1_plus_2); // commutative
        assert_eq!((&(&m1 + &m2) + &m2).rows, expected_rows_1_plus_2_plus_2); // associative
        assert_eq!((&m1 + &(&m2 + &m2)).rows, expected_rows_1_plus_2_plus_2); // associative
        assert_eq!((&m1 + &Matrix::zero(2, 3)).rows, rows_1); // additive identity (zero matrix) does not change values
        assert_eq!((&m1 + &m1_neg).rows, expected_rows_zeroes); // additive inverse yields zero matrix
        assert_eq!((&m1 + &m1.additive_inverse()).rows, expected_rows_zeroes); // additive inverse tested with method
        assert_eq!((&m1 + &(-&m1)).rows, expected_rows_zeroes); // additive inverse tested with unary neg operator
    }

    #[test]
    fn matrix_trait_add_matrix_matrix_f64_owned() {
        let rows_1 = [vec![0.0_f64, -1.0, 2.0], vec![3.0_f64, -4.0, 5.0]];
        let rows_2 = [vec![0.0_f64, 10.0, -8.0], vec![5.0, -12.0, 3.0]];

        let m1 = Matrix::from_rows(&rows_1);
        let m2 = Matrix::from_rows(&rows_2);

        let expected_rows_1_plus_2 = [vec![0.0_f64, 9.0, -6.0], vec![8.0_f64, -16.0, 8.0]];

        // can only test this once because move occurs without use of references
        assert_eq!((m1 + m2).rows, expected_rows_1_plus_2);
    }

    #[test]
    fn matrix_trait_add_matrix_matrix_f64_ref() {
        let rows_1 = [vec![0.0_f64, -1.0, 2.0], vec![3.0_f64, -4.0, 5.0]];
        let rows_2 = [vec![0.0_f64, 10.0, -8.0], vec![5.0, -12.0, 3.0]];
        let rows_1_neg = [vec![0.0_f64, 1.0, -2.0], vec![-3.0_f64, 4.0, -5.0]];

        let m1 = Matrix::from_rows(&rows_1);
        let m1_neg = Matrix::from_rows(&rows_1_neg);
        let m2 = Matrix::from_rows(&rows_2);

        let expected_rows_1_plus_2 = [vec![0.0_f64, 9.0, -6.0], vec![8.0_f64, -16.0, 8.0]];
        let expected_rows_1_plus_2_plus_2 =
            [vec![0.0_f64, 19.0, -14.0], vec![13.0_f64, -28.0, 11.0]];
        let expected_rows_zeroes = [vec![0.0_f64, 0.0, 0.0], vec![0.0_f64, 0.0, 0.0]];

        assert_eq!((&m1 + &m2).rows, expected_rows_1_plus_2);
        assert_eq!((&m2 + &m1).rows, expected_rows_1_plus_2); // commutative
        assert_eq!((&(&m1 + &m2) + &m2).rows, expected_rows_1_plus_2_plus_2); // associative
        assert_eq!((&m1 + &(&m2 + &m2)).rows, expected_rows_1_plus_2_plus_2); // associative
        assert_eq!((&m1 + &Matrix::zero(2, 3)).rows, rows_1); // additive identity (zero matrix) does not change values
        assert_eq!((&m1 + &m1_neg).rows, expected_rows_zeroes); // additive inverse yields zero matrix
        assert_eq!((&m1 + &m1.additive_inverse()).rows, expected_rows_zeroes); // additive inverse tested with method
        assert_eq!((&m1 + &(-&m1)).rows, expected_rows_zeroes); // additive inverse tested with unary neg operator
    }

    #[test]
    fn matrix_trait_add_matrix_matrix_complex_i32_owned() {
        let rows_1 = [
            vec![Complex::new(0_i32, -1), Complex::new(-1_i32, 0)],
            vec![Complex::new(3_i32, -4), Complex::new(-4_i32, 3)],
        ];
        let rows_2 = [
            vec![Complex::new(0_i32, -10), Complex::new(-10_i32, 0)],
            vec![Complex::new(30_i32, -40), Complex::new(-40_i32, 30)],
        ];

        let m1 = Matrix::from_rows(&rows_1);
        let m2 = Matrix::from_rows(&rows_2);

        let expected_rows_1_plus_2 = [
            vec![Complex::new(0_i32, -11), Complex::new(-11_i32, 0)],
            vec![Complex::new(33_i32, -44), Complex::new(-44_i32, 33)],
        ];

        // can only test this once because move occurs without use of references
        assert_eq!((m1 + m2).rows, expected_rows_1_plus_2);
    }

    #[test]
    fn matrix_trait_add_matrix_matrix_complex_i32_ref() {
        let rows_1 = [
            vec![Complex::new(0_i32, -1), Complex::new(-1_i32, 0)],
            vec![Complex::new(3_i32, -4), Complex::new(-4_i32, 3)],
        ];
        let rows_2 = [
            vec![Complex::new(0_i32, -10), Complex::new(-10_i32, 0)],
            vec![Complex::new(30_i32, -40), Complex::new(-40_i32, 30)],
        ];
        let rows_1_neg = [
            vec![Complex::new(0_i32, 1), Complex::new(1_i32, 0)],
            vec![Complex::new(-3_i32, 4), Complex::new(4_i32, -3)],
        ];

        let m1 = Matrix::from_rows(&rows_1);
        let m2 = Matrix::from_rows(&rows_2);
        let m1_neg = Matrix::from_rows(&rows_1_neg);

        let expected_rows_1_plus_2 = [
            vec![Complex::new(0_i32, -11), Complex::new(-11_i32, 0)],
            vec![Complex::new(33_i32, -44), Complex::new(-44_i32, 33)],
        ];

        let expected_rows_1_plus_2_plus_2 = [
            vec![Complex::new(0_i32, -21), Complex::new(-21_i32, 0)],
            vec![Complex::new(63_i32, -84), Complex::new(-84_i32, 63)],
        ];

        let expected_rows_zeroes = [
            vec![Complex::new(0_i32, 0), Complex::new(0_i32, 0)],
            vec![Complex::new(0_i32, 0), Complex::new(0_i32, 0)],
        ];

        assert_eq!((&m1 + &m2).rows, expected_rows_1_plus_2);
        assert_eq!((&m2 + &m1).rows, expected_rows_1_plus_2); // commutative
        assert_eq!((&(&m1 + &m2) + &m2).rows, expected_rows_1_plus_2_plus_2); // associative
        assert_eq!((&m1 + &(&m2 + &m2)).rows, expected_rows_1_plus_2_plus_2); // associative
        assert_eq!((&m1 + &Matrix::zero(2, 2)).rows, rows_1); // additive identity (zero matrix) does not change values
        assert_eq!((&m1 + &m1_neg).rows, expected_rows_zeroes); // additive inverse yields zero matrix
        assert_eq!((&m1 + &m1.additive_inverse()).rows, expected_rows_zeroes); // additive inverse tested with method
        assert_eq!((&m1 + &(-&m1)).rows, expected_rows_zeroes); // additive inverse tested with unary neg operator
    }

    #[test]
    fn matrix_trait_add_matrix_matrix_complex_f64_owned() {
        let rows_1 = [
            vec![Complex::new(0.0_f64, -1.0), Complex::new(-1.0_f64, 0.0)],
            vec![Complex::new(3.0_f64, -4.0), Complex::new(-4.0_f64, 3.0)],
        ];
        let rows_2 = [
            vec![Complex::new(0.0_f64, -10.0), Complex::new(-10.0_f64, 0.0)],
            vec![Complex::new(30.0_f64, -40.0), Complex::new(-40.0_f64, 30.0)],
        ];

        let m1 = Matrix::from_rows(&rows_1);
        let m2 = Matrix::from_rows(&rows_2);

        let expected_rows_1_plus_2 = [
            vec![Complex::new(0.0_f64, -11.0), Complex::new(-11.0_f64, 0.0)],
            vec![Complex::new(33.0_f64, -44.0), Complex::new(-44.0_f64, 33.0)],
        ];

        // can only test this once because move occurs without use of references
        assert_eq!((m1 + m2).rows, expected_rows_1_plus_2);
    }

    #[test]
    fn matrix_trait_add_matrix_matrix_complex_f64_ref() {
        let rows_1 = [
            vec![Complex::new(0.0_f64, -1.0), Complex::new(-1.0_f64, 0.0)],
            vec![Complex::new(3.0_f64, -4.0), Complex::new(-4.0_f64, 3.0)],
        ];
        let rows_2 = [
            vec![Complex::new(0.0_f64, -10.0), Complex::new(-10.0_f64, 0.0)],
            vec![Complex::new(30.0_f64, -40.0), Complex::new(-40.0_f64, 30.0)],
        ];
        let rows_1_neg = [
            vec![Complex::new(0.0_f64, 1.0), Complex::new(1.0_f64, 0.0)],
            vec![Complex::new(-3.0_f64, 4.0), Complex::new(4.0_f64, -3.0)],
        ];

        let m1 = Matrix::from_rows(&rows_1);
        let m2 = Matrix::from_rows(&rows_2);
        let m1_neg = Matrix::from_rows(&rows_1_neg);

        let expected_rows_1_plus_2 = [
            vec![Complex::new(0.0_f64, -11.0), Complex::new(-11.0_f64, 0.0)],
            vec![Complex::new(33.0_f64, -44.0), Complex::new(-44.0_f64, 33.0)],
        ];

        let expected_rows_1_plus_2_plus_2 = [
            vec![Complex::new(0.0_f64, -21.0), Complex::new(-21.0_f64, 0.0)],
            vec![Complex::new(63.0_f64, -84.0), Complex::new(-84.0_f64, 63.0)],
        ];

        let expected_rows_zeroes = [
            vec![Complex::new(0.0_f64, 0.0), Complex::new(0.0_f64, 0.0)],
            vec![Complex::new(0.0_f64, 0.0), Complex::new(0.0_f64, 0.0)],
        ];

        assert_eq!((&m1 + &m2).rows, expected_rows_1_plus_2);
        assert_eq!((&m2 + &m1).rows, expected_rows_1_plus_2); // commutative
        assert_eq!((&(&m1 + &m2) + &m2).rows, expected_rows_1_plus_2_plus_2); // associative
        assert_eq!((&m1 + &(&m2 + &m2)).rows, expected_rows_1_plus_2_plus_2); // associative
        assert_eq!((&m1 + &Matrix::zero(2, 2)).rows, rows_1); // additive identity (zero matrix) does not change values
        assert_eq!((&m1 + &m1_neg).rows, expected_rows_zeroes); // additive inverse yields zero matrix
        assert_eq!((&m1 + &m1.additive_inverse()).rows, expected_rows_zeroes); // additive inverse tested with method
        assert_eq!((&m1 + &(-&m1)).rows, expected_rows_zeroes); // additive inverse tested with unary neg operator
    }

    #[test]
    #[should_panic]
    fn matrix_trait_add_panics_on_different_row_dim() {
        let m1: Matrix<i32> = Matrix::zero(2, 3);
        let m2: Matrix<i32> = Matrix::zero(3, 3);

        let _ = m1 + m2;
    }

    #[test]
    #[should_panic]
    fn matrix_trait_add_panics_on_different_column_dim() {
        let m1: Matrix<i32> = Matrix::zero(3, 3);
        let m2: Matrix<i32> = Matrix::zero(3, 4);

        let _ = m1 + m2;
    }

    #[test]
    #[should_panic]
    fn matrix_trait_add_panics_on_different_row_and_column_dim() {
        let m1: Matrix<i32> = Matrix::zero(2, 3);
        let m2: Matrix<i32> = Matrix::zero(3, 4);

        let _ = m1 + m2;
    }

    // ================================
    //
    // Sub trait tests
    //
    // ================================

    #[test]
    fn matrix_trait_sub_matrix_matrix_i32_owned() {
        let rows_1 = [vec![0_i32, -1, 2], vec![3, -4, 5]];
        let rows_2 = [vec![0_i32, 10, -8], vec![5, -12, 3]];

        let m1 = Matrix::from_rows(&rows_1);
        let m2 = Matrix::from_rows(&rows_2);

        let expected_rows_1_minus_2 = [vec![0_i32, -11, 10], vec![-2, 8, 2]];

        // can only test this once because move occurs without use of references
        assert_eq!((m1 - m2).rows, expected_rows_1_minus_2);
    }

    #[test]
    fn matrix_trait_sub_matrix_matrix_i32_ref() {
        let rows_1 = [vec![0_i32, -1, 2], vec![3, -4, 5]];
        let rows_2 = [vec![0_i32, 10, -8], vec![5, -12, 3]];

        let m1 = Matrix::from_rows(&rows_1);
        let m2 = Matrix::from_rows(&rows_2);

        let expected_rows_1_minus_2 = [vec![0_i32, -11, 10], vec![-2_i32, 8, 2]];
        let expected_rows_1_minus_2_neg = [vec![0_i32, 11, -10], vec![2_i32, -8, -2]];

        let expected_rows_1_minus_2_minus_2 = [vec![0_i32, -21, 18], vec![-7_i32, 20, -1]];
        let expected_rows_zeroes = [vec![0_i32, 0_i32, 0_i32], vec![0_i32, 0_i32, 0_i32]];

        assert_eq!((&m1 - &m2).rows, expected_rows_1_minus_2);
        assert_eq!((&m2 - &m1).rows, expected_rows_1_minus_2_neg); // anti-commutative
        assert_eq!((&(&m1 - &m2) - &m2).rows, expected_rows_1_minus_2_minus_2); // non-associative
        assert_eq!((&m1 - &(&m2 - &m2)).rows, rows_1); // non-associative
        assert_eq!((&m1 - &Matrix::zero(2, 3)).rows, rows_1); // additive identity (zero matrix) subtraction does not change values
        assert_eq!((&m1 - &(-m1.additive_inverse())).rows, expected_rows_zeroes); // subtraction with the negative of the additive inverse yields a zero matrix
        assert_eq!((&m1 - &m1).rows, expected_rows_zeroes); // subtraction with self yields the zero matrix
    }

    #[test]
    fn matrix_trait_sub_matrix_matrix_f64_owned() {
        let rows_1 = [vec![0.0_f64, -1.0, 2.0], vec![3.0_f64, -4.0, 5.0]];
        let rows_2 = [vec![0.0_f64, 10.0, -8.0], vec![5.0_f64, -12.0, 3.0]];

        let m1 = Matrix::from_rows(&rows_1);
        let m2 = Matrix::from_rows(&rows_2);

        let expected_rows_1_minus_2 =
            [vec![0.0_f64, -11.0_f64, 10.0_f64], vec![-2.0_f64, 8.0, 2.0]];

        // can only test this once because move occurs without use of references
        assert_eq!((m1 - m2).rows, expected_rows_1_minus_2);
    }

    #[test]
    fn matrix_trait_sub_matrix_matrix_f64_ref() {
        let rows_1 = [vec![0.0_f64, -1.0, 2.0], vec![3.0_f64, -4.0, 5.0]];
        let rows_2 = [vec![0.0_f64, 10.0, -8.0], vec![5.0_f64, -12.0, 3.0]];

        let m1 = Matrix::from_rows(&rows_1);
        let m2 = Matrix::from_rows(&rows_2);

        let expected_rows_1_minus_2 = [vec![0.0_f64, -11.0, 10.0], vec![-2.0_f64, 8.0, 2.0]];
        let expected_rows_1_minus_2_neg = [vec![0.0_f64, 11.0, -10.0], vec![2_f64, -8.0, -2.0]];

        let expected_rows_1_minus_2_minus_2 =
            [vec![0.0_f64, -21.0, 18.0], vec![-7.0_f64, 20.0, -1.0]];
        let expected_rows_zeroes = [vec![0.0_f64, 0.0, 0.0], vec![0.0_f64, 0.0, 0.0]];

        assert_eq!((&m1 - &m2).rows, expected_rows_1_minus_2);
        assert_eq!((&m2 - &m1).rows, expected_rows_1_minus_2_neg); // anti-commutative
        assert_eq!((&(&m1 - &m2) - &m2).rows, expected_rows_1_minus_2_minus_2); // non-associative
        assert_eq!((&m1 - &(&m2 - &m2)).rows, rows_1); // non-associative
        assert_eq!((&m1 - &Matrix::zero(2, 3)).rows, rows_1); // additive identity (zero matrix) subtraction does not change values
        assert_eq!((&m1 - &(-m1.additive_inverse())).rows, expected_rows_zeroes); // subtraction with the negative of the additive inverse yields a zero matrix
        assert_eq!((&m1 - &m1).rows, expected_rows_zeroes); // subtraction with self yields the zero matrix
    }

    #[test]
    fn matrix_trait_sub_matrix_matrix_complex_i32_owned() {
        let rows_1 = [
            vec![Complex::new(0_i32, -1), Complex::new(-1_i32, 0)],
            vec![Complex::new(3_i32, -4), Complex::new(-4_i32, 3)],
        ];
        let rows_2 = [
            vec![Complex::new(0_i32, -10), Complex::new(-10_i32, 0)],
            vec![Complex::new(30_i32, -40), Complex::new(-40_i32, 30)],
        ];

        let m1 = Matrix::from_rows(&rows_1);
        let m2 = Matrix::from_rows(&rows_2);

        let expected_rows_1_minus_2 = [
            vec![Complex::new(0_i32, 9), Complex::new(9_i32, 0)],
            vec![Complex::new(-27_i32, 36), Complex::new(36_i32, -27)],
        ];

        // can only test this once because move occurs without use of references
        assert_eq!((m1 - m2).rows, expected_rows_1_minus_2);
    }

    #[test]
    fn matrix_trait_sub_matrix_matrix_complex_i32_ref() {
        let rows_1 = [
            vec![Complex::new(0_i32, -1), Complex::new(-1_i32, 0)],
            vec![Complex::new(3_i32, -4), Complex::new(-4_i32, 3)],
        ];
        let rows_2 = [
            vec![Complex::new(0_i32, -10), Complex::new(-10_i32, 0)],
            vec![Complex::new(30_i32, -40), Complex::new(-40_i32, 30)],
        ];

        let m1 = Matrix::from_rows(&rows_1);
        let m2 = Matrix::from_rows(&rows_2);

        let expected_rows_1_minus_2 = [
            vec![Complex::new(0_i32, 9), Complex::new(9_i32, 0)],
            vec![Complex::new(-27_i32, 36), Complex::new(36_i32, -27)],
        ];

        let expected_rows_1_minus_2_neg = [
            vec![Complex::new(0_i32, -9), Complex::new(-9_i32, 0)],
            vec![Complex::new(27_i32, -36), Complex::new(-36_i32, 27)],
        ];

        let expected_rows_1_minus_2_minus_2 = [
            vec![Complex::new(0_i32, 19), Complex::new(19_i32, 0)],
            vec![Complex::new(-57_i32, 76), Complex::new(76_i32, -57)],
        ];

        let expected_rows_zeroes = [
            vec![Complex::new(0_i32, 0), Complex::new(0_i32, 0)],
            vec![Complex::new(0_i32, 0), Complex::new(0_i32, 0)],
        ];

        assert_eq!((&m1 - &m2).rows, expected_rows_1_minus_2);
        assert_eq!((&m2 - &m1).rows, expected_rows_1_minus_2_neg); // anti-commutative
        assert_eq!((&(&m1 - &m2) - &m2).rows, expected_rows_1_minus_2_minus_2); // non-associative
        assert_eq!((&m1 - &(&m2 - &m2)).rows, rows_1); // non-associative
        assert_eq!((&m1 - &Matrix::zero(2, 2)).rows, rows_1); // additive identity (zero matrix) subtraction does not change values
        assert_eq!((&m1 - &(-m1.additive_inverse())).rows, expected_rows_zeroes); // subtraction with the negative of the additive inverse yields a zero matrix
        assert_eq!((&m1 - &m1).rows, expected_rows_zeroes); // subtraction with self yields the zero matrix
    }

    #[test]
    fn matrix_trait_sub_matrix_matrix_complex_f64_owned() {
        let rows_1 = [
            vec![Complex::new(0.0_f64, -1.0), Complex::new(-1.0_f64, 0.0)],
            vec![Complex::new(3.0_f64, -4.0), Complex::new(-4.0_f64, 3.0)],
        ];
        let rows_2 = [
            vec![Complex::new(0.0_f64, -10.0), Complex::new(-10.0_f64, 0.0)],
            vec![Complex::new(30.0_f64, -40.0), Complex::new(-40.0_f64, 30.0)],
        ];

        let m1 = Matrix::from_rows(&rows_1);
        let m2 = Matrix::from_rows(&rows_2);

        let expected_rows_1_minus_2 = [
            vec![Complex::new(0.0_f64, 9.0), Complex::new(9.0_f64, 0.0)],
            vec![Complex::new(-27.0_f64, 36.0), Complex::new(36.0_f64, -27.0)],
        ];

        // can only test this once because move occurs without use of references
        assert_eq!((m1 - m2).rows, expected_rows_1_minus_2);
    }

    #[test]
    fn matrix_trait_sub_matrix_matrix_complex_f64_ref() {
        let rows_1 = [
            vec![Complex::new(0.0_f64, -1.0), Complex::new(-1.0_f64, 0.0)],
            vec![Complex::new(3.0_f64, -4.0), Complex::new(-4.0_f64, 3.0)],
        ];
        let rows_2 = [
            vec![Complex::new(0.0_f64, -10.0), Complex::new(-10.0_f64, 0.0)],
            vec![Complex::new(30.0_f64, -40.0), Complex::new(-40.0_f64, 30.0)],
        ];

        let m1 = Matrix::from_rows(&rows_1);
        let m2 = Matrix::from_rows(&rows_2);

        let expected_rows_1_minus_2 = [
            vec![Complex::new(0.0_f64, 9.0), Complex::new(9.0_f64, 0.0)],
            vec![Complex::new(-27.0_f64, 36.0), Complex::new(36.0_f64, -27.0)],
        ];

        let expected_rows_1_minus_2_neg = [
            vec![Complex::new(0.0_f64, -9.0), Complex::new(-9.0_f64, 0.0)],
            vec![Complex::new(27.0_f64, -36.0), Complex::new(-36.0_f64, 27.0)],
        ];

        let expected_rows_1_minus_2_minus_2 = [
            vec![Complex::new(0.0_f64, 19.0), Complex::new(19.0_f64, 0.0)],
            vec![Complex::new(-57.0_f64, 76.0), Complex::new(76.0_f64, -57.0)],
        ];

        let expected_rows_zeroes = [
            vec![Complex::new(0.0_f64, 0.0), Complex::new(0.0_f64, 0.0)],
            vec![Complex::new(0.0_f64, 0.0), Complex::new(0.0_f64, 0.0)],
        ];

        assert_eq!((&m1 - &m2).rows, expected_rows_1_minus_2);
        assert_eq!((&m2 - &m1).rows, expected_rows_1_minus_2_neg); // anti-commutative
        assert_eq!((&(&m1 - &m2) - &m2).rows, expected_rows_1_minus_2_minus_2); // non-associative
        assert_eq!((&m1 - &(&m2 - &m2)).rows, rows_1); // non-associative
        assert_eq!((&m1 - &Matrix::zero(2, 2)).rows, rows_1); // additive identity (zero matrix) subtraction does not change values
        assert_eq!((&m1 - &(-m1.additive_inverse())).rows, expected_rows_zeroes); // subtraction with the negative of the additive inverse yields a zero matrix
        assert_eq!((&m1 - &m1).rows, expected_rows_zeroes); // subtraction with self yields the zero matrix
    }

    #[test]
    #[should_panic]
    fn matrix_trait_sub_panics_on_different_row_dim() {
        let m1: Matrix<i32> = Matrix::zero(2, 3);
        let m2: Matrix<i32> = Matrix::zero(3, 3);

        let _ = m1 - m2;
    }

    #[test]
    #[should_panic]
    fn matrix_trait_sub_panics_on_different_column_dim() {
        let m1: Matrix<i32> = Matrix::zero(3, 3);
        let m2: Matrix<i32> = Matrix::zero(3, 4);

        let _ = m1 - m2;
    }

    #[test]
    #[should_panic]
    fn matrix_trait_sub_panics_on_different_row_and_column_dim() {
        let m1: Matrix<i32> = Matrix::zero(2, 3);
        let m2: Matrix<i32> = Matrix::zero(3, 4);

        let _ = m1 - m2;
    }
}
