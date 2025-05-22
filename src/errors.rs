//! Error types.

/// Errors that occur while working with [`crate::types::vector::Vector`]
#[derive(Debug)]
pub enum VectorError {
    /// Occurs when an operation that requires data in a [`crate::types::vector::Vector`] is
    /// requested with an empty [`crate::types::vector::Vector`]
    EmptyVectorError(String),
    /// Occurs when there is invalid data during an attempt to convert
    /// from [`slice`] data.
    TryFromSliceError(String),
    /// Occurs when there is invalid data during an attempt to convert
    /// from [`Vec`] data.
    TryFromVecError(String),
    /// ValueError occurs when an invalid value is used in an operation
    ValueError(String),
}

impl std::fmt::Display for VectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            VectorError::EmptyVectorError(s) => {
                write!(f, "VectorError::EmptyVectorError: {s}")
            }
            VectorError::TryFromVecError(s) => {
                write!(f, "VectorError::TryFromVecError: {s}")
            }
            VectorError::TryFromSliceError(s) => {
                write!(f, "VectorError::TryFromSliceError: {s}")
            }
            VectorError::ValueError(s) => {
                write!(f, "VectorError::ValueError: {s}")
            }
        }
    }
}

/// Errors that occur while working with [`crate::types::matrices::Matrix`]
#[derive(Debug)]
pub enum MatrixError {
    /// Occurs when an operation receives an invalid empty matrix
    EmptyMatrixError(String),
    /// Occurs when an operation receives an invalid matrix shape
    InvalidMatrixShapeError(String),
    /// Occurs when there is invalid data during a fallible attempt to convert
    /// a [`crate::Matrix`] row or column to another type
    TryFromMatrixError(String),
}

impl std::fmt::Display for MatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            MatrixError::EmptyMatrixError(s) => {
                write!(f, "MatrixError::EmptyMatrixError: {}", s)
            }
            MatrixError::InvalidMatrixShapeError(s) => {
                write!(f, "MatrixError::InvalidMatrixShapeError: {}", s)
            }
            MatrixError::TryFromMatrixError(s) => {
                write!(f, "MatrixError::TryFromMatrixError: {}", s)
            }
        }
    }
}
