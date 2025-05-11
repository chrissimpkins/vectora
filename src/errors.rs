//! Error types.

/// Errors that occur while working with vector types.
#[derive(Debug)]
pub enum VectorError {
    /// Occurs when an operation that requires data in a vector type is
    /// requested with an empty vector type.
    EmptyVectorError(String),
    /// Occurs when two vectors need to be the same length to complete an operation, and are not.
    MismatchedLengthError(String),
    /// Occurs when a value extends outside of a mandatory range.
    OutOfRangeError(String),
    /// Occurs when there is invalid data during an attempt to convert
    /// from [`slice`] data.
    TryFromSliceError(String),
    /// Occurs when there is invalid data during an attempt to convert
    /// from [`Vec`] data.
    TryFromVecError(String),
    /// ValueError occurs when an invalid value is used in an operation
    ValueError(String),
    /// Errors with invalid zero vectors
    ZeroVectorError(String),
}

impl std::fmt::Display for VectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            VectorError::EmptyVectorError(s) => {
                write!(f, "VectorError::EmptyVectorError: {s}")
            }
            VectorError::MismatchedLengthError(s) => {
                write!(f, "VectorError::MismatchedLengthError: {}", s)
            }
            VectorError::OutOfRangeError(s) => {
                write!(f, "VectorError::OutOfRangeError: {}", s)
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
            VectorError::ZeroVectorError(s) => {
                write!(f, "VectorError::ZeroVectorError: {}", s)
            }
        }
    }
}
