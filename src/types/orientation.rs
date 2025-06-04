//! Orientation marker types for vectors.
//!
//! These zero-sized types are used as type parameters to distinguish between row and column vectors
//! at compile time for both `Vector` and `FlexVector`.

use std::fmt;

/// Marker type for row vectors.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Row;

/// Marker type for column vectors.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Column;

/// ...
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum VectorOrientation {
    /// ...
    Row,
    /// ...
    Column,
}

impl fmt::Display for VectorOrientation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorOrientation::Row => write!(f, "Row"),
            VectorOrientation::Column => write!(f, "Column"),
        }
    }
}
