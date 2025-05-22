//! Orientation marker types for vectors.
//!
//! These zero-sized types are used as type parameters to distinguish between row and column vectors
//! at compile time for both `Vector` and `FlexVector`.

use crate::types::traits::VectorOrientationName;

/// Marker type for row vectors.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Row;

/// Marker type for column vectors.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Column;

impl VectorOrientationName for Row {
    fn orientation_name() -> &'static str {
        "Row"
    }
}

impl VectorOrientationName for Column {
    fn orientation_name() -> &'static str {
        "Column"
    }
}

/// ...
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum VectorOrientation {
    /// ...
    Row,
    /// ...
    Column,
}
