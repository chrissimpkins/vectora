# Changelog

## v0.8.0

### Added

- `parallel` feature
- Optional `parallel` feature rayon parallel iterators over `Vector` scalars
- Optional `parallel` feature rayon immutable parallel slices over `Vector` scalars
- Optional `parallel` feature rayon mutable parallel slices over `Vector` scalars
- Optional `parallel` feature `Vector::into_par_iter` method
- Optional `parallel` feature `Vector::par_iter` method
- Optional `parallel` feature `Vector::par_iter_mut` method
- Optional `parallel` feature `Vector::as_parallel_slice` method
- Optional `parallel` feature `Vector::as_parallel_slice_mut` method

### Changed

- docs: revised Getting Started guide with available optional feature support
- docs: revised Getting Started guide installation instructions with new, optional feature Cargo.toml configuration
- docs: revised Getting Started guide, Iteration and Loops section, with information on new, optional parallel iterator support

## v0.7.0

### Added

- `macros` module
- `vector!` macro for initialization of `Vector` with infallible data
- `try_vector!` macro for initialization of `Vector` with fallible data
- `Vector::min` method for element-wise minimum value with `Num` data types that support the `std::cmp::Ord` trait
- `Vector::min_fp` method for element-wise minimum value with floating point data types that do not support the `std::cmp::Ord` trait
- `Vector::max` method for element-wise maximum value with `Num` data types that support the `std::cmp::Ord` trait
- `Vector::max_fp` method for element-wise maximum value with floating point data types that do not support the `std::cmp::Ord` trait

### Changed

- docs: revise Getting Started guide docs with `vector!` macro initialization strategy for infallible data collection types
- docs: revise Getting Started guide docs with `try_vector!` macro initialization strategy for fallible data collection types
- docs: revise API docs numeric type syntax in some examples

## v0.6.0

### Added

- `Vector::mean` method for arithmetic mean statistic with floating point data
- `Vector::mean_geo` method for geometric mean statistic with floating point data
- `Vector::mean_harmonic` method for harmonic mean statistic with floating point data
- `Vector::median` method for median statistic with floating point data
- `Vector::variance` method for variance statistic with floating point data (supports population and sample variance)
- `Vector::stddev` method for standard deviation statistic with floating point data (supports population and sample std dev)
- `Vector::to_usize` method for explicit type cast support
- `Vector::to_u8` method for explicit type cast support
- `Vector::to_u16` method for explicit type cast support
- `Vector::to_u32` method for explicit type cast support
- `Vector::to_u64` method for explicit type cast support
- `Vector::to_u128` method for explicit type cast support
- `Vector::to_isize` method for explicit type cast support
- `Vector::to_i8` method for explicit type cast support
- `Vector::to_i16` method for explicit type cast support
- `Vector::to_i32` method for explicit type cast support
- `Vector::to_i64` method for explicit type cast support
- `Vector::to_i128` method for explicit type cast support
- `Vector::to_f32` method for explicit type cast support
- `Vector::to_f64` method for explicit type cast support

### Changed

- docs: Getting Started guide revised with new "Descriptive Statistics" section
- docs: Getting Started guide revised with new "Numeric Type Casts" section
- docs: Getting Started guide updated with Complex number type cast support example
- docs: API docs revised with updated Vector numeric type cast information and internal links

## v0.5.1

### Added

- None

### Changed

- Cargo.toml config: reduced keyword number from 7 to 5. crates.io does not accept > 5 keywords

## v0.5.0

### Added

- `Vector` struct `pretty` method for pretty-print formatted `String` of data contents
- Implement `Display` trait for `Vector` type
- cargo fuzz + libFuzzer based fuzzing infrastructure support

### Changed

- Cargo.toml configuration: updated keyword metadata
- tests: add unit tests of Vector type introspection with `Any` trait implementation
- docs: Getting Started guide: revised to indicate the maximum length of a `Vector` type
- docs: Developer: revised to include new documentation of how to execute only unit tests and only doc tests
- docs: Developer: revised to include documentation of cargo fuzz based fuzzing on the library

## v0.4.0

### Added

- `Vector` struct `enumerate` method for enumeration over (index, value) tuples
- `Vector` struct `product` method for element-wise products (supports int, float, and complex number types)
- `Vector` struct `sum` method for element-wise sums (supports int, float, and complex number types)
- `Vector::product` Criterion benchmark tests

### Changed

- None

## v0.3.1

### Added

- None

### Changed

- docs: Getting Started guide: updated integer overflow / underflow text
- docs: Getting Started guide: typo fix

## v0.3.0

### Added

- `Vector` struct complex number support with the `num::Complex` data type
- PartialEq trait implementation for `Vector` of `num::Complex` types with integer real and imaginary parts
- PartialEq trait implementation for `Vector` of `num::Complex` types with floating point real and imaginary parts
- AbsDiffEq trait implementation for `Vector` of `num::Complex` types with floating point real and imaginary parts
- RelativeEq trait implementation for `Vector` of `num::Complex` types with floating point real and imaginary parts
- UlpsEq trait implementation for `Vector` of `num::Complex` types with floating point real and imaginary parts
- Mul trait implementation to support `Vector` of `num::Complex` types scalar multiplication with integer scalar values
- Mul trait implementation to support `Vector` of `num::Complex` types scalar multiplication with floating point scalar values
- Lossless `Vector` of `num::Complex` types unsigned integer to signed integer real and imaginary part cast support
- Lossless `Vector` of `num::Complex` types signed integer to signed integer real and imaginary part cast support
- Lossless `Vector` of `num::Complex` types unsigned integer to unsigned integer real and imaginary part cast support
- Lossless `Vector` of `num::Complex` types unsigned integer to signed integer real and imaginary part cast support
- Lossless `Vector` of `num::Complex` types unsigned integer to float real and imaginary part cast support
- Lossless `Vector` of `num::Complex` types signed integer to float real and imaginary part cast support
- Lossless `Vector` `f32` to `f64` floating point type cast support

### Changed

- refactor Neg trait implementation to support `Vector` of `num::Complex` types
- docs: Getting Started guide: major revision to document the new support for `Vector` of complex numbers
- docs: Getting Started guide: added new`Vector` unary negation operator documentation
- docs: API: major revision to document the new support for `Vector` of complex numbers
- docs: API: updated Vector `dot` method documentation to indicate that the method is not intended for complex number types
- docs: API: updated Vector `Float` trait bound methods documentation to indicate that these methods are intended for real, floating point types
- docs: README: minor revisions to document the new support for `Vector` of complex numbers

## v0.2.1

### Added

- integer and float Vector PartialEq testing with `==` operator benchmarks
- rustc nightly toolchain CI testing on Ubuntu Linux, macOS, and Windows
- scheduled nightly CI testing with rustc stable, beta, and nightly toolchains on Ubuntu Linux, macOS, and Windows

### Changed

- minor documentation updates

## v0.2.0

### Added

- customizable f32 and f64 `Vector` absolute epsilon difference partial equivalence relation support (#11)
- customizable f32 and f64 `Vector` relative epsilon difference partial equivalence relation support (#11)
- customizable f32 and f64 `Vector` units in last place (ULPs) difference partial equivalence relation support (#11)
- `Vector` initialization Criterion benchmarks (#6, #9)

### Changed

- `Vector` initialization with std lib `Vec` reference types execution time improvement (#7)

## v0.1.3

### Added

- Minimum Rust version compatibility policy added to the `lib.rs` (docs.rs/vectora) and source repository README documentation

### Changed

- None

## v0.1.2

### Added

- None

### Changed

- typo fixes in the source documentation

## v0.1.1

### Added

- how to add vectora package to Cargo.toml dependencies documentation
- maintainer documentation

### Changed

- source cleanup: removed unused, commented out imports
- src/lib.rs documentation revisions
- README.md documentation revisions and formatting

## v0.1.0

- initial release
