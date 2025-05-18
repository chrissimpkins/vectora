//! A vector computation library
//!
//! # Contents
//!
//! - [About](#about)
//! - [Safety Guarantee](#safety-guarantee)
//! - [Versioning](#versioning)
//! - [Minimum Rust Version Compatibility Policy](#minimum-rust-version-compatibility-policy)
//! - [Source](#source)
//! - [Changes](#changes)
//! - [Issues](#issues)
//! - [Contributing](#contributing)
//! - [License](#license)
//! - [Getting Started](#getting-started)
//! - [Advanced Usage](#advanced-usage)
//!
//! # About
//!
//! Vectora is a library for n-dimensional vector computation with real and complex scalar types.
//! The main library entry point is the [`Vector`] struct.  Please see the [Gettting Started guide](#getting-started)
//! for a detailed library overview with examples.
//!
//! # Safety Guarantee
//!
//! The current default distribution does not contain `unsafe` code blocks.
//!
//! # Versioning
//!
//! This project uses [semantic versioning](https://semver.org/) and is currently in a pre-v1.0 stage
//! of development.  The public API should not be considered stable across release versions at this
//! time.
//!
//! # Minimum Rust Version Compatibility Policy
//!
//! This project parameterizes generics by constants and relies on the [constant generics feature support
//! that was stabilized in Rust v1.51](https://github.com/rust-lang/rust/pull/79135).  The minimum
//! supported `rustc` version is believed to be v1.51.0.
//!
//! # Source
//!
//! The source files are available at <https://github.com/chrissimpkins/vectora>.
//!
//! # Changes
//!
//! Please see the [CHANGELOG.md](https://github.com/chrissimpkins/vectora/blob/main/CHANGELOG.md) document in the source repository.
//!
//! # Issues
//!
//! The [issue tracker](https://github.com/chrissimpkins/vectora/issues) is available on the GitHub repository.
//! Don't be shy.  Please report any issues that you identify so that we can address them.
//!
//! # Contributing
//!
//! Contributions are welcomed.  Developer documentation is available in the source
//! repository [README](https://github.com/chrissimpkins/vectora).
//!
//! Submit your source or documentation changes as a GitHub pull request on
//! the [source repository](https://github.com/chrissimpkins/vectora).
//!
//! # License
//!
//! Vectora is released under the [Apache License v2.0](https://github.com/chrissimpkins/vectora/blob/main/LICENSE.md).
//! Please review the full text of the license for details.
//!
//! # Getting Started
//!
//! ## Quick Start
//!
//! ```
//! use vectora::Vector;
//!
//! let mut v = Vector::<i32, 3>::from([1, 2, 3]);
//! v += Vector::from([4, 5, 6]);
//! v *= 2;
//! assert_eq!(v, Vector::from([10, 14, 18]));
//! ```
//!
//! ## Adding Vectora to Your Project
//!
//! In your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! vectora = "0.8.1"
//! ```
//!
//! ## Creating Vectors
//!
//! - **Default values:** `Vector::<i32, 3>::new()`
//! - **Additive identity (all zeroes):** `Vector::<i32, 3>::zero()`
//! - **Multiplicative identity (all ones):** `Vector::<i32, 3>::one()`
//! - **Filled with constant:** `Vector::<i32, 3>::filled(7)`
//! - **From array:** `Vector::from([1, 2, 3])`
//! - **From slice/vec:** `Vector::<i32, 3>::from_slice(&[1,2,3])?`, `Vector::<i32, 3>::from_vec(vec![1,2,3])?`
//!
//! ## Arithmetic
//!
//! - `+`, `-`, `*` for vector addition, subtraction, and scalar multiplication
//! - `+=`, `-=`, `*=` for in-place operations
//!
//! ## Supported Types
//!
//! - Integers: i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, usize, isize
//! - Floats: f32, f64
//! - Complex: [`num::Complex<T>`]
//!
//! ## More Examples
//!
//! See the [API docs](https://docs.rs/vectora) for advanced features, error handling, and trait implementations.
//!
//! # Advanced Usage
//!
//! ## Optional Crate Features
//!
//! Optional features are defined in your `Cargo.toml` configuration file:
//!
//! ```yaml
//! [dependencies]
//! vectora = { version = "VERSION_NUMBER", features = ["parallel"] }
//! ```
//!
//! Replace `VERSION_NUMBER` in the example above with the vectora crate version number.
//!
//! Conditional compilation and optional dependency installation are available for the following features:
//!
//! - **`parallel`**: Installs an optional [rayon crate](https://docs.rs/crate/rayon/latest) dependency and broadens the [`Vector`] API with
//!   parallel iterator and parallel slice support. This feature includes implementations of `Vector::into_par_iter`, `Vector::par_iter`,
//!   `Vector::par_iter_mut`, `Vector::as_parallel_slice`, and `Vector::as_parallel_slice_mut` methods with support for the rayon trait-defined
//!   [parallel iterator](https://docs.rs/rayon/latest/rayon/iter/trait.ParallelIterator.html#provided-methods),
//!   [immutable parallel slice](https://docs.rs/rayon/latest/rayon/slice/trait.ParallelSlice.html), and
//!   [mutable parallel slice](https://docs.rs/rayon/latest/rayon/slice/trait.ParallelSliceMut.html) APIs.
//!
//!
//! ## Numeric Type Support
//!
//! This library supports computation with real and complex scalar number types.
//!
//! ### Integers
//!
//! Support is available for the following primitive integer data types:
//!
//! - [`i8`]
//! - [`i16`]
//! - [`i32`]
//! - [`i64`]
//! - [`i128`]
//! - [`u8`]
//! - [`u16`]
//! - [`u32`]
//! - [`u64`]
//! - [`u128`]
//! - [`usize`]
//! - [`isize`]
//!
//! **Note**: overflowing integer arithmetic uses the default
//! Rust standard library approach of panics in debug builds
//! and twos complement wrapping in release builds.  You will not encounter
//! undefined behavior with either build type, but this approach
//! may not be what you want. Please consider this issue and understand
//! the library source implementations if your use case requires support
//! for integer overflows/underflows, and you prefer to handle it differently.
//!
//! ### Floating Point Numbers
//!
//! Support is available for the following primitive IEEE 754-2008 floating point types:
//!
//! - [`f32`]
//! - [`f64`]
//!
//! ### Complex Numbers
//!
//! The [`Vector`] type supports **collections of** complex scalars as
//! represented by the [`num::Complex`] type.  Please review the num crate documentation
//! for additional details on the [`num::Complex`] number type.
//!
//! **Note**: This guide will not provide detailed examples with [`num::Complex`] data in order to
//! remain as concise as possible. With the notable exception of the
//! floating point only [`Vector`] methods that can be identified with a [`num::Float`] trait bound,
//! much of the public API supports the [`num::Complex`] type.  These areas should be evident in the
//! [`Vector`] API documentation descriptions and source trait bounds.  [`num::Complex`] support
//! *should* be available when a general [`num::Num`] trait bound is used in
//! the implementation. In these cases, you can replace integer or floating point
//! numbers in the following examples with [`num::Complex`] types.  Please raise an
//! issue on the repository if this is not the case.
//!
//! ## Numeric Type Casts
//!
//! Use the `to_[TYPE SIGNATURE]` methods for explicit [`Vector`] data type casts
//! to supported integer and floating point types.  Casts to unsupported numeric
//! types (e.g., signed integer to unsigned integer) return `None`.  Casts from
//! unsupported types (e.g., [`num::Complex`] number with a non-zero imaginary part)
//! return `None`.
//!
//! ```
//! # use vectora::Vector;
//! use num::Complex;
//!
//! let v_u8: Vector<u8, 3> = Vector::from([1, 2, 3]);
//! let v_i8: Vector<i8, 3> = Vector::from([-1, 2, 3]);
//! let v_complex: Vector<Complex<f32>, 2> = Vector::from([Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);
//!
//! let v_i32: Vector<i32, 3> = v_u8.to_i32().unwrap();
//! let v_f64: Vector<f64, 3> = v_u8.to_f64().unwrap();
//! let v_f64_2: Vector<f64, 2> = v_complex.to_f64().unwrap();
//!
//! assert!(v_i8.to_u32().is_none());
//! ```
//!
//! Implicit, lossless type casts can be performed between supported types
//! with the `into` method:
//!
//! ```
//! # use vectora::types::vector::Vector;
//! let v_i32: Vector<i32, 2> = Vector::from([1_i32, 2_i32]);
//!
//! let v_i128: Vector<i128, 2> = v_i32.into();
//! let v_f64: Vector<f64, 2> = v_i32.into();
//! ```
//!
//! And the [`Vector::to_num_cast`] method supports unchecked, closure-defined
//! type casts:
//!
//! ```
//! # use vectora::types::vector::Vector;
//! let v_i32: Vector<i32, 2> = Vector::from([1_i32, 2_i32]);
//!
//! let v_i128: Vector<i128, 2> = v_i32.to_num_cast(|x| x as i128);
//! let v_f64: Vector<f64, 2> = v_i32.to_num_cast(|x| x as f64);
//! ```
//!
//! Please review the API documentation for warnings and additional details.
//!
//! ## Access and Assignment with Indexing
//!
//! Use zero-based indices for access and assignment:
//!
//! ### Access
//!
//! ```
//! # use vectora::Vector;
//! let v1: Vector<f64, 3> = Vector::from([1.0, 2.0, 3.0]);
//!
//! let x = v1[0];
//! let y = v1[1];
//! let z = v1[2];
//! ```
//!
//! Attempts to access items beyond the length of the [`Vector`] panic:
//!
//! ```should_panic
//! # use vectora::Vector;
//! # let v1: Vector<f64, 3> = Vector::from([1.0, 2.0, 3.0]);
//! // panics!
//! let _ = v1[10];
//! ```
//!
//! ### Assignment
//!
//! ```
//! # use vectora::Vector;
//! let mut v1_m: Vector<f64, 3> = Vector::from([1.0, 2.0, 3.0]);
//!
//! v1_m[0] = 10.0;
//! v1_m[1] = 20.0;
//! v1_m[2] = 30.0;
//! ```
//!
//! Attempts to assign to items beyond the length of the [`Vector`] panic:
//!
//! ```should_panic
//! # use vectora::Vector;
//! # let mut v1_m: Vector<f64, 3> = Vector::from([1.0, 2.0, 3.0]);
//! // panics!
//! v1_m[10] = 100.0;
//! ```
//!
//! See the [`Vector::get`] and [`Vector::get_mut`] method documentation
//! for getters that perform bounds checks and do not panic.
//!
//! ## Slicing
//!
//! Coerce to a read-only [`slice`] of the [`Vector`]:
//!
//! ```
//! # use vectora::Vector;
//! let v = Vector::<i32, 3>::from([1, 2, 3]);
//! let v_slice = &v[0..2];
//!
//! assert_eq!(v_slice, [1, 2]);
//! ```
//!
//! ## Partial Equivalence Testing
//!
//! Partial equivalence relation support is available with the `==` operator
//! for integer, floating point, and [`num::Complex`] numeric types.
//!
//! ### Integer types
//!
//! [`Vector`] of real integer values:
//!
//! ```
//! # use vectora::Vector;
//! let v1: Vector<i32, 3> = Vector::from([10, 50, 100]);
//! let v2: Vector<i32, 3> = Vector::from([5*2, 25+25, 10_i32.pow(2)]);
//!
//! assert!(v1 == v2);
//! ```
//!
//! [`Vector`] of [`num::Complex`] numbers with integer real and imaginary parts:
//!
//! ```
//! # use vectora::Vector;
//! use num::Complex;
//!
//! let v1: Vector<Complex<i32>, 2> = Vector::from([Complex::new(1, 2), Complex::new(3, 4)]);
//! let v2: Vector<Complex<i32>, 2> = Vector::from([Complex::new(1, 2), Complex::new(3, 4)]);
//!
//! assert!(v1 == v2);
//! ```
//!
//! ### Float types
//!
//! We compare floating point types with the [approx](https://docs.rs/approx/latest/approx/)
//! crate relative epsilon equivalence relation implementation by default. This includes
//! fixed definitions of the epsilon and max relative difference values. See
//! [the section below](#custom-equivalence-relations-for-float-types) for customization
//! options with methods.
//!
//! Why a different strategy for floats?
//!
//! "Equivalence" with floating point numbers can be challenging:
//!
//! ```should_panic
//! // panics!
//! assert!(0.15_f64 + 0.15_f64 == 0.1_f64 + 0.2_f64);
//! ```
//!
//! You likely want these floating point sums to compare as approximately equivalent.
//!
//! With the [`Vector`] type, they do.
//!
//! [`Vector`] of floating point values:
//!
//! ```
//! # use vectora::Vector;
//! let v1: Vector<f64, 1> = Vector::from([0.15 + 0.15]);
//! let v2: Vector<f64, 1> = Vector::from([0.1 + 0.2]);
//!
//! assert!(v1 == v2);
//! ```
//!
//! [`Vector`] of [`num::Complex`] numbers with floating point real and imaginary parts:
//!
//! ```
//! # use vectora::Vector;
//! use num::Complex;
//!
//! let v1: Vector<Complex<f64>, 2> = Vector::from([Complex::new(0.15 + 0.15, 2.0), Complex::new(3.0, 4.0)]);
//! let v2: Vector<Complex<f64>, 2> = Vector::from([Complex::new(0.1 + 0.2, 2.0), Complex::new(3.0, 4.0)]);
//!
//! assert!(v1 == v2);
//! ```
//!
//! `assert_eq!` and `assert_ne!` macro assertions use the same
//! partial equivalence approach, as you'll note throughout these docs.
//!
//! You can implement the same equivalence relation approach for float types that
//! are **not** contained in a [`Vector`] with the [approx crate](https://docs.rs/approx/latest/approx/)
//! `relative_eq!`, `relative_ne!`, `assert_relative_eq!`, and `assert_relative_ne!`
//! macros.
//!
//! ### Custom equivalence relations for floating point types
//!
//! The library also provides method support for absolute, relative, and units in last place (ULPs)
//! approximate floating point equivalence relations. These methods allow custom epsilon, max relative,
//! and max ULPs difference tolerances to define relations when float data are near and far apart.  You
//! must call the method to use them.  It is not possible to modify the default approach used in the
//! `==` operator overload.
//!
//! See the API documentation for [`Vector`] implementations of the `approx` crate `AbsDiffEq`, `RelativeEq`,
//! `UlpsEq` traits in the links below:
//!
//! #### Absolute difference equivalence relation
//!
//! The absolute difference equivalence relation approach supports custom epsilon tolerance definitions.
//!
//! - [`Vector::abs_diff_eq`](types/vector/struct.Vector.html#impl-AbsDiffEq<Vector<f32%2C%20N>>) (`f32`)
//! - [`Vector::abs_diff_eq`](types/vector/struct.Vector.html#impl-AbsDiffEq<Vector<f64%2C%20N>>) (`f64`)
//! - [`Vector::abs_diff_eq`](types/vector/struct.Vector.html#impl-AbsDiffEq<Vector<Complex<f32>%2C%20N>>) (`Complex<f32>`)
//! - [`Vector::abs_diff_eq`](types/vector/struct.Vector.html#impl-AbsDiffEq<Vector<Complex<f64>%2C%20N>>) (`Complex<f64>`)
//!
//! #### Relative difference equivalence relation
//!
//! The relative difference equivalence relation approach supports custom epsilon (data that are near)
//! and max relative difference (data that are far apart) tolerance definitions.
//!
//! - [`Vector::relative_eq`](types/vector/struct.Vector.html#impl-RelativeEq<Vector<f32%2C%20N>>) (`f32`)
//! - [`Vector::relative_eq`](types/vector/struct.Vector.html#impl-RelativeEq<Vector<f64%2C%20N>>) (`f64`)
//! - [`Vector::relative_eq`](types/vector/struct.Vector.html#impl-RelativeEq<Vector<Complex<f32>%2C%20N>>) (`Complex<f32>`)
//! - [`Vector::relative_eq`](types/vector/struct.Vector.html#impl-RelativeEq<Vector<Complex<f64>%2C%20N>>) (`Complex<f64>`)
//!
//! #### Units in Last Place (ULPs) difference equivalence relation
//!
//! The ULPs difference equivalence relation approach supports custom epsilon (data that are near)
//! and max ULPs difference (data that are far apart) tolerance definitions.
//!
//! - [`Vector::ulps_eq`](types/vector/struct.Vector.html#impl-UlpsEq<Vector<f32%2C%20N>>) (`f32`)
//! - [`Vector::ulps_eq`](types/vector/struct.Vector.html#impl-UlpsEq<Vector<f64%2C%20N>>) (`f64`)
//! - [`Vector::ulps_eq`](types/vector/struct.Vector.html#impl-UlpsEq<Vector<Complex<f32>%2C%20N>>) (`Complex<f32>`)
//! - [`Vector::ulps_eq`](types/vector/struct.Vector.html#impl-UlpsEq<Vector<Complex<f64>%2C%20N>>) (`Complex<f64>`)
//!
//! ## Iteration and Loops
//!
//! ### Over immutable scalar references
//!
//! ```
//! # use vectora::Vector;
//! let v: Vector<i32, 3> = Vector::from([-1, 2, 3]);
//! let mut iter = v.iter();
//!
//! assert_eq!(iter.next(), Some(&-1));
//! assert_eq!(iter.next(), Some(&2));
//! assert_eq!(iter.next(), Some(&3));
//! assert_eq!(iter.next(), None);
//! ```
//!
//! The syntax for a loop over this type:
//!
//! ```
//! # use vectora::Vector;
//! # let v: Vector<i32, 3> = Vector::from([-1, 2, 3]);
//! for x in &v {
//!     // do things
//! }
//! ```
//!
//! ### Over mutable scalar references
//!
//! ```
//! # use vectora::Vector;
//! let mut v: Vector<i32, 3> = Vector::from([-1, 2, 3]);
//! let mut iter = v.iter_mut();
//!
//! assert_eq!(iter.next(), Some(&mut -1));
//! assert_eq!(iter.next(), Some(&mut 2));
//! assert_eq!(iter.next(), Some(&mut 3));
//! assert_eq!(iter.next(), None);
//! ```
//!
//! The syntax for a loop over this type:
//!
//! ```
//! # use vectora::Vector;
//! # let mut v: Vector<i32, 3> = Vector::from([-1, 2, 3]);
//! for x in &mut v {
//!     // do things
//! }
//! ```
//!
//! ### Over mutable scalar values
//!
//! ```
//! # use vectora::Vector;
//! let v: Vector<i32, 3> = Vector::from([-1, 2, 3]);
//! let mut iter = v.into_iter();
//!
//! assert_eq!(iter.next(), Some(-1));
//! assert_eq!(iter.next(), Some(2));
//! assert_eq!(iter.next(), Some(3));
//! assert_eq!(iter.next(), None);
//! ```
//!
//! The syntax for a loop over this type:
//!
//! ```
//! # use vectora::Vector;
//! # let v: Vector<i32, 3> = Vector::from([-1, 2, 3]);
//! for x in v {
//!     // do things
//! }
//! ```
//!
//! ### Parallel iteration (Optional crate feature)
//!
//! *Optional* rayon parallel iterator support may be installed with the vectora crate
//! `parallel` feature.  With activation of this feature, you may use the `Vector::into_par_iter`,
//! `Vector::par_iter`, or `Vector::par_iter_mut` methods to create a parallel iterator over owned
//! [`Vector`] scalars or scalar references.  The `parallel` feature provides access to the
//! [rayon parallel iterator API](https://docs.rs/rayon/latest/rayon/iter/trait.ParallelIterator.html#provided-methods).
//!
//! See the [Optional Crate Features](#optional-crate-features) section above for
//! installation instructions and refer to the [`Vector`] API docs for additional details.
//!
//! ## Vector Arithmetic
//!
//! Use operator overloads for vector arithmetic:
//!
//! ### Unary Negation
//!
//! The unary negation operator yields the additive inverse
//! [`Vector`]:
//!
//! ```
//! # use vectora::Vector;
//! let v: Vector<f64, 3> = Vector::from([1.0, 2.0, 3.0]);
//!
//! assert_eq!(-v, Vector::from([-1.0, -2.0, -3.0]));
//! assert_eq!(v + -v, Vector::<f64, 3>::zero());
//! ```
//!
//! ### Vector Addition
//!
//! ```
//! # use vectora::Vector;
//! let v1: Vector<f64, 3> = Vector::from([1.0, 2.0, 3.0]);
//! let v2: Vector<f64, 3> = Vector::from([4.0, 5.0, 6.0]);
//!
//! let v3 = v1 + v2;
//!
//! assert_eq!(v3, Vector::from([5.0, 7.0, 9.0]));
//! ```
//!
//! ### Vector Subtraction
//!
//! ```
//! # use vectora::Vector;
//! let v1: Vector<f64, 3> = Vector::from([1.0, 2.0, 3.0]);
//! let v2: Vector<f64, 3> = Vector::from([4.0, 5.0, 6.0]);
//!
//! let v3 = v2 - v1;
//!
//! assert_eq!(v3, Vector::from([3.0, 3.0, 3.0]));
//! ```
//!
//! ### Scalar Multiplication
//!
//! ```
//! # use vectora::Vector;
//! let v1: Vector<f64, 3> = Vector::from([1.0, 2.0, 3.0]);
//! let v2: Vector<f64, 3> = Vector::from([4.0, 5.0, 6.0]);
//!
//! let v3 = v1 * 10.0;
//! let v4 = v2 * -1.0;
//!
//! assert_eq!(v3, Vector::from([10.0, 20.0, 30.0]));
//! assert_eq!(v4, Vector::from([-4.0, -5.0, -6.0]));
//! ```
//!
//! #### Scalar multiplication with [`num::Complex`] numbers
//!
//! [`Vector`] of [`num::Complex`] types support multiplication with
//! real and complex numbers.
//!
//! ##### [`num::Complex`] * real
//!
//! ```
//! # use vectora::Vector;
//! use num::Complex;
//!
//! let v: Vector<Complex<i32>, 2> = Vector::from([Complex::new(1, 2), Complex::new(3, 4)]);
//!
//! assert_eq!(v * 10, Vector::<Complex<i32>, 2>::from([Complex::new(10, 20), Complex::new(30, 40)]));
//! ```
//!
//! ##### [`num::Complex`] * [`num::Complex`]
//!
//! ```
//! # use vectora::Vector;
//! use num::Complex;
//!
//! let v: Vector<Complex<f64>, 2> = Vector::from([Complex::new(3.0, 2.0), Complex::new(-3.0, -2.0)]);
//! let c: Complex<f64> = Complex::new(1.0, 7.0);
//!
//! assert_eq!(v * c, Vector::from([Complex::new(-11.0, 23.0), Complex::new(11.0, -23.0)]));
//! ```
//!
//! #### Operator Assignment In-Place
//!
//! ```
//! # use vectora::Vector;
//! let mut v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
//! let v2: Vector<i32, 3> = Vector::from([4, 5, 6]);
//!
//! v1 += v2; // In-place addition
//! assert_eq!(v1, Vector::from([5, 7, 9]));
//!
//! v1 -= Vector::from([1, 1, 1]); // In-place subtraction
//! assert_eq!(v1, Vector::from([4, 6, 8]));
//!
//! v1 *= 2; // In-place scalar multiplication
//! assert_eq!(v1, Vector::from([8, 12, 16]));
//! ```
//!
//! For vectors of complex numbers, you can also multiply by a real scalar:
//!
//! ```
//! # use vectora::Vector;
//! use num::Complex;
//!
//! let mut v = Vector::<Complex<f64>, 2>::from([Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
//! v *= 2.0;
//! assert_eq!(v, Vector::from([Complex::new(2.0, 4.0), Complex::new(6.0, 8.0)]));
//! ```
//!
//! ## Methods for Vector Operations
//!
//! Method support is available for common vector calculations.
//! Examples of some frequently used operations are shown below:
//!
//! ### Dot product
//!
//! ```
//! # use vectora::Vector;
//! use approx::assert_relative_eq;
//!
//! let v1: Vector<f64, 3> = Vector::from([1.0, 2.0, 3.0]);
//! let v2: Vector<f64, 3> = Vector::from([4.0, 5.0, 6.0]);
//!
//! let dot_prod = v1.dot(&v2);
//!
//! assert_relative_eq!(dot_prod, 32.0);
//! ```
//!
//! [ [API docs](types/vector/struct.Vector.html#method.dot) ]
//!
//! ### Vector Magnitude
//!
//! ```
//! # use vectora::Vector;
//! use approx::assert_relative_eq;
//!
//! let v1: Vector<f64, 3> = Vector::from([1.0, 2.0, 3.0]);
//!
//! let m = v1.magnitude();
//!
//! assert_relative_eq!(m, 3.7416573867739413);
//! ```
//!
//! [ [API docs](types/vector/struct.Vector.html#method.magnitude) ]
//!
//! ### Vector Distance
//!
//! ```
//! # use vectora::Vector;
//! use approx::assert_relative_eq;
//!
//! let v1: Vector<f64, 2> = Vector::from([2.0, 2.0]);
//! let v2: Vector<f64, 2> = Vector::from([4.0, 4.0]);
//!
//! assert_relative_eq!(v1.distance(&v2), 8.0_f64.sqrt());
//! assert_relative_eq!(v1.distance(&v1), 0.0_f64);
//! ```
//!
//! [ [API docs](types/vector/struct.Vector.html#method.distance) ]
//!
//! ### Opposite Vector
//!
//! ```
//! # use vectora::Vector;
//! use approx::assert_relative_eq;
//!
//! let v: Vector<f64, 3> = Vector::from([2.0, 2.0, 2.0]);
//!
//! assert_eq!(v.opposite(), Vector::from([-2.0, -2.0, -2.0]));
//! assert_relative_eq!(v.opposite().magnitude(), v.magnitude());
//! ```
//!
//! [ [API docs](types/vector/struct.Vector.html#method.opposite) ]
//!
//! ### Normalization
//!
//! ```
//! # use vectora::Vector;
//! use approx::assert_relative_eq;
//!
//! let v1: Vector<f64, 3> = Vector::from([1.0, 2.0, 3.0]);
//!
//! let unit_vector = v1.normalize();
//!
//! assert_relative_eq!(unit_vector.magnitude(), 1.0);
//! ```
//!
//! [ [API docs](types/vector/struct.Vector.html#method.normalize) ]
//!
//! ### Linear Interpolation
//!
//!```
//! # use vectora::Vector;
//! let v1: Vector<f64, 3> = Vector::from([1.0, 2.0, 3.0]);
//! let v2: Vector<f64, 3> = Vector::from([4.0, 5.0, 6.0]);
//!
//! let v3 = v1.lerp(&v2, 0.5).unwrap();
//!
//! assert_eq!(v3, Vector::from([2.5, 3.5, 4.5]));
//!```
//!
//! [ [API docs](types/vector/struct.Vector.html#method.lerp) ]
//!
//! ### Closure Mapping
//!
//! ```
//! # use vectora::Vector;
//! let v1: Vector<f64, 3> = Vector::from([-1.0, 2.0, 3.0]);
//!
//! let v3 = v1.map_closure(|x| x.powi(2));
//!
//! assert_eq!(v3, Vector::from([1.0, 4.0, 9.0]));
//! ```
//!
//! [ [API docs](types/vector/struct.Vector.html#method.map_closure) ]
//!
//! ### Function Mapping
//!
//!```
//!# use vectora::Vector;
//! let v1: Vector<f64, 3> = Vector::from([-1.0, 2.0, 3.0]);
//!
//! fn square(x: f64) -> f64 {
//!     x.powi(2)
//! }
//!
//! let v3 = v1.map_fn(square);
//!
//! assert_eq!(v3, Vector::from([1.0, 4.0, 9.0]));
//!```
//!
//! [ [API docs](types/vector/struct.Vector.html#method.map_fn) ]
//!
//! Many of these methods have mutable alternates that edit the [`Vector`] data
//! in place instead of allocating a new [`Vector`].  The mutable methods are
//! prefixed with `mut_*`.
//!
//! See the [`Vector` method implementations](types/vector/struct.Vector.html#implementations) docs
//! for the complete list of supported methods and additional examples.
//!
//! ## Descriptive Statistics
//!
//! Element-wise measures of central tendency and dispersion are available for floating
//! point [`Vector`] types.
//!
//! ### Arithmetic Mean
//!
//! ```
//! # use vectora::Vector;
//! use approx::assert_relative_eq;
//!
//! let v: Vector<f64, 6> = Vector::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//!
//! assert_relative_eq!(v.mean().unwrap(), 3.5);
//! ```
//!
//! ### Geometric mean
//!
//! ```
//! # use vectora::Vector;
//! use approx::assert_relative_eq;
//!
//! let v: Vector<f64, 6> = Vector::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//!
//! assert_relative_eq!(v.mean_geo().unwrap(), 2.993795165523909);
//! ```
//!
//! ### Harmonic mean
//!
//! ```
//! # use vectora::Vector;
//! use approx::assert_relative_eq;
//!
//! let v: Vector<f64, 6> = Vector::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//!
//! assert_relative_eq!(v.mean_harmonic().unwrap(), 2.4489795918367347);
//! ```
//!
//! ### Median
//!
//! ```
//! # use vectora::Vector;
//! use approx::assert_relative_eq;
//!
//! let v: Vector<f64, 6> = Vector::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//!
//! assert_relative_eq!(v.median().unwrap(), 3.5);
//! ```
//!
//! ### Variance
//!
//! ```
//! # use vectora::Vector;
//! use approx::assert_relative_eq;
//!
//! let v: Vector<f64, 6> = Vector::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//!
//! // population variance
//! assert_relative_eq!(v.variance(0.0).unwrap(), 2.9166666666666665);
//!
//! // sample variance (with Bessel's correction)
//! assert_relative_eq!(v.variance(1.0).unwrap(), 3.5);
//! ```
//!
//! ### Standard deviation
//!
//! ```
//! # use vectora::Vector;
//! use approx::assert_relative_eq;
//!
//! let v: Vector<f64, 6> = Vector::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//!
//! // population standard deviation
//! assert_relative_eq!(v.stddev(0.0).unwrap(), 1.707825127659933);
//!
//! // sample standard deviation (with Bessel's correction)
//! assert_relative_eq!(v.stddev(1.0).unwrap(), 1.8708286933869707);
//! ```
//!
//! ## Working with Rust Standard Library Types
//!
//! Casts to commonly used Rust standard library data collection types are straightforward.
//! Note that some of these type casts support mutable [`Vector`] owned data references,
//! allowing you to use standard library type operations to change the [`Vector`] data.
//!
//! ### [`array`] Representations
//!
//! Immutable:
//!
//! ```
//! # use vectora::Vector;
//! let v: Vector<i32, 3> = Vector::from([-1, 2, 3]);
//!
//! assert_eq!(v.as_array(), &[-1, 2, 3]);
//! assert_eq!(v.to_array(), [-1, 2, 3]);
//! ```
//!
//! Mutable:
//!
//! ```
//! # use vectora::Vector;
//! let mut v: Vector<i32, 3> = Vector::from([-1, 2, 3]);
//!
//! let m_arr = v.as_mut_array();
//!
//! assert_eq!(m_arr, &mut [-1, 2, 3]);
//!
//! m_arr[0] = -10;
//!
//! assert_eq!(m_arr, &mut [-10, 2, 3]);
//! assert_eq!(v, Vector::from([-10, 2, 3]));
//! ```
//!
//! ### [`slice`] Representations
//!
//! Immutable:
//!
//! ```
//! # use vectora::Vector;
//! let v: Vector<i32, 3> = Vector::from([-1, 2, 3]);
//!
//! assert_eq!(v.as_slice(), &[-1, 2, 3][..]);
//! ```
//!
//! Mutable:
//!
//! ```
//! # use vectora::Vector;
//! let mut v: Vector<i32, 3> = Vector::from([-1, 2, 3]);
//!
//! let m_sli = v.as_mut_slice();
//!
//! assert_eq!(m_sli, &mut [-1, 2, 3][..]);
//!
//! m_sli[0] = -10;
//!
//! assert_eq!(m_sli, &mut [-10, 2, 3]);
//! assert_eq!(v, Vector::from([-10, 2, 3]));
//! ```
//!
//!
//! ### [`Vec`] Representations
//!
//! Casts to [`Vec`] always allocate a new [`Vec`] with copied data.
//!
//! ```
//! # use vectora::Vector;
//! let v: Vector<i32, 3> = Vector::from([-1, 2, 3]);
//!
//! assert_eq!(v.to_vec(), vec![-1, 2, 3]);
//! ```
//!
//! See the [Initialization](#initialization) section for documentation of
//! the syntax to instantiate a [`Vector`] from a standard library [`Vec`] type.

#![warn(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links, unsafe_code)]
#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod errors;
pub mod macros;
pub mod types;

pub use types::flexvector::FlexVector;
pub use types::vector::Vector;

/// The vectora prelude: import this module to bring all core traits, types, and macros into scope.
///
/// # Example
/// ```
/// use vectora::prelude::*;
///
/// let v = vector![1, 2, 3];
/// let fv = flexvector![1.0, 2.0, 3.0];
/// let v2 = Vector::<i32, 3>::from([4, 5, 6]);
/// let fv2 = FlexVector::from_vec(vec![4.0, 5.0, 6.0]);
/// ```
///
/// This prelude includes:
/// - Core traits (`VectorBase`, `VectorOps`, `VectorOpsFloat`, `VectorOpsComplex`)
/// - The main types (`Vector`, `FlexVector`)
/// - User-facing macros (`vector!`, `flexvector!`, `try_vector!`, `try_flexvector!`)
pub mod prelude {
    // Traits
    pub use crate::types::traits::{VectorBase, VectorOps, VectorOpsComplex, VectorOpsFloat};
    // Types
    pub use crate::types::flexvector::FlexVector;
    pub use crate::types::vector::Vector;
    // Macros
    pub use crate::{flexvector, try_flexvector, try_vector, vector};
}
