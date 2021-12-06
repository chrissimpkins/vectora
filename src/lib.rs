//! A vector computation library
//!
//! # Contents
//!
//! - [About](#about)
//! - [Safety Guarantee](#safety-guarantee)
//! - [Versioning](#versioning)
//! - [Source](#source)
//! - [Issues](#issues)
//! - [Contributing](#contributing)
//! - [License](#license)
//! - [Getting Started](#getting-started)
//!     - [Initialization](#initialization)
//!     - [Access and Assignment with Indexing](#access-and-assignment-with-indexing)
//!     - [Slicing](#slicing)
//!     - [Partial Equivalence Testing](#partial-equivalence-testing)
//!     - [Iteration and Loops](#iteration-and-loops)
//!     - [Vector Arithmetic](#vector-arithmetic)
//!     - [Methods for Vector Operations](#methods-for-vector-operations)
//!     - [Working with Rust Standard Library Types](#working-with-rust-standard-library-types)
//!     
//!
//! # About
//!
//! Vectora is a library for n-dimensional vector computation. The main library entry point is the [`Vector`] struct.  
//! Please see the [Gettting Started guide](#getting-started) for a detailed library overview with examples.
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
//! # Source
//!
//! The source files are available at <https://github.com/chrissimpkins/vectora>.
//!
//! # Issues
//!
//! The [issue tracker](https://github.com/chrissimpkins/vectora/issues) is available on the GitHub repository.
//! Don't be shy.  Please report any issues that you identify so that we can address them.
//!
//! # Contributing
//!
//! Contributions are welcomed.  Open your change proposal as a GitHub pull request on
//! the [source repository](https://github.com/chrissimpkins/vectora).
//!
//! # License
//!
//! Vectora is released under the [Apache License v2.0](https://github.com/chrissimpkins/vectora/blob/main/LICENSE.md).
//! Please review the full text of the license for details.
//!
//! # Getting Started
//!
//! See the [`Vector`] page for detailed API documentation of the main library
//! entry point.
//!
//! The following section provides an overview of common tasks, and will get you up
//! and running with the library quickly.  
//!
//! The source examples below assume the following [`Vector`] import:
//!
//! ```
//! use vectora::Vector;
//! ```
//!
//! ## Initialization
//!
//! A [`Vector`] can have mutable values, but it cannot grow in length.  The
//! dimension length is fixed at instantiation and all fields are *initialized*
//! at instantiation.
//!
//! ### Zero Vector
//!
//! Use the [`Vector::zero`] method to initialize a [`Vector`] with zero values
//! of the respective numeric type:
//!
//! ```
//! # use vectora::Vector;
//! let v_zero_int: Vector<i32, 3> = Vector::zero();
//! let v_zero_float: Vector<f64, 2> = Vector::zero();
//! ```
//!
//! ### With Predefined Data in Other Types
//!
//! The recommended approach is to use [`Vector::from`] with an [`array`] of
//! ordered data when possible:
//!
//! ```
//! # use vectora::Vector;
//! // example three dimensional f64 Vector
//! let v1: Vector<f64, 3> = Vector::from([1.0, 2.0, 3.0]);
//!
//! // example two dimensional i32 Vector
//! let v2: Vector<i32, 2> = Vector::from([4, -5]);
//!
//! // with a library type alias
//! use vectora::types::vector::Vector3dF64;
//!
//! let v3: Vector3dF64 = Vector::from([1.0, 2.0, 3.0]);
//! ```
//!
//! or use one of the alternate initialization approaches with data
//! in iterator, [`array`], [`slice`], or [`Vec`] types:
//!
//! ```
//! # use vectora::Vector;
//! // from an iterator over an array or Vec with collect
//! let v4: Vector<i32, 3> = [1, 2, 3].into_iter().collect();
//! let v5: Vector<f64, 2> = vec![1.0, 2.0].into_iter().collect();
//!
//! // from a slice with try_from
//! let arr = [1, 2, 3];
//! let vec = vec![1.0, 2.0, 3.0];
//! let v6: Vector<i32, 3> = Vector::try_from(&arr[..]).unwrap();
//! let v7: Vector<f64, 3> = Vector::try_from(&vec[..]).unwrap();
//!
//! // from a Vec with try_from
//! let vec = vec![1, 2, 3];
//! let v8: Vector<i32, 3> = Vector::try_from(&vec).unwrap();
//! ```
//!
//! Please see the API docs for the approach to overflows and underflows with the
//! [`FromIterator`](types/vector/struct.Vector.html#impl-FromIterator<T>)
//! implementation that supports the `collect` approach.
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
//! Attmpts to access items beyond the length of the [`Vector`] panics:
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
//! Attmpts to assign to items beyond the length of the [`Vector`] panics:
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
//! Partial equivalence comparison support is available for integer and
//! float numeric types with the `==` operator.
//!
//! ### Integer types
//!
//! ```
//! # use vectora::Vector;
//! let a: Vector<i32, 3> = Vector::from([10, 50, 100]);
//! let b: Vector<i32, 3> = Vector::from([5*2, 25+25, 10_i32.pow(2)]);
//!
//! assert!(a == b);
//! ```
//!
//! ### Float types
//!
//! Float comparisons use the [approx](https://docs.rs/approx/latest/approx/) crate
//! relative epsilon float equivalence relation implementation.
//!
//! Why handle these differently than the standard library implementation?
//!
//! Some floating point numbers can be defined as different due to
//! floating point precision:
//!
//! ```should_panic
//! // panics!
//! assert!(0.15_f64 + 0.15_f64 == 0.1_f64 + 0.2_f64);
//! ```
//!
//! You likely mean for these float sums to compare as approximately equivalent.
//!
//! With the [`Vector`] type, they do:
//!
//! ```
//! # use vectora::Vector;
//! let a: Vector<f64, 1> = Vector::from([0.15 + 0.15]);
//! let b: Vector<f64, 1> = Vector::from([0.1 + 0.2]);
//!
//! assert!(a == b);
//! ```
//!
//! `assert_eq!` and `assert_ne!` macro assertions use the same
//! partial equivalence testing approach as you'll note throughout these docs.
//!
//! You can implement the same equivalence relation approach for float types that
//! are **not** contained in a [`Vector`] with the [approx crate](https://docs.rs/approx/latest/approx/)
//! `relative_eq!`, `relative_ne!`, `assert_relative_eq!`, and `assert_relative_ne!`
//! macros.
//!
//! ## Iteration and Loops
//!
//! ### Over immutable scalar component references
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
//! Syntax for a loop over this type:
//!
//! ```
//! # use vectora::Vector;
//! # let v: Vector<i32, 3> = Vector::from([-1, 2, 3]);
//! for x in &v {
//!     // do things
//! }
//! ```
//!
//! ### Over mutable scalar component references
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
//! Syntax for a loop over this type:
//!
//! ```
//! # use vectora::Vector;
//! # let mut v: Vector<i32, 3> = Vector::from([-1, 2, 3]);
//! for x in &mut v {
//!     // do things
//! }
//! ```
//!
//! ### Over mutable scalar component values
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
//! Syntax for a loop over this type:
//!
//! ```
//! # use vectora::Vector;
//! # let v: Vector<i32, 3> = Vector::from([-1, 2, 3]);
//! for x in v {
//!     // do things
//! }
//! ```
//!
//! ## Vector Arithmetic
//!
//! Use operator overloads for vector arithmetic:
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
//! Please note that overflowing integer arithmetic uses the default
//! Rust standard library approach of panics in debug builds
//! and twos complement wrapping in release builds.  You will not encounter
//! undefined behavior with either build type, but this approach
//! may not be what you want. Avoid these operator overloads if your use
//! case requires support for integer overflows/underflows and you
//! prefer to handle it differently.
//!
//! ## Methods for Vector Operations
//!
//! Method support is available for other common vector calculations.
//! Examples of some commonly used operations are shown below:
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
//! ## Working with Rust Standard Library Types
//!
//! Casting a [`Vector`] to a number of commonly used Rust standard library data collection
//! types is straightforward.  Note that some of these type casts support mutable [`Vector`]
//! owned data references, allowing you to use standard library type operations to change the
//! [`Vector`] data.
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
//! This always allocates a new [`Vec`] with copied data and
//! does not support mutation of the original [`Vector`] data.
//!
//! ```
//! # use vectora::Vector;
//! let v: Vector<i32, 3> = Vector::from([-1, 2, 3]);
//!
//! assert_eq!(v.to_vec(), vec![-1, 2, 3]);
//! ```
//!
//! See the [Initialization](#initialization) section for details on how
//! to instantiate a [`Vector`] from a standard library [`Vec`] type.

#![warn(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links, unsafe_code)]

pub mod errors;
pub mod types;

pub use types::vector::Vector;
