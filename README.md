<img src="https://raw.githubusercontent.com/chrissimpkins/vectora/img/img/vectora.png" width="350">

## A Rust library for vector computation

[![Crates.io](https://img.shields.io/crates/v/vectora)](https://crates.io/crates/vectora)
[![docs.rs](https://img.shields.io/docsrs/vectora)](https://docs.rs/vectora)
[![GitHub](https://img.shields.io/github/license/chrissimpkins/vectora)](LICENSE)

## Test Status

[![stable toolchain unit tests](https://github.com/chrissimpkins/vectora/actions/workflows/stable-unittests.yml/badge.svg)](https://github.com/chrissimpkins/vectora/actions/workflows/stable-unittests.yml)
[![beta toolchain unit tests](https://github.com/chrissimpkins/vectora/actions/workflows/beta-unittests.yml/badge.svg)](https://github.com/chrissimpkins/vectora/actions/workflows/beta-unittests.yml)
[![clippy lints](https://github.com/chrissimpkins/vectora/actions/workflows/lints.yml/badge.svg)](https://github.com/chrissimpkins/vectora/actions/workflows/lints.yml)
[![rustfmt check](https://github.com/chrissimpkins/vectora/actions/workflows/fmt.yml/badge.svg)](https://github.com/chrissimpkins/vectora/actions/workflows/fmt.yml)

## About

Vectora is a library for n-dimensional vector computation with real and complex scalar types. The main library entry point is the [`Vector`](https://docs.rs/vectora/latest/vectora/types/vector/struct.Vector.html) struct.  Please see the [Gettting Started Guide](https://docs.rs/vectora/latest/vectora/#getting-started) for a detailed library API overview with examples.

## User documentation

User documentation is available at https://docs.rs/vectora.

### Minimum Rust Version Compatibility Policy

This project parameterizes generics by constants and relies on the [constant generics feature support stabilized in Rust v1.51.0](https://github.com/rust-lang/rust/pull/79135).

The minimum supported `rustc` version is believed to be v1.51.0.

### Include Vectora in Your Project

Import the library in the `[dependencies]` section of your `Cargo.toml` file:

**Cargo.toml**

```toml
[dependencies]
vectora = "0.8.1"
```

## Developer documentation

### Contributing

#### Issues

The [issue tracker](https://github.com/chrissimpkins/vectora/issues) is available on the GitHub repository. Don't be shy. Please report any issues that you identify so that we can address them.

#### Source contributions

Contributions are welcomed.  Submit your changes as a GitHub pull request. Please add new tests for source contributions that our current test suite does not cover.

#### Clone the repository

```txt
git clone https://github.com/chrissimpkins/vectora.git
```

#### Testing

The project is tested with the latest GitHub Actions macOS, Linux (Ubuntu), and Windows environment runners using the stable and beta `rustc` toolchains.

##### Unit and doc test suite

Edit the source files, then run the unit and doc test suite locally with the command:

```txt
cargo test
```

##### Unit tests only

```txt
cargo test --lib
```

##### Doc tests only

```txt
cargo test --doc
```

##### Clippy lints

Clippy lints are not executed with the above commands.  Use the following to lint Rust source files with clippy:

```txt
cargo clippy -- -D warnings
```

##### Fuzzing

This crate supports [`cargo fuzz`](https://github.com/rust-fuzz/cargo-fuzz) + [`libFuzzer`](https://llvm.org/docs/LibFuzzer.html) based fuzzing with the nightly rustc toolchain in supported environments.

[Install the `rustc` nightly toolchain](https://rust-lang.github.io/rustup/concepts/channels.html#working-with-nightly-rust).

Then, install `cargo-fuzz` with:

```
cargo +nightly install -f cargo-fuzz
```

Edit the fuzz target source in the `fuzz/fuzz_vectora.rs` file and begin fuzzing with the command:

```
cargo +nightly fuzz run fuzz_vectora
```

Please see the [Fuzzing with cargo-fuzz chapter](https://rust-fuzz.github.io/book/cargo-fuzz.html) of the Rust Fuzz book for additional documentation.

#### Documentation contributions

The docs.rs documentation is authored in the Rust source files.  Edit the text and build a local version of the project documentation for review with the command:

```txt
cargo doc
```

The documentation `index.html` page can be found on the following relative path from the repository's root: `target/doc/vectora/index.html`.

Submit your doc edits as a GitHub pull request.

## Changes

Please see [CHANGELOG.md](CHANGELOG.md).

## License

Vectora is released under the [Apache License, v2.0](LICENSE).
