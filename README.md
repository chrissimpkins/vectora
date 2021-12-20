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
vectora = "0.4.0"
```

## Developer documentation

### Contributing

#### ![L4 Header](https://via.placeholder.com/12/B01721/000000?text=+) Issues

The [issue tracker](https://github.com/chrissimpkins/vectora/issues) is available on the GitHub repository. Don't be shy. Please report any issues that you identify so that we can address them.

#### ![L4 Header](https://via.placeholder.com/12/B01721/000000?text=+) Source contributions

Contributions are welcomed.  Submit your changes as a GitHub pull request. Please add new tests for source contributions that our current test suite does not cover.

#### ![L4 Header](https://via.placeholder.com/12/B01721/000000?text=+) Clone the repository

```txt
git clone https://github.com/chrissimpkins/vectora.git
```

#### ![L4 Header](https://via.placeholder.com/12/B01721/000000?text=+) Testing

The project is tested with the latest GitHub Actions macOS, Linux (Ubuntu), and Windows environment runners using the stable and beta `rustc` toolchains.

Edit the source files, then run the unit and doc test suite locally with the command:

```txt
cargo test
```

Run clippy lints with:

```txt
cargo clippy -- -D warnings
```

#### ![L4 Header](https://via.placeholder.com/12/B01721/000000?text=+) Documentation contributions

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
