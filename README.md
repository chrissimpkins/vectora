<img src="https://raw.githubusercontent.com/chrissimpkins/vectora/img/img/vectora.png" width="350">

## A Rust vector computation library

![Crates.io](https://img.shields.io/crates/v/vectora)
![docs.rs](https://img.shields.io/docsrs/vectora)
![GitHub](https://img.shields.io/github/license/chrissimpkins/vectora)

## Test Status

[![stable toolchain unit tests](https://github.com/chrissimpkins/vectora/actions/workflows/stable-unittests.yml/badge.svg)](https://github.com/chrissimpkins/vectora/actions/workflows/stable-unittests.yml)
[![beta toolchain unit tests](https://github.com/chrissimpkins/vectora/actions/workflows/beta-unittests.yml/badge.svg)](https://github.com/chrissimpkins/vectora/actions/workflows/beta-unittests.yml)
[![clippy lints](https://github.com/chrissimpkins/vectora/actions/workflows/lints.yml/badge.svg)](https://github.com/chrissimpkins/vectora/actions/workflows/lints.yml)
[![rustfmt check](https://github.com/chrissimpkins/vectora/actions/workflows/fmt.yml/badge.svg)](https://github.com/chrissimpkins/vectora/actions/workflows/fmt.yml)

## User documentation

User documentation is available at https://docs.rs/vectora.

Vectora is a library for n-dimensional vector computation. The main library entry point is the [`Vector`](#) struct.  Please see the [Gettting Started guide](#) for a detailed library API overview with examples.

## Developer documentation

### Contributing

#### Issues

The [issue tracker](https://github.com/chrissimpkins/vectora/issues) is available on the GitHub repository. Don't be shy. Please report any issues that you identify so that we can address them.

#### Source contributions

Contributions are welcomed.  Submit your changes as a GitHub pull request. Please add new tests for source contributions that are not covered by our current test suite.

#### Clone the repository

```txt
git clone https://github.com/chrissimpkins/vectora.git
```

#### Testing

Edit the source files, then run the unit tests and doc tests with the command:

```txt
cargo test
```

Run clippy lints with:

```txt
cargo clippy -- -D warnings
```

#### Documentation contributions

The docs.rs documentation is authored in the Rust source files.  Edit the text and build a local version of the project documentation for review with the command:

```txt
cargo doc
```

The documentation `index.html` page can be found on the following relative path from the root of the repository: `target/doc/vectora/index.html`.

Submit your doc edits as a GitHub pull request.

## License

Vectora is released under the [Apache License, v2.0](LICENSE).
