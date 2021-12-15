# Changelog

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
