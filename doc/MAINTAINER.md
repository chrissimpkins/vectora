# Maintainer Documentation

## Releases

1. Update the release version number in `Cargo.toml`, `README.md` (docs), and `src/lib.rs` (docs)
2. Push a git release tag
3. Publish to crates.io

### Update version number

The `[dependencies]` section version number update in the project documentation is automated with `cargo-release`:

```txt
cargo release --no-publish --no-push --no-tag --execute [SEMVER VERSION NUMBER]
```

### Publish to crates.io

```txt
cargo publish
```
