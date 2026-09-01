# cpge

The crate for CPGE (French acronym for *Classe Préparatoire aux Grandes Écoles*) stuffs. Contains for now mathematics 
tools (matrix, vectors...).

Implemented when I was in CPGE during my first year for the base version. I will continue to implement new features or
fix bugs in my spare time.

The crate even works on `no_env` environment and without allocator, but some features may not work.

## Installation

The crate is not published on [Crates.io](https://crates.io/) yet. You need to clone the Git repository to add the
crate.

```bash
cargo add cpge --git https://github.com/JulMan-Dev/cpge
```

Note that this crate compiles some native code one installation, may sure you have the required dependencies installed:

 - macOS for `gl` feature: Xcode Command Line Tools (or Xcode)

## Crate features

This crate exports some features:

 - `gl`: Renderer feature (graphs...), not working yet
 - `alloc`: Standard allocator support, requires a global allocator
 - `std`: (default) Standard library support

## Implemented features

- Matrices (Gaussian elimination, RREF, determinant, inverse)
- Vectors (operations, family, partial family)
- Polynomials (operations, roots, derivatives, Taylor series)
- Complex numbers (operations, roots)
- Geometry (points, lines, planes)

And more to come!

## Roadmap and future

For the future for this crate and incoming features, refer to the roadmap issue opened.
