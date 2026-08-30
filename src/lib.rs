//! The crate for CPGE (French acronym for *Classe Préparatoire aux Grandes Écoles*) stuffs. Contains for now
//! mathematics tools (matrix, vectors...).

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "alloc")]
extern crate alloc;
extern crate core;
#[cfg(feature = "std")]
extern crate std;

pub mod linear;
pub mod geometry;
pub mod traits;
pub mod polynomials;
pub mod combinatorial;
#[cfg(test)]
pub mod testing;
pub mod complex;
pub mod iter;
pub mod mem;
#[cfg(feature = "alloc")]
pub mod function;
#[cfg(feature = "gl")]
pub mod gl;
