pub mod application;
pub mod vector;
pub mod matrix;
pub mod family;
pub mod gauss;
pub mod basis;
#[cfg(feature = "alloc")]
pub mod heap_matrix;
pub mod solver;
pub mod spaces;
pub mod space;
mod product;
pub mod iter;
#[cfg(feature = "alloc")]
pub mod ops;
pub mod convolution;
pub mod view;

// re-export
pub use application::*;
pub use vector::*;
pub use matrix::*;
pub use family::*;
pub use gauss::*;
pub use basis::*;
#[cfg(feature = "alloc")]
pub use heap_matrix::*;
pub use solver::*;
pub use spaces::*;
pub use space::*;
pub use iter::*;
#[cfg(feature = "alloc")]
pub use ops::*;
pub use convolution::*;
pub use view::*;
