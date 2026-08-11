use crate::linear::heap_matrix::HeapMatrix;
use crate::linear::view::MatrixView;
use crate::linear::{Matrix, Vector};
use alloc::{boxed::Box, rc::Rc};
use alloc::borrow::ToOwned;
use core::ops::Deref;
use num_traits::Num;

/// This trait describes a type that can be freely transposed to matrix back and forth.
///
/// The implementation must follow some rules:
///  * It should be fast and efficient, at most O(n), or considered O(1) if R and C.
///  * It must guarantee `T::from_matrix(x.to_matrix()) == x` and
///    `T::from_matrix(matrix).to_matrix() == matrix`.
///  * It must not be unsafe.
///  * It should panic if and only if the reverse operation was incorrect.
pub trait TransparentMatrix {
    /// The cell type for matrix.
    type MatrixItem: Default + Copy + Num;

    /// This type must be an equivalent to `Self` but owned if `Self` is a reference or a pointer.
    type ResultType;

    /// Interprets `self` as a matrix.
    fn into_matrix(self) -> HeapMatrix<Self::MatrixItem>;

    /// Interprets the matrix as it was an instance of `Self`.
    fn from_matrix(matrix: &MatrixView<Self::MatrixItem>) -> Self::ResultType;
}

// TransparentMatrix implementation for references and smart pointers

impl<T> TransparentMatrix for &T
where
    T: TransparentMatrix<ResultType = T> + Clone,
{
    type MatrixItem = T::MatrixItem;
    type ResultType = T;

    fn into_matrix(self) -> HeapMatrix<Self::MatrixItem> { T::into_matrix(self.clone()) }

    fn from_matrix(matrix: &MatrixView<Self::MatrixItem>) -> Self::ResultType {
        T::from_matrix(matrix)
    }
}

impl<T> TransparentMatrix for &mut T
where
    T: TransparentMatrix<ResultType = T> + Clone,
{
    type MatrixItem = T::MatrixItem;
    type ResultType = T;

    fn into_matrix(self) -> HeapMatrix<Self::MatrixItem> { T::into_matrix(self.clone()) }

    fn from_matrix(matrix: &MatrixView<Self::MatrixItem>) -> Self::ResultType {
        T::from_matrix(matrix)
    }
}

/// For boxes, we do not enforce `T: Clone` since this trait defines methods that own parameters.
impl<T> TransparentMatrix for Box<T>
where
    T: TransparentMatrix<ResultType = T>,
{
    type MatrixItem = T::MatrixItem;
    type ResultType = T;

    fn into_matrix(self) -> HeapMatrix<Self::MatrixItem> { T::into_matrix(*self) }

    fn from_matrix(matrix: &MatrixView<Self::MatrixItem>) -> Self::ResultType {
        T::from_matrix(matrix)
    }
}

impl<T> TransparentMatrix for Rc<T>
where
    T: TransparentMatrix<ResultType = T> + Clone,
{
    type MatrixItem = T::MatrixItem;
    type ResultType = T;

    fn into_matrix(self) -> HeapMatrix<Self::MatrixItem> { T::into_matrix(self.deref().clone()) }

    fn from_matrix(matrix: &MatrixView<Self::MatrixItem>) -> Self::ResultType {
        T::from_matrix(matrix)
    }
}

impl<T, const R: usize, const C: usize> TransparentMatrix for Matrix<T, R, C>
where
    T: Default + Copy + Num,
{
    type MatrixItem = T;
    type ResultType = Self;

    fn into_matrix(self) -> HeapMatrix<Self::MatrixItem> { self.into_heap() }

    fn from_matrix(matrix: &MatrixView<Self::MatrixItem>) -> Self::ResultType {
        matrix.to_matrix().expect("invalid dimensions")
    }
}

impl<T> TransparentMatrix for HeapMatrix<T>
where
    T: Default + Copy + Num,
{
    type MatrixItem = T;
    type ResultType = Self;

    fn into_matrix(self) -> HeapMatrix<Self::MatrixItem> {
        self
    }

    fn from_matrix(matrix: &MatrixView<Self::MatrixItem>) -> Self::ResultType {
        matrix.to_owned()
    }
}

// we won't implement TransparentMatrix for VectorBasis as it may be unsafe.

impl<T, const N: usize> TransparentMatrix for Vector<T, N>
where
    T: Default + Copy + Num,
{
    type MatrixItem = T;
    type ResultType = Self;

    fn into_matrix(self) -> HeapMatrix<Self::MatrixItem> {
        let data = Box::new(self.scalars);

        HeapMatrix::new(data, N, 1)
    }

    fn from_matrix(matrix: &MatrixView<Self::MatrixItem>) -> Self::ResultType {
        let matrix: Matrix<T, N, 1> = matrix.to_matrix().expect("invalid dimensions");

        Self {
            scalars: matrix.data.as_flattened().try_into().expect("invalid dimensions")
        }
    }
}
