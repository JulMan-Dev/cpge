use crate::linear::family::VectorFamily;
use crate::linear::matrix::Matrix;
use crate::linear::vector::Vector;
use num_traits::{ConstOne, ConstZero, Num};
use core::borrow::Borrow;
use core::mem;

#[derive(Clone, Debug)]
pub struct VectorBasis<T, const N: usize>
where
    T: Default + Copy + Num,
{
    /// an array of vectors
    vectors: [[T; N]; N],
}

impl<T, const N: usize> VectorBasis<T, N>
where
    T: Default + Copy + Num,
{
    /// This function is unsafe because it doesn't safe the vectors passed form
    /// a basis. The caller must check before creating one.
    ///
    /// # Safety
    ///
    /// The caller must ensure `vectors` represents a basis.
    pub const unsafe fn new(vectors: [[T; N]; N]) -> Self {
        Self { vectors }
    }

    pub const fn vectors(&self) -> &[[T; N]; N] {
        &self.vectors
    }

    /// This function is unsafe because it lets the caller changing the scalars of
    /// the vectors, potentially making this not a basis anymore.
    ///
    /// # Safety
    ///
    /// The caller must ensure `self` remains a basis after mutation.
    pub const unsafe fn vectors_mut(&mut self) -> &mut [[T; N]; N] {
        &mut self.vectors
    }

    pub const fn standard_basis() -> Self
    where
        T: ConstOne + ConstZero,
    {
        let vectors = VectorFamily::<T, N>::const_standard_basis();

        // SAFETY: both share the same memory layout
        let vectors = unsafe {
            mem::transmute_copy(&vectors)
        };

        Self { vectors }
    }

    pub const fn as_matrix(&self) -> Matrix<T, N> {
        Matrix::from_array(self.vectors).transpose()
    }
}

impl<T, const N: usize> Borrow<[Vector<T, N>; N]> for VectorBasis<T, N>
where
    T: Default + Copy + Num,
{
    fn borrow(&self) -> &[Vector<T, N>; N] {
        // SAFETY: both share the same memory layout
        unsafe { mem::transmute_copy(&self.vectors) }
    }
}
