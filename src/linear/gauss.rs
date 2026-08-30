use crate::linear::matrix::Matrix;
use num_traits::Num;

/// Represents an elementary row operation for matrices.
///
/// This implements `Copy`, `Clone`, `Debug` and `Default` (which is no-op).
///
/// ```
/// # #[cfg(feature = "alloc")]
/// # {
/// # use cpge::linear::{Matrix, MatrixRowOperation};
/// let original = Matrix::from([[1, 2], [3, 4]]);
///
/// let steps = vec![MatrixRowOperation::Swap(0, 1)];
/// let mut m = original.replay_steps(&steps);
/// assert_eq!(m, Matrix::from([[3, 4], [1, 2]]));
///
/// m.replay_steps_mut(&steps); // revert to original
/// assert_eq!(m, original);
/// # }
/// ```
#[derive(Copy, Clone, Debug)]
pub enum MatrixRowOperation<T>
where
    T: Default + Copy + Num,
{
    Swap(usize, usize),
    Add(usize, T, usize),
    Mul(usize, T),
}

impl<T: Default + Copy + Num> Default for MatrixRowOperation<T> {
    fn default() -> Self { Self::Swap(0, 0) }
}

impl<T> MatrixRowOperation<T>
where
    T: Default + Copy + Num,
{
    pub fn as_matrix<const N: usize>(self) -> Matrix<T, N> {
        let mut i = Matrix::identity();
        i.view_mut(..).replay_steps_mut(&[self]);
        i
    }

    pub fn composition_matrix<const N: usize>(steps: &[Self]) -> Matrix<T, N> {
        steps.iter().rfold(Matrix::identity(), |acc, &l| {
            l.as_matrix::<N>() * acc
        })
    }
}
