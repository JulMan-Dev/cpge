use core::hint::cold_path;
use arrayvec::ArrayVec;
use crate::linear::basis::VectorBasis;
use crate::linear::matrix::Matrix;
use crate::linear::vector::Vector;
use num_traits::{ConstOne, ConstZero, Float, Num};
use crate::linear::{LinearSolver, VectorFamily};
use crate::mem::Owned;

#[derive(Debug, Clone)]
pub struct Application<T, const F: usize, const R: usize = F>
where
    T: Default + Copy + Num
{
    pub matrix: Matrix<T, R, F>,
    pub input_basis: VectorBasis<T, F>,
    pub output_basis: VectorBasis<T, R>,
}

impl<T, const F: usize, const R: usize> Application<T, F, R>
where
    T: Default + Copy + Num,
{
    pub fn from_fn_with_standard_basis<Func>(func: &Func) -> Self
    where
        T: ConstOne + ConstZero,
        Func: Fn(Vector<T, F>) -> Vector<T, R>
    {
        let mut images = [Vector::ZERO; F];
        let basis = VectorBasis::<T, F>::standard_basis();

        for (i, &vector) in basis.vectors().iter().enumerate() {
            let image = func(vector.into());

            images[i] = image;
        }

        let matrix = Matrix::from_vectors(&images);

        Self {
            matrix,
            input_basis: basis,
            output_basis: VectorBasis::standard_basis(),
        }
    }

    /// Gets the kernel of this application with respect to the input and output basis.
    pub fn kernel(&self) -> Owned<VectorFamily<T, F>, ArrayVec<Vector<T, F>, F>>
    where
        T: Float,
    {
        let solutions = LinearSolver::solve_homogenous(&self.matrix);

        use crate::linear::LinearSolutions::*;
        match solutions {
            EntireSpace => VectorFamily::standard_basis(),
            Infinite { kernel, particular } => {
                assert!(particular.is_near_zero(), "particular must be zero here");
                Owned::new(kernel)
            }
            Unique(particular) => {
                assert!(particular.is_near_zero(), "particular must be zero here");
                VectorFamily::only_zero()
            },
            // very unlikely to be encored (never)
            None => {
                cold_path();
                VectorFamily::empty()
            },
        }
    }
}

impl<T, const N: usize> Application<T, N>
where
    T: Default + Copy + Num,
{
    pub const fn const_identity() -> Self
    where
        T: ConstZero + ConstOne,
    {
        Self {
            matrix: Matrix::const_identity(),
            input_basis: VectorBasis::standard_basis(),
            output_basis: VectorBasis::standard_basis(),
        }
    }
}
