use crate::linear::matrix::Matrix;
use crate::linear::vector::Vector;
use crate::linear::VectorFamily;
use arrayvec::ArrayVec;
use core::fmt::Debug;
use core::marker::PhantomData;
use num_traits::{Float, Num, Zero};

#[derive(Clone, Debug)]
pub enum LinearSolutions<T, const D: usize>
where
    T: Default + Copy + Num,
{
    EntireSpace,
    Infinite {
        particular: Vector<T, D>,
        kernel: ArrayVec<Vector<T, D>, D>,
    },
    Unique(Vector<T, D>),
    None,
}

impl<T, const D: usize> LinearSolutions<T, D>
where
    T: Default + Copy + Num,
{
    /// Checks if `vec` is a solution. This requires `T` to be `Float` because it uses
    /// [`gaussian_elimination`](crate::linear::MatrixView::gaussian_elimination).
    pub fn is_in(&self, vec: &Vector<T, D>) -> bool
    where
        T: Float,
    {
        match self {
            Self::EntireSpace => true,
            Self::Infinite { particular, kernel } => {
                let d = vec.clone() - particular.clone();
                // checking if d is in the kernel
                let mut kernel = kernel.clone();
                kernel.push(d);
                let kernel: &VectorFamily<T, D> = kernel.as_ref().into();

                !kernel.is_independent()
            },
            Self::Unique(v) => (vec.clone() - v.clone()).is_near_zero(),
            Self::None => false,
        }
    }
}

/// The linear solver. It may solve any linear equations.
pub struct LinearSolver<T, const R: usize, const C: usize = R> {
    marker: PhantomData<T>,
}

impl<T, const R: usize, const C: usize> LinearSolver<T, R, C>
where
    T: Default + Copy + Num,
{
    /// Solves the homogenous system through its matrix.
    ///
    /// Note: The zero vector is always a solution of such systems.
    ///
    /// # Example
    ///
    /// ```
    /// # use cpge::linear::{LinearSolver, LinearSolutions, Matrix, Vector};
    /// # use std::assert_matches;
    /// # use num_traits::Zero;
    /// let system = Matrix::<f64, 3>::const_identity();
    /// let solution = LinearSolver::solve_homogenous(&system);
    /// assert_matches!(solution, LinearSolutions::Unique(_)); // only one solution,
    /// assert!(solution.is_in(&Zero::zero())); // and it's zero.
    ///
    /// let system = Matrix::from([
    ///     [1.0, 2.0, 1.0],
    ///     [2.0, 4.0, 2.0],
    /// ]);
    /// let solution = LinearSolver::solve_homogenous(&system);
    /// assert!(solution.is_in(&Zero::zero()));
    /// assert!(solution.is_in(&Vector::new([-2.0, 1.0, 0.0])));
    /// ```
    pub fn solve_homogenous(matrix: &Matrix<T, R, C>) -> LinearSolutions<T, C>
    where
        T: Float,
    {
        let rref = matrix.rref();

        if rref.rank_float() == C {
            // easy, only 0 is solution
            return LinearSolutions::Unique(Zero::zero());
        }

        if rref.rank_float() == 0 {
            // easy, everything is solution
            return LinearSolutions::EntireSpace;
        }

        // this will map for k in [0, C):
        // * None: no pivot found
        // * Some(i): pivot found at (i, k)
        let pivots: [Option<usize>; C] = {
            let mut out = [None; C];
            let mut last_k = None;

            for (i, row) in rref.rows().enumerate() {
                let found = row.iter()
                    .enumerate()
                    .skip(last_k.unwrap_or(0))
                    .find(|(_, x)| x.abs() > T::epsilon());

                match found {
                    Some((k, _)) => {
                        last_k.replace(k);
                        out[k].replace(i);
                    },
                    // zero row, all rows after are also zeroes, skipping
                    None => break,
                }
            }

            out
        };

        let mut kernel = ArrayVec::new();
        let mut pending = ArrayVec::<usize, C>::new();

        for item in pivots.iter().enumerate() {
            match item {
                (_, &Some(row)) => pending.push(row),
                (i, None) => {
                    let mut vector = [T::zero(); C];
                    vector[i] = T::one();

                    // dealing with pending pivots
                    for &row in &pending {
                        vector[row] = -rref[(row, i)];
                    }

                    kernel.push(Vector::new(vector));
                }
            }
        }

        LinearSolutions::Infinite { particular: Zero::zero(), kernel }
    }

    pub fn solve_with_vector(matrix: &Matrix<T, R, C>, right: &Vector<T, R>) -> LinearSolutions<T, C>
    where
        T: Float,
    {
        match Self::solve_homogenous(matrix) {
            LinearSolutions::EntireSpace => {
                if right.is_near_zero() {
                    // 0x0 + 0x1 + ... + 0xn = 0
                    // ...
                    // always true
                    LinearSolutions::EntireSpace
                } else {
                    LinearSolutions::None
                }
            }
            LinearSolutions::Infinite { particular: _, kernel: _ } => {
                todo!()
            }
            LinearSolutions::Unique(_) => { todo!() }
            LinearSolutions::None => { todo!() }
        }
    }

    #[allow(unused)]
    fn find_particular_solution(matrix: &Matrix<T, R, C>, right: &Vector<T, R>) -> Option<Vector<T, C>>
    where
        T: Float,
    {
        let (_rref, _right) = matrix.rref_with_right(right);

        todo!();
    }
}

        /*
        let mut prev_i = None;

            'a: for (k, u) in out.iter_mut().enumerate() {
                let col = &mut matrix.col(k);

                // checking if rows before prev_i are all zeroes.
                let (maybe_pivot, shift) = match prev_i {
                    None => (true, 0),
                    Some(i) => {
                        (col.take(i + 1).all(|x| x.abs() <= T::epsilon()), i + 1)
                    }
                };

                if !maybe_pivot {
                    // skipping further verification
                    continue;
                }

                // we look for [X, 0, ... 0] where X != 0
                let mut last_nonzero = None;

                loop {
                    let i = col.enumerate().find_map(|(i, &x)| {
                        (x.abs() > T::epsilon()).then_some(i)
                    });

                    match (i, &mut last_nonzero) {
                        (None, _) => break,
                        (Some(i), p @ None) => {
                            // "i" is 0 when "real i" is "shift"
                            p.replace(i + shift);
                        },
                        // found [X, ..., Y, ...] where X != 0 and Y != 0, not a pivot
                        (Some(i), Some(_)) => continue 'a,
                    };
                }

                let Some(last_nonzero) = last_nonzero else {
                    // column of zero, whatever
                    continue;
                };

                u.replace(last_nonzero);
                prev_i.replace(last_nonzero);
            }

            out
        };
    */
