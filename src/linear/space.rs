//! The vector space module.
//!
//! This implements vector space of `T ^ n`.

use crate::complex::BasicComplex;
use crate::linear::family::VectorFamily;
use crate::linear::gauss::MatrixRowOperation;
use crate::linear::matrix::Matrix;
use crate::linear::vector::Vector;
use crate::linear::{InnerDotProductSpace, MetricSpace, NormedVectorSpace};
use crate::mem::{AbstractVec, Owned};
use arrayvec::ArrayVec;
use core::array::from_ref;
use num_traits::{Float, Zero};

#[derive(Debug)]
pub struct VectorSpace<T, const N: usize, const CAP: usize>
where
    T: Default + Copy + Float,
{
    pub family: Owned<VectorFamily<T, N>, ArrayVec<Vector<T, N>, CAP>>,
    // independent family version of `family`, used to check inclusion
    minimal_space: ArrayVec<Vector<T, N>, N>,
    // allowing faster inclusion check
    rref_steps: Matrix<T, N>,
}

pub type ComplexVectorSpace<T, const N: usize, const CAP: usize> = VectorSpace<BasicComplex<T>, N, CAP>;

impl<T, const N: usize, const CAP: usize> VectorSpace<T, N, CAP>
where
    T: Default + Copy + Float,
{
    pub fn new(family: Owned<VectorFamily<T, N>, ArrayVec<Vector<T, N>, CAP>>) -> Self {
        let minimal = family.minimal_independent();
        let steps = {
            let mut vectors = minimal.to_inner().clone();
            vectors.resize(N, Vector::zero());
            let array: &[_; N] = vectors.as_array().expect("we just resized");

            // we know it will produce kinda identity, so we can just take the steps and ignore
            // the result matrix
            let mut matrix = Matrix::from_vectors(array);
            matrix.record_rref_mut()
        };

        Self {
            family,
            minimal_space: minimal.into_inner(),
            rref_steps: MatrixRowOperation::composition_matrix(&steps),
        }
    }

    /// Checks if `self` is a line.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cpge::linear::VectorSpace;
    /// const CAP: usize = 10; // indicative
    ///
    /// let space = VectorSpace::<f64, 2, CAP>::from(&[
    ///     [1.0, 2.0]
    /// ][..]);
    /// assert!(space.is_line());
    ///
    /// let space = VectorSpace::<f64, 2, CAP>::from(&[
    ///     [1.0, 0.0],
    ///     [2.0, 0.0]
    /// ][..]);
    /// assert!(space.is_line());
    ///
    /// let space = VectorSpace::<f64, 2, CAP>::from(&[
    ///     [1.0, 0.0],
    ///     [1.0, 1.0]
    /// ][..]);
    /// assert!(!space.is_line());
    /// ```
    pub const fn is_line(&self) -> bool {
        self.minimal_space.len() == 1
    }

    /// Checks if `self` is a plane.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cpge::linear::VectorSpace;
    /// let space = VectorSpace::<f64, 3, 3>::from(&[
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    /// ][..]);
    /// assert!(space.is_plane());
    ///
    /// let space = VectorSpace::<f64, 2, 2>::from(&[
    ///     [1.0, 0.0],
    /// ][..]);
    /// assert!(space.is_line() && space.is_plane()); // both a plane and a line
    /// ```
    pub const fn is_plane(&self) -> bool {
        self.minimal_space.len() == N - 1
    }

    /// Checks if `self` spans the entire universe, `T ^ n`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cpge::linear::VectorSpace;
    /// let space: VectorSpace<f64, 3, 3> = VectorSpace::whole_space();
    /// assert!(space.is_whole_space(), "a");
    ///
    /// let space = VectorSpace::<f64, 2, 2>::from(&[
    ///     [1.0, 0.0],
    ///     [0.0, 2.0],
    /// ][..]);
    /// assert!(space.is_whole_space(), "b");
    /// ```
    pub const fn is_whole_space(&self) -> bool {
        self.minimal_space.len() == N
    }

    /// Returns a vector space that generates the whole `T ^ n`.
    ///
    /// It uses the [`standard_basis`](VectorFamily::standard_basis).
    pub fn whole_space() -> Self {
        assert!(CAP >= N, "CAP is lower than N, cannot store standard basis");

        let family = VectorFamily::standard_basis();
        let copy: ArrayVec<Vector<T, N>, CAP> = ArrayVec::try_from(family.to_inner().as_ref())
            .unwrap();
        let copy = Owned::new(copy);

        Self {
            family: copy,
            minimal_space: family.into_inner(),
            // no steps to reduce identity matrix
            rref_steps: Matrix::identity(),
        }
    }

    pub fn is_in(&self, vec: &Vector<T, N>) -> bool {
        let rref = {
            let mut vectors = self.minimal_space.clone();
            vectors.resize(N, Vector::zero());
            let array: &[_; N] = vectors.as_array().unwrap();

            self.rref_steps.clone() * Matrix::from_vectors(array)
        };
        let rref_right = self.rref_steps.clone() * Matrix::from_vectors(from_ref(vec));

        for (i, row) in rref.rows().enumerate().rev() {
            let is_zero = row.iter().all(|&x| x.abs() <= T::epsilon());

            if is_zero && rref_right[(i, 0)].abs() > T::epsilon() {
                // not possible, returning false
                return false;
            }
        }

        true
    }

    pub fn dot(lhs: &Vector<T, N>, rhs: &Vector<T, N>) -> T::DotOutput
    where
        T: InnerDotProductSpace<N>,
    {
        T::dot(lhs, rhs)
    }

    pub fn norm(this: &Vector<T, N>) -> T::Norm
    where
        T: NormedVectorSpace<N>,
    {
        T::norm(this)
    }

    pub fn distance(lhs: &Vector<T, N>, rhs: &Vector<T, N>) -> T::Distance
    where
        T: MetricSpace<N>,
    {
        T::distance(lhs, rhs)
    }
}

impl<T, const N: usize, const CAP: usize> From<&[[T; N]]> for VectorSpace<T, N, CAP>
where
    T: Default + Copy + Float,
{
    fn from(value: &[[T; N]]) -> Self {
        let vectors = value.iter()
            .map(|&scalars| Vector { scalars })
            .collect();

        Self::new(Owned::new(vectors))
    }
}

impl<T, const N: usize, const CAP: usize> From<&[Vector<T, N>]> for VectorSpace<T, N, CAP>
where
    T: Default + Copy + Float,
{
    fn from(value: &[Vector<T, N>]) -> Self {
        Self::new(Owned::new(ArrayVec::try_from(value).unwrap()))
    }
}


mod heap {
    use alloc::boxed::Box;
    use core::array::from_ref;
    use num_traits::{Float, Zero};
    use crate::complex::BasicComplex;
    use crate::linear::{InnerDotProductSpace, Matrix, MatrixRowOperation, MetricSpace, NormedVectorSpace, Vector, VectorFamily};
    use crate::mem::Owned;

    #[derive(Debug)]
    pub struct VectorSpaceHeap<T, const N: usize>
    where
        T: Default + Copy + Float,
    {
        pub family: Owned<VectorFamily<T, N>, Box<[Vector<T, N>]>>,
        // independent family version of `family`, used to check inclusion
        minimal_space: Box<[Vector<T, N>]>,
        // allowing faster inclusion check
        rref_steps: Matrix<T, N>,
    }

    impl<T, const N: usize> VectorSpaceHeap<T, N>
    where
        T: Default + Copy + Float,
    {
        pub fn new(family: Owned<VectorFamily<T, N>, Box<[Vector<T, N>]>>) -> Self {
            let minimal = family.minimal_independent_heap();
            let steps = {
                let mut vectors = minimal.to_inner().clone().into_vec();
                vectors.resize(N, Vector::zero());
                let array: &[_; N] = vectors.as_array().expect("we just resized");

                // we know it will produce kinda identity, so we can just take the steps and ignore
                // the result matrix
                let mut matrix = Matrix::from_vectors(array);
                matrix.record_rref_mut()
            };

            Self {
                family,
                minimal_space: minimal.into_inner(),
                rref_steps: MatrixRowOperation::composition_matrix(&steps),
            }
        }

        /// Checks if `self` is a line.
        ///
        /// # Examples
        ///
        /// ```
        /// # use cpge::linear::VectorSpaceHeap;
        /// let space = VectorSpaceHeap::<f64, 2>::from(&[
        ///     [1.0, 2.0]
        /// ][..]);
        /// assert!(space.is_line());
        ///
        /// let space = VectorSpaceHeap::from(&[
        ///     [1.0, 0.0],
        ///     [2.0, 0.0]
        /// ][..]);
        /// assert!(space.is_line());
        ///
        /// let space = VectorSpaceHeap::from(&[
        ///     [1.0, 0.0],
        ///     [1.0, 1.0]
        /// ][..]);
        /// assert!(!space.is_line());
        /// ```
        pub const fn is_line(&self) -> bool {
            self.minimal_space.len() == 1
        }

        /// Checks if `self` is a plane.
        ///
        /// # Examples
        ///
        /// ```
        /// # use cpge::linear::VectorSpaceHeap;
        /// let space = VectorSpaceHeap::from(&[
        ///     [1.0, 0.0, 0.0],
        ///     [0.0, 1.0, 0.0],
        /// ][..]);
        /// assert!(space.is_plane());
        ///
        /// let space = VectorSpaceHeap::from(&[
        ///     [1.0, 0.0],
        /// ][..]);
        /// assert!(space.is_line() && space.is_plane()); // both a plane and a line
        /// ```
        pub const fn is_plane(&self) -> bool {
            self.minimal_space.len() == N - 1
        }

        /// Checks if `self` spans the entire universe, `T ^ n`.
        ///
        /// # Examples
        ///
        /// ```
        /// # use cpge::linear::VectorSpaceHeap;
        /// let space: VectorSpaceHeap<f64, 3> = VectorSpaceHeap::whole_space();
        /// assert!(space.is_whole_space());
        ///
        /// let space = VectorSpaceHeap::from(&[
        ///     [1.0, 0.0],
        ///     [0.0, 2.0],
        /// ][..]);
        /// assert!(space.is_whole_space());
        /// ```
        pub const fn is_whole_space(&self) -> bool {
            self.minimal_space.len() == N
        }

        /// Returns a vector space that generates the whole `T ^ n`.
        ///
        /// It uses the [`standard_basis`](VectorFamily::standard_basis).
        pub fn whole_space() -> Self {
            let family = VectorFamily::standard_basis_heap();

            Self {
                family: family.clone(),
                minimal_space: family.into_inner(),
                // no steps to reduce identity matrix
                rref_steps: Matrix::identity(),
            }
        }

        pub fn is_in(&self, vec: &Vector<T, N>) -> bool {
            let rref = {
                let mut vectors = self.minimal_space.clone().into_vec();
                vectors.resize(N, Vector::zero());
                let array: &[_; N] = vectors.as_array().unwrap();

                self.rref_steps.clone() * Matrix::from_vectors(array)
            };
            let rref_right = self.rref_steps.clone() * Matrix::from_vectors(from_ref(vec));

            for (i, row) in rref.rows().enumerate().rev() {
                let is_zero = row.iter().all(|&x| x.abs() <= T::epsilon());

                if is_zero && rref_right[(i, 0)].abs() > T::epsilon() {
                    // not possible, returning false
                    return false;
                }
            }

            true
        }

        pub fn dot(lhs: &Vector<T, N>, rhs: &Vector<T, N>) -> T::DotOutput
        where
            T: InnerDotProductSpace<N>,
        {
            T::dot(lhs, rhs)
        }

        pub fn norm(this: &Vector<T, N>) -> T::Norm
        where
            T: NormedVectorSpace<N>,
        {
            T::norm(this)
        }

        pub fn distance(lhs: &Vector<T, N>, rhs: &Vector<T, N>) -> T::Distance
        where
            T: MetricSpace<N>,
        {
            T::distance(lhs, rhs)
        }
    }

    pub type ComplexVectorSpaceHeap<T, const N: usize> = VectorSpaceHeap<BasicComplex<T>, N>;

    impl<T, const N: usize> From<&[[T; N]]> for VectorSpaceHeap<T, N>
    where
        T: Default + Copy + Float,
    {
        fn from(value: &[[T; N]]) -> Self {
            let vectors = value.iter()
                .map(|&scalars| Vector { scalars })
                .collect();

            Self::new(Owned::new(vectors))
        }
    }

    impl<T, const N: usize> From<&[Vector<T, N>]> for VectorSpaceHeap<T, N>
    where
        T: Default + Copy + Float,
    {
        fn from(value: &[Vector<T, N>]) -> Self {
            Self::new(Owned::new(Box::from(value)))
        }
    }
}

pub use heap::*;
