use crate::linear::matrix::Matrix;
use num_traits::{ConstZero, Float, Num, Zero};
use core::mem;
use core::ops::{Index, Mul};
use crate::linear::{InnerDotProductSpace, MetricSpace, NormedVectorSpace};

/// This is a column vector.
///
/// This implements `Clone`, `Debug`, if `T: Debug`, and `PartialEq`.
///
/// This can be used to create matrices.
///
/// # Example
///
/// ```
/// # use cpge::linear::{Vector, Matrix};
/// let vectors: [Vector<usize, 3>; 2] = [
///     Vector::new([1, 2, 3]),
///     Vector::new([4, 5, 6]),
/// ];
/// let matrix = Matrix::from_vectors(&vectors);
/// assert_eq!(matrix, Matrix::from([
///     [1, 4],
///     [2, 5],
///     [3, 6],
/// ]));
///
/// let row_matrix = Matrix::from_row_vectors(&vectors);
/// // here the vectors are treated as row vectors.
/// assert_eq!(row_matrix, Matrix::from([
///     [1, 2, 3],
///     [4, 5, 6],
/// ]));
///
/// let matrix = matrix.transpose();
/// // from_row_vectors(&v) = from_vectors(&v).transpose()
/// assert_eq!(matrix, row_matrix);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct Vector<T, const N: usize>
where
    T: Default + Copy + Num,
{
    pub scalars: [T; N],
}

macro_rules! impl_vector_ops {
    ($(%name $name:ident $fn_name:ident,)*) => {
        mod internal_impl_ops {
            use num_traits::Num;

            type V<T, const N: usize> = super::Vector<T, N>;

            $(#[inline(always)]
            fn $fn_name<T, const N: usize>(lhs: &V<T, N>, rhs: &V<T, N>) -> V<T, N>
            where
                T: Default + Copy + Num,
            {
                let mut result = lhs.clone();

                for i in 0..N {
                    result.scalars[i] = core::ops::$name::$fn_name(lhs.scalars[i], rhs.scalars[i]);
                }

                result
            }

            impl<T, const N: usize> core::ops::$name for V<T, N>
            where
                T: Default + Copy + Num,
            {
                type Output = Self;

                fn $fn_name(self, rhs: Self) -> Self::Output {
                    $fn_name(&self, &rhs)
                }
            }

            impl<T, const N: usize> core::ops::$name<&V<T, N>> for V<T, N>
            where
                T: Default + Copy + Num,
            {
                type Output = Self;

                fn $fn_name(self, rhs: &Self) -> Self::Output {
                    $fn_name(&self, rhs)
                }
            }

            impl<T, const N: usize> core::ops::$name<V<T, N>> for &V<T, N>
            where
                T: Default + Copy + Num,
            {
                type Output = V<T, N>;

                fn $fn_name(self, rhs: V<T, N>) -> Self::Output {
                    $fn_name(self, &rhs)
                }
            }

            impl<T, const N: usize> core::ops::$name for &V<T, N>
            where
                T: Default + Copy + Num,
            {
                type Output = V<T, N>;

                fn $fn_name(self, rhs: Self) -> Self::Output {
                    $fn_name(self, rhs)
                }
            })*
        }
    };
}

impl_vector_ops!(
    %name Add add,
    %name Sub sub,
);

impl<T, const N: usize> Mul<T> for Vector<T, N>
where
    T: Default + Copy + Num,
{
    type Output = Vector<T, N>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut result = self.clone();

        for i in 0..N {
            result.scalars[i] = self.scalars[i] * rhs;
        }

        result
    }
}

impl<T, const N: usize> Zero for Vector<T, N>
where
    T: Default + Copy + Num,
{
    fn zero() -> Self {
        Self { scalars: [T::zero(); N] }
    }

    fn is_zero(&self) -> bool {
        self.scalars == Self::zero().scalars
    }
}

impl<T, const N: usize> ConstZero for Vector<T, N>
where
    T: Default + Copy + Num + ConstZero,
{
    const ZERO: Self = Self { scalars: [T::ZERO; N] };
}

impl<T, const N: usize> Default for Vector<T, N>
where
    T: Default + Copy + Num,
{
    fn default() -> Self {
        Self { scalars: [T::default(); N] }
    }
}

impl<T, const N: usize> Index<usize> for Vector<T, N>
where
    T: Default + Copy + Num,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.scalars[index]
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T, N>
where
    T: Copy + Default + Num
{
    fn from(scalars: [T; N]) -> Self {
        Self { scalars }
    }
}

impl<T, const N: usize> Vector<T, N>
where
    T: Copy + Default + Num,
{
    pub const fn new(scalars: [T; N]) -> Self {
        Self { scalars }
    }

    pub fn dot(&self, rhs: &Self) -> T::DotOutput
    where
        T: InnerDotProductSpace<N>
    {
        T::dot(self, rhs)
    }

    pub fn norm_squared(&self) -> T::DotOutput
    where
        T: InnerDotProductSpace<N>
    {
        self.dot(self)
    }

    pub fn norm(&self) -> T::Norm
    where
        T: NormedVectorSpace<N>,
    {
        T::norm(self)
    }

    /// Normalizes the vector, divides each component by the vector norm.
    pub fn normalize(&self) -> Self
    where
        T: NormedVectorSpace<N> + Mul<T::Norm, Output = T>,
        T::Norm: Float,
    {
        let recip = self.norm().recip();
        let scalars = self.scalars.map(|x| x * recip);
        Self { scalars }
    }

    pub fn distance_from(&self, rhs: &Vector<T, N>) -> T::Distance
    where
        T: MetricSpace<N>,
    {
        T::distance(self, rhs)
    }

    #[doc(hidden)]
    pub(crate) fn is_near_zero(&self) -> bool
    where
        T: Float,
    {
        self.scalars.iter().all(|&x| x.abs() <= T::epsilon())
    }
}

impl<T> Mul for &Vector<T, 3>
where
    T: Copy + Default + Num,
{
    type Output = Vector<T, 3>;

    fn mul(self, rhs: Self) -> Self::Output {
        let ([a, b, c], [d, e, f]) = (self.scalars, rhs.scalars);

        [b * f - c * e, c * d - a * f, a * e - b * d].into()
    }
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Default + Copy + Num,
{
    /// Converts `self` matrix to a static array of column vectors.
    ///
    /// This is a heavy operation.
    pub const fn to_vectors(&self) -> [Vector<T, R>; C] {
        self.transpose().to_row_vectors()
    }
    
    /// Converts `self` matrix to a static array of row vectors.
    ///
    /// This is a free operation.
    pub const fn to_row_vectors(&self) -> [Vector<T, C>; R] {
        // SAFETY: Matrix::<T, R, C>::data and [Vector<T, C>; R]
        // shares the same memory layout
        unsafe { mem::transmute_copy(&self.data) }
    }

    pub const fn from_row_vectors(vectors: &[Vector<T, C>; R]) -> Self {
        // SAFETY: Matrix::<T, R, C>::data and [Vector<T, C>; R]
        // shares the same memory layout
        let &data: &[[T; C]; R] = unsafe { mem::transmute(vectors) };

        Self::from_array(data)
    }

    pub const fn from_vectors(vectors: &[Vector<T, R>; C]) -> Self {
        Matrix::from_row_vectors(vectors).transpose()
    }
}
