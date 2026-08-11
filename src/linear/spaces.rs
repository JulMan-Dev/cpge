//! The traits for spaces (inner dot product, normed and metric spaces).

use num_traits::{Float, Num};
use crate::linear::Vector;

/// This trait defines the dot product for inner product spaces.
///
/// This implies `Self: NormedVectorSpace<N> + MetricSpace<N>`.
///
/// # Example
///
/// ```
/// # use cpge::linear::Vector;
/// // i32 implements InnerDotProduct for all N.
/// let u = Vector::new([1, 2, 3]);
/// let v = Vector::new([3, 2, 1]);
/// assert_eq!(u.dot(&v), 10);
/// ```
pub trait InnerDotProductSpace<const N: usize>
where
    Self: Default + Copy + Num,
{
    type DotOutput: Default + Copy + Num;

    /// This computes the dot product for vectors. It must follow some rule:
    ///
    /// * Symmetric: for all (u, v) in `Self ^ N`, [`dot(u, v)`](Self::dot) =
    ///   [`dot(v, u)`](Self::dot).
    fn dot(lhs: &Vector<Self, N>, rhs: &Vector<Self, N>) -> Self::DotOutput;
}

/// This trait defines normed vector spaces. It adds the norm operation, allowing to get the norm
/// for any vector in the space. This implies the zero vector is the origin for all
/// computations.
///
/// This implies `Self: MetricSpace<N>`.
///
/// # Example
///
/// ```
/// # use cpge::linear::Vector;
/// // f64 implements NormedVectorSpace for all N.
/// let u = Vector::new([1.0f64, 2.0, 3.0]);
/// let n = (1.0f64 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0).sqrt();
/// assert!((u.norm() - n).abs() <= f64::EPSILON);
/// ```
pub trait NormedVectorSpace<const N: usize>
where
    Self: Default + Copy + Num,
{
    type Norm: Default + Copy + Num;

    fn norm(this: &Vector<Self, N>) -> Self::Norm;
}

/// The trait defines metric spaces. It adds the distance operation, allowing to get the distance
/// between any vector in the space. Metric spaces may not have origin, only distance is enforced.
/// A metric space with zero as origin and a distance operation is a normed vector space.
///
/// # Example
///
/// ```
/// # use cpge::linear::Vector;
/// // f64 implements MetricSpace for all N.
/// let u = Vector::new([1.0f64, 0.0, 0.0]);
/// let v = Vector::new([0.0, 1.0, 0.0]);
/// assert_eq!(u.distance_from(&v), 2.0f64.sqrt());
/// ```
pub trait MetricSpace<const N: usize>
where
    Self: Default + Copy + Num,
{
    type Distance: Default + Copy + Num;

    fn distance(lhs: &Vector<Self, N>, rhs: &Vector<Self, N>) -> Self::Distance;
}

/// This is a mathematical implication, it cannot be overridden.
impl<T, const N: usize> NormedVectorSpace<N> for T
where
    T: InnerDotProductSpace<N>,
    T::DotOutput: Float,
{
    type Norm = T::DotOutput;

    fn norm(this: &Vector<Self, N>) -> Self::Norm {
        T::dot(this, this).sqrt()
    }
}

/// This is a mathematical implication, it cannot be overridden.
impl<T, const N: usize> MetricSpace<N> for T
where
    T: NormedVectorSpace<N>,
{
    type Distance = T::Norm;

    fn distance(lhs: &Vector<Self, N>, rhs: &Vector<Self, N>) -> Self::Distance {
        let diff = lhs - rhs;
        T::norm(&diff)
    }
}

macro_rules! impl_dot_product {
    ($($self:ty)+) => {
        #[doc(hidden)]
        mod internal_dot_product_impl {
            use ::num_traits::ConstZero;
            use crate::linear::{InnerDotProductSpace, Vector};

            $(impl<const N: usize> InnerDotProductSpace<N> for $self {
                type DotOutput = Self;

                fn dot(lhs: &Vector<Self, N>, rhs: &Vector<Self, N>) -> Self::DotOutput {
                    ::core::iter::Iterator::zip(lhs.scalars.iter(), rhs.scalars.iter())
                        .map(|(x, y)| x * y)
                        .fold(ConstZero::ZERO, |acc, x| acc + x)
                    // using ConstZero::ZERO avoids using 0 for integers and 0.0 for floats,
                    // avoiding defining two macros.
                }
            })+
        }
    };
}

impl_dot_product!(i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64);
