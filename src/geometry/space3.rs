use crate::linear::InnerDotProductSpace;
use crate::linear::family::VectorFamily;
use crate::linear::matrix::Matrix;
use crate::linear::vector::Vector;
use crate::mem::Owned;
use arrayvec::ArrayVec;
use num_traits::{Float, Num, Zero};

/// Represents a 3D plane. It could be either in Cartesian or in parametric notation.
///
/// You may need `T` to implement [`Float`], most all methods require this.
///
/// This uses vectors from [`linear`](crate::linear) module.
#[derive(Clone, Debug)]
pub enum Plane3<T>
where
    T: Default + Copy + Num + InnerDotProductSpace<3>,
{
    /// [a, b, c, d] in `ax + by + cy + d = 0`
    Cartesian([T; 4]),
    /// ((x0, y0, z0), ((a, b, c), (d, e, f))) in `x = x0 + at + dt' ; y = y0 + bt + et' ; z = z0 + ct + ft'`
    Parametric(Vector<T, 3>, (Vector<T, 3>, Vector<T, 3>)),
}

impl<T> Plane3<T>
where
    T: Default + Copy + Num + InnerDotProductSpace<3>,
{
    /// Forces `self` to be represented using [`Cartesian`](Plane3::Cartesian).
    ///
    /// Note: `T` must be a `Float` because this uses [`Matrix::gaussian_elimination`] which
    /// requires it.
    pub fn to_cartesian(&self) -> Self
    where
        T: Float,
    {
        let Self::Parametric(point, (u, v)) = self else {
            // already cartesian.
            return self.clone();
        };

        let (zero, one) = (T::zero(), T::one());

        // We need to do Gaussian reduction
        let mut matrix = Matrix::from_vectors(&[u.clone(), v.clone()]);
        let mut augmented = Matrix::<T, 3, 4>::from([
            [one, zero, zero, -point.scalars[0]],
            [zero, one, zero, -point.scalars[1]],
            [zero, zero, one, -point.scalars[2]],
        ]);

        // augmented will contain the coefficients for Cartesian once elimination done.
        // first we play on base `matrix`.
        let steps = matrix.record_gaussian_elimination_mut();

        if matrix.rank_float() < 2 {
            panic!("u and v must be linearly independent");
        }

        // replay on `augmented`
        augmented.replay_steps_mut(&steps);

        // taking the last row, it's the cartesian equation
        Self::Cartesian(augmented.data[2])
    }

    /// Forces `self` to be represented using [`Parametric`](Plane3::Parametric).
    ///
    /// Note: `T` must be a `Float` because this performs a division.
    pub fn to_parametric(&self) -> Self
    where
        T: Float,
    {
        let &Self::Cartesian([a, b, c, d]) = self else {
            // already parametric.
            return self.clone();
        };

        let (zero, minus_one) = (T::zero(), T::zero() - T::one());

        let normal = Vector::from([a, b, c]);
        debug_assert!(!normal.is_zero()); // avoid division by zero

        let u = Vector::from([b * minus_one, a, zero]);
        let v = &normal * &u; // cross

        // taking a point on the plane
        let point = Vector::from(if a.abs() > T::epsilon() {
            [-d / a, zero, zero]
        } else if b.abs() > T::epsilon() {
            [zero, -d / b, zero]
        } else {
            [zero, zero, -d / c]
        });

        Self::Parametric(point, (u, v))
    }

    /// Checks if the point is on the plane.
    pub fn is_in(&self, point: &Vector<T, 3>) -> bool {
        match self {
            Plane3::Cartesian([a, b, c, d]) => {
                // straight forward
                let [x, y, z] = point.scalars;

                (*a * x + *b * y + *c * z + *d).is_zero()
            }
            Plane3::Parametric(point1, (u, v)) => {
                let normal = u * v;
                let vec = point.clone() - point1.clone();

                normal.dot(&vec).is_zero()
            }
        }
    }

    /// Checks if the point is on the place. This should be used over [`is_in`](Self::is_in) for floats
    /// coordinates.
    pub fn is_in_float(&self, point: &Vector<T, 3>) -> bool
    where
        T: Float,
        T::DotOutput: Float,
    {
        match self {
            Self::Cartesian([a, b, c, d]) => {
                // straight forward
                let [x, y, z] = point.scalars;

                (*a * x + *b * y + *c * z + *d).abs() <= T::epsilon()
            }
            Self::Parametric(point1, (u, v)) => {
                let normal = u * v;
                let vec = point.clone() - point1.clone();

                normal.dot(&vec).abs() <= T::DotOutput::epsilon()
            }
        }
    }

    pub fn intersection(&self, rhs: &Self) -> Plane3Intersection<T>
    where
        T: Float,
    {
        use Plane3::*;

        match (self, rhs) {
            (Cartesian(x), Cartesian(y)) => {
                let mut rref = Matrix::from([*x, *y]);
                // here we don't really about the right member
                rref.rref_mut();

                match rref.rank_float() {
                    2 => { // a line!
                        // matrix representation helps here.
                        Plane3Intersection::Line(Line3::Cartesian(rref.data))
                    }
                    1 => {
                        let _eq_sum = rref.data[1].iter()
                            .take(3)
                            .fold(T::zero(), |acc, &x| acc + x);
                        let _right = rref[(1, 3)];

                        todo!();
                    },
                    _ => todo!(),
                }
            },
            _ => todo!(),
        }
    }
}

/// The result of an intersection of two 3D planes.
///
/// The notation of [`Plane3`] and [`Line3`]  is chosen by the implementation and depending
/// on the notation of the two planes.
pub enum Plane3Intersection<T>
where
    T: Default + Copy + Num + InnerDotProductSpace<3>,
{
    /// The two planes do not intersect.
    None,
    /// The two planes are equivalent.
    Plane(Plane3<T>),
    /// The two planes intersects on a line.
    Line(Line3<T>),
}

impl<T> PartialEq for Plane3<T>
where
    T: Default + Float + InnerDotProductSpace<3>,
    T::DotOutput: Float,
{
    fn eq(&self, other: &Self) -> bool {
        use Plane3::*;

        match (self, other) {
            (Cartesian(a), Cartesian(b)) => {
                let first_non_zero = (0..3).find(|&i| b[i] > T::epsilon());

                let Some(first_non_zero) = first_non_zero else {
                    // a must be [0, 0, 0, 0] for planes to be equals
                    return a.iter().all(|x| x.abs() <= T::epsilon());
                };

                let a1 = a.map(|x| x / b[first_non_zero]);
                let b1 = b.map(|x| x / b[first_non_zero]);

                a1 == b1
            },
            (Parametric(p1, (u1, v1)), Parametric(p2, (u2, v2))) => {
                // fast-exit: p1 in (P2) and p2 in (P1)?
                if !self.is_in_float(p2) || !other.is_in_float(p1) {
                    return false;
                }

                // u2 and v2 should be linear dependents of (u1, v1), checking this.
                let family1 = <&VectorFamily<_, _>>::from(&[u1.clone(), v1.clone(), u2.clone()][..]);
                let family2 = <&VectorFamily<_, _>>::from(&[u1.clone(), v1.clone(), v2.clone()][..]);

                !family1.is_independent() &&
                    !family2.is_independent()
            },
            (plane @ Cartesian(_), Parametric(p, (u, v))) |
            (Parametric(p, (u, v)), plane @ Cartesian(_)) => {
                // on release, we assume (u, v) is linear independent.
                if cfg!(debug_assertions) {
                    let mut matrix = Matrix::from_vectors(&[u.clone(), v.clone(), Default::default()]);
                    matrix.gaussian_elimination_mut();

                    let is_independent = matrix.rank_float() == 2;

                    // we don't even define a plane here anyway
                    if !is_independent {
                        return false;
                    }
                }

                plane.is_in_float(p) && plane.is_in_float(&(p.clone() + u.clone())) && plane.is_in_float(&(p.clone() + v.clone()))
            }
        }
    }
}

/// Represents a 3D line. It could be either in Cartesian or in parametric notation.
#[derive(Clone, Debug)]
pub enum Line3<T>
where
    T: Default + Copy + Num,
{
    Cartesian([[T; 4]; 2]),
    Parametric(Vector<T, 3>, Vector<T, 3>),
}

impl<T> Line3<T>
where
    T: Default + Copy + Num + InnerDotProductSpace<3>,
{
    pub fn is_line(&self) -> bool
    where
        T: Float,
        T::DotOutput: Float,
    {
        match self {
            Self::Cartesian([a, b]) => {
                let plane_a = Plane3::Cartesian(*a);
                let plane_b = Plane3::Cartesian(*b);

                PartialEq::ne(&plane_a, &plane_b)
            }
            Self::Parametric(_, vec) => vec.is_near_zero(),
        }
    }

    /// Converts `self` to a vector family for `T ^ 3`.
    ///
    /// The family can either contain:
    ///  * 0 vectors if it's a point or if it's not a vector space,
    ///  * 1 vector if it's a linear line,
    ///  * 2 vectors if it's a linear plane
    pub fn to_family(&self) -> Owned<VectorFamily<T, 3>, ArrayVec<Vector<T, 3>, 2>>
    where
        T: Float,
    {
        match *self {
            Self::Cartesian([a, b]) => {
                let left: Matrix<T, 2, 3> = Matrix::from([
                    *a[0..3].as_array().unwrap(),
                    *b[0..3].as_array().unwrap(),
                ]);
                let right = Matrix::from([[a[3]], [b[3]]]);

                let (rref, steps) = left.record_rref();
                let rref_right = right.replay_steps(&steps);

                match rref.rank_float() {
                    // cannot be a line, could only represent (0, 0, 0), not sure
                    0 => {
                        let _is_right_zero = 'a: {
                            for k in 0..3usize {
                                if rref_right[(k, 0)].abs() > T::epsilon() {
                                    break 'a false;
                                }
                            }

                            true
                        };

                        todo!();
                    }
                    // a plane
                    1 => {
                        todo!();
                    },
                    _ => todo!(),
                }
            },
            _ => todo!(),
        }
    }
}
