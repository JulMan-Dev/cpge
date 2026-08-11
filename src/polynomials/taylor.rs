//! Taylor polynomial formula.

use arrayvec::ArrayVec;
use crate::combinatorial::factorial;
use crate::polynomials::{basic_newton_binomial, Polynomial};
use num_traits::{Float, Num, NumCast, Zero};

/// Represents Taylor polynomial formula.
#[derive(Clone, Debug)]
pub struct TaylorPolynomial<T, const CAP: usize>
where
    T: Default + Copy + Num,
{
    /// The point where the formula is constructed.
    pub point: T,
    /// The coefficients for the formula.
    pub coefficients: ArrayVec<T, CAP>,
}

impl<T, const CAP: usize> TaylorPolynomial<T, CAP>
where
    T: Default + Copy + Num,
{
    /// Produces a new polynomial from the coefficients.
    ///
    /// # Example
    ///
    /// ```
    /// # use cpge::polynomials::HeapPolynomial;
    /// # use cpge::polynomials::taylor::HeapTaylorPolynomial;
    /// let f = HeapPolynomial::from([1.0, 2.0, 3.0]); // 1 + 2x + 3x^2
    /// let taylor: HeapTaylorPolynomial<f64> = f.taylor_at_0();
    /// let g = taylor.to_polynomial();
    /// assert_eq!(f, g); // f = g
    /// ```
    pub fn to_polynomial(&self) -> Polynomial<T, CAP>
    where
        T: Float,
    {
        let mut poly = Zero::zero();

        for (i, &k) in self.coefficients.iter().enumerate() {
            poly = poly + basic_newton_binomial(-self.point, i) * k *
                <T as NumCast>::from(factorial(i)).unwrap().recip();
        }

        poly
    }
}

pub mod heap {
    use alloc::boxed::Box;
    use num_traits::{Float, Num, NumCast, Zero};
    use crate::combinatorial::factorial;
    use crate::polynomials::poly::heap::{basic_newton_binomial_heap, HeapPolynomial};

    /// Represents Taylor polynomial formula.
    #[derive(Clone, Debug)]
    pub struct HeapTaylorPolynomial<T>
    where
        T: Default + Copy + Num,
    {
        /// The point where the formula is constructed.
        pub point: T,
        /// The coefficients for the formula.
        pub coefficients: Box<[T]>,
    }

    impl<T> HeapTaylorPolynomial<T>
    where
        T: Default + Copy + Num,
    {
        /// Produces a new polynomial from the coefficients.
        ///
        /// # Example
        ///
        /// ```
        /// # use cpge::polynomials::HeapPolynomial;
        /// # use cpge::polynomials::taylor::HeapTaylorPolynomial;
        /// let f = HeapPolynomial::from([1.0, 2.0, 3.0]); // 1 + 2x + 3x^2
        /// let taylor: HeapTaylorPolynomial<f64> = f.taylor_at_0();
        /// let g = taylor.to_polynomial();
        /// assert_eq!(f, g); // f = g
        /// ```
        pub fn to_polynomial(&self) -> HeapPolynomial<T>
        where
            T: Float,
        {
            let mut poly = Zero::zero();

            for (i, &k) in self.coefficients.iter().enumerate() {
                poly = poly + basic_newton_binomial_heap(-self.point, i) * k *
                    <T as NumCast>::from(factorial(i)).unwrap().recip();
            }

            poly
        }
    }

    #[test]
    fn test_to_polynomial() {
        let polys = [
            HeapPolynomial::from([0.0, 0.0, 2.0]), // 2x^2
            HeapPolynomial::from([1.0, 2.0, 3.0]), // 1 + 2x + 3x^2
        ];

        #[derive(Clone, Copy, Debug)]
        enum Point<T: Copy> {
            NonZero(T),
            Zero,
        }

        //use Point::*;

        let points = [
            Point::Zero,
            Point::NonZero(4.0),
        ];

        for poly in &polys {
            for point in &points {
                let taylor = match point {
                    Point::Zero => poly.taylor_at_0(),
                    &Point::NonZero(a) => poly.taylor(a),
                };

                let g = taylor.to_polynomial();

                assert_eq!(*poly, g);
            }
        }
    }
}

pub use heap::*;
