use crate::combinatorial::combination;
use crate::polynomials::taylor::TaylorPolynomial;
use crate::traits::spec_window::SpecWindow;
use arrayvec::ArrayVec;
use core::ops::{Add, Deref, Mul, Not};
use core::mem;
use core::mem::MaybeUninit;
use core::ops::DerefMut;
use num_traits::{ConstZero, Float, Num, NumCast, One, Zero};
use crate::mem::AbstractVec;

/// A polynomial whose degree is maximum `N`, on the stack.
#[derive(Clone, Debug, PartialOrd, PartialEq)]
#[repr(transparent)]
pub struct Polynomial<T, const CAP: usize>
where
    T: Default + Copy + Num,
{
    pub coefficients: ArrayVec<T, CAP>,
}

impl<T, const CAP: usize> Polynomial<T, CAP>
where
    T: Default + Copy + Num,
{
    pub fn new(coefficients: [T; CAP]) -> Self {
        Self { coefficients: ArrayVec::from(coefficients) }
    }

    pub fn one() -> Self {
        let mut vec = ArrayVec::new();
        vec.push(T::one());
        vec.spare_capacity_mut().fill_with(|| MaybeUninit::new(T::zero()));

        if cfg!(debug_assertions) {
            // SAFETY: we filled the entire vec
            unsafe { vec.set_len(CAP) };
        }

        // SAFETY: we filled the entire vec
        Self::new(unsafe { vec.into_inner_unchecked() })
    }

    /// Computes the derivative of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cpge::polynomials::Polynomial;
    /// let f = Polynomial::new([0, 2, 1]); // 2x + x^2
    /// let df = f.derivative();
    /// assert_eq!(df.coefficients, [2, 2][..]); // 2 + 2x
    /// ```
    #[inline(always)]
    pub fn derivative(&self) -> Self
    where
        T: NumCast,
    {
        let mut x = self.clone();
        x.derivative_mut();
        x
    }

    /// Computes the `n`th derivative of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cpge::polynomials::Polynomial;
    /// let f = Polynomial::new([0, 2, 1, 1]); // 2x + x^2 + x^3
    /// let g = f.nth_derivative(2);
    /// assert_eq!(g.coefficients, [2, 6][..]); // 2 + 6x
    pub fn nth_derivative(&self, n: usize) -> Self
    where
        T: NumCast,
    {
        if n == 0 {
            self.clone()
        } else {
            let mut p = self.clone();
            for _ in 0..n { p.derivative_mut() }
            p
        }
    }

    /// Computes Taylor coefficients from a polynomial for a point.
    pub fn taylor(&self, point: T) -> TaylorPolynomial<T, CAP>
    where
        T: NumCast,
    {
        // using .degree_float here is overkill, it's just for allocation.
        if self.degree().is_none() {
            return TaylorPolynomial { point, coefficients: ArrayVec::new() };
        };

        let mut g = self.clone();
        let mut coefficients = ArrayVec::new();

        for i in coefficients.iter_mut() {
            *i = g.apply(point);
            g.derivative_mut();
        }

        TaylorPolynomial { point, coefficients }
    }

    /// Computes Taylor coefficients from `self` polynomial at zero.
    ///
    /// This is faster than [`taylor(T::zero())`](Self::taylor) because it doesn't derive `self`
    /// several times.
    pub fn taylor_at_0(&self) -> TaylorPolynomial<T, CAP>
    where
        T: NumCast,
    {
        let mut coefficients = self.coefficients.clone();

        coefficients.iter_mut().enumerate()
            .fold(1usize, |acc, (i, p)| {
                let fact = acc * i.max(1);

                *p = *p * <T as NumCast>::from(fact).expect("T must be constructible from usize");

                fact
            });

        TaylorPolynomial { point: T::zero(), coefficients }
    }
}

impl<T, const CAP: usize> Deref for Polynomial<T, CAP>
where
    T: Default + Copy + Num,
{
    type Target = AbstractPolynomial<T>;

    fn deref(&self) -> &Self::Target {
        let vec: &dyn AbstractVec<T> = &self.coefficients;

        // SAFETY: AbstractPolynomial<T> and dyn AbstractVec<T> share the same layout
        unsafe { mem::transmute(vec) }
    }
}

impl<T, const CAP: usize> DerefMut for Polynomial<T, CAP>
where
    T: Default + Copy + Num,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        let vec: &mut dyn AbstractVec<T> = &mut self.coefficients;

        // SAFETY: AbstractPolynomial<T> and dyn AbstractVec<T> share the same layout
        unsafe { mem::transmute(vec) }
    }
}

#[cfg(feature = "alloc")]
mod heap {
    use alloc::boxed::Box;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::mem;
    use core::ops::{Add, Deref, DerefMut, Mul};
    use num_traits::{ConstZero, Float, Num, NumCast, One, Zero};
    use crate::combinatorial::combination;
    use crate::mem::AbstractVec;
    use crate::polynomials::AbstractPolynomial;
    use crate::polynomials::taylor::HeapTaylorPolynomial;

    /// A polynomial whose data is on the heap.
    #[derive(Clone, Debug, PartialOrd, PartialEq)]
    #[repr(transparent)]
    pub struct HeapPolynomial<T>
    where
        T: Default + Copy + Num,
    {
        // we need to store a Vec and not a boxed slice because AbstractPolynomial requires it.
        pub coefficients: Vec<T>,
    }

    impl<T> HeapPolynomial<T>
    where
        T: Default + Copy + Num,
    {
        pub fn compact(&mut self) {
            let new_length = self.degree().unwrap_or(0);

            if self.coefficients.len() == new_length {
                return;
            }

            self.coefficients.truncate(new_length);
        }

        pub fn compact_float(&mut self)
        where
            T: Float,
        {
            let new_length = self.degree_float().unwrap_or(0);

            if self.coefficients.len() == new_length {
                return;
            }

            self.coefficients.truncate(new_length);
        }

        /// Computes the derivative of `self`.
        ///
        /// # Example
        ///
        /// ```
        /// # use cpge::polynomials::{HeapPolynomial};
        /// let f = HeapPolynomial::from([0, 2, 1]); // 2x + x^2
        /// let df = f.derivative();
        /// assert_eq!(df.coefficients, &[2, 2][..]); // 2 + 2x
        /// ```
        #[inline(always)]
        pub fn derivative(&self) -> Self
        where
            T: NumCast,
        {
            let mut x = self.clone();
            x.derivative_mut();
            x
        }

        /// Computes the `n`th derivative of `self`.
        ///
        /// # Example
        ///
        /// ```
        /// # use cpge::polynomials::HeapPolynomial;
        /// let f = HeapPolynomial::from([0, 2, 1, 1]); // 2x + x^2 + x^3
        /// let g = f.nth_derivative(2);
        /// assert_eq!(g.coefficients, &[2, 6][..]); // 2 + 6x
        pub fn nth_derivative(&self, n: usize) -> Self
        where
            T: NumCast,
        {
            if n == 0 {
                self.clone()
            } else {
                let mut p = self.clone();
                for _ in 0..n { p.derivative_mut() }
                p
            }
        }

        /// Computes Taylor coefficients from a polynomial for a point.
        pub fn taylor(&self, point: T) -> HeapTaylorPolynomial<T>
        where
            T: NumCast,
        {
            // using .degree_float here is overkill, it's just for allocation.
            let Some(degree) = self.degree() else {
                return HeapTaylorPolynomial { point, coefficients: Box::default() };
            };

            let mut g = self.clone();
            let mut coefficients = Box::new_uninit_slice(degree + 1);

            for i in coefficients.iter_mut() {
                i.write(g.apply(point));
                g.derivative_mut();
            }

            // SAFETY: all elements are initialized.
            let coefficients = unsafe { coefficients.assume_init() };

            HeapTaylorPolynomial { point, coefficients }
        }

        /// Computes Taylor coefficients from `self` polynomial at zero.
        ///
        /// This is faster than [`taylor(T::zero())`](Self::taylor) because it doesn't derive `self`
        /// several times.
        pub fn taylor_at_0(&self) -> HeapTaylorPolynomial<T>
        where
            T: NumCast,
        {
            let mut coefficients = self.coefficients.clone();

            coefficients.iter_mut().enumerate()
                .fold(1usize, |acc, (i, p)| {
                    let fact = acc * i.max(1);

                    *p = *p * <T as NumCast>::from(fact).expect("T must be constructible from usize");

                    fact
                });

            HeapTaylorPolynomial {
                point: T::zero(),
                coefficients: coefficients.into_boxed_slice(),
            }
        }
    }

    impl<T> Mul<T> for HeapPolynomial<T>
    where
        T: Copy + Default + Num,
    {
        type Output = Self;

        fn mul(mut self, rhs: T) -> Self::Output {
            for p in self.coefficients.iter_mut() { *p = *p * rhs };

            Self { coefficients: self.coefficients }
        }
    }

    impl<T> Mul for HeapPolynomial<T>
    where
        T: Copy + Default + Num,
    {
        type Output = Self;

        #[allow(clippy::suspicious_arithmetic_impl)]
        fn mul(self, rhs: Self) -> Self::Output {
            let new_degree = {
                let (Some(a), Some(b)) = (self.degree(), rhs.degree()) else {
                    return Self::zero();
                };

                a + b
            };

            let mut coefficients = vec![T::zero(); new_degree];

            for (i, &x) in self.coefficients.iter().enumerate() {
                for (j, &y) in rhs.coefficients.iter().enumerate() {
                    let v = coefficients.get_mut(i + j).unwrap();

                    *v = *v + x * y;
                }
            }

            Self { coefficients }
        }
    }

    impl<T> Add for HeapPolynomial<T>
    where
        T: Copy + Default + Num,
    {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            let mut coefficients = Vec::with_capacity(
                usize::max(self.coefficients.len(), rhs.coefficients.len())
            );

            for i in 0usize.. {
                let (x, y) = (self.coefficients.get(i), rhs.coefficients.get(i));

                coefficients.push(match (x, y) {
                    (Some(&x), Some(&y)) => x + y,
                    (Some(&x), None) | (None, Some(&x)) => x,
                    (None, None) => break,
                });
            }

            Self { coefficients }
        }
    }

    // this doesn't require T: ConstZero since Vec::new is const.
    impl<T> ConstZero for HeapPolynomial<T>
    where
        T: Default + Copy + Num,
    {
        const ZERO: Self = Self { coefficients: Vec::new() };
    }

    impl<T> Zero for HeapPolynomial<T>
    where
        T: Default + Copy + Num,
    {
        fn zero() -> Self {
            Self { coefficients: Vec::new() }
        }

        fn is_zero(&self) -> bool {
            self.coefficients.iter().all(Zero::is_zero)
        }
    }

    impl<T> One for HeapPolynomial<T>
    where
        T: Default + Copy + Num,
    {
        fn one() -> Self {
            Self { coefficients: vec![T::one()] }
        }
    }

    impl<T, K> From<K> for HeapPolynomial<T>
    where
        T: Default + Copy + Num,
        K: IntoIterator<Item = T>,
    {
        fn from(value: K) -> Self {
            Self { coefficients: value.into_iter().collect() }
        }
    }

    /// Computes the polynomial equivalent to `(x + a)^n`.
    pub fn basic_newton_binomial_heap<T>(a: T, n: usize) -> HeapPolynomial<T>
    where
        T: Default + Copy + Num + Float,
    {
        if a.is_zero() {
            // trivial: it's x^n
            let mut coefficients = Vec::with_capacity(n + 1);

            for _ in 0..n {
                coefficients.push(T::zero());
            }

            coefficients.push(T::one());

            // SAFETY: all elements are initialized;
            return HeapPolynomial { coefficients };
        }

        if n.is_zero() {
            // trivial: it's one
            return One::one();
        }

        if n.is_one() {
            // trivial: it's (a + x), avoiding computations.
            return HeapPolynomial { coefficients: vec![a, T::one()] };
        }

        let coefficients = (0..=n)
            .map(|i| combination::<T>(i, n) * a.powi((n - i) as i32))
            .collect();

        HeapPolynomial { coefficients }
    }

    impl<T> Deref for HeapPolynomial<T>
    where
        T: Default + Copy + Num,
    {
        type Target = AbstractPolynomial<T>;

        fn deref(&self) -> &Self::Target {
            let vec: &dyn AbstractVec<T> = &self.coefficients;

            // SAFETY: AbstractPolynomial<T> and dyn AbstractVec<T> share the same layout
            unsafe { mem::transmute(vec) }
        }
    }

    impl<T> DerefMut for HeapPolynomial<T>
    where
        T: Default + Copy + Num,
    {
        fn deref_mut(&mut self) -> &mut Self::Target {
            let vec: &mut dyn AbstractVec<T> = &mut self.coefficients;

            // SAFETY: AbstractPolynomial<T> and dyn AbstractVec<T> share the same layout
            unsafe { mem::transmute(vec) }
        }
    }

    #[test]
    fn test_newton_binomial() {
        use crate::testing::Wrapped;

        let rights: &[(Wrapped<f64>, usize)] = &[
            (Wrapped(1.0), 2),
            (Wrapped(1.0), 3),
        ];

        let polys: &[HeapPolynomial<Wrapped<f64>>] = &[
            [Wrapped(1.0), Wrapped(2.0), Wrapped(1.0)].into(),
            [Wrapped(1.0), Wrapped(3.0), Wrapped(3.0), Wrapped(1.0)].into(),
        ];

        for ((a, n), expected) in
            Iterator::zip(rights.iter(), polys.iter()) {
            let computed = basic_newton_binomial_heap(*a, *n);

            assert_eq!(*expected, computed);
        }
    }
}

#[cfg(feature = "alloc")]
pub use heap::*;

#[repr(transparent)]
pub struct AbstractPolynomial<T>(dyn AbstractVec<T>)
where
    T: Default + Copy + Num;

impl<T> AbstractPolynomial<T>
where
    T: Default + Copy + Num,
{
    /// Computes the degree of `self` polynomial.
    pub fn degree(&self) -> Option<usize> {
        self.0.iter()
            .enumerate()
            .rev()
            .find_map(|(i, v)| v.is_zero().not().then_some(i))
    }

    /// Computes the degree of `self` polynomial if `T` is [`Float`].
    pub fn degree_float(&self) -> Option<usize>
    where
        T: Float,
    {
        self.0.iter()
            .enumerate()
            .rev()
            .find_map(|(i, v)| v.abs().ge(&T::epsilon()).then_some(i))
    }

    /// Computes `self(x)`.
    ///
    /// This doesn't need `T` to implement a `powi` because it computes it recursively.
    ///
    /// # Example
    ///
    /// ```
    /// # use cpge::polynomials::HeapPolynomial;
    /// let f = HeapPolynomial::from([1, 2]); // 1 + 2x
    /// let y = f.apply(2); // 1 + 2 * 2 = 5
    /// assert_eq!(y, 5);
    /// ```
    pub fn apply(&self, x: T) -> T {
        self.0.iter()
            .rev()
            .fold(T::zero(), |acc, &coeff| acc * x + coeff)
    }

    /// Computes the derivative of `self`. This changes in place, which may be preferred over
    /// `derivative` if you won't reuse the previous polynomial.
    pub fn derivative_mut(&mut self)
    where
        T: NumCast,
    {
        if let Some(degree) = self.degree() && degree >= 1 {
            for (i, (cur, &[next])) in self.0.spec_window().enumerate() {
                *cur = next * <T as NumCast>::from(i + 1).expect("T must be constructible from usize");
            }

            let _ = self.0.pop();
        } else {
            self.set_zero();
        }
    }

    pub fn nth_derivative_mut(&mut self, n: usize)
    where
        T: NumCast,
    {
        if n > 0 {
            for _ in 0..n { self.derivative_mut() }
        } else {
            self.set_zero();
        }
    }

    pub fn set_zero(&mut self) {
        self.0.fill(T::zero());
    }
}

impl<T, const CAP: usize> Mul<T> for Polynomial<T, CAP>
where
    T: Copy + Default + Num,
{
    type Output = Self;

    fn mul(mut self, rhs: T) -> Self::Output {
        for p in self.coefficients.iter_mut() { *p = *p * rhs };

        Self { coefficients: self.coefficients }
    }
}

impl<T, const CAP: usize> Add for Polynomial<T, CAP>
where
    T: Copy + Default + Num,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut coefficients = ArrayVec::new();

        for i in 0usize.. {
            let (x, y) = (self.coefficients.get(i), rhs.coefficients.get(i));

            coefficients.push(match (x, y) {
                (Some(&x), Some(&y)) => x + y,
                (Some(&x), None) | (None, Some(&x)) => x,
                (None, None) => break,
            });
        }

        Self { coefficients }
    }
}

// same as PolynomialHeap: ConstZero, it doesn't require T: ConstZero since ArrayVec::new_const
// is const.
impl<T, const CAP: usize> ConstZero for Polynomial<T, CAP>
where
    T: Default + Copy + Num,
{
    const ZERO: Self = Self { coefficients: ArrayVec::new_const() };
}

impl<T, const CAP: usize> Zero for Polynomial<T, CAP>
where
    T: Default + Copy + Num,
{
    fn zero() -> Self {
        Self { coefficients: ArrayVec::new() }
    }

    fn set_zero(&mut self) {
        // this is more memory efficient than the default implementation
        self.coefficients.clear()
    }

    fn is_zero(&self) -> bool {
        self.coefficients.iter().all(Zero::is_zero)
    }
}

/// Computes the polynomial equivalent to `(x + a)^n`.
pub fn basic_newton_binomial<T, const CAP: usize>(a: T, n: usize) -> Polynomial<T, CAP>
where
    T: Default + Copy + Num + Float,
{
    if a.is_zero() {
        // trivial: it's x^n
        let mut coefficients = ArrayVec::new();

        for _ in 0..n {
            coefficients.push(T::zero());
        }

        coefficients.push(T::one());

        // SAFETY: all elements are initialized;
        return Polynomial { coefficients };
    }

    if n.is_zero() {
        // trivial: it's one
        return Polynomial::one();
    }

    if n.is_one() {
        // trivial: it's (a + x), avoiding computations.
        return Polynomial { coefficients: ArrayVec::try_from(&[a, T::one()][..]).unwrap() };
    }

    let coefficients = (0..=n)
        .map(|i| combination::<T>(i, n) * a.powi((n - i) as i32))
        .collect();

    Polynomial { coefficients }
}
