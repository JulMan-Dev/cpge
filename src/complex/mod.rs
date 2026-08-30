pub mod julia;

use core::num::FpCategory;
use core::ops::{Add, Div, Mul, Neg, Rem, Sub};
use num_traits::{ConstOne, ConstZero, Float, Num, NumCast, One, ToPrimitive, Zero};

/// Basic complex. It supports every operation on it.
///
/// It implements: +, -, *, / on complexes and on reals.
///
/// ```
/// # use cpge::complex::BasicComplex;
/// let z = BasicComplex::Complex(1.0, 2.0); // 1 + 2i
/// let a = z.conjugate();
/// assert_eq!(a, BasicComplex::Complex(1.0, -2.0));
/// ```
#[derive(Copy, Clone, PartialOrd, PartialEq, Default, Debug)]
pub enum BasicComplex<T>
where
    T: Default + Copy + Num,
{
    #[default]
    Zero,
    Real(T),
    Imaginary(T),
    Complex(T, T),
}

use BasicComplex::*;
#[cfg(feature = "alloc")]
use crate::polynomials::{basic_newton_binomial_heap, HeapPolynomial};

impl<T> BasicComplex<T>
where
    T: Default + Copy + Num,
{
    /// Computes the conjugates of `self`.
    ///
    /// It's simply `a + ib -> a - ib`.
    pub fn conjugate(self) -> Self {
        match self {
            x @ (Zero | Real(_)) => x,
            Imaginary(x) => Imaginary(T::zero() - x),
            Complex(a, b) => Complex(a, T::zero() - b),
        }
    }

    /// Compacts `self` to ensure it is using the correct enum variant.
    ///
    /// ```
    /// # use cpge::complex::BasicComplex::*;
    ///
    /// assert_eq!(Real(0).compact(), Zero);
    /// assert_eq!(Complex(1, 0).compact(), Real(1));
    /// assert_eq!(Complex(0, 1).compact(), Imaginary(1));
    ///
    /// // identity if it cannot be compacted
    /// assert_eq!(Imaginary(1).compact(), Imaginary(1));
    /// assert_eq!(Complex(1, 1).compact(), Complex(1, 1));
    /// ```
    pub fn compact(self) -> Self {
        match self {
            Zero => Zero,
            Real(a) | Imaginary(a) if a.is_zero() => Zero,
            x @ (Real(_) | Imaginary(_)) => x,
            Complex(a, b) => match (a.is_zero(), b.is_zero()) {
                (true, true) => Zero,
                (true, false) => Imaginary(b),
                (false, true) => Real(a),
                (false, false) => Complex(a, b),
            }
        }
    }

    /// Converts `self` as a pair of `T`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cpge::complex::BasicComplex::*;
    /// let z = Complex(1, 2); // 1 + 2i
    /// assert_eq!(z.into_pair(), [1, 2]);
    /// let z = Imaginary(3); // 3i
    /// assert_eq!(z.into_pair(), [0, 3]);
    /// ```
    pub fn into_pair(self) -> [T; 2] {
        match self {
            Zero => [T::zero(); 2],
            Real(a) => [a, T::zero()],
            Imaginary(b) => [T::zero(), b],
            Complex(a, b) => [a, b],
        }
    }

    pub fn re(&self) -> T {
        match *self {
            Real(x) | Complex(x, _) => x,
            _ => T::zero(),
        }
    }

    pub fn im(&self) -> T {
        match *self {
            Imaginary(x) | Complex(_, x) => x,
            _ => T::zero(),
        }
    }

    pub fn magnitude(self) -> T
    where
        T: Float,
    {
        T::sqrt(self.into_pair().iter().fold(T::zero(), |acc, &x| acc + x * x))
    }

    /// Gets the imaginary unit.
    pub fn i() -> Self {
        Imaginary(T::one())
    }
}

impl<T> BasicComplex<T>
where
    T: Default + Copy + Num + ConstOne,
{
    /// The imaginary unit.
    pub const I: Self = Imaginary(T::ONE);
}

pub fn construct_from_usize<T>(mut n: usize) -> T
where
    T: Default + Copy + Num,
{
    let mut res = T::one();
    let (mut acc, mut k) = (T::one(), 1usize);

    while n != 1 {
        if k == n {
            return acc;
        }

        if k.is_multiple_of(n) {
            res = res * acc;
            n /= k;
        } else {
            acc = acc + T::one();
            k += 1;
        }
    }

    res
}

#[test]
fn test_construct_from_usize() {
    let s: &[(f64, usize)] = &[
        (3.0, 3),
        (52.0, 52),
    ];

    for &(f, n) in s {
        let g: f64 = construct_from_usize(n);

        assert_eq!(f, g);
    }
}

impl<T> Add<T> for BasicComplex<T>
where
    T: Default + Copy + Num,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        if rhs.is_zero() {
            self
        } else {
            match self {
                Zero => Real(rhs),
                Real(a) => Real(a + rhs),
                Imaginary(b) => Complex(rhs, b),
                Complex(a, b) => Complex(a + rhs, b),
            }
        }
    }
}

impl<T> Add for BasicComplex<T>
where
    T: Default + Copy + Num,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let ([a, b], [c, d]) = (self.into_pair(), rhs.into_pair());

        Complex(a + c, b + d)
    }
}

impl<T> Zero for BasicComplex<T>
where
    T: Default + Copy + Num,
{
    fn zero() -> Self { Zero }

    fn is_zero(&self) -> bool {
        self.into_pair().iter().all(Zero::is_zero)
    }
}

impl<T> ConstZero for BasicComplex<T>
where
    T: Default + Copy + Num,
{
    const ZERO: Self = Zero;
}

impl<T> One for BasicComplex<T>
where
    T: Default + Copy + Num,
{
    fn one() -> Self { Real(T::one()) }
}

impl<T> ConstOne for BasicComplex<T>
where
    T: Default + Copy + Num + ConstOne,
{
    const ONE: Self = Real(T::ONE);
}

impl<T> Neg for BasicComplex<T>
where
    T: Default + Copy + Num,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Zero => Zero,
            Real(a) => Real(T::zero() - a),
            Imaginary(b) => Imaginary(T::zero() - b),
            Complex(a, b) => Complex(T::zero() - a, T::zero() - b),
        }
    }
}

impl<T> Sub<T> for BasicComplex<T>
where
    T: Default + Copy + Num,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        self + (T::zero() - rhs)
    }
}

impl<T> Sub for BasicComplex<T>
where
    T: Default + Copy + Num,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<T> Mul<T> for BasicComplex<T>
where
    T: Default + Copy + Num,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        if rhs.is_zero() {
            Zero
        } else {
            match self {
                Zero => Zero,
                Real(a) => Real(a * rhs),
                Imaginary(b) => Imaginary(b * rhs),
                Complex(a, b) => Complex(a * rhs, b * rhs),
            }
        }
    }
}

impl<T> Mul for BasicComplex<T>
where
    T: Default + Copy + Num,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Zero, _) | (_, Zero) => Zero,
            (z, Real(x)) | (Real(x), z) => z * x,
            (Imaginary(a), Imaginary(b)) => Real(T::zero() - a * b),
            (Imaginary(x), Complex(a, b)) | (Complex(a, b), Imaginary(x)) => {
                Complex(T::zero() - x * b, x * a)
            },
            (Complex(a, b), Complex(c, d)) => Complex(a * c - b * d, a * d + b * c)
        }
    }
}

impl<T> Div<T> for BasicComplex<T>
where
    T: Default + Copy + Num,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        debug_assert!(!rhs.is_zero());

        match self {
            Zero => Zero,
            Real(a) => Real(a / rhs),
            Imaginary(b) => Imaginary(b / rhs),
            Complex(a, b) => Complex(a / rhs, b / rhs),
        }
    }
}

impl<T> Div for BasicComplex<T>
where
    T: Default + Copy + Num,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        // (a + ib) / (c + id) = (a + ib)(c - id) / (c^2 + d^2)
        let c = rhs.conjugate();
        let sum: T = rhs.into_pair().iter().fold(T::zero(), |acc, &x| acc + x * x);

        self * c / sum
    }
}

impl<T> Rem<T> for BasicComplex<T>
where
    T: Default + Copy + Num,
{
    type Output = Self;

    fn rem(self, rhs: T) -> Self::Output {
        match self {
            Zero => Zero,
            Real(a) => Real(a % rhs),
            Imaginary(b) => Imaginary(b % rhs),
            Complex(a, b) => Complex(a % rhs, b % rhs),
        }
    }
}

/// ⚠️ This implementation is not standard. Each component of the result is the remainder of the
/// component of the requests.
impl<T> Rem for BasicComplex<T>
where
    T: Default + Copy + Num,
{
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let ([a, b], [c, d]) = (self.into_pair(), rhs.into_pair());

        Complex(a % c, b % d)
    }
}

impl<T> Num for BasicComplex<T>
where
    T: Default + Copy + Num,
{
    type FromStrRadixErr = T::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        if let Some((re, im)) = str.split_once(',') {
            let re = T::from_str_radix(re, radix)?;
            let im = T::from_str_radix(im, radix)?;

            Ok(Complex(re, im))
        } else {
            Ok(Real(T::from_str_radix(str, radix)?))
        }
    }
}

impl<T> ToPrimitive for BasicComplex<T>
where
    T: Default + Copy + Num + ToPrimitive,
{
    fn to_i64(&self) -> Option<i64> {
        match *self {
            Zero => Some(0),
            Real(a) => a.to_i64(),
            Complex(a, b) if b.is_zero() => a.to_i64(),
            _ => None,
        }
    }

    fn to_u64(&self) -> Option<u64> {
        match *self {
            Self::Zero => Some(0),
            Real(a) => a.to_u64(),
            Complex(a, b) if b.is_zero() => a.to_u64(),
            _ => None,
        }
    }
}

impl<T> NumCast for BasicComplex<T>
where
    T: Default + Copy + Num + NumCast,
{
    fn from<K: ToPrimitive>(n: K) -> Option<Self> {
        let k: T = NumCast::from(n)?;

        Some(Real(k))
    }
}

#[cfg(feature = "alloc")]
impl<T> Float for BasicComplex<T>
where
    T: Default + Copy + Num + Float,
{
    /// Returns NaN real complex.
    fn nan() -> Self {
        Real(T::nan())
    }

    /// Returns inf real complex.
    fn infinity() -> Self {
        Real(T::infinity())
    }

    /// Returns -inf real complex.
    fn neg_infinity() -> Self {
        Real(T::neg_infinity())
    }

    /// Returns -0.0 real complex.
    fn neg_zero() -> Self {
        Real(T::neg_zero())
    }

    /// Returns `T::min_value()` as real complex.
    fn min_value() -> Self {
        Real(T::min_value())
    }

    /// Returns `T::min_positive_value()` as real complex.
    fn min_positive_value() -> Self {
        Real(T::min_positive_value())
    }

    /// Returns `T::max_value()` as real complex.
    fn max_value() -> Self {
        Real(T::max_value())
    }

    /// Checks if at least one of the component is NaN.
    fn is_nan(self) -> bool {
        self.into_pair().iter().any(|&x| x.is_nan())
    }

    /// Checks if at least one of the component is infinite.
    fn is_infinite(self) -> bool {
        self.into_pair().iter().any(|&x| x.is_infinite())
    }

    /// Checks if both components are finite.
    fn is_finite(self) -> bool {
        self.into_pair().iter().all(|&x| x.is_finite())
    }

    /// Checks if both components are normal.
    fn is_normal(self) -> bool {
        self.into_pair().iter().all(|&x| x.is_normal())
    }

    fn classify(self) -> FpCategory {
        match self {
            x if x.is_nan() => FpCategory::Nan,
            x if x.is_infinite() => FpCategory::Infinite,
            x if x.is_zero() => FpCategory::Zero,
            x if x.is_normal() => FpCategory::Normal,
            _ => FpCategory::Subnormal,
        }
    }

    /// Panics, floor cannot be implemented on complexes.
    fn floor(self) -> Self {
        panic!("cannot floor on complex");
    }

    /// Panics, ceil cannot be implemented on complexes.
    fn ceil(self) -> Self {
        panic!("cannot ceil on complex");
    }

    //// Panics, round cannot be implemented on complex.
    fn round(self) -> Self {
        panic!("cannot round on complex");
    }

    /// Panics, trunc cannot be implemented on complex.
    fn trunc(self) -> Self {
        panic!("cannot trunc on complex");
    }

    /// Panics, fract cannot be implemented on complex.
    fn fract(self) -> Self {
        panic!("cannot fract on complex");
    }

    /// Takes the [`abs()`](Float::abs) of each component.
    ///
    /// Use [`magnitude`](Self::magnitude) to get the magnitude.
    fn abs(self) -> Self {
        let [re, im] = self.into_pair();

        Complex(re.abs(), im.abs())
    }

    /// Takes the [`signum()`](Float::signum) of each component.
    fn signum(self) -> Self {
        let [re, im] = self.into_pair();

        Complex(re.signum(), im.signum())
    }

    /// Panics, is_sign_positive cannot be implemented on complexes.
    fn is_sign_positive(self) -> bool {
        panic!("cannot is_sign_positive on complex");
    }

    /// Panics, is_sign_negative cannot be implemented on complexes.
    fn is_sign_negative(self) -> bool {
        panic!("cannot is_sign_negative on complex")
    }

    /// Cannot be implemented using [`mul_add()`](Float::mul_add).
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }

    fn recip(self) -> Self {
        // 1 / (a + ib) = (a - ib) / (a^2 + b^2)
        let conj = self.conjugate();
        let sum = self.into_pair().iter().fold(T::zero(), |acc, &x| acc + x * x);

        conj / sum
    }

    fn powi(self, n: i32) -> Self {
        match (self, n) {
            (Zero, _) => Zero,
            (_, 0) => Self::one(),
            (Real(x), n) => Real(x.powi(n)),
            (_, n) if n < 0 => self.powi(-n).recip(),
            (Imaginary(x), n) if n % 4 == 0 => Real(x.powi(n)),
            (Imaginary(x), n) if n % 4 == 1 => Imaginary(x.powi(n)),
            (Imaginary(x), n) if n % 4 == 2 => -Real(x.powi(n)),
            (Imaginary(x), n) if n % 4 == 3 => -Imaginary(x.powi(n)),
            (Imaginary(_), _) => unreachable!(),
            (Complex(a, b), n) => {
                // newton binomial
                let poly: HeapPolynomial<Self> = basic_newton_binomial_heap(Real(a), n as usize); // we consider X = ib, easier
                poly.apply(Imaginary(b))
            }
        }
    }

    fn powf(self, x: Self) -> Self {
        Self::exp(self.ln() * x)
    }

    fn sqrt(self) -> Self {
        let r = self.magnitude();
        let re = self.re();

        let two = T::one() + T::one(); // easy two

        let re = ((r + re) / two).sqrt();
        let im = ((r - re) / two).sqrt();

        if self.im() < T::zero() {
            Complex(re, -im)
        } else {
            Complex(re, im)
        }
    }

    fn exp(self) -> Self {
        let [re, im] = self.into_pair();
        let exp_re = re.exp();
        let (sin, cos) = im.sin_cos();

        Complex(exp_re * cos, exp_re * sin)
    }

    fn exp2(self) -> Self {
        Self::exp(self * construct_from_usize::<Self>(2))
    }

    fn ln(self) -> Self {
        let [re, im] = self.into_pair();
        let r = self.magnitude();
        let theta = re.atan2(im);
        Complex(r.ln(), theta)
    }

    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    fn log2(self) -> Self {
        self.ln() / Real(construct_from_usize::<T>(2).ln())
    }

    fn log10(self) -> Self {
        self.ln() / Real(construct_from_usize::<T>(10).ln())
    }

    /// Panics, max cannot be implemented on complexes.
    fn max(self, _other: Self) -> Self {
        panic!("cannot max on complex");
    }

    /// Panics, max cannot be implemented on complexes.
    fn min(self, _other: Self) -> Self {
        panic!("cannot min on complex")
    }

    /// Panics, abs_sub cannot be implemented on complexes.
    fn abs_sub(self, _other: Self) -> Self {
        panic!("cannot abs_sub on complex")
    }

    fn cbrt(self) -> Self {
        let three = construct_from_usize(3);

        self.powf(Real(T::one() / three))
    }

    fn hypot(self, other: Self) -> Self {
        (self.powi(2) + other.powi(2)).sqrt()
    }

    fn sin(self) -> Self {
        let [re, im] = self.into_pair();

        Complex(re.sin() * im.cosh(), re.cos() * im.sinh())
    }

    fn cos(self) -> Self {
        let [re, im] = self.into_pair();

        Complex(re.cos() * im.cosh(), -re.sin() * im.sinh())
    }

    fn tan(self) -> Self {
        let [re, im] = self.into_pair();
        let two: T = T::one() + T::one();
        let sum = (two * re).cos() + (two * im).cosh();

        Complex((two * re).sin() / sum, (two * im).sinh() / sum)
    }

    fn asin(self) -> Self {
        // -ilog(iz + sqrt(1 - z^2))
        -Self::i() * (Self::i() * self + (Self::one() - self.powi(2)).sqrt()).log10()
    }

    fn acos(self) -> Self {
        // -ilog(z + isqrt(1 - z^2))
        -Self::i() * (self + Self::i() * (Self::one() - self.powi(2)).sqrt()).log10()
    }

    fn atan(self) -> Self {
        // i/2 log((1-iz)/(1+iz))
        let (a, b) = (Real(T::one()), Self::i() * self);
        Self::i() / (T::one() + T::one()) * Self::log10((a - b) / (a + b))
    }

    fn atan2(self, _other: Self) -> Self {
        todo!("find a correct implementation for this")
    }

    fn sin_cos(self) -> (Self, Self) {
        let [re, im] = self.into_pair();
        let (sin, cos) = re.sin_cos();
        let (sinh, cosh) = (im.sinh(), im.cosh());

        (Complex(sin * cosh, cos * sinh), Complex(cos * cosh, -sin * sinh))
    }

    fn exp_m1(self) -> Self {
        match self {
            Zero => -Self::one(),
            Real(k) => Real(k.exp_m1()),
            z @ Imaginary(_) => z.exp() - Self::one(),
            z @ Complex(a, _) => if a.is_zero() {
                -Self::one()
            } else {
                z.exp() - Self::one()
            }
        }
    }

    fn ln_1p(self) -> Self {
        match self {
            Zero => Zero, // ln(1) = 0
            Real(k) => Real(k.ln_1p()),
            z => (z + Self::one()).ln()
        }
    }

    fn sinh(self) -> Self {
        let [re, im] = self.into_pair();
        let k = Real(re.exp()) * Imaginary(im).exp();
        // Re(x.exp()) is better than Real(x).exp()

        (k - k.recip()) / (T::one() + T::one())
    }

    fn cosh(self) -> Self {
        let [re, im] = self.into_pair();
        let k = Real(re.exp()) * Imaginary(im).exp();
        // Re(x.exp()) is better than Real(x).exp()

        (k + k.recip()) / (T::one() + T::one())
    }

    fn tanh(self) -> Self {
        todo!()
    }

    fn asinh(self) -> Self {
        todo!()
    }

    fn acosh(self) -> Self {
        todo!()
    }

    fn atanh(self) -> Self {
        todo!()
    }

    /// Panics, integer_decode cannot be implemented on complexes.
    fn integer_decode(self) -> (u64, i16, i8) {
        panic!("cannot interger_decode on complexes")
    }
}
