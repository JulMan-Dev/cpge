use alloc::borrow::ToOwned;
use core::num::FpCategory;
use num_traits::{Float, Num, NumCast, One, ToPrimitive, Zero};
use crate::linear::Matrix;

/// This is an internal type that implements [`From<usize>`](From) when using `f64`.
#[derive(PartialEq, PartialOrd, Copy, Clone, Debug, Default)]
pub struct Wrapped<T>(pub T);

fn test() {
    let matrix = Matrix::from_array([[1, 2], [3, 4]]);

    let heap = (*matrix).to_owned();
}

macro_rules! impl_base {
    (%self $self:ident, $(%op $name:ident $fn_name:ident,)+) => {
        $(impl<T: core::ops::$name<Output = T>> core::ops::$name for $self<T> {
            type Output = Self;

            fn $fn_name(self, rhs: Self) -> Self::Output {
                $self(<T as core::ops::$name>::$fn_name(self.0, rhs.0))
            }
        })+
    };
}

impl_base!(
    %self Wrapped,
    %op Add add,
    %op Sub sub,
    %op Mul mul,
    %op Div div,
    %op Rem rem,
);

impl<T: core::ops::Neg<Output = T>> core::ops::Neg for Wrapped<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Wrapped(core::ops::Neg::neg(self.0))
    }
}

macro_rules! inherit_to_self_0 {
    (%base $base_type:tt %trait $base_trait:tt, $(fn $name:ident($($arg:ident: $argt:ty),*) -> Self;)*) => {
        $(#[inline]
        fn $name($($arg: $argt),*) -> Self {
            $base_type(<T as $base_trait>::$name($($arg),*))
        })*
    };
}

macro_rules! inherit_self_to_self_0 {
    (%base $base_type:tt %trait $base_trait:tt, $(fn $name:ident(self $(, $arg:ident: $argt:ty)*) -> Self;)*) => {
        $(#[inline]
        fn $name(self $(, $arg: $argt)*) -> Self {
            $base_type($base_trait::$name(self.0 $(, $arg)*))
        })*
    };
}

macro_rules! inherit_self_0 {
    (%base $base_type:tt %trait $base_trait:tt, $(fn $name:ident(self $(, $arg:ident: $argt:ty)*) -> $rt:tt;)*) => {
        $(#[inline]
        fn $name(self $(, $arg: $argt)*) -> $rt {
            $base_trait::$name(self.0 $(, $arg)*)
        })*
    };
}

impl<T: Float> Float for Wrapped<T> {
    inherit_to_self_0! {
        %base Wrapped %trait Float,
        fn nan() -> Self;
        fn infinity() -> Self;
        fn neg_infinity() -> Self;
        fn neg_zero() -> Self;
        fn min_value() -> Self;
        fn min_positive_value() -> Self;
        fn epsilon() -> Self;
        fn max_value() -> Self;
    }

    inherit_self_0! {
        %base F %trait Float,
        fn is_nan(self) -> bool;
        fn is_infinite(self) -> bool;
        fn is_finite(self) -> bool;
        fn is_normal(self) -> bool;
        fn is_subnormal(self) -> bool;
        fn classify(self) -> FpCategory;
        fn is_sign_positive(self) -> bool;
        fn is_sign_negative(self) -> bool;
        fn integer_decode(self) -> (u64, i16, i8);
    }

    inherit_self_to_self_0! {
        %base Wrapped %trait Float,
        fn floor(self) -> Self;
        fn ceil(self) -> Self;
        fn round(self) -> Self;
        fn trunc(self) -> Self;
        fn fract(self) -> Self;
        fn abs(self) -> Self;
        fn signum(self) -> Self;
        fn recip(self) -> Self;
        fn powi(self, n: i32) -> Self;
        fn sqrt(self) -> Self;
        fn exp(self) -> Self;
        fn exp2(self) -> Self;
        fn ln(self) -> Self;
        fn log2(self) -> Self;
        fn log10(self) -> Self;
        fn to_degrees(self) -> Self;
        fn to_radians(self) -> Self;
        fn cbrt(self) -> Self;
        fn cos(self) -> Self;
        fn sin(self) -> Self;
        fn tan(self) -> Self;
        fn asin(self) -> Self;
        fn acos(self) -> Self;
        fn atan(self) -> Self;
        fn exp_m1(self) -> Self;
        fn ln_1p(self) -> Self;
        fn sinh(self) -> Self;
        fn cosh(self) -> Self;
        fn tanh(self) -> Self;
        fn asinh(self) -> Self;
        fn acosh(self) -> Self;
        fn atanh(self) -> Self;
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Wrapped(T::mul_add(self.0, a.0, b.0))
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        Wrapped(T::powf(self.0, n.0))
    }

    #[inline]
    fn log(self, base: Self) -> Self {
        Wrapped(T::log(self.0, base.0))
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        Wrapped(T::max(self.0, other.0))
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        Wrapped(T::min(self.0, other.0))
    }

    #[inline]
    fn abs_sub(self, other: Self) -> Self {
        Wrapped(T::abs_sub(self.0, other.0))
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        Wrapped(T::hypot(self.0, other.0))
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        Wrapped(T::atan2(self.0, other.0))
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (a, b) = T::sin_cos(self.0);
        (Wrapped(a), Wrapped(b))
    }
}

impl<T: ToPrimitive> ToPrimitive for Wrapped<T> {
    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }
}

impl<T: NumCast> NumCast for Wrapped<T> {
    fn from<K: ToPrimitive>(n: K) -> Option<Self> {
        T::from(n).map(Wrapped)
    }
}

impl<T: One> One for Wrapped<T> {
    inherit_to_self_0! {
        %base Wrapped %trait One,
        fn one() -> Self;
    }
}

impl<T: Zero> Zero for Wrapped<T> {
    inherit_to_self_0! {
        %base Wrapped %trait Zero,
        fn zero() -> Self;
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<T: Num> Num for Wrapped<T> {
    type FromStrRadixErr = <T as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        <T as Num>::from_str_radix(str, radix).map(Wrapped)
    }
}

impl From<usize> for Wrapped<f64> {
    fn from(value: usize) -> Self {
        Self(value as f64)
    }
}
