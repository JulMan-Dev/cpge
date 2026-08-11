//! Product of matrices. This is included by default, and you may never need to import this
//! module manually. This does not export anything.

// supposed A and B matrices objects, we support:
//  - A * B
//  - &A * B
//  - A * &B
//  - &A * &B

macro_rules! invoke {
    // &, &
    ($fn:expr => (ref $lhr:expr, ref $rhs:expr)) => { $fn(&$lhr, &$rhs) };

    // &, _
    ($fn:expr => (ref $lhr:expr, $rhs:expr)) => { $fn(&$lhr, $rhs) };
    ($fn:expr => (ref $lhr:expr, _ $rhs:expr)) => { $fn(&$lhr, $rhs) };

    // _, &
    ($fn:expr => ($lhr:expr, ref $rhs:expr)) => { $fn($lhr, &$rhs) };
    ($fn:expr => (_ $lhr:expr, ref $rhs:expr)) => { $fn($lhr, &$rhs) };

    // _, _
    ($fn:expr => ($lhr:expr, $rhs:expr)) => { $fn($lhr, $rhs) };
    ($fn:expr => (_ $lhr:expr, $rhs:expr)) => { $fn($lhr, $rhs) };
    ($fn:expr => ($lhr:expr, _ $rhs:expr)) => { $fn($lhr, $rhs) };
    ($fn:expr => (_ $lhr:expr, _ $rhs:expr)) => { $fn($lhr, $rhs) };
}

pub(crate) use invoke;

macro_rules! impl_product {
    ($($lt:ty, $rt:ty, ($lhs:tt, $rhs:tt),)+) => {
        $(impl<T> Mul<$rt> for $lt
        where
            T: Default + Copy + Num,
        {
            type Output = HeapMatrix<T>;

            fn mul(self, rhs: $rt) -> Self::Output {
                $crate::linear::product::invoke!(do_product => ($lhs self, $rhs rhs))
            }
        })+
    };
}

pub(crate) use impl_product;

macro_rules! impl_scalar {
    ($($lt:ty, $rt:ty, ($lhs:tt, $rhs:tt),)+) => {
        $(impl<T> Mul<$rt> for $lt
        where
            T: Default + Copy + Num,
        {
            type Output = HeapMatrix<T>;

            fn mul(self, rhs: $rt) -> Self::Output {
                $crate::linear::product::invoke!(do_scalar_product => ($lhs self, $rhs rhs))
            }
        })+
    };
}

pub(crate) use impl_scalar;
