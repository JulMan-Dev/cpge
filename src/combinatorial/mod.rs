use num_traits::{Float, NumCast, Zero};

pub fn factorial(n: usize) -> usize {
    (1..=n).product()
}

/// Computes the number `k`-combination for `n` elements.
///
/// # Panic
///
/// This panics if `n < k`.
pub fn combination<T>(k: usize, n: usize) -> T
where
    T: Float,
{
    if n < k {
        panic!("n cannot smaller than k")
    }

    if k.is_zero() {
        return T::one();
    }

    let upper: usize = (n - k + 1..=n).product();
    let lower: usize = (1..=k).product();

    // for some reason, T::from(upper) doesn't work
    let upper: T = <T as NumCast>::from(upper).expect("usize must be convertible to usize");
    let lower: T = <T as NumCast>::from(lower).expect("usize must be convertible to usize");

    upper * lower.recip()
}
