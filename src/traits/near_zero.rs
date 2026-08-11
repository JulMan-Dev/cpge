use num_traits::Float;

pub trait NearZero {
    /// Checks if `self` is near zero for floats or exactly zero for integers.
    fn is_near_zero(&self) -> bool;
}

macro_rules! impl_near_zero_epsilon {
    ($($t:ty)*) => {
        $(impl NearZero for $t {
            fn is_near_zero(&self) -> bool { self.abs() <= Self::epsilon() }
        })*
    };
}

macro_rules! impl_near_zero_exact {
    ($($t:ty)*) => {
        $(impl NearZero for $t {
            fn is_near_zero(&self) -> bool { *self == 0 }
        })*
    }
}

impl_near_zero_epsilon!(f32 f64);
impl_near_zero_exact!(i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize);
