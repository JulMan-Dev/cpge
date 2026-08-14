use core::cmp::Ordering;
use arrayvec::ArrayVec;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::iter;

/// `Living` is an owned value that wraps an owned data that only lives during `'a` lifetime.
///
/// In general, `T` is an owned smart pointer, for example, [`MatrixView`](crate::linear::MatrixView).
///
/// This is not a smart pointer. It doesn't hold a pointer to `T`, it just owns a `T` and marks
/// it-self as holding a reference which lives `'a`. It works better if `T` doesn't implement
/// [`Clone`].
///
/// It is not possible to unwrap the inner value of the living object. But it is still possible to use
/// the inner value by reference as this implements [`Deref`] and [`DerefMut`] traits.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Living<'a, T, const MUT: bool = false> {
    inner: T,
    _marker: PhantomData<&'a ()>,
}

impl<'a, T, const MUT: bool> Living<'a, T, MUT> {
    pub const fn new(inner: T) -> Self {
        Self { inner, _marker: PhantomData }
    }
}

impl<'a, T> Living<'a, T, true> {
    pub const fn to_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

impl<'a, T, const MUT: bool> Deref for Living<'a, T, MUT> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a, T> DerefMut for Living<'a, T, true> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

pub trait AbstractVec<T>
where
    Self: Deref<Target = [T]> + DerefMut,
{
    fn push(&mut self, v: T);

    #[must_use]
    fn pop(&mut self) -> Option<T>;

    #[must_use]
    fn remove(&mut self, i: usize) -> T;

    fn truncate(&mut self, len: usize);

    fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone;

    fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>];
}

impl<T, const CAP: usize> AbstractVec<T> for ArrayVec<T, CAP> {
    fn push(&mut self, v: T) {
        Self::push(self, v);
    }

    fn pop(&mut self) -> Option<T> {
        Self::pop(self)
    }

    fn remove(&mut self, i: usize) -> T {
        Self::remove(self, i)
    }

    fn truncate(&mut self, len: usize) {
        Self::truncate(self, len);
    }

    fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone
    {
        match self.len().cmp(&new_len) {
            Ordering::Less => Self::extend(self, iter::repeat_n(value, new_len - self.len())),
            Ordering::Greater => Self::truncate(self, new_len),
            Ordering::Equal => {} // nothing to do
        }
    }

    fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        Self::spare_capacity_mut(self)
    }
}

mod heap {
    use alloc::vec::Vec;
    use core::mem::MaybeUninit;
    use crate::mem::AbstractVec;

    impl<T> AbstractVec<T> for Vec<T> {
        fn push(&mut self, v: T) {
            Self::push(self, v);
        }

        fn pop(&mut self) -> Option<T> {
            Self::pop(self)
        }

        fn remove(&mut self, i: usize) -> T {
            Self::remove(self, i)
        }

        fn truncate(&mut self, len: usize) {
            Self::truncate(self, len);
        }

        fn resize(&mut self, new_len: usize, value: T)
        where
            T: Clone
        {
            Self::resize(self, new_len, value);
        }

        fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
            Self::spare_capacity_mut(self)
        }
    }
}

pub trait FromOwned<Owned> {
    fn from_owned(owned: &Owned) -> &Self;

    fn from_owned_mut(owned: &mut Owned) -> &mut Self;
}

/// An owned instance of `O` that dereferences to a type `K`. This is used to share a `K` instance
/// when it is unsized. The storage is handled by `O`.
#[derive(Debug)]
pub struct Owned<K: ?Sized, O>
where
    K: FromOwned<O>,
{
    inner: O,
    _data: PhantomData<K>
}

impl<K: ?Sized, O> Owned<K, O>
where
    K: FromOwned<O>,
{
    /// Creates a new `Owned` instance from the given `O` instance.
    pub const fn new(inner: O) -> Self {
        Self { inner, _data: PhantomData }
    }

    /// Consumes the `Owned` instance and returns the inner storage instance.
    pub fn into_inner(self) -> O {
        self.inner
    }

    /// Returns a reference to the inner storage instance.
    pub fn to_inner(&self) -> &O {
        &self.inner
    }

    /// Returns a mutable reference to the inner storage instance.
    pub fn to_inner_mut(&mut self) -> &mut O {
        &mut self.inner
    }
}

impl<K: ?Sized, O> Clone for Owned<K, O>
where
    K: FromOwned<O>,
    O: Clone,
{
    fn clone(&self) -> Self {
        Self::new(self.inner.clone())
    }
}

impl<K: ?Sized, O> Deref for Owned<K, O>
where
    K: FromOwned<O>,
{
    type Target = K;

    fn deref(&self) -> &Self::Target {
        K::from_owned(&self.inner)
    }
}

impl<K: ?Sized, O> DerefMut for Owned<K, O>
where
    K: FromOwned<O>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        K::from_owned_mut(&mut self.inner)
    }
}
