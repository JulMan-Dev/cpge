use core::mem;

/// A trait that adds a method, [`spec_window`](Self::spec_window).
pub trait SpecWindow<T> {
    /// Returns a new sliding window iterator.
    fn spec_window<const N: usize>(&mut self) -> SWindow<'_, T, N>;
}

/// A sliding window iterator.
///
/// The iterator returns a tuple of a mutable reference to the current element and a slice of the
/// next `N` elements. May be used to rewrite elements in-place depending on next elements.
///
/// This iterator may only work on fixed slices.
pub struct SWindow<'a, T, const N: usize> {
    inner: &'a mut [T]
}

impl<'a, T, const N: usize> Iterator for SWindow<'a, T, N> {
    type Item = (&'a mut T, &'a [T; N]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.inner.len() > N {
            let inner = self.inner as *mut [T];
            // SAFETY: the pointer is valid for 'a
            let (left, others) = unsafe { &mut *inner }.split_first_mut().unwrap();
            let right = others[..N].as_array().unwrap();
            // SAFETY: others lives as long as self.inner.
            self.inner = unsafe { mem::transmute_copy(&others) };
            Some((left, right))
        } else {
            None
        }
    }
}

impl<T> SpecWindow<T> for [T] {
    fn spec_window<const N: usize>(&mut self) -> SWindow<'_, T, N> {
        SWindow { inner: self }
    }
}
