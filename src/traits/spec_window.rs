use core::mem;

pub trait SpecWindow<T> {
    fn spec_window<const N: usize>(&mut self) -> SWindow<'_, T, N>;
}

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
