use core::iter::FusedIterator;

pub trait UnpackableIter<A, B, C> {
    fn unpack_left(self) -> LeftUnpacker<Self>
    where
        Self: Sized + Iterator<Item = ((A, B), C)>;

    fn unpack_right(self) -> RightUnpacker<Self>
    where
        Self: Sized + Iterator<Item = (A, (B, C))>;
}

impl<T, A, B, C> UnpackableIter<A, B, C> for T
where
    T: Sized + Iterator
{
    fn unpack_left(self) -> LeftUnpacker<Self>
    where
        Self: Iterator<Item=((A, B), C)>
    {
        LeftUnpacker(self)
    }

    fn unpack_right(self) -> RightUnpacker<Self>
    where
        Self: Iterator<Item=(A, (B, C))>
    {
        RightUnpacker(self)
    }
}

pub struct LeftUnpacker<T>(T);
pub struct RightUnpacker<T>(T);

macro_rules! impl_unpacker_iterator_common {
    () => {
        fn size_hint(&self) -> (usize, Option<usize>) { self.0.size_hint() }
    };
}
macro_rules! impl_unpacker {
    (%map { $map_left:expr, $map_right:expr }) => {
        // LeftUnpacker
        impl<A, B, C, T: Iterator<Item = ((A, B), C)>> Iterator for LeftUnpacker<T> {
            type Item = (A, B, C);
            fn next(&mut self) -> Option<Self::Item> { self.0.next().map($map_left) }
            impl_unpacker_iterator_common!();
        }

        impl<A, B, C, T: Iterator<Item = ((A, B), C)> + DoubleEndedIterator> DoubleEndedIterator for LeftUnpacker<T> {
            fn next_back(&mut self) -> Option<Self::Item> { self.0.next_back().map($map_left) }
        }

        impl<A, B, C, T: Iterator<Item = ((A, B), C)> + ExactSizeIterator> ExactSizeIterator for LeftUnpacker<T> {
            fn len(&self) -> usize { self.0.len() }
        }

        impl<A, B, C, T: Iterator<Item = ((A, B), C)> + FusedIterator> FusedIterator for LeftUnpacker<T> {}

        // RightUnpacker
        impl<A, B, C, T: Iterator<Item = (A, (B, C))>> Iterator for RightUnpacker<T> {
            type Item = (A, B, C);
            fn next(&mut self) -> Option<Self::Item> { self.0.next().map($map_right) }
            impl_unpacker_iterator_common!();
        }

        impl<A, B, C, T: Iterator<Item = (A, (B, C))> + DoubleEndedIterator> DoubleEndedIterator for RightUnpacker<T> {
            fn next_back(&mut self) -> Option<Self::Item> { self.0.next_back().map($map_right) }
        }

        impl<A, B, C, T: Iterator<Item = (A, (B, C))> + ExactSizeIterator> ExactSizeIterator for RightUnpacker<T> {
            fn len(&self) -> usize { self.0.len() }
        }

        impl<A, B, C, T: Iterator<Item = (A, (B, C))> + FusedIterator> FusedIterator for RightUnpacker<T> {}
    };
}

impl_unpacker!(
    %map {
        |((a, b), c)| (a, b, c),
        |(a, (b, c))| (a, b, c)
    }
);
