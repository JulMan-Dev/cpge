use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::borrow::{Borrow, BorrowMut};
use crate::linear::view::{IntoGoggles, MatrixView, MatrixViewHeader, make_header, update_header};
use crate::linear::{Matrix, Vector};
use crate::mem::Living;
use num_traits::Num;
use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut, Index, IndexMut};

#[derive(Debug)]
#[repr(C)]
pub struct HeapMatrix<T>
where
    T: Default + Copy + Num,
{
    header: UnsafeCell<MatrixViewHeader>,
    pub data: Box<[T]>,
    pub rows: usize,
    pub cols: usize,
}

impl<T> HeapMatrix<T>
where
    T: Default + Copy + Num,
{
    pub fn empty() -> Self {
        Self {
            header: UnsafeCell::new(make_header(0, 0)),
            data: Box::from([]),
            rows: 0,
            cols: 0,
        }
    }

    pub const fn new(data: Box<[T]>, rows: usize, cols: usize) -> Self {
        Self {
            header: UnsafeCell::new(make_header(rows, cols)),
            data,
            rows,
            cols
        }
    }

    pub fn zero(rows: usize, cols: usize) -> Self {
        Self {
            header: UnsafeCell::new(make_header(rows, cols)),
            data: vec![T::zero(); rows * cols].into_boxed_slice(),
            rows,
            cols,
        }
    }

    pub fn view(&self, goggles: impl IntoGoggles) -> Living<'_, MatrixView<T>> {
        goggles.into_goggles().see_through(self)
    }

    pub fn view_mut(&mut self, goggles: impl IntoGoggles) -> Living<'_, MatrixView<T>, true> {
        goggles.into_goggles().see_through_mut(self)
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zero(n, n);

        for k in 0..n {
            m[(k, k)] = T::one();
        }

        m
    }

    pub fn transpose(&self) -> Self {
        let mut matrix = Self::zero(self.cols, self.rows);

        for (i, chunk) in self.data.chunks(self.cols).enumerate() {
            assert_eq!(chunk.len(), self.rows);

            for j in 0..self.rows {
                matrix[(i, j)] = self[(j, i)];
            }
        }

        matrix
    }

    pub fn from_row_vectors<const C: usize>(vectors: &[Vector<T, C>]) -> Self {
        let mut data = vec![T::zero(); vectors.len() * C];

        for (i, v) in vectors.iter().enumerate() {
            data.splice(i * C..(i + 1) * C, v.scalars);
        }

        Self::new(data.into_boxed_slice(), vectors.len(), C)
    }

    pub fn from_vectors<const R: usize>(vectors: &[Vector<T, R>]) -> Self {
        HeapMatrix::from_row_vectors(vectors).transpose()
    }
}

impl<T> Deref for HeapMatrix<T>
where
    T: Default + Num + Copy,
{
    type Target = MatrixView<T>;

    fn deref(&self) -> &Self::Target {
        // updating the header
        unsafe {
            update_header(&mut *self.header.get(), self.data.as_ptr().cast_mut().cast());
            MatrixView::from_header(&self.header)
        }
    }
}

impl<T> DerefMut for HeapMatrix<T>
where
    T: Default + Num + Copy,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        update_header(self.header.get_mut(), self.data.as_mut_ptr().cast());

        unsafe { MatrixView::from_header_mut(&mut self.header) }
    }
}

impl<T> Borrow<MatrixView<T>> for HeapMatrix<T>
where
    T: Default + Num + Copy,
{
    fn borrow(&self) -> &MatrixView<T> {
        self
    }
}

impl<T> BorrowMut<MatrixView<T>> for HeapMatrix<T>
where
    T: Default + Num + Copy,
{
    fn borrow_mut(&mut self) -> &mut MatrixView<T> {
        self
    }
}

impl<T> Index<(usize, usize)> for HeapMatrix<T>
where
    T: Default + Copy + Num,
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 * self.cols + index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for HeapMatrix<T>
where
    T: Default + Copy + Num,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 * self.cols + index.1]
    }
}

impl<T> PartialEq for HeapMatrix<T>
where
    T: Default + Copy + Num,
{
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        PartialEq::<MatrixView<T>>::eq(self, other)
    }
}

impl<T, const R: usize, const C: usize> PartialEq<Matrix<T, R, C>> for HeapMatrix<T>
where
    T: Default + Copy + Num,
{
    #[inline(always)]
    fn eq(&self, other: &Matrix<T, R, C>) -> bool {
        PartialEq::<MatrixView<T>>::eq(self, other)
    }
}

impl<T> PartialEq<MatrixView<T>> for HeapMatrix<T>
where
    T: Default + Copy + Num,
{
    fn eq(&self, other: &MatrixView<T>) -> bool {
        let (rows, cols) = other.dim();

        if self.rows == rows && self.cols == cols {
            Iterator::eq(self.values(), other.values())
        } else {
            false
        }
    }
}

impl<T> Eq for HeapMatrix<T> where T: Default + Copy + Num {}

impl<T> Clone for HeapMatrix<T>
where
    T: Default + Copy + Num,
{
    fn clone(&self) -> Self {
        Self {
            header: UnsafeCell::new(unsafe { *self.header.get() }),
            data: self.data.clone(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl<T, const R: usize, const C: usize> From<Matrix<T, R, C>> for HeapMatrix<T>
where
    T: Default + Copy + Num,
{
    fn from(value: Matrix<T, R, C>) -> Self {
        Self::new(Box::from(value.data.as_flattened()), R, C)
    }
}

impl<T, A> FromIterator<A> for HeapMatrix<T>
where
    T: Default + Copy + Num,
    A: IntoIterator<Item = T>,
{
    fn from_iter<K: IntoIterator<Item = A>>(iter: K) -> Self {
        let mut ret = Vec::new();
        let mut last_i = None;

        let mut cols = None::<usize>;

        for (i, row) in iter.into_iter().enumerate() {
            let previous_len = ret.len();
            ret.extend(row);

            let pushed = ret.len() - previous_len;

            match &mut cols {
                Some(x) => if pushed != *x { panic!("invalid amount of values in row") },
                x => { x.replace(pushed); },
            };

            last_i.replace(i);
        }

        match cols {
            None => Self::empty(),
            Some(cols) => {
                let rows: usize = ret.len() / cols; // it must be a multiple of cols

                Self::new(ret.into_boxed_slice(), rows, cols)
            }
        }
    }
}
