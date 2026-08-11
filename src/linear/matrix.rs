use crate::linear::heap_matrix::HeapMatrix;
use crate::linear::vector::Vector;
use crate::linear::view::{make_header, update_header, IntoGoggles, MatrixView, MatrixViewHeader, RowRecorder};
use crate::mem::Living;
use alloc::{boxed::Box, string::ToString};
use core::borrow::{Borrow, BorrowMut};
use core::cell::UnsafeCell;
use core::fmt::Write;
use core::marker::PhantomPinned;
use core::mem::MaybeUninit;
use core::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul};
use core::{fmt, mem};
use num_traits::{ConstOne, ConstZero, Float, Num, Zero};

/// A row-major `R`x`C` matrix.
#[derive(Debug)]
#[repr(C)]
pub struct Matrix<T, const R: usize, const C: usize = R>
where
    T: Default + Copy + Num,
{
    header: UnsafeCell<MatrixViewHeader>,
    pub data: [[T; C]; R],
    _pinned: PhantomPinned,
}

impl<T, const R: usize, const C: usize> ConstZero for Matrix<T, R, C>
where
    T: Default + ConstZero + Copy + Num,
{
    const ZERO: Self = Self::from_array([[T::ZERO; C]; R]);
}

/// Methods for all matrices.
impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Default + Copy + Num,
{
    pub const fn from_array(data: [[T; C]; R]) -> Self {
        Self { header: UnsafeCell::new(make_header(R, C)), data, _pinned: PhantomPinned }
    }

    pub fn view(&self, goggles: impl IntoGoggles) -> Living<'_, MatrixView<T>> {
        goggles.into_goggles().see_through(self)
    }

    pub fn view_mut(&mut self, goggles: impl IntoGoggles) -> Living<'_, MatrixView<T>, true> {
        goggles.into_goggles().see_through_mut(self)
    }

    pub fn into_heap(self) -> HeapMatrix<T> {
        HeapMatrix::new(
            Box::from(self.data.as_flattened()),
            R, C,
        )
    }

    /// Transposes the matrix, swapping rows and columns.
    ///
    /// # Example
    ///
    /// ```
    /// # use cpge::linear::Matrix;
    /// let original: Matrix<i32, 2, 3> = Matrix::from([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    /// ]);
    /// let transposed: Matrix<i32, 3, 2> = original.transpose();
    /// assert_eq!(transposed, Matrix::from([
    ///     [1, 4],
    ///     [2, 5],
    ///     [3, 6],
    /// ]));
    ///
    /// // this is used for example in from_row_vectors/to_row_vectors
    /// let from_cols = original.to_vectors();
    /// let from_rows = original.transpose().to_row_vectors();
    /// assert_eq!(from_cols, from_rows);
    /// ```
    pub const fn transpose(&self) -> Matrix<T, C, R> {
        let mut data = [[MaybeUninit::<T>::uninit(); R]; C];

        let mut i = 0;
        while i < R {
            let mut j = 0;
            while j < C {
                data[j][i].write(self.data[i][j]);

                j += 1;
            }

            i += 1;
        }

        // SAFETY: all elements initialized above
        let data = unsafe { mem::transmute_copy(&data) };
        Matrix {
            header: UnsafeCell::new(make_header(R, C)),
            data,
            _pinned: PhantomPinned,
        }
    }

    pub fn set_row(&mut self, row: usize, values: &[T; C]) -> bool {
        if row < R {
            self.data[row] = *values;
            true
        } else {
            false
        }
    }
}

/// Methods for square matrix.
impl<T, const N: usize> Matrix<T, N>
where
    T: Default + Copy + Num,
{
    /// Gets the `N * N` identity matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use cpge::iter::Cross;
    /// # use cpge::linear::Matrix;
    /// // checking if it is really identity
    /// let m: Matrix<i32, 3> = Matrix::identity();
    /// for (i, j) in Cross::new(0..3, 0..3) {
    ///     let to_check = (i == j) as i32;
    ///     assert_eq!(m[(i, j)], to_check);
    /// }
    ///
    /// // AI = IA = A
    /// let a = Matrix::from([
    ///     [2, 3, 4],
    ///     [1, 0, 3],
    ///     [8, 1, 9],
    /// ]);
    /// assert_eq!(&*a * &*m, &*m * &*a);
    /// assert_eq!(&*a * &*m, a);
    /// ```
    pub fn identity() -> Self {
        let mut matrix = Self::zero();

        for k in 0..N {
            matrix[(k, k)] = T::one();
        }

        matrix
    }

    /// Same as [`identity`](Self::identity) but works at compile-time.
    pub const fn const_identity() -> Self
    where
        T: ConstZero + ConstOne
    {
        let mut data = Self::ZERO.data;

        let mut i = 0;
        while i < N  {
            data[i][i] = T::ONE;
            i += 1;
        }

        Self { header: UnsafeCell::new(make_header(N, N)), data, _pinned: PhantomPinned }
    }
}

impl<T, const R: usize, const C: usize> Deref for Matrix<T, R, C>
where
    T: Copy + Default + Num,
{
    type Target = MatrixView<T>;

    fn deref(&self) -> &Self::Target {
        unsafe {
            update_header(&mut *self.header.get(), self.data.as_ptr().cast_mut().cast());

            MatrixView::from_header(&self.header)
        }
    }
}

impl<T, const R: usize, const C: usize> DerefMut for Matrix<T, R, C>
where
    T: Copy + Default + Num,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        update_header(self.header.get_mut(), self.data.as_mut_ptr().cast());

        unsafe { MatrixView::from_header_mut(&mut self.header) }
    }
}

impl<T, const R: usize, const C: usize> Borrow<MatrixView<T>> for Matrix<T, R, C>
where
    T: Default + Num + Copy,
{
    fn borrow(&self) -> &MatrixView<T> {
        self
    }
}

impl<T, const R: usize, const C: usize> BorrowMut<MatrixView<T>> for Matrix<T, R, C>
where
    T: Default + Num + Copy,
{
    fn borrow_mut(&mut self) -> &mut MatrixView<T> {
        self
    }
}

impl<T, const R: usize, const C: usize> PartialEq for Matrix<T, R, C>
where
    T: Copy + Default + Num,
{
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        // only data represents the matrix
        self.data == other.data
    }
}

impl<T, const R: usize, const C: usize> PartialEq<HeapMatrix<T>> for Matrix<T, R, C>
where
    T: Copy + Default + Num,
{
    #[inline(always)]
    fn eq(&self, other: &HeapMatrix<T>) -> bool {
        PartialEq::<MatrixView<T>>::eq(self, other)
    }
}

impl<T, const R: usize, const C: usize> PartialEq<MatrixView<T>> for Matrix<T, R, C>
where
    T: Copy + Default + Num,
{
    fn eq(&self, other: &MatrixView<T>) -> bool {
        if other.dim() == (R, C) {
            Iterator::eq(self.values(), other.values())
        } else {
            false
        }
    }
}

impl<T, const R: usize, const C: usize> Eq for Matrix<T, R, C> where T: Copy + Default + Num {}

impl<T, const R: usize, const C: usize> Clone for Matrix<T, R, C>
where
    T: Copy + Default + Num,
{
    fn clone(&self) -> Self {
        Self {
            header: UnsafeCell::new(unsafe { *self.header.get() }),
            data: self.data,
            _pinned: PhantomPinned,
        }
    }
}

impl<T, const R: usize, const C: usize> Mul<Vector<T, C>> for Matrix<T, R, C>
where
    T: Default + Copy + Num,
{
    type Output = Vector<T, R>;

    fn mul(self, rhs: Vector<T, C>) -> Self::Output {
        let mut res: Self::Output = Default::default();

        for i in 0..R {
            let row = self.data[i];
            let sum = row
                .iter()
                .enumerate()
                .fold(T::default(), |acc, (j, v)| *v * rhs[j] + acc);
            res.scalars[i] = sum;
        }

        res
    }
}

impl<T, const R: usize, const C: usize> Add for Matrix<T, R, C>
where
    T: Copy + Default + Num,
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        for i in 0..R {
            for j in 0..C {
                self[(i, j)] = self[(i, j)] + rhs[(i, j)]
            }
        }

        self
    }
}

impl<T, const R: usize, const C: usize> Zero for Matrix<T, R, C>
where
    T: Default + Copy + Num,
{
    fn zero() -> Self {
        Self::from_array([[T::zero(); C]; R])
    }

    fn is_zero(&self) -> bool {
        self.data == Self::zero().data
    }
}

impl<T, const R: usize, const C: usize> Index<(usize, usize)> for Matrix<T, R, C>
where
    T: Default + Copy + Num,
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0][index.1]
    }
}

impl<T, const R: usize, const C: usize> IndexMut<(usize, usize)> for Matrix<T, R, C>
where
    T: Default + Copy + Num,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0][index.1]
    }
}

impl<T, const R: usize, const C: usize> fmt::Display for Matrix<T, R, C>
where
    T: Default + Copy + Num + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let strings = self.data.map(|row| row.map(|x| x.to_string()));
        let max_len = {
            let mut out = [0; C];

            for (i, cur) in out.iter_mut().enumerate() {
                for row in &strings {
                    let len = row[i].len();

                    if len > *cur {
                        *cur = len;
                    }
                }
            }

            out
        };

        let len = strings.len();
        for (i, row) in strings.into_iter().enumerate() {
            f.write_str("[ ")?;

            for (i, cell) in row.into_iter().enumerate() {
                write!(f, "{cell:<len$}", len = max_len[i] + 1)?;
            }

            f.write_str("]")?;

            if i < len {
                f.write_char('\n')?;
            }
        }

        Ok(())
    }
}

impl<T, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T, R, C>
where
    T: Default + Copy + Num,
{
    fn from(data: [[T; C]; R]) -> Self {
        Self::from_array(data)
    }
}

impl<T, const R: usize, const C: usize> TryFrom<HeapMatrix<T>> for Matrix<T, R, C>
where
    T: Default + Num + Copy,
{
    type Error = HeapMatrix<T>;

    fn try_from(value: HeapMatrix<T>) -> Result<Self, Self::Error> {
        let (rows, cols) = value.dim();

        if rows == R && cols == C {
            let (chunks, &[]) = value.data.as_chunks::<C>() else {
                return Err(value);
            };
            let data = chunks.as_array().unwrap();
            Ok(Self::from_array(*data))
        } else {
            Err(value)
        }
    }
}

impl<T, A, const R: usize, const C: usize> FromIterator<A> for Matrix<T, R, C>
where
    T: Default + Copy + Num,
    A: IntoIterator<Item = T>,
{
    fn from_iter<K: IntoIterator<Item = A>>(iter: K) -> Self {
        let mut ret = Self::zero();
        let mut last_i = None;

        for (i, row) in iter.into_iter().enumerate() {
            if i == R {
                panic!("too many rows");
            }

            let mut row_buffer = [MaybeUninit::uninit(); C];
            let mut seen = 0;

            let row = &mut row.into_iter();
            for (k, v) in row.enumerate().take(C) {
                seen = k;
                row_buffer[k].write(v);
            }

            if row.count() != 0 && seen != C - 1 {
                panic!("invalid amount of values in row")
            }

            // SAFETY: all elements are initiliazed.
            ret.data[i].copy_from_slice(unsafe { row_buffer.assume_init_ref() });
            last_i.replace(i);
        }

        if last_i.is_none_or(|k| k + 1 != R) {
            panic!("invalid amount of rows");
        }

        ret
    }
}

impl<T, const R: usize, const C: usize, const S: usize> Mul<Matrix<T, S, C>> for Matrix<T, R, S>
where
    T: Default + Copy + Num,
{
    type Output = Matrix<T, R, C>;

    fn mul(self, rhs: Matrix<T, S, C>) -> Self::Output {
        let mut ret = Matrix::zero();

        for i in 0..R {
            for j in 0..C {
                let p = &mut ret[(i, j)];

                for k in 0..S {
                    *p = *p + self[(i, k)] * rhs[(k, j)];
                }
            }
        }

        ret
    }
}

pub trait DoubleRREF<T, const R: usize, const A: usize, const B: usize> {
    fn rref_mut(&mut self);
}

impl<T, const R: usize, const A: usize, const B: usize> DoubleRREF<T, R, A, B> for (Matrix<T, R, A>, Matrix<T, R, B>)
where
    T: Default + Copy + Num + Float,
{
    fn rref_mut(&mut self) {
        self.0.core_rref_mut(&mut RowRecorder::<T, 0>::Matrix(&mut self.1));
    }
}
