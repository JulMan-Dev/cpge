//! This module allows getting a view from a matrix. This takes a reference to a matrix and produces
//! a view matrix, without typed sizes. This may be used to get a subset of a matrix, or getting a
//! view to an entire matrix.

use crate::linear::convolution::ConvolutionRows;
use crate::linear::{Matrix, MatrixRowOperation, Positions};
use crate::mem::{AbstractVec, Living};
use alloc::vec;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::mem;
use core::mem::MaybeUninit;
use core::ops::{Bound, Index, IndexMut, RangeBounds};
use core::ptr::null_mut;
use arrayvec::ArrayVec;
use num_traits::{Float, Num, Zero};

pub(crate) type MatrixViewHeader = (*mut (), usize, [usize; 5]);

/// Creates a new header for matrices. The returned value is unsafe for casting to [`MatrixView`],
/// you must update the header first.
pub(crate) const fn make_header(rows: usize, cols: usize) -> MatrixViewHeader {
    (null_mut(), rows * cols, [0, rows, 0, cols, cols])
}

/// Updates the header.
pub(crate) fn update_header(this: &mut MatrixViewHeader, ptr: *mut ()) {
    this.0 = ptr;
}

/// A matrix view like a slice, `&[T]`, is for an array, `[T; N]`, or a vector, `Vec<T>`, but more
/// advanced.
///
/// This may be seen as a smart constant pointer to a matrix.
#[repr(C)]
pub struct MatrixView<T>
where
    T: Default + Copy + Num,
{
    inner: *mut [T],
    rows: (usize, usize),
    cols: (usize, usize),
    stride: usize,
}

impl<T> MatrixView<T>
where
    T: Default + Copy + Num,
{
    pub(crate) unsafe fn from_header(header: &UnsafeCell<MatrixViewHeader>) -> &Self {
        unsafe {
            let ptr: *mut Self = mem::transmute(header.get());
            &*ptr
        }
    }

    pub(crate) unsafe fn from_header_mut(header: &mut UnsafeCell<MatrixViewHeader>) -> &mut Self {
        unsafe { mem::transmute(header.get_mut()) }
    }

    pub(crate) const fn clone(this: &Self) -> Self {
        Self {
            inner: this.inner,
            rows: this.rows,
            cols: this.cols,
            stride: this.stride,
        }
    }

    /// Makes an empty matrix view. This doesn't require an owned matrix because it doesn't allocate
    /// nor contain any cells. The number of rows and columns are both zero.
    pub const fn empty() -> Self {
        Self {
            // SAFETY: we are making a null pointer, avoiding allocating
            inner: unsafe { MaybeUninit::zeroed().assume_init() },
            rows: (0, 0),
            cols: (0, 0),
            stride: 0,
        }
    }

    const fn as_slice(&self) -> &[T] {
        unsafe { &*self.inner }
    }

    const fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { &mut *self.inner }
    }

    /// Makes a sub-view matrix
    #[inline]
    pub fn view(&self, goggles: impl IntoGoggles) -> Living<'_, Self> {
        goggles.into_goggles().see_through(self)
    }

    #[inline]
    pub fn view_mut(&mut self, goggles: impl IntoGoggles) -> Living<'_, Self, true> {
        goggles.into_goggles().see_through_mut(self)
    }

    #[inline]
    pub const fn dim(&self) -> (usize, usize) {
        let diff_row = self.rows.1 - self.rows.0;
        let diff_col = self.cols.1 - self.cols.0;

        if diff_row == 0 || diff_col == 0 {
            (0, 0)
        } else {
            (diff_row, diff_col)
        }
    }

    /// Gets the goggle range that describes the positions in this matrix view.
    ///
    /// # Example
    ///
    /// ```
    /// # use cpge::linear::Matrix;
    /// # use crate::cpge::linear::view::IntoGoggles;
    /// let matrix = Matrix::from([
    ///     [1, 2],
    ///     [3, 4],
    /// ]);
    /// assert_eq!(matrix.exact_goggles(), (..2, ..2).into_goggles());
    /// ```
    #[inline]
    pub const fn exact_goggles(&self) -> Goggles {
        let (rows, cols) = self.dim();

        Goggles {
            rows: (0, Some(rows)),
            cols: (0, Some(cols)),
        }
    }

    #[inline]
    pub const fn count_rows(&self) -> usize {
        self.dim().0
    }

    #[inline]
    pub const fn count_cols(&self) -> usize {
        self.dim().1
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        let (rows, cols) = self.dim();

        rows == 0 || cols == 0
    }

    pub const fn is_pos_in(&self, pos: (usize, usize)) -> bool {
        let (rows, cols) = self.dim();

        pos.0 < rows && pos.1 < cols
    }

    pub fn to_matrix<const R: usize, const C: usize>(&self) -> Option<Matrix<T, R, C>> {
        let (rows, cols) = self.dim();

        if rows == R && cols == C {
            let mut matrix = Matrix::<T, R, C>::zero();

            for i in 0..R {
                matrix.set_row(i, &self.get_row_array_exact(i)
                    .unwrap()
                    .map(|&x| x));
            }

            Some(matrix)
        } else {
            None
        }
    }

    pub const fn get(&self, pos: (usize, usize)) -> Option<&T> {
        if self.is_pos_in(pos) {
            Some(&self.as_slice()[(self.rows.0 + pos.0) * self.stride + pos.1 + self.cols.0])
        } else {
            None
        }
    }

    pub const fn get_mut(&mut self, pos: (usize, usize)) -> Option<&mut T> {
        if self.is_pos_in(pos) {
            let stride = self.stride;
            Some(&mut self.as_slice_mut()[pos.0 * stride + pos.1])
        } else {
            None
        }
    }

    pub const fn values(&'_ self) -> ViewValues<'_, T> {
        ViewValues { view: self, pos: Some((0, 0)) }
    }

    pub const fn values_mut(&'_ mut self) -> ViewValuesMut<'_, T> {
        ViewValuesMut { view: self, pos: Some((0, 0)) }
    }

    /// Makes an iterator that yields values for convolution result matrix.
    ///
    /// This computation is lazy. When called, no computation is done. You can collect into a stack
    /// matrix as [`Matrix`] implements [`FromIterator`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use cpge::linear::Matrix;
    /// # use num_traits::Zero;
    /// let input = Matrix::from([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9],
    /// ]);
    /// let kernel = Matrix::<i32, 2>::const_identity();
    ///
    /// let mut output: Matrix<i32, 2> = input.convolution(&kernel).collect();
    /// assert_eq!(output, Matrix::from([
    ///     [6, 8],
    ///     [12, 14],
    /// ]));
    /// ```
    pub const fn convolution<'b>(&'b self, kernel: &'b Self) -> ConvolutionRows<'b, T> {
        ConvolutionRows::new(self, kernel)
    }

    pub fn get_row_array_exact<const N: usize>(&self, row: usize) -> Option<[&T; N]> {
        let (rows, cols) = self.dim();

        if row < rows && cols == N {
            let mut vec = ArrayVec::<_, N>::new();

            for i in 0..cols {
                vec.push(&self[(row, i)]);
            }

            // SAFETY: we filled all the elements
            unsafe { Some(vec.into_inner_unchecked()) }
        } else {
            None
        }
    }

    /// Gets the four quadrants from `self` view for the `pivot` point.
    ///
    /// This may return none if the pivot point cannot be used. Note that you can use edge pivots to
    /// get.
    pub const fn quadrants(&self, pivot: (usize, usize)) -> Option<[Living<'_, MatrixView<T>>; 4]> {
        let (rows, cols) = self.dim();
        let valid_range = Goggles::exact(0, rows + 1, 0, cols+ 1);
        let quadrants = Goggles::quadrants(pivot);

        let mut ret = [const { Living::new(Self::empty()) }; 4];
        let mut i = 0;

        while i < 4 {
            let range = quadrants[i].make_contained(valid_range);

            if range.size().unwrap() == 0 {
                return None;
            }

            ret[i] = range.see_through(self);
            i += 1;
        }

        Some(ret)
    }

    pub const fn quadrants_mut(&mut self, pivot: (usize, usize)) -> Option<[Living<'_, MatrixView<T>, true>; 4]> {
        let (rows, cols) = self.dim();
        let valid_range = Goggles::exact(0, rows + 1, 0, cols+ 1);
        let quadrants = Goggles::quadrants(pivot);

        let mut ret = [const { Living::new(Self::empty()) }; 4];
        let mut i = 0;
        let this = self as *mut Self;

        while i < 4 {
            let range = quadrants[i].make_contained(valid_range);

            if range.size().unwrap() == 0 {
                return None;
            }

            ret[i] = range.see_through_mut(unsafe { &mut *this });

            i += 1;
        }

        Some(ret)
    }

    /// Makes an array of mutable references given an array of positions.
    ///
    /// # Safety
    ///
    /// This method does not do any check on positions. It may produce multiple mutable references
    /// to the same element. The caller must ensure that every position passed is different.
    pub const unsafe fn get_disjoint_mut_unchecked<const N: usize>(
        &mut self,
        positions: [(usize, usize); N],
    ) -> Option<[&mut T; N]> {
        let mut ret = [const { MaybeUninit::uninit() }; N];
        let mut i = 0;
        let this = self as *mut Self;

        while i < N {
            // SAFETY: the caller must make sure positions doesn't overlap
            ret[i].write(unsafe {
                match (&mut *this).get_mut(positions[i]) {
                    Some(k) => k,
                    None => return None,
                }
            });

            i += 1;
        }

        // SAFETY: the loop initialized every element
        Some(unsafe {
            mem::transmute_copy::<_, MaybeUninit<[_; N]>>(&ret).assume_init()
        })
    }

    pub const fn get_disjoint_mut<const N: usize>(
        &mut self,
        positions: [(usize, usize); N],
    ) -> Option<[&mut T; N]> {
        let mut i = 0;

        while i < N {
            let mut j = 0;
            let (lower_i, upper_i) = positions[i];

            while j < i {
                let (lower_j, upper_j) = positions[j];

                if lower_i == lower_j && upper_i == upper_j {
                    return None;
                }

                j += 1;
            }

            i += 1;
        }

        // SAFETY: every position are different
        unsafe { self.get_disjoint_mut_unchecked(positions) }
    }

    /// Computes the rank of the matrix. The matrix must be row echelon, do Gaussian elimination
    /// first.
    ///
    /// This is preferred over [`rank`](Self::rank) if the matrix contains floats
    /// because it checks if scalars are near zero and not exactly zero, using
    /// epsilon.
    pub fn rank_float(&self) -> usize
    where
        T: Float,
    {
        self.rows()
            .filter(|row| row.iter().any(|x| x.abs() > T::epsilon()))
            .count()
    }

    /// Computes the rank of the matrix. The matrix must be row echelon, do Gaussian elimination
    /// first.
    pub fn rank(&self) -> usize {
        self.rows()
            .filter(|row| row.iter().any(|x| !x.is_zero()))
            .count()
    }

    #[inline]
    pub const fn is_square(&self) -> Option<usize> {
        let (rows, cols) = self.dim();

        if rows == cols {
            Some(rows)
        } else {
            None
        }
    }

    /// Swaps `i` and `j` lines in `self`.
    ///
    /// # Panic
    ///
    /// This panics if `i` or `j` are over or equal `R`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cpge::linear::matrix::Matrix;
    /// let mut matrix = Matrix::from([[1, 2], [3, 4]]);
    /// matrix.swap_lines(0, 1);
    /// assert_eq!(matrix.data, [[3, 4], [1, 2]]);
    /// ```
    pub const fn swap_lines(&mut self, i: usize, j: usize) {
        let (cols, rows) = self.dim();

        assert!(i < rows && j < rows, "invalid indices");

        if i != j {
            let mut k = 0;

            while k < cols {
                let [i, j] = self.get_disjoint_mut([(i, k), (j, k)]).unwrap();

                (*i, *j) = (*j, *i);
                k += 1;
            }
        }
    }

    /// Adds to the `i`th line `x` times the `j`th line.
    ///
    /// # Panic
    ///
    /// This panics if `i` or `j` are over or equal `R`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cpge::linear::matrix::Matrix;
    /// let mut matrix = Matrix::from([[1, 2], [3, 4]]);
    /// matrix.add_to_line(1, 2, 0); // L1 <- L1 + 2 * L0
    /// assert_eq!(matrix.data, [[1, 2], [5, 8]]);
    /// ```
    pub fn add_to_line(&mut self, i: usize, x: T, j: usize) {
        let (rows, cols) = self.dim();

        assert!(i < rows && j < rows, "invalid indices");

        for k in 0..cols {
            self[(i, k)] = self[(i, k)] + x * self[(j, k)];
        }
    }

    /// Multiplies the `i`th line by `x`.
    ///
    /// # Panic
    ///
    /// This panics if `i` is over or equal `R`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cpge::linear::matrix::Matrix;
    /// let mut matrix = Matrix::from([[1, 2], [3, 4]]);
    /// matrix.mul_line(0, 2);
    /// assert_eq!(matrix.data, [[2, 4], [3, 4]]);
    /// ```
    pub fn mul_line(&mut self, i: usize, x: T) {
        let (rows, cols) = self.dim();
        assert!(i < rows, "invalid index");

        for k in 0..cols {
            self[(i, k)] = self[(i, k)] * x;
        }
    }

    pub fn replay_steps_mut(&mut self, steps: &[MatrixRowOperation<T>]) {
        for step in steps {
            use crate::linear::gauss::MatrixRowOperation::*;

            match *step {
                Swap(i, j) => self.swap_lines(i, j),
                Add(i, x, j) => self.add_to_line(i, x, j),
                Mul(i, x) => self.mul_line(i, x),
            }
        }
    }

    fn core_gaussian_elimination_mut<const N: usize>(&mut self, recorder: &mut RowRecorder<T, N>)
    where
        T: Float,
    {
        use MatrixRowOperation::*;

        let (rows, cols) = self.dim();

        let mut i = 0;
        let mut pivots: Vec<bool> = vec![false; Ord::max(rows, cols)];

        loop {
            if i == rows || i == cols {
                break;
            }

            // working on row `i`.
            // looking on each row
            let found = self.rows()
                .enumerate()
                .skip(i)
                .find(|(_, x)| x[i].abs() > T::epsilon());

            let Some((j, v)) = found else {
                // whole column is zero
                pivots[i] = false;
                i += 1;
                continue;
            };
            let v_i = *v[i];

            self.swap_lines(i, j);
            recorder.push(Swap(i, j));

            // making pivot equal to 1, easing further computations
            self.mul_line(i, v_i.recip());
            recorder.push(Mul(i, v_i.recip()));

            // neutralization of lower lines
            for j in i + 1..rows {
                let scalar = self[(j, i)];

                if scalar.abs() > T::epsilon() {
                    self.add_to_line(j, scalar.neg(), i);
                    recorder.push(Add(j, scalar.recip(), i));
                }
            }

            pivots[i] = true;
            i += 1;
        }
    }

    #[inline]
    pub fn gaussian_elimination_mut(&mut self)
    where
        T: Float,
    {
        self.core_gaussian_elimination_mut(&mut RowRecorder::<T, 0>::None);
    }

    #[inline]
    pub fn record_composition_gaussian_elimination_mut<const N: usize>(&mut self) -> Matrix<T, N>
    where
        T: Float,
    {
        let mut matrix = Matrix::identity();
        self.core_gaussian_elimination_mut(&mut RowRecorder::Composition(&mut matrix));
        matrix
    }

    #[inline]
    pub fn record_in_gaussian_elimination_mut(&mut self, vec: &mut dyn AbstractVec<MatrixRowOperation<T>>)
    where
        T: Float,
    {
        self.core_gaussian_elimination_mut(&mut RowRecorder::<T, 0>::Vec(vec));
    }

    fn core_reduced_row_echelon_mut<const N: usize>(&mut self, recorder: &mut RowRecorder<T, N>)
    where
        T: Float,
    {
        let rank = self.rank_float();

        for i in (0..rank).rev() {
            if i == 0 {
                // end of work
                continue;
            }

            let (pivot_index, _) = self.row(i)
                .enumerate()
                .find(|(_, x)| x.abs() > T::epsilon())
                .expect("pivot must exist");

            for j in 0..i {
                let scalar = self[(j, pivot_index)];

                self.add_to_line(j, scalar.neg(), i);
                recorder.push(MatrixRowOperation::Add(j, scalar.recip(), i));
            }
        }
    }

    /// Computes the unique reduced row echelon form for `self`.
    ///
    /// The matrix must be row echelon, do Gaussian elimination before.
    pub fn reduced_row_echelon_mut(&mut self)
    where
        T: Float,
    {
        self.core_reduced_row_echelon_mut(&mut RowRecorder::<T, 0>::None);
    }

    #[inline]
    pub fn record_composition_reduced_row_echelon_mut<const N: usize>(&mut self) -> Matrix<T, N>
    where
        T: Float,
    {
        let mut matrix = Matrix::identity();
        self.core_reduced_row_echelon_mut(&mut RowRecorder::Composition(&mut matrix));
        matrix
    }

    #[inline]
    pub fn record_in_reduced_row_echelon_mut(&mut self, vec: &mut dyn AbstractVec<MatrixRowOperation<T>>)
    where
        T: Float,
    {
        self.core_reduced_row_echelon_mut(&mut RowRecorder::<T, 0>::Vec(vec));
    }

    pub(super) fn core_rref_mut<const N: usize>(&mut self, recorder: &mut RowRecorder<T, N>)
    where
        T: Float,
    {
        self.core_gaussian_elimination_mut(recorder);
        self.core_reduced_row_echelon_mut(recorder);
    }

    pub fn rref_mut(&mut self)
    where
        T: Float,
    {
        self.core_rref_mut(&mut RowRecorder::<T, 0>::None);
    }

    #[inline]
    pub fn record_composition_rref_mut<const N: usize>(&mut self) -> Matrix<T, N>
    where
        T: Float,
    {
        let mut matrix = Matrix::identity();
        self.core_rref_mut(&mut RowRecorder::Composition(&mut matrix));
        matrix
    }

    #[inline]
    pub fn record_in_rref_mut(&mut self, vec: &mut dyn AbstractVec<MatrixRowOperation<T>>)
    where
        T: Float,
    {
        self.core_rref_mut(&mut RowRecorder::<T, 0>::Vec(vec));
    }

    pub fn is_identity(&self) -> bool {
        if self.is_square().is_some() {
            for (pos, e) in self.values().with_positions() {
                let test = if pos.0 == pos.1 { e.is_one() } else { e.is_zero() };

                if !test {
                    return false;
                }
            }

            true
        } else {
            false
        }
    }

    pub fn is_identity_float(&self) -> bool
    where
        T: Float,
    {
        if self.is_square().is_some() {
            let (one, zero) = (T::one(), T::zero());

            for (pos, e) in self.values().with_positions() {
                let predicate = if pos.0 == pos.1 { one } else { zero };

                if (*e - predicate).abs() > T::epsilon() {
                    return false;
                }
            }

            true
        } else {
            false
        }
    }
}

impl<T: Default + Copy + Num> Index<(usize, usize)> for MatrixView<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index).expect("Index out of bounds")
    }
}

impl<T: Default + Copy + Num> IndexMut<(usize, usize)> for MatrixView<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.get_mut(index).expect("Index out of bounds")
    }
}

impl<T: Default + Copy + Num> PartialEq for MatrixView<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.dim() == other.dim() {
            Iterator::eq(self.values(), other.values())
        } else {
            false
        }
    }
}

impl<T: Default + Copy + Num, const R: usize, const C: usize> PartialEq<Matrix<T, R, C>> for MatrixView<T> {
    #[inline(always)]
    fn eq(&self, other: &Matrix<T, R, C>) -> bool {
        PartialEq::<MatrixView<T>>::eq(self, other)
    }
}

impl<T: Default + Copy + Num> Eq for MatrixView<T> {}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum InvertError {
    NotSquare,
    NotInvertible,
}

pub(super) enum RowRecorder<'a, T: Default + Copy + Num, const N: usize = 0> {
    None,
    Vec(&'a mut dyn AbstractVec<MatrixRowOperation<T>>),
    Matrix(&'a mut MatrixView<T>),
    Composition(&'a mut Matrix<T, N>),
}

impl<T: Default + Copy + Num, const N: usize> RowRecorder<'_, T, N> {
    pub fn push(&mut self, action: MatrixRowOperation<T>) {
        match self {
            Self::None => {},
            Self::Vec(vec) => vec.push(action),
            Self::Matrix(matrix) => matrix.replay_steps_mut(&[action]),
            Self::Composition(matrix) => {
                let operation = action.as_matrix();
                **matrix = operation * matrix.clone();
            },
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Goggles {
    rows: (usize, Option<usize>),
    cols: (usize, Option<usize>),
}

impl Goggles {
    #[inline]
    pub const fn exact(row_lower: usize, row_upper: usize, col_lower: usize, col_upper: usize) -> Self {
        Self {
            rows: (row_lower, Some(row_upper)),
            cols: (col_lower, Some(col_upper)),
        }
    }

    #[inline]
    pub const fn quadrants(pivot: (usize, usize)) -> [Self; 4] {
        let (row, col) = pivot;

        [
            Self { rows: (0, Some(row)), cols: (0, Some(col)) },
            Self { rows: (row, None), cols: (0, Some(col)) },
            Self { rows: (0, Some(row)), cols: (col, None) },
            Self { rows: (row, None), cols: (col, None) },
        ]
    }

    pub const fn shift_by(self, offset: (usize, usize)) -> Self {
        Self {
            rows: {
                let lower = self.rows.0 + offset.0;
                let upper = match self.rows.1 {
                    Some(k) => Some(k + offset.0),
                    None => None,
                };

                (lower, upper)
            },
            cols: {
                let lower = self.cols.0 + offset.1;
                let upper = match self.cols.1 {
                    Some(k) => Some(k + offset.1),
                    None => None,
                };

                (lower, upper)
            },
        }
    }

    /// Gives the next position in the goggle given the original one. This may return none if the
    /// original position is already the last.
    pub const fn next(self, pos: (usize, usize)) -> Option<(usize, usize)> {
        let (row, col) = pos;
        let is_row_full = match self.cols {
            (lower, Some(upper)) => col >= upper - lower - 1,
            (_, None) => false,
        };

        if is_row_full {
            // increment row
            if let (lower, Some(upper)) = self.rows && row >= upper - lower - 1 {
                None
            } else {
                Some((row + 1, 0))
            }
        } else {
            Some((row, col + 1))
        }
    }

    pub const fn nth(self, pos: (usize, usize), mut n: usize) -> Option<(usize, usize)> {
        if n == 0 {
            return Some(pos);
        }

        let stride = match self.cols {
            (lower, Some(upper)) => upper - lower,
            (_, None) => return Some((pos.0, pos.1 + n)),
        };

        // rounding to next row
        let new_pos = if stride > pos.1 {
            let diff = stride - pos.1;

            if diff > n {
                return Some((pos.0, pos.1 + n));
            } else if diff == n {
                return Some((pos.0 + 1, 0));
            } else {
                n -= diff;
                (pos.0 + 1, 0)
            }
        } else {
            // pos is not in the goggle if we reach here
            return None;
        };

        let q = n.div_euclid(stride);
        let r = n.rem_euclid(stride);
        let new_pos = (new_pos.0 + q, new_pos.1 + r);

        if self.is_relative_in(new_pos) {
            Some(new_pos)
        } else {
            None
        }
    }

    pub const fn see_through<T>(self, matrix: &MatrixView<T>) -> Living<'_, MatrixView<T>>
    where
        T: Default + Copy + Num,
    {
        let mut view = MatrixView::clone(matrix);
        let goggles = view.exact_goggles();

        if let Some(result) = goggles.compose(self) {
            let (rows, cols) = goggles.make_contained(result).into_region(view.dim());
            view.rows = rows;
            view.cols = cols;
            Living::new(view)
        } else {
            Living::new(MatrixView::empty())
        }
    }

    pub const fn see_through_mut<T>(self, matrix: &mut MatrixView<T>) -> Living<'_, MatrixView<T>, true>
    where
        T: Default + Copy + Num,
    {
        let mut view = MatrixView::clone(matrix);
        let goggles = view.exact_goggles();

        if let Some(result) = goggles.compose(self) {
            let (rows, cols) = goggles.make_contained(result).into_region(view.dim());
            view.rows = rows;
            view.cols = cols;
            Living::new(view)
        } else {
            Living::new(MatrixView::empty())
        }
    }

    /// Checks if `target` represents a relative subregion of `self`. Unbound goggles are considered
    /// subregion.
    pub const fn can_compose_with(self, target: Self) -> bool {
        // sorry for producing such ugly conditional code...

        (match (self.rows, target.rows) {
            // if the origin is unbound, the target may be anything
            ((_, None), (_, _)) => true,
            ((origin_lower, Some(origin_upper)), (target_lower, None))
            => target_lower < origin_upper - origin_lower,
            ((origin_lower, Some(origin_upper)), (target_lower, Some(target_upper)))
            => target_lower < origin_upper - origin_lower && target_upper <= origin_upper - origin_lower,
        }) && match (self.cols, target.cols) {
            // if the origin is unbound, the target may be anything
            ((_, None), (_, _)) => true,
            ((origin_lower, Some(origin_upper)), (target_lower, None))
            => target_lower < origin_upper - origin_lower,
            ((origin_lower, Some(origin_upper)), (target_lower, Some(target_upper)))
            => target_lower < origin_upper - origin_lower && target_upper <= origin_upper - origin_lower,
        }
    }

    pub const fn compose(self, target: Self) -> Option<Self> {
        if self.can_compose_with(target) {
            let rows = {
                let lower = self.rows.0 + target.rows.0;
                let upper = match target.rows.1 {
                    None => self.rows.1,
                    Some(k) => Some(self.rows.0 + k),
                };

                (lower, upper)
            };

            let cols = {
                let lower = self.cols.0 + target.cols.0;
                let upper = match target.cols.1 {
                    None => self.cols.1,
                    Some(k) => Some(self.cols.0 + k),
                };

                (lower, upper)
            };

            Some(Self { rows, cols })
        } else {
            None
        }
    }

    pub const fn make_contained(self, target: Self) -> Self {
        macro_rules! const_op {
            (max => $lhs:expr, $rhs:expr) => {{
                let lhs = $lhs;
                let rhs = $rhs;

                if lhs >= rhs {
                    lhs
                } else {
                    rhs
                }
            }};
            (min => $lhs:expr, $rhs:expr) => {{
                let lhs = $lhs;
                let rhs = $rhs;

                if lhs <= rhs {
                    lhs
                } else {
                    rhs
                }
            }};
        }

        let rows = {
            let lower = const_op!(max => self.rows.0, target.rows.0);
            let upper = match (self.rows.1, target.rows.1) {
                (Some(k), None) | (None, Some(k)) => Some(k),
                (Some(a), Some(b)) => Some(const_op!(min => a, b)),
                (None, None) => None,
            };

            (lower, upper)
        };

        let cols = {
            let lower = const_op!(max => self.cols.0, target.cols.0);
            let upper = match (self.cols.1, target.cols.1) {
                (Some(k), None) | (None, Some(k)) => Some(k),
                (Some(a), Some(b)) => Some(const_op!(min => a, b)),
                (None, None) => None,
            };

            (lower, upper)
        };

        Self { rows, cols }
    }

    pub const fn into_region(self, end: (usize, usize)) -> ((usize, usize), (usize, usize)) {
        let (rows, cols) = end;

        let this = Goggles::exact(0, rows, 0, cols).make_contained(self);

        (match this.rows {
            (a, Some(b)) => (a, b),
            (_, None) => unreachable!(),
        }, match this.cols {
            (a, Some(b)) => (a, b),
            (_, None) => unreachable!(),
        })
    }

    pub const fn size(self) -> Option<usize> {
        let rows = match self.rows {
            (a, Some(b)) if a == b => return Some(0),
            (lower, Some(upper)) => Some(upper - lower),
            (_, None) => None,
        };

        let cols = match self.cols {
            (a, Some(b)) if a == b => return Some(0),
            (lower, Some(upper)) => Some(upper - lower),
            (_, None) => None,
        };

        match (rows, cols) {
            (Some(rows), Some(cols)) => Some(rows * cols),
            _ => None,
        }
    }

    pub const fn is_relative_in(self, pos: (usize, usize)) -> bool {
        let rows = match self.rows {
            (a, Some(b)) if a == b => return false,
            (lower, Some(upper)) => pos.0 < upper - lower,
            (_, None) => true,
        };

        let cols = match self.cols {
            (a, Some(b)) if a == b => return false,
            (lower, Some(upper)) => pos.1 < upper - lower,
            (_, None) => true,
        };

        rows & cols
    }
}

pub trait IntoGoggles {
    fn into_goggles(self) -> Goggles;
}

impl IntoGoggles for Goggles {
    fn into_goggles(self) -> Goggles {
        self
    }
}

impl IntoGoggles for usize {
    fn into_goggles(self) -> Goggles {
        Goggles {
            rows: (self, Some(self + 1)),
            cols: (0, None),
        }
    }
}

macro_rules! impl_range {
    ($($r:ty)+) => {
        $(impl IntoGoggles for $r {
            fn into_goggles(self) -> Goggles {
                Goggles {
                    rows: (match RangeBounds::start_bound(&self) {
                        Bound::Included(&a) => a,
                        Bound::Excluded(&a) => a + 1,
                        Bound::Unbounded => 0,
                    }, match RangeBounds::end_bound(&self) {
                        Bound::Included(&a) => Some(a + 1),
                        Bound::Excluded(&a) => Some(a),
                        Bound::Unbounded => None,
                    }),
                    cols: (0, None),
                }
            }
        })+
    };
}

impl_range! {
    core::ops::RangeFull
    core::ops::RangeFrom<usize>
    core::ops::RangeTo<usize>
    core::ops::RangeInclusive<usize>
    core::ops::RangeToInclusive<usize>
}

macro_rules! impl_double {
    ($(($lt:ty, $rt:ty))+) => {
        $(impl IntoGoggles for ($lt, $rt) {
            fn into_goggles(self) -> Goggles {
                let row = self.0.into_goggles();
                let col = self.1.into_goggles();

                Goggles {
                    rows: row.rows,
                    cols: col.rows,
                }
            }
        })+
    };
}

impl_double! {
    (usize, usize)
    (usize, core::ops::RangeFull)
    (usize, core::ops::RangeFrom<usize>)
    (usize, core::ops::RangeTo<usize>)
    (usize, core::ops::RangeInclusive<usize>)
    (usize, core::ops::RangeToInclusive<usize>)
    (core::ops::RangeFull, usize)
    (core::ops::RangeFull, core::ops::RangeFull)
    (core::ops::RangeFull, core::ops::RangeFrom<usize>)
    (core::ops::RangeFull, core::ops::RangeTo<usize>)
    (core::ops::RangeFull, core::ops::RangeInclusive<usize>)
    (core::ops::RangeFull, core::ops::RangeToInclusive<usize>)
    (core::ops::RangeFrom<usize>, usize)
    (core::ops::RangeFrom<usize>, core::ops::RangeFull)
    (core::ops::RangeFrom<usize>, core::ops::RangeFrom<usize>)
    (core::ops::RangeFrom<usize>, core::ops::RangeTo<usize>)
    (core::ops::RangeFrom<usize>, core::ops::RangeInclusive<usize>)
    (core::ops::RangeFrom<usize>, core::ops::RangeToInclusive<usize>)
    (core::ops::RangeTo<usize>, usize)
    (core::ops::RangeTo<usize>, core::ops::RangeFull)
    (core::ops::RangeTo<usize>, core::ops::RangeFrom<usize>)
    (core::ops::RangeTo<usize>, core::ops::RangeTo<usize>)
    (core::ops::RangeTo<usize>, core::ops::RangeInclusive<usize>)
    (core::ops::RangeTo<usize>, core::ops::RangeToInclusive<usize>)
    (core::ops::RangeInclusive<usize>, usize)
    (core::ops::RangeInclusive<usize>, core::ops::RangeFull)
    (core::ops::RangeInclusive<usize>, core::ops::RangeFrom<usize>)
    (core::ops::RangeInclusive<usize>, core::ops::RangeTo<usize>)
    (core::ops::RangeInclusive<usize>, core::ops::RangeInclusive<usize>)
    (core::ops::RangeInclusive<usize>, core::ops::RangeToInclusive<usize>)
    (core::ops::RangeToInclusive<usize>, usize)
    (core::ops::RangeToInclusive<usize>, core::ops::RangeFull)
    (core::ops::RangeToInclusive<usize>, core::ops::RangeFrom<usize>)
    (core::ops::RangeToInclusive<usize>, core::ops::RangeTo<usize>)
    (core::ops::RangeToInclusive<usize>, core::ops::RangeInclusive<usize>)
    (core::ops::RangeToInclusive<usize>, core::ops::RangeToInclusive<usize>)
}

pub struct ViewValues<'a, T>
where
    T: 'a + Default + Num + Copy,
{
    view: &'a MatrixView<T>,
    pos: Option<(usize, usize)>,
}

impl<'a, T: 'a + Default + Num + Copy> ViewValues<'a, T> {
    pub fn with_positions(self) -> Positions<Self> {
        let goggles = self.view.exact_goggles();
        let cur = self.pos.unwrap();
        Positions::new_at(self, goggles, cur)
    }
}

impl<'a, T> Iterator for ViewValues<'a, T>
where
    T: 'a + Default + Num + Copy,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(pos) = self.pos && self.view.is_pos_in(pos) {
            let ret = &self.view[pos];
            self.pos = self.view.exact_goggles().next(pos);
            Some(ret)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (rows, cols) = self.view.dim();
        let size = rows * cols;

        (size, Some(size))
    }
}

impl<'a, T> ExactSizeIterator for ViewValues<'a, T>
where
    T: 'a + Default + Num + Copy,
{
}

pub struct ViewValuesMut<'a, T>
where
    T: 'a + Default + Num + Copy,
{
    view: &'a mut MatrixView<T>,
    pos: Option<(usize, usize)>,
}

impl<'a, T: 'a + Default + Num + Copy> ViewValuesMut<'a, T> {
    pub const fn with_positions(self) -> Positions<Self> {
        let goggles = self.view.exact_goggles();
        let cur = self.pos.unwrap();
        Positions::new_at(self, goggles, cur)
    }
}

impl<'a, T> Iterator for ViewValuesMut<'a, T>
where
    T: 'a + Default + Num + Copy,
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(pos) = self.pos && self.view.is_pos_in(pos) {
            let new_pos = self.view.exact_goggles().next(pos);
            let ret = &mut self.view[pos];
            self.pos = new_pos;
            Some(unsafe { &mut *(ret as *mut T) })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (rows, cols) = self.view.dim();
        let size = rows * cols;

        (size, Some(size))
    }
}

impl<'a, T> ExactSizeIterator for ViewValuesMut<'a, T>
where
    T: 'a + Default + Num + Copy,
{
}

mod heap {
    use core::ops::Mul;
use alloc::borrow::ToOwned;
    use alloc::boxed::Box;
    use alloc::vec;
    use alloc::vec::Vec;
    use num_traits::{Float, Num, Zero};
    use crate::linear::{HeapMatrix, InvertError, MatrixRowOperation, MatrixView, TransparentMatrix, Vector, VectorFamily};
    use crate::linear::iter::{MatrixCellIter, MatrixCellIterMut, MatrixCols, MatrixColsMut, MatrixRows, MatrixRowsMut};
    use crate::linear::product::{impl_product, impl_scalar};
    use crate::linear::view::RowRecorder;
    use crate::mem::Owned;

    impl<T> MatrixView<T>
    where
        T: Default + Num + Copy,
    {
        /// Converts `self` matrix to an array of column vectors.
        pub fn to_vectors<const N: usize>(&self) -> Owned<VectorFamily<T, N>, Box<[Vector<T, N>]>> {
            let (rows, cols) = self.dim();
            assert_eq!(N, rows, "N must be the rows count");

            let mut vectors = vec![Vector::<T, N>::zero(); cols];

            let mut i = 0;
            while i < cols {
                let mut j = 0;
                while j < rows {
                    vectors[i].scalars[j] = self[(j, i)];

                    j += 1;
                }

                i += 1;
            }

            Owned::new(vectors.into_boxed_slice())
        }

        /// Copies this view matrix to a heap matrix.
        pub fn to_heap(&self) -> HeapMatrix<T> {
            let (rows, cols) = self.dim();

            if rows == 0 || cols == 0 {
                HeapMatrix::empty()
            } else {
                let inner = self.values().copied().collect();

                HeapMatrix::new(inner, rows, cols)
            }
        }

        /// Converts `self` matrix to an array of row vectors.
        pub fn to_row_vectors<const N: usize>(&self) -> Owned<VectorFamily<T, N>, Box<[Vector<T, N>]>> {
            self.transpose().to_vectors()
        }

        /// Gets a values iterator for a matrix row.
        ///
        /// The returned iterator yields exactly `C` elements, or nothing if `row` >= `R`.
        ///
        /// # Example
        ///
        /// ```
        /// # use cpge::linear::Matrix;
        /// let matrix = Matrix::from([[1, 2], [3, 4]]);
        /// let mut iter = matrix.row(1);
        /// assert_eq!(iter.next(), Some(&3));
        /// assert_eq!(iter.next(), Some(&4));
        /// assert_eq!(iter.next(), None);
        ///
        /// let mut iter = matrix.row(2);
        /// assert_eq!(iter.len(), 0);
        /// ```
        pub const fn row(&'_ self, row: usize) -> MatrixCellIter<'_, T> {
            MatrixCellIter::new_row(self, row)
        }


        /// Gets a mutable values iterator from a matrix row.
        ///
        /// The returned iterator yields exactly `C` elements, or nothing if `row` >= `R`.
        ///
        /// # Example
        ///
        /// ```
        /// # use cpge::linear::Matrix;
        /// let mut matrix = Matrix::from([
        ///     [1.0, 2.0, 0.0],
        ///     [0.0, 4.0, 2.0],
        ///     [3.0, 9.0, 6.0]
        /// ]);
        ///
        /// // normalize each row by its "pivot" (first non-zero).
        /// for i in 0..3 {
        ///     let pivot: f64 = match matrix.row(i).find(|&&x| f64::abs(x) > f64::EPSILON) {
        ///         Some(&k) => k,
        ///         None => unreachable!(),
        ///     };
        ///
        ///     for u in matrix.row_mut(i) {
        ///         *u /= pivot;
        ///     }
        /// }
        ///
        /// let m = Matrix::from([
        ///     [1.0, 2.0, 0.0],
        ///     [0.0, 1.0, 0.5],
        ///     [1.0, 3.0, 2.0],
        /// ]);
        /// assert_eq!(matrix, m);
        /// ```
        pub const fn row_mut(&'_ mut self, row: usize) -> MatrixCellIterMut<'_, T> {
            MatrixCellIterMut::new_row(self, row)
        }

        /// Gets a values iterator for a matrix column.
        ///
        /// The returned iterator yields exactly `R` elements, or nothing if `col` >= `C`.
        ///
        /// # Example
        ///
        /// ```
        /// # use cpge::linear::Matrix;
        /// let matrix = Matrix::from([[1, 2], [3, 4]]);
        /// let mut iter = matrix.col(0);
        /// assert_eq!(iter.next(), Some(&1));
        /// assert_eq!(iter.next(), Some(&3));
        /// assert_eq!(iter.next(), None);
        ///
        /// let mut iter = matrix.col(2);
        /// assert_eq!(iter.len(), 0);
        /// ```
        pub const fn col(&'_ self, col: usize) -> MatrixCellIter<'_, T> {
            MatrixCellIter::new_col(self, col)
        }

        /// Gets a mutable values iterator from a matrix column.
        ///
        /// The returned iterator yields exactly `R` elements, or nothing if `col` >= `C`.
        ///
        /// # Example
        ///
        /// ```
        /// # use cpge::linear::Matrix;
        /// let mut matrix = Matrix::from([
        ///     [3.0, 0.0],
        ///     [4.0, 5.0],
        /// ]);
        ///
        /// // normalize each vectors
        /// for i in 0..2 {
        ///     let norm: f64 = matrix.col(i).map(|x| x * x).sum::<f64>().sqrt();
        ///
        ///     for u in matrix.col_mut(i) {
        ///         *u /= norm;
        ///     }
        /// }
        ///
        /// let m = Matrix::from([
        ///     [0.6, 0.0],
        ///     [0.8, 1.0],
        /// ]);
        /// assert_eq!(matrix, m);
        /// ```
        pub const fn col_mut(&'_ mut self, col: usize) -> MatrixCellIterMut<'_, T> {
            MatrixCellIterMut::new_col(self, col)
        }

        /// Gets a row iterator from `self`.
        ///
        /// It yields exactly `R` rows.
        ///
        /// # Example
        ///
        /// ```
        /// # use cpge::linear::Matrix;
        /// let matrix: Matrix<i32, 3> = Matrix::const_identity();
        /// let mut iter = matrix.rows();
        /// assert_eq!(iter.next(), Some(Box::from([&1, &0, &0])));
        /// assert_eq!(iter.next(), Some(Box::from([&0, &1, &0])));
        /// assert_eq!(iter.next(), Some(Box::from([&0, &0, &1])));
        /// assert_eq!(iter.next(), None);
        /// ```
        pub const fn rows(&'_ self) -> MatrixRows<'_, T> {
            MatrixRows::new(self)
        }

        /// Gets an iterator of mutable rows from `self`.
        ///
        /// It yields exactly `R` rows.
        ///
        /// # Example
        ///
        /// ```
        /// # use cpge::linear::Matrix;
        /// let mut matrix = Matrix::from([
        ///     [1, 2, 3, 0],
        ///     [0, 1, 2, 0],
        ///     [0, 0, 1, 0],
        /// ]);
        ///
        /// // last column will hold sum of row.
        /// for row in matrix.rows_mut() {
        ///     *row[3] = row[0..3].iter().fold(0, |acc, x| acc + **x);
        /// }
        ///
        /// let last_column: Vec<_> = matrix.col(3).collect();
        /// assert_eq!(last_column, vec![&6, &3, &1]);
        /// ```
        pub const fn rows_mut(&'_ mut self) -> MatrixRowsMut<'_, T> {
            MatrixRowsMut::new(self)
        }

        /// Gets a column iterator from `self`.
        ///
        /// It yields exactly `C` columns.
        ///
        /// As matrices are row-major, it yields arrays of pointers. [`rows`](Self::rows) yields array
        /// pointers because no computation is required.
        ///
        /// # Example
        ///
        /// ```
        /// # use cpge::linear::Matrix;
        /// let matrix: Matrix<i32, 3> = Matrix::const_identity();
        /// let mut iter = matrix.cols();
        /// assert_eq!(iter.next(), Some(Box::from([&1, &0, &0])));
        /// assert_eq!(iter.next(), Some(Box::from([&0, &1, &0])));
        /// assert_eq!(iter.next(), Some(Box::from([&0, &0, &1])));
        /// assert_eq!(iter.next(), None);
        /// ```
        pub const fn cols(&'_ self) -> MatrixCols<'_, T> {
            MatrixCols::new(self)
        }

        /// Gets an iterator of mutable columns from `self`.
        ///
        /// It yields exactly `C` columns.
        ///
        /// As matrices are row-major, it yields arrays of pointers. [`rows_mut`](Self::rows_mut) yields
        /// array pointers because no computation is required.
        ///
        /// # Example
        ///
        /// ```
        /// # use cpge::linear::Matrix;
        /// # use cpge::polynomials::HeapPolynomial;
        /// # use num_traits::ConstZero;
        /// let f = HeapPolynomial::from([1, 2, 3]); // 1 + 2X + 3X^2
        /// let df = f.derivative();
        /// let ddf = df.derivative();
        /// let mut mat: Matrix<usize, 3> = Matrix::ZERO;
        ///
        /// for (i, col) in mat.cols_mut().enumerate() {
        ///     for (&p, u) in Iterator::zip([&f, &df, &ddf].iter(), col) {
        ///         *u = p.apply(i);
        ///     }
        /// }
        ///
        /// let m = Matrix::from([
        ///     [1, 6, 17],   // f(0), f(1), f(2)
        ///     [2, 8, 14],   // df(0), df(1), df(2)
        ///     [6, 6, 6],    // ddf(0), ddf(1), ddf(2)
        /// ]);
        /// assert_eq!(mat, m);
        /// ```
        pub const fn cols_mut(&'_ mut self) -> MatrixColsMut<'_, T> {
            MatrixColsMut::new(self)
        }

        pub fn get_row_array(&self, row: usize) -> Option<Box<[&T]>> {
            let (rows, cols) = self.dim();

            if row < rows {
                let mut vec = Vec::new();

                for i in 0..cols {
                    vec.push(&self[(row, i)]);
                }

                Some(vec.into_boxed_slice())
            } else {
                None
            }
        }

        pub fn get_mut_row_array(&mut self, row: usize) -> Option<Box<[&mut T]>> {
            let (rows, cols) = self.dim();

            if row < rows {
                let mut vec = Vec::new();

                for i in 0..cols {
                    // SAFETY: the previous cell mut refs are not this ref
                    vec.push(unsafe {
                        let this: &mut Self = &mut *(self as *mut _);
                        &mut this[(row, i)]
                    });
                }

                Some(vec.into_boxed_slice())
            } else {
                None
            }
        }

        pub fn get_col_array(&self, col: usize) -> Option<Box<[&T]>> {
            let (rows, cols) = self.dim();

            if col < cols {
                let mut vec = Vec::new();

                for i in 0..rows {
                    vec.push(&self[(i, col)]);
                }

                Some(vec.into_boxed_slice())
            } else {
                None
            }
        }

        pub fn get_mut_col_array(&mut self, col: usize) -> Option<Box<[&mut T]>> {
            let (rows, cols) = self.dim();

            if col < cols {
                let mut vec = Vec::new();

                for i in 0..rows {
                    // SAFETY: the previous cell mut refs are not this ref
                    vec.push(unsafe {
                        let this: &mut Self = &mut *(self as *mut _);
                        &mut this[(i, col)]
                    });
                }

                Some(vec.into_boxed_slice())
            } else {
                None
            }
        }

        /// Copies the view into an owned matrix and transposes it.
        pub fn transpose(&self) -> HeapMatrix<T> {
            self.to_heap().transpose()
        }

        /// Does matrix product with provided right side.
        ///
        /// The right side could anything that is equivalent to a matrix. The result is always the
        /// result matrix, not in the type of the right side.
        ///
        /// # Example
        ///
        /// ```
        /// # use cpge::linear::{Matrix, Vector, TransparentMatrix};
        /// let m = Matrix::from([
        ///     [1, 2],
        ///     [3, 4],
        /// ]);
        ///
        /// let vector = Vector::new([1, 2]);
        /// let vector1 = Vector::from_matrix(&m.product(vector)); // vector converted to matrix
        /// assert_eq!(vector1, Vector::new([5, 11]));
        /// ```
        pub fn product<K>(&self, rhs: K) -> HeapMatrix<T>
        where
            K: TransparentMatrix<MatrixItem = T>,
        {
            self * &*rhs.into_matrix()
        }

        /// Does matrix product with provided left side.
        pub fn product_left<K>(&self, lhs: K) -> HeapMatrix<T>
        where
            K: TransparentMatrix<MatrixItem = T>,
        {
            &*lhs.into_matrix() * self
        }

        pub fn replay_steps(&self, steps: &[MatrixRowOperation<T>]) -> HeapMatrix<T> {
            let mut matrix = self.to_owned();
            matrix.replay_steps_mut(steps);
            matrix
        }

        #[inline]
        pub fn gaussian_elimination(&self) -> HeapMatrix<T>
        where
            T: Float,
        {
            let mut matrix = self.to_owned();
            matrix.gaussian_elimination_mut();
            matrix
        }

        #[inline]
        pub fn record_gaussian_elimination_mut(&mut self) -> Box<[MatrixRowOperation<T>]>
        where
            T: Float,
        {
            let mut vec = Vec::new();
            self.core_gaussian_elimination_mut(&mut RowRecorder::<T, 0>::Vec(&mut vec));
            vec.into_boxed_slice()
        }

        #[inline]
        pub fn record_gaussian_elimination(&self) -> (HeapMatrix<T>, Box<[MatrixRowOperation<T>]>)
        where
            T: Float,
        {
            let mut matrix = self.to_owned();
            let steps = matrix.record_gaussian_elimination_mut();
            (matrix, steps)
        }

        #[inline]
        pub fn gaussian_elimination_mut_with_right<K>(&mut self, right: K) -> K::ResultType
        where
            K: TransparentMatrix<MatrixItem = T>,
            T: Float,
        {
            let mut matrix = right.into_matrix();
            assert_eq!(matrix.count_rows(), self.count_rows());

            self.core_gaussian_elimination_mut(&mut RowRecorder::<T, 0>::Matrix(&mut matrix));
            K::from_matrix(&matrix)
        }

        #[inline]
        pub fn gaussian_elimination_with_right<K>(&self, right: K) -> (HeapMatrix<T>, K::ResultType)
        where
            K: TransparentMatrix<MatrixItem = T>,
            T: Float,
        {
            let mut matrix = self.to_owned();
            let right = matrix.gaussian_elimination_mut_with_right(right);
            (matrix, right)
        }

        #[inline]
        pub fn reduced_row_echelon(&self) -> HeapMatrix<T>
        where
            T: Float,
        {
            let mut matrix = self.to_owned();
            matrix.reduced_row_echelon_mut();
            matrix
        }

        #[inline]
        pub fn record_reduced_row_echelon_mut(&mut self) -> Box<[MatrixRowOperation<T>]>
        where
            T: Float,
        {
            let mut vec = Vec::new();
            self.core_gaussian_elimination_mut(&mut RowRecorder::<T, 0>::Vec(&mut vec));
            vec.into_boxed_slice()
        }

        #[inline]
        pub fn record_reduced_row_echelon(&self) -> (HeapMatrix<T>, Box<[MatrixRowOperation<T>]>)
        where
            T: Float,
        {
            let mut matrix = self.to_owned();
            let steps = matrix.record_reduced_row_echelon_mut();
            (matrix, steps)
        }

        #[inline]
        pub fn reduced_row_echelon_mut_with_right<K>(&mut self, right: K) -> K::ResultType
        where
            T: Float,
            K: TransparentMatrix<MatrixItem = T>,
        {
            let mut matrix = right.into_matrix();
            assert_eq!(matrix.count_rows(), self.count_rows());

            self.core_reduced_row_echelon_mut(&mut RowRecorder::<T, 0>::Matrix(&mut matrix));
            K::from_matrix(&matrix)
        }

        #[inline]
        pub fn reduced_row_echelon_with_right<K>(&self, right: K) -> (HeapMatrix<T>, K::ResultType)
        where
            T: Float,
            K: TransparentMatrix<MatrixItem = T>,
        {
            let mut matrix = self.to_owned();
            let right = matrix.reduced_row_echelon_mut_with_right(right);
            (matrix, right)
        }

        #[inline]
        pub fn rref(&self) -> HeapMatrix<T>
        where
            T: Float,
        {
            let mut matrix = self.to_owned();
            matrix.rref_mut();
            matrix
        }

        #[inline]
        pub fn record_rref_mut(&mut self) -> Box<[MatrixRowOperation<T>]>
        where
            T: Float,
        {
            let mut vec = Vec::new();
            self.core_rref_mut(&mut RowRecorder::<T, 0>::Vec(&mut vec));
            vec.into_boxed_slice()
        }

        #[inline]
        pub fn record_rref(&self) -> (HeapMatrix<T>, Box<[MatrixRowOperation<T>]>)
        where
            T: Float,
        {
            let mut matrix = self.to_owned();
            let steps = matrix.record_rref_mut();
            (matrix, steps)
        }

        #[inline]
        pub fn rref_mut_with_right<K>(&mut self, right: K) -> K::ResultType
        where
            K: TransparentMatrix<MatrixItem = T>,
            T: Float,
        {
            let mut matrix = right.into_matrix();
            assert_eq!(matrix.count_rows(), self.count_rows());

            self.core_rref_mut(&mut RowRecorder::<T, 0>::Matrix(&mut matrix));
            K::from_matrix(&matrix)
        }

        #[inline]
        pub fn rref_with_right<K>(&self, right: K) -> (HeapMatrix<T>, K::ResultType)
        where
            K: TransparentMatrix<MatrixItem = T>,
            T: Float,
        {
            let mut matrix = self.to_owned();
            let right = matrix.rref_mut_with_right(right);
            (matrix, right)
        }

        /// Tries to compute the invert matrix of `self`.
        ///
        /// To check if `self` is invertible without needing `self^-1`, do not do
        /// `self.invert().is_ok()`, do `self.is_invertible()`, its faster.
        pub fn invert(&self) -> Result<HeapMatrix<T>, InvertError>
        where
            T: Float,
        {
            let Some(n) = self.is_square() else {
                return Err(InvertError::NotSquare);
            };

            let mut left = self.to_heap();
            let right = left.rref_mut_with_right(HeapMatrix::identity(n));

            if left.rank_float() == n {
                Ok(right)
            } else {
                Err(InvertError::NotInvertible)
            }
        }

        /// Checks if `self` is invertible.
        ///
        /// ```
        /// # use cpge::linear::matrix::Matrix;
        /// let m = Matrix::<f64, 5>::const_identity();
        /// let n = Matrix::from([[1.0, 2.0], [4.0, 8.0]]);
        ///
        /// assert!(m.is_invertible());
        /// assert!(!n.is_invertible());
        /// ```
        pub fn is_invertible(&self) -> bool
        where
            T: Float,
        {
            if let Some(n) = self.is_square() {
                let mut matrix = self.to_owned();
                matrix.rref_mut();
                matrix.rank_float() == n
            } else {
                false
            }
        }

        /// Computes the determinant of `self`.
        ///
        /// Note: `T` must be `Float` because this can use [`gaussian_elimination`](Self::gaussian_elimination).
        pub fn determinant_mut(&mut self) -> Option<T>
        where
            T: Float,
        {
            let n = self.is_square()?;

            Some(match *self.as_slice() {
                [a, b, c, d] => {
                    a * c - b * d
                }
                [a, b, c, d, e, f, g, h, i] => {
                    a * (e * i - f * h) - d * (b * i - c * h) + g * (b * f - c * e)
                }
                _ => {
                    // TODO: use another algorithm that doesn't rely on Gaussian elimination, allowing to
                    //       remove the `T: Float` precondition thus allowing integers matrices.

                    // generic form, using Gaussian reduction
                    let steps = self.record_gaussian_elimination_mut();

                    let swaps = steps.iter()
                        .filter(|k| matches!(k, MatrixRowOperation::Swap(_, _)))
                        .count();

                    let k = if swaps.is_multiple_of(2) {
                        T::one()
                    } else {
                        T::zero() - T::one()
                    };

                    let det_matrix = (0..n)
                        .map(|k| self[(k, k)])
                        .fold(T::one(), |acc, x| acc * x);

                    k * det_matrix
                }
            })
        }
    }

    // ToOwned is unsound without alloc
    impl<T: Default + Copy + Num> ToOwned for MatrixView<T> {
        type Owned = HeapMatrix<T>;

        #[inline(always)]
        fn to_owned(&self) -> Self::Owned {
            self.to_heap()
        }
    }

    impl<T: Default + Copy + Num> PartialEq<HeapMatrix<T>> for MatrixView<T> {
        #[inline(always)]
        fn eq(&self, other: &HeapMatrix<T>) -> bool {
            PartialEq::<MatrixView<T>>::eq(self, other)
        }
    }

    #[inline(always)]
    fn do_product<T>(lhs: &MatrixView<T>, rhs: &MatrixView<T>) -> HeapMatrix<T>
    where
        T: Default + Copy + Num,
    {
        let (left_rows, left_cols) = lhs.dim();
        let (right_rows, right_cols) = rhs.dim();

        assert_eq!(left_cols, right_rows);

        let mut ret = HeapMatrix::zero(left_rows, right_cols);

        for i in 0..left_rows {
            for j in 0..right_cols {
                let p = &mut ret[(i, j)];

                for k in 0..left_cols {
                    *p = *p + lhs[(i, k)] * rhs[(k, j)];
                }
            }
        }

        ret
    }

    impl_product!(
        MatrixView<T>, MatrixView<T>, (ref, ref),
        &MatrixView<T>, MatrixView<T>, (_, ref),
        MatrixView<T>, &MatrixView<T>, (ref, _),
        &MatrixView<T>, &MatrixView<T>, (_, _),
    );

    #[inline(always)]
    fn do_scalar_product<T>(lhs: &MatrixView<T>, rhs: &T) -> HeapMatrix<T>
    where
        T: Default + Copy + Num,
    {
        let mut this = lhs.to_heap();

        for k in this.values_mut() {
            *k = *k * *rhs;
        }

        this
    }

    impl_scalar!(
        MatrixView<T>, T, (ref, ref),
        &MatrixView<T>, T, (_, ref),
        MatrixView<T>, &T, (ref, _),
        &MatrixView<T>, &T, (_, _),
    );
}
