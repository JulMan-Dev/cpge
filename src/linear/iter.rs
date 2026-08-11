use crate::linear::view::Goggles;

pub mod heap {
    //#region MatrixCellIter

    use alloc::boxed::Box;
    use core::iter::FusedIterator;
    use num_traits::Num;
    use crate::linear::MatrixView;

    #[derive(Clone)]
    pub struct MatrixCellIter<'a, T>
    where
        T: Default + Copy + Num,
    {
        matrix: &'a MatrixView<T>,
        index: usize,
        current: usize, // current for Iterator
        current_rev: usize, // current for DoubleEndedIterator, exclusive
        mode: IterMode,
    }

    #[derive(Clone)]
    enum IterMode {
        Row,
        Col,
    }

    impl<'a, T> MatrixCellIter<'a, T>
    where
        T: Default + Copy + Num,
    {
        pub(crate) const fn new_row(matrix: &'a MatrixView<T>, index: usize) -> Self {
            let current_rev = if index < matrix.count_rows() { matrix.count_cols() } else { 0 };

            Self { matrix, index, current: 0, current_rev, mode: IterMode::Row }
        }

        pub(crate) const fn new_col(matrix: &'a MatrixView<T>, index: usize) -> Self {
            let current_rev = if index < matrix.count_cols() { matrix.count_rows() } else { 0 };

            Self { matrix, index, current: 0, current_rev, mode: IterMode::Col }
        }
    }

    impl<'a, T> Iterator for MatrixCellIter<'a, T>
    where
        T: Default + Copy + Num,
    {
        type Item = &'a T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current >= self.current_rev {
                return None;
            }

            let next = match self.mode {
                IterMode::Row => self.matrix.get((self.index, self.current)),
                IterMode::Col => self.matrix.get((self.current, self.index)),
            };
            self.current += 1;
            next
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let k = self.current_rev - self.current;
            (k, Some(k))
        }
    }

    impl<'a, T> ExactSizeIterator for MatrixCellIter<'a, T>
    where
        T: Default + Copy + Num,
    {}

    impl<'a, T> DoubleEndedIterator for MatrixCellIter<'a, T>
    where
        T: Default + Copy + Num,
    {
        fn next_back(&mut self) -> Option<Self::Item> {
            if self.current >= self.current_rev {
                return None;
            }

            self.current_rev -= 1;
            let next = match self.mode {
                IterMode::Row => self.matrix.get((self.index, self.current_rev)),
                IterMode::Col => self.matrix.get((self.current_rev, self.index)),
            };

            next
        }
    }

    impl<'a, T> FusedIterator for MatrixCellIter<'a, T>
    where
        T: Default + Copy + Num,
    {}

    //#endregion

    //#region MatrixCellIterMut

    pub struct MatrixCellIterMut<'a, T>
    where
        T: Default + Copy + Num,
    {
        matrix: &'a mut MatrixView<T>,
        index: usize,
        current: usize, // current for Iterator
        current_rev: usize, // current for DoubleEndedIterator, exclusive
        mode: IterMode,
    }

    impl<'a, T> MatrixCellIterMut<'a, T>
    where
        T: Default + Copy + Num,
    {
        pub(crate) const fn new_row(matrix: &'a mut MatrixView<T>, index: usize) -> Self {
            let current_rev = if index < matrix.count_rows() { matrix.count_cols() } else { 0 };

            Self { matrix, index, current: 0, current_rev, mode: IterMode::Row }
        }

        pub(crate) const fn new_col(matrix: &'a mut MatrixView<T>, index: usize) -> Self {
            let current_rev = if index < matrix.count_cols() { matrix.count_rows() } else { 0 };

            Self { matrix, index, current: 0, current_rev, mode: IterMode::Col }
        }
    }

    impl<'a, T> Iterator for MatrixCellIterMut<'a, T>
    where
        T: Default + Copy + Num,
    {
        type Item = &'a mut T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current >= self.current_rev {
                return None;
            }

            let current = self.current;
            self.current += 1;
            let matrix: &mut _ = self.matrix;

            let next = match self.mode {
                IterMode::Row if current < matrix.count_cols() => &mut matrix[(self.index, current)],
                IterMode::Col if current < matrix.count_rows() => &mut matrix[(current, self.index)],
                _ => return None,
            };

            // SAFETY: * next was a mut reference just before ;
            //         * next() may not produce a mut ref to same memory.
            unsafe { (next as *mut T).as_mut() }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let k = self.current_rev - self.current;
            (k, Some(k))
        }
    }

    impl<'a, T> ExactSizeIterator for MatrixCellIterMut<'a, T>
    where
        T: Default + Copy + Num,
    {}

    impl<'a, T> DoubleEndedIterator for MatrixCellIterMut<'a, T>
    where
        T: Default + Copy + Num,
    {
        fn next_back(&mut self) -> Option<Self::Item> {
            if self.current >= self.current_rev {
                return None;
            }

            self.current_rev -= 1;
            let current = self.current_rev;
            let matrix: &mut _ = self.matrix;

            let next = match self.mode {
                IterMode::Row if current < matrix.count_cols() => &mut matrix[(self.index, current)],
                IterMode::Col if current < matrix.count_rows() => &mut matrix[(current, self.index)],
                _ => return None,
            };

            // SAFETY: * next was a mut reference just before ;
            //         * next_back() may not produce a mut ref to same memory
            unsafe { (next as *mut T).as_mut() }
        }
    }

    impl<'a, T> FusedIterator for MatrixCellIterMut<'a, T>
    where
        T: Default + Copy + Num,
    {}

    //#endregion

    //#region MatrixRows

    #[derive(Clone)]
    pub struct MatrixRows<'a, T>
    where
        T: Default + Copy + Num,
    {
        matrix: &'a MatrixView<T>,
        current: usize,
        current_rev: usize, // exclusive
    }

    impl<'a, T> MatrixRows<'a, T>
    where
        T: Default + Copy + Num,
    {
        pub(crate) const fn new(matrix: &'a MatrixView<T>) -> Self {
            Self { matrix, current: 0, current_rev: matrix.count_rows() }
        }
    }

    impl<'a, T> Iterator for MatrixRows<'a, T>
    where
        T: Default + Copy + Num,
    {
        type Item = Box<[&'a T]>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current >= self.current_rev {
                return None;
            }

            let next = self.matrix.get_row_array(self.current);
            self.current += 1;

            next
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let k = self.current_rev - self.current;
            (k, Some(k))
        }
    }

    impl<'a, T> ExactSizeIterator for MatrixRows<'a, T>
    where
        T: Default + Copy + Num,
    {}

    impl<'a, T> DoubleEndedIterator for MatrixRows<'a, T>
    where
        T: Default + Copy + Num,
    {
        fn next_back(&mut self) -> Option<Self::Item> {
            if self.current >= self.current_rev {
                return None;
            }

            self.current_rev -= 1;
            self.matrix.get_row_array(self.current_rev)
        }
    }

    impl<'a, T> FusedIterator for MatrixRows<'a, T>
    where
        T: Default + Copy + Num,
    {}

    //#endregion

    //#region MatrixRowsMut

    pub struct MatrixRowsMut<'a, T>
    where
        T: Default + Copy + Num,
    {
        matrix: &'a mut MatrixView<T>,
        current: usize,
        current_rev: usize, // exclusive
    }

    impl<'a, T> MatrixRowsMut<'a, T>
    where
        T: Default + Copy + Num,
    {
        pub(crate) const fn new(matrix: &'a mut MatrixView<T>) -> Self {
            Self {
                current_rev: matrix.count_rows(),
                matrix,
                current: 0,
            }
        }
    }

    impl<'a, T> Iterator for MatrixRowsMut<'a, T>
    where
        T: Default + Copy + Num,
    {
        type Item = Box<[&'a mut T]>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current >= self.current_rev {
                return None;
            }

            // SAFETY: we use this reference to get a disjoint array from previous iterations
            let matrix: &mut MatrixView<T> = unsafe { &mut *(self.matrix as *mut _) };
            let next = matrix.get_mut_row_array(self.current);
            self.current += 1;
            next
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let k = self.current_rev - self.current;
            (k, Some(k))
        }
    }

    impl<'a, T> ExactSizeIterator for MatrixRowsMut<'a, T>
    where
        T: Default + Copy + Num,
    {}

    impl<'a, T> DoubleEndedIterator for MatrixRowsMut<'a, T>
    where
        T: Default + Copy + Num,
    {
        fn next_back(&mut self) -> Option<Self::Item> {
            if self.current >= self.current_rev {
                return None;
            }

            self.current_rev -= 1;
            // SAFETY: we use this reference to get a disjoint array from previous iterations
            let matrix: &mut MatrixView<T> = unsafe { &mut *(self.matrix as *mut _) };
            matrix.get_mut_row_array(self.current_rev)
        }
    }

    impl<'a, T> FusedIterator for MatrixRowsMut<'a, T>
    where
        T: Default + Copy + Num,
    {}

    //#endregion

    //#region MatrixCols

    #[derive(Clone)]
    pub struct MatrixCols<'a, T>
    where
        T: Default + Copy + Num,
    {
        matrix: &'a MatrixView<T>,
        current: usize,
        current_rev: usize, // exclusive
    }

    impl<'a, T> MatrixCols<'a, T>
    where
        T: Default + Copy + Num,
    {
        pub(crate) const fn new(matrix: &'a MatrixView<T>) -> Self {
            Self { matrix, current: 0, current_rev: matrix.count_cols() }
        }
    }

    impl<'a, T> Iterator for MatrixCols<'a, T>
    where
        T: Default + Copy + Num,
    {
        type Item = Box<[&'a T]>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current >= self.current_rev {
                return None;
            }

            let next = self.matrix.get_col_array(self.current);
            self.current += 1;

            next
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let k = self.current_rev - self.current;
            (k, Some(k))
        }
    }

    impl<'a, T> ExactSizeIterator for MatrixCols<'a, T>
    where
        T: Default + Copy + Num,
    {}

    impl<'a, T> DoubleEndedIterator for MatrixCols<'a, T>
    where
        T: Default + Copy + Num,
    {
        fn next_back(&mut self) -> Option<Self::Item> {
            if self.current >= self.current_rev {
                return None;
            }

            self.current_rev -= 1;
            self.matrix.get_col_array(self.current_rev)
        }
    }

    impl<'a, T> FusedIterator for MatrixCols<'a, T>
    where
        T: Default + Copy + Num,
    {}

    //#endregion

    //#region MatrixColsMut

    pub struct MatrixColsMut<'a, T>
    where
        T: Default + Copy + Num,
    {
        matrix: &'a mut MatrixView<T>,
        current: usize,
        current_rev: usize, // exclusive
    }

    impl<'a, T> MatrixColsMut<'a, T>
    where
        T: Default + Copy + Num,
    {
        pub(crate) const fn new(matrix: &'a mut MatrixView<T>) -> Self {
            Self {
                current_rev: matrix.count_cols(),
                matrix,
                current: 0,
            }
        }
    }

    impl<'a, T> Iterator for MatrixColsMut<'a, T>
    where
        T: Default + Copy + Num,
    {
        type Item = Box<[&'a mut T]>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current >= self.current_rev {
                return None;
            }

            // SAFETY: we use this reference to get a disjoint array from previous iterations
            let matrix: &mut MatrixView<T> = unsafe { &mut *(self.matrix as *mut _) };
            let next = matrix.get_mut_col_array(self.current);
            self.current += 1;

            next
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let k = self.current_rev - self.current;
            (k, Some(k))
        }
    }

    impl<'a, T> ExactSizeIterator for MatrixColsMut<'a, T>
    where
        T: Default + Copy + Num,
    {
    }

    impl<'a, T> DoubleEndedIterator for MatrixColsMut<'a, T>
    where
        T: Default + Copy + Num,
    {
        fn next_back(&mut self) -> Option<Self::Item> {
            if self.current >= self.current_rev {
                return None;
            }

            self.current_rev -= 1;
            // SAFETY: we use this reference to get a disjoint array from previous iterations
            let matrix: &mut MatrixView<T> = unsafe { &mut *(self.matrix as *mut _) };
            matrix.get_mut_col_array(self.current_rev)
        }
    }

    impl<'a, T> FusedIterator for MatrixColsMut<'a, T>
    where
        T: Default + Copy + Num,
    {
    }

    //#endregion
}

pub struct Positions<I> {
    iter: I,
    goggles: Goggles,
    cur: Option<(usize, usize)>,
}

impl<I: Iterator> Positions<I> {
    pub const fn new(iter: I, goggles: Goggles) -> Self {
        Self::new_at(iter, goggles, (0, 0))
    }

    pub const fn new_at(iter: I, goggles: Goggles, cur: (usize, usize)) -> Self {
        Self { iter, goggles, cur: Some(cur) }
    }
}

impl<I: Iterator> Iterator for Positions<I> {
    type Item = ((usize, usize), I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(cur) = self.cur {
            let next = (cur, self.iter.next()?);
            self.cur = self.goggles.next(cur);
            Some(next)
        } else {
            None
        }
    }
}
