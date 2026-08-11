use crate::linear::view::MatrixView;
use num_traits::Num;

pub struct ConvolutionRows<'a, T>
where
    T: Default + Copy + Num,
{
    input: &'a MatrixView<T>,
    kernel: &'a MatrixView<T>,
    pos: usize,
}

impl<'a, T> ConvolutionRows<'a, T>
where
    T: Default + Copy + Num,
{
    pub(crate) const fn new(input: &'a MatrixView<T>, kernel: &'a MatrixView<T>) -> Self {
        Self { input, kernel, pos: 0 }
    }
}

impl<'a, T> Iterator for ConvolutionRows<'a, T>
where
    T: Default + Copy + Num,
{
    type Item = ConvolutionRow<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.size_hint().0 != 0 {
            self.pos += 1;

            Some(ConvolutionRow {
                input: self.input,
                kernel: self.kernel,
                row: self.pos - 1,
                pos: 0,
            })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let input_rows = self.input.count_rows();
        let kernel_rows = self.kernel.count_rows();
        let length = input_rows - kernel_rows + 1;
        let remaining = length - self.pos;
        (remaining, Some(remaining))
    }
}

pub struct ConvolutionRow<'a, T>
where
    T: Default + Copy + Num,
{
    input: &'a MatrixView<T>,
    kernel: &'a MatrixView<T>,
    row: usize,
    pos: usize,
}

impl<'a, T> Iterator for ConvolutionRow<'a, T>
where
    T: Default + Copy + Num,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let goggles = self.kernel
            .exact_goggles()
            .shift_by((self.row, self.pos));

        if self.input.exact_goggles().can_compose_with(goggles) {
            let value = Iterator::zip(
                self.input.view(goggles).values(),
                self.kernel.values()
            ).fold(T::zero(), |acc, (&left, &right)| acc + left * right);

            self.pos += 1;

            Some(value)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let input_cols = self.input.count_cols();
        let kernel_cols = self.kernel.count_cols();
        let length = input_cols - kernel_cols + 1;
        let remaining = length - self.pos;
        (remaining, Some(remaining))
    }
}
