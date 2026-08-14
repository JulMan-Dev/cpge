/// A cross-product iterator.
#[derive(Clone, Debug)]
pub struct Cross<U: Iterator, V: Iterator + Clone>
where
    U::Item: Clone,
{
    first: U,
    current: Option<U::Item>,
    second: V,
    second_copy: V,
}

impl<U: Iterator + Sized, V: Iterator + Clone + Sized> Cross<U, V>
where
    U::Item: Clone,
    V::Item: Clone,
{
    pub fn new(first: U, second: V) -> Self {
        Self {
            first,
            current: None,
            second_copy: second.clone(),
            second,
        }
    }
}

impl<U: Iterator, V: Iterator + Clone> Iterator for Cross<U, V>
where
    U::Item: Clone,
    V::Item: Clone,
{
    type Item = (U::Item, V::Item);

    fn next(&mut self) -> Option<Self::Item> {
        // we acquire the second first to see if we need to acquire a newer first
        let (second, need_first) = match self.second_copy.next() {
            Some(a) => (a, false), // yes!
            None => {
                // cloning again from original
                self.second_copy = self.second.clone();

                // we should have a next now, if not we abort
                (self.second_copy.next()?, true)
            }
        };

        let first = if need_first || self.current.is_none() {
            // we abort if no next
            self.current.insert(self.first.next()?).clone()
        } else {
            self.current.clone().unwrap()
        };

        Some((first, second))
    }
}
