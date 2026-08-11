use crate::linear::basis::VectorBasis;
use crate::linear::heap_matrix::HeapMatrix;
use crate::linear::matrix::Matrix;
use crate::linear::vector::Vector;
use crate::mem::{FromOwned, Owned};
use alloc::{boxed::Box, vec};
use arrayvec::ArrayVec;
use core::mem;
use num_traits::{ConstOne, ConstZero, Float, Num, Zero};
use core::ops::Deref;

/// Represents a family of vectors with `N` coordinates.
#[repr(transparent)]
#[derive(Debug)]
pub struct VectorFamily<T, const N: usize>
where
    T: Default + Copy + Num,
{
    pub vectors: [Vector<T, N>],
}

impl<T, const N: usize> VectorFamily<T, N>
where
    T: Default + Copy + Num,
{
    /// Returns a vector family that contains no elements, neither zero.
    pub fn empty() -> Owned<Self, ArrayVec<Vector<T, N>, N>> {
        Owned::new(ArrayVec::new())
    }

    /// Returns a vector family that contains no elements, neither zero.
    pub fn empty_heap() -> Owned<Self, Box<[Vector<T, N>]>> {
        Owned::new(Box::from([]))
    }

    /// Returns a vector family that contains only the zero vector.
    pub fn only_zero() -> Owned<Self, ArrayVec<Vector<T, N>, N>> {
        let mut vec = ArrayVec::new();
        vec.push(Vector::zero());
        Owned::new(vec)
    }

    /// Returns a vector family that contains only the zero vector.
    pub fn only_zero_heap() -> Owned<Self, Box<[Vector<T, N>]>> {
        Owned::new(Box::new([Vector::zero()]))
    }

    /// Computes the standard basis for `F^n`. This cannot be constant because
    /// it uses the heap.
    ///
    /// Use [`const_standard_basis`](Self::const_standard_basis) if you can.
    pub fn standard_basis() -> Owned<Self, ArrayVec<Vector<T, N>, N>> {
        let matrix = Matrix::identity();

        Owned::new(ArrayVec::from(matrix.to_row_vectors()))
    }

    /// Computes the standard basis for `F^n`. This cannot be constant because
    /// it uses the heap.
    ///
    /// Use [`const_standard_basis`](Self::const_standard_basis) if you can.
    pub fn standard_basis_heap() -> Owned<Self, Box<[Vector<T, N>]>> {
        let mut vectors = Box::new_uninit_slice(N);

        for (i, vector) in vectors.iter_mut().enumerate() {
            let vector: &mut Vector<T, N> = vector.write(Vector::zero());
            vector.scalars[i] = T::one();
        }

        // SAFETY: all elements are initialized
        Owned::new(unsafe { vectors.assume_init() })
    }

    /// Computes the standard basis for `F^n`. This can be done at compile time.
    ///
    /// Only works if `T` is [`ConstOne`] and [`ConstZero`]. You need to use
    /// runtime variant [`standard_basis`](Self::standard_basis) elsewise.
    pub const fn const_standard_basis() -> Owned<Self, [Vector<T, N>; N]>
    where
        T: ConstOne + ConstZero,
    {
        // Identity is symmetric, so it's equivalent to to_vector()
        // but much faster.
        Owned::new(Matrix::<T, N>::const_identity().to_row_vectors())
    }

    /// Determines if `self` is a family of independent vectors.
    ///
    /// To check if it's a basis, use [`as_basis`](Self::as_basis).
    ///
    /// Uses Gaussian elimination.
    ///
    /// # Example
    ///
    /// ```
    /// # use cpge::linear::{vector::Vector, family::VectorFamily};
    /// let family: &VectorFamily<f64, 2> = (&[
    ///     Vector::from([1.0, 2.0]),
    ///     Vector::from([2.0, 4.0]),
    /// ][..]).into();
    /// assert!(!family.is_independent());
    ///
    /// let family: &VectorFamily<f64, 2> = (&[
    ///     Vector::from([1.0, 2.0]),
    ///     Vector::from([2.0, 2.0]),
    /// ][..]).into();
    /// assert!(family.is_independent());
    /// ```
    pub fn is_independent(&self) -> bool
    where
        T: Float,
    {
        // we need to do this check, elsewise vectors won't fit the matrix
        if self.vectors.len() > N {
            return false;
        }

        // We now do Gauss elimination and look for the rank.
        let mut vectors = self.vectors.to_vec();
        vectors.resize(N, Vector::zero());
        assert_eq!(vectors.len(), N);

        let array: &[_; N] = vectors.as_array().unwrap();
        let matrix = Matrix::from_vectors(array).gaussian_elimination();

        matrix.rank_float() == self.vectors.len()
    }

    /// Determines if `self` is a family of vectors that generators whole `T ^ N` space.
    ///
    /// To check if it's a basis, use [`as_basis`](Self::as_basis).
    ///
    /// Uses Gaussian elimination.
    pub fn is_generator(&self) -> bool
    where
        T: Float,
    {
        if self.vectors.len() < N {
            return false;
        }

        let matrix = HeapMatrix::from_vectors(&self.vectors[..])
            .gaussian_elimination();

        matrix.rank_float() == N
    }

    /// Tries to convert `self` as a vector basis. This checks if the family is independent and
    /// generator. Returns [`None`] if the family isn't a basis and [`Some`] if it is.
    ///
    /// Use `self.as_basis().is_some()` over `self.is_independent() && self.is_generator()`, it's
    /// faster.
    ///
    /// Uses Gaussian elimination.
    pub fn as_basis(&self) -> Option<VectorBasis<T, N>>
    where
        T: Float,
    {
        if self.vectors.len() != N {
            return None;
        }

        let array = self.vectors.as_array::<N>().expect("Vec wasn't resized to N");
        let matrix = Matrix::from_vectors(array);
        let echelon = matrix.gaussian_elimination();

        if echelon.rank_float() != N {
            return None;
        }

        let vectors = self.vectors
            .as_array().unwrap()
            .clone()
            .map(|x| x.scalars);

        // SAFETY: we just check it's a basis
        unsafe { Some(VectorBasis::new(vectors)) }
    }

    /// Returns a minimal independent vector family that is equivalent to `self`.
    ///
    /// The returned family may contain at most `n` vectors.
    pub fn minimal_independent(&self) -> Owned<Self, ArrayVec<Vector<T, N>, N>>
    where
        T: Float,
    {
        let mut iter = self.vectors.iter().cloned();
        let mut ret = if let Some(vec) = iter.next() {
            let mut ret = ArrayVec::new();
            ret.push(vec);
            ret
        } else { // self empty
            return Owned::new(ArrayVec::new());
        };

        for vec in iter {
            if ret.len() == N {
                break;
            }

            ret.push(vec);

            let family: &Self = ret.as_ref().into();
            let independent = family.is_independent();

            if !independent {
                // we remove the vector we just added because it was independent previously.
                let _ = ret.pop();
            }
        }

        Owned::new(ret)
    }

    /// Returns a minimal independent vector family that is equivalent to `self`.
    ///
    /// The returned family may contain at most `n` vectors.
    pub fn minimal_independent_heap(&self) -> Owned<Self, Box<[Vector<T, N>]>>
    where
        T: Float,
    {
        let mut iter = self.vectors.iter().cloned();
        let mut ret = if let Some(vec) = iter.next() {
            vec![vec]
        } else { // self empty
            return Self::empty_heap();
        };

        for vec in iter {
            if ret.len() == N {
                break;
            }

            ret.push(vec);

            let family: &Self = (&ret[..]).into();
            let independent = family.is_independent();

            if !independent {
                // we remove the vector we just added because it was independent previously.
                let _ = ret.pop();
            }
        }

        Owned::new(ret.into_boxed_slice())
    }

    #[inline]
    pub fn as_vectors(&self) -> &[Vector<T, N>] {
        &self.vectors
    }
}

impl<T, const N: usize> From<&[Vector<T, N>]> for &VectorFamily<T, N>
where
    T: Default + Copy + Num,
{
    fn from(value: &[Vector<T, N>]) -> Self {
        unsafe { mem::transmute(value) }
    }
}

impl<T, const N: usize> From<&mut [Vector<T, N>]> for &mut VectorFamily<T, N>
where
    T: Default + Copy + Num,
{
    fn from(value: &mut [Vector<T, N>]) -> Self {
        unsafe { mem::transmute(value) }
    }
}

impl<T, K, const N: usize> FromOwned<K> for VectorFamily<T, N>
where
    T: Default + Copy + Num,
    K: AsRef<[Vector<T, N>]> + AsMut<[Vector<T, N>]>,
{
    fn from_owned(owned: &K) -> &Self {
        owned.as_ref().into()
    }

    fn from_owned_mut(owned: &mut K) -> &mut Self {
        owned.as_mut().into()
    }
}

impl<T, const N: usize> Deref for VectorFamily<T, N>
where
    T: Default + Copy + Num,
{
    type Target = [Vector<T, N>];

    fn deref(&self) -> &Self::Target {
        &self.vectors
    }
}
