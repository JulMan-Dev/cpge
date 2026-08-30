use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use core::any::TypeId;
use core::array::from_fn;
use core::mem;
use core::ops::{Add, Deref};
use num_traits::Num;

#[derive(Debug)]
pub struct Operation<T>(Box<FunctionOperation<T>>)
where
    T: Copy;

impl<T> Operation<T>
where
    T: Copy,
{
    pub fn new_singleton(value: T) -> Self {
        Self(Box::new(FunctionOperation::Singleton(value)))
    }

    fn from_raw_operation(operation: FunctionOperation<T>) -> Self {
        Self(Box::new(operation))
    }
}

impl<T: Copy> Clone for Operation<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Copy> Deref for Operation<T> {
    type Target = FunctionOperation<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

type __MacroOperation<T, const N: usize> = Operation<T>;

pub struct Function<T, const N: usize = 1>
where
    T: Copy,
{
    body: Operation<T>,
}

impl<T, const N: usize> Function<T, N>
where
    T: Copy,
{
    pub fn from_lazy<K, F>(f: F) -> Self
    where
        K: Copy,
        F: LazyFunction<K, N, Output = T>,
    {
        let unknowns = from_fn(|i|
            Operation::from_raw_operation(FunctionOperation::Unknown(i)));

        Self { body: f.invoke(unknowns) }
    }
}

pub trait LazyFunction<T, const N: usize>
where
    T: Copy,
{
    type Output: Copy;

    fn invoke(&self, values: [Operation<T>; N]) -> Operation<Self::Output>;
}

impl<T> LazyFunction<T, 0> for FunctionOperation<T>
where
    T: Default + Copy + Num,
{
    type Output = T;

    fn invoke(&self, _: [Operation<T>; 0]) -> Operation<Self::Output> {
        Operation(Box::new(self.clone()))
    }
}

impl<T> LazyFunction<T, 0> for Operation<T>
where
    T: Default + Copy + Num,
{
    type Output = T;

    fn invoke(&self, _: [Operation<T>; 0]) -> Operation<Self::Output> {
        self.clone()
    }
}

impl<T, const N: usize> LazyFunction<T, N> for Function<T, N>
where
    T: Copy + 'static,
{
    type Output = T;

    fn invoke(&self, values: [Operation<T>; N]) -> Operation<Self::Output> {
        self.body.evaluate(get_vtable(), &values)
    }
}

macro_rules! impl_functions {
    () => {};
    ([0], $($tt:tt)*) => {
        impl<T, F, U> LazyFunction<T, 0> for F
        where
            T: Copy,
            U: Copy,
            F: Fn() -> Operation<U>,
        {
            type Output = U;

            fn invoke(&self, _: [Operation<T>; 0]) -> Operation<Self::Output> {
                self()
            }
        }

        impl_functions! { $($tt)* }
    };
    ([$($n:literal),+; $count:literal], $($tt:tt)*) => {
        impl<T, F, U> LazyFunction<T, $count> for F
        where
            T: Copy,
            U: Copy,
            F: Fn($(__MacroOperation<T, $n>),+) -> Operation<U>,
        {
            type Output = U;

            fn invoke(&self, values: [Operation<T>; $count]) -> Operation<Self::Output> {
                self($(values[$n].clone()),*)
            }
        }

        impl_functions! { $($tt)* }
    };
}

impl_functions! {
    [0],
    [0; 1],
    [0, 1; 2],
    [0, 1, 2; 3],
    // TODO: support more function arity
}

#[derive(Clone, Debug)]
pub enum FunctionOperation<T>
where
    T: Copy,
{
    Unknown(usize),
    Singleton(T),
    Add(Operation<T>, Operation<T>),
    Sub(Operation<T>, Operation<T>),
    Mul(Operation<T>, Operation<T>),
    Div(Operation<T>, Operation<T>),
}

#[derive(Default)]
#[repr(C)]
struct VTable<T> {
    add: Option<fn(T, T) -> T>,
    sub: Option<fn(T, T) -> T>,
    mul: Option<fn(T, T) -> T>,
    div: Option<fn(T, T) -> T>,
}

impl<T> FunctionOperation<T>
where
    T: Copy,
{
    pub const fn unwrap_value(&self) -> Option<T> {
        match self {
            Self::Singleton(v) => Some(*v),
            _ => None,
        }
    }

    pub(self) fn evaluate(&self, vtable: &VTable<T>, map: &[Operation<T>]) -> Operation<T> {
        macro_rules! pat_2 {
            ($o:ident @ $fn:ident, $a:ident, $b:ident) => {{
                let a = $a.evaluate(vtable, map);
                let b = $b.evaluate(vtable, map);

                Operation::from_raw_operation(if let Some((a, b)) = a.unwrap_value().zip(b.unwrap_value()) {
                    let v = vtable.add.unwrap()(a, b);
                    FunctionOperation::Singleton(v)
                } else {
                    $o.clone()
                })
            }};
        }

        match self {
            Self::Unknown(i) => panic!("found unknown {i} operation"),
            v @ Self::Singleton(_) => Operation::from_raw_operation(v.clone()),
            o @ Self::Add(a, b) => pat_2!(o @ add, a, b),
            o @ Self::Sub(a, b) => pat_2!(o @ sub, a, b),
            o @ Self::Mul(a, b) => pat_2!(o @ mul, a, b),
            o @ Self::Div(a, b) => pat_2!(o @ div, a, b),
        }
    }
}

static mut VTABLES: BTreeMap<TypeId, VTable<()>> = BTreeMap::new();

fn get_vtable<T>() -> &'static VTable<T> {
    let key = TypeId::of::<T>();

    // SAFETY: we acquire a const reference, lives 'static
    #[allow(static_mut_refs)]
    let vtable: &VTable<()> = unsafe {
        VTABLES.entry(key).or_default()
    };

    // SAFETY: VTable<()> and VTable<T> shares the same layout.
    unsafe { mem::transmute(vtable) }
}

unsafe fn get_mut_vtable<T>() -> &'static mut VTable<T> {
    let key = TypeId::of::<T>();

    // SAFETY: the caller should not leak the references out of its scope
    #[allow(static_mut_refs)]
    let vtable = unsafe {
        VTABLES.entry(key).or_default()
    };

    // SAFETY: VTable<()> and VTable<T> shares the same layout.
    unsafe { mem::transmute(vtable) }
}

impl<T> Add<Operation<T>> for Operation<T>
where
    T: Copy + Add<T, Output = T> + 'static,
{
    type Output = Operation<T>;

    fn add(self, rhs: Operation<T>) -> Self::Output {
        // SAFETY: the vtable pointer is dropped at the end of the unsafe block
        unsafe {
            get_mut_vtable::<T>().add.get_or_insert(T::add);
        }

        Operation::from_raw_operation(FunctionOperation::Add(self, rhs))
    }
}
