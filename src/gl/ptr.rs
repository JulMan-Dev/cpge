use core::fmt;
use core::ptr::NonNull;

/// An opaque pointer struct that can be sent and synced across threads. Unlike default raw pointers
/// that don't implement `Send` and `Sync`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct OpaqueInner(NonNull<()>);

unsafe impl Send for OpaqueInner {}
unsafe impl Sync for OpaqueInner {}

impl fmt::Debug for OpaqueInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}

impl OpaqueInner {
    #[inline]
    pub const fn new<T>(non_null: NonNull<T>) -> Self {
        Self(non_null.cast())
    }

    #[inline]
    pub const fn dangling() -> Self {
        Self(NonNull::dangling())
    }

    /// # Safety
    ///
    /// The caller must ensure that the pointer is non-null.
    #[inline]
    pub const unsafe fn new_unchecked<T>(ptr: *mut T) -> Self {
        Self::new(unsafe { NonNull::new_unchecked(ptr) })
    }

    #[inline]
    pub const fn from_ref<T>(ptr: &T) -> Self {
        Self::new(NonNull::from_ref(ptr))
    }

    #[inline]
    pub const fn from_mut<T>(ptr: &mut T) -> Self {
        Self::new(NonNull::from_mut(ptr))
    }

    /// # Safety
    ///
    /// The inner pointer must be non-null, well-aligned, and represent `T`.
    #[inline]
    pub const unsafe fn as_ref<'a, T>(self) -> &'a T {
        unsafe { self.0.cast().as_ref() }
    }

    /// # Safety
    ///
    /// The inner pointer must be castable a mutable reference of `T`, meaning it must represent a
    /// `T`, must be the unique reference to `T` and be non-null and well-aligned.
    #[inline]
    pub const unsafe fn as_mut<'a, T>(self) -> &'a mut T {
        unsafe { self.0.cast().as_mut() }
    }

    /// # Safety
    ///
    /// The pointer must represent a `T`. This will cast to `*mut T` and drop the object.
    #[inline]
    pub unsafe fn drop_in_place<T>(self) {
        unsafe { self.0.cast::<T>().drop_in_place() }
    }
}

/// This module is only supported on macOS, as it is the only OS that uses Objective-C.
#[cfg(target_os = "macos")]
mod objc {
    use crate::gl::ptr::OpaqueInner;
    use objc2::Message;
    use objc2::rc::Retained;

    impl OpaqueInner {
        #[inline]
        pub fn from_objc<T: Message>(ptr: Retained<T>) -> Self {
            unsafe { Self::new_unchecked(Retained::into_raw(ptr)) }
        }

        /// # Safety
        ///
        /// The pointer must represent a `Retained<T>`, be well-aligned and non-null.
        #[inline]
        pub unsafe fn into_objc<T: Message>(self) -> Retained<T> {
            unsafe { Retained::from_raw(self.0.cast().as_ptr()).unwrap() }
        }

        /// # Safety
        ///
        /// The pointer must represent a `Retained<T>`. This will cast to `*const Retained<T>` and
        /// drop the object.
        #[inline]
        pub unsafe fn objc_drop_in_place<T: Message>(self) {
            // into_objc makes the Retained object, and Rust RAII drops it before it return.
            unsafe { self.into_objc::<T>() };
        }
    }
}
