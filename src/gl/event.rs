use crate::gl::ptr::OpaqueInner;
use alloc::string::String;
use std::sync::Arc;
use tokio::sync::SetOnce;

pub trait BackendEvent {
    fn timestamp(&self) -> u64;
}

pub(super) mod internal {
    use crate::gl::event::ApplicationEvent;
    use alloc::vec::Vec;
    use tokio::runtime::Handle;

    pub trait ApplicationEventSource {
        /// The implementation that polls application events in a platform-independent way.
        ///
        /// The backend implementation should be not blocking. This should return immediately if they
        /// are not events pending.
        ///
        /// The event loop calls this method to poll for new events. It is guaranteed to be invoked
        /// on the main system thread.
        fn poll_events(&self, events: &mut Vec<ApplicationEvent>);

        /// The Tokio runtime handle. This may be used to spawn asynchronous tasks.
        ///
        /// Note: if your task needs the main thread, you should use
        /// [`Context::spawn_on_main`](super::super::context::Context::spawn_on_main).
        fn async_handle(&self) -> &Handle;
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum MouseButton {
    Left,
    Right,
    Other,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum MouseAction {
    Down,
    Up,
    Dragged,
    Moved,
    Entered,
    Exited,
}

// For each event, the Clone implementation should be platform-dependent. The inner pointer should
// be cloned using OS-specific methods.

#[derive(Debug)]
pub struct MouseEvent {
    pub button: MouseButton,
    pub location: (f64, f64),
    pub delta_x: f64,
    pub delta_y: f64,
    pub action: MouseAction,
    pub(super) inner: OpaqueInner,
}

#[derive(Debug)]
pub struct KeyEvent {
    pub key: String,
    pub caps: bool,
    pub shift: bool,
    pub control: bool,
    pub alt: bool,
    pub meta: bool,
    pub function: bool,
    pub(super) inner: OpaqueInner,
}

#[derive(Debug)]
pub struct WheelEvent {
    pub delta_x: f64,
    pub delta_y: f64,
    pub inverted: bool,
    pub(super) inner: OpaqueInner,
}

#[derive(Debug)]
pub struct PeriodicEvent {
    pub(super) inner: OpaqueInner,
}

#[derive(Debug)]
pub struct ShouldTerminateEvent {
    pub(super) notifier: Arc<SetOnce<bool>>, // used to drop daemon thread if platform requires it
    pub(super) inner: OpaqueInner,
}

#[derive(Debug, Clone)]
pub enum ApplicationEvent {
    Mouse(MouseEvent),
    Key(KeyEvent),
    Wheel(WheelEvent),
    Periodic(PeriodicEvent),
    ShouldTerminate(ShouldTerminateEvent),
    WindowClosed,
}
