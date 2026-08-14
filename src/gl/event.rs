use crate::gl::ptr::OpaqueInner;
use alloc::string::String;
use std::sync::{Arc, RwLock};
use tokio::sync::{oneshot, SetOnce};

pub trait BackendEvent {
    fn timestamp(&self) -> u64;
}

pub(super) mod internal {
    use crate::gl::event::ApplicationEvent;
    use std::prelude::rust_2015::Vec;
    use tokio::runtime::Handle;

    pub trait ApplicationEventSource {
        fn poll_events(&self, events: &mut Vec<ApplicationEvent>);

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
