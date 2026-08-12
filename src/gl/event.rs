use alloc::string::String;
use std::marker::PhantomData;
use std::ptr::NonNull;
use tokio::sync::broadcast;
use crate::gl::ptr::OpaqueInner;
use crate::gl::subscribe_events;

pub(super) mod internal {
    use crate::gl::event::ApplicationEvent;
    use std::io;
    use std::prelude::rust_2015::Vec;

    pub trait BackendEvent {
        fn timestamp(&self) -> u64;
    }

    pub trait ApplicationEventSource {
        fn poll_events(&self, events: &mut Vec<ApplicationEvent>);
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

#[derive(Debug, Clone)]
pub struct MouseEvent {
    pub button: MouseButton,
    pub location: (f64, f64),
    pub delta_x: f64,
    pub delta_y: f64,
    pub action: MouseAction,
    pub(super) inner: OpaqueInner,
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct WheelEvent {
    pub delta_x: f64,
    pub delta_y: f64,
    pub inverted: bool,
    pub(super) inner: OpaqueInner,
}

#[derive(Debug, Clone)]
pub struct PeriodicEvent {
    pub(super) inner: OpaqueInner,
}

#[derive(Debug, Clone)]
pub struct ShouldTerminateEvent {
    pub(super) replied: bool, // used to reply on drop if not done
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

pub struct Events {
    receiver: broadcast::Receiver<ApplicationEvent>,
}

impl Events {
    pub fn context() -> Self {
        Self {
            receiver: subscribe_events(),
        }
    }

    pub async fn poll(&mut self) -> ApplicationEvent {
        self.receiver.recv().await.unwrap()
    }
}
