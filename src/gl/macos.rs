//! macOS window logic.

use alloc::string::ToString;
use alloc::borrow::Cow;
use alloc::string::String;
use std::io;
use alloc::vec::Vec;
use std::marker::PhantomData;
use std::ptr::NonNull;
use crate::gl::{Data, GL, init_vulkan};
use std::sync::{Arc, OnceLock};
use objc2::{class, msg_send, MainThreadMarker};
use objc2::ffi::nil;
use objc2::rc::Retained;
use objc2::runtime::NSObject;
use objc2_app_kit::{NSApplication, NSEvent, NSEventMask, NSEventModifierFlags, NSEventTrackingRunLoopMode, NSEventType};
use objc2_foundation::{NSDate, NSProcessInfo};
use crate::gl::event::{internal, MouseEvent, KeyEvent, WheelEvent, ApplicationEvent, MouseButton, MouseAction, ShouldTerminateEvent};
use crate::gl::event::ApplicationEvent::{Key, Mouse};
use crate::gl::ptr::OpaqueInner;

#[link(name = "cpge-native")]
unsafe extern "C-unwind" {
    fn cpge_init_application(width: isize, height: isize);
}

static GLOBAL_GL: OnceLock<GL> = OnceLock::new();

/// Initializes the macOS window and makes a poller for Tokio runtime.
pub fn start_application() -> MacOsPoller {
    GLOBAL_GL.get_or_init(|| GL::new().unwrap());

    unsafe {
        cpge_init_application(1280, 720);
        let marker = MainThreadMarker::new_unchecked();
        let application = NSApplication::sharedApplication(marker);
        application.finishLaunching();
    }

    MacOsPoller::new()
}

/// This is called by the Swift mainloop when the view is ready.
#[unsafe(export_name = "cpge_spawn_vulkan")]
extern "C-unwind" fn spawn_vulkan(layer: *mut (), data: *const Data) {
    let gl = GLOBAL_GL.get().expect("illegal call to cpge_spawn_vulkan");
    let instance = gl.instance().unwrap();
    let surface = instance.macos_surface(layer).unwrap();

    // SAFETY: the pointer was written by make_vulkan_data, so it is valid
    let data = unsafe { Arc::from_raw(data) };

    init_vulkan(instance, surface, data)
}

impl internal::BackendEvent for MouseEvent {
    fn timestamp(&self) -> u64 {
        let event: &NSEvent = unsafe { self.inner.as_ref() };

        let f = NSDate::new().timeIntervalSince1970() +
            event.timestamp() - NSProcessInfo::processInfo().systemUptime();
        f as u64
    }
}

impl internal::BackendEvent for KeyEvent {
    fn timestamp(&self) -> u64 {
        let event: &NSEvent = unsafe { self.inner.as_ref() };

        let f = NSDate::new().timeIntervalSince1970() +
            event.timestamp() - NSProcessInfo::processInfo().systemUptime();
        f as u64
    }
}

impl internal::BackendEvent for WheelEvent {
    fn timestamp(&self) -> u64 {
        let event: &NSEvent = unsafe { self.inner.as_ref() };

        let f = NSDate::new().timeIntervalSince1970() +
            event.timestamp() - NSProcessInfo::processInfo().systemUptime();
        f as u64
    }
}

pub struct MacOsPoller(Retained<NSApplication>);

impl MacOsPoller {
    pub(super) fn new() -> Self {
        // SAFETY: the caller must ensure that this is called on the main thread
        let marker = unsafe { MainThreadMarker::new_unchecked() };
        MacOsPoller(NSApplication::sharedApplication(marker))
    }

    pub fn poll(&self) -> Option<Retained<NSEvent>> {
        let event = self.0.nextEventMatchingMask_untilDate_inMode_dequeue(
            NSEventMask::Any,
            None,
            unsafe { NSEventTrackingRunLoopMode },
            true,
        )?;

        // backpressure to ensure AppKit also handles events
        self.0.sendEvent(&event);

        Some(event)
    }
}

impl internal::ApplicationEventSource for MacOsPoller {
    fn poll_events(&self, events: &mut Vec<ApplicationEvent>) {
        while let Some(event) = self.poll() {
            const LEFT_MOUSE_DOWN: usize = NSEventType::LeftMouseDown.0;
            const MOUSE_EXITED: usize = NSEventType::MouseExited.0;

            let event = match event.r#type() {
                ty @ (NSEventType(LEFT_MOUSE_DOWN..=MOUSE_EXITED) |
                NSEventType::OtherMouseDown |
                NSEventType::OtherMouseUp |
                NSEventType::OtherMouseDragged) => {
                    Mouse(MouseEvent {
                        button: match ty {
                            NSEventType::LeftMouseDown | NSEventType::LeftMouseUp | NSEventType::LeftMouseDragged => MouseButton::Left,
                            NSEventType::RightMouseDown | NSEventType::RightMouseUp | NSEventType::RightMouseDragged => MouseButton::Right,
                            _ => MouseButton::Other,
                        },
                        location: {
                            let location = event.locationInWindow();
                            (location.x, location.y)
                        },
                        delta_x: event.deltaX(),
                        delta_y: event.deltaY(),
                        action: match ty {
                            NSEventType::LeftMouseDown | NSEventType::RightMouseDown | NSEventType::OtherMouseDown => MouseAction::Down,
                            NSEventType::LeftMouseUp | NSEventType::RightMouseUp | NSEventType::OtherMouseUp => MouseAction::Up,
                            NSEventType::LeftMouseDragged | NSEventType::RightMouseDragged | NSEventType::OtherMouseDragged => MouseAction::Dragged,
                            NSEventType::MouseEntered => MouseAction::Entered,
                            NSEventType::MouseMoved => MouseAction::Moved,
                            NSEventType::MouseExited => MouseAction::Exited,
                            _ => unreachable!()
                        },
                        inner: OpaqueInner::from_objc(event),
                    })
                },
                NSEventType::KeyDown => {
                    let modifiers = event.modifierFlags();

                    Key(KeyEvent {
                        key: {
                            let chars = event.charactersIgnoringModifiers().unwrap();
                            let mut chars = chars.to_string();
                            chars.make_ascii_lowercase();
                            chars
                        },
                        caps: modifiers.contains(NSEventModifierFlags::CapsLock),
                        shift: modifiers.contains(NSEventModifierFlags::Shift),
                        control: modifiers.contains(NSEventModifierFlags::Control),
                        alt: modifiers.contains(NSEventModifierFlags::Option),
                        meta: modifiers.contains(NSEventModifierFlags::Command),
                        function: modifiers.contains(NSEventModifierFlags::Function),
                        inner: OpaqueInner::from_objc(event),
                    })
                }
                _ => continue,
            };

            events.push(event);
        }
    }
}

impl ShouldTerminateEvent {
    pub fn reply_not_ready(&mut self) {
        if self.replied {
            return;
        }

        self.replied = true;
        todo!("invoke swift to actually reply");
    }

    pub fn reply_ready(&mut self) {
        if self.replied {
            return;
        }

        self.replied = true;
        todo!("invoke swift to actually reply");
    }
}
