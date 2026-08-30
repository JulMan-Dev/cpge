//! macOS window logic.

use crate::gl::context::PlatformContext;
use crate::gl::event::ApplicationEvent::{Key, Mouse};
use crate::gl::event::{internal, ApplicationEvent, BackendEvent, KeyEvent, MouseAction, MouseButton, MouseEvent, PeriodicEvent, ShouldTerminateEvent, WheelEvent, WillTerminateEvent};
use crate::gl::ptr::OpaqueInner;
use crate::gl::{Data, GL, context, init_vulkan};
use alloc::string::ToString;
use alloc::vec::Vec;
use objc2::MainThreadMarker;
use objc2::rc::Retained;
use objc2_app_kit::{NSApplication, NSEvent, NSEventMask, NSEventModifierFlags, NSEventType, NSModalPanelRunLoopMode};
use objc2_foundation::{NSDate, NSDefaultRunLoopMode, NSProcessInfo};
use std::sync::{Arc, OnceLock, RwLock};
use std::{mem, thread};
use std::iter::once;
use std::sync::atomic::AtomicUsize;
use tokio::io::AsyncWriteExt;
use tokio::runtime::Handle;
use tokio::sync::broadcast::Receiver;
use tokio::sync::{broadcast, mpsc, SetOnce};
use tokio::task;

#[link(name = "cpge-native")]
unsafe extern "C-unwind" {
    fn cpge_init_application(width: isize, height: isize);
}

static GLOBAL_GL: OnceLock<GL> = OnceLock::new();

/// Initializes the macOS window and makes a poller for Tokio runtime.
pub fn start_application(tx: broadcast::Sender<ApplicationEvent>, handle: Handle) -> MacOsPoller {
    GLOBAL_GL.get_or_init(|| GL::new().unwrap());

    unsafe {
        cpge_init_application(1280, 720);
        let marker = MainThreadMarker::new_unchecked();
        let application = NSApplication::sharedApplication(marker);
        application.finishLaunching();
    }

    MacOsPoller::new(tx, handle)
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

impl BackendEvent for MouseEvent {
    fn timestamp(&self) -> f64 {
        let event: &NSEvent = unsafe { self.inner.as_ref() };

        NSDate::new().timeIntervalSince1970() +
            event.timestamp() - NSProcessInfo::processInfo().systemUptime()
    }
}

impl BackendEvent for KeyEvent {
    fn timestamp(&self) -> f64 {
        let event: &NSEvent = unsafe { self.inner.as_ref() };

        NSDate::new().timeIntervalSince1970() +
            event.timestamp() - NSProcessInfo::processInfo().systemUptime()
    }
}

impl BackendEvent for WheelEvent {
    fn timestamp(&self) -> f64 {
        let event: &NSEvent = unsafe { self.inner.as_ref() };

        NSDate::new().timeIntervalSince1970() +
            event.timestamp() - NSProcessInfo::processInfo().systemUptime()
    }
}

impl BackendEvent for ShouldTerminateEvent {
    fn timestamp(&self) -> f64 {
        use std::time::{SystemTime, UNIX_EPOCH};

        // ShouldTerminateEvent is not a NSEvent, it is an artificial event
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
    }
}

impl BackendEvent for WillTerminateEvent {
    fn timestamp(&self) -> f64 {
        use std::time::{SystemTime, UNIX_EPOCH};

        // same as ShouldTerminateEvent
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
    }
}

pub struct MacOsPoller {
    broadcast: broadcast::Sender<ApplicationEvent>,
    is_dying: RwLock<bool>,
    handle: Handle,
}

impl MacOsPoller {
    pub(super) fn new(tx: broadcast::Sender<ApplicationEvent>, handle: Handle) -> Self {
        MacOsPoller {
            broadcast: tx,
            is_dying: RwLock::new(false),
            handle,
        }
    }

    pub fn poll(&self) -> Option<Retained<NSEvent>> {
        Some({
            // do nothing if we are not on the main thread
            let marker = MainThreadMarker::new()?;
            let application = NSApplication::sharedApplication(marker);

            application.updateWindows();
            let event = application.nextEventMatchingMask_untilDate_inMode_dequeue(
                NSEventMask::Any,
                None,
                unsafe {
                    if !*self.is_dying.read().unwrap() {
                        NSDefaultRunLoopMode
                    } else {
                        NSModalPanelRunLoopMode
                    }
                },
                true,
            )?;

            // backpressure to ensure AppKit also handles events
            application.sendEvent(&event);
            event
        })
    }
}

impl PlatformContext for MacOsPoller {
    fn events(&self) -> Receiver<ApplicationEvent> {
        self.broadcast.subscribe()
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

    fn async_handle(&self) -> &Handle {
        &self.handle
    }
}

impl Clone for MouseEvent {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            button: self.button,
            location: self.location,
            delta_x: self.delta_x,
            delta_y: self.delta_y,
            action: self.action,
            inner: unsafe {
                let ptr: Retained<NSEvent> = self.inner.into_objc();
                let r = OpaqueInner::from_objc(ptr.clone());
                mem::forget(ptr);
                r
            },
        }
    }
}

impl Drop for MouseEvent {
    fn drop(&mut self) {
        unsafe { self.inner.objc_drop_in_place::<NSEvent>() }
    }
}

impl Clone for KeyEvent {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
            caps: self.caps,
            shift: self.shift,
            control: self.control,
            alt: self.alt,
            meta: self.meta,
            function: self.function,
            inner: unsafe {
                let ptr: Retained<NSEvent> = self.inner.into_objc();
                let r = OpaqueInner::from_objc(ptr.clone());
                mem::forget(ptr);
                r
            },
        }
    }
}

impl Drop for KeyEvent {
    fn drop(&mut self) {
        unsafe { self.inner.objc_drop_in_place::<NSEvent>() }
    }
}

impl Clone for WheelEvent {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            delta_x: self.delta_x,
            delta_y: self.delta_y,
            inverted: self.inverted,
            inner: unsafe {
                let ptr: Retained<NSEvent> = self.inner.into_objc();
                let r = OpaqueInner::from_objc(ptr.clone());
                mem::forget(ptr);
                r
            },
        }
    }
}

impl Drop for WheelEvent {
    fn drop(&mut self) {
        unsafe { self.inner.objc_drop_in_place::<NSEvent>() }
    }
}

impl Clone for PeriodicEvent {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: unsafe {
                let ptr: Retained<NSEvent> = self.inner.into_objc();
                let r = OpaqueInner::from_objc(ptr.clone());
                mem::forget(ptr);
                r
            },
        }
    }
}

impl Drop for PeriodicEvent {
    fn drop(&mut self) {
        unsafe { self.inner.objc_drop_in_place::<NSEvent>() }
    }
}

impl Clone for ShouldTerminateEvent {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            notifier: self.notifier.clone(),
            inner: self.inner,
        }
    }
}

impl Drop for ShouldTerminateEvent {
    fn drop(&mut self) {
        // 2 is the daemon thread and self; we can consider this is the last Arc
        if Arc::strong_count(&self.notifier) > 2 {
            return;
        }

        // we are the last one, notify is required
        self.reply_ready();
    }
}

impl Clone for WillTerminateEvent {
    fn clone(&self) -> Self {
        Self {
            counter: self.counter.clone(),
            inner: self.inner,
        }
    }
}

// no need for drop, as the event is processed when every Sender instance is dropped

#[unsafe(export_name = "cpge_macos_should_terminate")]
extern "C-unwind" fn swift_notify_should_terminate() {
    let ctx: &MacOsPoller = context().downcast_context().unwrap();

    let once = Arc::new(SetOnce::new());

    *ctx.is_dying.write().unwrap() = true;
    ctx.broadcast.send(ApplicationEvent::ShouldTerminate(ShouldTerminateEvent {
        notifier: once.clone(),
        inner: OpaqueInner::from_ref(ctx), // ctx is 'static, we can use it
    })).unwrap();

    let application = OpaqueInner::from_objc(
        NSApplication::sharedApplication(MainThreadMarker::new().unwrap())
    );

    // when this returns, AppKit owns the event loop, aka the main thread.
    // we need to spawn a system thread to run the Tokio event loop until we
    // responded to AppKit
    thread::Builder::new().name("cpge-gl-should_terminate".to_string()).spawn(move || {
        let value = ctx.handle.block_on(async {
            task::yield_now().await;
            *once.wait().await
        });
        unsafe { application.into_objc::<NSApplication>().replyToApplicationShouldTerminate(value) };
    }).unwrap();
}

impl ShouldTerminateEvent {
    /// Notifies the system that the application is not ready, for now, to terminate immediately.
    pub fn reply_not_ready(&mut self) {
        let _ = self.notifier.set(false);
    }

    /// Notifies the system that the application is ready to terminate immediately.
    ///
    /// Note that the system may terminate the application immediately after this call. Do the
    /// cleanup before calling this method.
    pub fn reply_ready(&mut self) {
        let _ = self.notifier.set(true);
    }
}

#[unsafe(export_name = "cpge_macos_will_terminate")]
extern "C-unwind" fn swift_notify_will_terminate() {
    let ctx: &MacOsPoller = context().downcast_context().unwrap();

    let counter = Arc::new(());
    ctx.broadcast.send(ApplicationEvent::WillTerminate(WillTerminateEvent {
        counter: counter.clone(),
        inner: OpaqueInner::dangling(),
    })).unwrap();

    while Arc::strong_count(&counter) > 1 {
        thread::yield_now();
    }

    // shutting down event loop now
    context().block_on_shutdown();
}
