//! macOS window logic.

extern crate objc;

use std::{eprint, eprintln, println, vec};
use std::process::abort;
use crate::gl::GL;
use std::sync::{Arc, OnceLock};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::VulkanError;

#[link(name = "cpge-native")]
unsafe extern "C-unwind" {
    fn cpge_init_application(width: isize, height: isize);

    fn cpge_mainloop();
}

static GLOBAL_GL: OnceLock<GL> = OnceLock::new();

/// Initializes the macOS window and loops the main loop.
pub fn start_application() {
    GLOBAL_GL.get_or_init(|| GL::new().unwrap());

    unsafe { cpge_init_application(800, 600) };
    unsafe { cpge_mainloop() };
}

/// This is called by the Swift mainloop when the view is ready.
#[unsafe(export_name = "cpge_spawn_vulkan")]
extern "C-unwind" fn spawn_vulkan(view: *mut ()) {
    let gl = GLOBAL_GL.get().expect("illegal call to cpge_spawn_vulkan");
    let instance = gl.instance().unwrap();
    let surface = instance.macos_surface(view).unwrap();

    let device = instance.first_physical_device().expect("cannot acquire gpu device");

    let queue_family_index = device
        .queue_family_properties()
        .iter()
        .position(|queue_family_properties| {
            queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS)
        })
        .expect("couldn't find a graphical queue family") as u32;

    let (device, mut queues) = Device::new(
        device,
        DeviceCreateInfo {
            // here we pass the desired queue family to use by index
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    ).expect("failed to create device");

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
}
