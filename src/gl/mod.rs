//! GL module for CPGE crate.
//!
//! This module is made to render graphs, such as fractals.
//!
//! The module is named `gl` but uses Vulkan as render API.

#[cfg(target_os = "macos")]
pub mod macos;

use alloc::borrow::ToOwned;
use alloc::sync::Arc;
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
use vulkano::{LoadingError, Validated, VulkanError, VulkanLibrary};
use vulkano::device::physical::PhysicalDevice;
use vulkano::swapchain::Surface;

#[derive(Debug, Clone)]
pub struct GL(pub Arc<VulkanLibrary>);

impl GL {
    pub fn new() -> Result<Self, LoadingError> {
        VulkanLibrary::new().map(Self)
    }

    pub fn instance(&self) -> Result<GLInstance, Validated<VulkanError>> {
        let mut enabled_extensions = InstanceExtensions::empty();

        #[cfg(target_os = "macos")]
        {
            enabled_extensions.khr_surface = true;
            enabled_extensions.ext_metal_surface = true;
            enabled_extensions.mvk_macos_surface = true;
        }

        Instance::new(
            self.0.clone(),
            InstanceCreateInfo {
                application_name: Some("CPGE".to_owned()),
                enabled_extensions,
                ..Default::default()
            },
        ).map(GLInstance)
    }
}

pub struct GLInstance(pub Arc<Instance>);

impl GLInstance {
    pub fn first_physical_device(&self) -> Result<Arc<PhysicalDevice>, VulkanError> {
        let mut iter = self.0.enumerate_physical_devices()?;

        if let Some(r) = iter.next() {
            Ok(r)
        } else {
            Err(VulkanError::InitializationFailed)
        }
    }

    #[cfg(target_os = "macos")]
    pub fn macos_surface(&self, view: *mut ()) -> Result<Arc<Surface>, Validated<VulkanError>> {
        unsafe { Surface::from_mac_os(self.0.clone(), view.cast(), None) }
    }
}
