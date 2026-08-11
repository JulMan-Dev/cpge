//! GL module for CPGE crate.
//!
//! This module is made to render graphs, such as fractals.
//!
//! The module is named `gl` but uses Vulkan as render API.

#[cfg(target_os = "macos")]
pub mod macos;

use alloc::borrow::ToOwned;
use alloc::sync::Arc;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::{LoadingError, Validated, VulkanError, VulkanLibrary};

#[derive(Debug, Clone)]
pub struct GL(Arc<VulkanLibrary>);

impl GL {
    pub fn new() -> Result<Self, LoadingError> {
        VulkanLibrary::new().map(Self)
    }

    pub fn instance(&self) -> Result<GLInstance, Validated<VulkanError>> {
        Instance::new(
            self.0.clone(),
            InstanceCreateInfo {
                application_name: Some("CPGE".to_owned()),
                ..Default::default()
            },
        ).map(GLInstance)
    }
}

pub struct GLInstance(Arc<Instance>);
