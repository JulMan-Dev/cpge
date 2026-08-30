//! GL module for CPGE crate.
//!
//! This module is made to render graphs, such as fractals.
//!
//! The module is named `gl` but uses Vulkan as render API.

#[cfg(target_os = "macos")]
pub mod macos;
pub mod event;
pub mod ptr;

use crate::gl::context::PlatformContext;
use crate::gl::event::internal::ApplicationEventSource;
use alloc::borrow::ToOwned;
use alloc::sync::Arc;
use alloc::vec::Vec;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use std::vec;
use std::{format, thread};
use tokio::runtime::Handle;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio::{runtime, task, time};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo};
use vulkano::{LoadingError, Validated, VulkanError, VulkanLibrary};

static CONTEXT: OnceLock<context::Context> = OnceLock::new();
static SHOULD_TERMINATE: AtomicBool = AtomicBool::new(false);

pub mod context {
    use crate::gl::event::{ApplicationEvent, internal};
    use alloc::boxed::Box;
    use alloc::vec::Vec;
    use std::any::Any;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use tokio::runtime::Handle;
    use tokio::sync::{broadcast, mpsc};
    use crate::gl::SHOULD_TERMINATE;

    pub struct Context {
        pub(super) inner: Arc<dyn PlatformContext>,
        pub(super) ids: Arc<AtomicUsize>,
        pub(super) spawner: mpsc::Sender<(usize, Pin<Box<dyn Future<Output = ()> + Send>>)>,
        pub(super) responder: broadcast::Receiver<usize>,
    }

    pub trait PlatformContext: internal::ApplicationEventSource + Send + Sync + Any + 'static {
        fn events(&self) -> broadcast::Receiver<ApplicationEvent>;
    }

    impl Context {
        /// Schedules the future to run on the main thread. This returns when the main thread
        /// notifies the task is done.
        ///
        /// This may be used to invoke system APIs that require running on the main thread. You
        /// generally don't need to invoke this manually, abstractions do this for you.
        pub async fn spawn_on_main<F>(&self, future: F)
        where
            F: Future<Output = ()> + Send + 'static,
        {
            let next_id = self.ids.fetch_add(1, Ordering::SeqCst);
            self.spawner.send((next_id, Box::pin(future))).await.unwrap();

            let mut responder = self.responder.resubscribe();
            while let Ok(id) = responder.recv().await {
                if id == next_id {
                    break;
                }
            }
        }

        pub(super) fn downcast_context<T: PlatformContext>(&self) -> Option<&T> {
            (&*self.inner as &(dyn Any + Send + Sync)).downcast_ref()
        }

        pub(super) fn block_on_shutdown(&self) {
            SHOULD_TERMINATE.store(true, Ordering::SeqCst);
            while !SHOULD_TERMINATE.load(Ordering::SeqCst) {
                thread::yield_now();
            }
        }
    }

    impl Clone for Context {
        fn clone(&self) -> Self {
            Self {
                inner: self.inner.clone(),
                ids: self.ids.clone(),
                spawner: self.spawner.clone(),
                responder: self.responder.resubscribe(),
            }
        }
    }

    impl internal::ApplicationEventSource for Context {
        fn poll_events(&self, events: &mut Vec<ApplicationEvent>) {
            self.inner.poll_events(events);
        }

        fn async_handle(&self) -> &Handle {
            self.inner.async_handle()
        }
    }

    impl PlatformContext for Context {
        fn events(&self) -> broadcast::Receiver<ApplicationEvent> {
            self.inner.events()
        }
    }
}

/// Boots the GL window and event loop. This must be called on the main thread.
///
/// Calling this method blocks until the event loop terminates.
///
/// Note that some platforms do not resume the thread when the system asks the application to
/// terminate. You can listen to the [`ShouldTerminate`](event::ApplicationEvent::ShouldTerminate)
/// event to handle cleanup before terminating.
///
/// This spawns a runtime system thread to ensure the event loop continues to run even if the system
/// blocks the main thread. The future may not run on the main thread, if some code requires this,
/// use [`Context::spawn_on_main`](context::Context::spawn_on_main).
///
/// If the passed future returns, the event loop terminates and this returns.
pub fn boot_gl<F, T>(f: F)
where
    F: Send + FnOnce() -> T + 'static,
    T: Future<Output = ()> + Send,
{
    let handle_rt = Arc::new(OnceLock::<Handle>::new());
    let (tx_context, rx_context) = oneshot::channel();

    // we spawn a runtime system thread.
    let handle = {
        let handle_rt = handle_rt.clone();
        let main = thread::current();

        thread::Builder::new().name("cpge-event-loop".to_owned()).spawn(move || {
            let rt = runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            handle_rt.set(rt.handle().clone()).unwrap();
            drop(handle_rt);
            main.unpark();

            rt.block_on(async {
                let _: () = rx_context.await.unwrap(); // wait for context to be initialized

                let handle = rt.spawn(async {
                    // no longer need yielding since we waited for the context
                    f().await;
                });

                loop {
                    if handle.is_finished() || SHOULD_TERMINATE.load(Ordering::Relaxed) {
                        break;
                    }

                    task::yield_now().await;
                }
            });

            drop(rt);
        }).unwrap()
    };

    // synchronously wait for runtime thread to set the runtime handle.
    thread::park();
    let handle_rt = Arc::into_inner(handle_rt)
        .expect("runtime thread did not drop the arc pointer")
        .into_inner()
        .expect("runtime thread did not send the handle");

    let (tx_tasks, mut rx_tasks) = mpsc::channel(32);
    let (tx_responder, rx_responder) = broadcast::channel(32);

    handle_rt.block_on(async {
        let (tx, _) = broadcast::channel(32);

        let context = {
            // we initialize the platform context from the main thread in case the backend requires
            // this (AppKit, for example, does)
            let application: Arc<dyn PlatformContext> = Arc::new(cfg_select! {
                target_os = "macos" => { macos::start_application(tx.clone(), handle_rt.clone()) }
            });

            let context = context::Context {
                inner: application,
                ids: Arc::new(Default::default()),
                spawner: tx_tasks,
                responder: rx_responder,
            };

            CONTEXT.set(context).ok().expect("a gl context already defined");
            CONTEXT.get().unwrap()
        };

        tx_context.send(()).unwrap();

        let mut ticker = time::interval(Duration::from_millis(10));
        let mut vec = Vec::new();

        loop {
            if handle.is_finished() || SHOULD_TERMINATE.load(Ordering::Relaxed) {
                break;
            }

            tokio::select! {
                _ = ticker.tick() => {
                    context.poll_events(&mut vec);

                    for event in vec.drain(..) {
                        tx.send(event).unwrap();
                    }
                },
                Some((id, task)) = rx_tasks.recv() => {
                    task.await;
                    tx_responder.send(id).unwrap();
                },
            }
        }
    });

    // synchronously wait for runtime thread to end before returning
    SHOULD_TERMINATE.store(true, Ordering::Relaxed);
    handle.join().unwrap();
    SHOULD_TERMINATE.store(false, Ordering::Relaxed);
}

/// Gets the current context.
///
/// # Panics
///
/// Panics if called outside a [`boot_gl`] context.
pub fn context() -> &'static context::Context {
    CONTEXT.get().expect("Not called in bool_gl context")
}

pub(super) fn mark_should_terminate() {
    SHOULD_TERMINATE.store(true, Ordering::Relaxed);
}

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
    pub fn first_physical_device(&self, surface: &Surface) -> Result<(Arc<PhysicalDevice>, u32), VulkanError> {
        self.0.enumerate_physical_devices()?
            .filter(|x| x.supported_extensions().contains(&DeviceExtensions {
                khr_swapchain: true,
                ..Default::default()
            }))
            .filter_map(|p| {
                p.queue_family_properties().iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.contains(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, surface).unwrap_or(false)
                    })
                    .map(|q| (p, q as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .ok_or(VulkanError::ExtensionNotPresent)
    }

    #[cfg(target_os = "macos")]
    pub fn macos_surface(&self, view: *mut ()) -> Result<Arc<Surface>, Validated<VulkanError>> {
        unsafe { Surface::from_mac_os(self.0.clone(), view.cast(), None) }
    }
}

#[derive(Default)]
pub struct Data {
    device: Option<Arc<Device>>,
    queue: Option<Arc<Queue>>,
    allocator: Option<Arc<StandardMemoryAllocator>>,
    viewport: Viewport,
}

/// Used to create Vulkan holder data.
#[unsafe(export_name = "cpge_make_vulkan_data")]
extern "C-unwind" fn make_vulkan_data(output: *mut *const Data) {
    unsafe {
        output.write(Arc::into_raw(Arc::new(Data::default())));
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460

            void main() {
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        ",
    }
}

fn get_render_pass(device: Arc<Device>, swapchain: &Arc<Swapchain>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                // Set the format the same as the swapchain.
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    ).unwrap()
}

fn get_framebuffers(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images.iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            ).unwrap()
        })
        .collect()
}

fn get_command_buffers(
    command_buffer_allocator: &Arc<StandardCommandBufferAllocator>,
    queue: &Arc<Queue>,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            ).unwrap();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap()
                .bind_pipeline_graphics(pipeline.clone())
                .unwrap()
                .end_render_pass(Default::default())
                .unwrap();

            builder.build().unwrap()
        })
        .collect()
}

pub fn init_vulkan(instance: GLInstance, surface: Arc<Surface>, mut data: Arc<Data>) {
    let mut data_mut = Arc::get_mut(&mut data).unwrap();

    let (device, queue_family_index) = instance.first_physical_device(&surface)
        .expect("cannot acquire gpu device");

    let (device, mut queues) = Device::new(
        device,
        DeviceCreateInfo {
            // here we pass the desired queue family to use by index
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: DeviceExtensions {
                khr_swapchain: true,
                ..DeviceExtensions::empty()
            },
            ..Default::default()
        },
    ).expect("failed to create device");
    let device = data_mut.device.insert(device);

    let queue = queues.next().expect("no queue");
    let queue = data_mut.queue.insert(queue);

    data_mut.viewport = Viewport {
        extent: [1280.0, 720.0],
        ..Default::default()
    };

    let (mut swapchain, images) = {
        let caps = device.physical_device()
            .surface_capabilities(&surface, Default::default())
            .expect("failed to get surface capabilities");

        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let image_format = device.physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        Swapchain::new(
            device.clone(),
            surface,
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count,
                image_format,
                image_extent: [1280, 720],
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha,
                ..Default::default()
            },
        ).unwrap()
    };

    let render_pass = get_render_pass(device.clone(), &swapchain);
    let framebuffers = get_framebuffers(&images, &render_pass);

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let memory_allocator = data_mut.allocator.insert(memory_allocator);

    let fs: Arc<ShaderModule> = fs::load(device.clone()).unwrap();
    let vs: Arc<ShaderModule> = vs::load(device.clone()).unwrap();

    let pipeline = {
        let vs = vs.entry_point("main").unwrap();
        let fs = fs.entry_point("main").unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        ).unwrap();

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(VertexInputState::default()),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [data_mut.viewport.clone()].into_iter().collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState {
                    rasterizer_discard_enable: false,
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        ).unwrap()
    };

    let command_buffer_allocator = Arc::new(
        StandardCommandBufferAllocator::new(device.clone(), Default::default())
    );

    let mut command_buffers = get_command_buffers(
        &command_buffer_allocator,
        queue,
        &pipeline,
        &framebuffers,
    );
}
