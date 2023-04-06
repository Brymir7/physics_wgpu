use std::{iter, time::Duration};
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};
const GRAVITY: f32 = 9.81;
use cgmath::{Rotation3, Zero};
use cgmath::InnerSpace;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}
impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}
struct PhysicalState {
    pub position: cgmath::Vector3<f32>,
    pub rotation: cgmath::Quaternion<f32>,
    pub scale: f32,
    pub velocity: [f32; 3],
}
impl PhysicalState {
    fn new(position: cgmath::Vector3<f32>) -> PhysicalState {
        let rotation = if position.is_zero() {
            // this is needed so an object at (0, 0, 0) won't get scaled to zero
            // as Quaternions can effect scale if they're not created correctly
            cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
        } else {
            cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(25.0))
        };
        PhysicalState {
            position,
            rotation,
            scale: 1.0,
            velocity: [0.0; 3],
        }
    }
    fn create_transformation_matrix(&self) -> cgmath::Matrix4<f32> {
        (cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation))
    }
}


struct Floor {
    position: [f32; 3],
    size: f32,
}
impl Floor {
    fn new(size: f32) -> Floor {
        Floor {
            position: [0.0, 0.0, 0.0],
            size,
        }
    }
    fn floor_positions(&self) -> Vec<[f32; 3]> {
        let x = self.position[0];
        let y = self.position[1];
        let z = self.position[2];

        let half_size = self.size / 2.0;
        let half_thickness = self.size * 0.005;
        [
            [-half_size, -half_thickness, -half_size],
            [half_size, -half_thickness, -half_size],
            [-half_size, half_thickness, -half_size],
            [-half_size, half_thickness, -half_size],
            [half_size, -half_thickness, -half_size],
            [half_size, half_thickness, -half_size],
            // Right face
            [half_size, -half_thickness, -half_size],
            [half_size, -half_thickness, half_size],
            [half_size, half_thickness, -half_size],
            [half_size, half_thickness, -half_size],
            [half_size, -half_thickness, half_size],
            [half_size, half_thickness, half_size],
            // Back face
            [half_size, -half_thickness, half_size],
            [-half_size, -half_thickness, half_size],
            [half_size, half_thickness, half_size],
            [half_size, half_thickness, half_size],
            [-half_size, -half_thickness, half_size],
            [-half_size, half_thickness, half_size],
            // Left face
            [-half_size, -half_thickness, half_size],
            [-half_size, -half_thickness, -half_size],
            [-half_size, half_thickness, half_size],
            [-half_size, half_thickness, half_size],
            [-half_size, -half_thickness, -half_size],
            [-half_size, half_thickness, -half_size],
            // Top face
            [-half_size, half_thickness, half_size],
            [half_size, half_thickness, half_size],
            [-half_size, half_thickness, -half_size],
            [-half_size, half_thickness, -half_size],
            [half_size, half_thickness, half_size],
            [half_size, half_thickness, -half_size],
            // Bottom face
            [-half_size, -half_thickness, half_size],
            [half_size, -half_thickness, half_size],
            [-half_size, -half_thickness, -half_size],
            [-half_size, -half_thickness, -half_size],
            [half_size, -half_thickness, half_size],
            [half_size, -half_thickness, -half_size],
        ]
        .to_vec()
    }
    fn floor_colors(&self) -> Vec<[i8; 3]> {
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        .to_vec()
    }
    fn vertex(&self, p: [f32; 3], c: [i8; 3]) -> Vertex {
        Vertex {
            position: [p[0] as f32, p[1] as f32, p[2] as f32],
            color: [c[0] as f32, c[1] as f32, c[2] as f32],
        }
    }
    fn create_vertices(&self) -> Vec<Vertex> {
        let pos = self.floor_positions();
        let col = self.floor_colors();
        let mut data: Vec<Vertex> = Vec::with_capacity(pos.len());
        for i in 0..pos.len() {
            data.push(self.vertex(pos[i], col[i]));
        }
        data.to_vec()
    }
}
struct Cube {
    position: [f32; 3],
    size: f32,
    physical_state: PhysicalState,
}
impl Cube {
    fn new(position: [f32; 3], size: f32) -> Self {
        Self {
            position,
            size,
            physical_state: PhysicalState::new(position.into()),
        }
    }
    fn cube_positions(&self) -> Vec<[f32; 3]> {
        let x = self.position[0];
        let y = self.position[1];
        let z = self.position[2];
        let half_size = self.size / 2.0;
        [
            [x - half_size, y - half_size, z + half_size],
            [x + half_size, y - half_size, z + half_size],
            [x - half_size, y + half_size, z + half_size],
            [x - half_size, y + half_size, z + half_size],
            [x + half_size, y - half_size, z + half_size],
            [x + half_size, y + half_size, z + half_size],
            // right (1, 0, 0)
            [x + half_size, y - half_size, z + half_size],
            [x + half_size, y - half_size, z - half_size],
            [x + half_size, y + half_size, z + half_size],
            [x + half_size, y + half_size, z + half_size],
            [x + half_size, y - half_size, z - half_size],
            [x + half_size, y + half_size, z - half_size],
            // back (0, 0, -1)
            [x + half_size, y - half_size, z - half_size],
            [x - half_size, y - half_size, z - half_size],
            [x + half_size, y + half_size, z - half_size],
            [x + half_size, y + half_size, z - half_size],
            [x - half_size, y - half_size, z - half_size],
            [x - half_size, y + half_size, z - half_size],
            // left (-1, 0, 0)
            [x - half_size, y - half_size, z - half_size],
            [x - half_size, y - half_size, z + half_size],
            [x - half_size, y + half_size, z - half_size],
            [x - half_size, y + half_size, z - half_size],
            [x - half_size, y - half_size, z + half_size],
            [x - half_size, y + half_size, z + half_size],
            [x - half_size, y + half_size, z + half_size],
            [x + half_size, y + half_size, z + half_size],
            [x - half_size, y + half_size, z - half_size],
            [x - half_size, y + half_size, z - half_size],
            [x + half_size, y + half_size, z + half_size],
            [x + half_size, y + half_size, z - half_size],
            // bottom (0, -1, 0)
            [x - half_size, y - half_size, z - half_size],
            [x + half_size, y - half_size, z - half_size],
            [x - half_size, y - half_size, z + half_size],
            [x - half_size, y - half_size, z + half_size],
            [x + half_size, y - half_size, z - half_size],
            [x + half_size, y - half_size, z + half_size],
        ]
        .to_vec()
    }
    fn cube_colors(&self) -> Vec<[i8; 3]> {
        [
            // front - blue
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            // right - red
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            // back - yellow
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            // left - aqua
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
            // top - green
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            // bottom - fuchsia
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
        ]
        .to_vec()
    }
    fn vertex(&self, p: [f32; 3], c: [i8; 3]) -> Vertex {
        Vertex {
            position: [p[0] as f32, p[1] as f32, p[2] as f32],
            color: [c[0] as f32, c[1] as f32, c[2] as f32],
        }
    }
    fn create_vertices(&self) -> Vec<Vertex> { 
        let pos = self.cube_positions();
        let col = self.cube_colors();
        let mut data: Vec<Vertex> = Vec::with_capacity(pos.len());
        for i in 0..pos.len() {
            data.push(self.vertex(pos[i], col[i]));
        }
        data.to_vec()
    }
    fn apply_gravity(&mut self, dt: f32) {
        self.physical_state.velocity[1] -= GRAVITY * dt;
    }
    fn update(&mut self, dt: f32) {
        self.physical_state.position[0] += self.physical_state.velocity[0] * dt;
        self.physical_state.position[1] += self.physical_state.velocity[1] * dt;
        self.physical_state.position[2] += self.physical_state.velocity[2] * dt;
    }
}
struct World {
    size: f32,
    half_size: f32,
    objects: Vec<Cube>,
}
impl World {
    fn new (size: f32, objects:Vec<Cube>) -> World {
        World {
            size,
            half_size: size * 0.5,
            objects: objects,
        }
    }
    fn update(&mut self, dt: std::time::Duration) {
        let dt_as_secs = dt.as_secs_f32();
        for object in &mut self.objects {
            if object.physical_state.position[1] - object.size >= -self.half_size {
                object.apply_gravity(dt_as_secs);
            }
            object.update(dt_as_secs);
            if let Some(new_velocity) = Self::handle_borders(object, dt_as_secs) {
                object.physical_state.velocity[1] = new_velocity;
            }
        }
    }
    fn handle_borders(object: &mut Cube, half_size: f32) -> Option<f32> {
        if object.physical_state.position[1] + object.size * 0.5 <= -half_size {
            return Some(-0.75*object.physical_state.velocity[1]);
        }
        None
    }
}
//indices enable to reuse data of vertex (triangles that use the same vertices for example)
const INDICES: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4];

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);
struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl CameraController {
    fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::W | VirtualKeyCode::Up => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::A | VirtualKeyCode::Left => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::D | VirtualKeyCode::Right => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::F | VirtualKeyCode::Right => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn update_camera(&self, camera: &mut Camera) {
        use cgmath::InnerSpace;
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        // Prevents glitching when camera gets too close to the
        // center of the scene.
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(camera.up);

        // Redo radius calc in case the fowrard/backward is pressed.
        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            // Rescale the distance between the target and eye so
            // that it doesn't change. The eye therefore still
            // lies on the circle made by the target and eye.
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
    }
}

struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}
impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        // 1.
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        // 2.
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        // 3.
        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }
}
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    // We can't use cgmath with bytemuck directly so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TransformUniform {
    transform_matrix: [[f32; 4]; 4],
}

impl TransformUniform {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            transform_matrix: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_transform_matrix(&mut self, cube: &Cube) -> [[f32; 4]; 4] {
        self.transform_matrix = cube.physical_state.create_transformation_matrix().into();
        self.transform_matrix
    }
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // attribute for the transformation matrix
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}
struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_controller: CameraController,
    camera_bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    world: World,
    floor: Floor,
    transform_uniform: TransformUniform, //used to make matrix into valid type -> uniform it
    transform_buffer: wgpu::Buffer,
    transform_bind_group: wgpu::BindGroup,
}

impl State {
    async fn new(window: Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                },
                // Some(&std::path::Path::new("trace")), // Trace path
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an Srgb surface texture. Using a different
        // one will result all the colors comming out darker. If you want to support non
        // Srgb surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .filter(|f| f.describe().srgb)
            .next()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);
        let camera = Camera {
            eye: (0.0, 1.0, 2.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });
        let transform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX, // The transformation matrix will be used in the vertex shader
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("transform_bind_group_layout"),
            });

        let camera_controller = CameraController::new(0.05);

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &transform_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",     // 1.
                buffers: &[Vertex::desc(), TransformUniform::desc()], // 2.
            },
            fragment: Some(wgpu::FragmentState {
                // 3.
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    // 4.
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                count: 1,                         // 2.
                mask: !0,                         // 3.
                alpha_to_coverage_enabled: false, // 4.
            },
            multiview: None, // 5.
        });
        
        let mut objects: Vec<Cube> = Vec::new();
        let cube = Cube::new([0.0, 0.0, 0.0], 1.0);
        objects.push(cube);
        let world = World::new(100.0, objects);
        let floor = Floor::new(10.0);

        //let concatenated_vertices = objects.iter().fold(Vec::new(), |mut acc, cube| {
        //    acc.extend_from_slice(&cube.create_vertices());
        //    acc
        //});
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&world.objects[0].create_vertices()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mut transform_uniform = TransformUniform::new();
        transform_uniform.update_transform_matrix(&world.objects[0]);
        let transform_uniform_matrices: Vec<TransformUniform> = world.objects.iter().map(|obj| {
            let mut transform = TransformUniform::new();
            transform.update_transform_matrix(obj);
            transform
        }).collect();
        let transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transform Uniform Buffer"),
            contents: bytemuck::cast_slice(&transform_uniform_matrices),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::VERTEX |  wgpu::BufferUsages::COPY_DST,
        });
        // Create the bind group
        let transform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &transform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 1,
                resource: transform_buffer.as_entire_binding(),
            }],
            label: Some("transform_bind_group"),
        });
        let num_indices = INDICES.len() as u32;
        Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            world,
            floor,
            transform_uniform,
            transform_buffer,
            transform_bind_group,
        }
    }

    fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    #[allow(unused_variables)]
    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera_controller.process_events(event)
    }

    fn update(&mut self, dt: std::time::Duration) {
        // physic logic
        //self.objects[0].update(dt);
        self.world.update(dt);
        let transform_uniforms: Vec<TransformUniform> = self.world.objects.iter().map(|obj| {
            let mut transform = TransformUniform::new();
            transform.update_transform_matrix(obj);
            transform
        }).collect();
        // camera

        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        self.queue.write_buffer(
            &self.transform_buffer,
            0,
            bytemuck::cast_slice(&transform_uniforms),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.5,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipeline); // 2.
                                                             // NEW!
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.transform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.transform_buffer.slice(..));
            //render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            //render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
            render_pass.draw(0..36, 0..self.world.objects.len() as u32);
        }
        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub async fn run() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    // State::new uses async code, so we're going to wait for it to finish
    let mut state = State::new(window).await;
    let mut last_render_time = Instant::now();
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() => {
                if !state.input(event) {
                    // UPDATED!
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            // new_inner_size is &&mut so w have to dereference it twice
                            state.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
            }
            Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                let now = Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;
                state.update(dt);
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        state.resize(state.size)
                    }
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,

                    Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                }
            }
            Event::RedrawEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                state.window().request_redraw();
            }
            _ => {}
        }
    });
}
