use burn::tensor::backend::Backend;

#[test]
fn wgpu_probe() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
        &device,
        burn::backend::wgpu::RuntimeOptions {
            memory_config: burn::backend::wgpu::MemoryConfiguration::ExclusivePages,
            ..Default::default()
        },
    );
    type B = burn::backend::Wgpu<f32, i64, u32>;
    let a = burn::tensor::Tensor::<B, 2>::from_floats([[1.0f32, 2.0], [3.0, 4.0]], &device);
    let b = a.clone().matmul(a);
    let sum: f32 = b.into_data().to_vec::<f32>().unwrap().iter().sum();
    println!("backend name: {}", B::name(&device));
    println!("matmul sum = {sum} (expect 54)");
    assert!((sum - 54.0).abs() < 1e-3);
}
