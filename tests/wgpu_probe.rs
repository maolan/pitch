#[test]
fn wgpu_probe() {
    let device = maolan_pitch::wgpu_device();
    let a = burn::tensor::Tensor::<2>::from_floats([[1.0f32, 2.0], [3.0, 4.0]], &device);
    let b = a.clone().matmul(a);
    let sum: f32 = b.into_data().try_to_vec::<f32>().unwrap().iter().sum();
    println!("device: {device:?}");
    println!("matmul sum = {sum} (expect 54)");
    assert!((sum - 54.0).abs() < 1e-3);
}
