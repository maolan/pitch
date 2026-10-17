//! Burn reimplementation of the FCPE (Fast Context-based Pitch Estimation)
//! network, matching the exported ONNX graph of the bundled `fcpe_c_v001`
//! checkpoint (a `CFNaiveMelPE` with `conv_only = true`, `n_layers = 6`,
//! `hidden_dims = 512`, `out_dims = 360`).
//!
//! Architecture (derived from the checkpoint tensors and ONNX graph):
//!
//! ```text
//! mel (1, T, 128)
//!   -> Conv1d(128 -> 512, k = 3, pad = 1)
//!   -> GroupNorm(4, 512)            (affine params stored as onnx::Mul_242 /
//!                                    onnx::Add_243 in the export)
//!   -> LeakyReLU(0.01)
//!   -> Conv1d(512 -> 512, k = 3, pad = 1)
//!   -> 6 x CEST conv module, residual around each:
//!        LayerNorm
//!        Conv1d(512 -> 2048, k = 1)
//!        GLU (first half * sigmoid(second half))
//!        Depthwise Conv1d(1024, k = 31, pad = 15, groups = 1024)
//!        SiLU
//!        Conv1d(1024 -> 512, k = 1)
//!   -> LayerNorm
//!   -> Linear(512 -> 360)           (weight stored pre-flattened as
//!                                    onnx::MatMul_247 by the exporter)
//!   -> sigmoid  => latent pitch-class distribution (1, T, 360)
//! ```
//!
//! The attention branch of the upstream conformer layer is not present in the
//! exported checkpoint (`conv_only = true`) and the per-layer `norm` is dead
//! code in that configuration, so neither appears here.

use anyhow::{Context, Result, anyhow, bail};
use burn::tensor::Tensor;
use burn::tensor::TensorData;
use burn::tensor::activation::{leaky_relu, sigmoid, silu};
use burn_store::{BurnpackStore, ModuleStore};
use std::path::Path;

use crate::mel::N_BINS;

const HIDDEN: usize = 512;
const EXPANSION: usize = 2;
const INNER: usize = HIDDEN * EXPANSION; // 1024
const N_LAYERS: usize = 6;
const OUT_DIMS: usize = 360;
const GROUPS: usize = 4;
const KERNEL: usize = 31;
const F0_MIN: f32 = 32.70;
const F0_MAX: f32 = 1975.5;
const EPS: f32 = 1e-5;

/// Cent values of the output bins: linspace(f0_to_cent(F0_MIN),
/// f0_to_cent(F0_MAX), OUT_DIMS), where f0_to_cent(f) = 1200 * log2(f / 10).
fn cent_table() -> Vec<f32> {
    let start = 1200.0 * (F0_MIN / 10.0).log2();
    let end = 1200.0 * (F0_MAX / 10.0).log2();
    (0..OUT_DIMS)
        .map(|i| start + (end - start) * i as f32 / (OUT_DIMS - 1) as f32)
        .collect()
}

struct ConvModuleWeights {
    norm_w: Tensor<1>,
    norm_b: Tensor<1>,
    pw_w: Tensor<3>,
    pw_b: Tensor<1>,
    dw_w: Tensor<3>,
    dw_b: Tensor<1>,
    out_w: Tensor<3>,
    out_b: Tensor<1>,
}

/// Loaded FCPE parameters.
pub struct FcpeWeights {
    in_w: Tensor<3>,
    in_b: Tensor<1>,
    gn_w: Tensor<1>,
    gn_b: Tensor<1>,
    in2_w: Tensor<3>,
    in2_b: Tensor<1>,
    layers: Vec<ConvModuleWeights>,
    norm_w: Tensor<1>,
    norm_b: Tensor<1>,
    out_w: Tensor<2>,
    out_b: Tensor<1>,
}

type TensorLookup = dyn Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>;

fn get_tensor(store: &TensorLookup, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let (data, actual) = store(name).ok_or_else(|| anyhow!("missing tensor {name}"))?;
    if actual != shape {
        bail!("tensor {name} has shape {actual:?}, expected {shape:?}");
    }
    Ok(data)
}

fn tensor1(device: &burn::tensor::Device, data: Vec<f32>) -> Tensor<1> {
    let len = data.len();
    Tensor::from_data(TensorData::new(data, [len]), device)
}

fn tensor2(device: &burn::tensor::Device, data: Vec<f32>, d0: usize, d1: usize) -> Tensor<2> {
    Tensor::from_data(TensorData::new(data, [d0, d1]), device)
}

fn tensor3(
    device: &burn::tensor::Device,
    data: Vec<f32>,
    d0: usize,
    d1: usize,
    d2: usize,
) -> Tensor<3> {
    Tensor::from_data(TensorData::new(data, [d0, d1, d2]), device)
}

impl FcpeWeights {
    /// Load weights from a Burn Pack (`.bpk`) exported from the FCPE ONNX
    /// checkpoint. Tensors are looked up by their dotted ONNX name.
    pub fn from_burnpack(path: &Path, device: &burn::tensor::Device) -> Result<Self> {
        use burn::tensor::DType;

        let snapshots = BurnpackStore::from_file(path)
            .get_all_tensors()
            .with_context(|| format!("failed to read burnpack {}", path.display()))?
            .clone();

        let store = move |name: &str| -> Option<(Vec<f32>, Vec<usize>)> {
            snapshots.values().find_map(|snap| {
                if snap.name == name {
                    let data = burn_store::bridge::to_data(snap).ok()?;
                    if data.dtype() == DType::F32 {
                        let shape = data.shape().to_vec();
                        let values: Vec<f32> = data.try_to_vec::<f32>().ok()?;
                        Some((values, shape))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
        };

        Self::from_lookup(&store, device)
    }

    fn from_lookup(store: &TensorLookup, device: &burn::tensor::Device) -> Result<Self> {
        let in_w = get_tensor(store, "model.input_stack.0.weight", &[HIDDEN, N_BINS, 3])?;
        let in_b = get_tensor(store, "model.input_stack.0.bias", &[HIDDEN])?;
        // GroupNorm affine parameters, exported as broadcastable [512, 1]
        // tensors feeding a Mul/Add pair after the InstanceNormalization.
        let gn_w = get_tensor(store, "onnx::Mul_242", &[HIDDEN, 1])?;
        let gn_b = get_tensor(store, "onnx::Add_243", &[HIDDEN, 1])?;
        let in2_w = get_tensor(store, "model.input_stack.3.weight", &[HIDDEN, HIDDEN, 3])?;
        let in2_b = get_tensor(store, "model.input_stack.3.bias", &[HIDDEN])?;

        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            let p = format!("model.net.encoder_layers.{i}.conformer.net");
            layers.push(ConvModuleWeights {
                norm_w: tensor1(
                    device,
                    get_tensor(store, &format!("{p}.0.weight"), &[HIDDEN])?,
                ),
                norm_b: tensor1(
                    device,
                    get_tensor(store, &format!("{p}.0.bias"), &[HIDDEN])?,
                ),
                pw_w: tensor3(
                    device,
                    get_tensor(store, &format!("{p}.2.weight"), &[INNER * 2, HIDDEN, 1])?,
                    INNER * 2,
                    HIDDEN,
                    1,
                ),
                pw_b: tensor1(
                    device,
                    get_tensor(store, &format!("{p}.2.bias"), &[INNER * 2])?,
                ),
                dw_w: tensor3(
                    device,
                    get_tensor(store, &format!("{p}.4.conv.weight"), &[INNER, 1, KERNEL])?,
                    INNER,
                    1,
                    KERNEL,
                ),
                dw_b: tensor1(
                    device,
                    get_tensor(store, &format!("{p}.4.conv.bias"), &[INNER])?,
                ),
                out_w: tensor3(
                    device,
                    get_tensor(store, &format!("{p}.6.weight"), &[HIDDEN, INNER, 1])?,
                    HIDDEN,
                    INNER,
                    1,
                ),
                out_b: tensor1(
                    device,
                    get_tensor(store, &format!("{p}.6.bias"), &[HIDDEN])?,
                ),
            });
        }

        let norm_w = get_tensor(store, "model.norm.weight", &[HIDDEN])?;
        let norm_b = get_tensor(store, "model.norm.bias", &[HIDDEN])?;
        // weight_norm Linear, already materialized by the exporter: y = x @ W + b
        let out_w = get_tensor(store, "onnx::MatMul_247", &[HIDDEN, OUT_DIMS])?;
        let out_b = get_tensor(store, "model.output_proj.bias", &[OUT_DIMS])?;

        Ok(Self {
            in_w: tensor3(device, in_w, HIDDEN, N_BINS, 3),
            in_b: tensor1(device, in_b),
            gn_w: tensor1(device, gn_w),
            gn_b: tensor1(device, gn_b),
            in2_w: tensor3(device, in2_w, HIDDEN, HIDDEN, 3),
            in2_b: tensor1(device, in2_b),
            layers,
            norm_w: tensor1(device, norm_w),
            norm_b: tensor1(device, norm_b),
            out_w: tensor2(device, out_w, HIDDEN, OUT_DIMS),
            out_b: tensor1(device, out_b),
        })
    }

    fn layer_norm(x: Tensor<3>, w: &Tensor<1>, b: &Tensor<1>) -> Tensor<3> {
        // normalize over the last dim (biased variance, eps = 1e-5)
        let mean = x.clone().mean_dim(2);
        let centered = x.clone().sub(mean.clone());
        let var = centered
            .clone()
            .powf_scalar(2.0)
            .mean_dim(2)
            .add_scalar(EPS)
            .sqrt();
        let n = x.dims()[2];
        let w = w.clone().reshape([1, 1, w.dims()[0]]).expand([1, 1, n]);
        let b = b.clone().reshape([1, 1, b.dims()[0]]).expand([1, 1, n]);
        centered.div(var).mul(w).add(b)
    }

    fn conv1d(
        x: Tensor<3>,
        weight: &Tensor<3>,
        bias: &Tensor<1>,
        padding: usize,
        groups: usize,
    ) -> Tensor<3> {
        let stride = 1;
        let options = burn::tensor::ops::ConvOptions::new_with_padding(
            [stride],
            [(padding, padding)],
            [1],
            groups,
        );
        burn::tensor::module::conv1d(x, weight.clone(), Some(bias.clone()), options)
    }

    /// Run the network on mel features `[n_frames][N_BINS]` and return the
    /// sigmoid latent pitch-class distribution as `[n_frames][OUT_DIMS]`.
    pub fn forward(&self, features: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let device = self.in_w.device();
        let n_frames = features.len();
        let flat: Vec<f32> = features.iter().flatten().copied().collect();
        let x = Tensor::<3>::from_data(TensorData::new(flat, [1, n_frames, N_BINS]), &device);

        // input stack: (B, T, C) -> (B, C, T) for conv1d
        let x = x.swap_dims(1, 2);
        let x = Self::conv1d(x, &self.in_w, &self.in_b, 1, 1);

        // GroupNorm(4, 512): normalize each group of 128 channels over
        // all of its (channel, time) values
        let mut group_normed: Vec<Tensor<3>> = Vec::with_capacity(GROUPS);
        let group_ch = HIDDEN / GROUPS;
        let time = x.dims()[2];
        for g in 0..GROUPS {
            let slice = x
                .clone()
                .slice([0..1, g * group_ch..(g + 1) * group_ch, 0..time]);
            let mean = slice.clone().mean().reshape([1, 1, 1]);
            let centered = slice.sub(mean);
            let var = centered.clone().powf_scalar(2.0).mean().reshape([1, 1, 1]);
            group_normed.push(centered.div(var.add_scalar(EPS).sqrt()));
        }
        let mut x = group_normed.remove(0);
        for part in group_normed {
            x = Tensor::cat(vec![x, part], 1);
        }
        let gn_w = self
            .gn_w
            .clone()
            .reshape([1, HIDDEN, 1])
            .expand([1, HIDDEN, time]);
        let gn_b = self
            .gn_b
            .clone()
            .reshape([1, HIDDEN, 1])
            .expand([1, HIDDEN, time]);
        let x = x.mul(gn_w).add(gn_b);

        let x = leaky_relu(x, 0.01);
        let x = Self::conv1d(x, &self.in2_w, &self.in2_b, 1, 1);
        let mut x = x.swap_dims(1, 2); // (1, T, 512)

        for layer in &self.layers {
            let y = Self::layer_norm(x.clone(), &layer.norm_w, &layer.norm_b);
            let y = y.swap_dims(1, 2); // (1, 512, T)
            let y = Self::conv1d(y, &layer.pw_w, &layer.pw_b, 0, 1);
            // GLU along the channel dim: first half * sigmoid(second half)
            let t = y.dims()[2];
            let a = y.clone().slice([0..1, 0..INNER, 0..t]);
            let b = y.slice([0..1, INNER..INNER * 2, 0..t]);
            let y = a.mul(sigmoid(b));
            let y = Self::conv1d(y, &layer.dw_w, &layer.dw_b, KERNEL / 2, INNER);
            let y = silu(y);
            let y = Self::conv1d(y, &layer.out_w, &layer.out_b, 0, 1);
            let y = y.swap_dims(1, 2); // (1, T, 512)
            x = x.add(y);
        }

        let x = Self::layer_norm(x, &self.norm_w, &self.norm_b);
        let t = x.dims()[1];
        let bias = self
            .out_b
            .clone()
            .reshape([1, 1, OUT_DIMS])
            .expand([1, t, OUT_DIMS]);
        let logits = x.matmul(self.out_w.clone().unsqueeze_dim::<3>(0)).add(bias);
        let latent = sigmoid(logits);
        let data = latent.into_data();
        let values: Vec<f32> = data.try_to_vec::<f32>().expect("latent tensor is f32");
        values
            .chunks(OUT_DIMS)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
}

/// Decode the sigmoid latent into per-frame f0 estimates using the
/// `local_argmax` decoder: a confidence-weighted centroid of the cent table
/// over the 9 bins surrounding the argmax.
///
/// Returns `(frequency_hz, confidence)` per frame; frequency is `None` for
/// frames whose peak confidence is at or below `threshold`.
pub fn decode_latent(latent: &[Vec<f32>], threshold: f32) -> Vec<(Option<f32>, f32)> {
    let table = cent_table();
    latent
        .iter()
        .map(|frame| {
            let (max_idx, &conf) = frame
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            if conf <= threshold {
                return (None, conf);
            }
            let lo = max_idx.saturating_sub(4);
            let hi = (max_idx + 5).min(OUT_DIMS);
            let mut weighted = 0.0_f32;
            let mut total = 0.0_f32;
            for i in lo..hi {
                weighted += table[i] * frame[i];
                total += frame[i];
            }
            let cents = if total > 0.0 { weighted / total } else { 0.0 };
            let f0 = 10.0 * 2.0_f32.powf(cents / 1200.0);
            (Some(f0), conf)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cent_table_endpoints() {
        let table = cent_table();
        assert_eq!(table.len(), OUT_DIMS);
        let start = 1200.0 * (F0_MIN / 10.0).log2();
        let end = 1200.0 * (F0_MAX / 10.0).log2();
        assert!((table[0] - start).abs() < 1e-4);
        assert!((table[OUT_DIMS - 1] - end).abs() < 1e-4);
    }

    #[test]
    fn decode_synthetic_peak() {
        // single peak at bin corresponding to ~440 Hz
        let table = cent_table();
        let target = 1200.0 * (440.0_f32 / 10.0).log2();
        let idx = table
            .iter()
            .enumerate()
            .min_by(|a, b| {
                (a.1 - target)
                    .abs()
                    .partial_cmp(&(b.1 - target).abs())
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap();
        let mut frame = vec![0.0_f32; OUT_DIMS];
        frame[idx] = 0.9;
        if idx > 0 {
            frame[idx - 1] = 0.3;
        }
        if idx + 1 < OUT_DIMS {
            frame[idx + 1] = 0.3;
        }
        let out = decode_latent(&[frame], 0.05);
        let (f0, conf) = out[0];
        assert!(conf > 0.05);
        let f0 = f0.unwrap();
        assert!((f0 - 440.0).abs() / 440.0 < 0.01, "f0 = {f0}");
    }

    #[test]
    fn decode_unvoiced_below_threshold() {
        let frame = vec![0.01_f32; OUT_DIMS];
        let out = decode_latent(&[frame], 0.05);
        assert_eq!(out[0].0, None);
    }
}
