//! PC-NSF-HiFiGAN neural vocoder (Burn reimplementation, from scratch).
//!
//! Pitch-controllable NSF-HiFiGAN vocoder from the openvpi/vocoders release
//! `pc-nsf-hifigan-44.1k-hop512-128bin-2025.02` (config: sampling_rate 44100,
//! hop 512, 128 mel bins, fmin 40, fmax 16000, upsample rates [8, 8, 2, 2, 2],
//! upsample kernels [16, 16, 4, 4, 4], resblock "1" with kernel sizes [3, 7, 11]
//! and dilations [[1, 3, 5]; 3], upsample_initial_channel 512, `mini_nsf: true`,
//! `pc_aug: true`).
//!
//! Architecture (derived from the upstream generator and the exported tensor
//! shapes):
//!
//! ```text
//! mel [1, 128, T]
//!   -> conv_pre: Conv1d(128 -> 512, k = 7, pad 3)
//!   -> for i, (rate, kernel) in [(8, 16), (8, 16), (2, 4), (2, 4), (2, 4)]:
//!        LeakyReLU(0.1)
//!        ConvTranspose1d(ch -> ch/2, kernel, stride = rate, pad (k - r) / 2)
//!        if i == 1: x += source_conv( harmonic excitation )   // mini-NSF
//!        x = mean over j in 0..3 of ResBlock1_j(x)            // MRF
//!   -> LeakyReLU(0.1)
//!   -> conv_post: Conv1d(16 -> 1, k = 7, pad 3)
//!   -> tanh
//! ```
//!
//! The mini-NSF excitation (`fastsinegen`) turns the per-mel-frame f0 contour
//! into a sinusoid at `source_sr = 44100 / (2 * 2 * 2) = 5512.5 Hz`
//! (`upp = 8 * 8 = 64` samples per mel frame), with linear-within-frame f0
//! interpolation and a wrapped cumulative phase-accumulation term, exactly as
//! the upstream `Generator.fastsinegen`.
//!
//! All convolutions carry weight normalization in the checkpoint as
//! `weight_v`/`weight_g` pairs; the weight is materialized at load time as
//! `g * v / ||v||` over all dimensions except the first (torch weight_norm
//! `dim = 0` semantics for both Conv1d and ConvTranspose1d).

use anyhow::{Context, Result, anyhow, bail};
use burn::tensor::TensorData;
use burn::tensor::activation::{leaky_relu, tanh};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, ops::ConvTransposeOptions};
use burn_store::{BurnpackStore, ModuleStore};
use std::path::Path;

use crate::vmel::N_MELS;

const INITIAL_CHANNEL: usize = 512;
const NUM_KERNELS: usize = 3;
const LRELU_SLOPE: f32 = 0.1;
const UPSAMPLE_RATES: [usize; 5] = [8, 8, 2, 2, 2];
const UPSAMPLE_KERNELS: [usize; 5] = [16, 16, 4, 4, 4];
const RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
const RESBLOCK_DILATIONS: [[usize; 3]; 3] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];
/// Sampling rate of the harmonic excitation: 44100 / prod(rates[2..]).
const SOURCE_SR: f32 = 44100.0 / 8.0;
/// Excitation samples per mel frame: prod(rates[..2]).
const UPP: usize = 64;
/// Audio samples per mel frame at 44.1 kHz.
const SAMPLES_PER_FRAME: usize = 512;
/// Chunking for long utterances (mel frames).
const CHUNK_FRAMES: usize = 400;
const CHUNK_CONTEXT: usize = 40;
const CROSSFADE_SAMPLES: usize = 512;

struct Conv1dW<B: Backend> {
    w: Tensor<B, 3>,
    b: Tensor<B, 1>,
}

struct ResBlockW<B: Backend> {
    convs1: [Conv1dW<B>; 3],
    convs2: [Conv1dW<B>; 3],
    dilations: [usize; 3],
    kernel_size: usize,
}

/// Loaded PC-NSF-HiFiGAN generator parameters.
pub struct PcNsfHifiGanWeights<B: Backend> {
    conv_pre: Conv1dW<B>,
    ups: Vec<Conv1dW<B>>,
    resblocks: Vec<ResBlockW<B>>,
    source_conv: Conv1dW<B>,
    conv_post: Conv1dW<B>,
}

/// PC-NSF-HiFiGAN vocoder.
pub struct PcNsfHifiGan<B: Backend = crate::B> {
    weights: PcNsfHifiGanWeights<B>,
}

/// Materialize a torch weight_norm pair: `w = g * v / ||v||` per dim-0 slice.
///
/// `v` is row-major `[d0, d1, d2]`, `g` is `[d0, 1, 1]`.
fn materialize_weight_norm(v: &[f32], g: &[f32], shape: [usize; 3]) -> Vec<f32> {
    let [d0, d1, d2] = shape;
    if v.len() != d0 * d1 * d2 || g.len() != d0 {
        return vec![0.0; d0 * d1 * d2];
    }
    let mut out = vec![0.0_f32; v.len()];
    for (i, &g_i) in g.iter().enumerate().take(d0) {
        let offset = i * d1 * d2;
        let norm = v[offset..offset + d1 * d2]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let scale = if norm > 0.0 { g_i / norm } else { 0.0 };
        for (dst, &src) in out[offset..offset + d1 * d2].iter_mut().zip(&v[offset..]) {
            *dst = src * scale;
        }
    }
    out
}

/// Mini-NSF harmonic excitation (upstream `Generator.fastsinegen`).
///
/// `f0` is one f0 value per mel frame (Hz; 0 = unvoiced). Returns
/// `f0.len() * upp` samples of sinusoidal excitation at `source_sr`, with
/// linear f0 interpolation inside each frame and wrapped cumulative
/// phase accumulation across frames.
pub fn fastsinegen(f0: &[f32], upp: usize, source_sr: f32) -> Vec<f32> {
    fn fmod1(x: f32) -> f32 {
        x - x.floor()
    }

    let mut out = Vec::with_capacity(f0.len() * upp);
    let mut acc = 0.0_f32;
    let frames = f0.len();
    for t in 0..frames {
        let s0 = f0[t] / source_sr;
        let ds0 = if t + 1 < frames {
            f0[t + 1] / source_sr - s0
        } else {
            0.0
        };
        let upp_f = upp as f32;
        let mut rad = vec![0.0_f32; upp];
        for (n_idx, slot) in rad.iter_mut().enumerate() {
            let n = (n_idx + 1) as f32;
            *slot = s0 * n + 0.5 * ds0 * n * (n - 1.0) / upp_f;
        }
        let rad2 = fmod1(rad[upp - 1] + 0.5) - 0.5;
        for slot in &mut rad {
            *slot += acc;
        }
        out.extend(rad.iter().map(|&r| (2.0 * std::f32::consts::PI * r).sin()));
        acc = fmod1(acc + rad2);
    }
    out
}

impl<B: Backend> PcNsfHifiGanWeights<B> {
    /// Load generator weights from a Burn Pack (`.bpk`) exported from the
    /// PC-NSF-HiFiGAN checkpoint. Tensors are looked up by their dotted
    /// state-dict name and weight-norm pairs are materialized at load time.
    pub fn from_burnpack(path: &Path, device: &B::Device) -> Result<Self> {
        use burn::tensor::DType;

        let snapshots = BurnpackStore::from_file(path)
            .zero_copy(true)
            .get_all_snapshots()
            .with_context(|| format!("failed to read burnpack {}", path.display()))?
            .clone();

        let store = move |name: &str| -> Option<(Vec<f32>, Vec<usize>)> {
            snapshots.values().find_map(|snap| {
                if snap.full_path() == name {
                    let data = snap.to_data().ok()?;
                    if data.dtype == DType::F32 {
                        let shape = data.shape.to_vec();
                        let values: Vec<f32> = data.to_vec::<f32>().ok()?;
                        Some((values, shape))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
        };

        let get = |name: &str, shape: &[usize]| -> Result<Vec<f32>> {
            let (data, actual) = store(name).ok_or_else(|| anyhow!("missing tensor {name}"))?;
            if actual != shape {
                bail!("tensor {name} has shape {actual:?}, expected {shape:?}");
            }
            Ok(data)
        };

        // Conv1d weight is [out, in, k] (bias [out]); ConvTranspose1d weight
        // is [in, out, k] (bias [out]). Weight norm is per dim-0 in both.
        let load_wn = |prefix: &str,
                       shape: [usize; 3],
                       bias_len: usize,
                       device: &B::Device|
         -> Result<Conv1dW<B>> {
            let v = get(&format!("{prefix}.weight_v"), &shape)?;
            let g = get(&format!("{prefix}.weight_g"), &[shape[0], 1, 1])?;
            let w = materialize_weight_norm(&v, &g, shape);
            Ok(Conv1dW {
                w: Tensor::from_data(TensorData::new(w, shape), device),
                b: tensor1(device, get(&format!("{prefix}.bias"), &[bias_len])?),
            })
        };

        let conv_pre = load_wn(
            "generator.conv_pre",
            [INITIAL_CHANNEL, N_MELS, 7],
            512,
            device,
        )?;

        let mut ups = Vec::with_capacity(UPSAMPLE_RATES.len());
        let mut resblocks = Vec::with_capacity(UPSAMPLE_RATES.len() * NUM_KERNELS);
        let mut ch = INITIAL_CHANNEL;
        for (i, (&_rate, &kernel)) in UPSAMPLE_RATES.iter().zip(&UPSAMPLE_KERNELS).enumerate() {
            ch /= 2;
            // ConvTranspose1d weight is [in, out, k]
            ups.push(load_wn(
                &format!("generator.ups.{i}"),
                [2 * ch, ch, kernel],
                ch,
                device,
            )?);
            for (j, (&rk, &dil)) in RESBLOCK_KERNELS.iter().zip(&RESBLOCK_DILATIONS).enumerate() {
                let idx = i * NUM_KERNELS + j;
                let mut convs1 = Vec::with_capacity(3);
                let mut convs2 = Vec::with_capacity(3);
                for l in 0..3 {
                    convs1.push(load_wn(
                        &format!("generator.resblocks.{idx}.convs1.{l}"),
                        [ch, ch, rk],
                        ch,
                        device,
                    )?);
                    convs2.push(load_wn(
                        &format!("generator.resblocks.{idx}.convs2.{l}"),
                        [ch, ch, rk],
                        ch,
                        device,
                    )?);
                }
                resblocks.push(ResBlockW {
                    convs1: [convs1.remove(0), convs1.remove(0), convs1.remove(0)],
                    convs2: [convs2.remove(0), convs2.remove(0), convs2.remove(0)],
                    dilations: dil,
                    kernel_size: rk,
                });
            }
        }

        let source_conv = Conv1dW {
            w: tensor3(
                device,
                get("generator.source_conv.weight", &[128, 1, 1])?,
                128,
                1,
                1,
            ),
            b: tensor1(device, get("generator.source_conv.bias", &[128])?),
        };
        let conv_post = load_wn("generator.conv_post", [1, 16, 7], 1, device)?;

        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            source_conv,
            conv_post,
        })
    }
}

fn tensor1<B: Backend>(device: &B::Device, data: Vec<f32>) -> Tensor<B, 1> {
    let len = data.len();
    Tensor::from_data(TensorData::new(data, [len]), device)
}

fn tensor3<B: Backend>(
    device: &B::Device,
    data: Vec<f32>,
    d0: usize,
    d1: usize,
    d2: usize,
) -> Tensor<B, 3> {
    Tensor::from_data(TensorData::new(data, [d0, d1, d2]), device)
}

/// hifigan get_padding: `(kernel * dilation - dilation) / 2`.
fn conv_padding(kernel_size: usize, dilation: usize) -> usize {
    (kernel_size * dilation - dilation) / 2
}

impl<B: Backend> PcNsfHifiGan<B> {
    pub fn new(weights: PcNsfHifiGanWeights<B>) -> Self {
        Self { weights }
    }

    fn conv1d(x: Tensor<B, 3>, conv: &Conv1dW<B>, padding: usize, dilation: usize) -> Tensor<B, 3> {
        let options = burn::tensor::ops::PaddedConvOptions::asymmetric(
            [1],
            [padding],
            [padding],
            [dilation],
            1,
        );
        burn::tensor::module::conv1d(x, conv.w.clone(), Some(conv.b.clone()), options)
    }

    fn resblock(x: Tensor<B, 3>, block: &ResBlockW<B>) -> Tensor<B, 3> {
        let mut x = x;
        for (l, (c1, c2)) in block.convs1.iter().zip(&block.convs2).enumerate() {
            let d = block.dilations[l];
            let xt = Self::conv1d(
                leaky_relu(x.clone(), f64::from(LRELU_SLOPE)),
                c1,
                conv_padding(block.kernel_size, d),
                d,
            );
            let xt = Self::conv1d(
                leaky_relu(xt, f64::from(LRELU_SLOPE)),
                c2,
                conv_padding(block.kernel_size, 1),
                1,
            );
            x = x.add(xt);
        }
        x
    }

    /// Run the generator on one mel chunk.
    ///
    /// `mel` is `[T][N_MELS]`, `har` is the excitation for exactly `T` mel
    /// frames (`T * UPP` samples). Returns `T * SAMPLES_PER_FRAME` samples.
    fn forward(&self, mel: &[Vec<f32>], har: &[f32]) -> Vec<f32> {
        let device = self.weights.conv_pre.w.device();
        let frames = mel.len();
        let flat: Vec<f32> = mel.iter().flatten().copied().collect();
        let mut x = Tensor::<B, 3>::from_data(TensorData::new(flat, [1, frames, N_MELS]), &device)
            .swap_dims(1, 2); // [1, 128, T]

        x = Self::conv1d(x, &self.weights.conv_pre, 3, 1);

        let har_t =
            Tensor::<B, 3>::from_data(TensorData::new(har.to_vec(), [1, 1, har.len()]), &device);

        for (i, up) in self.weights.ups.iter().enumerate() {
            x = leaky_relu(x, f64::from(LRELU_SLOPE));
            let rate = UPSAMPLE_RATES[i];
            let kernel = UPSAMPLE_KERNELS[i];
            let options = ConvTransposeOptions::new([rate], [(kernel - rate) / 2], [0], [1], 1);
            x = burn::tensor::module::conv_transpose1d(
                x,
                up.w.clone(),
                Some(up.b.clone()),
                options,
            );
            if i == 1 {
                let src = Self::conv1d(har_t.clone(), &self.weights.source_conv, 0, 1);
                x = x.add(src);
            }
            let mut xs: Option<Tensor<B, 3>> = None;
            for j in 0..NUM_KERNELS {
                let block = &self.weights.resblocks[i * NUM_KERNELS + j];
                let out = Self::resblock(x.clone(), block);
                xs = Some(match xs {
                    Some(acc) => acc.add(out),
                    None => out,
                });
            }
            if let Some(acc) = xs {
                x = acc.div_scalar(NUM_KERNELS as f32);
            }
        }

        x = leaky_relu(x, f64::from(LRELU_SLOPE));
        x = Self::conv1d(x, &self.weights.conv_post, 3, 1);
        x = tanh(x);

        x.into_data().to_vec::<f32>().unwrap_or_default()
    }

    /// Synthesize audio from log-mel features and an f0 contour.
    ///
    /// * `mel` — `[T][128]` log-mel features (see `vmel`).
    /// * `f0_hz` — one f0 value per mel frame in Hz (0 = unvoiced).
    ///
    /// Returns mono 44.1 kHz f32 PCM with `T * 512` samples (long inputs are
    /// processed in overlapping chunks crossfaded at the seams).
    pub fn synthesize(&self, mel: &[Vec<f32>], f0_hz: &[f32]) -> Vec<f32> {
        assert_eq!(
            mel.len(),
            f0_hz.len(),
            "mel and f0 must have the same number of frames"
        );
        assert!(!mel.is_empty(), "mel must not be empty");
        for row in mel {
            assert_eq!(row.len(), N_MELS, "mel rows must have {} bins", N_MELS);
        }

        let har = fastsinegen(f0_hz, UPP, SOURCE_SR);
        if mel.len() <= CHUNK_FRAMES + 2 * CHUNK_CONTEXT {
            return self.forward(mel, &har);
        }

        let mut out: Vec<f32> = Vec::new();
        let mut core_start = 0usize;
        while core_start < mel.len() {
            let core_end = (core_start + CHUNK_FRAMES).min(mel.len());
            let win_start = core_start.saturating_sub(CHUNK_CONTEXT);
            let win_end = (core_end + CHUNK_CONTEXT).min(mel.len());
            let y = self.forward(
                &mel[win_start..win_end],
                &har[win_start * UPP..win_end * UPP],
            );
            let offset = (core_start - win_start) * SAMPLES_PER_FRAME;
            let core_len = (core_end - core_start) * SAMPLES_PER_FRAME;
            let core_audio = &y[offset..offset + core_len];

            if out.len() >= CROSSFADE_SAMPLES && core_audio.len() >= CROSSFADE_SAMPLES {
                let n = CROSSFADE_SAMPLES;
                for (i, &sample) in core_audio.iter().take(n).enumerate() {
                    let t = (i + 1) as f32 / (n + 1) as f32;
                    let tail = out.len() - n + i;
                    out[tail] = out[tail] * (1.0 - t) + sample * t;
                }
                out.extend_from_slice(&core_audio[n..]);
            } else {
                out.extend_from_slice(core_audio);
            }
            core_start = core_end;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_norm_materializes_per_dim0() {
        // v = identity-ish [2, 1, 2], g = [3, 4]
        let v = vec![3.0f32, 4.0, 0.0, 5.0]; // norms: 5, 5
        let g = vec![10.0f32, 20.0];
        let w = materialize_weight_norm(&v, &g, [2, 1, 2]);
        // w[0] = v[0..2] * (10/5) = [6, 8]; w[1] = v[2..4] * (20/5) = [0, 20]
        assert!((w[0] - 6.0).abs() < 1e-5);
        assert!((w[1] - 8.0).abs() < 1e-5);
        assert!((w[2] - 0.0).abs() < 1e-6);
        assert!((w[3] - 20.0).abs() < 1e-5);
    }

    #[test]
    fn excitation_has_expected_period_for_constant_f0() {
        // 1 second of mel frames at ~86.13 fps (hop 512 @ 44.1k)
        let frames = 86;
        let f0 = vec![220.0f32; frames];
        let har = fastsinegen(&f0, UPP, SOURCE_SR);
        assert_eq!(har.len(), frames * UPP);
        // 220 Hz at 5512.5 Hz -> ~25.06 samples per period; count cycles
        let mut crossings = 0usize;
        for w in har.windows(2) {
            if w[0] <= 0.0 && w[1] > 0.0 {
                crossings += 1;
            }
        }
        // positive-going zero crossings == cycles (sine starts at sin(0) = 0)
        let seconds = har.len() as f32 / SOURCE_SR;
        let measured = crossings as f32 / seconds;
        assert!(
            (measured - 220.0).abs() < 2.0,
            "measured {measured} Hz, expected ~220"
        );
        assert!(har.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    }

    #[test]
    fn excitation_unvoiced_is_silent() {
        let har = fastsinegen(&[0.0f32; 10], UPP, SOURCE_SR);
        assert!(har.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn excitation_follows_glide() {
        // linear glide 220 -> 440 over ~1 s; average f0 ~330
        let frames: Vec<f32> = (0..86).map(|i| 220.0 + 220.0 * i as f32 / 86.0).collect();
        let har = fastsinegen(&frames, UPP, SOURCE_SR);
        let mut crossings = 0usize;
        for w in har.windows(2) {
            if w[0] <= 0.0 && w[1] > 0.0 {
                crossings += 1;
            }
        }
        let seconds = har.len() as f32 / SOURCE_SR;
        let measured = crossings as f32 / seconds;
        assert!(
            (measured - 330.0).abs() < 25.0,
            "measured {measured} Hz, expected ~330"
        );
    }

    #[test]
    fn conv_padding_matches_hifigan() {
        assert_eq!(conv_padding(3, 1), 1);
        assert_eq!(conv_padding(3, 3), 3);
        assert_eq!(conv_padding(3, 5), 5);
        assert_eq!(conv_padding(7, 3), 9);
        assert_eq!(conv_padding(11, 5), 25);
    }
}
