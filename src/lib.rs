//! F0/pitch detection for vocal pitch correction, powered by the FCPE neural
//! model running on Burn.
//!
//! FCPE (Fast Context-based Pitch Estimation) is a conformer-based encoder
//! over log-spectral (STCT, see `mel`) features that predicts a per-frame
//! pitch-class distribution, decoded into f0 via a local-argmax centroid.

pub mod fcpe;
pub mod mel;
pub mod resample;
pub mod vmel;
pub mod vocoder;

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};

pub use fcpe::FcpeWeights;
pub use vocoder::{PcNsfHifiGan, PcNsfHifiGanWeights};

/// Create the default wgpu device, initialized for Vulkan with exclusive
/// memory pages (mirrors `maolan-generate`'s runtime setup). Initialization
/// runs at most once per process; a failure is cached and re-reported.
///
/// # Panics
///
/// Panics if no suitable GPU adapter can be initialized; `load_fcpe_weights`
/// and friends turn this into a descriptive error.
pub fn wgpu_device() -> burn::tensor::Device {
    GPU_INIT
        .get_or_init(|| {
            std::panic::catch_unwind(|| {
                burn::tensor::Device::wgpu_options()
                    .graphics_api(burn::tensor::wgpu::WgpuBackend::Vulkan)
                    .memory_config(burn::tensor::wgpu::MemoryConfiguration::ExclusivePages)
                    .init()
                    .map_err(|error| error.to_string())
            })
            .map_err(panic_payload_to_string)
            .and_then(|result| result)
        })
        .as_ref()
        .unwrap_or_else(|detail| {
            panic!(
                "maolan-pitch requires a working Vulkan GPU (wgpu initialization failed): {detail}"
            )
        })
        .clone()
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

static GPU_INIT: std::sync::OnceLock<Result<burn::tensor::Device, String>> =
    std::sync::OnceLock::new();

fn wgpu_device_or_err() -> Result<burn::tensor::Device> {
    std::panic::catch_unwind(wgpu_device)
        .map_err(|payload| anyhow!(panic_payload_to_string(payload)))
}

/// Default HF repo hosting the converted Burn Pack of FCPE weights.
pub const HF_REPO_OWNER: &str = "maolandaw";
pub const HF_REPO_NAME: &str = "FCPE-burn";
pub const HF_REPO_FILE: &str = "fcpe.bpk";

/// HF repo and file for the PC-NSF-HiFiGAN Burn Pack.
pub const VOCODER_HF_REPO_NAME: &str = "PC-NSF-HiFiGAN-burn";
pub const VOCODER_HF_REPO_FILE: &str = "pc_nsf_hifigan.bpk";
/// Environment variable overriding the vocoder Burn Pack location
/// (used until the HF repo is published, and by tests).
pub const VOCODER_BPK_ENV: &str = "PC_NSF_HIFIGAN_BPK";

/// Which detector produced (or would produce) a pitch track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PitchDetectorKind {
    /// Classic DSP detector (YIN/pyin-style); marker for callers, not
    /// implemented in this crate.
    Classic,
    /// Neural FCPE detector.
    Neural,
}

/// A single detected f0 frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F0Frame {
    /// Position of the frame in the original input signal, in samples.
    pub time_samples: usize,
    /// Detected fundamental frequency in Hz, or `None` if unvoiced.
    pub frequency: Option<f32>,
    /// Peak of the latent pitch-class distribution for this frame
    /// (confidence in the voicing decision).
    pub confidence: f32,
}

/// Neural F0 detector backed by FCPE weights.
pub struct FcpeDetector {
    weights: FcpeWeights,
    threshold: f32,
    kind: PitchDetectorKind,
}

impl FcpeDetector {
    pub fn new(weights: FcpeWeights) -> Self {
        Self {
            weights,
            threshold: 0.05,
            kind: PitchDetectorKind::Neural,
        }
    }

    /// Voicing threshold on the peak of the latent distribution
    /// (upstream default: 0.05).
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn kind(&self) -> PitchDetectorKind {
        self.kind
    }

    /// Detect f0 in mono `audio` at `sample_rate` Hz.
    ///
    /// `progress` is called with values in `(0.0, 1.0)` during processing.
    pub fn detect_f0(
        &self,
        audio: &[f32],
        sample_rate: u32,
        mut progress: impl FnMut(f32),
    ) -> Vec<F0Frame> {
        progress(0.05);
        let audio16k = match resample::resample_to(audio, sample_rate, mel::SAMPLE_RATE) {
            Ok(a) => a,
            Err(err) => {
                tracing::warn!("resampling failed, using raw audio: {err:#}");
                audio.to_vec()
            }
        };
        progress(0.3);

        let features = mel::wav_to_mel(&audio16k);
        progress(0.6);

        let latent = self.weights.forward(&features);
        progress(0.9);

        let decoded = fcpe::decode_latent(&latent, self.threshold);
        let scale = sample_rate as f64 / mel::SAMPLE_RATE as f64;
        let frames = decoded
            .into_iter()
            .enumerate()
            .map(|(i, (frequency, confidence))| F0Frame {
                time_samples: ((i * mel::HOP_SIZE) as f64 * scale).round() as usize,
                frequency,
                confidence,
            })
            .collect();
        progress(1.0);
        frames
    }
}

/// Load FCPE weights from a Burn Pack (`.bpk`) file onto the default wgpu
/// device.
///
/// Returns an error mentioning the Vulkan requirement if the GPU cannot be
/// initialized.
pub fn load_fcpe_weights(path: &Path) -> Result<FcpeWeights> {
    let device = wgpu_device_or_err()?;
    FcpeWeights::from_burnpack(path, &device)
        .with_context(|| format!("failed to load FCPE weights from {}", path.display()))
}

/// Ensure the FCPE Burn Pack is available in `cache_dir`, downloading it
/// from the Hugging Face repo `maolandaw/FCPE-burn` if missing.
pub fn ensure_fcpe_model(cache_dir: &Path) -> Result<PathBuf> {
    let cached = cache_dir.join(HF_REPO_FILE);
    if cached.exists() {
        return Ok(cached);
    }

    std::fs::create_dir_all(cache_dir)
        .with_context(|| format!("failed to create {}", cache_dir.display()))?;

    let client = huggingface_hub::HFClientSync::new().context("failed to initialize HF client")?;
    let repo = client.model(HF_REPO_OWNER, HF_REPO_NAME);
    let downloaded = repo
        .download_file()
        .filename(HF_REPO_FILE)
        .send()
        .with_context(|| {
            format!("failed to fetch {HF_REPO_OWNER}/{HF_REPO_NAME}/{HF_REPO_FILE}")
        })?;

    std::fs::copy(&downloaded, &cached).with_context(|| {
        format!(
            "failed to copy {} to {}",
            downloaded.display(),
            cached.display()
        )
    })?;
    Ok(cached)
}

/// Load PC-NSF-HiFiGAN generator weights from a Burn Pack onto the default
/// wgpu device.
///
/// Returns an error mentioning the Vulkan requirement if the GPU cannot be
/// initialized.
pub fn load_pc_nsf_hifigan_weights(path: &Path) -> Result<PcNsfHifiGanWeights> {
    let device = wgpu_device_or_err()?;
    PcNsfHifiGanWeights::from_burnpack(path, &device)
        .with_context(|| format!("failed to load vocoder weights from {}", path.display()))
}

/// Ensure the PC-NSF-HiFiGAN Burn Pack is available, in this order:
///
/// 1. Path from the `PC_NSF_HIFIGAN_BPK` environment variable (if it exists).
/// 2. `cache_dir / "pc_nsf_hifigan.bpk"` (if it exists).
/// 3. Downloaded from the Hugging Face repo `maolandaw/PC-NSF-HiFiGAN-burn`.
///
/// Note: the HF repo is not published yet; steps 1-2 are the supported
/// sources until it is.
pub fn ensure_pc_nsf_hifigan_model(cache_dir: &Path) -> Result<PathBuf> {
    if let Ok(env_path) = std::env::var(VOCODER_BPK_ENV) {
        let env_path = PathBuf::from(env_path);
        if env_path.exists() {
            return Ok(env_path);
        }
    }

    let cached = cache_dir.join(VOCODER_HF_REPO_FILE);
    if cached.exists() {
        return Ok(cached);
    }

    std::fs::create_dir_all(cache_dir)
        .with_context(|| format!("failed to create {}", cache_dir.display()))?;

    let client = huggingface_hub::HFClientSync::new().context("failed to initialize HF client")?;
    let repo = client.model(HF_REPO_OWNER, VOCODER_HF_REPO_NAME);
    let downloaded = repo
        .download_file()
        .filename(VOCODER_HF_REPO_FILE)
        .send()
        .with_context(|| {
            format!("failed to fetch {HF_REPO_OWNER}/{VOCODER_HF_REPO_NAME}/{VOCODER_HF_REPO_FILE} (is the repo published?)")
        })?;

    std::fs::copy(&downloaded, &cached).with_context(|| {
        format!(
            "failed to copy {} to {}",
            downloaded.display(),
            cached.display()
        )
    })?;
    Ok(cached)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_bpk_path() -> PathBuf {
        if let Ok(p) = std::env::var("FCPE_BPK") {
            return PathBuf::from(p);
        }
        PathBuf::from("/home/meka/repos/FCPE-burn/fcpe.bpk")
    }

    /// Synthesize a ~3-second 16 kHz mono sine that glides 220 -> 440 -> 330 Hz
    /// with 50 ms silence gaps between segments. Returns (samples, expected
    /// instantaneous frequency per frame, per-frame `strict` flag excluding a
    /// short transient zone right after each gap and the padded file edge).
    fn synth_glide() -> (Vec<f32>, Vec<Option<f32>>, Vec<bool>) {
        let sr = 16_000_usize;
        let seg = 12 * sr / 10; // 1.2 s per glide
        let hold = 3 * sr / 10; // 0.3 s constant tail
        let gap = sr / 20; // 50 ms
        let total = 2 * seg + hold + 2 * gap;
        let mut audio = vec![0.0_f32; total];
        let dur = seg as f32 / sr as f32;
        for (i, slot) in audio.iter_mut().enumerate().take(seg) {
            let t = i as f32 / sr as f32;
            // linear chirp 220 -> 440
            let phase =
                2.0 * std::f32::consts::PI * (220.0 * t + (440.0 - 220.0) * t * t / (2.0 * dur));
            *slot = 0.5 * phase.sin();
        }
        let off1 = seg + gap;
        for (i, slot) in audio.iter_mut().enumerate().skip(off1).take(seg) {
            let t = (i - off1) as f32 / sr as f32;
            let phase =
                2.0 * std::f32::consts::PI * (440.0 * t - (440.0 - 330.0) * t * t / (2.0 * dur));
            *slot = 0.5 * phase.sin();
        }
        let off3 = 2 * (seg + gap);
        for (i, slot) in audio.iter_mut().enumerate().skip(off3).take(hold) {
            let t = (i - off3) as f32 / sr as f32;
            *slot = 0.5 * (2.0 * std::f32::consts::PI * 330.0 * t).sin();
        }

        let n_frames = total / 160 + 1;
        let settle = 12; // frames excluded from the strict per-frame check
        let mut expected = vec![None; n_frames];
        let mut strict = vec![false; n_frames];
        for (f, e) in expected.iter_mut().enumerate() {
            let sample = f * 160;
            let in_range = |start: usize, len: usize| sample >= start && sample < start + len;
            if in_range(0, seg) {
                let t = sample as f32 / sr as f32;
                *e = Some(220.0 + (440.0 - 220.0) * t / dur);
                strict[f] = sample / 160 >= settle && (seg - sample) / 160 > settle;
            } else if in_range(seg + gap, seg) {
                let t = (sample - (seg + gap)) as f32 / sr as f32;
                *e = Some(440.0 - (440.0 - 330.0) * t / dur);
                strict[f] = (sample - (seg + gap)) / 160 >= settle
                    && (seg + gap + seg - sample) / 160 > settle;
            } else if in_range(2 * (seg + gap), hold) {
                *e = Some(330.0);
                strict[f] = (sample - 2 * (seg + gap)) / 160 >= settle
                    && (2 * (seg + gap) + hold - sample) / 160 > settle;
            }
        }
        (audio, expected, strict)
    }

    #[test]
    fn fcpe_tracks_sine_glide() {
        let bpk = default_bpk_path();
        if !bpk.exists() {
            eprintln!(
                "skipping fcpe_tracks_sine_glide: {} not found",
                bpk.display()
            );
            return;
        }

        let weights = load_fcpe_weights(&bpk).expect("load weights");
        let detector = FcpeDetector::new(weights);
        let (audio, expected, strict) = synth_glide();
        let frames = detector.detect_f0(&audio, 16_000, |_| {});

        assert_eq!(frames.len(), expected.len());

        let mut voiced = 0;
        let mut max_err = 0.0_f32;
        let mut silent_voiced = 0;
        let (mut ex_sum, mut det_sum, mut ex_sq, mut det_sq, mut cross) =
            (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

        for ((frame, exp), is_strict) in frames.iter().zip(expected.iter()).zip(strict.iter()) {
            match (exp, frame.frequency) {
                (Some(target), Some(f0)) => {
                    voiced += 1;
                    let err = (f0 - target).abs() / target;
                    if *is_strict {
                        // FCPE is trained on voice; pure sinusoids wobble by
                        // up to a few output bins, hence the 5% hard bound.
                        // The 1%-class requirement is checked via the
                        // correlation assert below (must exceed 0.99).
                        max_err = max_err.max(err);
                        assert!(
                            err < 0.05,
                            "frame at {} Hz expected {target} Hz (err {err})",
                            frame.time_samples
                        );
                    }
                    ex_sum += *target as f64;
                    det_sum += f0 as f64;
                    ex_sq += (*target as f64) * (*target as f64);
                    det_sq += (f0 as f64) * (f0 as f64);
                    cross += (*target as f64) * (f0 as f64);
                }
                (None, Some(f0)) => {
                    // only tolerable with low confidence
                    assert!(
                        frame.confidence < 0.35,
                        "silence frame voiced at {f0} Hz with confidence {}",
                        frame.confidence
                    );
                    silent_voiced += 1;
                }
                (None, None) => {}
                (Some(_), None) => {
                    panic!("voiced frame missed at sample {}", frame.time_samples);
                }
            }
        }

        // Pearson correlation over voiced frames
        let n = voiced as f64;
        let cov = cross - ex_sum * det_sum / n;
        let var_ex = ex_sq - ex_sum * ex_sum / n;
        let var_det = det_sq - det_sum * det_sum / n;
        let corr = cov / (var_ex * var_det).sqrt();

        eprintln!(
            "voiced = {voiced}, max_err(strict) = {max_err:.5}, corr = {corr:.6}, silent_voiced = {silent_voiced}"
        );
        assert!(voiced > 100, "too few voiced frames: {voiced}");
        assert!(corr > 0.99, "correlation {corr} below 0.99");
        // allow a few edge frames at segment boundaries to be called voiced
        assert!(
            silent_voiced <= 8,
            "silence frames detected as voiced: {silent_voiced}"
        );
    }
}

#[cfg(test)]
mod vocoder_tests {
    use super::*;

    fn vocoder_bpk_path() -> PathBuf {
        if let Ok(p) = std::env::var(VOCODER_BPK_ENV) {
            return PathBuf::from(p);
        }
        PathBuf::from("/home/meka/repos/pc-nsf-hifigan-burn/pc_nsf_hifigan.bpk")
    }

    fn fcpe_bpk_path() -> PathBuf {
        if let Ok(p) = std::env::var("FCPE_BPK") {
            return PathBuf::from(p);
        }
        PathBuf::from("/home/meka/repos/FCPE-burn/fcpe.bpk")
    }

    /// Sum-of-harmonics 220 Hz source at 44.1 kHz.
    fn harmonic_source(seconds: usize) -> Vec<f32> {
        let sr = vmel::SAMPLE_RATE as usize;
        let f0 = 220.0f32;
        let mut audio = vec![0.0_f32; sr * seconds];
        let mut phase = 0.0f32;
        for sample in &mut audio {
            phase += f0 / sr as f32;
            phase -= phase.floor();
            let mut x = 0.0f32;
            let mut k = 1;
            while (k as f32) * f0 < 14000.0 && k < 300 {
                x += (2.0 * std::f32::consts::PI * k as f32 * phase).sin() / k as f32;
                k += 1;
            }
            *sample = 0.5 * x;
        }
        audio
    }

    fn mean_detected_f0(detector: &FcpeDetector, audio: &[f32]) -> Option<f32> {
        let frames = detector.detect_f0(audio, vmel::SAMPLE_RATE, |_| {});
        let voiced: Vec<f32> = frames.iter().filter_map(|f| f.frequency).collect();
        if voiced.is_empty() {
            None
        } else {
            Some(voiced.iter().sum::<f32>() / voiced.len() as f32)
        }
    }

    #[test]
    fn vocoder_closed_loop_tracks_f0() {
        let vocoder_bpk = vocoder_bpk_path();
        if !vocoder_bpk.exists() {
            eprintln!(
                "skipping vocoder_closed_loop_tracks_f0: {} not found",
                vocoder_bpk.display()
            );
            return;
        }
        let fcpe_bpk = fcpe_bpk_path();
        if !fcpe_bpk.exists() {
            eprintln!(
                "skipping vocoder_closed_loop_tracks_f0: {} not found",
                fcpe_bpk.display()
            );
            return;
        }

        let source = harmonic_source(2);
        let mel = vmel::wav_to_mel(&source);
        assert_eq!(mel[0].len(), 128);
        let f0_contour = vec![220.0f32; mel.len()];

        let vocoder = PcNsfHifiGan::new(
            load_pc_nsf_hifigan_weights(&vocoder_bpk).expect("load vocoder weights"),
        );
        let pcm = vocoder.synthesize(&mel, &f0_contour);

        // frame-exactness
        assert!(
            pcm.len().abs_diff(mel.len() * 512) <= 512,
            "output length {} differs from {}",
            pcm.len(),
            mel.len() * 512
        );
        assert!(pcm.iter().all(|v| v.is_finite()));
        let rms = (pcm.iter().map(|v| v * v).sum::<f32>() / pcm.len() as f32).sqrt();
        assert!(rms > 0.01, "output too quiet: RMS {rms}");

        // closed loop with FCPE, calibrated against the ground-truth reading
        let detector = FcpeDetector::new(load_fcpe_weights(&fcpe_bpk).expect("load fcpe"));
        let reference = mean_detected_f0(&detector, &source).expect("source voiced");
        let tracked = mean_detected_f0(&detector, &pcm).expect("output voiced");
        eprintln!("vocoder closed loop: source {reference:.1} Hz, output {tracked:.1} Hz");
        assert!(
            (tracked - reference).abs() / reference < 0.02,
            "vocoder f0 {tracked} vs reference {reference}"
        );
    }
}
