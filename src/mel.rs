//! Spectral frontend matching the FCPE `fcpe_c_v001` checkpoint's expected
//! input.
//!
//! Although the bundled `config_dict` advertises a librosa mel frontend
//! (num_mels = 128, fmin = 0, fmax = 8000), the checkpoint was in practice
//! trained on the **first 128 bins of a 640-point log-magnitude STCT**
//! (verified empirically: pure tones land on the predicted bins and are
//! decoded to the correct pitch only with this frontend). The parameters:
//!
//! - sample rate: 16000 Hz
//! - n_fft = win_size = 640 (25 Hz per bin), hop_size = 160
//! - hann window (periodic, torch.hann_window)
//! - center = false with manual reflect padding (240 left, 241 right)
//! - magnitude: `sqrt(re^2 + im^2 + 1e-9)`
//! - keep bins 0..128 (0..3200 Hz)
//! - log dynamic range compression with clip value 1e-5
//! - frame count adjusted to `n_samples // hop + 1` (repeat last frame)

use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

pub const SAMPLE_RATE: u32 = 16000;
pub const N_FFT: usize = 640;
pub const WIN_SIZE: usize = 640;
pub const HOP_SIZE: usize = 160;
pub const N_BINS: usize = 128;
pub const CLIP_VAL: f32 = 1e-5;

const SYMMETRIC_HANN: bool = false;

fn hann_window_periodic(size: usize) -> Vec<f32> {
    let denom = if SYMMETRIC_HANN { size - 1 } else { size };
    (0..size)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / denom as f32).cos())
        .collect()
}

/// torch.nn.functional.pad with mode = "reflect" on both ends.
/// Mirrors the signal without repeating the edge sample.
fn reflect_pad(signal: &[f32], left: usize, right: usize) -> Vec<f32> {
    let len = signal.len();
    let mut out = Vec::with_capacity(len + left + right);
    for i in 0..left {
        out.push(signal[left - i]);
    }
    out.extend_from_slice(signal);
    for k in 0..right {
        out.push(signal[len - 2 - k.min(len - 2)]);
    }
    out
}

/// Compute the FCPE input features for mono 16 kHz audio.
///
/// Returns row-major `[n_frames][N_BINS]` log-spectral features, where
/// `n_frames = audio.len() / HOP_SIZE + 1`.
pub fn wav_to_mel(audio: &[f32]) -> Vec<Vec<f32>> {
    let mut window = hann_window_periodic(WIN_SIZE);
    window.resize(N_FFT, 0.0);

    let pad_left = (WIN_SIZE - HOP_SIZE) / 2;
    let natural_right = (WIN_SIZE - HOP_SIZE).div_ceil(2);
    let needed_right =
        (WIN_SIZE as isize - audio.len() as isize - pad_left as isize).max(0) as usize;
    let pad_right = natural_right.max(needed_right);
    let padded = if pad_right < audio.len() {
        reflect_pad(audio, pad_left, pad_right)
    } else {
        // torch falls back to constant padding when the right pad is too large
        let mut out = vec![0.0_f32; audio.len() + pad_left + pad_right];
        out[pad_left..pad_left + audio.len()].copy_from_slice(audio);
        out
    };

    let n_frames = if padded.len() >= WIN_SIZE {
        (padded.len() - WIN_SIZE) / HOP_SIZE + 1
    } else {
        0
    };

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut frame = vec![Complex32::new(0.0, 0.0); N_FFT];

    let mut features = vec![vec![0.0_f32; N_BINS]; n_frames];
    for (t, feature_row) in features.iter_mut().enumerate() {
        let start = t * HOP_SIZE;
        for (i, bin) in frame.iter_mut().enumerate() {
            *bin = Complex32::new(padded[start + i] * window[i], 0.0);
        }
        fft.process(&mut frame);
        for (f, out) in feature_row.iter_mut().enumerate() {
            let re = frame[f].re;
            let im = frame[f].im;
            *out = (re * re + im * im + 1e-9).sqrt().max(CLIP_VAL).ln();
        }
    }

    // Match the upstream wrapper: frame count must be exactly len // hop + 1
    let target_frames = audio.len() / HOP_SIZE + 1;
    if target_frames > features.len() && !features.is_empty() {
        let last = features.last().cloned().unwrap();
        features.resize(target_frames, last);
    } else if target_frames < features.len() {
        features.truncate(target_frames);
    }

    features
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                0.5 * (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE as f32).sin()
            })
            .collect()
    }

    #[test]
    fn frame_count_matches_wrapper() {
        let mel = wav_to_mel(&sine(220.0, 16000));
        assert_eq!(mel.len(), 16000 / HOP_SIZE + 1);
        assert_eq!(mel[0].len(), N_BINS);
        // log-compressed values are negative for unit-scale signals
        assert!(mel[10][20] < 0.0);
    }

    #[test]
    fn spectral_peak_lands_on_expected_bin() {
        // 25 Hz per bin: a pure tone peaks at bin round(f / 25)
        for (freq, bin) in [(250.0f32, 10usize), (1000.0, 40), (2000.0, 80)] {
            let mel = wav_to_mel(&sine(freq, 16000));
            let (peak, _) = mel[50]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            assert!(
                (peak as i32 - bin as i32).abs() <= 1,
                "freq {freq}: peak bin {peak}, expected {bin}"
            );
        }
    }

    #[test]
    fn silence_is_low_energy() {
        let mel = wav_to_mel(&vec![0.0_f32; 16000]);
        // sqrt(1e-9) floor keeps values around -10.4 even for digital silence
        assert!(mel[50].iter().all(|&v| v < -9.0));
    }

    #[test]
    fn reflect_padding() {
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let padded = reflect_pad(&x, 2, 2);
        assert_eq!(padded, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }
}
