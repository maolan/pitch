//! Log-mel frontend for the PC-NSF-HiFiGAN vocoder (44.1 kHz).
//!
//! Matches the frontend the checkpoint was trained with (DiffSinger's
//! `nvSTFT.get_mel`, see `modules/nsf_hifigan/nvSTFT.py`):
//!
//! - sample rate: 44100 Hz
//! - n_fft = win_size = 2048, hop_size = 512
//! - hann window (periodic, torch.hann_window)
//! - center = false with manual reflect padding (768 left, 769 right)
//! - magnitude: `|STFT|` (no epsilon inside)
//! - 128 mel bins, Slaney scale (librosa defaults: htk = false,
//!   norm = "slaney"), fmin = 40, fmax = 16000 (from the release config.json)
//! - log dynamic range compression with clip value 1e-5 (natural log)

use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

pub const SAMPLE_RATE: u32 = 44100;
pub const N_FFT: usize = 2048;
pub const WIN_SIZE: usize = 2048;
pub const HOP_SIZE: usize = 512;
pub const N_MELS: usize = 128;
pub const FMIN: f32 = 40.0;
pub const FMAX: f32 = 16000.0;
pub const CLIP_VAL: f32 = 1e-5;

fn hz_to_mel(freqs: &[f32]) -> Vec<f32> {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0_f32;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f32.ln() / 27.0;
    freqs
        .iter()
        .map(|&f| {
            if f < min_log_hz {
                f / f_sp
            } else {
                min_log_mel + (f / min_log_hz).log10() / logstep
            }
        })
        .collect()
}

fn mel_to_hz(mels: &[f32]) -> Vec<f32> {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0_f32;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f32.ln() / 27.0;
    mels.iter()
        .map(|&m| {
            if m < min_log_mel {
                m * f_sp
            } else {
                min_log_hz * ((m - min_log_mel) * logstep).exp()
            }
        })
        .collect()
}

/// librosa.filters.mel(sr = 44100, n_fft = 2048, n_mels = 128,
/// fmin = 40, fmax = 16000, htk = false, norm = "slaney").
fn mel_filterbank() -> Vec<Vec<f32>> {
    let n_freqs = N_FFT / 2 + 1;
    let fftfreqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * (SAMPLE_RATE as f32 / 2.0) / (n_freqs - 1) as f32)
        .collect();

    let min_mel = hz_to_mel(&[FMIN])[0];
    let max_mel = hz_to_mel(&[FMAX])[0];
    let mel_ticks: Vec<f32> = (0..N_MELS + 2)
        .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (N_MELS + 1) as f32)
        .collect();
    let mel_f = mel_to_hz(&mel_ticks);

    let mut weights = vec![vec![0.0_f32; n_freqs]; N_MELS];
    for i in 0..N_MELS {
        let (f_lo, f_c, f_hi) = (mel_f[i], mel_f[i + 1], mel_f[i + 2]);
        // Slaney area normalization: enorm = 2 / (f_hi - f_lo)
        let enorm = 2.0 / (f_hi - f_lo);
        for (j, &f) in fftfreqs.iter().enumerate() {
            let lower = (f - f_lo) / (f_c - f_lo);
            let upper = (f_hi - f) / (f_hi - f_c);
            weights[i][j] = lower.min(upper).max(0.0) * enorm;
        }
    }
    weights
}

fn hann_window_periodic(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos())
        .collect()
}

/// torch.nn.functional.pad with mode = "reflect".
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

/// Compute the vocoder input features for mono 44.1 kHz audio.
///
/// Returns row-major `[n_frames][N_MELS]` log-mel features, where
/// `n_frames = (padded_len - WIN_SIZE) / HOP_SIZE + 1`.
pub fn wav_to_mel(audio: &[f32]) -> Vec<Vec<f32>> {
    let mel_fb = mel_filterbank();
    let window = hann_window_periodic(WIN_SIZE);

    let pad_left = (WIN_SIZE - HOP_SIZE) / 2;
    let natural_right = (WIN_SIZE - HOP_SIZE).div_ceil(2);
    let needed_right =
        (WIN_SIZE as isize - audio.len() as isize - pad_left as isize).max(0) as usize;
    let pad_right = natural_right.max(needed_right);
    let padded = if pad_right < audio.len() {
        reflect_pad(audio, pad_left, pad_right)
    } else {
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
    let n_freqs = N_FFT / 2 + 1;
    let mut frame = vec![Complex32::new(0.0, 0.0); N_FFT];

    let mut features = vec![vec![0.0_f32; N_MELS]; n_frames];
    for (t, feature_row) in features.iter_mut().enumerate() {
        let start = t * HOP_SIZE;
        for (i, bin) in frame.iter_mut().enumerate() {
            *bin = Complex32::new(padded[start + i] * window[i], 0.0);
        }
        fft.process(&mut frame);
        for m in 0..N_MELS {
            let fb_row = &mel_fb[m];
            let mut acc = 0.0_f32;
            for f in 0..n_freqs {
                let w = fb_row[f];
                if w == 0.0 {
                    continue;
                }
                let re = frame[f].re;
                let im = frame[f].im;
                acc += w * (re * re + im * im).sqrt();
            }
            feature_row[m] = acc.max(CLIP_VAL).ln();
        }
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
    fn frame_count_matches_frontend() {
        // 1 second of audio
        let mel = wav_to_mel(&sine(220.0, 44100));
        // (44100 + 768 + 769 - 2048) / 512 + 1 = 85
        assert_eq!(mel.len(), (44100 + 768 + 769 - 2048) / 512 + 1);
        assert_eq!(mel[0].len(), N_MELS);
    }

    #[test]
    fn mel_peak_in_expected_band() {
        // a 1 kHz tone should energize mel bins in the 800-1250 Hz range
        let mel = wav_to_mel(&sine(1000.0, 44100));
        let row = &mel[40];
        let (peak, &peak_val) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert!(peak_val > row[0]);
        // Slaney mel(1 kHz) = 15 exactly; bin ~60 of 128 spanning 40..16000 Hz
        assert!((55..=65).contains(&peak), "peak bin {peak}");
    }

    #[test]
    fn values_are_log_compressed() {
        let mel = wav_to_mel(&sine(440.0, 44100));
        assert!(mel.iter().flatten().all(|v| v.is_finite()));
        // clamped floor is ln(1e-5) ~= -11.5; a 0.5-amplitude sine peaks near ln(256)
        assert!(mel[40].iter().all(|&v| (-11.6..=10.0).contains(&v)));
        // Slaney area normalization keeps pure-tone mel values small
        // (a 0.5-amplitude sine peaks around ln(12) ~ 2.5)
        let peak = mel[40].iter().fold(0.0f32, |a, &b| a.max(b));
        assert!(peak > 2.0, "peak {peak} too low");
    }
}
