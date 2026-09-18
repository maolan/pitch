//! Resampling helper wrapping rubato to bring arbitrary-rate mono f32 audio
//! to the FCPE model rate (16 kHz).

use anyhow::{Context, Result};
use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

/// Resample mono audio to `target_rate` using rubato's sinc interpolator.
pub fn resample_to(audio: &[f32], from_rate: u32, target_rate: u32) -> Result<Vec<f32>> {
    if from_rate == target_rate || audio.is_empty() {
        return Ok(audio.to_vec());
    }

    let params = SincInterpolationParameters {
        f_cutoff: Some(0.95),
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        ..SincInterpolationParameters::new(256, WindowFunction::BlackmanHarris2)
    };

    let frames = audio.len();
    let mut resampler = Async::<f32>::new_sinc(
        target_rate as f64 / from_rate as f64,
        2.0,
        &params,
        frames.clamp(1, 8192),
        1,
        FixedAsync::Input,
    )
    .context("failed to create resampler")?;

    let channel_buffers: Vec<Vec<f32>> = vec![audio.to_vec()];
    let input = SequentialSliceOfVecs::new(&channel_buffers, 1, frames)
        .context("failed to build resampler input")?;
    let resampled = resampler
        .process_all(&input, frames, None)
        .context("resampling failed")?;

    Ok(resampled.take_data())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_for_matching_rate() {
        let audio = vec![0.1_f32; 100];
        let out = resample_to(&audio, 16000, 16000).unwrap();
        assert_eq!(out, audio);
    }

    #[test]
    fn downsamples_by_ratio() {
        // 1 second of 48 kHz audio -> 16000 samples
        let audio: Vec<f32> = (0..48000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
            .collect();
        let out = resample_to(&audio, 48000, 16000).unwrap();
        assert!((out.len() as i64 - 16000).abs() < 64);
        // still a sine at ~440 Hz: check zero crossings
        let crossings = out
            .iter()
            .zip(out.iter().skip(1))
            .filter(|(a, b)| a.signum() != b.signum())
            .count();
        let freq = crossings as f32 / 2.0 / (out.len() as f32 / 16000.0);
        assert!((freq - 440.0).abs() < 10.0, "freq = {freq}");
    }
}
