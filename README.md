# maolan-pitch

Neural pitch analysis and pitch-correction resynthesis for the
[Maolan](https://maolan.rs) DAW, running on [Burn](https://burn.dev) 0.21
with the wgpu (Vulkan) backend.

The crate provides two model pipelines, both implemented from scratch in
Rust (BSD-2-Clause) and loading converted Burn Pack (`.bpk`) weights:

- **FCPE F0 detection** — a conformer-based pitch tracker for singing
  voice. Input audio of any sample rate is resampled to 16 kHz and fed to
  the network as a log-spectral (STCT) frontend; output is a per-frame f0
  track with a voicing confidence.
- **PC-NSF-HiFiGAN vocoder** — a pitch-controllable neural vocoder
  (44.1 kHz, hop 512, 128 mel bins). Given a log-mel spectrogram and an
  f0 contour it re-synthesizes a voice singing that contour with the
  original timbre. This is the engine behind Maolan's *resynthesis* pitch
  correction mode (Melodyne-style, as opposed to pitch-shifting the
  original recording).

## Requirements

A working Vulkan GPU. Burn's wgpu backend is the only backend; there is
no CPU fallback. Initialization failures are reported as descriptive
errors (never panics) via `load_fcpe_weights` / `load_pc_nsf_hifigan_weights`.

## Model weights

Weights are not bundled. On first use they are downloaded from
HuggingFace into the cache directory you pass:

| Model | HF repo | File | Weights license |
|---|---|---|---|
| FCPE | `maolandaw/FCPE-burn` | `fcpe.bpk` (~43 MB) | MIT (upstream [CNChTu/FCPE](https://github.com/CNChTu/FCPE)) |
| PC-NSF-HiFiGAN | `maolandaw/PC-NSF-HiFiGAN-burn` | `pc_nsf_hifigan.bpk` (~57 MB) | AGPLv3 (upstream [openvpi/vocoders](https://github.com/openvpi/vocoders)) |

The conversion tooling that produces these `.bpk` archives lives in the
respective HF repos, so each artifact carries its corresponding source.
The PC-NSF-HiFiGAN weights are copyleft: shipping them (or converted
derivatives) obligates you under the AGPLv3, while this crate itself
remains BSD-2-Clause.

Environment overrides for tests/local development:
`FCPE_BPK` and `PC_NSF_HIFIGAN_BPK` point directly at local `.bpk` files,
bypassing the download.

## Usage

```rust
use maolan_pitch::{ensure_fcpe_model, load_fcpe_weights, FcpeDetector};

let bpk = ensure_fcpe_model(&cache_dir)?;        // download once
let detector = FcpeDetector::new(load_fcpe_weights(&bpk)?);
let frames = detector.detect_f0(&mono_audio, 48_000, |p| println!("{p:.0}%"));
// frames: Vec<F0Frame { time_samples, frequency: Option<f32>, confidence }>
```

```rust
use maolan_pitch::{ensure_pc_nsf_hifigan_model, load_pc_nsf_hifigan_weights, PcNsfHifiGan};

let bpk = ensure_pc_nsf_hifigan_model(&cache_dir)?;
let vocoder = PcNsfHifiGan::new(load_pc_nsf_hifigan_weights(&bpk)?);
let pcm_44k1 = vocoder.synthesize(&mel_frames, &f0_hz);
```

Everything is generic over `burn::tensor::backend::Backend` with the
default alias `maolan_pitch::B` = wgpu; a CPU backend could be
reintroduced by adding an alias and a loader for it.

## Validation binary

```bash
cargo run --bin pitch_test                 # synthetic 220→440→330 Hz glide;
                                           # asserts f0 correlation > 0.99
cargo run --bin pitch_test detect \
    --model fcpe.bpk --wav vocal.wav       # print (time, hz, confidence)
cargo run --bin pitch_test vocoder         # closed loop: synthesize 220 Hz,
                                           # re-detect with FCPE, compare
```

## Tests

```bash
cargo test
```

Includes unit tests for the mel frontends, the NSF excitation generator,
weight-norm materialization, and — when the local `.bpk` files are
present (via `FCPE_BPK` / `PC_NSF_HIFIGAN_BPK` or the default search
paths) — end-to-end validation of both models on the GPU. Missing weights
skip those tests gracefully.

## Repository layout

```
src/lib.rs       public API, model download/cache, wgpu device setup
src/fcpe.rs      FCPE network (Burn), latent decoding
src/mel.rs       16 kHz STCT frontend for FCPE
src/vmel.rs      44.1 kHz log-mel frontend for the vocoder
src/vocoder.rs   PC-NSF-HiFiGAN generator (Burn), NSF excitation
src/resample.rs  rubato-based resampling to model rates
src/bin/pitch_test.rs   validation CLI
tests/wgpu_probe.rs     headless GPU adapter probe
```

## License

BSD-2-Clause, matching the Maolan project. This covers only the Rust code
in this repository — the model weights loaded at runtime carry their own
licenses as listed above.
