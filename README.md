# String Audio Visualizer

An audio-reactive hanging-thread renderer with CRT phosphor glow and fisheye lens distortion. Vertical threads hang from the top of the frame, swaying gently in a slow, unsynchronised drift. When the audio moves through them, the relevant threads are plucked — bowing outward in a brief decaying oscillation, their sub-strands fringing apart before settling back into one — giving the impression of a curtain of loose strings that simply exists in the room, and feels the music pass through it.

Built as a sister project to [`PARTICLE-VISUALIZER`](https://github.com/AmoghShet/PARTICLE-VISUALIZER.git) & [`OSCILLOSCOPE-VISUALIZER`](https://github.com/AmoghShet/OSCILLOSCOPE-VISUALIZER.git), sharing the same rendering pipeline and visual aesthetic; was used to create the music visualizer for [`Mo - Lullabye (Draft)`](https://www.youtube.com/watch?v=wkN10XcN6KA)

---

## Preview

> *Rendered for "Lullabye (Draft)" — A minor, 133 BPM, atmospheric indie folk.*

The default color (`255 255 255` BGR) renders as pure white — each thread a thin filament of light, warm under glow, hanging in the dark.

---

## How It Works

### Thread Geometry

Each thread is a vertical polyline pinned at the top of the frame and hanging freely to the bottom. It has no horizontal motion of its own — it does not travel anywhere. It just hangs. Two displacement layers act on it independently: a slow passive sway, and an audio-triggered pluck.

### Passive Sway

Every thread is assigned a random sway period (between `--sway_period_min` and `--sway_period_max` seconds) and a random phase offset, so no two threads ever move in sync. The sway shape is weighted as `y^0.7` along the thread's length — the top anchor barely moves, the bottom hangs free and swings more. The overall effect reads as a still room with just a little air in it.

### Pluck Response

A pluck fires when a thread's normalized frequency-band energy rises sharply above `--pluck_threshold` relative to the previous frame. This catches guitar plucks, vocal onsets, and transient attacks without triggering on sustained notes. When triggered, a damped oscillation is added to the thread's displacement:

```
displacement = pluck_amp × sin(pluck_phase) × sin(π × y_norm)
```

The `sin(π × y_norm)` envelope pins the bow to zero at both the top anchor and the bottom tip, so the thread bows in the middle — the fundamental mode of a plucked string. `pluck_amp` decays by `--pluck_decay` every frame, returning the thread to its natural sway. Multiple triggers accumulate so sustained loud passages keep threads visibly alive.

### Frequency Mapping

Threads are assigned to frequency bands **logarithmically** from left to right — matching the perceptual spacing of human hearing rather than a linear Hz scale. The leftmost threads respond to low frequencies (guitar body, bass, sub-bass); center threads to mids (vocals, overtones); rightmost threads to highs (falsetto, reverb shimmer, transient air). A guitar pluck physically travels left to right through the curtain as its harmonics climb the spectrum.

### Bowstring Fringe

Each visible thread is composed of `--n_strands` sub-strands. At rest, they are nearly coincident — within sub-pixel distance of each other — and the bundle reads as a single line. When plucked, each strand separates slightly according to its own fixed character value, fanning outward by a fraction of the pluck displacement (`--fringe_scale`). The spread is driven by the raw pluck envelope (always positive, always decaying), so strands fan cleanly outward and converge back — they do not flicker or cross. The innermost strand is the brightest; outer strands dim progressively, giving the bundle a natural core-and-edge quality even at rest.

### CRT Glow

Each frame is composited from three additive layers:

- **Wide soft halo** — broad Gaussian at `2.5× sigma`, blended at 35% opacity
- **Medium bloom** — tighter Gaussian at `1× sigma`, blended at 60% opacity
- **Crisp core** — the raw polylines at full brightness

The glow sigma is deliberately softer than the sibling visualizers (`--glow_sigma 5.0` vs. `7.0`) — candlelight rather than neon.

### Fisheye

A barrel distortion remap is pre-computed once at render start and applied to each output frame. The edges of the curtain bow gently inward, as if seen on a physical CRT screen.

---

## Requirements

```bash
pip install librosa numpy opencv-python scipy tqdm
```

FFmpeg must be installed and available on your `PATH`:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

---

## Usage

### Quick preview (first 10 seconds)

```bash
python curtain_visualizer.py --input Lullabye.wav --output preview.mp4 --preview
```

### Full render

```bash
python curtain_visualizer.py \
  --input Lullabye.wav --output lullabye.mp4 \
  --fps 60 --n_threads 35 \
  --thread_color 255 255 255 \
  --sway_amplitude 6.0 --pluck_strength 14.0 --pluck_decay 0.91 \
  --n_strands 4 --fringe_scale 0.12
```

### Warm amber variant

```bash
python curtain_visualizer.py \
  --input Lullabye.wav --output lullabye_amber.mp4 \
  --fps 60 --n_threads 35 \
  --thread_color 60 180 255 \
  --sway_amplitude 6.0 --pluck_strength 14.0 --pluck_decay 0.91 \
  --n_strands 4 --fringe_scale 0.12
```

### Denser curtain (more threads, veil-like)

```bash
python curtain_visualizer.py \
  --input track.wav --output curtain_dense.mp4 \
  --fps 60 --n_threads 200 \
  --thread_color 255 255 255 \
  --sway_amplitude 4.0 --pluck_strength 10.0 --pluck_decay 0.93 \
  --n_strands 3 --fringe_scale 0.08
```

### Sparse, each thread legible

```bash
python curtain_visualizer.py \
  --input track.wav --output curtain_sparse.mp4 \
  --fps 60 --n_threads 20 \
  --thread_color 255 255 255 \
  --sway_amplitude 8.0 --pluck_strength 18.0 --pluck_decay 0.89 \
  --n_strands 5 --fringe_scale 0.16
```

---

## All Parameters

### Input / Output

| Flag | Default | Description |
|------|---------|-------------|
| `--input` / `-i` | *(required)* | Path to input audio file (WAV recommended) |
| `--output` / `-o` | `curtain.mp4` | Output video path |
| `--width` | `1920` | Output width in pixels |
| `--height` | `1080` | Output height in pixels |
| `--fps` | `60` | Output frame rate |
| `--preview` | off | Render only the first N seconds |
| `--preview_seconds` | `10.0` | Duration of preview in seconds |

### Threads

| Flag | Default | Description |
|------|---------|-------------|
| `--n_threads` | `300` | Number of hanging threads. 20–50 gives distinct, legible strings. 200–400 approaches a dense veil |
| `--thread_color B G R` | `255 255 255` | Thread color in BGR order. Default is pure white |
| `--n_y_points` | `80` | Vertical resolution per thread — number of polyline points from top to bottom. Higher = smoother curve when plucked |
| `--freq_min` | `40.0` | Frequency in Hz assigned to the leftmost thread |
| `--freq_max` | `16000.0` | Frequency in Hz assigned to the rightmost thread |

### Passive Sway

| Flag | Default | Description |
|------|---------|-------------|
| `--sway_amplitude` | `6.0` | Maximum lateral sway displacement at the bottom of a thread in pixels |
| `--sway_period_min` | `3.0` | Fastest individual sway cycle in seconds |
| `--sway_period_max` | `7.0` | Slowest individual sway cycle in seconds. Each thread gets its own random period in this range, preventing synchronisation |

### Pluck Response

| Flag | Default | Description |
|------|---------|-------------|
| `--pluck_strength` | `14.0` | Maximum bow displacement at the midpoint of a plucked thread in pixels. Keep modest — the song is gentle |
| `--pluck_decay` | `0.91` | Per-frame pluck amplitude multiplier (0–1). Lower = settles faster. `0.85` returns quickly; `0.96` rings out long |
| `--pluck_omega` | `18.0` | Pluck oscillation angular frequency in rad/s. Higher = faster visible vibration on pluck |
| `--pluck_threshold` | `0.25` | Normalized energy floor (0–1) required to trigger a pluck. Raise to make only strong transients register; lower to make the curtain react to quieter passages |

### Bowstring Fringe

| Flag | Default | Description |
|------|---------|-------------|
| `--n_strands` | `4` | Sub-threads per visible thread. At rest they are nearly coincident; they fan apart on pluck and converge as the pluck decays |
| `--fringe_scale` | `0.12` | Fringe spread as a fraction of pluck displacement. `0.06` = barely there. `0.20` = clearly visible split |

### Appearance

| Flag | Default | Description |
|------|---------|-------------|
| `--glow` / `--no_glow` | on | Enable or disable CRT phosphor glow |
| `--glow_sigma` | `5.0` | Glow blur radius in pixels. Softer than the sibling visualizers by design — candlelight, not neon |
| `--fisheye` / `--no_fisheye` | on | Enable or disable barrel lens distortion |
| `--fisheye_strength` | `0.45` | Fisheye intensity (0–1). `0` = flat. `0.7+` = strong warp |

---

## Parameter Reference: What Changes What

### The threads feel too reactive / plucking too often
Raise `--pluck_threshold` toward `0.4`. The curtain will only respond to clear transients and loud onsets, ignoring the sustained body of notes.

### The threads settle too slowly after a pluck
Lower `--pluck_decay` toward `0.85`. The oscillation decays faster and the thread returns to its gentle sway within half a second.

### The threads ring out too briefly / I want more resonance
Raise `--pluck_decay` toward `0.96`. Plucked threads will oscillate visibly for several seconds, overlapping with new triggers on busier passages.

### The fringe is too subtle to see
Raise `--fringe_scale` toward `0.20–0.25`. On a sparse thread count (20–40) each strand will be individually legible on a pluck. On a dense count the effect reads as texture rather than individual strands — both are intentional outcomes.

### The fringe is too obvious / looks like separate threads
Lower `--fringe_scale` toward `0.06`. The bundle will read as a single thread at all times, with only a barely-perceptible breathing of the edge on strong plucks.

### The sway feels too mechanical / all threads moving together
Widen the gap between `--sway_period_min` and `--sway_period_max` (e.g. `2.0` and `9.0`). Because each thread has a random period drawn from this range, a wider range produces more staggered, organic motion.

### I want more of the acoustic body of the guitar to move the curtain
Lower `--freq_min` to `80` and `--pluck_threshold` to `0.20`. The low-mid threads (guitar body, pick attack) will respond more readily.

### Stronger fisheye warp
Set `--fisheye_strength 0.7`. Above `0.8` corners go fully black — use with care.

---

## Output Pipeline

1. **Audio loading** — `librosa` loads the file and peak-normalises it so `--pluck_strength` behaves consistently regardless of source loudness.
2. **Feature extraction** — STFT is computed once upfront. Per-thread frequency-bin energies are extracted for every frame and normalized to each thread's own time history, so every frequency band has equivalent dynamic range.
3. **Thread initialization** — base X positions, sway parameters, strand characters, and brightness values are seeded once with a fixed RNG so renders are fully deterministic.
4. **Frame rendering** — for each frame, pluck state is updated, sway and pluck displacements are computed, and each thread's strands are drawn as polylines. OpenCV writes frames to a temporary silent `.mp4` via the `mp4v` codec.
5. **Audio mux** — FFmpeg re-encodes to H.264 (`libx264`, CRF 18) and muxes in the original audio as AAC 192k. The temporary silent file is deleted.

---

## Notes

- Input audio is **peak-normalized globally** at load time. This keeps `--pluck_strength` consistent regardless of source loudness.
- The frequency-to-thread mapping is **logarithmic**, not linear. This matches the perceptual spacing of human hearing and ensures the guitar-body frequencies (100–400 Hz) occupy the same visual real estate as the vocal/falsetto range (400–4000 Hz), rather than being compressed into a thin sliver on the left edge.
- Strand fringe spread is driven by the **raw pluck envelope** (always positive), not the oscillating displacement. This means strands fan outward and return cleanly — they do not flicker or cross — and the fringe always reads as a single coherent bundle at rest.
- The fisheye remap is computed **once per render** and never fed back into the canvas, so distortion does not accumulate.
- Renders are **fully deterministic**. The RNG is seeded at `42` for thread initialization. Given identical inputs and flags, two runs will produce identical output.
