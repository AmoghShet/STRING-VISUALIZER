# String Audio Visualizer

An audio-reactive hanging-thread renderer with CRT phosphor glow and fisheye lens distortion. Vertical threads hang from the top of the frame, swaying gently in a slow, unsynchronised drift. When the audio moves, a randomly chosen thread is plucked — bowing outward in a brief decaying oscillation, its sub-strands fringing apart before settling back into one — giving the impression of a curtain of loose strings that simply exists in the room, and feels the music pass through it.

Built as a sister project to [`PARTICLE-VISUALIZER`](https://github.com/AmoghShet/PARTICLE-VISUALIZER.git) & [`OSCILLOSCOPE-VISUALIZER`](https://github.com/AmoghShet/OSCILLOSCOPE-VISUALIZER.git), sharing the same rendering pipeline and visual aesthetic.

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

Sway amplitude scales with the track's overall loudness in real time — during quiet passages the curtain barely drifts; as the song opens up the threads sway more visibly. `--loudness_floor` sets how much motion remains at the quietest moment so the curtain never goes completely still.

### Pluck Response

Onset detection runs on the audio signal and fires a pluck event at each transient — guitar pick attacks, note beginnings, any sharp energy rise. When an onset fires, **one randomly chosen thread** is plucked. The thread that moves is not tied to any frequency band; it is simply whichever eligible thread happens to be picked at that moment. This gives the curtain a fingerpicked quality — different strings waking up each time — rather than the same cluster of threads lighting up every note.

Each thread has a per-string cooldown (`--cooldown_sec`) so a recently-plucked thread cannot be immediately re-selected, naturally spreading plucks across the curtain over time.

The pluck displacement is:

```
displacement = pluck_strength × sin(pluck_phase) × sin(π × y_norm)
```

The `sin(π × y_norm)` envelope pins the bow to zero at both the top anchor and the bottom tip, so the thread bows in the middle — the fundamental mode of a plucked string. The amplitude decays by `--pluck_decay` every frame, returning the thread to its natural sway.

### Frequency Filtering for Onset Detection

By default onset detection reads the full mixed audio signal. You can narrow it to a specific frequency band using `--freq_filter_low` and `--freq_filter_high`, which apply a bandpass filter to the signal before onset detection runs. This lets you target only the part of the spectrum that carries the musical gesture you care about.

For a fingerpicked acoustic guitar, the pick attack transient lives in roughly the 2–6 kHz range. Filtering to that band means the curtain responds to the sharpness of each pluck rather than to the body of the note or any low-end bloom beneath it.

### Bowstring Fringe

Each visible thread is composed of `--n_strands` sub-strands. At rest they are separated by a fixed random offset (controlled by `--strand_spread` in pixels), so the bundle is visibly fibrous even when still. When plucked, each strand fans out further according to its own character value, spreading by a fraction of the pluck displacement (`--fringe_scale`). The spread is driven by the raw pluck envelope (always positive, always decaying), so strands fan cleanly outward and converge back — they do not flicker or cross. The innermost strand is the brightest; outer strands dim progressively, giving the bundle a natural core-and-edge quality.

Fringe spread also scales with loudness alongside sway, so during louder passages the strands fan a little wider on each pluck.

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

### Lullabye (recommended config)

```bash
python curtain_visualizer.py --input Lullabye.wav --output lullabye.mp4 \
  --fps 60 --thread_color 255 255 255 \
  --n_threads 40 --n_strands 4 --strand_spread 5.0 --fringe_scale 0.35 \
  --sway_amplitude 7.0 --pluck_strength 22.0 --pluck_decay 0.93 \
  --freq_filter_low 2500 --freq_filter_high 3000 \
  --loudness_floor 0.15 \
  --preview
```

### Warm amber variant

```bash
python curtain_visualizer.py --input Lullabye.wav --output lullabye_amber.mp4 \
  --fps 60 --thread_color 60 180 255 \
  --n_threads 40 --n_strands 4 --strand_spread 5.0 --fringe_scale 0.35 \
  --sway_amplitude 7.0 --pluck_strength 22.0 --pluck_decay 0.93 \
  --freq_filter_low 2500 --freq_filter_high 3000 \
  --loudness_floor 0.15
```

### Denser curtain (veil-like)

```bash
python curtain_visualizer.py --input track.wav --output curtain_dense.mp4 \
  --fps 60 --thread_color 255 255 255 \
  --n_threads 200 --n_strands 3 --strand_spread 2.0 --fringe_scale 0.12 \
  --sway_amplitude 4.0 --pluck_strength 10.0 --pluck_decay 0.93 \
  --loudness_floor 0.2
```

### Sparse, each thread legible

```bash
python curtain_visualizer.py --input track.wav --output curtain_sparse.mp4 \
  --fps 60 --thread_color 255 255 255 \
  --n_threads 20 --n_strands 5 --strand_spread 6.0 --fringe_scale 0.40 \
  --sway_amplitude 8.0 --pluck_strength 18.0 --pluck_decay 0.89 \
  --loudness_floor 0.1
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
| `--n_threads` | `40` | Number of hanging threads. 20–50 gives distinct, legible strings. 200+ approaches a dense veil |
| `--thread_color B G R` | `255 255 255` | Thread color in BGR order. Default is pure white |
| `--n_y_points` | `80` | Vertical resolution per thread — number of polyline points from top to bottom. Higher = smoother curve when plucked |

### Passive Sway

| Flag | Default | Description |
|------|---------|-------------|
| `--sway_amplitude` | `7.0` | Maximum lateral sway displacement at the bottom of a thread in pixels. Scales with loudness at runtime |
| `--sway_period_min` | `3.5` | Fastest individual sway cycle in seconds |
| `--sway_period_max` | `8.0` | Slowest individual sway cycle in seconds. Each thread gets its own random period in this range, preventing synchronisation |
| `--loudness_floor` | `0.2` | Minimum sway and fringe scale at silence (0 = fully still, 1 = no dynamics). Keeps the curtain alive during quiet passages |

### Pluck Response

| Flag | Default | Description |
|------|---------|-------------|
| `--pluck_strength` | `22.0` | Maximum bow displacement at the midpoint of a plucked thread in pixels |
| `--pluck_decay` | `0.93` | Per-frame pluck amplitude multiplier (0–1). Lower = settles faster. `0.85` returns quickly; `0.96` rings out long |
| `--pluck_omega` | `16.0` | Pluck oscillation angular frequency in rad/s. Higher = faster visible vibration on pluck |
| `--onset_sensitivity` | `0.07` | Onset detection threshold (0–1). Lower = more plucks detected; higher = only strong transients trigger |
| `--pluck_count` | `1` | How many randomly chosen threads to pluck per onset event. `1` = single fingerpick feel; `2` = occasional two-string moments |
| `--cooldown_sec` | `0.4` | Seconds before the same string can be plucked again. Forces variety across the curtain over time |

### Frequency Filtering

| Flag | Default | Description |
|------|---------|-------------|
| `--freq_filter_low` | off | Highpass / bandpass low cutoff in Hz for onset detection. e.g. `2000` = only react to content above 2 kHz. Leave unset to use the full signal |
| `--freq_filter_high` | off | Bandpass upper cutoff in Hz. Use together with `--freq_filter_low` to isolate a specific band (e.g. `2500 3000` for acoustic guitar pick attack) |

### Bowstring Fringe

| Flag | Default | Description |
|------|---------|-------------|
| `--n_strands` | `4` | Sub-threads per visible thread |
| `--strand_spread` | `4.0` | Pixel half-range of strand rest separation. Higher = strands visibly apart even at rest; lower = bundle reads as a single line |
| `--fringe_scale` | `0.22` | Fringe spread as a fraction of pluck displacement. Scales with loudness. `0.10` = subtle. `0.35+` = clearly visible split on each pluck |

### Appearance

| Flag | Default | Description |
|------|---------|-------------|
| `--glow` / `--no_glow` | on | Enable or disable CRT phosphor glow |
| `--glow_sigma` | `5.0` | Glow blur radius in pixels. Softer than the sibling visualizers by design — candlelight, not neon |
| `--fisheye` / `--no_fisheye` | on | Enable or disable barrel lens distortion |
| `--fisheye_strength` | `0.45` | Fisheye intensity (0–1). `0` = flat. `0.7+` = strong warp |

---

## Parameter Reference: What Changes What

### The curtain feels too reactive / plucking too often
Raise `--onset_sensitivity` toward `0.15`. The curtain will only respond to the sharpest transients. Alternatively narrow the frequency band (`--freq_filter_low`, `--freq_filter_high`) so onset detection ignores parts of the signal that are too busy.

### The curtain feels too sparse / missing obvious plucks
Lower `--onset_sensitivity` toward `0.04`. If you are using a frequency filter, try widening the band slightly downward — e.g. `--freq_filter_low 1500` instead of `2500` to catch more of the note onset rather than just the very tip of the pick attack.

### The same few strings keep getting plucked
Raise `--cooldown_sec` (e.g. `0.6–0.8`). A longer cooldown forces the random selection to reach threads that haven't been touched recently. You can also raise `--pluck_count` to `2` to spread each onset across two threads.

### The threads settle too slowly after a pluck
Lower `--pluck_decay` toward `0.85`. The oscillation decays faster and the thread returns to its gentle sway within half a second.

### The threads ring out too briefly / I want more resonance
Raise `--pluck_decay` toward `0.96`. Plucked threads will oscillate visibly for several seconds.

### The strands look like one solid line, not a frayed thread
Raise `--strand_spread` (e.g. `5.0–8.0`) to increase rest separation, and raise `--fringe_scale` (e.g. `0.30–0.45`) to increase the fan on each pluck. Both together make the fibrous quality clearly legible.

### The strands look like separate threads rather than one frayed string
Lower `--strand_spread` toward `1.0–2.0`. The bundle will read as a single thread at rest, with strands only separating perceptibly mid-pluck.

### The curtain barely moves during quiet passages
Lower `--loudness_floor` toward `0.0`. The sway and fringe will shrink more during silence, making the contrast between quiet and loud moments more dramatic.

### The curtain moves too much even in quiet passages
Raise `--loudness_floor` toward `0.4`. Sway and fringe will never drop below 40% of their maximum, keeping the curtain consistently alive throughout.

### The sway feels too mechanical / all threads moving together
Widen the gap between `--sway_period_min` and `--sway_period_max` (e.g. `2.0` and `10.0`). A wider range produces more staggered, organic motion.

### Stronger fisheye warp
Set `--fisheye_strength 0.7`. Above `0.8` corners go fully black — use with care.

---

## Output Pipeline

1. **Audio loading** — `librosa` loads the file and peak-normalises it so `--pluck_strength` behaves consistently regardless of source loudness.
2. **RMS loudness** — a per-frame RMS envelope is computed and smoothed with a slow-attack follower (~0.33s rise time). This drives the sway amplitude and fringe scale in real time so the curtain breathes with the song's dynamics.
3. **Onset detection** — if `--freq_filter_low` or `--freq_filter_high` are set, the audio is bandpass filtered first. Onset strength is computed on the (optionally filtered) signal; a local-maximum threshold with a 60ms cooldown produces a frame-accurate boolean onset array.
4. **Thread initialization** — base X positions, sway parameters, strand characters, and brightness values are seeded once with a fixed RNG so renders are fully deterministic.
5. **Frame rendering** — for each frame, any onset fires a pluck on a randomly chosen eligible thread (respecting per-string cooldown). Sway and pluck displacements are computed, scaled by current loudness, and each thread's strands are drawn as polylines. OpenCV writes frames to a temporary silent `.mp4` via the `mp4v` codec.
6. **Audio mux** — FFmpeg re-encodes to H.264 (`libx264`, CRF 18) and muxes in the original audio as AAC 192k. The temporary silent file is deleted.

---

## Notes

- Input audio is **peak-normalized globally** at load time. This keeps `--pluck_strength` consistent regardless of source loudness.
- Plucked strings are chosen **randomly**, not by frequency. The audio controls *when* a pluck fires; a random eligible string controls *which* thread moves. This gives a fingerpicked feel rather than a frequency-analyzer feel.
- The frequency filter (`--freq_filter_low` / `--freq_filter_high`) affects **onset detection only** — it does not change which strings exist or how they are laid out. It is purely a lens for deciding what in the audio counts as a pluck event.
- Strand fringe spread is driven by the **raw pluck envelope** (always positive), not the oscillating displacement. This means strands fan outward and return cleanly — they do not flicker or cross — and the fringe always reads as a single coherent bundle.
- The fisheye remap is computed **once per render** and never fed back into the canvas, so distortion does not accumulate.
- Renders are **fully deterministic**. The geometry RNG is seeded at `42`; the pluck selection RNG at `99`. Given identical inputs and flags, two runs produce identical output.
