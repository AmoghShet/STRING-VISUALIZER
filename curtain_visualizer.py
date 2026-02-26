"""
Curtain Audio Visualizer — clean rewrite
=========================================
Strings hang from the top of the frame, swaying gently.
Audio drives WHEN a string gets plucked. A random string gets plucked.
That's it.

Two things move strings:
  1. sway   — passive, gentle sinusoidal drift. Always on. Ignores audio entirely.
  2. pluck  — a single decaying oscillation on ONE randomly chosen string,
              triggered by onset detection on the full audio signal.
              The string that gets plucked is random, not frequency-linked,
              so it always looks like fingers moving across a guitar neck.

Nothing else. No energy bow. No jitter. No per-frequency reactivity.

Requirements:
    pip install librosa numpy opencv-python tqdm

Usage:
    python curtain_visualizer.py --input Lullabye.wav --output out.mp4 --preview

    python curtain_visualizer.py --input Lullabye.wav --output out.mp4 \\
        --n_threads 40 --thread_color 255 255 255 \\
        --pluck_strength 22.0 --pluck_decay 0.93 --sway_amplitude 7.0 --preview
"""

import argparse
import numpy as np
import cv2
import librosa
import subprocess
import os
import tempfile
from tqdm import tqdm


# ── Fisheye ────────────────────────────────────────────────────────────────────

def build_fisheye_maps(width, height, strength=0.45):
    cx, cy = width / 2.0, height / 2.0
    xs = (np.arange(width)  - cx) / cx
    ys = (np.arange(height) - cy) / cy
    xv, yv = np.meshgrid(xs, ys)
    r     = np.sqrt(xv**2 + yv**2)
    r_src = r * (1.0 + strength * r**2)
    scale = np.where(r > 1e-8, r_src / r, 1.0)
    map_x = (xv * scale * cx + cx).astype(np.float32)
    map_y = (yv * scale * cy + cy).astype(np.float32)
    return map_x, map_y


# ── Onset detection ────────────────────────────────────────────────────────────

def detect_onsets(y, sr, fps, hop_length, sensitivity=0.07):
    """
    Returns bool array (n_frames,): True = onset fires this frame.
    Uses librosa onset strength with local-max + threshold + cooldown.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    peak = onset_env.max()
    if peak > 1e-8:
        onset_env /= peak

    n_frames        = len(onset_env)
    is_onset        = np.zeros(n_frames, dtype=bool)
    cooldown        = 0
    cooldown_frames = max(1, int(fps * 0.06))

    for i in range(1, n_frames - 1):
        if cooldown > 0:
            cooldown -= 1
            continue
        if (onset_env[i] > sensitivity
                and onset_env[i] >= onset_env[i - 1]
                and onset_env[i] >= onset_env[i + 1]):
            is_onset[i] = True
            cooldown = cooldown_frames

    return is_onset


# ── Renderer ───────────────────────────────────────────────────────────────────

def render(
    audio_path:        str,
    output_path:       str,
    width:             int   = 1920,
    height:            int   = 1080,
    fps:               int   = 60,
    n_threads:         int   = 40,
    thread_color:      tuple = (255, 255, 255),
    sway_amplitude:    float = 7.0,
    sway_period_min:   float = 3.5,
    sway_period_max:   float = 8.0,
    pluck_strength:    float = 22.0,
    pluck_decay:       float = 0.93,
    pluck_omega:       float = 16.0,
    onset_sensitivity: float = 0.07,
    pluck_count:       int   = 1,
    cooldown_sec:      float = 0.4,
    loudness_floor:    float = 0.2,    # minimum sway/fringe at silence (0=fully still, 1=no dynamics)
    freq_filter_low:   float = None,   # highpass / bandpass low cutoff (Hz). None = full signal
    freq_filter_high:  float = None,   # bandpass high cutoff (Hz). None = no upper limit
    n_strands:         int   = 4,
    strand_spread:     float = 4.0,    # px half-range of strand rest separation (was hardcoded 0.5)
    fringe_scale:      float = 0.22,
    glow:              bool  = True,
    glow_sigma:        float = 5.0,
    fisheye:           bool  = True,
    fisheye_strength:  float = 0.45,
    n_y_points:        int   = 80,
    preview_seconds:   float = None,
):
    # ── Audio ──────────────────────────────────────────────────────────────────
    print(f"[1/4] Loading: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(y) / sr
    print(f"      {duration:.2f}s  SR={sr}")

    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y /= peak

    hop_length   = int(sr / fps)
    total_frames = int(duration * fps)

    if preview_seconds is not None:
        total_frames = min(total_frames, int(preview_seconds * fps))
        print(f"      PREVIEW → {total_frames} frames")

    # ── Per-frame RMS loudness (0..1) ──────────────────────────────────────────
    # Drives sway amplitude and fringe scale so the curtain breathes with the
    # song's dynamics — quiet passages drift gently, loud ones sway noticeably.
    rms_frames = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    rms_peak   = rms_frames.max()
    if rms_peak > 1e-8:
        rms_frames = rms_frames / rms_peak
    # Smooth with a slow-attack follower so changes feel like breathing, not twitching
    rms_smooth    = np.zeros(len(rms_frames), dtype=np.float32)
    rms_smooth[0] = rms_frames[0]
    rms_alpha     = 0.05   # ~20-frame lag at 60fps ≈ 0.33s rise time
    for k in range(1, len(rms_frames)):
        rms_smooth[k] = rms_alpha * rms_frames[k] + (1.0 - rms_alpha) * rms_smooth[k - 1]

    def get_loudness(fi):
        return float(rms_smooth[min(fi, len(rms_smooth) - 1)])

    # ── Onsets ─────────────────────────────────────────────────────────────────
    print("[2/4] Detecting onsets ...")

    # Optionally filter to a frequency band before onset detection.
    # This lets you make the visualizer react only to e.g. high-end pick
    # attacks rather than the full signal (bass notes, low mids, etc).
    y_for_onsets = y
    if freq_filter_low is not None or freq_filter_high is not None:
        from scipy.signal import butter, sosfilt
        lo = freq_filter_low  if freq_filter_low  is not None else 1.0
        hi = freq_filter_high if freq_filter_high is not None else sr / 2.0 - 1.0
        hi = min(hi, sr / 2.0 - 1.0)
        if lo >= hi:
            raise ValueError(f"freq_filter_low ({lo}) must be < freq_filter_high ({hi})")
        sos = butter(4, [lo / (sr / 2.0), hi / (sr / 2.0)], btype="band", output="sos")
        y_for_onsets = sosfilt(sos, y).astype(np.float32)
        print(f"      Onset detection on band {lo:.0f}–{hi:.0f} Hz")

    onsets = detect_onsets(y_for_onsets, sr, fps, hop_length, sensitivity=onset_sensitivity)
    print(f"      {int(onsets[:total_frames].sum())} onset events")

    # ── Geometry ───────────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)

    margin  = width * 0.04
    x_bases = np.linspace(margin, width - margin, n_threads)
    x_bases += rng.uniform(-2.0, 2.0, n_threads)

    y_pts  = np.linspace(0, height - 1, n_y_points, dtype=np.float32)
    y_norm = y_pts / (height - 1)

    sway_shape  = (y_norm ** 0.7).astype(np.float32)
    pluck_shape = np.sin(np.pi * y_norm).astype(np.float32)

    sway_phases  = rng.uniform(0, 2 * np.pi, n_threads)
    sway_periods = rng.uniform(sway_period_min, sway_period_max, n_threads)
    sway_amps    = rng.uniform(sway_amplitude * 0.4, sway_amplitude, n_threads)

    brightnesses     = rng.uniform(0.55, 1.0, n_threads).astype(np.float32)
    thicknesses      = rng.choice([1, 1, 1, 2], size=n_threads)
    strand_offsets   = rng.uniform(-strand_spread, strand_spread, (n_threads, n_strands)).astype(np.float32)
    strand_fringe    = rng.uniform(-1.0, 1.0, (n_threads, n_strands)).astype(np.float32)
    sort_idx         = np.argsort(np.abs(strand_fringe), axis=1)
    strand_fringe    = np.take_along_axis(strand_fringe, sort_idx, axis=1)
    strand_b_factors = np.linspace(1.0, 0.5, n_strands, dtype=np.float32)
    base_color       = np.array(thread_color, dtype=np.float32)

    # ── State ──────────────────────────────────────────────────────────────────
    pluck_amp   = np.zeros(n_threads, dtype=np.float64)
    pluck_phase = np.zeros(n_threads, dtype=np.float64)
    cooldown    = np.zeros(n_threads, dtype=np.int32)
    cooldown_frames = max(1, int(fps * cooldown_sec))
    pluck_rng   = np.random.default_rng(99)

    dt = 1.0 / fps

    if fisheye:
        map_x, map_y = build_fisheye_maps(width, height, fisheye_strength)

    tmp_video = tempfile.mktemp(suffix="_noaudio.mp4")
    writer    = cv2.VideoWriter(
        tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    print(f"[3/4] Rendering {total_frames} frames ...")

    for fi in tqdm(range(total_frames)):
        t = fi * dt

        # ── Loudness-driven dynamics ───────────────────────────────────────────
        # loudness is 0..1. We keep a minimum floor so strings never go fully
        # still — the curtain always breathes a little even in the quietest parts.
        loudness      = get_loudness(fi)
        loud_scale    = loudness_floor + (1.0 - loudness_floor) * loudness
        live_sway     = sway_amps    * loud_scale
        live_fringe   = fringe_scale * loud_scale

        # ── Fire pluck on onset: pick random eligible string(s) ───────────────
        if fi < len(onsets) and onsets[fi]:
            eligible = np.where(cooldown == 0)[0]
            if eligible.size > 0:
                n_pick = min(pluck_count, eligible.size)
                chosen = pluck_rng.choice(eligible, size=n_pick, replace=False)
                pluck_amp[chosen]   = pluck_strength
                pluck_phase[chosen] = 0.0
                cooldown[chosen]    = cooldown_frames

        cooldown    = np.maximum(0, cooldown - 1)
        pluck_phase += pluck_omega * dt
        pluck_amp   *= pluck_decay

        # ── Displacement ───────────────────────────────────────────────────────
        sway_disp  = live_sway * np.sin(2.0 * np.pi * t / sway_periods + sway_phases)
        pluck_disp = pluck_amp * np.sin(pluck_phase)

        # ── Draw ───────────────────────────────────────────────────────────────
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(n_threads):
            b_t      = float(brightnesses[i])
            x_base   = sway_disp[i] * sway_shape + pluck_disp[i] * pluck_shape
            fringe_e = float(pluck_amp[i]) * live_fringe

            for s in range(n_strands):
                b     = b_t * float(strand_b_factors[s])
                color = (int(base_color[0]*b), int(base_color[1]*b), int(base_color[2]*b))

                x_disp  = x_base + strand_offsets[i, s]
                x_disp += fringe_e * float(strand_fringe[i, s]) * pluck_shape

                x_coords = np.clip(
                    (x_bases[i] + x_disp).astype(np.int32), 0, width - 1
                )
                pts = np.stack(
                    [x_coords, y_pts.astype(np.int32)], axis=1
                ).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], False, color,
                              int(thicknesses[i]), cv2.LINE_AA)

        if glow:
            bloom = cv2.GaussianBlur(canvas, (0, 0), glow_sigma,       glow_sigma)
            halo  = cv2.GaussianBlur(canvas, (0, 0), glow_sigma * 2.5, glow_sigma * 2.5)
            frame = cv2.add(canvas, (bloom * 0.60).clip(0, 255).astype(np.uint8))
            frame = cv2.add(frame,  (halo  * 0.35).clip(0, 255).astype(np.uint8))
        else:
            frame = canvas

        if fisheye:
            frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)

        writer.write(frame)

    writer.release()

    # ── Mux ────────────────────────────────────────────────────────────────────
    print("[4/4] Muxing audio ...")
    dur_args = ["-t", str(preview_seconds)] if preview_seconds else []
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", tmp_video, "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        *dur_args, output_path,
    ], capture_output=True, text=True)
    os.remove(tmp_video)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("ffmpeg failed")
    print(f"Done → {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  "-i", required=True)
    p.add_argument("--output", "-o", default="curtain.mp4")
    p.add_argument("--width",  type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps",    type=int, default=60)

    p.add_argument("--n_threads",    type=int,   default=40)
    p.add_argument("--thread_color", type=int, nargs=3, default=[255, 255, 255],
                   metavar=("B", "G", "R"))

    p.add_argument("--sway_amplitude",  type=float, default=7.0)
    p.add_argument("--sway_period_min", type=float, default=3.5)
    p.add_argument("--sway_period_max", type=float, default=8.0)

    p.add_argument("--pluck_strength",    type=float, default=22.0)
    p.add_argument("--pluck_decay",       type=float, default=0.93)
    p.add_argument("--pluck_omega",       type=float, default=16.0)
    p.add_argument("--onset_sensitivity", type=float, default=0.07,
                   help="Lower = more onsets (default 0.07)")
    p.add_argument("--pluck_count",  type=int,   default=1,
                   help="Strings plucked per onset. 1 = single fingerpick (default).")
    p.add_argument("--cooldown_sec", type=float, default=0.4,
                   help="Seconds before same string can be re-plucked (default 0.4).")
    p.add_argument("--loudness_floor", type=float, default=0.2,
                   help="Minimum sway/fringe scale at silence (0=fully still, 1=no dynamics). Default 0.2.")

    p.add_argument("--freq_filter_low",  type=float, default=None,
                   help="Highpass cutoff for onset detection (Hz). "
                        "e.g. 2000 = only react to frequencies above 2kHz. "
                        "None = full signal (default).")
    p.add_argument("--freq_filter_high", type=float, default=None,
                   help="Optional bandpass upper cutoff (Hz). "
                        "Use with --freq_filter_low to isolate a band. "
                        "None = no upper limit (default).")

    p.add_argument("--n_strands",    type=int,   default=4)
    p.add_argument("--strand_spread", type=float, default=4.0,
                   help="Pixel half-range of strand rest separation. "
                        "Higher = strands visibly apart even at rest (default 4.0).")
    p.add_argument("--fringe_scale", type=float, default=0.22)
    p.add_argument("--n_y_points",   type=int,   default=80)

    p.add_argument("--glow",       action="store_true", default=True)
    p.add_argument("--no_glow",    dest="glow", action="store_false")
    p.add_argument("--glow_sigma", type=float, default=5.0)

    p.add_argument("--fisheye",          action="store_true", default=True)
    p.add_argument("--no_fisheye",       dest="fisheye", action="store_false")
    p.add_argument("--fisheye_strength", type=float, default=0.45)

    p.add_argument("--preview",         action="store_true")
    p.add_argument("--preview_seconds", type=float, default=10.0)

    a = p.parse_args()
    render(
        audio_path        = a.input,
        output_path       = a.output,
        width             = a.width,
        height            = a.height,
        fps               = a.fps,
        n_threads         = a.n_threads,
        thread_color      = tuple(a.thread_color),
        sway_amplitude    = a.sway_amplitude,
        sway_period_min   = a.sway_period_min,
        sway_period_max   = a.sway_period_max,
        pluck_strength    = a.pluck_strength,
        pluck_decay       = a.pluck_decay,
        pluck_omega       = a.pluck_omega,
        onset_sensitivity = a.onset_sensitivity,
        pluck_count       = a.pluck_count,
        cooldown_sec      = a.cooldown_sec,
        loudness_floor    = a.loudness_floor,
        freq_filter_low   = a.freq_filter_low,
        freq_filter_high  = a.freq_filter_high,
        n_strands         = a.n_strands,
        strand_spread     = a.strand_spread,
        fringe_scale      = a.fringe_scale,
        n_y_points        = a.n_y_points,
        glow              = a.glow,
        glow_sigma        = a.glow_sigma,
        fisheye           = a.fisheye,
        fisheye_strength  = a.fisheye_strength,
        preview_seconds   = a.preview_seconds if a.preview else None,
    )


if __name__ == "__main__":
    main()
