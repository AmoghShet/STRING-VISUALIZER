"""
Curtain Audio Visualizer
========================
'Lullabye' aesthetic — vertical threads hang from the top of the frame,
swaying gently like loose strings in a still room. Each thread maps to a
frequency band; when the audio moves through it, the thread is plucked —
a brief decaying bow — before settling back to its gentle sway.

CRT phosphor glow + fisheye lens distortion.
Sister visualizer to particle_visualizer.py and oscilloscope_visualizer.py.

Requirements:
    pip install librosa numpy opencv-python scipy tqdm

Usage:
    # Quick 10s preview:
    python curtain_visualizer.py --input Lullabye.wav --output preview.mp4 --preview

    # Full render (recommended settings for Lullabye):
    python curtain_visualizer.py \\
        --input Lullabye.wav --output lullabye.mp4 \\
        --fps 60 --n_threads 300 \\
        --thread_color 60 180 255 \\
        --sway_amplitude 6.0 --pluck_strength 14.0 --pluck_decay 0.91
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

def build_fisheye_maps(width, height, strength=0.5):
    """Pre-compute barrel distortion remap tables (call once, reuse every frame)."""
    cx, cy = width / 2.0, height / 2.0
    xs = (np.arange(width)  - cx) / cx
    ys = (np.arange(height) - cy) / cy
    xv, yv = np.meshgrid(xs, ys)
    r     = np.sqrt(xv**2 + yv**2)
    r_src = r * (1.0 + strength * r**2)
    scale = np.where(r > 1e-8, r_src / (r + 1e-8), 1.0)
    map_x = (xv * scale * cx + cx).astype(np.float32)
    map_y = (yv * scale * cy + cy).astype(np.float32)
    return map_x, map_y


# ── Core renderer ──────────────────────────────────────────────────────────────

def render_visualizer(
    audio_path: str,
    output_path: str,
    width:  int   = 1920,
    height: int   = 1080,
    fps:    int   = 60,
    n_threads:        int   = 300,
    thread_color:     tuple = (60, 180, 255),   # warm amber (BGR)
    glow:             bool  = True,
    glow_sigma:       float = 5.0,
    fisheye:          bool  = True,
    fisheye_strength: float = 0.45,
    # ── Passive sway (the "existing") ──────────────────────────────────────────
    sway_amplitude:  float = 6.0,   # max sway displacement in pixels
    sway_period_min: float = 3.0,   # fastest sway period in seconds
    sway_period_max: float = 7.0,   # slowest sway period in seconds
    # ── Pluck response (the "being touched") ──────────────────────────────────
    pluck_strength:  float = 14.0,  # max pluck bow displacement in pixels
    pluck_decay:     float = 0.91,  # per-frame amplitude decay (lower = settles faster)
    pluck_omega:     float = 18.0,  # oscillation angular frequency rad/s
    pluck_threshold: float = 0.25,  # normalized energy floor to trigger a pluck
    smooth_alpha:    float = 0.08,  # slow-follower alpha for onset detection (0.05–0.15)
    smooth_rise:     float = 1.08,  # how far above smoothed avg counts as "rising"
    cooldown_sec:    float = 0.15,  # seconds before a thread can re-trigger
    # ── Geometry ──────────────────────────────────────────────────────────────
    n_y_points:      int   = 80,    # vertical sample points per thread
    freq_min:        float = 40.0,  # lowest mapped frequency in Hz
    freq_max:        float = 16000.0,  # highest mapped frequency in Hz
    # ── Strand (bowstring fringe) ──────────────────────────────────────────────
    n_strands:       int   = 4,     # sub-threads per visible thread
    fringe_scale:    float = 0.12,  # fringe spread as fraction of pluck displacement
    preview_seconds: float = None,
):
    """
    Each thread is pinned at the top of the frame and hangs freely to the bottom.
    Two displacement layers combine at each Y point:

      sway  : whole-thread lateral drift, shaped as y_norm^0.7
               (top barely moves, bottom sways more, like a hanging cord)

      pluck : decaying oscillation shaped as sin(π·y_norm)
               (bows in the middle, zero at top and bottom, like a plucked string)

    Pluck detection uses a slow-follower (exponential moving average) so that
    any thread rising above its own recent baseline triggers — catching both
    sharp transients and gradual swells, not just hard onsets. A per-thread
    cooldown prevents jitter on sustained passages.

    Frequency assignment is logarithmic (freq_min → freq_max, left → right),
    matching the perceptual spacing of human hearing.
    """
    print(f"[1/4] Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(y) / sr
    print(f"      Duration: {duration:.2f}s  |  SR: {sr} Hz")

    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak
        print(f"      Peak-normalized (original peak: {peak:.4f})")

    samples_per_frame = int(sr / fps)
    total_frames      = int(duration * fps)

    if preview_seconds is not None:
        total_frames = min(total_frames, int(preview_seconds * fps))
        print(f"      [PREVIEW] First {preview_seconds}s -> {total_frames} frames")

    # ── Per-thread frequency analysis ─────────────────────────────────────────
    print("[2/4] Analysing audio ...")

    hop_length    = samples_per_frame
    n_fft         = 2048
    D             = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs         = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_stft_frames = D.shape[1]

    # Logarithmic frequency mapping: thread 0 = freq_min, thread N-1 = freq_max
    f_max_clamped = min(freq_max, freqs[-1])
    thread_freqs  = np.logspace(
        np.log10(freq_min), np.log10(f_max_clamped), n_threads
    )
    thread_bins = np.array(
        [np.argmin(np.abs(freqs - f)) for f in thread_freqs], dtype=np.int32
    )

    # Extract energy per thread per STFT frame
    thread_energy = D[thread_bins, :].copy()   # (n_threads, n_stft_frames)

    # Normalize each thread's energy over its own time history
    # so every frequency band has the same dynamic range.
    t_max = thread_energy.max(axis=1, keepdims=True)
    t_max = np.where(t_max < 1e-8, 1.0, t_max)
    thread_energy /= t_max   # values in [0, 1] per thread

    def get_energies(frame_idx):
        fi = min(frame_idx, n_stft_frames - 1)
        return thread_energy[:, fi]   # (n_threads,)

    # ── Thread geometry ───────────────────────────────────────────────────────
    print(f"      Initializing {n_threads} threads ...")

    rng = np.random.default_rng(42)

    # Base X positions: uniformly spaced with minimal jitter
    margin  = width * 0.04
    x_bases = np.linspace(margin, width - margin, n_threads)
    x_bases += rng.uniform(-2.0, 2.0, n_threads)

    # Y sample points along the thread (top = 0, bottom = height-1)
    y_points = np.linspace(0, height - 1, n_y_points, dtype=np.float32)
    y_norm   = y_points / (height - 1)   # 0 at top, 1 at bottom

    # Sway shape: top is pinned, bottom hangs free → bottom swings more
    sway_shape  = (y_norm ** 0.7).astype(np.float32)

    # Pluck shape: fundamental string mode — bows in the middle
    pluck_shape = np.sin(np.pi * y_norm).astype(np.float32)

    # Per-thread passive sway parameters (randomised so threads never sync)
    sway_phases  = rng.uniform(0, 2 * np.pi, n_threads)
    sway_periods = rng.uniform(sway_period_min, sway_period_max, n_threads)
    sway_amps    = rng.uniform(sway_amplitude * 0.4, sway_amplitude, n_threads)

    # Per-thread visual properties
    brightnesses = rng.uniform(0.5, 1.0, n_threads).astype(np.float32)
    thicknesses  = rng.choice([1, 1, 1, 2], size=n_threads)

    base_color = np.array(thread_color, dtype=np.float32)

    # ── Strand (bowstring fringe) init ────────────────────────────────────────
    strand_rest_offsets = rng.uniform(-0.4, 0.4, (n_threads, n_strands)).astype(np.float32)
    strand_fringe_chars = rng.uniform(-1.0, 1.0, (n_threads, n_strands)).astype(np.float32)
    sort_idx = np.argsort(np.abs(strand_fringe_chars), axis=1)
    strand_fringe_chars = np.take_along_axis(strand_fringe_chars, sort_idx, axis=1)

    strand_b_factors = np.linspace(1.0, 0.55, n_strands, dtype=np.float32)

    # ── Pluck state (per thread) ───────────────────────────────────────────────
    pluck_amp      = np.zeros(n_threads, dtype=np.float64)
    pluck_phase    = np.zeros(n_threads, dtype=np.float64)
    prev_energy    = np.zeros(n_threads, dtype=np.float64)
    smooth_energy  = np.zeros(n_threads, dtype=np.float64)  # slow follower
    cooldown       = np.zeros(n_threads, dtype=np.int32)     # frames until re-trigger

    cooldown_frames = int(fps * cooldown_sec)

    if fisheye:
        print(f"      Building fisheye maps (strength={fisheye_strength}) ...")
        map_x, map_y = build_fisheye_maps(width, height, strength=fisheye_strength)

    # ── Video writer ──────────────────────────────────────────────────────────
    tmp_video = tempfile.mktemp(suffix="_noaudio.mp4")
    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    writer    = cv2.VideoWriter(tmp_video, fourcc, fps, (width, height))

    print(f"[3/4] Rendering {total_frames} frames @ {fps}fps "
          f"({'PREVIEW' if preview_seconds else 'FULL'}) ...")

    dt = 1.0 / fps

    for frame_idx in tqdm(range(total_frames)):
        t        = frame_idx * dt
        energies = get_energies(frame_idx)

        # ── Pluck detection ───────────────────────────────────────────────────
        # smooth_energy is a slow exponential follower of the actual energy.
        # A thread is "rising" whenever its current energy is above threshold
        # AND meaningfully above its own recent average — this catches both
        # sharp transients and gradual swells, not just hard frame-to-frame
        # spikes. A per-thread cooldown prevents jitter on sustained passages.
        smooth_energy = smooth_alpha * energies + (1.0 - smooth_alpha) * smooth_energy

        rising = (
            (energies > pluck_threshold) &
            (energies > smooth_energy * smooth_rise) &
            (cooldown == 0)
        )

        if rising.any():
            pluck_amp[rising]   += energies[rising] * pluck_strength
            pluck_phase[rising]  = 0.0
            cooldown[rising]     = cooldown_frames

        # Tick down cooldown
        cooldown = np.maximum(0, cooldown - 1)

        prev_energy = energies.copy()

        # Advance oscillation and decay amplitude toward zero
        pluck_phase += pluck_omega * dt
        pluck_amp   *= pluck_decay

        # ── Passive sway (vectorised) ─────────────────────────────────────────
        sway_offset  = sway_amps * np.sin(2.0 * np.pi * t / sway_periods + sway_phases)

        # Pluck bow magnitude per thread (sign from oscillation phase)
        pluck_offset = pluck_amp * np.sin(pluck_phase)

        # ── Render ────────────────────────────────────────────────────────────
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(n_threads):
            b_thread = float(brightnesses[i])

            # Shared displacement for this thread (sway + main pluck bow)
            x_disp_base  = sway_offset[i]  * sway_shape
            x_disp_base += pluck_offset[i] * pluck_shape

            # Fringe spread: scales with current pluck amplitude envelope
            # (not the oscillating offset) so spread is always positive and
            # decays cleanly while individual strands still oscillate.
            fringe_envelope = float(pluck_amp[i]) * fringe_scale

            for s in range(n_strands):
                b = b_thread * float(strand_b_factors[s])
                color = (
                    int(base_color[0] * b),
                    int(base_color[1] * b),
                    int(base_color[2] * b),
                )

                x_disp  = x_disp_base.copy()
                x_disp += strand_rest_offsets[i, s]
                x_disp += fringe_envelope * float(strand_fringe_chars[i, s]) * pluck_shape

                x_coords = np.clip(
                    (x_bases[i] + x_disp).astype(np.int32), 0, width - 1
                )

                pts = np.stack(
                    [x_coords, y_points.astype(np.int32)], axis=1
                ).reshape((-1, 1, 2))

                cv2.polylines(
                    canvas, [pts], isClosed=False,
                    color=color, thickness=int(thicknesses[i]),
                    lineType=cv2.LINE_AA,
                )

        # ── CRT phosphor glow ─────────────────────────────────────────────────
        if glow:
            sig   = glow_sigma
            bloom = cv2.GaussianBlur(canvas, (0, 0), sigmaX=sig,       sigmaY=sig)
            halo  = cv2.GaussianBlur(canvas, (0, 0), sigmaX=sig * 2.5, sigmaY=sig * 2.5)
            frame_out = canvas.copy()
            frame_out = cv2.add(frame_out, (halo  * 0.35).clip(0, 255).astype(np.uint8))
            frame_out = cv2.add(frame_out, (bloom * 0.60).clip(0, 255).astype(np.uint8))
        else:
            frame_out = canvas.copy()

        # ── Fisheye — applied to output only, never fed back ──────────────────
        if fisheye:
            frame_out = cv2.remap(
                frame_out, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )

        writer.write(frame_out)

    writer.release()

    # ── Mux audio ─────────────────────────────────────────────────────────────
    print(f"[4/4] Muxing audio ...")

    output_duration_args = []
    if preview_seconds is not None:
        output_duration_args = ["-t", str(preview_seconds)]

    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_video,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        *output_duration_args,
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(tmp_video)

    if result.returncode != 0:
        print("[!] ffmpeg error:")
        print(result.stderr)
        raise RuntimeError("ffmpeg mux failed.")

    print(f"Done -> {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Curtain audio visualizer — 'Lullabye' aesthetic"
    )
    parser.add_argument("--input",  "-i", required=True)
    parser.add_argument("--output", "-o", default="curtain.mp4")
    parser.add_argument("--width",  type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps",    type=int, default=60)

    parser.add_argument("--n_threads",       type=int, default=300,
                        help="Number of threads/strings (default 300).")
    parser.add_argument("--thread_color",    type=int, nargs=3, default=[60, 180, 255],
                        metavar=("B", "G", "R"),
                        help="Thread colour in BGR (default: 60 180 255 — warm amber).")

    parser.add_argument("--sway_amplitude",  type=float, default=6.0,
                        help="Max passive sway displacement in pixels (default 6.0).")
    parser.add_argument("--sway_period_min", type=float, default=3.0,
                        help="Fastest individual sway cycle in seconds (default 3.0).")
    parser.add_argument("--sway_period_max", type=float, default=7.0,
                        help="Slowest individual sway cycle in seconds (default 7.0).")

    parser.add_argument("--pluck_strength",  type=float, default=14.0,
                        help="Max pluck bow displacement in pixels (default 14.0).")
    parser.add_argument("--pluck_decay",     type=float, default=0.91,
                        help="Per-frame pluck amplitude decay 0–1 (default 0.91).")
    parser.add_argument("--pluck_omega",     type=float, default=18.0,
                        help="Pluck oscillation angular frequency rad/s (default 18.0).")
    parser.add_argument("--pluck_threshold", type=float, default=0.08,
                        help="Normalized energy floor to trigger a pluck (default 0.08).")

    parser.add_argument("--smooth_alpha",    type=float, default=0.08,
                        help="Slow-follower alpha for onset detection (default 0.08). "
                             "Lower = longer memory, harder to re-trigger.")
    parser.add_argument("--smooth_rise",     type=float, default=1.08,
                        help="How far above smoothed avg counts as rising (default 1.08). "
                             "1.0 = any energy above avg triggers; 1.2 = needs 20%% rise.")
    parser.add_argument("--cooldown_sec",    type=float, default=0.15,
                        help="Seconds before a thread can re-trigger (default 0.15). "
                             "Prevents jitter on sustained passages.")

    parser.add_argument("--n_y_points",      type=int,   default=80,
                        help="Vertical resolution per thread (default 80).")
    parser.add_argument("--freq_min",        type=float, default=40.0,
                        help="Lowest frequency mapped to left edge in Hz (default 40).")
    parser.add_argument("--freq_max",        type=float, default=16000.0,
                        help="Highest frequency mapped to right edge in Hz (default 16000).")

    parser.add_argument("--n_strands",       type=int,   default=4,
                        help="Sub-threads per visible thread (default 4).")
    parser.add_argument("--fringe_scale",    type=float, default=0.12,
                        help="Fringe spread as fraction of pluck displacement (default 0.12).")

    parser.add_argument("--glow",            action="store_true",  default=True)
    parser.add_argument("--no_glow",         dest="glow",    action="store_false")
    parser.add_argument("--glow_sigma",      type=float, default=5.0,
                        help="Glow blur radius (default 5.0).")

    parser.add_argument("--fisheye",         action="store_true",  default=True)
    parser.add_argument("--no_fisheye",      dest="fisheye", action="store_false")
    parser.add_argument("--fisheye_strength",type=float, default=0.45)

    parser.add_argument("--preview",         action="store_true",
                        help="Render first N seconds only.")
    parser.add_argument("--preview_seconds", type=float, default=10.0)

    args = parser.parse_args()

    render_visualizer(
        audio_path       = args.input,
        output_path      = args.output,
        width            = args.width,
        height           = args.height,
        fps              = args.fps,
        n_threads        = args.n_threads,
        thread_color     = tuple(args.thread_color),
        glow             = args.glow,
        glow_sigma       = args.glow_sigma,
        fisheye          = args.fisheye,
        fisheye_strength = args.fisheye_strength,
        sway_amplitude   = args.sway_amplitude,
        sway_period_min  = args.sway_period_min,
        sway_period_max  = args.sway_period_max,
        pluck_strength   = args.pluck_strength,
        pluck_decay      = args.pluck_decay,
        pluck_omega      = args.pluck_omega,
        pluck_threshold  = args.pluck_threshold,
        smooth_alpha     = args.smooth_alpha,
        smooth_rise      = args.smooth_rise,
        cooldown_sec     = args.cooldown_sec,
        n_y_points       = args.n_y_points,
        freq_min         = args.freq_min,
        freq_max         = args.freq_max,
        n_strands        = args.n_strands,
        fringe_scale     = args.fringe_scale,
        preview_seconds  = args.preview_seconds if args.preview else None,
    )


if __name__ == "__main__":
    main()