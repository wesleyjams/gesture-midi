'''
Guided gesture dataset recorder.

Walks through every gesture in GESTURES, prompting you to perform each one
REPS times. Press SPACE in the spectrogram window to start a take and SPACE
again to stop it. Each take is saved as a raw WAV file and a row is appended
to a manifest CSV, so you can run this script many times and the dataset
keeps growing.

Acoustic gestures (claps, taps...) are recorded with the speaker silent.
Ultrasound gestures (swipes, flicks...) are recorded while the 18-22kHz sweep
from ultrasound_sweep_capture.py plays. WAVs are saved unfiltered; use
apply_mode_filter() at training time to isolate the band for each mode.

Keys (the spectrogram window must be focused):
  SPACE  start / stop a take
  r      redo: delete the last saved take and record it again
  s      skip the rest of the current gesture
  q      quit (everything already saved stays saved)

Usage:
  python record_gesture_dataset.py --list-devices
  python record_gesture_dataset.py --reps 10
  python record_gesture_dataset.py --reps 5 --only "Clap,Swipe Right"
'''

import argparse
import csv
import os
import queue
import re
import time
from collections import Counter
from datetime import datetime

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile
from scipy.signal import butter, chirp, get_window, sosfiltfilt

# -----------------------------
# Gestures: add new gestures here
# -----------------------------
# mode is "acoustic" (speaker silent) or "ultrasound" (sweep plays).
GESTURES = [
    ("Clap", "acoustic"),
    ("Snap", "acoustic"),
    ("Finger Nail Tap", "acoustic"),
    ("Finger Pad Tap", "acoustic"),
    ("Thumb Tap", "acoustic"),
    ("Knuckle Tap", "acoustic"),
    ("Tap Outside Right", "acoustic"),
    ("Tap Outside Left", "acoustic"),
    ("Background", "acoustic"),

    ("Swipe Right", "ultrasound"),
    ("Swipe Left", "ultrasound"),
    ("Covered Right", "ultrasound"),
    ("Covered Left", "ultrasound"),
    ("Flick Up Right", "ultrasound"),
    ("Flick Down Right", "ultrasound"),
    ("Flick Up Left", "ultrasound"),
    ("Flick Down Left", "ultrasound"),
    ("Background Sweep", "ultrasound"),
]

# -----------------------------
# Audio configuration (matches ultrasound_sweep_capture.py)
# -----------------------------
SAMPLE_RATE = 96_000

SWEEP_DURATION = 1.0
LOW_FREQUENCY = 18000.0
HIGH_FREQUENCY = 22050.0
OUTPUT_AMPLITUDE = 0.05 # keep this low to avoid blowing your speakers out!

BLOCK_SIZE = 1024
CHANNELS = 1

# Display settings
FFT_SIZE = 4096
DISPLAY_SECONDS = 10
DISPLAY_MIN_FREQUENCY = 0
DISPLAY_MAX_FREQUENCY = 24000

# CHANGE THESE TO MATCH YOUR SYSTEM (or pass --input-device / --output-device)
INPUT_DEVICE = 2
OUTPUT_DEVICE = 3

# -----------------------------
# Dataset configuration
# -----------------------------
DEFAULT_OUTPUT_DIR = "gesture_data"
DEFAULT_REPS = 10
MIN_TAKE_SECONDS = 0.1

MANIFEST_COLUMNS = [
    "file", "label", "slug", "mode", "session", "take",
    "sample_rate", "duration_s", "sweep_on", "low_hz", "high_hz", "recorded_at",
]


# -----------------------------
# Band separation
# -----------------------------
def apply_mode_filter(audio, mode, sample_rate=SAMPLE_RATE):
    """Return audio filtered to the band that matters for this mode.

    acoustic   -> keep audible content, remove anything near the sweep band
    ultrasound -> keep only the sweep band (LOW_FREQUENCY..HIGH_FREQUENCY)
    """
    # Guard band between 16kHz and the 18kHz sweep start absorbs filter roll-off
    if mode == "acoustic":
        sos = butter(8, 16000, btype="lowpass", fs=sample_rate, output="sos")
    elif mode == "ultrasound":
        high = min(HIGH_FREQUENCY + 500, sample_rate / 2 - 100)
        sos = butter(8, [LOW_FREQUENCY - 500, high], btype="bandpass", fs=sample_rate, output="sos")
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    return sosfiltfilt(sos, audio)


# -----------------------------
# Sweep generation (from ultrasound_sweep_capture.py)
# -----------------------------
def make_sweep():
    sample_count = int(SWEEP_DURATION * SAMPLE_RATE)
    t = np.arange(sample_count) / SAMPLE_RATE

    sweep = chirp(
        t,
        f0=LOW_FREQUENCY,
        f1=HIGH_FREQUENCY,
        t1=SWEEP_DURATION,
        method="logarithmic",
    )

    # Fade the sweep boundaries to prevent clicks
    fade_samples = int(0.02 * SAMPLE_RATE)
    fade = np.sin(np.linspace(0, np.pi / 2, fade_samples)) ** 2

    envelope = np.ones(sample_count)
    envelope[:fade_samples] = fade
    envelope[-fade_samples:] = fade[::-1]

    return (OUTPUT_AMPLITUDE * sweep * envelope).astype(np.float32)


def slugify(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


# -----------------------------
# Recorder
# -----------------------------
class GestureRecorder:
    def __init__(self, gestures, reps, output_dir):
        self.gestures = gestures
        self.reps = reps
        self.output_dir = output_dir
        self.manifest_path = os.path.join(output_dir, "manifest.csv")
        self.session = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Position in the prompt sequence
        self.gesture_index = 0
        self.take = 1
        self.saved_takes = [] # stack of (gesture_index, take, wav_path) for redo
        self.done = False

        # Sweep playback state (read by the audio callback)
        self.sweep = make_sweep()
        self.sweep_position = 0
        self.sweep_enabled = False

        # Recording state. Blocks are collected directly in the callback so
        # nothing is dropped, unlike the display queue below.
        self.is_recording = False
        self.record_blocks = []
        self.record_started_at = 0.0

        # Display-only queue; drops old blocks if plotting falls behind.
        self.display_queue = queue.Queue(maxsize=50)

    # --- Prompt sequence -------------------------------------------------

    @property
    def current(self):
        return self.gestures[self.gesture_index]

    def prompt_text(self):
        if self.done:
            return "Done! Close the window."
        label, mode = self.current
        state = "RECORDING... SPACE to stop" if self.is_recording else "SPACE to record"
        return (
            f"[{self.gesture_index + 1}/{len(self.gestures)}] {label} ({mode})"
            f" - take {self.take}/{self.reps} - {state}"
        )

    def enter_gesture(self):
        """Called whenever the current gesture changes; sets the speaker mode."""
        if self.done:
            self.sweep_enabled = False
            return
        label, mode = self.current
        if mode == "ultrasound" and not self.sweep_enabled:
            # Restart at the sweep's faded-in start so turning on doesn't click
            self.sweep_position = 0
            self.sweep_enabled = True
        elif mode != "ultrasound":
            self.sweep_enabled = False
        print(f"\n>>> {label} ({mode}) - {self.reps} takes. Sweep {'ON' if self.sweep_enabled else 'OFF'}")

    def advance(self):
        self.take += 1
        if self.take > self.reps:
            self.next_gesture()

    def next_gesture(self):
        self.gesture_index += 1
        self.take = 1
        if self.gesture_index >= len(self.gestures):
            self.done = True
        self.enter_gesture()

    # --- Audio callback --------------------------------------------------

    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(status)

        block = indata[:, 0].copy()

        if self.is_recording:
            self.record_blocks.append(block)

        try:
            self.display_queue.put_nowait(block)
        except queue.Full:
            try:
                self.display_queue.get_nowait()
                self.display_queue.put_nowait(block)
            except queue.Empty:
                pass

        if not self.sweep_enabled:
            outdata.fill(0)
            return

        # Generate repeating sweep output
        output = np.empty(frames, dtype=np.float32)
        first_count = min(frames, len(self.sweep) - self.sweep_position)
        output[:first_count] = self.sweep[
            self.sweep_position:self.sweep_position + first_count
        ]

        remaining = frames - first_count
        if remaining:
            output[first_count:] = self.sweep[:remaining]
            self.sweep_position = remaining
        else:
            self.sweep_position += first_count
            if self.sweep_position >= len(self.sweep):
                self.sweep_position = 0

        outdata[:, 0] = output

    # --- Saving ----------------------------------------------------------

    def start_take(self):
        self.record_blocks = []
        self.record_started_at = time.time()
        self.is_recording = True
        print(f"  recording take {self.take}...")

    def stop_take(self):
        self.is_recording = False
        blocks = self.record_blocks
        self.record_blocks = []

        if not blocks:
            print("  no audio captured; discarded.")
            return
        audio = np.concatenate(blocks)
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_TAKE_SECONDS:
            print(f"  take too short ({duration:.2f}s); discarded.")
            return

        label, mode = self.current
        slug = slugify(label)
        wav_dir = os.path.join(self.output_dir, "wav", slug)
        os.makedirs(wav_dir, exist_ok=True)
        wav_path = os.path.join(wav_dir, f"{self.session}_{self.take:03d}.wav")
        wavfile.write(wav_path, SAMPLE_RATE, audio.astype(np.float32))

        self.append_manifest_row({
            "file": os.path.relpath(wav_path, self.output_dir),
            "label": label,
            "slug": slug,
            "mode": mode,
            "session": self.session,
            "take": self.take,
            "sample_rate": SAMPLE_RATE,
            "duration_s": f"{duration:.3f}",
            "sweep_on": int(self.sweep_enabled),
            "low_hz": LOW_FREQUENCY,
            "high_hz": HIGH_FREQUENCY,
            "recorded_at": datetime.fromtimestamp(self.record_started_at).isoformat(timespec="seconds"),
        })
        self.saved_takes.append((self.gesture_index, self.take, wav_path))
        print(f"  saved {wav_path} ({duration:.2f}s)")
        self.advance()

    def append_manifest_row(self, row):
        file_exists = os.path.exists(self.manifest_path)
        with open(self.manifest_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def read_manifest(self):
        if not os.path.exists(self.manifest_path):
            return []
        with open(self.manifest_path, newline="") as f:
            return list(csv.DictReader(f))

    def redo_last(self):
        if not self.saved_takes:
            print("  nothing to redo in this session.")
            return

        gesture_index, take, wav_path = self.saved_takes.pop()
        if os.path.exists(wav_path):
            os.remove(wav_path)

        relative = os.path.relpath(wav_path, self.output_dir)
        rows = [r for r in self.read_manifest() if r["file"] != relative]
        with open(self.manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        changed_gesture = gesture_index != self.gesture_index or self.done
        self.gesture_index = gesture_index
        self.take = take
        self.done = False
        print(f"  deleted {wav_path}; redo take {take}.")
        if changed_gesture:
            self.enter_gesture()

    def print_summary(self):
        session_counts = Counter(
            self.gestures[g][0] for g, _, _ in self.saved_takes
        )
        total_counts = Counter(r["label"] for r in self.read_manifest())

        print("\nSession summary (this session / all sessions):")
        for label, _ in self.gestures:
            print(f"  {label:<20} {session_counts[label]:>4} / {total_counts[label]:>5}")
        print(f"Manifest: {self.manifest_path}")

    # --- Keyboard --------------------------------------------------------

    def on_key_press(self, event):
        if event.key == "q":
            if self.is_recording:
                self.is_recording = False
                print("  recording discarded.")
            plt.close(event.canvas.figure)
            return

        if self.done:
            if event.key == "r":
                self.redo_last()
            return

        if event.key == " ":
            if self.is_recording:
                self.stop_take()
            else:
                self.start_take()
        elif self.is_recording:
            # Ignore redo/skip mid-take so a stray key can't corrupt the take
            return
        elif event.key == "r":
            self.redo_last()
        elif event.key == "s":
            print(f"  skipping rest of {self.current[0]}.")
            self.next_gesture()


# -----------------------------
# Spectrogram display (from ultrasound_sweep_capture.py)
# -----------------------------
frequency_bins = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
min_bin = np.searchsorted(frequency_bins, DISPLAY_MIN_FREQUENCY)
max_bin = np.searchsorted(frequency_bins, DISPLAY_MAX_FREQUENCY)
window = get_window("hann", FFT_SIZE)


def compute_fft(block):
    """Return the positive-frequency magnitude spectrum in dB."""
    if len(block) < FFT_SIZE:
        padded = np.zeros(FFT_SIZE, dtype=np.float32)
        padded[-len(block):] = block
        block = padded
    else:
        block = block[-FFT_SIZE:]

    block = block - np.mean(block)
    spectrum = np.fft.rfft(block * window)
    magnitude = np.abs(spectrum) / np.sum(window)
    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))

    return magnitude_db[min_bin:max_bin]


def disable_conflicting_keymaps():
    """Stop matplotlib from treating our keys as save / quit / home."""
    for keymap in ("keymap.save", "keymap.quit", "keymap.home", "keymap.fullscreen"):
        plt.rcParams[keymap] = [
            k for k in plt.rcParams[keymap] if k not in (" ", "r", "s", "q")
        ]


def run(recorder, input_device, output_device):
    column_count = max(1, int(DISPLAY_SECONDS * SAMPLE_RATE / BLOCK_SIZE))
    spectrogram_data = np.full(
        (max_bin - min_bin, column_count), -100.0, dtype=np.float32,
    )

    disable_conflicting_keymaps()
    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(
        spectrogram_data,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[-DISPLAY_SECONDS, 0, DISPLAY_MIN_FREQUENCY, DISPLAY_MAX_FREQUENCY],
        cmap="magma",
        vmin=-90,
        vmax=-20,
    )
    # Mark the sweep band so you can see acoustic vs ultrasound content
    ax.axhline(LOW_FREQUENCY, color="cyan", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time relative to now (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    title = ax.set_title(recorder.prompt_text())
    fig.colorbar(image, ax=ax).set_label("Magnitude (dB)")

    def update_plot(_frame):
        latest_column = None
        while True:
            try:
                block = recorder.display_queue.get_nowait()
            except queue.Empty:
                break
            latest_column = compute_fft(block)

        if latest_column is not None:
            spectrogram_data[:, :-1] = spectrogram_data[:, 1:]
            spectrogram_data[:, -1] = latest_column
            image.set_data(spectrogram_data)

        title.set_text(recorder.prompt_text())
        title.set_color("red" if recorder.is_recording else "black")
        return (image, title)

    fig.canvas.mpl_connect("key_press_event", recorder.on_key_press)

    stream = sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        channels=(CHANNELS, CHANNELS),
        device=(input_device, output_device),
        callback=recorder.audio_callback,
        latency="low",
    )

    recorder.enter_gesture()
    try:
        with stream:
            # blit=False so the title (outside the axes) redraws each frame
            animation = FuncAnimation(
                fig, update_plot, interval=50, blit=False, cache_frame_data=False,
            )
            plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(fig)
        recorder.print_summary()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="takes per gesture")
    parser.add_argument("--only", help='comma-separated subset of gestures, e.g. "Clap,Swipe Right"')
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR, help="dataset directory")
    parser.add_argument("--input-device", type=int, default=INPUT_DEVICE)
    parser.add_argument("--output-device", type=int, default=OUTPUT_DEVICE)
    parser.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    gestures = GESTURES
    if args.only:
        wanted = {slugify(name) for name in args.only.split(",")}
        gestures = [g for g in GESTURES if slugify(g[0]) in wanted]
        unknown = wanted - {slugify(g[0]) for g in gestures}
        if unknown:
            raise SystemExit(f"Unknown gesture(s): {', '.join(sorted(unknown))}")

    os.makedirs(args.out, exist_ok=True)
    recorder = GestureRecorder(gestures, args.reps, args.out)

    print(f"Session {recorder.session}: {len(gestures)} gestures x {args.reps} takes -> {args.out}/")
    print("Focus the spectrogram window. SPACE record/stop, r redo, s skip, q quit.")
    run(recorder, args.input_device, args.output_device)


if __name__ == "__main__":
    main()
