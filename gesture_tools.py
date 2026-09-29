'''
gesture_tools.py - the shared toolbox for the gesture MIDI instrument.

record.py, train.py and play.py all import from this file, so recording,
training and playing use exactly the same settings and the same code.

Read it top to bottom:
  1. SETTINGS   every number you might want to change
  2. GESTURES   your gestures and the MIDI message each one sends
  3. SOUND      the ultrasound sweep, the audio stream, measuring loudness
  4. DETECTOR   finds WHEN something happens (the SVM decides WHAT it was)
  5. FEATURES   turns a clip of audio into numbers for the SVM
'''

import queue

import numpy as np
import sounddevice as sd
from scipy.signal import chirp


# =====================================================================
# 1. SETTINGS
# =====================================================================

# Audio
SAMPLE_RATE = 96_000 # samples per second (96 kHz lets us capture ultrasound)
BLOCK_SIZE = 512 # samples per block, about 5 ms. Smaller = faster reactions
INPUT_DEVICE = 2 # microphone. To see your devices run: python -m sounddevice
OUTPUT_DEVICE = 3 # speakers

# Ultrasound sweep: an inaudible tone gliding from low to high, repeating forever
SWEEP_LOW = 18_000 # Hz
SWEEP_HIGH = 22_050 # Hz
SWEEP_DURATION = 0.05 # seconds per sweep
SWEEP_VOLUME = 0.05 # keep this low to protect your speakers
SWEEP_SAMPLES = int(SWEEP_DURATION * SAMPLE_RATE)

# Frequency bands we measure (Hz). 21 edges = 20 bands, 9 edges = 8 bands.
# Acoustic bands are spaced like piano octaves and stop well below the sweep.
ACOUSTIC_LOW = 200
ACOUSTIC_HIGH = 16_000
ACOUSTIC_BANDS = np.geomspace(ACOUSTIC_LOW, ACOUSTIC_HIGH, 21)
ULTRASOUND_BANDS = np.linspace(SWEEP_LOW, SWEEP_HIGH, 9)

# Detector
CALIBRATION_SECONDS = 2.0 # hands away and quiet while the room is measured
ACOUSTIC_THRESHOLD = 10 # dB louder than the quiet room = a sound started
ULTRASOUND_THRESHOLD = 3 # dB of average change in the sweep = your hand is there
ACOUSTIC_COOLDOWN = 0.2 # seconds before another sound can start
ULTRASOUND_RELEASE = 0.15 # seconds the sweep must be calm before a hand gesture ends
QUIET_AFTER_SOUND = 1.0 # loud sounds disturb the sweep too, so ignore it briefly

# Clips: how much audio the SVM looks at around the moment something started
ACOUSTIC_BEFORE = 0.01 # seconds
ACOUSTIC_AFTER = 0.1 # also the delay before a sound's MIDI message is sent
ULTRASOUND_BEFORE = 0.2 # the resting sweep, used as the reference
ULTRASOUND_AFTER = 0.4 # the hand gesture itself
ULTRASOUND_SLICES = 4 # split the gesture into slices to capture motion over time

# MIDI
MIDI_PORT_NAME = "Laptop Gestures"
HIT_NOTE = 60 # sent instantly for every sound, before it is classified (None = off)
CONTROLLER_CC = 20 # follows how much your hand changes the sweep (None = off)
CONTROLLER_RANGE = 10 # dB of sweep change that sends the maximum value (127)
VELOCITY_RANGE = 30 # dB above the room that sends the maximum velocity (127)
NOTE_LENGTH = 0.1 # seconds between note on and note off

# Files
DATA_FOLDER = "data"
MODEL_FILE = "models.joblib"


# =====================================================================
# 2. GESTURES
# =====================================================================
# One line per gesture: (label, mode, MIDI message)
#   mode: "acoustic"   the gesture makes a sound (tap, clap...)
#         "ultrasound" a silent hand movement over the laptop (swipe, cover...)
#   MIDI: ("note", number)  plays a note once
#         ("hold", number)  CC number goes to 127 while the gesture lasts, then back to 0
#         None              send nothing (for "Not a Gesture")
# Keep a "Not a Gesture" in each mode: record accidental bumps and movements
# for it, so the instrument learns to ignore them.

GESTURES = [
    ("Clap", "acoustic", ("note", 36)),
    ("Nail Tap", "acoustic", ("note", 38)),
    ("Knuckle Tap", "acoustic", ("note", 40)),
    ("Not a Gesture", "acoustic", None),

    ("Covered", "ultrasound", ("hold", 21)),
    ("Swipe Right", "ultrasound", ("note", 48)),
    ("Swipe Left", "ultrasound", ("note", 50)),
    ("Not a Gesture", "ultrasound", None),
]


def find_midi(label, mode):
    """Look up the MIDI message for a gesture."""
    for gesture_label, gesture_mode, midi in GESTURES:
        if gesture_label == label and gesture_mode == mode:
            return midi
    return None


# =====================================================================
# 3. SOUND
# =====================================================================

def make_sweep():
    """One sweep from SWEEP_LOW to SWEEP_HIGH, faded in and out so it doesn't click."""
    t = np.arange(SWEEP_SAMPLES) / SAMPLE_RATE # the time of every sample
    sweep = chirp(t, f0=SWEEP_LOW, f1=SWEEP_HIGH, t1=SWEEP_DURATION, method="logarithmic")

    fade = int(0.005 * SAMPLE_RATE) # 5 ms
    envelope = np.ones(SWEEP_SAMPLES)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    return (SWEEP_VOLUME * sweep * envelope).astype(np.float32)


def start_audio():
    """
    Start playing the sweep and listening to the microphone.
    Returns (stream, blocks). `blocks` is a queue that fills up with microphone
    blocks; take them out with blocks.get(). Call stream.stop() when done.
    """
    sweep = make_sweep()
    blocks = queue.Queue()
    position = 0 # where we are in the sweep

    def callback(indata, outdata, frames, time_info, status):
        nonlocal position # lets this function change `position` above
        blocks.put(indata[:, 0].copy())

        # Play the next `frames` samples of the sweep, wrapping back to the start
        indexes = (position + np.arange(frames)) % len(sweep)
        outdata[:, 0] = sweep[indexes]
        position = (position + frames) % len(sweep)

    stream = sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        channels=1,
        device=(INPUT_DEVICE, OUTPUT_DEVICE),
        callback=callback,
    )
    stream.start()
    return stream, blocks


def collect_audio(blocks, seconds):
    """Wait for the next `seconds` of microphone audio and return it as one array."""
    pieces = []
    for _ in range(int(seconds * SAMPLE_RATE / BLOCK_SIZE)):
        pieces.append(blocks.get())
    return np.concatenate(pieces)


def collect_waiting_audio(blocks):
    """Return all the microphone audio waiting in the queue (empties it)."""
    pieces = []
    while not blocks.empty():
        pieces.append(blocks.get())
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces)


def band_levels(audio, band_edges, smooth_edges=True):
    """
    How loud each frequency band is, in dB.
    band_edges = [200, 400, 800] gives two bands: 200-400 Hz and 400-800 Hz.

    smooth_edges fades the audio in and out (a "Hanning window") so loud
    frequencies don't smear into other bands. Use it for sounds. Turn it off
    for the sweep: we always measure whole sweeps, and fading would count the
    frequencies in the middle more than the ones at the edges.
    """
    if smooth_edges:
        audio = audio * np.hanning(len(audio))

    # FFT: the strength of every frequency in the audio
    spectrum = np.abs(np.fft.rfft(audio)) ** 2
    frequencies = np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)

    levels = []
    for low, high in zip(band_edges[:-1], band_edges[1:]):
        in_band = (frequencies >= low) & (frequencies < high)
        power = spectrum[in_band].mean()
        levels.append(10 * np.log10(power + 1e-12)) # dB; +1e-12 avoids log(0)
    return np.array(levels)


def sound_level(block):
    """How loud a block is in the acoustic range (the sweep is not included), in dB."""
    return band_levels(block, [ACOUSTIC_LOW, ACOUSTIC_HIGH])[0]


def sweep_levels(audio):
    """How loud each ultrasound band is, in dB. `audio` should hold a whole number of sweeps."""
    return band_levels(audio, ULTRASOUND_BANDS, smooth_edges=False)


def calibrate(blocks):
    """
    Measure the quiet room and the resting sweep.
    Returns (room_level, resting_sweep):
      room_level     how loud the room is when nothing happens (dB)
      resting_sweep  the level of each ultrasound band with no hand nearby (dB)
    """
    print(f"Calibrating for {CALIBRATION_SECONDS:.0f} s: hands away and quiet please...")
    collect_audio(blocks, 0.5) # skip the first half second while the speakers start
    audio = collect_audio(blocks, CALIBRATION_SECONDS)

    block_levels = []
    for start in range(0, len(audio) - BLOCK_SIZE, BLOCK_SIZE):
        block_levels.append(sound_level(audio[start:start + BLOCK_SIZE]))
    room_level = np.median(block_levels)

    every_sweep = []
    for start in range(0, len(audio) - SWEEP_SAMPLES, SWEEP_SAMPLES):
        every_sweep.append(sweep_levels(audio[start:start + SWEEP_SAMPLES]))
    resting_sweep = np.mean(every_sweep, axis=0)

    # How much the sweep wobbles on its own. Should be well below ULTRASOUND_THRESHOLD.
    wobble = np.mean([np.mean(np.abs(levels - resting_sweep)) for levels in every_sweep])
    print(f"  room level {room_level:.1f} dB, sweep wobble {wobble:.1f} dB "
          f"(ultrasound threshold {ULTRASOUND_THRESHOLD} dB)")
    if wobble > ULTRASOUND_THRESHOLD / 2:
        print("  WARNING: the sweep is unsteady. Check the speaker volume and keep hands away.")
    return room_level, resting_sweep


# =====================================================================
# 4. DETECTOR
# =====================================================================

class Detector:
    """
    Watches the audio one block at a time and reports when something starts.

    The same detector splits recordings into examples (record.py) and listens
    live (play.py), so training clips and live clips are cut in exactly the
    same way.
    """

    def __init__(self, room_level, resting_sweep):
        self.room_level = room_level
        self.resting_sweep = resting_sweep
        self.samples_heard = 0 # counts every sample since the detector started
        self.last_sound_start = -10 * SAMPLE_RATE # "a long time ago"
        self.sound_active = False # True while a sound is above the threshold
        self.hand_active = False
        self.calm_samples = 0
        self.sweep_change = 0.0 # how much the sweep differs from rest right now (dB)

    def update(self, recent_audio):
        """
        Call once for every new block. `recent_audio` must end with the newest block.
        Returns a list of events. Each event is (mode, what, onset, strength):
          mode      "acoustic" or "ultrasound"
          what      "start" or "end"
          onset     the sample number where it happened
          strength  how far above the threshold's reference it was (dB)
        """
        self.samples_heard += BLOCK_SIZE
        onset = self.samples_heard - BLOCK_SIZE # the first sample of the newest block
        events = []

        # ---- Acoustic: did the newest block just become much louder than the room? ----
        # A sound only counts when it RISES past the threshold. It must fall back
        # below it before another sound can start, so one long sound = one event.
        loudness = sound_level(recent_audio[-BLOCK_SIZE:]) - self.room_level
        time_since_sound = self.samples_heard - self.last_sound_start
        if loudness > ACOUSTIC_THRESHOLD:
            if not self.sound_active and time_since_sound > ACOUSTIC_COOLDOWN * SAMPLE_RATE:
                self.last_sound_start = onset
                events.append(("acoustic", "start", onset, loudness))
            self.sound_active = True
        else:
            self.sound_active = False

        # ---- Ultrasound: pause briefly after a loud sound ----
        if self.samples_heard - self.last_sound_start < QUIET_AFTER_SOUND * SAMPLE_RATE:
            return events
        if len(recent_audio) < SWEEP_SAMPLES:
            return events # not enough audio yet for a whole sweep

        # How much has the latest full sweep changed from the resting sweep?
        latest_sweep = sweep_levels(recent_audio[-SWEEP_SAMPLES:])
        self.sweep_change = np.mean(np.abs(latest_sweep - self.resting_sweep))

        if self.sweep_change > ULTRASOUND_THRESHOLD:
            self.calm_samples = 0
            if not self.hand_active:
                self.hand_active = True
                events.append(("ultrasound", "start", onset, self.sweep_change))
        elif self.hand_active:
            self.calm_samples += BLOCK_SIZE
            if self.calm_samples > ULTRASOUND_RELEASE * SAMPLE_RATE:
                self.hand_active = False
                events.append(("ultrasound", "end", onset, 0.0))

        return events


def find_events(audio, room_level, resting_sweep):
    """Run a recording through a fresh Detector, block by block, exactly like live."""
    detector = Detector(room_level, resting_sweep)
    events = []
    for end in range(BLOCK_SIZE, len(audio) + 1, BLOCK_SIZE):
        events += detector.update(audio[:end])
    return events


# =====================================================================
# 5. FEATURES
# =====================================================================

def cut_clip(audio, onset, mode, audio_start=0):
    """
    Cut the clip the SVM looks at around an onset.
    audio_start is the sample number of audio[0]. It is 0 for a recording,
    and something larger when `audio` is only the most recent part of a live stream.
    Returns None if the audio doesn't contain the whole clip.
    """
    if mode == "acoustic":
        before, after = ACOUSTIC_BEFORE, ACOUSTIC_AFTER
    else:
        before, after = ULTRASOUND_BEFORE, ULTRASOUND_AFTER

    start = onset - int(before * SAMPLE_RATE) - audio_start
    end = onset + int(after * SAMPLE_RATE) - audio_start
    if start < 0 or end > len(audio):
        return None
    return audio[start:end]


def get_features(clip, mode):
    """Turn a clip into the list of numbers the SVM learns from."""
    if mode == "acoustic":
        # The start (attack) and end (decay) of the sound: 20 band levels each = 40 numbers
        attack, decay = np.array_split(clip, 2)
        return np.concatenate([
            band_levels(attack, ACOUSTIC_BANDS),
            band_levels(decay, ACOUSTIC_BANDS),
        ])

    # Ultrasound: how the hand changed each band of the sweep, slice by slice.
    # Measuring the CHANGE from the resting sweep (instead of the raw level) cancels
    # out differences in speaker volume and laptop position between sessions.
    # The settings make every part a whole number of sweeps (0.2 s = 4, each 0.1 s slice = 2).
    before_length = int(ULTRASOUND_BEFORE * SAMPLE_RATE)
    resting = sweep_levels(clip[:before_length])
    slices = np.array_split(clip[before_length:], ULTRASOUND_SLICES)

    features = []
    for piece in slices:
        features.append(sweep_levels(piece) - resting)
    return np.concatenate(features) # 4 slices x 8 bands = 32 numbers
