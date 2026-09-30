'''
gesture_tools.py - the shared toolbox for the gesture MIDI instrument.

record.py, train.py and play.py all import from this file, so recording,
training and playing use exactly the same settings and the same code.

Read it top to bottom:
  1. SETTINGS   every number you might want to change
  2. GESTURES   your gestures and the MIDI note each one plays
  3. SOUND      the microphone, measuring loudness
  4. DETECTOR   finds WHEN a sound starts (the SVM decides WHAT it was)
  5. FEATURES   turns a clip of audio into numbers for the SVM
'''

import queue

import numpy as np
import sounddevice as sd


# =====================================================================
# 1. SETTINGS
# =====================================================================

# Audio
SAMPLE_RATE = 96_000 # samples per second (your recordings use this, so keep it)
BLOCK_SIZE = 512 # samples per block, about 5 ms. Smaller = faster reactions
INPUT_DEVICE = 3 # microphone. To see your devices run: python -m sounddevice

# Frequency bands we measure (Hz): 21 edges = 20 bands, spaced like piano octaves
LOWEST_FREQUENCY = 200
HIGHEST_FREQUENCY = 16_000
BANDS = np.geomspace(LOWEST_FREQUENCY, HIGHEST_FREQUENCY, 21)

# Detector
CALIBRATION_SECONDS = 2.0 # stay quiet while the room is measured
SOUND_THRESHOLD = 20 # dB louder than the quiet room = a sound started
SOUND_COOLDOWN = 1.0 # seconds before another sound can start

# Clips: how much audio the SVM looks at around the moment a sound started
CLIP_BEFORE = 0.01 # seconds
CLIP_AFTER = 0.1 # also the delay before a sound's MIDI note is sent

# MIDI
MIDI_PORT_NAME = "Laptop Gestures"
HIT_NOTE = None # e.g. 60: sent instantly for every sound, before it is classified.
                # Off, because key and trackpad clicks would trigger it too.
VELOCITY_RANGE = 30 # dB above the room that sends the maximum velocity (127)
NOTE_LENGTH = 0.1 # seconds between note on and note off

# Files
DATA_FOLDER = "data"
MODEL_FILE = "model.joblib"


# =====================================================================
# 2. GESTURES
# =====================================================================
# One line per gesture: (label, MIDI note). None = play nothing.
# To remove a gesture, delete its line: train.py skips recordings of it.
# "Not a Gesture": record accidental bumps and noises for it, so the
# instrument learns to ignore them.
# "Key Click" and "Trackpad Click" work the same way: the instrument learns to
# recognise them, then ignores them, so you can play Ableton's computer keyboard.

GESTURES = [
    ("Clap", 36),
    ("Finger Tap", 38), # any finger, thumb included: they sound too alike to tell apart
    ("Palm Tap", 41),
    ("Screen Tap", 45),
    ("Key Click", None),
    ("Trackpad Click", None),
    ("Not a Gesture", None),
]


def find_note(label):
    """Look up the MIDI note for a gesture (None = play nothing)."""
    for gesture_label, note in GESTURES:
        if gesture_label == label:
            return note
    return None


# =====================================================================
# 3. SOUND
# =====================================================================

def start_audio():
    """
    Start listening to the microphone.
    Returns (stream, blocks). `blocks` is a queue that fills up with microphone
    blocks; take them out with blocks.get(). Call stream.stop() when done.
    """
    blocks = queue.Queue()

    def callback(indata, frames, time_info, status):
        blocks.put(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        channels=1,
        device=INPUT_DEVICE,
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


def band_levels(audio, band_edges):
    """
    How loud each frequency band is, in dB.
    band_edges = [200, 400, 800] gives two bands: 200-400 Hz and 400-800 Hz.
    """
    # Fade the audio in and out (a "Hanning window") so loud frequencies
    # don't smear into other bands
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
    """How loud a block is, in dB."""
    return band_levels(block, [LOWEST_FREQUENCY, HIGHEST_FREQUENCY])[0]


def calibrate(blocks):
    """Measure how loud the room is when nothing happens (dB)."""
    print(f"Calibrating for {CALIBRATION_SECONDS:.0f} s: quiet please...")
    collect_audio(blocks, 0.5) # skip the first half second while the microphone starts
    audio = collect_audio(blocks, CALIBRATION_SECONDS)

    block_levels = []
    for start in range(0, len(audio) - BLOCK_SIZE, BLOCK_SIZE):
        block_levels.append(sound_level(audio[start:start + BLOCK_SIZE]))
    room_level = np.median(block_levels)

    print(f"  room level {room_level:.1f} dB (threshold: {SOUND_THRESHOLD} dB above it)")
    return room_level


# =====================================================================
# 4. DETECTOR
# =====================================================================

class Detector:
    """
    Listens one block at a time and reports when a sound starts.

    The same detector splits recordings into examples (record.py) and listens
    live (play.py), so training clips and live clips are cut in exactly the
    same way.
    """

    def __init__(self, room_level):
        self.room_level = room_level
        self.samples_heard = 0 # counts every sample since the detector started
        self.last_sound_start = -10 * SAMPLE_RATE # "a long time ago"
        self.sound_active = False # True while a sound is above the threshold

    def update(self, block):
        """
        Call once for every new block.
        Returns (onset, loudness) if a sound just started, otherwise None:
          onset     the sample number where it started
          loudness  how far above the quiet room it was (dB)
        """
        self.samples_heard += BLOCK_SIZE
        onset = self.samples_heard - BLOCK_SIZE # the first sample of this block

        # A sound only counts when it RISES past the threshold. It must fall back
        # below it before another sound can start, so one long sound = one event.
        loudness = sound_level(block) - self.room_level
        if loudness <= SOUND_THRESHOLD:
            self.sound_active = False
            return None

        was_active = self.sound_active
        self.sound_active = True
        cooling_down = onset - self.last_sound_start < SOUND_COOLDOWN * SAMPLE_RATE
        if was_active or cooling_down:
            return None

        self.last_sound_start = onset
        return onset, loudness


def find_sounds(audio, room_level):
    """Run a recording through a fresh Detector, block by block, exactly like live.
    Returns a list of (onset, loudness)."""
    detector = Detector(room_level)
    sounds = []
    for start in range(0, len(audio) - BLOCK_SIZE + 1, BLOCK_SIZE):
        sound = detector.update(audio[start:start + BLOCK_SIZE])
        if sound is not None:
            sounds.append(sound)
    return sounds


# =====================================================================
# 5. FEATURES
# =====================================================================

def cut_clip(audio, onset, audio_start=0):
    """
    Cut the clip the SVM looks at around an onset.
    audio_start is the sample number of audio[0]. It is 0 for a recording,
    and something larger when `audio` is only the most recent part of a live stream.
    Returns None if the audio doesn't contain the whole clip.
    """
    start = onset - int(CLIP_BEFORE * SAMPLE_RATE) - audio_start
    end = onset + int(CLIP_AFTER * SAMPLE_RATE) - audio_start
    if start < 0 or end > len(audio):
        return None
    return audio[start:end]


def get_features(clip):
    """
    Turn a clip into the list of numbers the SVM learns from:
    the start (attack) and end (decay) of the sound, 20 band levels each = 40 numbers.
    """
    attack, decay = np.array_split(clip, 2)
    return np.concatenate([band_levels(attack, BANDS), band_levels(decay, BANDS)])
