'''
Turns a clip of audio into a list of numbers (a "feature vector")
that the SVM can learn from.

Both train_gesture_model.py and run_gesture_model.py use get_features(),
so the model sees audio processed exactly the same way in training and
when running live. If you change anything here, retrain the model.

How it works:
  1. Filter the audio so we only keep the frequencies for this mode
       acoustic   -> below 16 kHz (claps, taps...)
       ultrasound -> around the 18-22 kHz sweep (swipes, flicks...)
  2. Cut out the important part of the clip
       acoustic   -> 0.3 seconds starting just before the loudest moment
       ultrasound -> one full sweep from the middle of the clip
  3. Measure how loud each frequency band is over time
  4. For each band keep two numbers: the average and the loudest value
'''

import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

from record_gesture_dataset import SAMPLE_RATE, SWEEP_DURATION, apply_mode_filter

# How much audio each mode looks at (seconds)
ACOUSTIC_WINDOW = 0.3
ACOUSTIC_BEFORE_PEAK = 0.03 # include a little audio before the peak
ULTRASOUND_WINDOW = SWEEP_DURATION # one full sweep, so every frequency is heard once

# Frequency band edges (Hz). 21 edges make 20 bands.
# Acoustic bands are spaced like piano octaves: narrow at low pitches, wide at high.
# Ultrasound bands are evenly spaced across the sweep.
ACOUSTIC_BANDS = np.geomspace(200, 16000, 21)
ULTRASOUND_BANDS = np.linspace(17500, 22050, 21)


def load_wav(path):
    """Read a WAV file and return its samples as a numpy array."""
    sample_rate, audio = wavfile.read(path)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path} is {sample_rate} Hz, expected {SAMPLE_RATE} Hz")
    return audio.astype(np.float32)


def cut_window(audio, start, length):
    """Return `length` samples starting at `start`, padding with silence if the clip is too short."""
    window = audio[start:start + length]
    if len(window) < length:
        missing = length - len(window)
        window = np.pad(window, (0, missing))
    return window


def band_loudness(audio, band_edges):
    """
    Return a 2D array: one row per band, one column per moment in time.
    Each value is how loud that band was at that moment, in decibels.
    """
    # frequencies: the frequency of each row of `power`
    # power:       energy at every (frequency, time) point
    frequencies, times, power = spectrogram(audio, fs=SAMPLE_RATE, nperseg=2048)

    rows = []
    for low, high in zip(band_edges[:-1], band_edges[1:]):
        in_band = (frequencies >= low) & (frequencies < high)
        band_power = power[in_band].mean(axis=0) # average across the band's frequencies
        rows.append(10 * np.log10(band_power + 1e-12)) # convert to dB; +1e-12 avoids log(0)
    return np.array(rows)


def get_features(audio, mode):
    """Turn raw audio into a feature vector for the given mode ("acoustic" or "ultrasound")."""
    # 1. Keep only this mode's frequencies. Filtering before cutting keeps the
    #    filter's start-up wobble at the edges of the clip, outside our window.
    audio = apply_mode_filter(audio, mode)

    # 2. Cut out the important part
    if mode == "acoustic":
        loudest_sample = np.argmax(np.abs(audio))
        start = max(0, loudest_sample - int(ACOUSTIC_BEFORE_PEAK * SAMPLE_RATE))
        window = cut_window(audio, start, int(ACOUSTIC_WINDOW * SAMPLE_RATE))
        band_edges = ACOUSTIC_BANDS
    else:
        length = int(ULTRASOUND_WINDOW * SAMPLE_RATE)
        start = max(0, (len(audio) - length) // 2)
        window = cut_window(audio, start, length)
        band_edges = ULTRASOUND_BANDS

    # 3. Loudness of each band over time
    loudness = band_loudness(window, band_edges)

    # 4. Average and loudest value of each band -> 40 numbers
    average = loudness.mean(axis=1)
    loudest = loudness.max(axis=1)
    return np.concatenate([average, loudest])
