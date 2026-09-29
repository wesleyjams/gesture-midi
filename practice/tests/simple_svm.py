'''
The simplest sound classifier: practice version.

  1. COLLECT   record short clips of each sound and label them
  2. FEATURES  turn each clip into a short list of numbers
  3. TRAIN     teach an SVM to tell the labels apart, and test it
  4. PREDICT   record new clips and let the model guess

Run (from this folder, with the env active):
  python simple_svm.py
'''

import os

import numpy as np
import sounddevice as sd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SAMPLE_RATE = 44_100 # measurements per second
CLIP_SECONDS = 1.0 # every recording is this long
LABELS = ["clap", "snap", "silence"] # the sounds we want to recognize
SAMPLES_PER_LABEL = 10 # how many examples to record of each
N_BANDS = 20 # how many numbers describe each clip
MAX_FREQUENCY = 10_000 # ignore frequencies above this (Hz)
DATA_FILE = "practice_data.npz"


# 1. COLLECT -------------------------------------------------------------

def record_clip():
    """Record CLIP_SECONDS of audio from the default microphone."""
    audio = sd.rec(int(CLIP_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait() # block until the recording is finished
    return audio[:, 0] # shape (samples, 1) -> (samples,)


def collect_data():
    X = [] # features: one row per clip
    y = [] # labels: one per clip, same order as X
    for label in LABELS:
        print(f"\n--- {label} ---")
        for i in range(SAMPLES_PER_LABEL):
            input(f"[{i + 1}/{SAMPLES_PER_LABEL}] Press Enter, then: {label}")
            X.append(get_features(record_clip()))
            y.append(label)
    return np.array(X), np.array(y)


# 2. FEATURES ------------------------------------------------------------

def get_features(audio):
    """Clip -> N_BANDS numbers: how loud each frequency range is (in dB)."""
    spectrum = np.abs(np.fft.rfft(audio)) # strength of each frequency
    freqs = np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE) # which frequency each value is

    edges = np.linspace(0, MAX_FREQUENCY, N_BANDS + 1) # e.g. 0, 500, 1000, ... Hz
    features = []
    for low, high in zip(edges[:-1], edges[1:]):
        in_band = (freqs >= low) & (freqs < high)
        energy = np.sum(spectrum[in_band] ** 2)
        features.append(10 * np.log10(energy + 1e-10)) # dB; +1e-10 avoids log(0)
    return features


# 3. TRAIN ---------------------------------------------------------------

def train(X, y):
    # Hide 25% of the clips from training so we can test on unseen examples
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0,
    )

    # Put every feature on the same scale (learned from training data only)
    scaler = StandardScaler().fit(X_train)

    model = SVC(kernel="rbf")
    model.fit(scaler.transform(X_train), y_train)

    accuracy = model.score(scaler.transform(X_test), y_test)
    print(f"\nTest accuracy: {accuracy:.0%} (chance is {1 / len(LABELS):.0%})")
    return scaler, model


# 4. PREDICT -------------------------------------------------------------

def predict_loop(scaler, model):
    while True:
        command = input("\nPress Enter and make a sound (q to quit): ")
        if command == "q":
            break
        features = get_features(record_clip())
        guess = model.predict(scaler.transform([features]))[0]
        print(f"I heard: {guess}")


def main():
    if os.path.exists(DATA_FILE) and input(f"Use saved {DATA_FILE}? [y/n] ") == "y":
        data = np.load(DATA_FILE)
        X, y = data["X"], data["y"]
    else:
        X, y = collect_data()
        np.savez(DATA_FILE, X=X, y=y)
        print(f"Saved {len(y)} clips to {DATA_FILE}")

    scaler, model = train(X, y)
    predict_loop(scaler, model)


if __name__ == "__main__":
    main()
