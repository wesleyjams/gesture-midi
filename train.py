'''
Step 2: train the models.

Reads data/manifest.csv, cuts a clip at every gesture found by record.py,
turns each clip into features, and trains one SVM per mode:
  acoustic   (sounds: taps, claps...)
  ultrasound (silent hand movements: swipes, covers...)

To test honestly, each model is first trained WITHOUT the newest session and
tested on it, like playing on a new day. Then it is retrained on everything
and saved to models.joblib for play.py.

  python train.py
'''

import csv
import os

import joblib
import numpy as np
from scipy.io import wavfile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from gesture_tools import DATA_FOLDER, MODEL_FILE, cut_clip, get_features


def load_examples(mode):
    """Return features (X), labels (y) and sessions for every example of this mode."""
    X = []
    y = []
    sessions = []
    recordings = {} # file name -> audio, so each WAV is only read once

    with open(os.path.join(DATA_FOLDER, "manifest.csv"), newline="") as file:
        for row in csv.DictReader(file):
            if row["mode"] != mode:
                continue
            if row["file"] not in recordings:
                sample_rate, audio = wavfile.read(os.path.join(DATA_FOLDER, row["file"]))
                recordings[row["file"]] = audio

            clip = cut_clip(recordings[row["file"]], int(row["onset"]), mode)
            X.append(get_features(clip, mode))
            y.append(row["label"])
            sessions.append(row["session"])

    return np.array(X), np.array(y), np.array(sessions)


def make_model():
    """
    A pipeline runs its steps in order and can be saved as one object:
      1. StandardScaler puts every feature on the same scale
      2. SVC is the Support Vector Machine that learns to tell gestures apart
    """
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, class_weight="balanced"))


def test_model(X, y, sessions):
    """Train without the newest session, test on it, and print the results."""
    newest = sorted(set(sessions))[-1]
    is_test = sessions == newest
    if is_test.all():
        print("Only one session so far: record another one to get a test score.")
        return

    model = make_model()
    model.fit(X[~is_test], y[~is_test])
    predictions = model.predict(X[is_test])
    real = y[is_test]

    labels = sorted(set(y))
    print(f"Tested on session {newest}: {accuracy_score(real, predictions):.0%} correct "
          f"(guessing would get {1 / len(labels):.0%})")
    print(classification_report(real, predictions, zero_division=0))

    print("Confusion matrix (rows = real gesture, columns = what the model said):")
    for label, row in zip(labels, confusion_matrix(real, predictions, labels=labels)):
        print(f"  {label:<16}" + "".join(f"{count:4}" for count in row))


def main():
    models = {}
    for mode in ["acoustic", "ultrasound"]:
        print(f"\n========== {mode} ==========")
        X, y, sessions = load_examples(mode)
        if len(set(y)) < 2:
            print("Need examples of at least 2 gestures, skipping.")
            continue
        print(f"{len(y)} examples of {len(set(y))} gestures from {len(set(sessions))} session(s)")

        test_model(X, y, sessions)

        # The final model learns from every example
        final_model = make_model()
        final_model.fit(X, y)
        models[mode] = final_model

    joblib.dump(models, MODEL_FILE)
    print(f"\nSaved {' and '.join(models)} models to {MODEL_FILE}. Next step: python play.py")


if __name__ == "__main__":
    main()
