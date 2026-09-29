'''
Trains two SVM models from the recordings made by record_gesture_dataset.py:
  - an acoustic model   (claps, taps, snaps...)
  - an ultrasound model (swipes, flicks, covers...)

Steps for each model:
  1. Read the manifest to find every recording and its label
  2. Turn every recording into features (see gesture_features.py)
  3. Test honestly: train on all sessions except the newest one, then test
     on that newest session, which the model has never seen
  4. Print the accuracy and which gestures get confused
  5. Retrain on ALL the data and save the models to gesture_models.joblib

Run:
  python train_gesture_model.py
'''

import csv
import os

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from gesture_features import get_features, load_wav

DATA_FOLDER = "gesture_data"
MODEL_FILE = "gesture_models.joblib"


def load_dataset(mode):
    """Return features (X), labels (y) and session names for every recording of this mode."""
    X = []
    y = []
    sessions = []

    manifest_path = os.path.join(DATA_FOLDER, "manifest.csv")
    with open(manifest_path, newline="") as file:
        for row in csv.DictReader(file):
            if row["mode"] != mode:
                continue
            audio = load_wav(os.path.join(DATA_FOLDER, row["file"]))
            X.append(get_features(audio, mode))
            y.append(row["label"])
            sessions.append(row["session"])

    return np.array(X), np.array(y), np.array(sessions)


def make_model():
    """
    A pipeline runs its steps in order, so we can save and use it as one object:
      1. StandardScaler puts every feature on the same scale
      2. SVC is the Support Vector Machine that learns to separate the gestures
    """
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, class_weight="balanced"),
    )


def test_model(X, y, sessions):
    """Train on some recordings, test on others, and print how well it did."""
    all_sessions = sorted(set(sessions))

    if len(all_sessions) >= 2:
        # Hold out the newest session: a fair test of how the model handles a new day
        test_session = all_sessions[-1]
        is_test = sessions == test_session
        X_train, y_train = X[~is_test], y[~is_test]
        X_test, y_test = X[is_test], y[is_test]
        print(f"Testing on session {test_session}, training on the other {len(all_sessions) - 1}")
    else:
        # Only one session: hold out a random 25% instead (this score will look better than reality)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=0,
        )
        print("Only one session recorded, testing on a random 25% (record more sessions for a fairer test)")

    model = make_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    labels = sorted(set(y))
    print(f"Accuracy: {accuracy_score(y_test, predictions):.0%} (guessing would get {1 / len(labels):.0%})")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Confusion matrix (rows = real gesture, columns = predicted):")
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    for label, row in zip(labels, matrix):
        counts = " ".join(f"{count:3}" for count in row)
        print(f"  {label:<18} {counts}")


def main():
    models = {}

    for mode in ["acoustic", "ultrasound"]:
        print(f"\n========== {mode} ==========")
        X, y, sessions = load_dataset(mode)
        if len(set(y)) < 2:
            print("Need at least 2 different gestures to train, skipping.")
            continue
        print(f"{len(y)} recordings, {len(set(y))} gestures, {X.shape[1]} features each")

        test_model(X, y, sessions)

        # The final model learns from every recording we have
        final_model = make_model()
        final_model.fit(X, y)
        models[mode] = final_model

    joblib.dump(models, MODEL_FILE)
    print(f"\nSaved {list(models)} models to {MODEL_FILE}")


if __name__ == "__main__":
    main()
