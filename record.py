'''
Step 1: record examples of every gesture.

For each gesture in GESTURES (see gesture_tools.py):
  1. Press Enter to start recording
  2. Do the gesture about 20 times, with a short pause between each one
  3. Press Enter to stop
The detector then finds each sound in the recording, exactly the way
play.py will find them live, and tells you how many it found.

Each recording is saved as a WAV file in data/, and every sound found in it
becomes one row in data/manifest.csv (which file, what label, and where).

Run it several times on different days: more sessions = a better instrument.
  python record.py
'''

import csv
import os
import time

from scipy.io import wavfile

from gesture_tools import (
    DATA_FOLDER, GESTURES, SAMPLE_RATE,
    calibrate, collect_waiting_audio, cut_clip, find_sounds, start_audio,
)

# Ignore anything this close to the start or end of a recording:
# that's where the Enter key clicks are.
EDGE_SECONDS = 0.3

MANIFEST_FILE = os.path.join(DATA_FOLDER, "manifest.csv")


def split_recording(audio, room_level):
    """Find where each sound starts in a recording. Returns a list of sample numbers."""
    edge = int(EDGE_SECONDS * SAMPLE_RATE)
    onsets = []
    for onset, loudness in find_sounds(audio, room_level):
        if onset < edge or onset > len(audio) - edge:
            continue # too close to an Enter key click
        if cut_clip(audio, onset) is None:
            continue # not enough audio around it to make a full clip
        onsets.append(onset)
    return onsets


def save_recording(audio, label, session, onsets):
    """Save the WAV file and add one manifest row per sound found."""
    file_name = f"{session}_{label.replace(' ', '_').lower()}.wav"
    wavfile.write(os.path.join(DATA_FOLDER, file_name), SAMPLE_RATE, audio)

    new_file = not os.path.exists(MANIFEST_FILE)
    with open(MANIFEST_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(["file", "label", "session", "onset"])
        for onset in onsets:
            writer.writerow([file_name, label, session, onset])


def main():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    session = time.strftime("%Y%m%d-%H%M%S") # e.g. 20260929-143000

    stream, blocks = start_audio()
    room_level = calibrate(blocks)

    for number, (label, note) in enumerate(GESTURES, start=1):
        while True:
            print(f"\n[{number}/{len(GESTURES)}] {label}")
            answer = input("  Enter = start recording, s = skip: ")
            if answer == "s":
                break

            collect_waiting_audio(blocks) # throw away everything before the start
            input("  RECORDING. Do the gesture ~20 times with a pause between each. Enter = stop ")
            audio = collect_waiting_audio(blocks)

            onsets = split_recording(audio, room_level)
            answer = input(f"  Found {len(onsets)} in {len(audio) / SAMPLE_RATE:.0f} s. "
                           "Enter = save, r = record again: ")
            if answer == "r":
                continue

            save_recording(audio, label, session, onsets)
            print(f"  Saved {len(onsets)} examples of {label}.")
            break

    stream.stop()
    print(f"\nSession {session} done. Next step: python train.py")


if __name__ == "__main__":
    main()
