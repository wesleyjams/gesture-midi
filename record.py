'''
Step 1: record examples of every gesture.

For each gesture in GESTURES (see gesture_tools.py):
  1. Press Enter to start recording
  2. Do the gesture about 20 times, with a short pause between each one
  3. Press Enter to stop
The detector then finds each gesture in the recording, exactly the way
play.py will find them live, and tells you how many it found.

Each recording is saved as a WAV file in data/, and every gesture found in it
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
    calibrate, collect_waiting_audio, cut_clip, find_events, start_audio,
)

# Ignore anything this close to the start or end of a recording:
# that's where the Enter key clicks are.
EDGE_SECONDS = 0.3

MANIFEST_FILE = os.path.join(DATA_FOLDER, "manifest.csv")


def split_recording(audio, mode, room_level, resting_sweep):
    """Find where each gesture starts in a recording. Returns a list of sample numbers."""
    edge = int(EDGE_SECONDS * SAMPLE_RATE)
    onsets = []
    for event_mode, what, onset, strength in find_events(audio, room_level, resting_sweep):
        if event_mode != mode or what != "start":
            continue
        if onset < edge or onset > len(audio) - edge:
            continue # too close to an Enter key click
        if cut_clip(audio, onset, mode) is None:
            continue # not enough audio around it to make a full clip
        onsets.append(onset)
    return onsets


def save_recording(audio, label, mode, session, onsets):
    """Save the WAV file and add one manifest row per gesture found."""
    file_name = f"{session}_{label.replace(' ', '_').lower()}_{mode}.wav"
    wavfile.write(os.path.join(DATA_FOLDER, file_name), SAMPLE_RATE, audio)

    new_file = not os.path.exists(MANIFEST_FILE)
    with open(MANIFEST_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(["file", "label", "mode", "session", "onset"])
        for onset in onsets:
            writer.writerow([file_name, label, mode, session, onset])


def main():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    session = time.strftime("%Y%m%d-%H%M%S") # e.g. 20260929-143000

    stream, blocks = start_audio()
    room_level, resting_sweep = calibrate(blocks)

    for number, (label, mode, midi) in enumerate(GESTURES, start=1):
        while True:
            print(f"\n[{number}/{len(GESTURES)}] {label}  ({mode})")
            answer = input("  Enter = start recording, s = skip: ")
            if answer == "s":
                break

            collect_waiting_audio(blocks) # throw away everything before the start
            input("  RECORDING. Do the gesture ~20 times with a pause between each. Enter = stop ")
            audio = collect_waiting_audio(blocks)

            onsets = split_recording(audio, mode, room_level, resting_sweep)
            answer = input(f"  Found {len(onsets)} in {len(audio) / SAMPLE_RATE:.0f} s. "
                           "Enter = save, r = record again: ")
            if answer == "r":
                continue

            save_recording(audio, label, mode, session, onsets)
            print(f"  Saved {len(onsets)} examples of {label}.")
            break

    stream.stop()
    print(f"\nSession {session} done. Next step: python train.py")


if __name__ == "__main__":
    main()
