'''
Step 3: play! Turns your gestures into MIDI notes for Ableton.

Opens a MIDI port called "Laptop Gestures" and offers two modes:

  1 = play      Listens to the microphone and sends MIDI:
                  - every sound sends HIT_NOTE instantly, if it is turned on
                  - 0.1 s later the sound is classified and plays its own note
                    (key and trackpad clicks are classified too, and play nothing)
  2 = mapping   Type a number to send one gesture's note. Use this in
                Ableton's MIDI Map Mode to connect gestures to controls
                without having to perform them.

  python play.py      (Ctrl+C to stop)
'''

import threading

import joblib
import mido
import numpy as np

from gesture_tools import (
    GESTURES, HIT_NOTE, MIDI_PORT_NAME, MODEL_FILE, NOTE_LENGTH, VELOCITY_RANGE,
    Detector, calibrate, collect_audio, cut_clip, find_note, get_features, start_audio,
)

MEMORY_SECONDS = 1.0 # how much recent audio to keep (must be longer than a clip)


# -----------------------------
# Sending MIDI
# -----------------------------
def send_note(port, note, velocity):
    """Play a note: note on now, note off NOTE_LENGTH seconds later."""
    port.send(mido.Message("note_on", note=note, velocity=velocity))
    note_off = mido.Message("note_off", note=note)
    threading.Timer(NOTE_LENGTH, port.send, [note_off]).start() # sends it later, in the background


def to_velocity(loudness):
    """How loud a sound was (dB above the room) as a note velocity from 1 to 127."""
    return int(np.clip(loudness / VELOCITY_RANGE * 127, 1, 127))


# -----------------------------
# Mode 2: mapping
# -----------------------------
def mapping_mode(port):
    # Every note the instrument can play: (name, note)
    notes = []
    for label, note in GESTURES:
        if note is not None:
            notes.append((label, note))
    if HIT_NOTE is not None:
        notes.append(("Hit note (every sound)", HIT_NOTE))

    print("\nIn Ableton: press Cmd+M, click the control you want, then send its note here.")
    for number, (name, note) in enumerate(notes, start=1):
        print(f"  {number}. {name:<24} note {note}")

    while True:
        answer = input("Number to send (q to quit): ")
        if answer == "q":
            break
        if not answer.isdigit() or not 1 <= int(answer) <= len(notes):
            continue
        name, note = notes[int(answer) - 1]
        send_note(port, note, 100)
        print(f"  sent {name}")


# -----------------------------
# Mode 1: play
# -----------------------------
def play(port):
    model = joblib.load(MODEL_FILE)
    stream, blocks = start_audio()
    detector = Detector(calibrate(blocks))

    recent_audio = collect_audio(blocks, MEMORY_SECONDS) # start with real audio, not silence
    waiting = [] # sounds that started but whose clip isn't complete yet: (onset, loudness)

    print("Playing! Ctrl+C to stop.\n")
    while True:
        # 1. Add the newest microphone block to our memory, dropping the oldest audio
        block = blocks.get()
        recent_audio = np.concatenate([recent_audio[len(block):], block])

        # 2. Ask the detector if a sound just started
        sound = detector.update(block)
        if sound is not None:
            onset, loudness = sound
            if HIT_NOTE is not None:
                send_note(port, HIT_NOTE, to_velocity(loudness)) # instant, before we know which sound it was
            waiting.append(sound)

        # 3. Classify every sound whose clip is now complete
        audio_start = detector.samples_heard - len(recent_audio) # sample number of recent_audio[0]
        still_waiting = []
        for onset, loudness in waiting:
            clip = cut_clip(recent_audio, onset, audio_start)
            if clip is None:
                still_waiting.append((onset, loudness)) # not finished yet
                continue

            # The model gives a chance for every gesture; the highest one is its guess
            chances = model.predict_proba([get_features(clip)])[0]
            best = np.argmax(chances)
            label = model.classes_[best]
            print(f"{label:<16} {chances[best]:4.0%} sure   {loudness:5.1f} dB")

            note = find_note(label)
            if note is not None:
                send_note(port, note, to_velocity(loudness))
        waiting = still_waiting


def main():
    port = mido.open_output(MIDI_PORT_NAME, virtual=True)
    print(f'MIDI port "{MIDI_PORT_NAME}" is open.')
    print("In Ableton: Settings > Link, Tempo & MIDI > turn on Remote for it, and Track off")
    print("(Track off = the notes control what you mapped, but never play an instrument).")

    choice = input("\n1 = play, 2 = mapping mode: ")
    try:
        if choice == "2":
            mapping_mode(port)
        else:
            play(port)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        port.close()


if __name__ == "__main__":
    main()
