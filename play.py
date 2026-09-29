'''
Step 3: play! Turns your gestures into MIDI for Ableton.

Opens a MIDI port called "Laptop Gestures" and offers two modes:

  1 = play      Listens to the microphone and sends MIDI:
                  - every sound sends HIT_NOTE instantly (velocity = how loud)
                  - 0.1 s later the sound is classified and sends its own MIDI
                  - silent hand gestures are classified 0.4 s after they start
                  - the hand controller CC follows your hand all the time
  2 = mapping   Type a number to send one gesture's MIDI message. Use this in
                Ableton's MIDI Map Mode to connect gestures to controls
                without having to perform them.

  python play.py      (Ctrl+C to stop)
'''

import threading

import joblib
import mido
import numpy as np

from gesture_tools import (
    CONTROLLER_CC, CONTROLLER_RANGE, GESTURES, HIT_NOTE, MIDI_PORT_NAME, MODEL_FILE,
    NOTE_LENGTH, SAMPLE_RATE, VELOCITY_RANGE,
    Detector, calibrate, collect_audio, cut_clip, find_midi, get_features, start_audio,
)

MEMORY_SECONDS = 1.0 # how much recent audio to keep (must be longer than any clip)


# -----------------------------
# Sending MIDI
# -----------------------------
def send_note(port, note, velocity):
    """Play a note: note on now, note off NOTE_LENGTH seconds later."""
    port.send(mido.Message("note_on", note=note, velocity=velocity))
    note_off = mido.Message("note_off", note=note)
    threading.Timer(NOTE_LENGTH, port.send, [note_off]).start() # sends it later, in the background


def send_cc(port, number, value):
    """Set a controller (CC) to a value from 0 to 127."""
    port.send(mido.Message("control_change", control=number, value=value))


def to_midi_value(amount, maximum):
    """Scale an amount (0 to maximum) to a MIDI value (0 to 127)."""
    return int(np.clip(amount / maximum * 127, 0, 127))


# -----------------------------
# Mode 2: mapping
# -----------------------------
def mapping_mode(port):
    # Everything the instrument can send: (name, message)
    messages = []
    for label, mode, midi in GESTURES:
        if midi is not None:
            messages.append((f"{label} ({mode})", midi))
    if HIT_NOTE is not None:
        messages.append(("Hit note (every sound)", ("note", HIT_NOTE)))
    if CONTROLLER_CC is not None:
        messages.append(("Hand controller", ("cc", CONTROLLER_CC)))

    print("\nIn Ableton: press Cmd+M, click the control you want, then send its message here.")
    for number, (name, midi) in enumerate(messages, start=1):
        print(f"  {number}. {name:<28} {midi[0]} {midi[1]}")

    while True:
        answer = input("Number to send (q to quit): ")
        if answer == "q":
            break
        if not answer.isdigit() or not 1 <= int(answer) <= len(messages):
            continue
        name, (kind, number) = messages[int(answer) - 1]
        if kind == "note":
            send_note(port, number, 100)
        else:
            send_cc(port, number, 127)
        print(f"  sent {name}")


# -----------------------------
# Mode 1: play
# -----------------------------
def play(port):
    models = joblib.load(MODEL_FILE)
    stream, blocks = start_audio()
    room_level, resting_sweep = calibrate(blocks)
    detector = Detector(room_level, resting_sweep)

    recent_audio = collect_audio(blocks, MEMORY_SECONDS) # start with real audio, not silence
    waiting = [] # gestures that started but whose clip isn't complete yet: (mode, onset, velocity)
    held_cc = None # the CC number being held at 127, if any
    last_controller_value = None

    print("Playing! Ctrl+C to stop.\n")
    while True:
        # 1. Add the newest microphone block to our memory, dropping the oldest audio
        block = blocks.get()
        recent_audio = np.concatenate([recent_audio[len(block):], block])

        # 2. Ask the detector if anything started or ended
        for mode, what, onset, strength in detector.update(recent_audio):
            if mode == "acoustic":
                velocity = max(1, to_midi_value(strength, VELOCITY_RANGE))
                if HIT_NOTE is not None:
                    send_note(port, HIT_NOTE, velocity) # instant, before we know which sound it was
                waiting.append((mode, onset, velocity))
            elif what == "start":
                waiting.append((mode, onset, 100))
            elif what == "end" and held_cc is not None:
                send_cc(port, held_cc, 0) # the hand left: release the hold
                held_cc = None

        # 3. Classify every gesture whose clip is now complete
        audio_start = detector.samples_heard - len(recent_audio) # sample number of recent_audio[0]
        still_waiting = []
        for mode, onset, velocity in waiting:
            clip = cut_clip(recent_audio, onset, mode, audio_start)
            if clip is None:
                still_waiting.append((mode, onset, velocity)) # not finished yet
                continue
            if mode not in models:
                continue # no model trained for this mode

            label = models[mode].predict([get_features(clip, mode)])[0]
            midi = find_midi(label, mode)
            print(f"{mode:<11} {label}")
            if midi is None:
                continue

            kind, number = midi
            if kind == "note":
                send_note(port, number, velocity)
            elif kind == "hold":
                send_cc(port, number, 127)
                if detector.hand_active:
                    held_cc = number # released when the detector says the hand left
                else:
                    send_cc(port, number, 0) # the hand already left: just a short pulse
        waiting = still_waiting

        # 4. Hand controller: follows how much your hand changes the sweep
        if CONTROLLER_CC is not None:
            value = to_midi_value(detector.sweep_change, CONTROLLER_RANGE)
            if value != last_controller_value: # only send when it changes
                send_cc(port, CONTROLLER_CC, value)
                last_controller_value = value


def main():
    port = mido.open_output(MIDI_PORT_NAME, virtual=True)
    print(f'MIDI port "{MIDI_PORT_NAME}" is open.')
    print("In Ableton: Settings > Link, Tempo & MIDI > turn on Track and Remote for it.")

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
