'''
Recognizes gestures live, using the models from train_gesture_model.py.

The speakers play the ultrasound sweep the whole time, and the microphone
keeps the last 2 seconds of audio in memory. Two models watch that audio:

  acoustic   When a sound gets louder than LOUDNESS_THRESHOLD, wait a moment
             for the sound to finish, then classify it once.
  ultrasound Every ULTRASOUND_EVERY seconds, classify the latest audio.
             Only print when the answer changes.

"Background" answers mean nothing happened, so they are never printed.

Run:
  python run_gesture_model.py      (Ctrl+C to stop)
'''

import queue

import joblib
import numpy as np
import sounddevice as sd

from gesture_features import get_features
from record_gesture_dataset import (
    BLOCK_SIZE, INPUT_DEVICE, OUTPUT_DEVICE, SAMPLE_RATE, SWEEP_DURATION,
    apply_mode_filter, make_sweep,
)

MODEL_FILE = "gesture_models.joblib"

# Acoustic settings
# Measured from the recordings: with the sweep filtered out, the background sits
# around -58 dB and 90% of even soft nail taps reach -48 dB. Raise this if random
# noises trigger it; lower it if soft taps are missed.
LOUDNESS_THRESHOLD = -48 # dB
WAIT_AFTER_SOUND = 0.3 # seconds to keep listening after a sound starts
ACOUSTIC_CLIP = 0.6 # seconds of audio to classify (includes a bit before the sound)
COOLDOWN = 0.5 # seconds to ignore new sounds after one is detected

# Ultrasound settings
ULTRASOUND_EVERY = 0.5 # seconds between ultrasound checks
ULTRASOUND_CLIP = SWEEP_DURATION + 0.5 # a full sweep plus a margin on each side
QUIET_AFTER_SOUND = 1.0 # loud sounds leak into the ultrasound band, so pause after one

MEMORY = 2.0 # seconds of recent audio to keep


# -----------------------------
# Audio: the microphone and speakers run in the background
# -----------------------------
sweep = make_sweep()
sweep_position = 0
new_blocks = queue.Queue() # microphone audio waiting to be processed


def audio_callback(indata, outdata, frames, time_info, status):
    """Called by sounddevice for every block: save the mic audio, play the next bit of sweep."""
    global sweep_position

    new_blocks.put(indata[:, 0].copy())

    # Take the next `frames` samples of the sweep, wrapping back to the start
    positions = (sweep_position + np.arange(frames)) % len(sweep)
    outdata[:, 0] = sweep[positions]
    sweep_position = (sweep_position + frames) % len(sweep)


# -----------------------------
# Helpers
# -----------------------------
def loudness_without_sweep(recent_audio):
    """How loud the newest block is (in dB) after filtering out the ultrasound sweep."""
    # Filter a little more than one block so the filter has audio to settle on
    filtered = apply_mode_filter(recent_audio[-4 * BLOCK_SIZE:], "acoustic")
    newest_block = filtered[-BLOCK_SIZE:]
    rms = np.sqrt(np.mean(newest_block ** 2)) # root mean square: the average strength
    return 20 * np.log10(rms + 1e-12)


def classify(model, audio, mode):
    features = get_features(audio, mode)
    return model.predict([features])[0]


def is_background(label):
    return label.lower().startswith("background")


# -----------------------------
# Main loop
# -----------------------------
def main():
    models = joblib.load(MODEL_FILE)
    acoustic_model = models["acoustic"]
    ultrasound_model = models["ultrasound"]
    print("Acoustic gestures:  ", ", ".join(acoustic_model.classes_))
    print("Ultrasound gestures:", ", ".join(ultrasound_model.classes_))
    print("Listening... (Ctrl+C to stop)\n")

    recent_audio = np.zeros(int(MEMORY * SAMPLE_RATE), dtype=np.float32)

    # Counters, all measured in samples
    waiting_for_sound_to_finish = False
    time_since_sound_started = 0
    time_since_last_sound = int(MEMORY * SAMPLE_RATE)
    time_since_ultrasound_check = 0
    last_ultrasound_label = None

    stream = sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        channels=1,
        device=(INPUT_DEVICE, OUTPUT_DEVICE),
        callback=audio_callback,
    )

    with stream:
        while True:
            # Wait for the next block from the microphone and add it to memory,
            # dropping the same amount of the oldest audio
            block = new_blocks.get()
            recent_audio = np.concatenate([recent_audio[len(block):], block])

            time_since_last_sound += len(block)
            time_since_ultrasound_check += len(block)

            # ---- Acoustic: wait for a loud sound, then classify it once ----
            if waiting_for_sound_to_finish:
                time_since_sound_started += len(block)
                if time_since_sound_started >= WAIT_AFTER_SOUND * SAMPLE_RATE:
                    clip = recent_audio[-int(ACOUSTIC_CLIP * SAMPLE_RATE):]
                    label = classify(acoustic_model, clip, "acoustic")
                    if not is_background(label):
                        print(f"acoustic:   {label}")
                    waiting_for_sound_to_finish = False

            elif time_since_last_sound > COOLDOWN * SAMPLE_RATE:
                if loudness_without_sweep(recent_audio) > LOUDNESS_THRESHOLD:
                    waiting_for_sound_to_finish = True
                    time_since_sound_started = 0
                    time_since_last_sound = 0

            # ---- Ultrasound: check regularly, print only when the answer changes ----
            if time_since_ultrasound_check >= ULTRASOUND_EVERY * SAMPLE_RATE:
                time_since_ultrasound_check = 0

                if time_since_last_sound < QUIET_AFTER_SOUND * SAMPLE_RATE:
                    continue # a loud sound just happened; skip this check

                clip = recent_audio[-int(ULTRASOUND_CLIP * SAMPLE_RATE):]
                label = classify(ultrasound_model, clip, "ultrasound")
                if label != last_ultrasound_label:
                    if not is_background(label):
                        print(f"ultrasound: {label}")
                    last_ultrasound_label = label


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
