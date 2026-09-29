# Gesture MIDI

Turn your laptop into a MIDI instrument for Ableton. Tap, clap or wave your hand over the laptop, and it sends MIDI notes and controller messages that you can map to anything in Ableton.

No extra hardware: it uses the laptop's own microphone and speakers.

**Contents**
1. [How it works](#1-how-it-works)
2. [The files](#2-the-files)
3. [Setup](#3-setup)
4. [Step 1: Record](#4-step-1-record)
5. [Step 2: Train](#5-step-2-train)
6. [Step 3: Play in Ableton](#6-step-3-play-in-ableton)
7. [Recording a good dataset](#7-recording-a-good-dataset)
8. [Tuning](#8-tuning)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. How it works

### Two kinds of gestures

- **Acoustic gestures make a sound**: a tap, a clap, a knuckle knock on the laptop. The microphone hears them directly.
- **Ultrasound gestures are silent hand movements**: covering the laptop, swiping over it. To sense them, the speakers constantly play a **sweep**, a tone gliding from 18,000 to 22,050 Hz every 0.05 seconds. It's too high for you to hear. Your hand reflects and blocks it, and the microphone picks up the change.

### When, then what

Recognizing a gesture takes two separate questions:

1. **WHEN did something happen?** The **detector** answers this with simple measurements:
   - *Acoustic:* the sound suddenly got more than 10 dB louder than the quiet room.
   - *Ultrasound:* the sweep changed more than 3 dB from how it sounds with no hand nearby.
2. **WHAT was it?** A machine-learning model, an **SVM** (Support Vector Machine), answers this. It looks at a short clip starting at that moment and compares it with the examples you recorded.

The **same detector** splits your recordings into examples *and* listens live. So the clips the model learns from are cut in exactly the same way as the clips it sees while you play. That's the most important design choice in this project.

```
 microphone ──▶ detector: "something started at sample 48,000"
                   │
                   ├──▶ instant HIT note (acoustic only, ~1 ms)
                   │
                   └──▶ wait for the clip ──▶ features ──▶ SVM: "Nail Tap" ──▶ MIDI note 38
                        (0.1 s acoustic,
                         0.4 s ultrasound)
```

### From sound to numbers (features)

An SVM can't read audio directly; it needs a short, fixed list of numbers. `get_features()` makes that list:

- **Acoustic:** the clip is split into its *attack* (first half) and *decay* (second half). For each half, it measures how loud 20 frequency bands are, from low thumps to bright clicks. That's 40 numbers.
- **Ultrasound:** the sweep is split into 8 frequency bands. The model gets **how much each band changed** compared with just before the gesture, in 4 time slices. That's 32 numbers. Using the *change* means the speaker volume and where the laptop sits don't matter. The time slices tell a swipe going one way from a swipe going the other.

How loud each band is gets measured with an **FFT** (Fast Fourier Transform), which splits audio into its frequencies.

### MIDI

`play.py` creates a MIDI port called **Laptop Gestures**. Ableton sees it like a plugged-in keyboard. It sends:

| What | MIDI | When |
|---|---|---|
| Hit note | note 60, velocity = how loud | the instant any sound starts (~1 ms) |
| Acoustic gesture | its own note | 0.1 s after the sound starts |
| Ultrasound gesture | its own note, or a "hold" | 0.4 s after the hand arrives |
| Hold (e.g. Covered) | CC → 127, back to 0 when your hand leaves | while the gesture lasts |
| Hand controller | CC 20, value 0–127 | continuously: how much your hand changes the sweep |

The hit note exists because 0.1 s is noticeable in rhythm. Use the hit note for drums that must be tight, and the gesture notes for switching, launching and effects.

---

## 2. The files

| File | What it does |
|---|---|
| `gesture_tools.py` | The toolbox everything else uses: **settings**, **your gestures and their MIDI**, sound measurement, the detector, features |
| `record.py` | Step 1: record examples of each gesture |
| `train.py` | Step 2: train the models and see how accurate they are |
| `play.py` | Step 3: play, or map gestures in Ableton |
| `data/` | Your recordings (created by `record.py`) |
| `models.joblib` | Your trained models (created by `train.py`) |
| `archive/v1/` | The earlier version of this project, kept for reference |
| `practice/` | Small practice scripts |

---

## 3. Setup

Once:
```bash
cd gesture-midi
/opt/homebrew/bin/python3.11 -m venv env    # Python 3.11: python-rtmidi has no ready-made package for 3.14 yet
source env/bin/activate
pip install -r requirements.txt
```

Every new terminal: `source env/bin/activate`.

**Audio devices:** run `python -m sounddevice` to list them. Set `INPUT_DEVICE` (the microphone) and `OUTPUT_DEVICE` (the speakers) at the top of `gesture_tools.py`. Use the **built-in** mic and speakers; most headphones and Bluetooth devices can't play ultrasound.

**Ableton's own sound** should go to **headphones or an audio interface**, not the laptop speakers. Otherwise the microphone hears your music and mistakes it for gestures.

**Choose your gestures:** edit the `GESTURES` list in `gesture_tools.py`. One line per gesture:
```python
("Nail Tap",  "acoustic",   ("note", 38)),   # plays note 38
("Covered",   "ultrasound", ("hold", 21)),   # CC 21 = 127 while covered, 0 when you let go
("Not a Gesture", "acoustic", None),         # sends nothing
```
Keep one **"Not a Gesture"** per mode. You'll record accidental bumps and movements for it, so the instrument learns to ignore them.

---

## 4. Step 1: Record

```bash
python record.py
```

1. **Calibration:** keep your hands away and stay quiet for 2 seconds. The script measures the room and the resting sweep, and prints something like `room level -55 dB, sweep wobble 0.4 dB`.
2. For each gesture:
   - press **Enter** to start
   - do the gesture about **20 times, with a short pause between each**
   - press **Enter** to stop
3. It tells you how many it found: `Found 19 in 25 s`. If that matches what you did, press **Enter** to save. Type **r** to record it again, or **s** at the start prompt to skip a gesture.

The first and last 0.3 s of each recording are ignored, because that's where the Enter-key clicks are.

For **"Not a Gesture"**, do things you *don't* want to trigger anything: touch the table, shift the laptop, reach for the trackpad, move your hand around nearby.

Record **at least 3 sessions**, ideally on different days.

---

## 5. Step 2: Train

```bash
python train.py
```

For each mode it prints:
```
120 examples of 4 gestures from 3 session(s)
Tested on session 20260929-143000: 87% correct (guessing would get 25%)
```
To keep the test honest, the model is trained **without your newest session** and tested on it, like playing on a new day. Then it's retrained on everything and saved.

**The confusion matrix** shows which gestures get mixed up:
```
                 Clap Knuckle Nail  Not
  Clap             19    0     1     0     ← 19 of 20 claps correct
  Knuckle Tap       0   14     0     6     ← knuckle taps often taken for "Not a Gesture"
```
Rows are what you really did, columns are what the model guessed. Numbers off the diagonal tell you which gestures need more careful recording, or are too similar to tell apart.

---

## 6. Step 3: Play in Ableton

**Connect Ableton (once):**
1. Run `python play.py`. The port is created while the script runs.
2. In Ableton: **Settings → Link, Tempo & MIDI**. Find **Laptop Gestures** under *Input* and turn on **Track** (notes play instruments) and **Remote** (notes and CCs can control knobs and buttons).

**Map gestures to controls:** choose mode **2** (mapping):
1. In Ableton press **Cmd+M** (MIDI Map Mode).
2. Click the control you want: a knob, a clip, a mute button.
3. In the terminal, type the gesture's number to send its message. It's now mapped.
4. Repeat, then press **Cmd+M** again to leave.

For notes to play an instrument (e.g. a Drum Rack), arm the track instead. Its pads respond to the gestures' note numbers.

**Play:** run `python play.py` and choose **1**. After calibrating, it prints every gesture it recognizes and sends the MIDI.

---

## 7. Recording a good dataset

The model learns whatever your examples have in common, including things you didn't mean to teach it.

- **Record in the conditions you'll play in:** same laptop position, same speaker volume, Ableton running.
- **Keep the speaker volume the same** every session. The sweep's level is part of what the ultrasound model hears.
- **Vary what shouldn't matter:** soft and hard hits, fast and slow swipes, slightly different spots.
- **Pause between repetitions** (about 1 s for sounds, 1.5–2 s for hand gestures). The detector needs a clear gap to separate them.
- **Choose gestures that sound different.** Two taps in the same place with the same finger will get confused. A clap, a nail tap and a knuckle knock are easy to tell apart.
- **Keep numbers balanced:** about the same number of examples for each gesture.
- **Check the count** after each recording. If it found far more or fewer than you did, press **r**.

---

## 8. Tuning

All settings are at the top of `gesture_tools.py`.

| Setting | Default | Change it when… |
|---|---|---|
| `ACOUSTIC_THRESHOLD` | 10 dB | Soft taps are missed → lower. Random noises trigger → raise |
| `ULTRASOUND_THRESHOLD` | 3 dB | Hand gestures are missed → lower (stay well above the "sweep wobble" from calibration). False hand triggers → raise |
| `ACOUSTIC_AFTER` | 0.1 s | Shorter = faster response, possibly less accurate. **Retrain** after changing |
| `CONTROLLER_RANGE` | 10 dB | The hand controller never reaches 127 → lower. It maxes out too easily → raise |
| `VELOCITY_RANGE` | 30 dB | Hits are always loud → raise. Always soft → lower |
| `HIT_NOTE`, `CONTROLLER_CC` | 60, 20 | Set to `None` to turn them off |

After changing anything under **Clips** or **FEATURES**, run `python train.py` again. The detector settings take effect straight away. If you change them a lot, re-record, so your examples are split the same way the detector now works.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | `source env/bin/activate` |
| Stream / device error | Check `INPUT_DEVICE` and `OUTPUT_DEVICE` with `python -m sounddevice`; use the built-in mic and speakers |
| "WARNING: the sweep is unsteady" | Keep your hands away during calibration; check the laptop volume isn't muted |
| `Found 0` for a hand gesture | Lower `ULTRASOUND_THRESHOLD`; move your hand closer to the speakers |
| Laptop Gestures not in Ableton | `play.py` must be running; restart Ableton's MIDI settings page |
| Music triggers gestures | Send Ableton's audio to headphones or an interface, not the laptop speakers |
| Good test score, poor live playing | Record a session in the exact setup you play in |
