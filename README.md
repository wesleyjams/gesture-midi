# Gesture MIDI

Turn your laptop into a MIDI instrument for Ableton. Tap or clap on the laptop, and it sends MIDI notes that you map to controls in Ableton: launch clips, mute tracks, switch effects on and off.

No extra hardware: it uses the laptop's own microphone.

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

### When, then what

Recognizing a gesture takes two separate questions:

1. **WHEN did a sound start?** The **detector** answers this with one simple measurement: the sound suddenly got more than `SOUND_THRESHOLD` dB louder than the quiet room.
2. **WHAT was it?** A machine-learning model, an **SVM** (Support Vector Machine), answers this. It looks at a short clip starting at that moment and compares it with the examples you recorded.

The **same detector** splits your recordings into examples *and* listens live. So the clips the model learns from are cut in exactly the same way as the clips it sees while you play. That's the most important design choice in this project.

```
 microphone ──▶ detector: "a sound started at sample 48,000"
                   │
                   ├──▶ instant HIT note (if turned on)
                   │
                   └──▶ wait 0.1 s for the clip ──▶ features ──▶ SVM: "Clap" ──▶ MIDI note 36
```

### From sound to numbers (features)

An SVM can't read audio directly; it needs a short, fixed list of numbers. `get_features()` makes that list: the clip is split into its *attack* (first half) and *decay* (second half). For each half, it measures how loud 20 frequency bands are, from low thumps to bright clicks. That's 40 numbers.

How loud each band is gets measured with an **FFT** (Fast Fourier Transform), which splits audio into its frequencies.

### MIDI

`play.py` creates a MIDI port called **Laptop Gestures**. Ableton sees it like a plugged-in keyboard. It sends:

| What | MIDI | When |
|---|---|---|
| Gesture | its own note, velocity = how loud | 0.1 s after the sound starts |
| Hit note (off by default) | e.g. note 60, velocity = how loud | the instant any sound starts (~1 ms) |

The hit note exists because 0.1 s is noticeable in rhythm. It is **off by default**: it fires before the sound is classified, so every key press and trackpad click would trigger it too. Turn it on (`HIT_NOTE = 60`) only if you won't touch the keyboard while playing.

### Playing Ableton's keyboard at the same time

Key presses and trackpad clicks are sounds too. So the instrument learns them as two extra gestures, **Key Click** and **Trackpad Click**, which play nothing. When you press a key, the model recognises a key click and ignores it.

---

## 2. The files

| File | What it does |
|---|---|
| `gesture_tools.py` | The toolbox everything else uses: **settings**, **your gestures and their notes**, sound measurement, the detector, features |
| `record.py` | Step 1: record examples of each gesture |
| `train.py` | Step 2: train the model and see how accurate it is |
| `play.py` | Step 3: play, or map gestures in Ableton |
| `data/` | Your recordings (created by `record.py`) |
| `model.joblib` | Your trained model (created by `train.py`) |
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

**Microphone:** run `python -m sounddevice` to list your devices, and set `INPUT_DEVICE` at the top of `gesture_tools.py` to the built-in microphone's number. The numbers change when you plug something in (like headphones), so check again if you get a device error.

**Ableton's own sound** should go to **headphones or an audio interface**, not the laptop speakers. Otherwise the microphone hears your music and mistakes it for gestures.

**Choose your gestures:** edit the `GESTURES` list in `gesture_tools.py`. One line per gesture:
```python
("Clap", 36),              # plays note 36
("Not a Gesture", None),   # plays nothing
```
Keep **"Not a Gesture"**: you'll record accidental bumps and noises for it, so the instrument learns to ignore them. Keep **"Key Click"** and **"Trackpad Click"** too if you'll use the keyboard while playing.

---

## 4. Step 1: Record

```bash
python record.py
```

1. **Calibration:** stay quiet for 2 seconds. The script measures the room and prints something like `room level -67 dB`.
2. For each gesture:
   - press **Enter** to start
   - do the gesture about **20 times, with a short pause between each**
   - press **Enter** to stop
3. It tells you how many it found: `Found 19 in 25 s`. If that matches what you did, press **Enter** to save. Type **r** to record it again, or **s** at the start prompt to skip a gesture.

The first and last 0.3 s of each recording are ignored, because that's where the Enter-key clicks are.

For **"Not a Gesture"**, make the noises you *don't* want to trigger anything: touch the table, shift the laptop, move in your chair.

For **"Key Click"**, type into the terminal the way you play Ableton's keyboard: single keys and chords, soft and hard, with short pauses. What you type is ignored; press **Enter** when done. For **"Trackpad Click"**, click and tap the trackpad. Record both every session.

Record **at least 3 sessions**, ideally on different days.

---

## 5. Step 2: Train

```bash
python train.py
```

It prints:
```
120 examples of 4 gestures from 3 session(s)
Tested on session 20260929-143000: 87% correct (guessing would get 25%)
```
To keep the test honest, the model is trained **without your newest session** and tested on it, like playing on a new day. Then it's retrained on everything and saved.

**The confusion matrix** shows which gestures get mixed up:
```
                 Clap Finger Palm  Not
  Clap             19    0     1     0     ← 19 of 20 claps correct
  Finger Tap        0   14     0     6     ← finger taps often taken for "Not a Gesture"
```
Rows are what you really did, columns are what the model guessed. Numbers off the diagonal tell you which gestures need more careful recording, or are too similar to tell apart.

---

## 6. Step 3: Play in Ableton

**Connect Ableton (once):**
1. Run `python play.py`. The port is created while the script runs.
2. In Ableton: **Settings → Link, Tempo & MIDI**. Find **Laptop Gestures** under *Input*:
   - **Remote on**: the notes control whatever you map them to.
   - **Track off**: the notes never reach an instrument, so gestures make no sound, even while a track is armed for Ableton's computer keyboard.

**Map gestures to controls:** choose mode **2** (mapping):
1. In Ableton press **Cmd+M** (MIDI Map Mode).
2. Click the control you want: a clip, a mute button, a device on/off.
3. In the terminal, type the gesture's number to send its note. It's now mapped.
4. Repeat, then press **Cmd+M** again to leave.

A mapped note works like pressing the button: each hit toggles it on or off, or launches the clip.

Want a gesture to play an instrument instead (e.g. a Drum Rack)? Turn **Track** on and arm that track. Its pads respond to the gestures' note numbers.

**Play:** run `python play.py` and choose **1**. After calibrating, it prints every sound it recognizes:
```
Clap              96% sure    23.4 dB
```
`% sure` is how confident the model is. `dB` is how far the sound rose above the quiet room. Both help with tuning: sounds you didn't mean to make that are only just above `SOUND_THRESHOLD` tell you to raise it.

---

## 7. Recording a good dataset

The model learns whatever your examples have in common, including things you didn't mean to teach it.

- **Record in the conditions you'll play in:** same laptop position, same room, Ableton running.
- **Vary what shouldn't matter:** soft and hard hits, slightly different spots.
- **Pause between repetitions**, at least `SOUND_COOLDOWN` (1 s). The detector needs a clear gap to separate them.
- **Choose gestures that sound different.** Two taps in the same place with the same finger will get confused. A clap, a palm tap and a finger tap are easy to tell apart.

**Removing a gesture:** delete its line from `GESTURES` and run `python train.py`. Its recordings stay in `data/` but are skipped.
- **Keep numbers balanced:** about the same number of examples for each gesture.
- **Check the count** after each recording. If it found far more or fewer than you did, press **r**.

---

## 8. Tuning

All settings are at the top of `gesture_tools.py`.

| Setting | Default | Change it when… |
|---|---|---|
| `SOUND_THRESHOLD` | 20 dB | Soft taps are missed → lower. Random noises trigger → raise |
| `SOUND_COOLDOWN` | 1.0 s | You want to play faster → lower. One hit counts twice → raise |
| `CLIP_AFTER` | 0.1 s | Shorter = faster response, possibly less accurate. **Retrain** after changing |
| `VELOCITY_RANGE` | 30 dB | Hits are always loud → raise. Always soft → lower |
| `HIT_NOTE` | None | Set to a note (e.g. 60) for an instant note on every sound. Key clicks will trigger it too |

After changing anything under **Clips** or **FEATURES**, run `python train.py` again. The detector settings take effect straight away. If you change them a lot, re-record, so your examples are split the same way the detector now works.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | `source env/bin/activate` |
| `Invalid number of channels` or another device error | The device numbers changed (e.g. you plugged in headphones). Run `python -m sounddevice` and update `INPUT_DEVICE` |
| Laptop Gestures not in Ableton | `play.py` must be running; restart Ableton's MIDI settings page |
| Lots of triggers while you're quiet | Raise `SOUND_THRESHOLD`; record the room's noises as "Not a Gesture" |
| Music triggers gestures | Send Ableton's audio to headphones or an interface, not the laptop speakers |
| Good test score, poor live playing | Record a session in the exact setup you play in |
