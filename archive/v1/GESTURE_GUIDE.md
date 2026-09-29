# Gesture Recognition with Sound: A Beginner's Guide

This guide explains how the gesture recognition system in this folder works, from the physics of sound to a trained machine-learning model reacting to your hands in real time. No prior knowledge is assumed.

**Contents**
1. [The big picture](#1-the-big-picture)
2. [Core ideas you need](#2-core-ideas-you-need)
3. [The files](#3-the-files)
4. [Setup](#4-setup)
5. [Step 1: Record a dataset](#5-step-1-record-a-dataset)
6. [Step 2: Train the models](#6-step-2-train-the-models)
7. [Step 3: Run the models live](#7-step-3-run-the-models-live)
8. [Building a good dataset](#8-building-a-good-dataset)
9. [Adding new gestures](#9-adding-new-gestures)
10. [Troubleshooting](#10-troubleshooting)
11. [Where things stand and what to try next](#11-where-things-stand-and-what-to-try-next)
12. [Glossary](#12-glossary)

---

## 1. The big picture

The laptop's microphone is used as a sensor. There are two kinds of gestures:

- **Acoustic gestures** make their own sound: a clap, a snap, a knuckle tap on the laptop. The microphone hears them directly.
- **Ultrasound gestures** make no sound: a swipe, a flick, or covering part of the laptop with your hand. To sense these, the speakers play a tone that's too high-pitched for humans to hear (18,000–22,050 Hz). Your hand reflects and blocks that tone, and the microphone picks up the changes.

Getting from "sound in the room" to "the computer says *Clap*" takes three stages:

```
 ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
 │ 1. RECORD        │     │ 2. TRAIN         │     │ 3. RUN           │
 │                  │     │                  │     │                  │
 │ You perform each │ ──▶ │ Turn every clip  │ ──▶ │ Listen live,     │
 │ gesture many     │     │ into numbers and │     │ turn audio into  │
 │ times; each take │     │ teach an SVM to  │     │ the same numbers,│
 │ is saved as a    │     │ tell gestures    │     │ ask the SVM what │
 │ labeled WAV file │     │ apart            │     │ gesture it was   │
 └──────────────────┘     └──────────────────┘     └──────────────────┘
 record_gesture_           train_gesture_           run_gesture_
 dataset.py                model.py                 model.py
```

This is **supervised machine learning**. You give the computer many *labeled examples* ("this clip is a Clap", "this clip is a Snap"), and it learns a rule that separates them. The model can only be as good as the examples you give it, which is why [Section 8](#8-building-a-good-dataset) matters so much.

---

## 2. Core ideas you need

### 2.1 Sound is vibration, and a microphone measures it as numbers

A microphone measures air pressure many times per second. Each measurement is one **sample**. These scripts record **96,000 samples per second** (the *sample rate*, 96 kHz), so one second of audio is a list of 96,000 numbers.

### 2.2 Frequency and hearing

Sound is made of vibrations at different speeds, measured in **Hertz (Hz)**, meaning vibrations per second. Low frequencies sound deep (a bass drum is about 60 Hz), and high frequencies sound sharp (a whistle is about 3,000 Hz). Adult humans hear roughly **20 Hz to 16,000–20,000 Hz**. Anything above that is **ultrasound**.

A rule called the **Nyquist limit** says you can only capture frequencies up to *half* the sample rate. Recording at 96 kHz captures up to 48 kHz, which is why these scripts use 96 kHz instead of the more common 44.1 kHz (that would only reach 22 kHz, right at the edge of our sweep).

### 2.3 The FFT and the spectrogram: seeing frequencies

A raw list of samples doesn't tell you much. The **FFT (Fast Fourier Transform)** takes a short chunk of audio and reports how much energy is at each frequency, like a graphic equalizer.

Do an FFT on every short chunk, one after another, and stack the results side by side. That gives a **spectrogram**: time runs left to right, frequency runs bottom to top, and color shows loudness. The recording window shows you one live.

```
 Frequency
 22 kHz ┤          ╱      ╱      ╱   ← the ultrasound sweep: a rising line
 18 kHz ┤ ───────╱──────╱──────╱─── ← (cyan dashed line marks 18 kHz)
        ┤
        ┤      █                       ← a clap: a short vertical burst
  5 kHz ┤      █        ▓              ← a tap: shorter, quieter burst
        ┤      █        ▓
    0   ┼──────────────────────────▶ Time
```

In practice, the log sweep curves rather than drawing a straight line, but the idea is the same.

A few terms come up when you compute an FFT:
- **FFT size** (here 4096 samples, about 43 ms) is how long each chunk is. A longer chunk gives finer frequency detail but blurrier timing.
- **Frequency bin** is one "slot" in the FFT output. At 96 kHz with 4096 samples, each bin is about 23 Hz wide.
- **Window (Hann)** is a smooth fade applied to each chunk before the FFT. It prevents fake frequencies caused by chopping audio abruptly.
- **dB (decibels)** is a logarithmic loudness scale. +10 dB means 10× the power. Every loudness value in these scripts is in dB, because that's much closer to how differences are actually noticed.

### 2.4 How the ultrasound sweep senses your hand

The speakers repeatedly play a **sweep** (also called a *chirp*): a tone that glides from 18,000 Hz up to 22,050 Hz over 1 second, then starts again. It's quiet (amplitude 0.05) and too high to hear.

The sound travels from speaker to microphone both directly and by bouncing off nearby surfaces. When your hand moves nearby:
- **Covering** blocks or reflects the sound, so some frequencies get louder or quieter at the mic.
- **Moving** shifts the reflected frequency slightly, the **Doppler effect**. It's the same reason an ambulance siren drops in pitch as it passes you.

Why a sweep and not a single tone? Your hand affects different frequencies differently. Sweeping through a range gives the microphone a whole "fingerprint" of how the environment responds, not just one number.

### 2.5 Filters: keeping the two kinds of gesture apart

A **filter** removes some frequencies and keeps others:
- A **low-pass** filter keeps frequencies *below* a cutoff.
- A **band-pass** filter keeps only frequencies *between* two cutoffs.

The function `apply_mode_filter()` in [record_gesture_dataset.py](record_gesture_dataset.py) does this separation:

| Mode | Filter | Keeps | Why |
|---|---|---|---|
| acoustic | low-pass at 16 kHz | 0–16 kHz | Claps and taps live here; the sweep (18 kHz+) is removed |
| ultrasound | band-pass 17.5–22.55 kHz | the sweep band | Only the part your hand changes |

The gap between 16 kHz and 18 kHz is a **guard band**. Real filters don't cut off like a wall, they fade out gradually. The gap stops the sweep from leaking into the acoustic data.

The filter is applied with `sosfiltfilt`, which runs the filter forward and then backward. This keeps sharp events like a clap from shifting in time.

### 2.6 Features: turning a clip into a fixed list of numbers

A machine-learning model like an SVM needs every example to be the **same length list of numbers**, called a **feature vector**. Takes have different lengths, and a raw clip is tens of thousands of samples, so we summarize each clip into a small set of meaningful numbers. This is **feature extraction**, done in [gesture_features.py](gesture_features.py).

**Acoustic features (99 numbers per clip):**
1. Filter the clip (low-pass 16 kHz).
2. Find the loudest moment (the hit of the clap or tap).
3. Cut a 0.3-second window starting 30 ms before that peak.
4. Build a spectrogram and group its frequencies into 24 **bands** from 200 Hz to 16 kHz. The bands are spaced like piano octaves, so there are more at low frequencies.
5. Split the window into 3 time segments (*attack*, *body*, *decay*) and record the average loudness of every band in each segment: 72 numbers.
6. Add the maximum loudness of each band (24 numbers), plus 3 extras: peak loudness, how long the sound lasts (*sustain*), and how bright or sharp it is (*spectral centroid*).

That's how a clap (loud, broad, fast decay) ends up looking different from a knuckle tap (duller, lower) as numbers.

**Ultrasound features (60 numbers per clip):**
1. Filter the clip (band-pass around the sweep).
2. Take the **middle 1.0 second** of the take. That's exactly one full sweep cycle, so every frequency in the sweep gets visited once, no matter where in the cycle the window starts.
3. Group the sweep range into 20 bands and record, for each band, the **mean**, **max** and **standard deviation** (how much it wobbles) of the loudness.

**Important:** training and live recognition both call the same `extract_features()` function. If they processed audio even slightly differently, the model would see numbers at runtime that look nothing like what it learned from, and accuracy would drop without any obvious error.

### 2.7 The SVM: the actual classifier

An **SVM (Support Vector Machine)** is a classic machine-learning model. Picture every take as a dot in a space with one axis per feature (99 axes for acoustic). Claps cluster in one region, snaps in another. The SVM learns **boundaries** that separate the clusters as widely as possible.

We use an **RBF kernel**, which lets those boundaries curve around clusters instead of being straight lines. Two settings control it:
- **C** controls how strictly it fits the training data. High C gives tight, wiggly boundaries that risk memorizing. Low C gives smoother, more forgiving boundaries.
- **gamma** controls how far one example's influence reaches. High gamma gives very local, bumpy boundaries.

We don't guess these values. The training script tries 16 combinations and keeps whichever scores best (a **grid search**).

Before the SVM, a **StandardScaler** rescales every feature to a similar range. Without it, a feature measured in big numbers would drown out one measured in small numbers.

**Two separate models.** Acoustic and ultrasound gestures get *separate* SVMs. They're recorded under different conditions (sweep off vs. on) and look at different frequency bands. Keeping them apart makes each problem simpler.

### 2.8 Testing honestly: train/test splits and cross-validation

A model always looks great on the examples it trained on, like a student grading their own homework. The real question is how it does on **new** examples.

**Cross-validation** checks this. The data is split into, say, 3 parts ("folds"). The model trains on 2 and is tested on the 3rd, and this repeats until every fold has been the test set once. The average score is your estimate.

This project goes one step further. If you have **3 or more recording sessions**, the folds are split **by session**: the model is tested on a session it never saw during training. That's much stricter and more realistic, because a real user is always a "new session". Different days mean different room noise, a different laptop position and different hand habits.

### 2.9 Confidence

After training, the model is wrapped in a **calibrator** (`CalibratedClassifierCV`). That lets it output a **probability** for each gesture, e.g. "Clap 0.82, Snap 0.10, …". The live script ignores predictions below a threshold (`--min-confidence`, default 0.6), so it stays quiet when unsure instead of guessing.

---

## 3. The files

| File | What it does |
|---|---|
| [record_gesture_dataset.py](record_gesture_dataset.py) | Prompts you through each gesture, records takes with the space bar, saves WAVs + `manifest.csv`. Also holds the **gesture list** and `apply_mode_filter()`. |
| [gesture_features.py](gesture_features.py) | Turns raw audio into feature vectors. Shared by training and live running. |
| [train_gesture_model.py](train_gesture_model.py) | Reads the dataset, extracts features, trains + evaluates the two SVMs, saves `gesture_models.joblib`. |
| [run_gesture_model.py](run_gesture_model.py) | Plays the sweep, listens, and prints recognized gestures live. |
| [ultrasound_sweep_capture.py](ultrasound_sweep_capture.py) | The original class example the sweep and spectrogram code came from. Useful for exploring. |

What gets created on disk:

```
gesture_data/
  manifest.csv                      ← one row per take: file, label, mode, session, ...
  wav/
    clap/20260925-093816_001.wav    ← <session timestamp>_<take number>.wav
    clap/20260925-093816_002.wav
    swipe_right/...
gesture_models.joblib               ← the trained models (after training)
```

The `manifest.csv` is the table of contents of your dataset. The training script reads it to find every file and its label. Don't hand-edit it while recording; use the `r` key to redo takes instead.

---

## 4. Setup

You only need to do this once. See also [README.md](README.md).

```bash
cd py-apps/acoustics
python3 -m venv env                # create an isolated Python environment
source env/bin/activate            # activate it (do this in every new terminal)
pip install -r requirements.txt    # install numpy, scipy, sounddevice, scikit-learn, ...
```

Once the environment is active, `python` means the environment's Python. All commands below assume it's active. (Without activating, use `env/bin/python` in place of `python`.)

### Pick your audio devices

```bash
python record_gesture_dataset.py --list-devices
```

Example output:
```
>  2 MacBook Pro Microphone, Core Audio (1 in, 0 out)
<  3 MacBook Pro Speakers, Core Audio (0 in, 2 out)
```

The defaults are input **2** and output **3**. If yours differ, either edit `INPUT_DEVICE` / `OUTPUT_DEVICE` near the top of [record_gesture_dataset.py](record_gesture_dataset.py) (the other scripts reuse them), or pass `--input-device N --output-device M` to each script.

**Use the built-in mic and speakers.** Bluetooth headphones and many USB devices can't handle 96 kHz or ultrasound, and the sweep would never reach the mic.

---

## 5. Step 1: Record a dataset

```bash
python record_gesture_dataset.py --reps 10
```

A spectrogram window opens. **Click on it** so it receives your key presses. The title tells you what to do:

```
[3/18] Finger Nail Tap (acoustic) - take 4/10 - SPACE to record
```

| Key | Action |
|---|---|
| **Space** | Start recording (title turns red). Press again to stop and save. |
| **r** | Redo: deletes the last saved take and lets you record it again. |
| **s** | Skip the rest of the current gesture. |
| **q** | Quit. Everything already saved stays saved. |

What happens behind the scenes:
- For **acoustic** gestures the speakers are **silent**.
- For **ultrasound** gestures the sweep starts **as soon as the gesture comes up**, before you press space, so it's already steady when you record. You'll see the sweep appear above the cyan line.
- Takes shorter than 0.1 s are thrown away (usually an accidental double-press).
- At the end you get a summary: takes per gesture this session / across all sessions.

Useful options:
```bash
python record_gesture_dataset.py --reps 5 --only "Clap,Swipe Right"   # just some gestures
python record_gesture_dataset.py --out other_dataset                   # separate dataset folder
```

**Running the script again adds to the dataset.** Every run gets its own session timestamp, so files never overwrite each other. That's how your dataset grows.

---

## 6. Step 2: Train the models

```bash
python train_gesture_model.py
```

This reads `gesture_data/manifest.csv`, extracts features from every WAV, trains both SVMs and saves them to `gesture_models.joblib`. Options: `--data <folder>` and `--out <file>`.

### Reading the output

```
=== acoustic ===
  270 takes, 99 features, 3 session(s)
  CV: 3-fold split by session (tests on unseen sessions)
  best params: {'svc__C': 10, 'svc__gamma': 'scale'}  CV accuracy: 0.560
```
- **CV accuracy** is the headline number: the fraction of held-out takes classified correctly. Compare it to **chance**, which is 1 ÷ number of gestures (9 gestures → 11%).
- If it says **"within sessions (optimistic)"**, you have fewer than 3 sessions, and the score will look better than real life.

The **classification report** gives three numbers per gesture:
- **Precision**: when the model said "Clap", how often was it really a clap?
- **Recall**: of all the real claps, how many did it catch?
- **F1**: one number balancing the two. Aim for 0.8 or higher.

The **confusion matrix** is the most useful diagnostic:

```
                     Bkg Clap Nail  Pad ...
    Background        22   0    0    4
    Clap               0  27    0    0      ← 27 of 30 claps correct
    Finger Pad Tap    11   0    6    9      ← pad taps often mistaken for Background
```
Rows are what the gesture *really* was; columns are what the model *guessed*. A perfect model has numbers only on the diagonal. Big numbers off the diagonal tell you **which pairs of gestures sound alike**. That's where to focus when recording more data, or it's a sign those gestures might not be separable with this setup.

**Retrain whenever you add data.** Training always uses everything in the manifest.

---

## 7. Step 3: Run the models live

```bash
python run_gesture_model.py
```

Perform gestures near the laptop and they're printed as they're recognized:

```
[   12.4s] acoustic   Clap                 (0.91)
[   18.0s] ultrasound Covered Right        (0.83)
```

### How it decides what happened

The sweep plays the whole time, and both models run side by side:

**Acoustic: event-triggered.** The script keeps track of the room's background loudness (the *noise floor*). When a sound jumps **15 dB above** it, that's an *event*. The script grabs a clip from 0.15 s before to 0.45 s after, runs the acoustic model **once**, and then ignores new triggers for 0.4 s. Because the acoustic filter removes everything above 16 kHz, the sweep playing in the background doesn't affect it.

**Ultrasound: continuous.** Every 0.25 s, the last 1.5 s of audio goes through the ultrasound model. A gesture is only reported after **3 predictions in a row agree**, which filters out one-off flukes, and it isn't reported again until the model returns to "Background Sweep".

**Keeping them apart.** A loud clap also has energy in the ultrasound band. So for 1 second after any acoustic event, ultrasound predictions are paused.

Predictions for any label starting with "Background" are never printed. Those classes exist so the model has a "nothing happened" answer instead of being forced to pick a gesture.

### Tuning

| Option | Default | Change it when… |
|---|---|---|
| `--threshold-db` | 15 | Soft taps are missed → lower it (e.g. 10). Random noises trigger → raise it. |
| `--min-confidence` | 0.6 | Nothing gets reported → lower it. Too many wrong guesses → raise it. |
| `--no-sweep` | off | You only want acoustic gestures (ultrasound model is disabled). |
| `--model` | gesture_models.joblib | You trained to a different file. |

If you change anything in [gesture_features.py](gesture_features.py), increase `FEATURE_VERSION` in that file and retrain. The live script refuses to run a model trained with a different feature version, which protects you from the mismatch problem in [2.6](#26-features-turning-a-clip-into-a-fixed-list-of-numbers).

### Using the model in your own code

```python
import joblib
from gesture_features import extract_features, load_wav

bundle = joblib.load("gesture_models.joblib")
model = bundle["models"]["acoustic"]              # or "ultrasound"

audio = load_wav("some_clip.wav")                 # raw, unfiltered, 96 kHz mono
features = extract_features(audio, "acoustic")    # same processing as training
probabilities = model.predict_proba([features])[0]
best = probabilities.argmax()
print(model.classes_[best], probabilities[best])
```

---

## 8. Building a good dataset

A model learns *whatever is consistent in its examples*, including things you didn't intend. If every clap was recorded in a quiet morning session and every snap in a noisy evening one, the model might learn "quiet = clap". Good data is the single biggest lever on accuracy.

### How much data

- **Minimum to get started:** 10 takes per gesture. Training refuses gestures with fewer than 2.
- **Reasonable:** 30–50 takes per gesture, spread across **at least 3 sessions**.
- **Balanced:** try to have a similar count for every gesture. The SVM uses `class_weight="balanced"` to compensate, but real balance is better.

### Vary what should *not* matter

Record across multiple sessions, changing things the model should ignore:
- **Different times and background noise** (quiet room, fan on, people talking).
- **Slightly different hand positions and speeds**: fast and slow swipes, hard and soft taps.
- **Different laptop positions** on the desk and **screen angles**, if you'll use it that way later.
- **Different people**, if others will use it.

If you'll only ever use it in one place in one way, you can vary less. But the model will then only work well in that place.

### Keep constant what *does* matter

- **Do the gesture the same way each time** in terms of *what it is*. "Swipe Right" should always be a left-to-right pass at roughly the same height.
- **Keep the speaker volume the same** between sessions. The sweep level is part of what the ultrasound model learns. Pick a volume and stick with it.
- Don't change `SAMPLE_RATE`, the sweep frequencies or the sweep duration between sessions. If you do, start a new dataset folder with `--out`.

### Recording each take well

- **Acoustic gestures:** press space, pause briefly, do the gesture **once**, pause briefly, press space. One clap per take, not three.
- **Ultrasound gestures:** make takes **at least 1.5 seconds long**, with the gesture **in the middle**. The features use the *middle 1.0 s* of each take. A 0.8 s take gets padded with silence, and a gesture at the very start gets cut off. (This is currently the biggest weakness in the existing dataset.)
- **Stay still between takes**, especially for ultrasound. Any hand motion near the laptop is signal.
- **Messed up? Press `r` right away.** One mislabeled take teaches the model something false.
- **Your keyboard presses are in the recording.** The space-bar click is an acoustic tap on the laptop. Having it in *every* take (including Background) means the model learns to ignore it, which is fine, as long as you're consistent.

### The Background classes are real gestures

"Background" (sweep off) and "Background Sweep" (sweep on) teach the model what *nothing happening* looks like. Record them in the same way and in the same amounts as other gestures. Include ordinary non-gesture noise in some: typing, shifting in your seat, room noise. This is what stops the live script from reporting false gestures.

### Check your data

- Watch the spectrogram while recording. Can *you* see a difference between two gestures? If you can't, the model probably can't either.
- After training, read the **confusion matrix**. Gestures that are consistently confused need more (or more careful) data. Or they may simply not be distinguishable with one mic and one speaker; see [Section 11](#11-where-things-stand-and-what-to-try-next).

---

## 9. Adding new gestures

1. Open [record_gesture_dataset.py](record_gesture_dataset.py) and add one line to the `GESTURES` list:
   ```python
   ("Double Clap", "acoustic"),
   ("Hover", "ultrasound"),
   ```
   Pick `"acoustic"` if the gesture makes a sound, or `"ultrasound"` if it's a silent hand movement.
2. Record just the new gestures:
   ```bash
   python record_gesture_dataset.py --reps 10 --only "Double Clap,Hover"
   ```
   Repeat over several sessions, like the rest of the dataset.
3. Retrain: `python train_gesture_model.py`.

The training and live scripts pick up new labels automatically from the manifest. You never edit them to add gestures.

**Renaming** a gesture creates a new label, so old takes under the old name won't count. **Removing** a gesture from `GESTURES` only stops it being prompted. Its old takes are still in the manifest and still get trained on.

---

## 10. Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| Keys do nothing | Click the spectrogram window first; it needs keyboard focus. |
| No sweep visible above the cyan line | Wrong output device, volume muted, or Bluetooth/USB output. Use built-in speakers. Check `--list-devices`. |
| `Invalid sample rate` / stream error | That device can't do 96 kHz. Use the built-in mic/speakers. |
| `ModuleNotFoundError: numpy` | Environment not active. Run `source env/bin/activate` or use `env/bin/python`. |
| Training: "need at least 2 takes" | Record more takes of the gesture listed. |
| "within sessions (optimistic)" | Fewer than 3 sessions recorded. Record more sessions for an honest score. |
| Live: nothing ever reported | Lower `--min-confidence` (try 0.4) and `--threshold-db` (try 10). |
| Live: constant false gestures | Raise those two values; record more Background takes. |
| Live: "trained with feature version X" | You changed `gesture_features.py`. Retrain. |
| Accuracy great in training, poor live | Live conditions differ from the recordings (volume, position, noise). Record a session in the conditions you'll actually use. |

---

## 11. Where things stand and what to try next

Results from the first 3 sessions (30 takes per gesture), tested on unseen sessions:

| Model | Accuracy | Chance | Notes |
|---|---|---|---|
| Acoustic | 56% | 11% | Clap, Snap and Nail Tap are distinguished well. Tap Outside Left vs. Right is confused (one mic can't tell direction well). Soft Finger Pad Taps are mistaken for Background. |
| Ultrasound | 25% | 11% | Covered vs. Background works (77% on its own). Swipe direction is weak; Flick Up vs. Down is at chance. |

Ideas, roughly in order of effort:
1. **Longer ultrasound takes (1.5 s+, gesture centered).** Many current takes are shorter than the 1 s analysis window. This is the cheapest fix.
2. **Start with coarser classes.** Train Covered / Swipe / Flick / Background first, then add directions once those are reliable.
3. **Shorter sweep** (e.g. 0.1–0.2 s in `SWEEP_DURATION`). The analysis window can then be shorter and capture fast motions like flicks in more detail. This needs a new dataset, because the sweep is part of what was recorded.
4. **Better acoustic features.** MFCCs are the standard in speech and sound recognition. If you add them, remember to bump `FEATURE_VERSION`.
5. **More sensors for direction.** Left/right is fundamentally hard with one microphone. Two mics, or a stereo sweep with different frequencies per speaker, would give the model direction information.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Sample / sample rate** | One microphone measurement / how many per second (96,000 here). |
| **Hz, kHz** | Vibrations per second; 1 kHz = 1,000 Hz. |
| **Nyquist limit** | Highest frequency you can capture = sample rate ÷ 2. |
| **Ultrasound** | Sound above human hearing (roughly >18–20 kHz). |
| **Sweep / chirp** | A tone that glides across a range of frequencies. |
| **Doppler effect** | Frequency shift from a moving reflector (your hand). |
| **FFT** | Converts a chunk of audio into energy per frequency. |
| **Spectrogram** | Many FFTs side by side: frequency vs. time. |
| **dB** | Logarithmic loudness unit. |
| **Filter (low-pass, band-pass)** | Removes frequencies outside a range. |
| **Guard band** | Unused gap between filter ranges to prevent leakage. |
| **Take** | One recording of one gesture. |
| **Session** | One run of the recording script; identified by a timestamp. |
| **Manifest** | The CSV index of every take and its label. |
| **Feature / feature vector** | Numbers summarizing a clip; the model's input. |
| **Label / class** | The gesture name the model predicts. |
| **SVM** | Support Vector Machine: finds boundaries separating classes. |
| **RBF kernel, C, gamma** | Lets SVM boundaries curve; C = strictness, gamma = reach of each example. |
| **StandardScaler** | Rescales features to comparable ranges. |
| **Grid search** | Trying every combination of settings and keeping the best. |
| **Cross-validation (CV)** | Repeatedly training on part of the data and testing on the rest. |
| **Overfitting** | Memorizing training examples instead of learning the general pattern. |
| **Precision / recall / F1** | How trustworthy a prediction is / how many real cases are caught / a balance of both. |
| **Confusion matrix** | Table of true vs. predicted labels; shows which gestures get mixed up. |
| **Calibration / confidence** | Turning the model's output into a probability you can threshold. |
| **Noise floor** | The background loudness level when nothing is happening. |
