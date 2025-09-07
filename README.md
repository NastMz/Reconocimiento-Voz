# VoiceCMD Monorepo

Two packages in one repo:

- **`voicecmd`** — reusable library + CLI for a simple voice command recognizer using **FFT → split into N parts → average energy per part**. Stores command profiles in **SQLite**.
- **`voicecmd-snake`** — small Snake game demo to exercise the recognizer with **keyboard** or **live voice**.

> The goal is to provide a clean, OOP, reusable recognizer you can drop into any project (the Snake demo is just a harness).

---

## Features

- OOP library: `AudioRecorder`, `FeatureExtractor`, `ProfileRepository`, `Trainer`, `Recognizer`, `LiveRecognizer`.
- Cross-platform capture via **sounddevice**/**soundfile** (no PyAudio).
- CLI (`voicecmd`) to list devices, record samples, train profiles, and recognize files.
- Live recognition: sliding window, noise calibration (RMS), confidence threshold.
- SQLite persistence (`voicecmd.db`), no hardcoded averages.
- Snake demo (`snake-voice`) for quick end-to-end testing.

---

## Repository layout

```
voicecmd-monorepo/
├─ README.md
├─ data/                          # default recordings folder (created as needed)
└─ packages/
   ├─ voicecmd/
   │  ├─ pyproject.toml
   │  └─ src/voicecmd/
   │     ├─ __init__.py
   │     ├─ config.py
   │     ├─ audio.py
   │     ├─ features.py
   │     ├─ repository.py
   │     ├─ training.py
   │     ├─ recognition.py
   │     └─ cli.py              # `voicecmd` entrypoint
   └─ voicecmd-snake/
      ├─ pyproject.toml
      └─ src/voicecmd_snake/
         ├─ __init__.py
         ├─ voice_controller.py
         └─ game.py             # `snake-voice` entrypoint
```

---

## Prerequisites

- **Python 3.10+**
- Microphone available to the OS.
- **Linux only** (for audio libs):

  ```bash
  sudo apt-get update
  sudo apt-get install -y libportaudio2 libsndfile1
  ```

- **WSL**: audio input is unreliable; run on Windows/macOS/Linux natively.

---

## Quick start (venv)

### 1) Create and activate a venv

macOS / Linux:

```bash
cd voicecmd-monorepo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

Windows (PowerShell):

```powershell
cd voicecmd-monorepo
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
```

### 2) Install both packages (editable)

```bash
pip install -e packages/voicecmd -e packages/voicecmd-snake
```

---

## Train your voice profiles (CLI)

List input devices:

```bash
voicecmd devices
```

Record multiple samples per command (repeat for each):

```bash
voicecmd add-command UP
voicecmd record UP           # add --seconds 2.0 to adjust length

voicecmd add-command DOWN
voicecmd record DOWN

voicecmd add-command LEFT
voicecmd record LEFT

voicecmd add-command RIGHT
voicecmd record RIGHT
```

Train (compute per-command average vectors):

```bash
voicecmd train                # uses default num_parts=100
```

Recognize a WAV file:

```bash
voicecmd recognize path/to/file.wav
```

**Defaults**

- Recordings saved under `data/<COMMAND>/...`
- Profiles saved in `voicecmd.db` (in the current working directory)

---

## Run the demo

Keyboard only:

```bash
snake-voice
```

With voice control:

```bash
snake-voice --voice
```

Useful options:

```bash
snake-voice --voice \
  --device-index 1 \     # from `voicecmd devices`
  --conf 0.60 \          # confidence threshold (0..1)
  --window 2.0 \         # analysis window seconds
  --hop 0.25 \           # step between predictions
  --factor 3.0 \         # RMS threshold = factor × noise baseline
  --calib 0.8            # calibration seconds
```

Controls:

- Keyboard: arrows / WASD
- Voice: “UP”, “DOWN”, “LEFT”, “RIGHT” (match the commands you trained)

---

## How it works (essence kept)

1. Read audio → **FFT**
2. Split FFT vector into **N parts** (default `num_parts=100`)
3. Compute **mean energy per part** → 1D feature vector
4. Train = average vectors per command
5. Infer = compare new vector vs. profiles (**L1 distance after L2 normalization**) and pick the smallest; return a simple confidence score.

---

## Use the library in your own code

```python
from pathlib import Path
from voicecmd.repository import ProfileRepository
from voicecmd.recognition import Recognizer

repo = ProfileRepository(Path("voicecmd.db"))
rec = Recognizer(repo, num_parts=100)
cmd, conf = rec.recognize_wav("some_clip.wav")
print(cmd, conf)
repo.close()
```

For live streaming, use `LiveRecognizer` (see `voicecmd_snake/voice_controller.py` for a minimal adaptor that pushes predictions into a queue).

---

## Troubleshooting

- **No mic device / permission denied**: grant microphone access to your terminal/IDE (macOS: System Settings → Privacy & Security → Microphone).
- **Too many false activations**: raise `--factor` (e.g., 3.5–4.5) or `--conf` (e.g., 0.65).
- **High latency**: reduce `--window` (1.2–1.6 s) and/or `--hop` (0.2–0.25 s).
- **Poor accuracy**: record more samples, keep distance to mic consistent, ensure `num_parts` in training and inference match.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Credits

- Audio I/O: [sounddevice](https://python-sounddevice.readthedocs.io/), [soundfile](https://pysoundfile.readthedocs.io/).
- Demo: [pygame](https://www.pygame.org/).
- Core idea preserved: FFT → band energies → averages per command.
