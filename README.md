# VoiceCMD Monorepo

Two packages in one repo:

- **`voicecmd`** — reusable library + CLI for a simple voice command recognizer using **FFT → split into N parts → average energy per part**. Stores command profiles in **SQLite**. Includes a **GUI** to record, train, and test live.
- **`voicecmd-snake`** — small Snake game demo to exercise the recognizer with **keyboard** or **live voice**.

> The goal is to provide a clean, OOP, reusable recognizer you can drop into any project (the Snake demo is just a harness).

---

## Features

- OOP library: `AudioRecorder`, `FeatureExtractor`, `ProfileRepository`, `Trainer`, `Recognizer`, `LiveRecognizer`.
- Cross-platform capture via **sounddevice**/**soundfile** (no PyAudio).
- CLI (`voicecmd`) to list devices, record samples, train profiles, recognize files, and evaluate accuracy.
- **GUI (`voicecmd-gui`)**: three tabs — **Dataset** (record & train), **Live** (streaming recognition with noise calibration), and **Evaluation** (batch testing and accuracy measurement).
- Live recognition: sliding window, noise calibration (RMS), confidence threshold.
- **Batch evaluation**: Test model accuracy on directories of audio files with detailed per-file results.
- **Robust training**: Automatically handles missing or corrupted audio files without failing.
- **Consistent data paths**: All components use `~/.voicecmd/` regardless of working directory.
- SQLite persistence (`voicecmd.db`), no hardcoded averages.
- Snake demo (`snake-voice`) for quick end-to-end testing.

---

## Repository layout

```text
voicecmd-monorepo/
├─ README.md
├─ ~/.voicecmd/                   # user data directory (created automatically)
│  ├─ voicecmd.db                 # SQLite database with profiles
│  └─ data/                       # audio recordings
│     ├─ UP/
│     ├─ DOWN/
│     ├─ LEFT/
│     └─ RIGHT/
└─ packages/
   ├─ voicecmd/
   │  ├─ pyproject.toml
   │  └─ src/voicecmd/
   │     ├─ __init__.py
   │     ├─ config.py            # path configuration functions
   │     ├─ audio.py
   │     ├─ features.py
   │     ├─ repository.py
   │     ├─ training.py
   │     ├─ recognition.py
   │     ├─ cli.py              # `voicecmd` entrypoint
   │     └─ gui.py              # `voicecmd-gui` entrypoint (Tkinter)
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
- **Linux (audio libs)**:

  ```bash
  sudo apt-get update
  sudo apt-get install -y libportaudio2 libsndfile1
  ```

- **Tkinter (GUI)**:

  - macOS / Windows (official Python): usually bundled.
  - Linux:
    ```bash
    sudo apt-get install -y python3-tk
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

> If you just added the GUI entrypoint, re-run:
>
> ```bash
> pip install -e packages/voicecmd
> ```

---

## Data Storage

VoiceCMD uses standardized paths so all components (CLI, GUI, Snake) share the same data:

- **Database**: `~/.voicecmd/voicecmd.db`
- **Recordings**: `~/.voicecmd/data/`

This allows you to run any command from any directory and access the same trained models.

**Verify paths**:

```bash
python -c "from voicecmd.config import get_database_path, get_data_dir; print('DB:', get_database_path()); print('Data:', get_data_dir())"
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

Evaluate accuracy on a directory of files:

```bash
voicecmd eval-dir path/to/test/directory EXPECTED_LABEL
voicecmd eval-dir data/UP UP --recursive         # test all UP samples
voicecmd eval-dir validation DOWN --glob "*.wav" # custom pattern
```

### Defaults

- Recordings saved under `~/.voicecmd/data/<COMMAND>/...`
- Profiles saved in `~/.voicecmd/voicecmd.db`
- Works from any directory

---

## GUI

Launch:

```bash
voicecmd-gui
```

### Tabs

- **Dataset (Record/Train)**

  - Choose or create a **Command** (e.g., `UP`).
  - Pick your **Microphone** from the list.
  - Set **Duration (s)** for each take.
  - Click **Record** several times per command (watch **RMS** and progress).
  - Click **Train profiles** to compute/overwrite averages per command.
  - **Recognize WAV…** lets you test any file against trained profiles.

- **Live recognition**

  - Pick your **Microphone**.
  - Adjust:

    - **Window (s)**: analysis window (e.g., 1.6–2.0 s)
    - **Hop (s)**: step between predictions (e.g., 0.2–0.3 s)
    - **Threshold factor**: RMS threshold relative to baseline noise (e.g., 3.0–4.0)
    - **Calibration (s)**: noise calibration time

  - Click **Start** to calibrate and begin predictions. The GUI shows **RMS**, last **Prediction** and **Confidence**.

- **Evaluation**

  - Browse to select a **Directory** containing test audio files.
  - Enter the **Expected Label** for all files in that directory.
  - Optionally modify the **File Pattern** (default: `*.wav`) or enable **Recursive Search**.
  - Click **Start Evaluation** to test accuracy on all matching files.
  - View real-time progress and detailed per-file results with overall accuracy statistics.

---

## Model Evaluation

### CLI Evaluation

Test model accuracy on directories of audio files:

```bash
# Evaluate all .wav files in a directory
voicecmd eval-dir path/to/test/UP UP

# Recursive search with custom pattern
voicecmd eval-dir data/validation DOWN --glob "test_*.wav" --recursive

# Output shows per-file results and overall accuracy
# OK  file1.wav                     -> UP (conf=0.85)
# ERR file2.wav                     -> DOWN (conf=0.67)
#
# Accuracy UP: 8/10 = 80.00%
```

### GUI Evaluation

Use the **Evaluation** tab for interactive testing:

1. **Browse** for directory containing test files
2. Enter **Expected Label** (command name)
3. Optionally modify **File Pattern** or enable **Recursive Search**
4. Click **Start Evaluation** to begin batch testing
5. Monitor **Progress** and view **Real-time Results**
6. Review **Overall Accuracy** and **Per-file Details**

Benefits:

- Visual progress tracking
- Detailed per-file results with confidence scores
- Overall accuracy statistics
- Easy directory browsing and pattern matching

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
- Voice: "UP", "DOWN", "LEFT", "RIGHT" (match the commands you trained)

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
from voicecmd.config import get_database_path
from voicecmd.repository import ProfileRepository
from voicecmd.recognition import Recognizer

repo = ProfileRepository(get_database_path())
rec = Recognizer(repo, num_parts=100)
cmd, conf = rec.recognize_wav("some_clip.wav")
print(cmd, conf)
repo.close()
```

For live streaming, use `LiveRecognizer` (see `voicecmd_snake/voice_controller.py` for a minimal adaptor that pushes predictions into a queue).

---

## Troubleshooting

- **No mic device / permission denied**: grant microphone access to your terminal/IDE (macOS: System Settings → Privacy & Security → Microphone).
- **GUI doesn't start on Linux**: install Tkinter — `sudo apt-get install -y python3-tk`.
- **Different apps see different data**: verify all components use `~/.voicecmd/` with the path verification command above.
- **Training fails with missing files**: The system automatically skips missing or corrupted files and continues training with available recordings. Check the output for `[WARN]` messages about missing files.
- **Poor evaluation accuracy**: Ensure test files match the expected command, record more training samples, and verify audio quality.
- **Too many false activations**: raise `--factor` (e.g., 3.5–4.5) or `--conf` (e.g., 0.65).
- **High latency**: reduce `--window` (1.2–1.6 s) and/or `--hop` (0.2–0.25 s).
- **Poor accuracy**: record more samples, keep distance to mic consistent, ensure `num_parts` in training and inference match.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Credits

- Audio I/O: [sounddevice](https://python-sounddevice.readthedocs.io/), [soundfile](https://pysoundfile.readthedocs.io/).
- GUI: Tkinter (built-in with CPython on most platforms).
- Demo: [pygame](https://www.pygame.org/).
