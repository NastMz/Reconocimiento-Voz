"""
Command-line interface for VoiceCMD voice command recognition system.

This module provides a comprehensive CLI for managing voice command recognition
workflows, from initial setup and data collection through training and inference.
The interface is built using Typer for modern, user-friendly command-line
interactions with automatic help generation and type validation.

The CLI supports the complete voice recognition pipeline:
1. **Device Management**: List and select audio input devices
2. **Command Setup**: Create and manage voice command definitions
3. **Data Collection**: Record training audio samples with metadata
4. **Model Training**: Generate recognition profiles from recorded samples
5. **Inference**: Recognize voice commands from audio files

Key Features:
- Automatic database and directory management
- Timestamped audio file organization
- Progress feedback and error handling
- Configurable feature extraction parameters
- Cross-platform audio device support

File Organization:
- Database: voicecmd.db (SQLite database for metadata and profiles)
- Audio Data: data/ directory with command-specific subdirectories
- File Naming: COMMAND_YYYYMMDD_HHMMSS.wav format for chronological organization

Commands:
    devices: List available audio input devices
    add-command: Create a new voice command definition
    record: Record training samples for a command
    train: Generate recognition profiles from training data
    recognize: Test recognition on audio files

Dependencies:
    - typer: Modern CLI framework with automatic help and validation
    - pathlib: Cross-platform path handling
    - datetime: Timestamp generation for file naming
    - All voicecmd modules: Core recognition system components

Usage Examples:
    # List audio devices
    voicecmd devices
    
    # Set up a new command
    voicecmd add-command hello
    
    # Record training samples
    voicecmd record hello --device-index 1 --seconds 2.5
    
    # Train recognition models
    voicecmd train --num-parts 100
    
    # Test recognition
    voicecmd recognize test_audio.wav --num-parts 100
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import typer

from .config import AudioConfig, FeatureConfig, get_database_path, get_data_dir
from .audio import AudioDevices, AudioRecorder, WavWriter
from .repository import ProfileRepository
from .training import Trainer
from .recognition import Recognizer

# Main CLI application with descriptive help text
app = typer.Typer(help="VoiceCMD — CLI for voice command recognition system")

# Configuration constants for file organization
DATA_DIR = get_data_dir()  # Base directory for audio recordings
DB_PATH = get_database_path()  # SQLite database for metadata and profiles


def _repo() -> ProfileRepository:
    """
    Create and return a ProfileRepository instance.

    Provides a centralized way to create repository connections with
    consistent database path configuration.

    Returns:
        ProfileRepository: Connected repository instance

    Note:
        Callers are responsible for calling close() on the returned repository.
    """
    return ProfileRepository(DB_PATH)


@app.command()
def devices():
    """
    List all available audio input devices on the system.

    Displays a numbered list of all audio input devices (microphones, line-in,
    etc.) that can be used for voice command recording. The device indices
    shown can be used with the --device-index option in the record command
    to select a specific input device.

    Output Format:
        Each line shows: <device_index>: <device_name>

    Examples:
        $ voicecmd devices
        0: Built-in Microphone
        1: USB Audio Device
        2: External Microphone Array

    Use Cases:
        - Initial setup to identify available recording devices
        - Troubleshooting audio input issues
        - Selecting optimal microphone for voice command recording
        - Multi-device environments requiring specific device selection

    Note:
        Device indices may change when hardware is connected/disconnected.
        Run this command again if your audio setup changes.
    """
    for i, name in AudioDevices.list_input():
        typer.echo(f"{i}: {name}")


@app.command()
def add_command(name: str):
    """
    Create a new voice command definition in the system.

    Registers a new command name in the database, making it available for
    training data collection. Command names are automatically converted to
    uppercase for consistency and stored with creation timestamps.

    Args:
        name: The command name to create (e.g., "hello", "stop", "play").
              Will be converted to uppercase automatically.

    Examples:
        $ voicecmd add-command hello
        OK: HELLO

        $ voicecmd add-command "play music"
        OK: PLAY MUSIC

    Database Impact:
        - Creates new entry in commands table
        - Sets creation timestamp
        - Enables the command for recording and training

    Use Cases:
        - Initial system setup with command vocabulary
        - Adding new commands to existing system
        - Preparing for training data collection

    Note:
        - Command names are case-insensitive but stored in uppercase
        - Duplicate names are handled gracefully (no error)
        - Use descriptive names that match expected user speech
    """
    r = _repo()
    try:
        r.upsert_command(name.upper())
        typer.echo(f"OK: {name.upper()}")
    finally:
        r.close()


@app.command()
def record(
    name: str,
    device_index: int = typer.Option(
        None, help="Audio device index (use 'devices' command to list)"),
    seconds: float = typer.Option(2.0, help="Recording duration in seconds")
):
    """
    Record training audio samples for a voice command.

    Captures audio from the specified input device and saves it as a timestamped
    WAV file in the data directory. The recording metadata is automatically
    stored in the database for use during training. Each recording is organized
    in command-specific subdirectories with chronological file naming.

    Args:
        name: Command name to record (must exist, use add-command first)
        device_index: Audio input device index (None = system default)
        seconds: Recording duration in seconds (default: 2.0)

    File Organization:
        - Directory: data/COMMAND_NAME/
        - Filename: COMMAND_NAME_YYYYMMDD_HHMMSS.wav
        - Format: 16-bit PCM WAV, mono channel

    Examples:
        # Record with default device and duration
        $ voicecmd record hello
        Recorded: data/HELLO/HELLO_20231015_143022.wav

        # Record with specific device and duration
        $ voicecmd record stop --device-index 1 --seconds 3.0
        Recorded: data/STOP/STOP_20231015_143055.wav

        # Multiple samples for better training
        $ voicecmd record hello
        $ voicecmd record hello
        $ voicecmd record hello

    Best Practices:
        - Record 5-10 samples per command for robust training
        - Vary pronunciation, tone, and speed slightly
        - Use consistent background noise conditions
        - Speak clearly and at normal volume
        - Wait for the full duration (avoid cutting off)

    Workflow Integration:
        1. Use 'devices' to identify audio input device
        2. Use 'add-command' to create command definition
        3. Use 'record' multiple times to collect training data
        4. Use 'train' to generate recognition profiles

    Technical Details:
        - Audio format: 44.1kHz sample rate, 16-bit depth, mono
        - Automatic directory creation for organization
        - Metadata includes path, sample rate, channels, duration
        - Timestamps ensure unique filenames and chronological ordering
    """
    cfg = AudioConfig(duration=seconds)
    rec = AudioRecorder(cfg, device_index=device_index)
    x = rec.record()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = DATA_DIR / name.upper() / f"{name.upper()}_{stamp}.wav"
    WavWriter.write(out, x, cfg.rate)
    r = _repo()
    try:
        cmd_id = r.upsert_command(name.upper())
        r.add_recording(cmd_id, str(out), cfg.rate, cfg.channels, cfg.duration)
        typer.echo(f"Recorded: {out}")
    finally:
        r.close()


@app.command()
def train(num_parts: int = typer.Option(100, help="Number of frequency bands for feature extraction")):
    """
    Generate recognition profiles from recorded training data.

    Processes all recorded audio samples to create feature profiles for voice
    command recognition. The training process extracts spectral features from
    each recording and computes centroid profiles that represent each command's
    acoustic characteristics.

    Args:
        num_parts: Number of frequency bands for feature extraction (default: 100)
                  Higher values provide finer resolution but increase computation

    Training Process:
        1. Load all commands and their associated recordings
        2. Extract spectral features from each audio file
        3. Compute centroid (mean) features for each command
        4. Store profiles in database for recognition use

    Examples:
        # Train with default feature resolution
        $ voicecmd train
        [OK] Profile updated: HELLO (num_parts=100)
        [OK] Profile updated: STOP (num_parts=100)
        [WARN] No recordings for PLAY

        # Train with high-resolution features
        $ voicecmd train --num-parts 200
        [OK] Profile updated: HELLO (num_parts=200)
        [OK] Profile updated: STOP (num_parts=200)

    Output Messages:
        - [OK]: Profile successfully created/updated
        - [WARN]: Command exists but has no training recordings

    Feature Configuration:
        - num_parts=50: Fast training, lower accuracy
        - num_parts=100: Balanced performance (recommended)
        - num_parts=200: High accuracy, slower processing

    Requirements:
        - At least one recording per command for successful training
        - All audio files must be accessible at stored paths
        - Sufficient disk space for profile storage

    Best Practices:
        - Use 5-10 recordings per command for robust profiles
        - Ensure consistent audio quality across recordings
        - Re-train after adding new recordings for updated profiles
        - Use same num_parts value for training and recognition

    Technical Details:
        - Features based on FFT spectral energy distribution
        - Centroids computed as arithmetic mean of all recordings
        - Profiles stored with feature configuration metadata
        - Existing profiles with same configuration are overwritten
    """
    r = _repo()
    try:
        Trainer(r, num_parts).train_all()
    finally:
        r.close()


@app.command()
def recognize(
    wav: Path,
    num_parts: int = typer.Option(
        100, help="Number of frequency bands (must match training)")
):
    """
    Recognize voice commands from audio files.

    Analyzes a WAV audio file to identify the most likely voice command based
    on trained profiles. Returns the recognized command name and confidence
    score for evaluation and decision making.

    Args:
        wav: Path to WAV audio file to analyze
        num_parts: Number of frequency bands (must match training configuration)

    Output Format:
        - Success: "Prediction: COMMAND_NAME (conf=X.XX)"
        - No profiles: "No trained profiles available."

    Examples:
        # Recognize a command
        $ voicecmd recognize test_hello.wav
        Prediction: HELLO (conf=0.85)

        # High-resolution recognition (if trained with 200 parts)
        $ voicecmd recognize test_hello.wav --num-parts 200
        Prediction: HELLO (conf=0.92)

        # No matching profiles
        $ voicecmd recognize test_unknown.wav
        No trained profiles available.

    Confidence Interpretation:
        - 0.8-1.0: Very high confidence, reliable recognition
        - 0.6-0.8: High confidence, likely correct
        - 0.4-0.6: Medium confidence, possible but uncertain
        - 0.0-0.4: Low confidence, likely incorrect or noise

    Requirements:
        - Trained profiles must exist (run 'train' command first)
        - num_parts must match the value used during training
        - Audio file must be valid WAV format

    Use Cases:
        - Testing recognition accuracy on known samples
        - Batch processing of audio files
        - Validating training effectiveness
        - Debugging recognition issues

    Integration:
        - Use in scripts for automated testing
        - Combine with confidence thresholds for decision logic
        - Process multiple files to evaluate system performance

    Technical Details:
        - Uses same feature extraction as training
        - Manhattan distance matching against stored profiles
        - Confidence based on inverse distance relationship
        - Supports mono and stereo audio (converted to mono)
    """
    r = _repo()
    try:
        name, conf = Recognizer(r, num_parts).recognize_wav(str(wav))
        if name is None:
            typer.echo("No trained profiles available.")
        else:
            typer.echo(f"Prediction: {name} (conf={conf:.2f})")
    finally:
        r.close()


@app.command("eval-dir")
def eval_dir(
    dir: Path,
    label: str,
    glob: str = "*.wav",
    recursive: bool = False,
    num_parts: int = 100,
    db: Path = DB_PATH,
):
    """
    Evaluate recognition accuracy on a directory of WAV files.
    """
    r = ProfileRepository(db)
    try:
        recog = Recognizer(r, num_parts)
        it = dir.rglob(glob) if recursive else dir.glob(glob)
        files = [p for p in it if p.is_file()]
        if not files:
            typer.echo("No files found.")
            raise typer.Exit(code=2)

        total = 0
        ok = 0
        for f in sorted(files):
            pred, conf = recog.recognize_wav(str(f))
            total += 1
            hit = (pred or "").upper() == label.upper()
            ok += 1 if hit else 0
            typer.echo(
                f"{'OK ' if hit else 'ERR'} {f.name:40s} -> {pred} (conf={conf:.2f})")

        acc = ok / total if total else 0.0
        typer.echo(f"\nAccuracy {label.upper()}: {ok}/{total} = {acc:.2%}")
    finally:
        r.close()
