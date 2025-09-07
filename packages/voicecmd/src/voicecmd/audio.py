"""
Audio processing module for voice command recognition system.

This module provides utilities for audio device management, recording,
file I/O operations, and real-time audio streaming.

Classes:
    AudioDevices: Utilities for listing and selecting input audio devices
    WavWriter: Handles writing audio data to WAV files
    AudioRecorder: Records audio from microphone with configurable settings
    RingBuffer: Circular buffer for efficient real-time audio streaming

Dependencies:
    - numpy: For audio data manipulation
    - sounddevice: For audio I/O operations
    - soundfile: For audio file handling
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


import numpy as np
import sounddevice as sd
import soundfile as sf


from .config import AudioConfig


class AudioDevices:
    """
    Utilities for discovering and managing input audio devices.

    This class provides static methods to query the system for available
    audio input devices and retrieve their properties.
    """

    @staticmethod
    def list_input() -> list[tuple[int, str]]:
        """
        List all available input audio devices on the system.

        Queries the system for audio devices that have input capabilities
        (microphones, line-in, etc.) and returns their device indices and names.

        Returns:
            list[tuple[int, str]]: A list of tuples containing:
                - Device index (int): The device ID used by sounddevice
                - Device name (str): Human-readable device name

        Examples:
            >>> devices = AudioDevices.list_input()
            >>> for idx, name in devices:
            ...     print(f"Device {idx}: {name}")
            Device 0: Built-in Microphone
            Device 2: USB Audio Device
        """
        devices = sd.query_devices()
        out: list[tuple[int, str]] = []
        for idx, d in enumerate(devices):
            if int(d.get("max_input_channels", 0)) > 0:
                out.append((idx, d.get("name", f"Device {idx}")))
        return out


class WavWriter:
    """
    Handles writing audio data to WAV files with automatic directory creation.

    This class provides utilities for persisting audio data to disk in WAV format,
    supporting both float32 and int16 audio data types with automatic conversion
    to 16-bit PCM format.
    """

    @staticmethod
    def write(path: Path, data: np.ndarray, rate: int) -> None:
        """
        Write audio data to a WAV file.

        Creates the parent directories if they don't exist and writes the audio
        data to the specified path in 16-bit PCM format.

        Args:
            path (Path): The file path where the WAV file will be saved
            data (np.ndarray): Audio data array. Can be float32 in range [-1,1]
                             or int16 format
            rate (int): Sample rate in Hz (e.g., 44100, 22050, 16000)

        Examples:
            >>> import numpy as np
            >>> from pathlib import Path
            >>> 
            >>> # Generate a simple sine wave
            >>> rate = 16000
            >>> duration = 1.0
            >>> t = np.linspace(0, duration, int(rate * duration))
            >>> audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            >>> 
            >>> # Save to file
            >>> WavWriter.write(Path("output/test.wav"), audio_data, rate)
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        # Accept float32 [-1,1] or int16
        sf.write(str(path), data, rate, subtype="PCM_16")


@dataclass
class AudioRecorder:
    """
    Records audio from microphone with configurable settings.

    This class provides a convenient interface for recording audio from
    microphones or other input devices, with support for different sample
    rates, channels, and recording durations.

    Attributes:
        config (AudioConfig): Audio configuration containing sample rate,
                             channels, and default duration settings
        device_index (int | None): Specific audio device to use. If None,
                                  uses the system default input device

    Examples:
        >>> from voicecmd.config import AudioConfig
        >>> 
        >>> # Create recorder with default device
        >>> config = AudioConfig(rate=16000, channels=1, duration=3.0)
        >>> recorder = AudioRecorder(config)
        >>> 
        >>> # Record 5 seconds of audio
        >>> audio_data = recorder.record(seconds=5.0)
        >>> 
        >>> # Record using configured duration
        >>> audio_data = recorder.record()
    """

    config: AudioConfig
    device_index: int | None = None

    def record(self, seconds: float | None = None) -> np.ndarray:
        """
        Record audio from the configured input device.

        Records audio for the specified duration, automatically handling
        multi-channel to mono conversion and data type conversion from
        float32 to int16 for pipeline compatibility.

        Args:
            seconds (float | None): Recording duration in seconds. If None,
                                  uses the duration from config

        Returns:
            np.ndarray: Audio data as int16 array in range [-32767, 32767].
                       Always returns mono audio regardless of input channels

        Raises:
            sounddevice.PortAudioError: If there's an issue with the audio device

        Examples:
            >>> recorder = AudioRecorder(config, device_index=0)
            >>> 
            >>> # Record for 3 seconds
            >>> audio = recorder.record(3.0)
            >>> print(f"Recorded {len(audio)} samples")
            >>> 
            >>> # Record using default duration from config
            >>> audio = recorder.record()
        """
        sec = float(seconds if seconds is not None else self.config.duration)
        frames = int(sec * self.config.rate)
        # dtype float32 [-1,1]
        data = sd.rec(
            frames,
            samplerate=self.config.rate,
            channels=self.config.channels,
            dtype="float32",
            device=self.device_index,
            blocking=True,
        )
        sd.wait()
        # Mono if there are multiple channels
        if data.ndim > 1:
            data = data.mean(axis=1)
        # Returns int16 to maintain compatibility with pipeline
        data = np.clip(data, -1.0, 1.0)
        return (data * 32767.0).astype(np.int16)


class RingBuffer:
    """
    Circular buffer implementation for efficient real-time audio streaming.

    A ring buffer (circular buffer) that maintains a fixed-size buffer for
    audio data, automatically overwriting old data when the buffer is full.
    This is particularly useful for real-time audio processing where you
    need to maintain a sliding window of recent audio samples.

    Attributes:
        buf (np.ndarray): Internal buffer storing audio samples as int16
        capacity (int): Maximum number of samples the buffer can hold
        _write (int): Current write position in the buffer
        _filled (int): Number of samples currently stored in the buffer

    Examples:
        >>> # Create a buffer for 1 second of audio at 16kHz
        >>> buffer = RingBuffer(capacity=16000)
        >>> 
        >>> # Add some audio data
        >>> new_audio = np.array([100, 200, 300], dtype=np.int16)
        >>> buffer.push(new_audio)
        >>> 
        >>> # Read the last 100 samples
        >>> recent_audio = buffer.read_latest(100)
    """

    def __init__(self, capacity: int):
        """
        Initialize the ring buffer with specified capacity.

        Args:
            capacity (int): Maximum number of int16 samples the buffer can hold
        """
        self.buf = np.zeros(capacity, dtype=np.int16)
        self.capacity = capacity
        self._write = 0
        self._filled = 0

    def push(self, x: np.ndarray):
        """
        Add new audio data to the buffer.

        Appends the audio data to the buffer, automatically wrapping around
        when the buffer is full. If the input data is larger than the buffer
        capacity, only the last `capacity` samples are kept.

        Args:
            x (np.ndarray): Audio data to add to the buffer. Should be int16 format

        Examples:
            >>> buffer = RingBuffer(1000)
            >>> audio_chunk = np.array([1, 2, 3, 4, 5], dtype=np.int16)
            >>> buffer.push(audio_chunk)
            >>> print(f"Buffer now contains {buffer._filled} samples")
        """
        n = len(x)
        n = min(n, self.capacity)
        idx = (self._write + np.arange(n)) % self.capacity
        self.buf[idx] = x[-n:]
        self._write = (self._write + n) % self.capacity
        self._filled = min(self.capacity, self._filled + n)

    def read_latest(self, n: int) -> np.ndarray:
        """
        Read the most recent n samples from the buffer.

        Retrieves the last n samples that were added to the buffer,
        maintaining their chronological order. If fewer than n samples
        are available, returns all available samples.

        Args:
            n (int): Number of recent samples to retrieve

        Returns:
            np.ndarray: Array containing the n most recent samples in
                       chronological order (oldest first)

        Examples:
            >>> buffer = RingBuffer(1000)
            >>> # Add some data
            >>> buffer.push(np.array([1, 2, 3, 4, 5], dtype=np.int16))
            >>> 
            >>> # Get the last 3 samples
            >>> recent = buffer.read_latest(3)
            >>> print(recent)  # Should show [3, 4, 5]
            >>> 
            >>> # Try to get more samples than available
            >>> all_data = buffer.read_latest(10)  # Returns all 5 samples
        """
        n = min(n, self._filled)
        start = (self._write - n) % self.capacity
        idx = (start + np.arange(n)) % self.capacity
        return self.buf[idx]
