"""
Audio feature extraction module for voice command recognition.

This module provides utilities for extracting meaningful features from audio signals
using Fast Fourier Transform (FFT) analysis. The primary focus is on spectral energy
distribution across frequency bands, which is particularly effective for voice
command classification and audio pattern recognition.

The module implements frequency domain analysis by:
1. Converting time-domain audio signals to frequency domain using FFT
2. Dividing the frequency spectrum into configurable bands
3. Computing energy levels for each frequency band
4. Returning a feature vector representing the spectral energy distribution

Classes:
    FeatureExtractor: Main class for extracting FFT-based spectral features
                     from audio data (WAV files or numpy arrays)

Dependencies:
    - numpy: For numerical computations and array operations
    - scipy.io.wavfile: For reading WAV audio files
    - scipy.fft.rfft: For real-valued Fast Fourier Transform computation
"""

from __future__ import annotations
import numpy as np
from scipy.io import wavfile
from scipy.fft import rfft


class FeatureExtractor:
    """
    Extract spectral energy features from audio signals using FFT analysis.

    This class implements feature extraction based on frequency domain analysis,
    converting audio signals into numerical feature vectors that represent the
    energy distribution across different frequency bands. This approach is
    particularly effective for voice command recognition and audio classification
    tasks.

    The extraction process:
    1. Applies Real FFT to convert time-domain audio to frequency domain
    2. Divides the frequency spectrum into configurable number of bands
    3. Computes the average energy (squared magnitude) for each band
    4. Returns a feature vector with energy values for each frequency band

    Attributes:
        num_parts (int): Number of frequency bands to divide the spectrum into.
                        Higher values provide finer frequency resolution but
                        increase computational cost and feature dimensionality.

    Examples:
        >>> # Create extractor with default 100 frequency bands
        >>> extractor = FeatureExtractor()
        >>> 
        >>> # Extract features from WAV file
        >>> features = extractor.from_wav("voice_command.wav")
        >>> print(f"Feature vector shape: {features.shape}")
        Feature vector shape: (100,)
        >>> 
        >>> # Extract features from audio array
        >>> import numpy as np
        >>> audio_data = np.random.randn(16000)  # 1 second at 16kHz
        >>> features = extractor.from_array(audio_data)
        >>> 
        >>> # Use different number of frequency bands
        >>> fine_extractor = FeatureExtractor(num_parts=200)
        >>> detailed_features = fine_extractor.from_wav("audio.wav")
        >>> print(f"Detailed features shape: {detailed_features.shape}")
        Detailed features shape: (200,)

    Note:
        The choice of num_parts affects the trade-off between:
        - Frequency resolution vs computational efficiency
        - Feature vector size vs processing speed
        - Classification accuracy vs memory usage

        Typical values:
        - 32-64: Fast processing, suitable for real-time applications
        - 100: Good balance (default), suitable for most voice commands
        - 150-300: High resolution, suitable for complex audio analysis
    """

    def __init__(self, num_parts: int = 100):
        """
        Initialize the feature extractor with specified frequency resolution.

        Args:
            num_parts (int): Number of frequency bands to divide the spectrum into.
                           Must be a positive integer. Default is 100, which provides
                           a good balance between feature resolution and computational
                           efficiency for voice command recognition.

        Raises:
            ValueError: If num_parts is not a positive integer.

        Examples:
            >>> # Default configuration for general use
            >>> extractor = FeatureExtractor()
            >>> 
            >>> # High-resolution configuration for detailed analysis
            >>> detailed_extractor = FeatureExtractor(num_parts=200)
            >>> 
            >>> # Fast configuration for real-time processing
            >>> fast_extractor = FeatureExtractor(num_parts=50)
        """
        if num_parts <= 0:
            raise ValueError("num_parts must be a positive integer")
        self.num_parts = int(num_parts)

    def from_wav(self, path: str) -> np.ndarray:
        """
        Extract spectral energy features from a WAV audio file.

        Reads an audio file from disk and extracts frequency domain features
        by computing the energy distribution across frequency bands. Automatically
        handles stereo to mono conversion if necessary.

        Args:
            path (str): Path to the WAV audio file. The file should be in a
                       standard audio format supported by scipy.io.wavfile
                       (typically 16-bit or 24-bit PCM WAV files).

        Returns:
            np.ndarray: Feature vector of shape (num_parts,) containing the
                       average energy for each frequency band. Values are
                       non-negative float64 numbers representing spectral energy.

        Raises:
            FileNotFoundError: If the specified audio file doesn't exist.
            ValueError: If the file is not a valid WAV file or is corrupted.
            scipy.io.wavfile.WavFileWarning: If there are issues with WAV format.

        Examples:
            >>> extractor = FeatureExtractor(num_parts=100)
            >>> 
            >>> # Extract features from a voice command recording
            >>> features = extractor.from_wav("data/hello.wav")
            >>> print(f"Energy in first frequency band: {features[0]:.3f}")
            >>> 
            >>> # Process multiple files
            >>> file_paths = ["cmd1.wav", "cmd2.wav", "cmd3.wav"]
            >>> all_features = [extractor.from_wav(path) for path in file_paths]
            >>> feature_matrix = np.vstack(all_features)
            >>> print(f"Feature matrix shape: {feature_matrix.shape}")

        Note:
            - Stereo files are automatically converted to mono by averaging channels
            - The function works with various sample rates and bit depths
            - For consistent results, use audio files with similar characteristics
        """
        _, data = wavfile.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1).astype(data.dtype)
        x = data.astype(np.float64)
        return self._fft_energy(x)

    def from_array(self, x: np.ndarray) -> np.ndarray:
        """
        Extract spectral energy features from a numpy audio array.

        Processes audio data that is already loaded in memory as a numpy array.
        This method is useful when working with audio data from real-time
        recording, preprocessing pipelines, or when the audio is already
        loaded from other sources.

        Args:
            x (np.ndarray): Audio data array. Can be 1D (mono) or 2D (stereo).
                          For 2D arrays, channels should be along axis 1.
                          Data type can be int16, int32, float32, or float64.
                          Shape should be (samples,) for mono or (samples, channels)
                          for multi-channel audio.

        Returns:
            np.ndarray: Feature vector of shape (num_parts,) containing the
                       average energy for each frequency band. Values are
                       non-negative float64 numbers representing spectral energy.

        Raises:
            ValueError: If the input array is empty or has invalid dimensions.
            TypeError: If the input is not a numpy array or compatible type.

        Examples:
            >>> import numpy as np
            >>> extractor = FeatureExtractor(num_parts=64)
            >>> 
            >>> # Process mono audio data (e.g., from microphone)
            >>> mono_audio = np.random.randn(8000).astype(np.int16)
            >>> features = extractor.from_array(mono_audio)
            >>> print(f"Mono features shape: {features.shape}")
            Mono features shape: (64,)
            >>> 
            >>> # Process stereo audio data
            >>> stereo_audio = np.random.randn(8000, 2).astype(np.float32)
            >>> features = extractor.from_array(stereo_audio)
            >>> print(f"Stereo features shape: {features.shape}")
            Stereo features shape: (64,)
            >>> 
            >>> # Process audio from AudioRecorder
            >>> from voicecmd.audio import AudioRecorder
            >>> from voicecmd.config import AudioConfig
            >>> 
            >>> config = AudioConfig(rate=16000, duration=2.0)
            >>> recorder = AudioRecorder(config)
            >>> recorded_audio = recorder.record()
            >>> features = extractor.from_array(recorded_audio)

        Note:
            - Multi-channel audio is automatically converted to mono by averaging
            - Input data is converted to float64 for numerical precision
            - Works with various input data types and ranges
            - No normalization is applied; consider preprocessing if needed
        """
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x.astype(np.float64)
        return self._fft_energy(x)

    def _fft_energy(self, x: np.ndarray) -> np.ndarray:
        """
        Compute frequency band energies using FFT analysis.

        This private method performs the core feature extraction algorithm:
        1. Applies Real FFT to convert time-domain signal to frequency domain
        2. Pads the FFT result if necessary to ensure even division into bands
        3. Splits the frequency spectrum into the specified number of bands
        4. Computes the average energy (mean squared magnitude) for each band

        The Real FFT is used instead of full FFT since audio signals are real-valued,
        which provides computational efficiency and avoids redundant information.

        Args:
            x (np.ndarray): Input audio signal as a 1D array of float64 values.
                          Should be preprocessed (mono, correct data type) before
                          calling this method.

        Returns:
            np.ndarray: Energy values for each frequency band, shape (num_parts,).
                       Each value represents the average squared magnitude of
                       frequency components in that band. Higher values indicate
                       more energy at those frequencies.

        Algorithm Details:
            - Uses scipy.fft.rfft for efficient real-valued FFT computation
            - Padding ensures the frequency spectrum can be evenly divided
            - Energy is computed as mean(|FFT_coefficients|²) per band
            - Results are returned as float64 for numerical precision

        Examples:
            >>> # This is a private method, typically called internally
            >>> extractor = FeatureExtractor(num_parts=4)
            >>> audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 1000))
            >>> energies = extractor._fft_energy(audio)
            >>> print(f"Energy distribution: {energies}")
            >>> # Would show higher energy in the band containing 440 Hz

        Note:
            - The method assumes input is already preprocessed (mono, float64)
            - Padding with zeros doesn't affect the energy computation significantly
            - Energy values are always non-negative due to squared magnitude computation
            - Higher sample rates will spread energy across more frequency bins
        """
        # Apply Real FFT to convert to frequency domain
        X = rfft(x)
        n = len(X)

        # Pad FFT result if needed to ensure even division into bands
        r = (-n) % self.num_parts
        if r:
            X = np.pad(X, (0, r), mode='constant', constant_values=0)

        # Split frequency spectrum into equal-sized bands
        parts = np.array_split(X, self.num_parts)

        # Compute average energy (mean squared magnitude) for each band
        energies = np.array([np.mean(np.abs(p) ** 2)
                            for p in parts], dtype=np.float64)
        return energies
