"""
Configuration module for voice command recognition system.

This module defines configuration classes that encapsulate all the parameters
needed for audio processing, feature extraction, and voice recognition tasks.
The configurations are implemented as frozen dataclasses to ensure immutability
and type safety.

Classes:
    AudioConfig: Configuration for audio capture and recording settings
    FeatureConfig: Configuration for audio feature extraction parameters

The frozen dataclass decorator ensures that configuration objects are immutable
after creation, preventing accidental modification during runtime.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    """
    Configuration class for audio capture and recording settings.

    This immutable configuration class defines all parameters necessary for
    audio recording and processing operations. It encapsulates sample rate,
    channel configuration, buffer management, and timing settings used
    throughout the voice recognition pipeline.

    The class is frozen to prevent accidental modification of audio parameters
    during runtime, ensuring consistent behavior across the application.

    Attributes:
        rate (int): Sample rate in Hz. Controls the number of audio samples
                   captured per second. Common values:
                   - 44100: CD quality audio (default)
                   - 22050: Half CD quality, good for voice
                   - 16000: Standard for speech recognition
                   - 8000: Telephone quality

        channels (int): Number of audio channels to capture. Options:
                       - 1: Mono recording (default, recommended for voice)
                       - 2: Stereo recording

        chunk (int): Buffer size for real-time audio streaming in samples.
                    Controls the granularity of audio processing:
                    - Smaller values: Lower latency, higher CPU usage
                    - Larger values: Higher latency, lower CPU usage
                    Default is 1024 samples (good balance for most use cases)

        duration (float): Default duration for audio capture sessions in seconds.
                         This is used as the default recording time when no
                         specific duration is requested. Default is 2.0 seconds.

    Examples:
        >>> # Create default configuration
        >>> config = AudioConfig()
        >>> print(f"Sample rate: {config.rate} Hz")
        Sample rate: 44100 Hz

        >>> # Create configuration optimized for speech recognition
        >>> speech_config = AudioConfig(
        ...     rate=16000,      # Lower sample rate for speech
        ...     channels=1,      # Mono for voice commands
        ...     chunk=512,       # Smaller chunks for lower latency
        ...     duration=3.0     # 3 seconds per command
        ... )

        >>> # Create high-quality audio configuration
        >>> hq_config = AudioConfig(
        ...     rate=48000,      # High sample rate
        ...     channels=2,      # Stereo recording
        ...     chunk=2048,      # Larger chunks for stability
        ...     duration=5.0     # Longer recordings
        ... )

        >>> # Configuration is immutable
        >>> # config.rate = 22050  # This would raise an error

    Note:
        The chunk size affects real-time performance. For voice commands,
        smaller chunks (256-1024) provide better responsiveness, while
        larger chunks (2048-4096) are better for batch processing.
    """

    rate: int = 44100
    channels: int = 1
    chunk: int = 1024  # used for streaming
    duration: float = 2.0  # seconds per capture


@dataclass(frozen=True)
class FeatureConfig:
    """
    Configuration class for audio feature extraction parameters.

    This immutable configuration class defines parameters used during the
    feature extraction phase of audio processing. Feature extraction converts
    raw audio signals into numerical representations that can be used for
    machine learning and pattern recognition tasks.

    The class is frozen to ensure that feature extraction parameters remain
    consistent throughout the processing pipeline, preventing bugs that could
    arise from parameter changes during runtime.

    Attributes:
        num_parts (int): Number of frequency bands (bins) to divide the audio
                        spectrum into during spectral analysis. This parameter
                        controls the frequency resolution of the extracted features.

                        Effects of different values:
                        - Lower values (20-50): Coarse frequency resolution,
                          faster processing, less memory usage
                        - Medium values (50-150): Balanced resolution and performance
                          (default: 100)
                        - Higher values (150+): Fine frequency resolution,
                          slower processing, more memory usage

                        The optimal value depends on:
                        - Audio content complexity
                        - Available computational resources
                        - Required classification accuracy

    Examples:
        >>> # Create default feature configuration
        >>> config = FeatureConfig()
        >>> print(f"Frequency bands: {config.num_parts}")
        Frequency bands: 100

        >>> # Configuration for fast processing (lower resolution)
        >>> fast_config = FeatureConfig(num_parts=50)
        >>> # Suitable for real-time applications or limited resources

        >>> # Configuration for high accuracy (higher resolution)
        >>> precise_config = FeatureConfig(num_parts=200)
        >>> # Suitable for offline analysis or when accuracy is critical

        >>> # Configuration for simple voice commands
        >>> voice_config = FeatureConfig(num_parts=64)
        >>> # Good balance for speech recognition tasks

        >>> # Configuration is immutable
        >>> # config.num_parts = 150  # This would raise an error

    Note:
        The number of frequency parts directly impacts:
        - Feature vector dimensionality
        - Computational complexity (O(n) where n = num_parts)
        - Memory usage for storing features
        - Training time for machine learning models

        Consider your specific use case when choosing this parameter:
        - Voice commands: 32-64 parts usually sufficient
        - Music analysis: 100-200 parts recommended
        - General audio: 100 parts (default) is a good starting point
    """

    num_parts: int = 100  # N bands
