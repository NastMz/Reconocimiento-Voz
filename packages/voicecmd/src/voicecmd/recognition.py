"""
Voice command recognition module for real-time and batch audio processing.

This module provides the inference engine for voice command recognition, supporting
both offline processing (files/arrays) and real-time streaming recognition. The
recognition system uses distance-based matching against trained command profiles
with confidence scoring and adaptive noise thresholding for live recognition.

Recognition Algorithm:
1. Extract spectral features from input audio using the same configuration as training
2. Normalize both current features and stored profiles using L2 normalization
3. Compute Manhattan distance (L1 norm) between normalized feature vectors
4. Select the command with minimum distance as the best match
5. Calculate confidence score based on inverse distance relationship

Key Features:
- Batch recognition from WAV files or numpy arrays
- Real-time streaming recognition with adaptive noise thresholding
- Confidence scoring for reliability assessment
- Multi-threaded live processing with callback system
- Automatic ambient noise calibration and voice activity detection

Classes:
    Recognizer: Core recognition engine for batch processing
    LiveRecognizer: Real-time streaming recognition with noise adaptation

Functions:
    _normalize: L2 normalization utility for feature vectors

Dependencies:
    - threading: For concurrent real-time processing
    - numpy: For numerical computations and array operations
    - sounddevice: For real-time audio streaming
    - repository: For loading trained command profiles
    - features: For consistent feature extraction
    - audio: For circular buffer management
    - config: For audio configuration parameters

Mathematical Foundation:
    Distance metric: d = Σ|ref_i - current_i| (Manhattan distance after L2 norm)
    Confidence: conf = 1 / (1 + d/n) where n is feature vector length
    This provides confidence values in range (0, 1] with higher values for better matches.
"""

from __future__ import annotations
import threading
import numpy as np
import sounddevice as sd

from .repository import ProfileRepository
from .features import FeatureExtractor
from .audio import RingBuffer
from .config import AudioConfig


def _normalize(v: np.ndarray) -> np.ndarray:
    """
    Apply L2 normalization to a feature vector.

    Normalizes the input vector to unit length using the L2 (Euclidean) norm.
    This normalization ensures that feature vectors of different magnitudes
    can be compared fairly, focusing on their shape rather than amplitude.

    Args:
        v (np.ndarray): Input feature vector to normalize. Should be a 1D array
                       of numeric values (typically spectral energies).

    Returns:
        np.ndarray: L2-normalized vector as float64. If the input vector has
                   near-zero norm (< 1e-12), returns the original vector
                   unchanged to avoid division by zero.

    Mathematical Details:
        For vector v, the L2 norm is: ||v||₂ = √(Σvᵢ²)
        Normalized vector: v_norm = v / ||v||₂

        Special case: If ||v||₂ < 1e-12, return v unchanged (zero vector handling)

    Examples:
        >>> import numpy as np
        >>> 
        >>> # Normalize a typical feature vector
        >>> features = np.array([1.0, 2.0, 3.0, 4.0])
        >>> normalized = _normalize(features)
        >>> print(f"Original: {features}")
        >>> print(f"Normalized: {normalized}")
        >>> print(f"Norm: {np.linalg.norm(normalized):.6f}")
        Original: [1. 2. 3. 4.]
        Normalized: [0.18257419 0.36514837 0.54772256 0.73029674]
        Norm: 1.000000
        >>> 
        >>> # Handle zero vector
        >>> zero_vec = np.array([0.0, 0.0, 0.0])
        >>> normalized_zero = _normalize(zero_vec)
        >>> print(f"Zero vector unchanged: {normalized_zero}")
        Zero vector unchanged: [0. 0. 0.]

    Note:
        The threshold 1e-12 prevents numerical instability when normalizing
        vectors that are effectively zero. This commonly occurs with silent
        audio segments or heavily attenuated signals.
    """
    v = v.astype(np.float64)
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n


class Recognizer:
    """
    Core voice command recognition engine for batch audio processing.

    This class implements the main recognition algorithm using distance-based
    matching against trained command profiles. It supports both WAV file
    processing and direct numpy array recognition, making it suitable for
    both offline analysis and integration with real-time audio pipelines.

    Recognition Process:
    1. Load all trained command profiles with matching feature configuration
    2. Extract features from input audio using the same parameters as training
    3. Apply L2 normalization to both current features and stored profiles
    4. Compute Manhattan distance between normalized feature vectors
    5. Select command with minimum distance as the best match
    6. Calculate confidence score based on inverse distance relationship

    Attributes:
        repo (ProfileRepository): Repository containing trained command profiles
        fe (FeatureExtractor): Feature extractor with matching configuration
        num_parts (int): Number of frequency bands for feature extraction

    Examples:
        >>> from pathlib import Path
        >>> from voicecmd.repository import ProfileRepository
        >>> 
        >>> # Initialize recognizer
        >>> repo = ProfileRepository(Path("commands.db"))
        >>> recognizer = Recognizer(repo, num_parts=100)
        >>> 
        >>> # Recognize from WAV file
        >>> command, confidence = recognizer.recognize_wav("test_audio.wav")
        >>> if command:
        ...     print(f"Recognized: {command} (confidence: {confidence:.3f})")
        ... else:
        ...     print("No recognition result")
        >>> 
        >>> # Recognize from audio array
        >>> import numpy as np
        >>> audio_data = np.random.randint(-32767, 32767, 16000, dtype=np.int16)
        >>> command, confidence = recognizer.recognize_array(audio_data)
        >>> 
        >>> repo.close()

    Algorithm Details:
        Distance Metric: d = Σ|ref_normalized_i - current_normalized_i|
        Confidence Score: conf = 1 / (1 + d/n) where n = feature vector length

        The Manhattan distance after L2 normalization provides good discrimination
        between different command patterns while being robust to amplitude variations.

    Performance Characteristics:
        - Time Complexity: O(k×n) where k = number of commands, n = feature length
        - Space Complexity: O(k×n) for storing profiles plus O(n) for current features
        - Typical recognition time: 1-10ms for 10 commands with 100 features

    Note:
        The feature extraction configuration (num_parts) must match between
        training and recognition phases. Mismatched configurations will result
        in no available profiles and recognition failure.
    """

    def __init__(self, repo: ProfileRepository, num_parts: int):
        """
        Initialize the recognizer with repository and feature configuration.

        Sets up the recognition engine with access to trained command profiles
        and consistent feature extraction configuration.

        Args:
            repo (ProfileRepository): Repository containing trained command profiles.
                                    Should be the same repository used during training.
            num_parts (int): Number of frequency bands for feature extraction.
                           Must match the configuration used during training.

        Examples:
            >>> repo = ProfileRepository(Path("voice_commands.db"))
            >>> 
            >>> # Standard configuration
            >>> recognizer = Recognizer(repo, num_parts=100)
            >>> 
            >>> # High-resolution configuration
            >>> detailed_recognizer = Recognizer(repo, num_parts=200)
        """
        self.repo = repo
        self.fe = FeatureExtractor(num_parts)
        self.num_parts = num_parts

    def recognize_wav(self, path: str) -> tuple[str | None, float]:
        """
        Recognize a voice command from a WAV audio file.

        Loads and processes a WAV file to identify the most likely voice command
        based on trained profiles. This method is suitable for offline processing
        and batch recognition tasks.

        Args:
            path (str): Path to the WAV audio file. The file should contain a
                       single voice command and be in a format supported by
                       scipy.io.wavfile (typically 16-bit or 24-bit PCM).

        Returns:
            tuple[str | None, float]: A tuple containing:
                - Command name (str) if recognition succeeded, None if no profiles exist
                - Confidence score (float) in range (0, 1], higher is better

        Raises:
            FileNotFoundError: If the specified audio file doesn't exist
            ValueError: If the file is not a valid WAV file or is corrupted

        Examples:
            >>> recognizer = Recognizer(repo, num_parts=100)
            >>> 
            >>> # Recognize a voice command
            >>> command, confidence = recognizer.recognize_wav("hello.wav")
            >>> if command:
            ...     print(f"Command: {command}, Confidence: {confidence:.3f}")
            ...     if confidence > 0.7:
            ...         print("High confidence recognition")
            ...     elif confidence > 0.4:
            ...         print("Medium confidence recognition")
            ...     else:
            ...         print("Low confidence recognition")
            ... else:
            ...     print("No trained profiles available")
            >>> 
            >>> # Batch processing
            >>> files = ["cmd1.wav", "cmd2.wav", "cmd3.wav"]
            >>> results = [recognizer.recognize_wav(f) for f in files]
            >>> for file, (cmd, conf) in zip(files, results):
            ...     print(f"{file}: {cmd} ({conf:.3f})")

        Note:
            - Confidence scores are relative to the available command set
            - Low confidence may indicate background noise or unknown commands
            - Consider setting confidence thresholds based on your application needs
        """
        profiles = self.repo.load_profiles(self.num_parts)
        if not profiles:
            return None, 0.0
        current = _normalize(self.fe.from_wav(path))
        names = list(profiles.keys())
        diffs = [float(np.sum(np.abs(_normalize(profiles[n]) - current)))
                 for n in names]
        idx = int(np.argmin(diffs))
        best = names[idx]
        d = diffs[idx]
        conf = 1.0 / (1.0 + d / (len(current) + 1e-9))
        return best, conf

    def recognize_array(self, x: np.ndarray) -> tuple[str | None, float]:
        """
        Recognize a voice command from a numpy audio array.

        Processes audio data that is already loaded in memory to identify the
        most likely voice command. This method is suitable for real-time
        processing and integration with audio streaming pipelines.

        Args:
            x (np.ndarray): Audio data array. Can be 1D (mono) or 2D (multi-channel).
                          Data type can be int16, int32, float32, or float64.
                          Should contain a single voice command for best results.

        Returns:
            tuple[str | None, float]: A tuple containing:
                - Command name (str) if recognition succeeded, None if no profiles exist
                - Confidence score (float) in range (0, 1], higher is better

        Examples:
            >>> import numpy as np
            >>> recognizer = Recognizer(repo, num_parts=100)
            >>> 
            >>> # Recognize from recorded audio
            >>> from voicecmd.audio import AudioRecorder
            >>> from voicecmd.config import AudioConfig
            >>> 
            >>> config = AudioConfig(rate=16000, duration=2.0)
            >>> recorder = AudioRecorder(config)
            >>> audio_data = recorder.record()
            >>> 
            >>> command, confidence = recognizer.recognize_array(audio_data)
            >>> print(f"Recognized: {command} (confidence: {confidence:.3f})")
            >>> 
            >>> # Process stereo audio
            >>> stereo_audio = np.random.randint(-1000, 1000, (8000, 2), dtype=np.int16)
            >>> command, confidence = recognizer.recognize_array(stereo_audio)
            >>> 
            >>> # Real-time processing integration
            >>> def process_audio_chunk(chunk):
            ...     command, conf = recognizer.recognize_array(chunk)
            ...     if command and conf > 0.6:
            ...         print(f"Command detected: {command}")
            ...         return command
            ...     return None

        Note:
            - Multi-channel audio is automatically converted to mono
            - Input data is converted to appropriate format internally
            - For real-time use, consider using LiveRecognizer for better performance
        """
        profiles = self.repo.load_profiles(self.num_parts)
        if not profiles:
            return None, 0.0
        current = _normalize(self.fe.from_array(x))
        names = list(profiles.keys())
        diffs = [float(np.sum(np.abs(_normalize(profiles[n]) - current)))
                 for n in names]
        idx = int(np.argmin(diffs))
        best = names[idx]
        d = diffs[idx]
        conf = 1.0 / (1.0 + d / (len(current) + 1e-9))
        return best, conf


class LiveRecognizer:
    """
    Real-time voice command recognition with adaptive noise thresholding.

    This class provides continuous voice command recognition from live audio
    input using a multi-threaded architecture with automatic noise calibration
    and voice activity detection. It uses a sliding window approach with
    configurable hop intervals for responsive recognition while managing
    computational efficiency.

    Key Features:
    - Automatic ambient noise calibration during startup
    - Adaptive RMS-based voice activity detection
    - Sliding window processing with configurable overlap
    - Multi-threaded architecture for real-time performance
    - Callback system for event handling (RMS, predictions, status)
    - Robust audio streaming with error handling

    Architecture:
    1. Audio Input Thread: Captures audio via sounddevice.InputStream
    2. Processing Thread: Handles noise calibration and recognition
    3. Ring Buffer: Provides efficient sliding window management
    4. Callback System: Delivers results to application layer

    Attributes:
        recognizer (Recognizer): Core recognition engine for pattern matching
        audio (AudioConfig): Audio configuration for streaming parameters
        device_index (int | None): Specific audio device or None for default
        window_secs (float): Duration of analysis window in seconds
        hop_secs (float): Interval between recognition attempts in seconds
        calib_secs (float): Duration of noise calibration period in seconds
        threshold_factor (float): Multiplier for noise threshold calculation

    Examples:
        >>> from pathlib import Path
        >>> from voicecmd.repository import ProfileRepository
        >>> from voicecmd.config import AudioConfig
        >>> 
        >>> # Set up recognition system
        >>> repo = ProfileRepository(Path("commands.db"))
        >>> recognizer = Recognizer(repo, num_parts=100)
        >>> audio_config = AudioConfig(rate=16000, channels=1, chunk=512)
        >>> 
        >>> # Initialize live recognizer
        >>> live = LiveRecognizer(
        ...     recognizer=recognizer,
        ...     audio=audio_config,
        ...     window_secs=2.0,
        ...     hop_secs=0.3,
        ...     threshold_factor=3.0
        ... )
        >>> 
        >>> # Set up callbacks
        >>> def on_command(name, confidence):
        ...     if confidence > 0.7:
        ...         print(f"Command: {name} (confidence: {confidence:.3f})")
        >>> 
        >>> def on_status(message):
        ...     print(f"Status: {message}")
        >>> 
        >>> live.on_pred = on_command
        >>> live.on_status = on_status
        >>> 
        >>> # Start recognition
        >>> live.start()
        >>> # ... recognition runs in background ...
        >>> live.stop()
        >>> repo.close()

    Calibration Process:
        1. Captures ambient noise for calib_secs duration
        2. Computes baseline RMS level from ambient samples
        3. Sets threshold = max(200, baseline × threshold_factor)
        4. Uses threshold for voice activity detection during recognition

    Recognition Process:
        1. Continuously captures audio in sliding windows
        2. Computes RMS level for voice activity detection
        3. When RMS exceeds threshold, performs command recognition
        4. Delivers results via callback system with confidence scores

    Performance Considerations:
        - Window duration affects latency vs accuracy trade-off
        - Hop interval controls recognition frequency and CPU usage
        - Threshold factor affects sensitivity to background noise
        - Buffer size should accommodate window duration with margin
    """

    def __init__(
        self,
        recognizer: Recognizer,
        audio: AudioConfig,
        device_index: int | None = None,
        window_secs: float = 2.0,
        hop_secs: float = 0.3,
        calib_secs: float = 0.8,
        threshold_factor: float = 3.0,
    ):
        """
        Initialize the live recognizer with configuration parameters.

        Sets up the real-time recognition system with specified timing
        parameters, audio configuration, and noise adaptation settings.

        Args:
            recognizer (Recognizer): Core recognition engine with trained profiles
            audio (AudioConfig): Audio streaming configuration (rate, channels, chunk size)
            device_index (int | None): Specific audio input device index, or None for default
            window_secs (float): Analysis window duration in seconds (typically 1-3 seconds)
            hop_secs (float): Recognition interval in seconds (typically 0.1-0.5 seconds)
            calib_secs (float): Noise calibration duration in seconds (typically 0.5-2 seconds)
            threshold_factor (float): Noise threshold multiplier (typically 2-5)

        Examples:
            >>> # Responsive configuration for quick commands
            >>> live_fast = LiveRecognizer(
            ...     recognizer, audio_config,
            ...     window_secs=1.5, hop_secs=0.2, threshold_factor=2.5
            ... )
            >>> 
            >>> # Robust configuration for noisy environments
            >>> live_robust = LiveRecognizer(
            ...     recognizer, audio_config,
            ...     window_secs=3.0, hop_secs=0.5, threshold_factor=4.0
            ... )
            >>> 
            >>> # Specific device configuration
            >>> live_device = LiveRecognizer(
            ...     recognizer, audio_config,
            ...     device_index=1  # Use specific microphone
            ... )

        Note:
            - Larger windows provide better accuracy but higher latency
            - Smaller hop intervals increase responsiveness but use more CPU
            - Higher threshold factors reduce false positives in noisy environments
        """
        self.recognizer = recognizer
        self.audio = audio
        self.device_index = device_index
        self.window_secs = float(window_secs)
        self.hop_secs = float(hop_secs)
        self.calib_secs = float(calib_secs)
        self.threshold_factor = float(threshold_factor)

        # Internal state management
        self._stream: sd.InputStream | None = None
        self._ring = RingBuffer(
            int(self.audio.rate * max(0.6, self.window_secs)))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # Callback functions for event handling
        # callable(rms_value, threshold) - RMS level updates
        self.on_rms = None
        self.on_pred = None  # callable(name, confidence) - Recognition results
        self.on_status = None  # callable(text) - Status messages

    def _cb(self, indata, frames, time_info, status):
        """
        Audio input callback for sounddevice.InputStream.

        This method is called automatically by the audio streaming system
        whenever new audio data is available. It processes the incoming
        audio data and adds it to the ring buffer for analysis.

        Args:
            indata: Audio input data as float32 in range [-1, 1]
            frames: Number of frames in this callback (typically chunk size)
            time_info: Timing information (not used)
            status: Stream status flags (not used)

        Note:
            This is an internal callback method called by sounddevice.
            It performs minimal processing to avoid blocking the audio thread.
        """
        # indata: float32 [-1,1]
        x = indata[:, 0] if indata.ndim > 1 else indata
        x = np.clip(x, -1.0, 1.0)
        x = (x * 32767.0).astype(np.int16)
        self._ring.push(x)

    def _emit_status(self, text: str):
        """
        Emit a status message via callback if configured.

        Args:
            text (str): Status message to emit
        """
        if self.on_status:
            self.on_status(text)

    def _emit_rms(self, rms: float, thr: float):
        """
        Emit RMS level information via callback if configured.

        Args:
            rms (float): Current RMS level
            thr (float): Current threshold level
        """
        if self.on_rms:
            self.on_rms(rms, thr)

    def _emit_pred(self, name: str, conf: float):
        """
        Emit recognition prediction via callback if configured.

        Args:
            name (str): Recognized command name (or "-" for no detection)
            conf (float): Confidence score
        """
        if self.on_pred:
            self.on_pred(name, conf)

    def start(self):
        """
        Start real-time voice command recognition.

        Initializes the audio streaming system and begins the recognition
        pipeline in a separate thread. The process includes:
        1. Opening audio input stream
        2. Calibrating ambient noise threshold
        3. Beginning continuous recognition with sliding windows

        The method returns immediately while recognition continues in the
        background. Use the callback system to receive recognition results
        and status updates.

        Examples:
            >>> # Set up callbacks first
            >>> def handle_command(name, confidence):
            ...     if confidence > 0.7:
            ...         print(f"Executing command: {name}")
            >>> 
            >>> def handle_status(message):
            ...     print(f"System: {message}")
            >>> 
            >>> live.on_pred = handle_command
            >>> live.on_status = handle_status
            >>> 
            >>> # Start recognition
            >>> live.start()
            >>> print("Recognition started, speak commands...")
            >>> 
            >>> # Recognition runs until stop() is called

        Process Flow:
            1. **Stream Setup**: Opens audio input stream with configured parameters
            2. **Calibration Phase**: Measures ambient noise for threshold calculation
            3. **Recognition Phase**: Continuous sliding window analysis

        Thread Safety:
            This method is thread-safe and can be called multiple times.
            If recognition is already running, subsequent calls are ignored.

        Note:
            - Ensure callbacks are set before calling start()
            - The method returns immediately; recognition runs asynchronously
            - Call stop() to terminate recognition and clean up resources
        """
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        # Open stream
        self._stream = sd.InputStream(
            samplerate=self.audio.rate,
            channels=self.audio.channels,
            dtype="float32",
            device=self.device_index,
            blocksize=self.audio.chunk,
            callback=self._cb,
        )
        self._stream.start()
        self._emit_status("Calibrating noise…")

        def worker():
            # Calibration phase
            time_slices = int(
                max(1, self.calib_secs / (self.audio.chunk / self.audio.rate)))
            xs = []
            for _ in range(time_slices):
                if self._stop.is_set():
                    break
                sd.sleep(int(1000 * (self.audio.chunk / self.audio.rate)))
                xs.append(self._ring.read_latest(self.audio.chunk))
            if xs:
                arr = np.concatenate(xs).astype(np.float64)
                baseline = float(np.sqrt((arr * arr).mean() + 1e-12))
            else:
                baseline = 300.0
            thr = max(200.0, baseline * self.threshold_factor)
            self._emit_status(f"RMS threshold ≈ {thr:.0f}")

            # Recognition phase: Window + hop processing
            win_n = int(self.window_secs * self.audio.rate)
            hop_n = max(1, int(self.hop_secs * self.audio.rate))
            last = 0
            while not self._stop.is_set():
                sd.sleep(int(1000 * (self.audio.chunk / self.audio.rate)))
                buf = self._ring.read_latest(win_n).astype(np.float64)
                if len(buf) < win_n:
                    continue
                rms = float(np.sqrt((buf * buf).mean() + 1e-12))
                self._emit_rms(rms, thr)
                last += self.audio.chunk
                if last >= hop_n:
                    last = 0
                    if rms >= thr:
                        name, conf = self.recognizer.recognize_array(
                            buf.astype(np.int16))
                        if name is None:
                            self._emit_status("No profiles.")
                            self._emit_pred("-", 0.0)
                        else:
                            self._emit_pred(name, conf)
                    else:
                        self._emit_pred("-", 0.0)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def stop(self):
        """
        Stop real-time voice command recognition and clean up resources.

        Gracefully terminates the recognition system by:
        1. Signaling the processing thread to stop
        2. Waiting for thread completion (with timeout)
        3. Closing and cleaning up the audio stream

        This method should always be called when finished with recognition
        to ensure proper resource cleanup and prevent audio device lock-up.

        Examples:
            >>> # Start recognition
            >>> live.start()
            >>> 
            >>> # ... do other work while recognition runs ...
            >>> 
            >>> # Stop recognition when done
            >>> live.stop()
            >>> print("Recognition stopped")
            >>> 
            >>> # Or use in try/finally pattern
            >>> live.start()
            >>> try:
            ...     # ... main application logic ...
            ... finally:
            ...     live.stop()

        Cleanup Process:
            1. **Signal Stop**: Sets internal stop flag for thread coordination
            2. **Thread Join**: Waits up to 1 second for processing thread to finish
            3. **Stream Cleanup**: Stops and closes audio input stream safely

        Thread Safety:
            This method is thread-safe and can be called multiple times.
            If recognition is not running, the call is safely ignored.

        Note:
            - Always call stop() to prevent resource leaks
            - The method blocks briefly (max 1 second) for clean shutdown
            - After calling stop(), the recognizer can be restarted with start()
        """
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
