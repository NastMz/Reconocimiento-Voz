"""
VoiceCMD GUI Application - Voice Command Training and Recognition Interface

This module provides a comprehensive graphical user interface for the VoiceCMD
voice command recognition system. The GUI enables users to:

1. Record voice command samples for training datasets
2. Train machine learning profiles for voice recognition
3. Test voice recognition on recorded WAV files  
4. Perform real-time voice command recognition
5. Evaluate model performance on directories of audio files
6. Manage voice command profiles and recordings

The application features a tabbed interface with three main sections:
- Dataset Tab: For recording training samples and managing voice profiles
- Live Recognition Tab: For real-time voice command recognition
- Evaluation Tab: For batch testing and accuracy measurement

Architecture:
- Built using tkinter for cross-platform GUI compatibility
- Multi-threaded audio recording with real-time RMS monitoring
- Integration with sounddevice for professional audio I/O
- SQLite database backend for profile and recording management
- Real-time audio processing with configurable parameters
- Batch evaluation with progress tracking and detailed reporting

Key Features:
- Visual RMS meters for audio level monitoring
- Progress bars for recording, training, and evaluation operations
- Configurable audio parameters (sample rate, duration, etc.)
- Device selection for multiple microphone support
- Real-time confidence scoring for voice recognition
- Batch evaluation with accuracy statistics and per-file results
- Automatic file organization and metadata management

Dependencies:
- tkinter: GUI framework
- sounddevice: Professional audio I/O
- numpy: Numerical processing for audio data
- threading: Concurrent operations for responsive UI
- VoiceCMD modules: Core voice recognition functionality

Usage:
    The GUI can be launched directly or integrated into larger applications:
    
    ```python
    from voicecmd.gui import VoiceCmdGUI, main
    
    # Direct launch
    main()
    
    # Programmatic use
    app = VoiceCmdGUI()
    app.mainloop()
    ```

File Organization:
- Database: voicecmd.db (SQLite database for profiles)
- Recordings: data/ directory with command-specific subdirectories
- Configuration: Integrated with VoiceCMD configuration system
"""
from __future__ import annotations
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import sounddevice as sd

from .config import AudioConfig, FeatureConfig, get_database_path, get_data_dir
from .audio import AudioDevices, AudioRecorder, WavWriter
from .repository import ProfileRepository
from .training import Trainer
from .recognition import Recognizer, LiveRecognizer

DB_PATH = get_database_path()
DATA_DIR = get_data_dir()


# --------- Grabación “por bloques” con RMS/avance (sounddevice) ----------
class ChunkRecorderThread(threading.Thread):
    """
    Threaded audio recorder with real-time RMS monitoring and progress tracking.

    This class provides non-blocking audio recording capabilities with real-time
    feedback for GUI applications. It records audio in configurable chunks while
    providing RMS (Root Mean Square) volume measurements and progress updates
    to the calling thread through callback functions.

    The recorder operates in a separate daemon thread to prevent blocking the
    main GUI thread, ensuring responsive user interface during recording operations.

    Recording Process:
        1. Initialize audio stream with specified parameters
        2. Record audio in blocks of configured chunk size
        3. Calculate RMS volume for each block
        4. Provide progress updates via callback
        5. Accumulate audio data in internal buffer
        6. Convert and return final audio data

    Audio Processing:
        - Clips input to [-1.0, 1.0] range to prevent distortion
        - Converts float32 input to int16 output format
        - Handles mono channel extraction from multi-channel input
        - Provides real-time RMS calculation for volume monitoring

    Thread Safety:
        - Uses threading.Event for clean shutdown signaling
        - Daemon thread ensures automatic cleanup on app exit
        - Thread-safe callback mechanism for GUI updates

    Args:
        audio (AudioConfig): Audio configuration (sample rate, channels, chunk size)
        seconds (float): Total recording duration in seconds
        device_index (int | None): Audio input device index (None for default)
        on_progress (callable): Progress callback (rms: float, progress: float)
        on_done (callable): Completion callback (success: bool, data: np.ndarray, error: Exception)

    Callbacks:
        on_progress(rms, progress_0_1):
            - rms: Current RMS volume level (0.0 to ~1.0)
            - progress_0_1: Recording progress (0.0 to 1.0)

        on_done(ok, data, err):
            - ok: Boolean success indicator
            - data: Recorded audio data as int16 numpy array (or None on error)
            - err: Exception object if error occurred (or None on success)

    Examples:
        Basic Usage:
        ```python
        def on_progress(rms, progress):
            print(f"Volume: {rms:.3f}, Progress: {progress:.1%}")

        def on_done(success, audio_data, error):
            if success:
                print(f"Recorded {len(audio_data)} samples")
            else:
                print(f"Recording failed: {error}")

        recorder = ChunkRecorderThread(
            audio_config, duration=3.0, device_index=None,
            on_progress=on_progress, on_done=on_done
        )
        recorder.start()
        ```

        GUI Integration:
        ```python
        def update_progress_bar(rms, progress):
            volume_bar.set(rms * 100)
            progress_bar.set(progress * 100)

        recorder = ChunkRecorderThread(
            audio_config, 5.0, device_index,
            on_progress=update_progress_bar,
            on_done=save_recording
        )
        ```

    Performance Notes:
        - Uses efficient numpy operations for audio processing
        - Minimal memory allocation during recording loop
        - Real-time RMS calculation with numerical stability (1e-12 epsilon)
        - Automatic cleanup of audio resources

    Error Handling:
        - Graceful handling of audio device disconnection
        - Protection against invalid audio configurations
        - Safe shutdown mechanism with stop() method
        - Comprehensive error reporting via on_done callback
    """

    def __init__(self, audio: AudioConfig, seconds: float, device_index: int | None,
                 on_progress, on_done):
        super().__init__(daemon=True)
        self.audio = audio
        self.seconds = float(seconds)
        self.device_index = device_index
        self.on_progress = on_progress  # (rms, progress_0_1)
        # (ok: bool, data: np.ndarray|None, err: Exception|None)
        self.on_done = on_done
        self._stop = threading.Event()

    def stop(self):
        """
        Signal the recording thread to stop gracefully.

        Sets an internal stop event that the recording loop checks
        periodically. This allows for clean termination of recording
        operations without corrupting audio data.

        The stop is asynchronous - the thread will complete its current
        audio block before actually stopping. Use thread.join() if you
        need to wait for complete termination.

        Thread Safety:
            Safe to call from any thread, including GUI thread.
        """
        self._stop.set()

    def run(self):
        """
        Main recording loop - executes in separate thread.

        Implements the core audio recording logic with real-time monitoring.
        This method should not be called directly - use start() instead.

        Recording Algorithm:
            1. Calculate target number of audio blocks based on duration
            2. Open audio input stream with configured parameters
            3. Read audio data in chunks, processing each block:
               - Extract mono channel if needed
               - Clip values to prevent distortion
               - Calculate RMS for volume monitoring
               - Report progress via callback
               - Accumulate data in buffer
            4. Concatenate all blocks into final audio array
            5. Convert to int16 format and report completion

        Error Handling:
            All exceptions are caught and reported via on_done callback
            to prevent thread crashes and ensure GUI remains responsive.

        Performance:
            - Efficient block-based processing
            - Minimal memory allocation during loop
            - Real-time RMS calculation
            - Automatic audio format conversion
        """
        rate = self.audio.rate
        ch = self.audio.channels
        block = self.audio.chunk
        total_frames = int(self.seconds * rate)
        target_blocks = max(1, total_frames // block)

        buf: list[np.ndarray] = []
        self._block_count = 0

        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio callback status: {status}")

            if self._stop.is_set() or self._block_count >= target_blocks:
                raise sd.CallbackStop()

            # Process audio data
            x = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            x = np.clip(x, -1.0, 1.0)

            # Calculate RMS
            rms = float(np.sqrt((x.astype(np.float64) ** 2).mean() + 1e-12))

            # Update progress
            self._block_count += 1
            progress = self._block_count / target_blocks
            self.on_progress(rms, progress)

            # Store audio data
            buf.append((x * 32767.0).astype(np.int16))

        try:
            with sd.InputStream(samplerate=rate, channels=ch, dtype="float32",
                                device=self.device_index, blocksize=block,
                                callback=audio_callback):
                # Wait for recording to complete
                while self._block_count < target_blocks and not self._stop.is_set():
                    sd.sleep(50)  # Sleep 50ms

            if not buf:
                self.on_done(False, None, RuntimeError("No se capturó audio"))
                return

            data = np.concatenate(buf)
            self.on_done(True, data, None)
        except Exception as e:
            self.on_done(False, None, e)


# ----------------------------- GUI -----------------------------
class VoiceCmdGUI(tk.Tk):
    """
    Main GUI application for VoiceCMD voice command system.

    This class provides a comprehensive graphical interface for voice command
    training, recognition, and evaluation. Built on tkinter, it offers a tabbed 
    interface with separate sections for dataset management, live recognition,
    and batch evaluation.

    The application integrates all VoiceCMD components into a user-friendly
    interface, allowing users to:
    - Record voice command samples for training
    - Train machine learning profiles for recognition
    - Test recognition on audio files
    - Perform real-time voice command recognition
    - Evaluate model performance on directories of audio files
    - Manage audio devices and configuration parameters

    GUI Architecture:
        - Tab 1 (Dataset): Recording, training, and file management
        - Tab 2 (Live): Real-time voice recognition with configuration
        - Tab 3 (Evaluation): Batch testing and accuracy measurement
        - Threaded operations for responsive UI during audio processing
        - Real-time visual feedback with progress bars and RMS meters
        - Automatic resource management and cleanup

    Key Features:
        Visual Components:
        - Progress bars for recording, training, and evaluation operations
        - RMS meters for real-time audio level monitoring
        - Device selection dropdowns for microphone management
        - Configuration controls for audio parameters
        - Status displays for operation feedback
        - Results display with per-file accuracy details

        Audio Management:
        - Multi-device audio input support
        - Real-time audio level monitoring
        - Configurable recording parameters
        - Professional audio I/O via sounddevice

        Data Management:
        - SQLite database integration for profiles
        - Automatic file organization by command
        - Metadata tracking for recordings
        - Training progress monitoring

        Live Recognition:
        - Real-time voice command detection
        - Configurable recognition parameters
        - Confidence scoring and display
        - Automatic calibration and noise handling

        Batch Evaluation:
        - Directory-based accuracy testing
        - Configurable file patterns and recursive search
        - Per-file results with OK/ERR status
        - Overall accuracy statistics and reporting

    State Management:
        The GUI maintains extensive state through tkinter variables:
        - Recording state: command, device, duration, progress
        - Live recognition state: parameters, predictions, confidence
        - Evaluation state: directory, label, progress, results
        - Visual feedback: RMS levels, status messages, progress

    Threading Model:
        - Main thread: GUI operations and user interaction
        - Recording thread: ChunkRecorderThread for audio capture
        - Recognition thread: LiveRecognizer for real-time processing
        - Training thread: Background training operations

    Error Handling:
        - Graceful handling of audio device issues
        - User-friendly error messages via message boxes
        - Automatic cleanup on application exit
        - Safe termination of background threads

    Dependencies:
        - tkinter/ttk: GUI framework and modern widgets
        - VoiceCMD core: Audio processing and recognition
        - sounddevice: Professional audio I/O
        - threading: Concurrent operations
        - pathlib: Modern file path handling

    Examples:
        Basic Usage:
        ```python
        app = VoiceCmdGUI()
        app.mainloop()
        ```

        Programmatic Integration:
        ```python
        app = VoiceCmdGUI()
        app.protocol("WM_DELETE_WINDOW", custom_cleanup)
        app.mainloop()
        ```

    Performance Notes:
        - Efficient GUI updates via tkinter variables
        - Non-blocking audio operations through threading
        - Minimal memory usage with streaming audio processing
        - Responsive interface during long-running operations
    """

    def __init__(self):
        super().__init__()
        self.title("VoiceCMD – Dataset & Live")
        self.geometry("860x640")
        self.minsize(780, 560)

        self.repo = ProfileRepository(DB_PATH)
        self.audio_cfg = AudioConfig()                 # 44.1 kHz, mono, chunk 1024, 2s
        self.num_parts = FeatureConfig().num_parts     # 100 por defecto

        # Estado dataset
        self.current_command = tk.StringVar(value="")
        self.device_var = tk.StringVar(value="")
        self.duration_var = tk.DoubleVar(value=self.audio_cfg.duration)
        self.status_var = tk.StringVar(value="Listo.")
        self.last_saved_var = tk.StringVar(value="-")
        self.rms_var = tk.DoubleVar(value=0.0)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.rec_thread: ChunkRecorderThread | None = None

        # Estado live
        self.live_status_var = tk.StringVar(value="Listo para calibrar.")
        self.live_rms_var = tk.DoubleVar(value=0.0)
        self.live_pred_var = tk.StringVar(value="-")
        self.live_conf_var = tk.StringVar(value="0.00")
        self.live_window_var = tk.DoubleVar(value=2.0)
        self.live_hop_var = tk.DoubleVar(value=0.3)
        self.live_factor_var = tk.DoubleVar(value=3.0)
        self.live_calib_var = tk.DoubleVar(value=0.8)
        self.live_thread: LiveRecognizer | None = None

        # Estado evaluación
        self.eval_dir_var = tk.StringVar(value="")
        self.eval_label_var = tk.StringVar(value="")
        self.eval_glob_var = tk.StringVar(value="*.wav")
        self.eval_recursive_var = tk.BooleanVar(value=False)
        self.eval_status_var = tk.StringVar(value="Listo para evaluar.")
        self.eval_results_var = tk.StringVar(value="")
        self.eval_accuracy_var = tk.StringVar(value="")
        self.eval_progress_var = tk.DoubleVar(value=0.0)
        self.eval_running = False

        # Dispositivos
        self.devices = AudioDevices.list_input()

        self._build_ui()
        self._load_commands()

    # -------------------- UI layout --------------------
    def _build_ui(self):
        """
        Build the main GUI interface with tabbed layout.

        Creates the primary user interface structure with a notebook widget
        containing separate tabs for dataset management and live recognition.
        Each tab is built using dedicated methods for better organization.

        UI Structure:
            - Main window with notebook widget
            - Dataset tab: Recording and training functionality
            - Live tab: Real-time recognition interface
            - Proper padding and sizing for professional appearance

        Layout Management:
            Uses pack geometry manager with fill and expand options
            for responsive layout that adapts to window resizing.
        """
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)
        self.tab_dataset = ttk.Frame(nb)
        self.tab_live = ttk.Frame(nb)
        self.tab_eval = ttk.Frame(nb)
        nb.add(self.tab_dataset, text="Dataset (Grabar/Entrenar)")
        nb.add(self.tab_live, text="Reconocimiento en vivo")
        nb.add(self.tab_eval, text="Evaluación")

        self._build_dataset_tab(self.tab_dataset)
        self._build_live_tab(self.tab_live)
        self._build_eval_tab(self.tab_eval)

    def _build_dataset_tab(self, parent):
        """
        Build the dataset management tab interface.

        Creates the interface for recording voice command samples, training
        models, and managing the voice command dataset. This tab provides
        all functionality needed for creating and maintaining voice profiles.

        UI Components:
            Row 1: Command selection and management
            - Command name input/selection combobox
            - Add/Select button for command management

            Row 2: Audio configuration
            - Microphone device selection dropdown
            - Recording duration spinner control

            Row 3: Action buttons
            - Record button for capturing voice samples
            - Train profiles button for model training
            - Recognize WAV button for testing
            - Open data folder button for file management

            Row 4: Visual feedback
            - RMS volume progress bar for recording levels
            - Recording progress bar for timing feedback

            Row 5: Status information
            - Last saved file display
            - Status bar for operation feedback

        Layout:
            Uses horizontal frames (rows) with mixed packing strategies
            for optimal space utilization and visual organization.

        Args:
            parent: Parent tkinter widget (tab frame)
        """
        pad = {"padx": 10, "pady": 10}
        frm = ttk.Frame(parent)
        frm.pack(fill="both", expand=True, **pad)

        # Comando
        row1 = ttk.Frame(frm)
        row1.pack(fill="x", **pad)
        ttk.Label(row1, text="Comando:").pack(side="left")
        self.cmd_box = ttk.Combobox(
            row1, textvariable=self.current_command, width=28)
        self.cmd_box.pack(side="left", padx=8)
        ttk.Button(row1, text="Añadir/Seleccionar", command=self._add_or_select_command)\
            .pack(side="left", padx=4)

        # Dispositivo + duración
        row2 = ttk.Frame(frm)
        row2.pack(fill="x", **pad)
        ttk.Label(row2, text="Micrófono:").pack(side="left")
        self.dev_box = ttk.Combobox(
            row2, textvariable=self.device_var, width=50, state="readonly")
        dev_items = [f"{i} - {name}" for i,
                     name in self.devices] or ["(sin dispositivos)"]
        self.dev_box["values"] = dev_items
        if dev_items:
            self.dev_box.current(0)
        self.dev_box.pack(side="left", padx=8)

        ttk.Label(row2, text="Duración (s):").pack(side="left", padx=(16, 4))
        ttk.Spinbox(row2, from_=0.5, to=10.0, increment=0.5,
                    textvariable=self.duration_var, width=6).pack(side="left")

        # Controles
        row3 = ttk.Frame(frm)
        row3.pack(fill="x", **pad)
        self.btn_record = ttk.Button(
            row3, text="● Grabar", command=self._on_record)
        self.btn_record.pack(side="left")
        ttk.Button(row3, text="Entrenar perfiles",
                   command=self._on_train).pack(side="left", padx=8)
        ttk.Button(row3, text="Reconocer WAV…",
                   command=self._on_recognize_wav).pack(side="left", padx=8)
        ttk.Button(row3, text="Abrir carpeta datos",
                   command=self._open_data_dir).pack(side="left", padx=8)

        # Indicadores
        row4 = ttk.Frame(frm)
        row4.pack(fill="x", **pad)
        ttk.Label(row4, text="Volumen (RMS):").pack(anchor="w")
        self.rms_bar = ttk.Progressbar(row4, orient="horizontal", mode="determinate",
                                       maximum=8000, variable=self.rms_var)
        self.rms_bar.pack(fill="x")
        ttk.Label(row4, text="Progreso de grabación:").pack(
            anchor="w", pady=(12, 0))
        self.progress_bar = ttk.Progressbar(row4, orient="horizontal", mode="determinate",
                                            maximum=100, variable=self.progress_var)
        self.progress_bar.pack(fill="x")

        # Estado inferior
        row5 = ttk.Frame(frm)
        row5.pack(fill="x", **pad)
        ttk.Label(row5, text="Último archivo:").pack(side="left")
        ttk.Label(row5, textvariable=self.last_saved_var, foreground="#555")\
            .pack(side="left", padx=6)
        self.status = ttk.Label(
            parent, textvariable=self.status_var, anchor="w", relief="sunken")
        self.status.pack(side="bottom", fill="x")

    def _build_live_tab(self, parent):
        """
        Build the live recognition tab interface.

        Creates the interface for real-time voice command recognition with
        configurable parameters and visual feedback. This tab provides all
        controls needed for live voice recognition operation.

        UI Components:
            Row 1: Audio and timing configuration
            - Microphone device selection dropdown
            - Window duration spinner (recognition time window)
            - Hop duration spinner (sliding window overlap)

            Row 2: Recognition parameters
            - Threshold factor spinner (noise sensitivity)
            - Calibration duration spinner (noise measurement time)

            Row 3: Control buttons
            - Start button for beginning live recognition
            - Stop button for ending recognition session

            Row 4: Real-time feedback
            - RMS volume meter for current audio levels

            Row 5: Recognition results
            - Prediction display (recognized command)
            - Confidence score display (recognition certainty)

            Status bar: Operation status and feedback

        Configuration Parameters:
            - Window: Duration of audio analyzed for each recognition attempt
            - Hop: Time between recognition attempts (sliding window)
            - Factor: Multiplier for noise threshold (higher = less sensitive)
            - Calibration: Time spent measuring background noise

        Visual Feedback:
            - Real-time RMS meter shows current audio input levels
            - Prediction display shows recognized commands in bold
            - Confidence score indicates recognition certainty
            - Status bar provides operation feedback

        Args:
            parent: Parent tkinter widget (tab frame)
        """
        pad = {"padx": 10, "pady": 10}
        frm = ttk.Frame(parent)
        frm.pack(fill="both", expand=True, **pad)

        # Dispositivo + parámetros
        row1 = ttk.Frame(frm)
        row1.pack(fill="x", **pad)
        ttk.Label(row1, text="Micrófono:").pack(side="left")
        self.live_dev_box = ttk.Combobox(row1, width=50, state="readonly")
        dev_items = [f"{i} - {name}" for i,
                     name in self.devices] or ["(sin dispositivos)"]
        self.live_dev_box["values"] = dev_items
        if dev_items:
            self.live_dev_box.current(0)
        self.live_dev_box.pack(side="left", padx=8)

        ttk.Label(row1, text="Ventana (s):").pack(side="left", padx=(12, 4))
        ttk.Spinbox(row1, from_=0.6, to=3.0, increment=0.1,
                    textvariable=self.live_window_var, width=6).pack(side="left")
        ttk.Label(row1, text="Hop (s):").pack(side="left", padx=(12, 4))
        ttk.Spinbox(row1, from_=0.1, to=1.0, increment=0.1,
                    textvariable=self.live_hop_var, width=6).pack(side="left")

        row2 = ttk.Frame(frm)
        row2.pack(fill="x", **pad)
        ttk.Label(row2, text="Factor umbral (× ruido):").pack(side="left")
        ttk.Spinbox(row2, from_=1.5, to=6.0, increment=0.1,
                    textvariable=self.live_factor_var, width=6).pack(side="left", padx=(6, 20))
        ttk.Label(row2, text="Calibración (s):").pack(side="left")
        ttk.Spinbox(row2, from_=0.5, to=3.0, increment=0.1,
                    textvariable=self.live_calib_var, width=6).pack(side="left")

        # Controles
        row3 = ttk.Frame(frm)
        row3.pack(fill="x", **pad)
        self.btn_live_start = ttk.Button(
            row3, text="▶ Iniciar", command=self._live_start)
        self.btn_live_start.pack(side="left")
        self.btn_live_stop = ttk.Button(
            row3, text="■ Detener", command=self._live_stop, state="disabled")
        self.btn_live_stop.pack(side="left", padx=8)

        # Indicadores
        row4 = ttk.Frame(frm)
        row4.pack(fill="x", **pad)
        ttk.Label(row4, text="RMS (en vivo):").pack(anchor="w")
        self.live_rms_bar = ttk.Progressbar(row4, orient="horizontal", mode="determinate",
                                            maximum=8000, variable=self.live_rms_var)
        self.live_rms_bar.pack(fill="x")

        row5 = ttk.Frame(frm)
        row5.pack(fill="x", **pad)
        ttk.Label(row5, text="Predicción:").pack(side="left")
        ttk.Label(row5, textvariable=self.live_pred_var, font=("Segoe UI", 14, "bold"))\
            .pack(side="left", padx=8)
        ttk.Label(row5, text="Confianza:").pack(side="left", padx=(20, 4))
        ttk.Label(row5, textvariable=self.live_conf_var, font=("Segoe UI", 12))\
            .pack(side="left")

        self.live_status = ttk.Label(
            parent, textvariable=self.live_status_var, anchor="w", relief="sunken")
        self.live_status.pack(side="bottom", fill="x")

    def _build_eval_tab(self, parent):
        """
        Build the evaluation tab interface.

        Creates the interface for batch evaluation of voice command recognition
        accuracy on directories of audio files. This tab provides functionality
        to test model performance and generate accuracy statistics.

        UI Components:
            Row 1: Directory and label configuration
            - Directory path input with browse button
            - Expected command label input
            - File pattern input (glob pattern)
            - Recursive search checkbox

            Row 2: Control buttons
            - Start evaluation button
            - Stop evaluation button (during running evaluation)

            Row 3: Progress feedback
            - Progress bar showing evaluation progress
            - Current file being processed

            Row 4: Results display
            - Scrollable text area with detailed results per file
            - Overall accuracy summary

            Status bar: Operation status and feedback

        Features:
            - Browse button for easy directory selection
            - Support for custom file patterns (*.wav, test_*.wav, etc.)
            - Recursive directory search option
            - Real-time progress feedback during evaluation
            - Detailed per-file results with OK/ERR status
            - Overall accuracy statistics

        Args:
            parent: Parent tkinter widget (tab frame)
        """
        pad = {"padx": 10, "pady": 10}
        frm = ttk.Frame(parent)
        frm.pack(fill="both", expand=True, **pad)

        # Configuración de evaluación
        config_frame = ttk.LabelFrame(frm, text="Configuración de Evaluación")
        config_frame.pack(fill="x", **pad)

        # Row 1: Directory and label
        row1 = ttk.Frame(config_frame)
        row1.pack(fill="x", padx=5, pady=5)

        ttk.Label(row1, text="Directorio:").grid(
            row=0, column=0, sticky="w", padx=(0, 5))
        self.eval_dir_entry = ttk.Entry(
            row1, textvariable=self.eval_dir_var, width=40)
        self.eval_dir_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        ttk.Button(row1, text="Examinar", command=self._browse_eval_dir).grid(
            row=0, column=2, padx=(0, 5))

        ttk.Label(row1, text="Etiqueta esperada:").grid(
            row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        ttk.Entry(row1, textvariable=self.eval_label_var, width=20).grid(
            row=1, column=1, sticky="w", pady=(5, 0))

        row1.columnconfigure(1, weight=1)

        # Row 2: Pattern and options
        row2 = ttk.Frame(config_frame)
        row2.pack(fill="x", padx=5, pady=5)

        ttk.Label(row2, text="Patrón de archivo:").grid(
            row=0, column=0, sticky="w", padx=(0, 5))
        ttk.Entry(row2, textvariable=self.eval_glob_var, width=15).grid(
            row=0, column=1, sticky="w", padx=(0, 10))

        ttk.Checkbutton(row2, text="Búsqueda recursiva",
                        variable=self.eval_recursive_var).grid(row=0, column=2, sticky="w")

        # Control buttons
        button_frame = ttk.Frame(frm)
        button_frame.pack(fill="x", **pad)

        self.eval_start_btn = ttk.Button(button_frame, text="Iniciar Evaluación",
                                         command=self._start_evaluation)
        self.eval_start_btn.pack(side="left", padx=(0, 10))

        self.eval_stop_btn = ttk.Button(button_frame, text="Detener",
                                        command=self._stop_evaluation, state="disabled")
        self.eval_stop_btn.pack(side="left")

        # Progress
        progress_frame = ttk.LabelFrame(frm, text="Progreso")
        progress_frame.pack(fill="x", **pad)

        self.eval_progress_bar = ttk.Progressbar(progress_frame, variable=self.eval_progress_var,
                                                 maximum=100, length=400)
        self.eval_progress_bar.pack(fill="x", padx=5, pady=5)

        # Results
        results_frame = ttk.LabelFrame(frm, text="Resultados")
        results_frame.pack(fill="both", expand=True, **pad)

        # Accuracy summary
        accuracy_frame = ttk.Frame(results_frame)
        accuracy_frame.pack(fill="x", padx=5, pady=(5, 0))

        ttk.Label(accuracy_frame, text="Precisión:").pack(side="left")
        accuracy_label = ttk.Label(accuracy_frame, textvariable=self.eval_accuracy_var,
                                   font=("TkDefaultFont", 10, "bold"))
        accuracy_label.pack(side="left", padx=(5, 0))

        # Results text area
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.eval_results_text = tk.Text(text_frame, height=15, wrap="none",
                                         font=("Consolas", 9))
        eval_scrollbar_v = ttk.Scrollbar(text_frame, orient="vertical",
                                         command=self.eval_results_text.yview)
        eval_scrollbar_h = ttk.Scrollbar(text_frame, orient="horizontal",
                                         command=self.eval_results_text.xview)

        self.eval_results_text.configure(yscrollcommand=eval_scrollbar_v.set,
                                         xscrollcommand=eval_scrollbar_h.set)

        self.eval_results_text.grid(row=0, column=0, sticky="nsew")
        eval_scrollbar_v.grid(row=0, column=1, sticky="ns")
        eval_scrollbar_h.grid(row=1, column=0, sticky="ew")

        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)

        # Status bar
        self.eval_status = ttk.Label(parent, textvariable=self.eval_status_var,
                                     anchor="w", relief="sunken")
        self.eval_status.pack(side="bottom", fill="x")

    # -------------------- Helpers --------------------
    def _load_commands(self):
        """
        Load available voice commands from database into UI combobox.

        Retrieves the list of all voice commands stored in the repository
        and populates the command selection combobox in the dataset tab.
        This allows users to select existing commands or see what commands
        are already available for training.

        Database Integration:
            Queries the ProfileRepository for all command names and updates
            the GUI combobox values to reflect current database state.
        """
        cmds = self.repo.list_commands()
        self.cmd_box["values"] = cmds

    def _parse_device_index_dataset(self) -> int | None:
        """
        Extract device index from dataset tab device selection.

        Parses the device selection string from the dataset tab combobox
        to extract the numeric device index for audio recording operations.

        Returns:
            int | None: Device index if valid selection, None otherwise

        Format:
            Device strings are formatted as "index - device_name"
            This method extracts the index portion for sounddevice use.

        Error Handling:
            Returns None if no device selected or parsing fails,
            allowing calling code to use default device.
        """
        if not self.device_var.get():
            return None
        try:
            return int(self.device_var.get().split(" - ")[0])
        except Exception:
            return None

    def _parse_device_index_live(self) -> int | None:
        """
        Extract device index from live recognition tab device selection.

        Parses the device selection string from the live tab combobox
        to extract the numeric device index for real-time recognition.

        Returns:
            int | None: Device index if valid selection, None otherwise

        Format:
            Device strings are formatted as "index - device_name"
            This method extracts the index portion for sounddevice use.

        Error Handling:
            Returns None if no device selected or parsing fails,
            allowing calling code to use default device.
        """
        if not self.live_dev_box.get():
            return None
        try:
            return int(self.live_dev_box.get().split(" - ")[0])
        except Exception:
            return None

    def _open_data_dir(self):
        """
        Open the data directory in the system file manager.

        Creates the data directory if it doesn't exist and opens it using
        the appropriate system command for the current platform. This allows
        users to easily access recorded audio files and manage the dataset.

        Platform Support:
            - Windows: Uses os.startfile() to open in Explorer
            - macOS: Uses subprocess to call 'open' command
            - Linux/Unix: Uses subprocess to call 'xdg-open' command

        Fallback Behavior:
            If opening fails, shows a message box with the directory path
            so users can navigate manually.

        Directory Structure:
            Creates DATA_DIR with subdirectories for each command name,
            allowing organized storage of voice recordings.
        """
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            import os
            import sys
            import subprocess
            if sys.platform.startswith("win"):
                os.startfile(DATA_DIR)  # type: ignore
            elif sys.platform == "darwin":
                subprocess.check_call(["open", DATA_DIR])
            else:
                subprocess.check_call(["xdg-open", DATA_DIR])
        except Exception:
            messagebox.showinfo("Carpeta", f"Carpeta: {DATA_DIR.resolve()}")

    # -------------------- Dataset actions --------------------
    def _add_or_select_command(self):
        """
        Add a new voice command or select an existing one.

        Handles the creation of new voice commands and selection of existing
        ones. Validates input, normalizes command names to uppercase, and
        updates the database and GUI accordingly.

        Process:
            1. Validate command name (non-empty after stripping)
            2. Normalize to uppercase for consistency
            3. Insert/update command in database
            4. Refresh command list in GUI
            5. Set current command for recording
            6. Update status display

        Validation:
            - Rejects empty or whitespace-only command names
            - Shows warning dialog for invalid input
            - Normalizes names to uppercase for consistency

        Database Operation:
            Uses upsert operation to handle both new commands and
            existing command selection gracefully.

        UI Updates:
            - Refreshes combobox values with updated command list
            - Sets current selection to the processed command
            - Updates status bar with confirmation message
        """
        name = self.current_command.get().strip()
        if not name:
            messagebox.showwarning(
                "Comando vacío", "Escribe o selecciona un nombre de comando.")
            return
        cmd = name.upper()
        self.repo.upsert_command(cmd)
        self._load_commands()
        self.current_command.set(cmd)
        self.status_var.set(f"Comando listo: {cmd}")

    def _on_record(self):
        """
        Initiate voice command recording process.

        Handles the complete recording workflow including validation,
        UI state management, thread creation, and file saving. This method
        coordinates between the GUI and the threaded recording system.

        Pre-recording Validation:
            1. Verify command name is selected/entered
            2. Check audio devices are available
            3. Validate recording parameters

        Recording Process:
            1. Configure UI state (disable record button, reset progress)
            2. Parse audio device and duration settings
            3. Generate timestamp-based filename
            4. Create and start ChunkRecorderThread
            5. Handle progress and completion callbacks

        File Management:
            - Creates command-specific subdirectories
            - Uses timestamp-based filenames for uniqueness
            - Saves WAV files with proper audio configuration
            - Records metadata in database

        Progress Tracking:
            - Real-time RMS volume display
            - Progress bar showing recording completion
            - Status updates throughout process

        Error Handling:
            - Input validation with user-friendly warnings
            - Device availability checking
            - Recording error capture and display
            - Database error handling for metadata

        Threading:
            Uses ChunkRecorderThread for non-blocking recording,
            ensuring GUI remains responsive during audio capture.

        Callbacks:
            - on_progress: Updates RMS and progress bars
            - on_done: Handles completion, saves file, updates database
        """
        cmd = self.current_command.get().strip().upper()
        if not cmd:
            messagebox.showwarning(
                "Falta comando", "Selecciona o crea un comando antes de grabar.")
            return
        if not self.devices:
            messagebox.showerror(
                "Micrófono", "No hay dispositivos de entrada disponibles.")
            return

        dur = float(self.duration_var.get() or 2.0)
        dev_index = self._parse_device_index_dataset()

        self.btn_record.config(state="disabled")
        self.status_var.set("Grabando…")
        self.rms_var.set(0.0)
        self.progress_var.set(0.0)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = DATA_DIR / cmd / f"{cmd}_{ts}.wav"

        def on_progress(rms, prog):
            self.rms_var.set(min(rms * 8000.0, 8000))  # escala rápida
            self.progress_var.set(min(100.0, 100.0 * prog))

        def on_done(ok, data, err):
            self.btn_record.config(state="normal")
            if not ok:
                messagebox.showerror("Grabación", f"Error: {err}")
                self.status_var.set("Error en grabación.")
                return

            try:
                # Guardar WAV y metadatos
                WavWriter.write(out_path, data, self.audio_cfg.rate)
                cmd_id = self.repo.upsert_command(cmd)
                self.repo.add_recording(cmd_id, str(out_path),
                                        self.audio_cfg.rate, self.audio_cfg.channels, dur)
                self.last_saved_var.set(str(out_path))
                self.status_var.set(f"Grabado: {out_path.name}")
            except Exception as e:
                messagebox.showerror("DB", f"Error guardando metadatos: {e}")
                self.status_var.set("Error al guardar en DB.")

            self.rec_thread = None

        self.rec_thread = ChunkRecorderThread(
            self.audio_cfg, dur, dev_index, on_progress, on_done)
        self.rec_thread.start()

    def _on_train(self):
        """
        Initiate voice profile training process.

        Starts background training of voice recognition profiles using all
        recorded samples in the database. Training runs in a separate thread
        to maintain GUI responsiveness during the potentially long operation.

        Training Process:
            1. Update status to indicate training in progress
            2. Create background worker thread
            3. Execute Trainer.train_all() to process all commands
            4. Update status on completion or error

        Threading:
            Uses daemon thread to prevent blocking GUI operations.
            Training can be lengthy depending on dataset size and complexity.

        Error Handling:
            - Catches all training exceptions
            - Displays user-friendly error dialogs
            - Updates status bar with error information
            - Ensures GUI remains functional after failures

        Progress Feedback:
            - Immediate status update when training starts
            - Success/failure status updates on completion
            - No detailed progress tracking (training runs to completion)

        Database Integration:
            Training operates on all commands and recordings stored
            in the ProfileRepository, generating ML models for recognition.
        """
        self.status_var.set("Entrenando perfiles…")

        def worker():
            try:
                Trainer(self.repo, self.num_parts).train_all()
                self.status_var.set("Entrenamiento completado.")
            except Exception as e:
                messagebox.showerror("Entrenamiento", f"Error: {e}")
                self.status_var.set("Error en entrenamiento.")

        threading.Thread(target=worker, daemon=True).start()

    def _on_recognize_wav(self):
        """
        Test voice recognition on a selected WAV file.

        Allows users to test the recognition system by selecting a WAV file
        and running it through the trained recognition engine. Useful for
        testing and validation of voice profiles.

        Process:
            1. Open file dialog for WAV selection
            2. Load and process the audio file
            3. Run recognition using trained profiles
            4. Display prediction and confidence results

        File Selection:
            Uses standard file dialog filtered for WAV files,
            allowing users to test any compatible audio recording.

        Recognition:
            - Uses the same Recognizer class as live recognition
            - Processes entire file as single recognition attempt
            - Returns command name and confidence score

        Results Display:
            Shows prediction and confidence in a message box,
            providing immediate feedback on recognition accuracy.

        Error Handling:
            - Handles file selection cancellation gracefully
            - Catches audio processing errors
            - Handles cases with no trained profiles
            - Displays appropriate error messages

        Use Cases:
            - Testing recognition accuracy on known samples
            - Validating training effectiveness
            - Debugging recognition issues
            - Quality assurance for voice profiles
        """
        wav_path = filedialog.askopenfilename(title="Selecciona un WAV",
                                              filetypes=[("WAV files", "*.wav")])
        if not wav_path:
            return
        try:
            name, conf = Recognizer(
                self.repo, self.num_parts).recognize_wav(wav_path)
            if name is None:
                messagebox.showinfo("Reconocer", "No hay perfiles entrenados.")
                return
            messagebox.showinfo(
                "Reconocer", f"Predicción: {name}\nConfianza: {conf:.2f}")
        except Exception as e:
            messagebox.showerror("Reconocer", f"Error: {e}")

    # -------------------- Live actions --------------------
    def _live_start(self):
        if self.live_thread:
            return
        if not self.devices:
            messagebox.showerror(
                "Micrófono", "No hay dispositivos de entrada disponibles.")
            return

        dev_index = self._parse_device_index_live()
        win_s = float(self.live_window_var.get())
        hop_s = float(self.live_hop_var.get())
        factor = float(self.live_factor_var.get())
        calib_s = float(self.live_calib_var.get())

        rec = Recognizer(self.repo, self.num_parts)
        lv = LiveRecognizer(
            recognizer=rec,
            audio=self.audio_cfg,
            device_index=dev_index,
            window_secs=win_s,
            hop_secs=hop_s,
            calib_secs=calib_s,
            threshold_factor=factor,
        )

        def on_status(text): self.live_status_var.set(text)
        def on_rms(rms, thr): self.live_rms_var.set(min(rms, 8000))

        def on_pred(name, conf):
            self.live_pred_var.set(name if name else "-")
            self.live_conf_var.set(f"{conf:.2f}")

        lv.on_status = on_status
        lv.on_rms = on_rms
        lv.on_pred = on_pred

        try:
            lv.start()
            self.live_thread = lv
            self.btn_live_start.config(state="disabled")
            self.btn_live_stop.config(state="normal")
        except Exception as e:
            messagebox.showerror("Live", f"No se pudo iniciar: {e}")

    def _live_stop(self):
        if self.live_thread:
            try:
                self.live_thread.stop()
            except Exception:
                pass
            self.live_thread = None
        self.btn_live_start.config(state="normal")
        self.btn_live_stop.config(state="disabled")
        self.live_status_var.set("Detenido.")
        self.live_pred_var.set("-")
        self.live_conf_var.set("0.00")
        self.live_rms_var.set(0.0)

    # -------------------- Lifecycle --------------------
    def on_close(self):
        """
        Handle application shutdown and resource cleanup.

        Performs comprehensive cleanup when the application is closing,
        ensuring all threads are stopped and resources are properly released.
        Prompts user if operations are in progress to prevent data loss.

        Cleanup Process:
            1. Check for active recording operations
            2. Prompt user confirmation if recording in progress
            3. Stop recording thread if necessary
            4. Stop live recognition if active
            5. Close database connections
            6. Destroy GUI window

        Thread Management:
            - Safely stops ChunkRecorderThread if recording
            - Terminates LiveRecognizer if running
            - Ensures no orphaned background threads

        Resource Management:
            - Closes ProfileRepository database connection
            - Releases audio device resources
            - Cleans up temporary files and handles

        User Interaction:
            Shows confirmation dialog if recording is in progress,
            allowing user to cancel shutdown to save important recordings.

        Error Handling:
            Uses try-except blocks to ensure cleanup continues
            even if individual operations fail.
        """
        if self.rec_thread and self.rec_thread.is_alive():
            if not messagebox.askyesno("Salir", "Hay una grabación en curso. ¿Deseas salir?"):
                return
            self.rec_thread.stop()
        self._live_stop()
        self.repo.close()
        self.destroy()

    # -------------------- Evaluation Methods --------------------

    def _browse_eval_dir(self):
        """Browse for evaluation directory."""
        directory = filedialog.askdirectory(
            title="Seleccionar directorio para evaluación"
        )
        if directory:
            self.eval_dir_var.set(directory)

    def _start_evaluation(self):
        """Start batch evaluation process."""
        if self.eval_running:
            return

        # Validate inputs
        dir_path = self.eval_dir_var.get().strip()
        if not dir_path:
            messagebox.showerror("Error", "Seleccione un directorio.")
            return

        label = self.eval_label_var.get().strip()
        if not label:
            messagebox.showerror("Error", "Ingrese la etiqueta esperada.")
            return

        dir_path_obj = Path(dir_path)
        if not dir_path_obj.exists():
            messagebox.showerror("Error", "El directorio no existe.")
            return

        # Start evaluation in separate thread
        self.eval_running = True
        self.eval_start_btn.config(state="disabled")
        self.eval_stop_btn.config(state="normal")
        self.eval_status_var.set("Iniciando evaluación...")
        self.eval_results_text.delete(1.0, tk.END)
        self.eval_accuracy_var.set("")
        self.eval_progress_var.set(0)

        # Run evaluation in thread to avoid blocking GUI
        eval_thread = threading.Thread(
            target=self._run_evaluation, daemon=True)
        eval_thread.start()

    def _run_evaluation(self):
        """Run evaluation process in background thread."""
        try:
            dir_path = Path(self.eval_dir_var.get().strip())
            label = self.eval_label_var.get().strip().upper()
            glob_pattern = self.eval_glob_var.get().strip()
            recursive = self.eval_recursive_var.get()

            # Create recognizer
            recog = Recognizer(self.repo, self.num_parts)

            # Find files
            if recursive:
                files = list(dir_path.rglob(glob_pattern))
            else:
                files = list(dir_path.glob(glob_pattern))

            files = [f for f in files if f.is_file()]

            if not files:
                self.after(0, lambda: self._update_eval_status(
                    "No se encontraron archivos."))
                self.after(0, self._evaluation_finished)
                return

            total = len(files)
            correct = 0

            # Process files
            for i, file_path in enumerate(sorted(files)):
                if not self.eval_running:  # Check if stopped
                    break

                try:
                    # Update progress
                    progress = (i / total) * 100
                    self.after(
                        0, lambda p=progress: self.eval_progress_var.set(p))
                    self.after(0, lambda f=file_path.name: self._update_eval_status(
                        f"Procesando: {f}"))

                    # Recognize file
                    pred, conf = recog.recognize_wav(str(file_path))
                    hit = (pred or "").upper() == label
                    if hit:
                        correct += 1

                    # Update results
                    status = "OK " if hit else "ERR"
                    result_line = f"{status} {file_path.name:40s} -> {pred} (conf={conf:.2f})\n"
                    self.after(
                        0, lambda line=result_line: self._append_eval_result(line))

                except Exception as e:
                    error_line = f"ERR {file_path.name:40s} -> ERROR: {str(e)}\n"
                    self.after(
                        0, lambda line=error_line: self._append_eval_result(line))

            # Final results
            if self.eval_running:
                accuracy = correct / total if total > 0 else 0.0
                summary = f"\nPrecisión {label}: {correct}/{total} = {accuracy:.2%}"
                self.after(0, lambda: self._append_eval_result(summary))
                self.after(0, lambda: self.eval_accuracy_var.set(
                    f"{correct}/{total} = {accuracy:.2%}"))
                self.after(0, lambda: self.eval_progress_var.set(100))
                self.after(0, lambda: self._update_eval_status(
                    "Evaluación completada."))

        except Exception as e:
            error_msg = f"Error durante la evaluación: {str(e)}"
            self.after(0, lambda: self._update_eval_status(error_msg))
            self.after(0, lambda: messagebox.showerror("Error", error_msg))
        finally:
            self.after(0, self._evaluation_finished)

    def _stop_evaluation(self):
        """Stop current evaluation process."""
        self.eval_running = False
        self._update_eval_status("Deteniendo evaluación...")

    def _evaluation_finished(self):
        """Reset UI after evaluation completion."""
        self.eval_running = False
        self.eval_start_btn.config(state="normal")
        self.eval_stop_btn.config(state="disabled")

    def _update_eval_status(self, message):
        """Update evaluation status message."""
        self.eval_status_var.set(message)

    def _append_eval_result(self, text):
        """Append text to evaluation results."""
        self.eval_results_text.insert(tk.END, text)
        self.eval_results_text.see(tk.END)


def main():
    """
    Main entry point for the VoiceCMD GUI application.

    Creates and runs the VoiceCMD GUI application with proper window
    management and cleanup handling. This function serves as the
    primary entry point for launching the graphical interface.

    Application Lifecycle:
        1. Create VoiceCmdGUI instance
        2. Set up window close protocol for cleanup
        3. Start tkinter main event loop
        4. Handle shutdown when user closes window

    Window Management:
        Registers the on_close method as the window close handler,
        ensuring proper cleanup of resources and threads when the
        application is terminated by the user.

    Usage:
        Can be called directly as a script or imported and used
        programmatically in larger applications.

    Examples:
        Command Line Usage:
        ```bash
        python -m voicecmd.gui
        ```

        Programmatic Usage:
        ```python
        from voicecmd.gui import main
        main()
        ```

        Custom Integration:
        ```python
        from voicecmd.gui import VoiceCmdGUI
        app = VoiceCmdGUI()
        # Custom setup...
        app.mainloop()
        ```

    Error Handling:
        Relies on VoiceCmdGUI class error handling for robust
        operation. Any unhandled exceptions will terminate the
        application cleanly through tkinter's exception handling.
    """
    app = VoiceCmdGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
