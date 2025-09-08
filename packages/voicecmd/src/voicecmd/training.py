"""
Training module for voice command recognition system.

This module provides the training pipeline for creating voice command profiles
from recorded audio samples. The training process involves feature extraction
from multiple audio recordings of each command and computing representative
profiles (centroids) that can be used for recognition.

Training Pipeline:
1. Load audio recordings for each command from the repository
2. Extract spectral features from each recording using FFT analysis
3. Compute the centroid (mean) of all feature vectors for each command
4. Store the resulting profile in the repository for later recognition

The centroid-based approach assumes that the feature space forms clusters
around each command, and the mean represents the most characteristic features
of that command. This simple yet effective method works well for distinguishing
between different voice commands.

Classes:
    Trainer: Main training orchestrator that manages the complete pipeline

Dependencies:
    - numpy: For numerical computations and array operations
    - repository: For data persistence and retrieval
    - features: For audio feature extraction

Algorithm Details:
    The training uses a centroid-based clustering approach where:
    - Each command's recordings are converted to feature vectors
    - The mean (centroid) of all vectors represents the command profile
    - Profiles are stored with their feature configuration for compatibility
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from .repository import ProfileRepository
from .features import FeatureExtractor
from .config import get_data_dir


def resolve_recording_path(relative_path: str) -> Path:
    """
    Resolve a recording path to an absolute path.

    This function handles both relative and absolute paths, ensuring that
    recordings stored with relative paths can be found regardless of the
    current working directory.

    Args:
        relative_path: Path as stored in the database (may be relative or absolute)

    Returns:
        Path: Absolute path to the recording file

    Examples:
        >>> # Relative path gets resolved to data directory
        >>> path = resolve_recording_path("data/UP/UP_123.wav")
        >>> print(path)
        /home/user/.voicecmd/data/UP/UP_123.wav

        >>> # Absolute path remains unchanged
        >>> path = resolve_recording_path("/absolute/path/file.wav")
        >>> print(path)
        /absolute/path/file.wav
    """
    path = Path(relative_path)

    # If already absolute, return as-is
    if path.is_absolute():
        return path

    # For relative paths, check if they exist relative to current directory first
    if path.exists():
        return path.resolve()

    # Otherwise, resolve relative to the standard data directory
    data_dir = get_data_dir()
    resolved_path = data_dir.parent / relative_path
    return resolved_path.resolve()


class Trainer:
    """
    Voice command training orchestrator using centroid-based clustering.

    This class manages the complete training pipeline for voice command
    recognition. It coordinates feature extraction from audio recordings
    and computes representative profiles for each command using a centroid
    (mean) approach. The trainer is designed to work with the repository
    system for data persistence and maintains consistency in feature
    extraction configuration.

    The training algorithm:
    1. Iterates through all commands in the repository
    2. Loads all audio recordings for each command
    3. Extracts spectral features from each recording
    4. Computes the centroid (mean) of all feature vectors
    5. Stores the resulting profile for recognition use

    Attributes:
        repo (ProfileRepository): Repository instance for data access
        fe (FeatureExtractor): Feature extractor with matching configuration
        num_parts (int): Number of frequency bands for feature extraction

    Examples:
        >>> from pathlib import Path
        >>> from voicecmd.repository import ProfileRepository
        >>> 
        >>> # Initialize repository and trainer
        >>> repo = ProfileRepository(Path("voice_commands.db"))
        >>> trainer = Trainer(repo, num_parts=100)
        >>> 
        >>> # Assume recordings have been added to repository
        >>> # Train all commands
        >>> trainer.train_all()
        [OK] Profile updated: hello (num_parts=100)
        [OK] Profile updated: stop (num_parts=100)
        [WARN] No recordings for play
        >>> 
        >>> # Close repository when done
        >>> repo.close()

    Mathematical Foundation:
        For a command with recordings R = {r₁, r₂, ..., rₙ} and their
        feature vectors F = {f₁, f₂, ..., fₙ}, the profile P is computed as:

        P = (1/n) * Σᵢ fᵢ

        This centroid represents the "average" spectral characteristics
        of the command and serves as a template for recognition.

    Note:
        The trainer requires that audio recordings are already stored in
        the repository with proper metadata. Feature extraction configuration
        (num_parts) must be consistent between training and recognition phases.
    """

    def __init__(self, repo: ProfileRepository, num_parts: int):
        """
        Initialize the trainer with repository and feature configuration.

        Sets up the training pipeline with the specified repository for data
        access and feature extractor configuration. The num_parts parameter
        must match between training and recognition to ensure compatibility.

        Args:
            repo (ProfileRepository): Repository instance containing commands
                                    and recordings. Must be properly initialized
                                    and contain training data.
            num_parts (int): Number of frequency bands for feature extraction.
                           Must be a positive integer. This value should match
                           the configuration used during recognition.

        Examples:
            >>> from pathlib import Path
            >>> from voicecmd.repository import ProfileRepository
            >>> 
            >>> # Standard configuration for voice commands
            >>> repo = ProfileRepository(Path("commands.db"))
            >>> trainer = Trainer(repo, num_parts=100)
            >>> 
            >>> # High-resolution configuration for detailed analysis
            >>> detailed_trainer = Trainer(repo, num_parts=200)
            >>> 
            >>> # Fast configuration for real-time systems
            >>> fast_trainer = Trainer(repo, num_parts=50)

        Note:
            The feature extractor is automatically configured with the
            specified num_parts parameter, ensuring consistency throughout
            the training process.
        """
        self.repo = repo
        self.fe = FeatureExtractor(num_parts)
        self.num_parts = num_parts

    def train_all(self):
        """
        Train all commands by computing centroid profiles from recordings.

        Executes the complete training pipeline for all commands stored in
        the repository. For each command, this method:
        1. Retrieves all associated audio recordings
        2. Validates that audio files exist and are accessible
        3. Extracts spectral features from each valid recording
        4. Computes the centroid (mean) of all feature vectors
        5. Stores the resulting profile for recognition use

        The method provides progress feedback and warnings for commands
        without training data or missing files. Commands with no valid
        recordings are skipped with a warning message. Missing or corrupted
        audio files are ignored rather than causing training to fail.

        Robust Error Handling:
            - Missing audio files are skipped with warnings
            - Corrupted files are ignored with error messages
            - Training continues with remaining valid files
            - Commands need at least one valid recording to create a profile

        Raises:
            sqlite3.Error: If database operations fail (unrecoverable)

        Examples:
            >>> from pathlib import Path
            >>> from voicecmd.repository import ProfileRepository
            >>> 
            >>> # Set up repository with some training data
            >>> repo = ProfileRepository(Path("training.db"))
            >>> 
            >>> # Add commands and recordings (some files may be missing)
            >>> hello_id = repo.upsert_command("hello")
            >>> repo.add_recording(hello_id, "/data/hello_001.wav", 16000, 1, 2.5)  # exists
            >>> repo.add_recording(hello_id, "/data/hello_002.wav", 16000, 1, 2.3)  # missing
            >>> repo.add_recording(hello_id, "/data/hello_003.wav", 16000, 1, 2.7)  # exists
            >>> 
            >>> stop_id = repo.upsert_command("stop")
            >>> repo.add_recording(stop_id, "/data/stop_001.wav", 16000, 1, 1.8)    # missing
            >>> repo.add_recording(stop_id, "/data/stop_002.wav", 16000, 1, 2.1)    # missing
            >>> 
            >>> # Empty command (no recordings)
            >>> play_id = repo.upsert_command("play")
            >>> 
            >>> # Train all commands
            >>> trainer = Trainer(repo, num_parts=100)
            >>> trainer.train_all()
            [WARN] 1 missing files for hello: ['/data/hello_002.wav']
            [OK] Profile updated: hello (num_parts=100, 2/3 recordings)
            [WARN] 2 missing files for stop: ['/data/stop_001.wav', '/data/stop_002.wav']
            [WARN] No valid recordings found for stop
            [WARN] No recordings for play
            >>> 
            >>> # Verify profiles were created
            >>> profiles = repo.load_profiles(num_parts=100)
            >>> print(f"Trained commands: {list(profiles.keys())}")
            Trained commands: ['hello']

        Training Process Details:
            1. **Command Iteration**: Processes commands in alphabetical order
            2. **File Validation**: Checks existence before feature extraction
            3. **Feature Extraction**: Each valid audio file is converted to a feature vector
            4. **Error Recovery**: Missing/corrupted files are skipped with warnings
            5. **Centroid Computation**: Mean of all valid vectors represents the command
            6. **Profile Storage**: Centroids are stored with configuration metadata

        Mathematical Details:
            For a command with n valid recordings and feature vectors [f₁, f₂, ..., fₙ]:

            Centroid = (f₁ + f₂ + ... + fₙ) / n

            Where each fᵢ is a vector of length num_parts containing spectral energies.

        Performance Considerations:
            - Training time scales linearly with number of valid recordings
            - File validation adds minimal overhead compared to feature extraction
            - Memory usage depends on num_parts and number of simultaneous recordings
            - Missing files cause warnings but don't slow down processing

        Output Format:
            - [OK] messages indicate successful profile creation with valid file count
            - [WARN] messages indicate missing files, commands without data, or processing errors
            - Progress is reported for each command processed

        Note:
            - Commands without valid recordings are skipped but remain in the database
            - Existing profiles with matching (command_id, num_parts) are overwritten
            - Missing files are reported but don't halt training
            - At least one valid recording is required to create a profile
            - Feature extraction errors are caught and reported as warnings
        """
        for name in self.repo.list_commands():
            cmd_id = self.repo.upsert_command(name)
            paths = self.repo.list_recordings(cmd_id)
            if not paths:
                print(f"[WARN] No recordings for {name}")
                continue

            # Extract features from all recordings, skipping missing files
            resolved_paths = [resolve_recording_path(p) for p in paths]
            seqs = []
            missing_files = []

            for i, (original_path, resolved_path) in enumerate(zip(paths, resolved_paths)):
                try:
                    if not resolved_path.exists():
                        missing_files.append(original_path)
                        continue

                    features = self.fe.from_wav(str(resolved_path))
                    seqs.append(features)
                except Exception as e:
                    print(f"[WARN] Failed to process {original_path}: {e}")
                    continue

            # Report missing files if any
            if missing_files:
                print(
                    f"[WARN] {len(missing_files)} missing files for {name}: {missing_files[:3]}{'...' if len(missing_files) > 3 else ''}")

            # Check if we have enough valid recordings
            if not seqs:
                print(f"[WARN] No valid recordings found for {name}")
                continue

            # Stack feature vectors and compute centroid
            arr = np.stack(seqs, axis=0)
            centroid = arr.mean(axis=0)

            # Save the computed profile
            self.repo.save_profile(cmd_id, self.num_parts, centroid)
            print(
                f"[OK] Profile updated: {name} (num_parts={self.num_parts}, {len(seqs)}/{len(paths)} recordings)")
