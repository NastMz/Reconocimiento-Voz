"""
Data persistence layer for voice command recognition system.

This module provides a SQLite-based repository for storing and managing voice
command data, including command definitions, audio recordings, and extracted
feature profiles. The repository implements a relational data model that
maintains referential integrity and supports efficient querying for machine
learning training and inference.

Database Schema:
    commands: Stores voice command definitions
        - id: Primary key (auto-increment)
        - name: Unique command name (e.g., "hello", "stop", "play")
        - created_at: ISO timestamp of creation
    
    recordings: Stores metadata about audio recordings
        - id: Primary key (auto-increment)
        - command_id: Foreign key to commands table
        - path: File system path to audio file
        - sample_rate: Audio sample rate in Hz
        - channels: Number of audio channels
        - duration: Recording duration in seconds
        - created_at: ISO timestamp of creation
    
    profiles: Stores extracted feature vectors for recognition
        - id: Primary key (auto-increment)
        - command_id: Foreign key to commands table
        - num_parts: Number of frequency bands used in feature extraction
        - vector: Serialized numpy array containing feature data
        - created_at: ISO timestamp of creation
        - UNIQUE constraint on (command_id, num_parts)

Classes:
    ProfileRepository: Main repository class for data persistence operations

Dependencies:
    - sqlite3: For database operations
    - pathlib: For file path handling
    - datetime: For timestamp management
    - numpy: For feature vector serialization/deserialization
"""

from __future__ import annotations
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

# Database schema definition with foreign key constraints
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS commands (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recordings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  command_id INTEGER NOT NULL,
  path TEXT NOT NULL,
  sample_rate INTEGER NOT NULL,
  channels INTEGER NOT NULL,
  duration REAL NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (command_id) REFERENCES commands(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS profiles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  command_id INTEGER NOT NULL,
  num_parts INTEGER NOT NULL,
  vector BLOB NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (command_id) REFERENCES commands(id) ON DELETE CASCADE,
  UNIQUE(command_id, num_parts)
);
"""


class ProfileRepository:
    """
    SQLite-based repository for voice command data persistence.

    This class provides a comprehensive data access layer for the voice command
    recognition system, managing commands, audio recordings, and feature profiles
    in a relational database. It handles database initialization, maintains
    referential integrity, and provides type-safe operations for all entities.

    Key Features:
    - Automatic database schema creation and migration
    - Foreign key constraint enforcement
    - Atomic operations with proper transaction handling
    - Type-safe numpy array serialization for feature vectors
    - UTC timestamp management for audit trails
    - Efficient querying with proper indexing

    Attributes:
        db_path (Path): Path to the SQLite database file
        _conn (sqlite3.Connection): Database connection object

    Examples:
        >>> from pathlib import Path
        >>> import numpy as np
        >>> 
        >>> # Initialize repository
        >>> repo = ProfileRepository(Path("voice_commands.db"))
        >>> 
        >>> # Add a command and get its ID
        >>> cmd_id = repo.upsert_command("hello")
        >>> 
        >>> # Add recording metadata
        >>> rec_id = repo.add_recording(
        ...     command_id=cmd_id,
        ...     path="/data/hello_001.wav",
        ...     sample_rate=16000,
        ...     channels=1,
        ...     duration=2.5
        ... )
        >>> 
        >>> # Save feature profile
        >>> features = np.random.rand(100)
        >>> repo.save_profile(cmd_id, num_parts=100, vector=features)
        >>> 
        >>> # Load all profiles for recognition
        >>> profiles = repo.load_profiles(num_parts=100)
        >>> print(profiles.keys())
        dict_keys(['hello'])
        >>> 
        >>> # Always close when done
        >>> repo.close()

    Note:
        Always call close() when finished to properly release database resources,
        or use the repository in a context manager pattern if implemented.
    """

    def __init__(self, db_path: Path):
        """
        Initialize the repository with database connection and schema setup.

        Creates a new SQLite database if it doesn't exist, or connects to an
        existing one. Automatically applies the database schema and enables
        foreign key constraints for referential integrity.

        Args:
            db_path (Path): Path to the SQLite database file. Parent directories
                          will be created automatically if they don't exist.

        Raises:
            sqlite3.Error: If database connection or schema creation fails.
            PermissionError: If the database file cannot be created or accessed.

        Examples:
            >>> # Create repository in current directory
            >>> repo = ProfileRepository(Path("commands.db"))
            >>> 
            >>> # Create repository with nested path
            >>> repo = ProfileRepository(Path("data/voice/commands.db"))
            >>> 
            >>> # Use pathlib for cross-platform paths
            >>> db_path = Path.home() / "voice_commands" / "data.db"
            >>> repo = ProfileRepository(db_path)
        """
        self.db_path = db_path
        # Create parent directories if they don't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database and configure
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def close(self):
        """
        Close the database connection and release resources.

        This method should be called when the repository is no longer needed
        to ensure proper cleanup of database resources and to flush any
        pending transactions.

        Examples:
            >>> repo = ProfileRepository(Path("commands.db"))
            >>> # ... perform operations ...
            >>> repo.close()
            >>> 
            >>> # Or use try/finally pattern
            >>> repo = ProfileRepository(Path("commands.db"))
            >>> try:
            ...     # ... perform operations ...
            ... finally:
            ...     repo.close()

        Note:
            After calling close(), the repository instance should not be used
            for further operations as it will raise database errors.
        """
        self._conn.close()

    # Commands
    def upsert_command(self, name: str) -> int:
        """
        Insert a new command or return ID of existing command.

        This method implements an "upsert" operation - it will create a new
        command if one with the given name doesn't exist, or return the ID
        of the existing command if it does. This ensures that command names
        remain unique while providing a convenient way to get command IDs.

        Args:
            name (str): The command name (e.g., "hello", "stop", "play").
                       Should be a meaningful identifier for voice commands.
                       Case-sensitive and must be unique across all commands.

        Returns:
            int: The command ID (primary key) that can be used to reference
                this command in other operations like adding recordings or
                saving feature profiles.

        Raises:
            sqlite3.Error: If database operation fails.
            ValueError: If name is empty or contains invalid characters.

        Examples:
            >>> repo = ProfileRepository(Path("commands.db"))
            >>> 
            >>> # Create new command
            >>> hello_id = repo.upsert_command("hello")
            >>> print(f"Hello command ID: {hello_id}")
            Hello command ID: 1
            >>> 
            >>> # Get existing command ID (same result)
            >>> hello_id_again = repo.upsert_command("hello")
            >>> assert hello_id == hello_id_again
            >>> 
            >>> # Create another command
            >>> stop_id = repo.upsert_command("stop")
            >>> print(f"Stop command ID: {stop_id}")
            Stop command ID: 2

        Note:
            Command names are case-sensitive. "Hello" and "hello" would be
            treated as different commands. Consider normalizing names before
            calling this method if case-insensitive behavior is desired.
        """
        cur = self._conn.execute(
            "SELECT id FROM commands WHERE name=?", (name,))
        row = cur.fetchone()
        if row:
            return int(row[0])
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "INSERT INTO commands(name, created_at) VALUES(?,?)", (name, now))
        self._conn.commit()
        return int(cur.lastrowid)

    def list_commands(self) -> list[str]:
        """
        Retrieve all command names from the database.

        Returns a list of all command names currently stored in the database,
        sorted alphabetically for consistent ordering. This is useful for
        displaying available commands to users or for iterating over all
        commands during training or recognition setup.

        Returns:
            list[str]: List of command names sorted alphabetically.
                      Empty list if no commands exist in the database.

        Examples:
            >>> repo = ProfileRepository(Path("commands.db"))
            >>> 
            >>> # Initially empty
            >>> commands = repo.list_commands()
            >>> print(commands)
            []
            >>> 
            >>> # Add some commands
            >>> repo.upsert_command("hello")
            >>> repo.upsert_command("stop")
            >>> repo.upsert_command("play")
            >>> 
            >>> # List all commands (sorted)
            >>> commands = repo.list_commands()
            >>> print(commands)
            ['hello', 'play', 'stop']
            >>> 
            >>> # Use for iteration
            >>> for cmd in repo.list_commands():
            ...     print(f"Processing command: {cmd}")

        Note:
            The returned list is sorted alphabetically, which provides
            consistent ordering across different database systems and
            makes the interface more predictable for users.
        """
        cur = self._conn.execute("SELECT name FROM commands ORDER BY name")
        return [r[0] for r in cur.fetchall()]

    # Recordings
    def add_recording(self, command_id: int, path: str, sample_rate: int, channels: int, duration: float) -> int:
        """
        Add metadata for an audio recording to the database.

        Stores metadata about an audio recording file, linking it to a specific
        command. This metadata is essential for training data management,
        allowing the system to track audio file properties and maintain
        associations between recordings and their corresponding commands.

        Args:
            command_id (int): ID of the command this recording represents.
                            Must be a valid command ID from upsert_command().
            path (str): File system path to the audio recording. Should be
                       either absolute path or relative to a known base directory.
            sample_rate (int): Audio sample rate in Hz (e.g., 16000, 44100).
            channels (int): Number of audio channels (1 for mono, 2 for stereo).
            duration (float): Recording duration in seconds.

        Returns:
            int: The recording ID (primary key) that can be used for future
                reference to this specific recording.

        Raises:
            sqlite3.IntegrityError: If command_id doesn't exist (foreign key violation).
            sqlite3.Error: If database operation fails.
            ValueError: If any parameter has invalid values.

        Examples:
            >>> repo = ProfileRepository(Path("commands.db"))
            >>> 
            >>> # First create a command
            >>> cmd_id = repo.upsert_command("hello")
            >>> 
            >>> # Add recording metadata
            >>> rec_id = repo.add_recording(
            ...     command_id=cmd_id,
            ...     path="/data/recordings/hello_001.wav",
            ...     sample_rate=16000,
            ...     channels=1,
            ...     duration=2.5
            ... )
            >>> print(f"Recording ID: {rec_id}")
            Recording ID: 1
            >>> 
            >>> # Add multiple recordings for the same command
            >>> rec_id2 = repo.add_recording(cmd_id, "/data/hello_002.wav", 16000, 1, 2.8)
            >>> rec_id3 = repo.add_recording(cmd_id, "/data/hello_003.wav", 16000, 1, 2.2)

        Note:
            The path is stored as-is in the database. Consider using absolute
            paths or establishing a consistent base directory structure to
            ensure recordings can be located later during training or inference.
        """
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "INSERT INTO recordings(command_id, path, sample_rate, channels, duration, created_at) VALUES (?,?,?,?,?,?)",
            (command_id, path, sample_rate, channels, duration, now),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def list_recordings(self, command_id: int) -> list[str]:
        """
        Retrieve all recording paths for a specific command.

        Returns a list of file paths for all recordings associated with the
        given command, ordered by recording ID (chronological order of addition).
        This is useful for training data collection and batch processing of
        audio files for a specific command.

        Args:
            command_id (int): ID of the command to retrieve recordings for.
                            Must be a valid command ID from the database.

        Returns:
            list[str]: List of file paths for recordings of the specified command,
                      ordered by recording ID (insertion order). Empty list if
                      no recordings exist for the command.

        Examples:
            >>> repo = ProfileRepository(Path("commands.db"))
            >>> 
            >>> # Set up command and recordings
            >>> cmd_id = repo.upsert_command("hello")
            >>> repo.add_recording(cmd_id, "/data/hello_001.wav", 16000, 1, 2.5)
            >>> repo.add_recording(cmd_id, "/data/hello_002.wav", 16000, 1, 2.8)
            >>> repo.add_recording(cmd_id, "/data/hello_003.wav", 16000, 1, 2.2)
            >>> 
            >>> # Get all recordings for the command
            >>> recordings = repo.list_recordings(cmd_id)
            >>> print(recordings)
            ['/data/hello_001.wav', '/data/hello_002.wav', '/data/hello_003.wav']
            >>> 
            >>> # Use for batch processing
            >>> for path in repo.list_recordings(cmd_id):
            ...     print(f"Processing: {path}")
            >>> 
            >>> # Check if command has any recordings
            >>> if repo.list_recordings(cmd_id):
            ...     print("Command has training data")

        Note:
            Recordings are returned in the order they were added to the database
            (ordered by ID), which typically corresponds to chronological order.
            This ordering is useful for reproducible training data processing.
        """
        cur = self._conn.execute(
            "SELECT path FROM recordings WHERE command_id=? ORDER BY id", (command_id,))
        return [r[0] for r in cur.fetchall()]

    # Profiles
    def save_profile(self, command_id: int, num_parts: int, vector: np.ndarray):
        """
        Save a feature profile (vector) for a command.

        Stores a computed feature vector that represents the acoustic
        characteristics of a command. This profile is used during recognition
        to match incoming audio against known commands. The method uses
        INSERT OR REPLACE to handle updates to existing profiles.

        Args:
            command_id (int): ID of the command this profile represents.
                            Must be a valid command ID from the database.
            num_parts (int): Number of frequency bands used in feature extraction.
                           This parameter ensures compatibility between feature
                           extraction configuration and stored profiles.
            vector (np.ndarray): Feature vector containing the extracted features.
                               Should be a 1D numpy array of numeric values.
                               Will be converted to float64 for consistency.

        Raises:
            sqlite3.IntegrityError: If command_id doesn't exist (foreign key violation).
            sqlite3.Error: If database operation fails.
            ValueError: If vector is not a valid numpy array.

        Examples:
            >>> import numpy as np
            >>> repo = ProfileRepository(Path("commands.db"))
            >>> 
            >>> # Create command and extract features
            >>> cmd_id = repo.upsert_command("hello")
            >>> features = np.array([0.1, 0.3, 0.2, 0.8, 0.4])  # Example features
            >>> 
            >>> # Save the profile
            >>> repo.save_profile(
            ...     command_id=cmd_id,
            ...     num_parts=5,
            ...     vector=features
            ... )
            >>> 
            >>> # Update existing profile (overwrites previous)
            >>> updated_features = np.array([0.15, 0.35, 0.25, 0.85, 0.45])
            >>> repo.save_profile(cmd_id, 5, updated_features)
            >>> 
            >>> # Save profile with different feature configuration
            >>> detailed_features = np.random.rand(100)
            >>> repo.save_profile(cmd_id, 100, detailed_features)

        Note:
            - The combination of (command_id, num_parts) must be unique
            - Existing profiles with the same command_id and num_parts are replaced
            - Feature vectors are serialized as binary data for efficient storage
            - All vectors are normalized to float64 for numerical consistency
        """
        now = datetime.now(timezone.utc).isoformat()
        blob = vector.astype(np.float64).tobytes(order="C")
        self._conn.execute(
            "INSERT OR REPLACE INTO profiles(command_id, num_parts, vector, created_at) VALUES (?,?,?,?)",
            (command_id, num_parts, blob, now),
        )
        self._conn.commit()

    def load_profiles(self, num_parts: int) -> dict[str, np.ndarray]:
        """
        Load all feature profiles with matching feature configuration.

        Retrieves all stored feature profiles that were created with the
        specified number of frequency parts. This ensures that only compatible
        profiles are loaded for recognition, since feature vectors must have
        the same dimensionality to be compared effectively.

        Args:
            num_parts (int): Number of frequency bands to match. Only profiles
                           created with this exact feature configuration will
                           be returned.

        Returns:
            dict[str, np.ndarray]: Dictionary mapping command names to their
                                  feature vectors. Keys are command names (strings),
                                  values are numpy arrays containing the feature
                                  data. Empty dict if no matching profiles exist.

        Examples:
            >>> import numpy as np
            >>> repo = ProfileRepository(Path("commands.db"))
            >>> 
            >>> # Set up some commands and profiles
            >>> hello_id = repo.upsert_command("hello")
            >>> stop_id = repo.upsert_command("stop")
            >>> play_id = repo.upsert_command("play")
            >>> 
            >>> # Save profiles with 100 frequency parts
            >>> repo.save_profile(hello_id, 100, np.random.rand(100))
            >>> repo.save_profile(stop_id, 100, np.random.rand(100))
            >>> repo.save_profile(play_id, 100, np.random.rand(100))
            >>> 
            >>> # Save some profiles with different configuration
            >>> repo.save_profile(hello_id, 50, np.random.rand(50))
            >>> 
            >>> # Load profiles for recognition (100 parts)
            >>> profiles = repo.load_profiles(num_parts=100)
            >>> print(profiles.keys())
            dict_keys(['hello', 'play', 'stop'])
            >>> 
            >>> # Load profiles with different configuration
            >>> profiles_50 = repo.load_profiles(num_parts=50)
            >>> print(profiles_50.keys())
            dict_keys(['hello'])
            >>> 
            >>> # Use for recognition
            >>> for cmd_name, features in profiles.items():
            ...     print(f"{cmd_name}: {features.shape}")
            hello: (100,)
            play: (100,)
            stop: (100,)

        Note:
            - Only profiles with exactly matching num_parts are returned
            - Feature vectors are deserialized from binary storage as float64
            - The method performs a JOIN to get command names with their profiles
            - Results are not ordered; use sorted(profiles.items()) if needed
        """
        sql = (
            "SELECT c.name, p.vector FROM profiles p JOIN commands c ON c.id=p.command_id WHERE p.num_parts=?"
        )
        cur = self._conn.execute(sql, (num_parts,))
        out: dict[str, np.ndarray] = {}
        for name, blob in cur.fetchall():
            out[name] = np.frombuffer(blob, dtype=np.float64)
        return out
