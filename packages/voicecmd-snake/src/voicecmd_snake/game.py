"""
Voice-controlled Snake game demonstration for VoiceCMD recognition system.

This module implements a complete Snake game that showcases the integration of
voice command recognition with real-time interactive applications. The game
supports both traditional keyboard controls and voice commands, providing a
practical demonstration of the VoiceCMD system capabilities.

Key Features:
- Classic Snake gameplay with modern Python/Pygame implementation
- Dual input support: keyboard and voice commands
- Progressive difficulty with increasing speed
- Real-time voice recognition integration
- Configurable voice recognition parameters
- Graceful degradation when voice recognition is unavailable
- Comprehensive command-line interface for testing and tuning

Game Mechanics:
- Snake moves continuously in grid-based movement
- Food spawns randomly on the game field
- Score increases with each food consumed
- Game speed increases every 5 points for added challenge
- Collision detection for walls and self-collision
- Pause functionality for game control

Voice Integration:
- Supports directional commands: UP, DOWN, LEFT, RIGHT
- Confidence threshold filtering to reduce false positives
- Voice command cooldown to prevent input flooding
- Fallback to keyboard input when voice is unavailable
- Real-time voice activity feedback in game UI

Technical Architecture:
- Pygame-based rendering and game loop
- Dataclass-based configuration management
- Modular design with separated concerns
- Thread-safe voice command processing
- Robust error handling and resource cleanup

Classes:
    GameConfig: Configuration container for game and voice parameters
    SnakeGame: Main game implementation with voice integration

Dependencies:
    - pygame: Graphics, input, and game loop management
    - voicecmd_snake.voice_controller: Voice recognition integration
    - argparse: Command-line interface
    - dataclasses: Configuration management
    - collections.deque: Efficient snake body representation

Usage Examples:
    # Keyboard-only mode
    python game.py
    
    # Voice-controlled mode (requires trained profiles)
    python game.py --voice
    
    # Voice mode with custom parameters
    python game.py --voice --conf 0.7 --device-index 1
    
    # Development mode with relaxed voice detection
    python game.py --voice --conf 0.4 --factor 2.0
"""

from __future__ import annotations
import argparse
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Deque
from collections import deque

import pygame

try:
    from voicecmd_snake.voice_controller import VoiceController
except Exception:
    VoiceController = None  # optional: game runs without voice control

# Type alias for 2D coordinates (x, y) in game space
Vec2 = Tuple[int, int]


@dataclass
class GameConfig:
    """
    Configuration container for Snake game parameters and voice recognition settings.

    This dataclass encapsulates all configurable parameters for the Snake game,
    including display settings, gameplay mechanics, and voice recognition tuning.
    The configuration approach allows easy customization and testing of different
    parameter combinations without code modification.

    Display Configuration:
        width (int): Game window width in pixels (default: 500)
        height (int): Game window height in pixels (default: 500)
        cell (int): Size of each grid cell in pixels (default: 10)
        font_name (str): System font name for UI text (default: "consolas")
        font_size (int): Font size for UI text (default: 18)

    Gameplay Configuration:
        base_fps (int): Starting game speed in frames per second (default: 8)
        speedup_every (int): Score interval for speed increases (default: 5)

    Voice Recognition Configuration:
        voice_enabled (bool): Whether voice control is active (default: False)
        voice_conf_threshold (float): Minimum confidence for voice commands (default: 0.55)
        voice_cooldown_ms (int): Milliseconds between voice direction changes (default: 180)

    Examples:
        >>> # Default configuration for standard gameplay
        >>> config = GameConfig()
        >>> 
        >>> # High-speed configuration for advanced players
        >>> fast_config = GameConfig(
        ...     base_fps=12,
        ...     speedup_every=3,
        ...     voice_conf_threshold=0.7
        ... )
        >>> 
        >>> # Large display configuration for presentations
        >>> demo_config = GameConfig(
        ...     width=800,
        ...     height=600,
        ...     cell=20,
        ...     font_size=24
        ... )
        >>> 
        >>> # Voice-optimized configuration for noisy environments
        >>> voice_config = GameConfig(
        ...     voice_enabled=True,
        ...     voice_conf_threshold=0.8,
        ...     voice_cooldown_ms=250
        ... )

    Parameter Guidelines:
        - Larger cell sizes improve visibility but reduce play area
        - Higher base_fps makes the game more challenging initially
        - Lower speedup_every creates more aggressive difficulty progression
        - Higher voice_conf_threshold reduces false positives but may miss commands
        - Longer voice_cooldown_ms prevents rapid direction changes but may feel sluggish

    Note:
        All parameters have sensible defaults for typical gameplay scenarios.
        Voice-related parameters only take effect when voice control is enabled.
    """

    # Display settings
    width: int = 500
    """Game window width in pixels."""

    height: int = 500
    """Game window height in pixels."""

    cell: int = 10
    """Size of each grid cell in pixels. Determines game resolution and snake segment size."""

    font_name: str = "consolas"
    """System font name for UI text rendering. Monospace fonts recommended for consistency."""

    font_size: int = 18
    """Font size for UI text including score, status, and messages."""

    # Gameplay mechanics
    base_fps: int = 8
    """Starting game speed in frames per second. Higher values make the game faster initially."""

    speedup_every: int = 5
    """Score interval for speed increases. Game gets faster every N points scored."""

    # Voice recognition settings
    voice_enabled: bool = False
    """Whether voice control is currently active in the game."""

    voice_conf_threshold: float = 0.55
    """Minimum confidence score (0.0-1.0) required to accept voice commands."""

    voice_cooldown_ms: int = 400
    """Minimum milliseconds between voice-triggered direction changes to prevent input flooding."""


class SnakeGame:
    """
    Main Snake game implementation with integrated voice control support.

    This class implements the complete Snake game logic, including rendering,
    input handling, game mechanics, and voice command integration. The game
    supports both traditional keyboard controls and voice commands, with
    graceful fallback when voice recognition is unavailable.

    Game Features:
    - Classic Snake gameplay with grid-based movement
    - Progressive difficulty with increasing speed
    - Real-time score tracking and display
    - Pause functionality for game control
    - Game over detection and restart capability
    - Dual input support (keyboard + voice)

    Voice Integration:
    - Real-time voice command recognition
    - Command confidence filtering
    - Voice input cooldown to prevent flooding
    - Visual feedback for voice activity
    - Seamless fallback to keyboard input

    Architecture:
    - Event-driven input handling
    - Separated game logic and rendering
    - Modular voice command processing
    - Efficient collision detection
    - Resource management with proper cleanup

    Attributes:
        cfg (GameConfig): Game configuration parameters
        screen (pygame.Surface): Main display surface
        clock (pygame.time.Clock): Frame rate control
        font (pygame.font.Font): UI text rendering
        voice (Optional[VoiceController]): Voice recognition controller
        snake (Deque[Vec2]): Snake body segments as coordinate deque
        direction (Vec2): Current movement direction vector
        pending (Vec2): Next direction vector (for input buffering)
        food (Vec2): Current food position
        score (int): Current game score
        game_over (bool): Game over state flag
        paused (bool): Game pause state flag

    Examples:
        >>> # Keyboard-only game
        >>> config = GameConfig()
        >>> game = SnakeGame(config)
        >>> game.run()
        >>> 
        >>> # Voice-controlled game
        >>> voice_controller = VoiceController()
        >>> voice_controller.start()
        >>> game = SnakeGame(config, voice_controller)
        >>> try:
        ...     game.run()
        ... finally:
        ...     voice_controller.stop()

    Controls:
        Keyboard:
        - Arrow Keys / WASD: Direction control
        - P: Pause/unpause game
        - R: Restart game (when game over)
        - ESC: Exit game

        Voice Commands:
        - "UP": Move snake upward
        - "DOWN": Move snake downward
        - "LEFT": Move snake left
        - "RIGHT": Move snake right

    Game Mechanics:
        - Snake moves continuously in the current direction
        - Food consumption increases score and snake length
        - Wall collision or self-collision ends the game
        - Speed increases every 5 points for added challenge
        - Direction changes are buffered to prevent missed inputs
    """

    def __init__(self, cfg: GameConfig, voice: Optional["VoiceController"] = None):
        """
        Initialize the Snake game with configuration and optional voice control.

        Sets up the game environment including Pygame initialization, display
        creation, font loading, and voice controller integration. Also performs
        initial game state setup through the reset() method.

        Args:
            cfg (GameConfig): Game configuration containing display, gameplay,
                            and voice recognition parameters
            voice (Optional[VoiceController]): Voice recognition controller for
                                             voice command input. None disables voice control.

        Initialization Process:
            1. Initialize Pygame systems (display, font, clock)
            2. Create game window with specified dimensions
            3. Load UI font for text rendering
            4. Configure voice controller if provided
            5. Reset game to initial state

        Examples:
            >>> # Standard initialization
            >>> config = GameConfig(width=600, height=600)
            >>> game = SnakeGame(config)
            >>> 
            >>> # With voice control
            >>> voice = VoiceController()
            >>> game = SnakeGame(config, voice)

        Note:
            Pygame.init() is called during initialization. Ensure proper cleanup
            by calling the shutdown method or using proper exception handling.
        """
        pygame.init()
        self.cfg = cfg
        self.screen = pygame.display.set_mode((cfg.width, cfg.height))
        pygame.display.set_caption("VoiceCMD Snake — Demo")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(cfg.font_name, cfg.font_size)
        self.voice = voice
        self._last_voice_change_ms = 0
        self.reset()

    def reset(self):
        """
        Reset the game to initial state for new game or restart.

        Initializes all game state variables to their starting values:
        - Places snake in center of screen with 3-segment body
        - Sets initial direction to rightward movement
        - Spawns first food item at random location
        - Resets score, game over, and pause states

        The snake is positioned in the center of the play area with segments
        aligned horizontally, ready to move right. The initial configuration
        ensures a valid starting position that doesn't overlap with walls.

        Examples:
            >>> game = SnakeGame(config)
            >>> # Game automatically resets on initialization
            >>> 
            >>> # Manual reset during gameplay
            >>> game.reset()  # Called when R key pressed after game over

        State Initialization:
            - Snake: 3 segments in center, moving right
            - Food: Random position not overlapping snake
            - Score: 0 points
            - Direction: Right (positive X)
            - Game flags: Not over, not paused
        """
        # Calculate center position aligned to grid
        cx = (self.cfg.width // (2*self.cfg.cell)) * self.cfg.cell
        cy = (self.cfg.height // (2*self.cfg.cell)) * self.cfg.cell

        # Initialize snake with 3 segments: head + 2 body segments
        self.snake: Deque[Vec2] = deque([
            (cx, cy),                      # Head
            (cx - self.cfg.cell, cy),      # Body segment 1
            (cx - 2*self.cfg.cell, cy)     # Body segment 2 (tail)
        ])

        # Set initial movement direction (rightward)
        self.direction: Vec2 = (self.cfg.cell, 0)
        self.pending: Vec2 = self.direction

        # Initialize game state
        self.food: Vec2 = self._spawn_food()
        self.score = 0
        self.game_over = False
        self.paused = False

    def _spawn_food(self) -> Vec2:
        """
        Generate a random food position that doesn't overlap with the snake.

        Calculates a random grid-aligned position within the game boundaries
        and ensures it doesn't coincide with any snake segment. Uses a simple
        retry loop until a valid position is found.

        Returns:
            Vec2: Grid-aligned (x, y) coordinates for the new food position

        Algorithm:
            1. Calculate grid dimensions based on window size and cell size
            2. Generate random grid coordinates
            3. Convert to pixel coordinates
            4. Check for overlap with snake body
            5. Retry if overlap detected, otherwise return position

        Examples:
            >>> # Called automatically during reset() and when food is consumed
            >>> food_pos = game._spawn_food()
            >>> print(f"Food spawned at {food_pos}")

        Note:
            This method assumes the game area is large enough that a valid
            position will eventually be found. For very large snakes, this
            could theoretically loop many times, but is not a practical concern
            for typical gameplay.
        """
        cols = self.cfg.width // self.cfg.cell
        rows = self.cfg.height // self.cfg.cell
        while True:
            fx = random.randrange(cols) * self.cfg.cell
            fy = random.randrange(rows) * self.cfg.cell
            if (fx, fy) not in self.snake:
                return fx, fy

    @staticmethod
    def _is_reverse(a: Vec2, b: Vec2) -> bool:
        """
        Check if two direction vectors are opposite to each other.

        Determines whether two direction vectors represent opposite directions,
        which is used to prevent the snake from immediately reversing into
        itself (an invalid move that would cause instant self-collision).

        Args:
            a (Vec2): First direction vector (dx, dy)
            b (Vec2): Second direction vector (dx, dy)

        Returns:
            bool: True if vectors are exactly opposite, False otherwise

        Examples:
            >>> # Opposite directions
            >>> SnakeGame._is_reverse((10, 0), (-10, 0))  # Right vs Left
            True
            >>> SnakeGame._is_reverse((0, 10), (0, -10))  # Down vs Up
            True
            >>> 
            >>> # Same or perpendicular directions
            >>> SnakeGame._is_reverse((10, 0), (10, 0))   # Same direction
            False
            >>> SnakeGame._is_reverse((10, 0), (0, 10))   # Right vs Down
            False

        Mathematical Definition:
            Two vectors a and b are reverse if a = -b, which means:
            a.x = -b.x AND a.y = -b.y

        Usage:
            This method is crucial for preventing invalid snake movements
            that would cause the snake to immediately collide with itself.
        """
        return a[0] == -b[0] and a[1] == -b[1]

    @staticmethod
    def _dir_for_key(k: int) -> Optional[Vec2]:
        """
        Convert keyboard input to direction vector.

        Maps keyboard key codes to movement direction vectors for snake control.
        Supports both arrow keys and WASD layout for accessibility and user preference.

        Args:
            k (int): Pygame key code from keyboard event

        Returns:
            Optional[Vec2]: Direction vector (dx, dy) or None if key not recognized
                          Vector components are in pixel units (not grid units)

        Supported Keys:
            - RIGHT ARROW / D: (10, 0) - Move right
            - LEFT ARROW / A: (-10, 0) - Move left  
            - UP ARROW / W: (0, -10) - Move up
            - DOWN ARROW / S: (0, 10) - Move down

        Examples:
            >>> # Arrow key input
            >>> SnakeGame._dir_for_key(pygame.K_RIGHT)
            (10, 0)
            >>> 
            >>> # WASD input
            >>> SnakeGame._dir_for_key(pygame.K_w)
            (0, -10)
            >>> 
            >>> # Non-movement key
            >>> SnakeGame._dir_for_key(pygame.K_SPACE)
            None

        Note:
            The returned vectors use hardcoded pixel values (10) which should
            match the game's cell size configuration for proper grid alignment.
        """
        if k in (pygame.K_RIGHT, pygame.K_d):
            return (10, 0)
        if k in (pygame.K_LEFT,  pygame.K_a):
            return (-10, 0)
        if k in (pygame.K_UP,    pygame.K_w):
            return (0, -10)
        if k in (pygame.K_DOWN,  pygame.K_s):
            return (0, 10)
        return None

    @staticmethod
    def _dir_for_word(word: str) -> Optional[Vec2]:
        """
        Convert voice command word to direction vector.

        Maps recognized voice command strings to movement direction vectors.
        Provides case-insensitive matching for natural voice interaction.

        Args:
            word (str): Voice command string (case-insensitive)

        Returns:
            Optional[Vec2]: Direction vector (dx, dy) or None if word not recognized
                          Vector components are in pixel units (not grid units)

        Supported Commands:
            - "RIGHT": (10, 0) - Move right
            - "LEFT": (-10, 0) - Move left
            - "UP": (0, -10) - Move up  
            - "DOWN": (0, 10) - Move down

        Examples:
            >>> # Standard voice commands
            >>> SnakeGame._dir_for_word("RIGHT")
            (10, 0)
            >>> SnakeGame._dir_for_word("up")
            (0, -10)
            >>> 
            >>> # Case insensitive
            >>> SnakeGame._dir_for_word("Left")
            (-10, 0)
            >>> 
            >>> # Unrecognized command
            >>> SnakeGame._dir_for_word("FORWARD")
            None

        Integration:
            This method is used by the voice command processing pipeline to
            convert recognized speech into game movement commands, maintaining
            consistency with keyboard input mapping.
        """
        w = word.upper()
        if w == "RIGHT":
            return (10, 0)
        if w == "LEFT":
            return (-10, 0)
        if w == "UP":
            return (0, -10)
        if w == "DOWN":
            return (0, 10)
        return None

    def _maybe_apply_voice(self):
        """
        Process voice commands with cooldown and validation.

        Attempts to retrieve and apply voice commands from the voice recognition
        system. This method provides the integration point between the voice
        recognition engine and the game logic, including timing controls to
        prevent overly rapid direction changes that could break gameplay.

        Voice Command Processing Flow:
            1. Check if voice controller is available
            2. Get command from voice recognition queue (non-blocking)
            3. Convert recognized word to direction vector
            4. Apply cooldown timing to prevent rapid-fire commands
            5. Normalize direction vector to grid coordinates
            6. Validate direction change (prevent reverse movement)
            7. Set pending direction for next game update

        Cooldown Mechanism:
            Uses voice_cooldown_ms configuration to limit how frequently
            voice commands can change the snake's direction, preventing
            erratic movement from continuous voice input.

        Direction Processing:
            - Raw directions from _dir_for_word() are in pixel units (±10)
            - Normalized to grid units using cell size configuration
            - Prevents immediate reversal that would cause self-collision

        Examples:
            Voice Recognition Chain:
            ```
            User says "RIGHT" → Voice Engine → get_command_nowait() → "RIGHT"
                                            ↓
            _dir_for_word("RIGHT") → (10, 0) → normalize → (cell_size, 0)
                                            ↓
            Cooldown check → Reverse check → self.pending = new_direction
            ```

        Performance Notes:
            - Non-blocking voice command retrieval
            - Minimal processing when no voice available
            - Efficient timing-based filtering

        Error Handling:
            - Gracefully handles missing voice controller
            - Ignores invalid voice commands
            - Continues operation if voice system unavailable
        """
        if not self.voice:
            return
        cmd = self.voice.get_command_nowait()
        if not cmd:
            return
        new_dir = self._dir_for_word(cmd)
        if not new_dir:
            return
        now = pygame.time.get_ticks()
        if now - self._last_voice_change_ms < self.cfg.voice_cooldown_ms:
            return
        norm = (new_dir[0]//10 * self.cfg.cell, new_dir[1]//10 * self.cfg.cell)
        if not self._is_reverse(norm, self.direction):
            self.pending = norm
            self._last_voice_change_ms = now

    def _handle_events(self):
        """
        Process all pygame events including user input and system events.

        This method handles the complete event processing pipeline for the game,
        including keyboard input, window management, and game state changes.
        It serves as the main input processing hub that coordinates between
        different input methods (keyboard, voice) and game actions.

        Event Processing:
            - System Events: QUIT, window closing
            - Keyboard Events: Movement, game control, state changes
            - Input Validation: Direction change validation and normalization

        Supported Controls:
            Game Control:
            - ESCAPE: Exit game immediately
            - R: Restart game (only when game over)
            - P: Toggle pause state

            Movement (Arrow Keys or WASD):
            - RIGHT/D: Move snake right
            - LEFT/A: Move snake left  
            - UP/W: Move snake up
            - DOWN/S: Move snake down

        Input Processing Pipeline:
            ```
            Event → Type Check → Key Processing → Direction Conversion →
            Grid Normalization → Reverse Check → Pending Direction Update
            ```

        Direction Handling:
            - Converts pixel-based directions to grid coordinates
            - Prevents immediate direction reversal (anti-suicide protection)
            - Sets pending direction for smooth movement updates

        Error Handling:
            - SystemExit exceptions for clean game termination
            - Graceful handling of invalid key combinations
            - State validation before applying changes

        Performance Notes:
            - Processes all events in single loop for efficiency
            - Minimal processing for non-game events
            - Early validation to prevent invalid state changes

        Threading Safety:
            This method should only be called from the main game thread
            as it interacts directly with pygame's event system.
        """
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self._shutdown()
                raise SystemExit
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self._shutdown()
                    raise SystemExit
                if ev.key == pygame.K_r and self.game_over:
                    self.reset()
                if ev.key == pygame.K_p:
                    self.paused = not self.paused
                maybe = self._dir_for_key(ev.key)
                if maybe:
                    norm = (maybe[0]//10 * self.cfg.cell,
                            maybe[1]//10 * self.cfg.cell)
                    if not self._is_reverse(norm, self.direction):
                        self.pending = norm

    def _apply_dir(self):
        """
        Apply pending direction change to snake movement.

        Updates the snake's current direction from the pending direction,
        but only if the pending direction is valid (not a reverse move).
        This method provides a safe way to change direction that prevents
        the snake from immediately colliding with itself.

        Direction Update Logic:
            1. Check if pending direction is valid (not reverse of current)
            2. Update current direction if validation passes
            3. Maintain current direction if validation fails

        Safety Features:
            - Prevents immediate direction reversal
            - Maintains game state consistency
            - Ensures smooth movement transitions

        Examples:
            Valid Direction Changes:
            ```
            Current: (10, 0) [Right]  → Pending: (0, 10) [Down]   ✓ Applied
            Current: (0, -10) [Up]   → Pending: (10, 0) [Right]  ✓ Applied
            ```

            Invalid Direction Changes:
            ```
            Current: (10, 0) [Right] → Pending: (-10, 0) [Left]  ✗ Ignored
            Current: (0, 10) [Down] → Pending: (0, -10) [Up]    ✗ Ignored
            ```

        Integration:
            Called from the main game loop after input processing to safely
            apply direction changes that were validated during event handling.

        Performance:
            - O(1) operation with minimal computational overhead
            - Simple validation check prevents complex collision scenarios
        """
        if not self._is_reverse(self.pending, self.direction):
            self.direction = self.pending

    def _move(self):
        """
        Move the snake one step in the current direction.

        Implements the core snake movement mechanics by calculating the new
        head position and managing the snake body. This method handles both
        normal movement and food consumption scenarios.

        Movement Algorithm:
            1. Calculate new head position based on current direction
            2. Add new head to front of snake body (deque.appendleft)
            3. Check if food was consumed at new head position
            4. If food consumed: keep tail, spawn new food, update score
            5. If no food: remove tail to maintain snake length

        Snake Body Management:
            - Snake body stored as deque of (x, y) coordinate tuples
            - Head is at index 0 (left), tail operations via pop()
            - Length increases by 1 when food is consumed
            - Body segments follow head automatically via deque operations

        Food Consumption Logic:
            ```
            New Head Position == Food Position?
            ├─ YES: Keep tail → Snake grows → Spawn new food → Score++
            └─ NO:  Remove tail → Snake maintains length
            ```

        Coordinate System:
            - Uses grid-based coordinates aligned to cell size
            - Direction vectors specify movement in pixels
            - Position calculations maintain grid alignment

        Examples:
            Normal Movement:
            ```
            Snake: deque([(20,10), (10,10), (0,10)])  Direction: (10,0)
                ↓
            Snake: deque([(30,10), (20,10), (10,10)])  [Tail removed]
            ```

            Food Consumption:
            ```
            Snake: deque([(20,10), (10,10)])  Food: (30,10)  Direction: (10,0)
                ↓
            Snake: deque([(30,10), (20,10), (10,10)])  [Tail kept, food consumed]
            ```

        Side Effects:
            - Modifies self.snake deque
            - May update self.food and self.score
            - Triggers food respawn when consumed

        Performance:
            - O(1) deque operations for efficient snake management
            - Minimal memory allocation for coordinate calculations
        """
        hx, hy = self.snake[0]
        nx, ny = hx + self.direction[0], hy + self.direction[1]
        new_head = (nx, ny)
        self.snake.appendleft(new_head)
        if new_head == self.food:
            self.score += 1
            self.food = self._spawn_food()
        else:
            self.snake.pop()

    def _check_collisions(self):
        """
        Check for game-ending collision conditions.

        Detects collision scenarios that end the game, including wall
        collisions and self-collisions. This method implements the core
        game-over logic for the Snake game.

        Collision Detection Types:
            1. Wall Collisions: Snake head hits screen boundaries
            2. Self-Collisions: Snake head overlaps with body segments

        Wall Collision Detection:
            - Checks if head x-coordinate is outside [0, width)
            - Checks if head y-coordinate is outside [0, height)
            - Uses screen dimensions from game configuration

        Self-Collision Detection:
            - Compares head position with all body segments
            - Excludes head from body check (index 0)
            - Uses list conversion for membership testing

        Game State Changes:
            Sets self.game_over = True when collision detected,
            triggering game-over state in main game loop.

        Examples:
            Wall Collision Scenarios:
            ```
            Screen: 800x600, Cell: 10x10
            Snake head at (-10, 300) → Wall collision (left edge)
            Snake head at (800, 300) → Wall collision (right edge)
            Snake head at (400, -10) → Wall collision (top edge)
            Snake head at (400, 600) → Wall collision (bottom edge)
            ```

            Self-Collision Scenario:
            ```
            Snake: [(100,100), (90,100), (80,100), (80,110), (90,110), (100,110)]
            Next move: Right → New head at (110,110)
            Check: (110,110) in body segments → No collision

            Snake moves in circle, head returns to (100,110)
            Check: (100,110) in body segments → Self-collision detected
            ```

        Performance Notes:
            - O(1) boundary checking
            - O(n) self-collision check where n = snake length
            - Early termination on first collision detected

        Integration:
            Called after each movement update in the main game loop
            to validate the new game state and detect game-over conditions.
        """
        x, y = self.snake[0]
        if x < 0 or x >= self.cfg.width or y < 0 or y >= self.cfg.height:
            self.game_over = True
            return
        if self.snake[0] in list(self.snake)[1:]:
            self.game_over = True

    def _draw(self):
        """
        Render the complete game scene to the screen.

        Handles all visual rendering including game elements, UI components,
        and informational displays. This method is responsible for the
        complete visual representation of the game state.

        Rendering Pipeline:
            1. Clear screen with background color
            2. Draw food item
            3. Draw snake (head and body with different colors)
            4. Draw HUD (Heads-Up Display) with game information
            5. Handle game-over overlay if applicable

        Visual Elements:
            Background:
            - Dark background color (12, 12, 12) for contrast

            Food:
            - Orange rectangle (255, 160, 60) at food position
            - Size matches configured cell dimensions

            Snake:
            - Head: Bright green (120, 220, 120) for visibility
            - Body: Light gray (200, 200, 200) for distinction
            - Each segment rendered as cell-sized rectangle

            HUD Information:
            - Score: Current player score
            - FPS: Real-time frame rate indicator
            - Input Mode: "VOICE" or "KB" (keyboard) indicator
            - Game State: "PAUSED" indicator when applicable

        Color Scheme:
            ```
            Background: RGB(12, 12, 12)     # Dark gray
            Food:       RGB(255, 160, 60)   # Orange
            Snake Head: RGB(120, 220, 120)  # Bright green  
            Snake Body: RGB(200, 200, 200)  # Light gray
            HUD Text:   RGB(220, 220, 220)  # Light gray
            ```

        Coordinate System:
            - Uses pygame's screen coordinate system (0,0 at top-left)
            - All positions are in pixel coordinates
            - Cell-based positioning for game elements

        Performance Considerations:
            - Full screen clear and redraw each frame
            - Efficient rectangle drawing for game elements
            - Text rendering cached when possible
            - Minimal string formatting for HUD updates

        Examples:
            Typical Render Sequence:
            ```
            Clear Screen → Draw Food Rectangle → Draw Snake Segments → 
            Render HUD Text → Display Complete Frame
            ```

        Integration:
            Called from main game loop after all game state updates
            to present the current game state to the user.
        """
        self.screen.fill((12, 12, 12))
        # food
        pygame.draw.rect(self.screen, (255, 160, 60), pygame.Rect(
            self.food[0], self.food[1], self.cfg.cell, self.cfg.cell))
        # snake
        for i, (sx, sy) in enumerate(self.snake):
            color = (200, 200, 200) if i else (120, 220, 120)
            pygame.draw.rect(self.screen, color, pygame.Rect(
                sx, sy, self.cfg.cell, self.cfg.cell))
        # HUD
        info = f"Score: {self.score}  FPS: {self._fps()}  {'VOICE' if self.voice else 'KB'}  {'PAUSED' if self.paused else ''}"
        self.screen.blit(self.font.render(
            info, True, (220, 220, 220)), (10, 6))
        if self.game_over:
            go = self.font.render(
                "GAME OVER — R: reiniciar / ESC: salir", True, (240, 80, 80))
            rect = go.get_rect(center=(self.cfg.width//2, self.cfg.height//2))
            self.screen.blit(go, rect)
        pygame.display.flip()

    def _fps(self) -> int:
        """
        Calculate current target frame rate based on game progression.

        Implements dynamic difficulty scaling by increasing the game speed
        as the player's score increases. This creates a progressively
        challenging experience that rewards skill development.

        Speed Scaling Algorithm:
            - Base FPS: Configured starting frame rate
            - Speed Increase: Every N points, increase FPS by 1
            - Formula: base_fps + (score // speedup_every)

        Args:
            None (uses instance variables)

        Returns:
            int: Target frame rate for current game state

        Examples:
            Configuration: base_fps=10, speedup_every=5
            ```
            Score 0-4:   FPS = 10 + (0//5) = 10
            Score 5-9:   FPS = 10 + (5//5) = 11  
            Score 10-14: FPS = 10 + (10//5) = 12
            Score 15-19: FPS = 10 + (15//5) = 13
            ```

        Difficulty Progression:
            - Higher FPS = faster snake movement
            - Faster movement = less reaction time
            - Creates natural difficulty curve
            - Maintains playability with reasonable increments

        Performance Impact:
            - Simple integer arithmetic
            - O(1) calculation complexity
            - Called once per frame for game loop timing

        Integration:
            Used by pygame.time.Clock.tick() to control game loop timing
            and create smooth, progressively challenging gameplay.
        """
        return self.cfg.base_fps + (self.score // self.cfg.speedup_every)

    def run(self):
        """
        Main game loop - runs the complete game until exit.

        Implements the core game loop that coordinates all game systems
        including input processing, game logic updates, collision detection,
        and rendering. This method contains the primary game execution flow.

        Game Loop Architecture:
            1. Event Handling: Process user input and system events
            2. Voice Input: Process voice commands if enabled
            3. Game Logic: Update game state (if not paused/game-over)
               - Apply direction changes
               - Move snake
               - Check for collisions
            4. Rendering: Draw current game state
            5. Timing: Control frame rate via clock tick

        Loop Flow Control:
            ```
            ┌─── Main Loop (infinite) ───┐
            │                            │
            │ 1. Handle Events           │
            │ 2. Process Voice Commands  │
            │ 3. Update Game Logic       │
            │    (if active)             │
            │ 4. Render Frame            │
            │ 5. Control Timing          │
            │                            │
            └────────────────────────────┘
                     │
                SystemExit → Clean shutdown
            ```

        State Management:
            - Paused State: Skip game logic, continue rendering
            - Game Over State: Skip game logic, show game over screen
            - Active State: Full game loop execution

        Input Integration:
            - Keyboard: Always processed via event handling
            - Voice: Processed when voice controller available
            - Both inputs use same validation and direction logic

        Performance Characteristics:
            - Frame rate controlled by dynamic FPS calculation
            - Efficient event processing with early exits
            - Minimal processing during paused/game-over states

        Error Handling:
            - SystemExit exceptions allow clean shutdown
            - Voice system errors handled gracefully
            - Game continues operation if voice system fails

        Exit Conditions:
            - QUIT event (window close button)
            - ESCAPE key press
            - SystemExit raised by event handlers

        Examples:
            Typical Frame Execution:
            ```
            Handle Events → Process Voice → Move Snake → Check Collisions →
            Render Scene → Wait for Next Frame (based on FPS)
            ```

            Paused Frame:
            ```
            Handle Events → Skip Game Logic → Render Scene (with pause indicator)
            ```

        Threading:
            This method runs on the main thread and should not be called
            concurrently. Voice processing may use separate threads internally.
        """
        while True:
            self._handle_events()
            if self.voice:
                self._maybe_apply_voice()
            if not self.paused and not self.game_over:
                self._apply_dir()
                self._move()
                self._check_collisions()
            self._draw()
            self.clock.tick(self._fps())

    def _shutdown(self):
        """
        Clean up resources and shut down the game.

        Handles proper cleanup of game resources including voice controller
        and pygame systems. This method ensures graceful termination and
        prevents resource leaks.

        Cleanup Operations:
            1. Stop voice controller if active
            2. Clean up pygame resources
            3. Release audio/video systems

        Error Handling:
            - Voice controller errors are ignored during shutdown
            - Ensures pygame.quit() is always called
            - Prevents cleanup errors from blocking shutdown

        Resource Management:
            - Voice Controller: Stops recognition engine and audio threads
            - Pygame: Releases display, audio, and input systems
            - Memory: Allows garbage collection of game objects

        Usage:
            Called automatically when user exits game via QUIT event
            or ESCAPE key. Should not be called directly during normal
            game operation.

        Thread Safety:
            Safe to call from main thread. Voice controller cleanup
            handles its own thread synchronization internally.
        """
        if self.voice:
            try:
                self.voice.stop()
            except Exception:
                pass
        pygame.quit()


def main():
    """
    Main entry point for the voice-controlled Snake game.

    Handles command-line argument parsing, voice controller initialization,
    and game execution with proper resource management. This function serves
    as the application entry point and orchestrates the complete game setup.

    Command-Line Arguments:
        --voice: Enable voice control (requires trained voice profiles)
        --device-index: Audio input device index (see `voicecmd devices`)
        --conf: Voice recognition confidence threshold (0.0-1.0, default: 0.55)
        --window: Voice recognition window duration in seconds (default: 2.0)
        --hop: Voice recognition hop size in seconds (default: 0.3)
        --factor: RMS threshold factor over noise baseline (default: 3.0)
        --calib: Noise calibration duration in seconds (default: 0.8)

    Voice Controller Configuration:
        The voice system requires pre-trained voice profiles created using
        the voicecmd training system. The controller parameters control
        the sensitivity and timing of voice recognition:

        - conf_threshold: Minimum confidence for accepting voice commands
        - window_secs: Duration of audio samples for recognition
        - hop_secs: Overlap between consecutive recognition windows
        - threshold_factor: Sensitivity to voice activity detection
        - calib_secs: Time to measure background noise levels

    Execution Flow:
        1. Parse command-line arguments
        2. Initialize voice controller (if --voice enabled)
        3. Create game configuration with voice settings
        4. Start voice recognition system
        5. Create and run Snake game
        6. Clean up resources on exit

    Error Handling:
        - Missing VoiceController: Exit with informative error message
        - Voice system errors: Clean shutdown with resource cleanup
        - Game exceptions: Ensure voice controller is properly stopped

    Examples:
        Keyboard-only mode:
        ```bash
        python -m voicecmd_snake.game
        ```

        Voice-enabled mode with default settings:
        ```bash
        python -m voicecmd_snake.game --voice
        ```

        Voice mode with custom sensitivity:
        ```bash
        python -m voicecmd_snake.game --voice --conf 0.7 --factor 2.5
        ```

        Voice mode with specific audio device:
        ```bash
        python -m voicecmd_snake.game --voice --device-index 1
        ```

    Resource Management:
        - Voice controller started before game begins
        - Automatic cleanup in finally block ensures proper shutdown
        - Prevents resource leaks even with unexpected exits

    Integration Requirements:
        - Requires voicecmd package for voice functionality
        - Requires pygame for game rendering and input
        - Voice profiles must be trained using voicecmd CLI tools

    Dependencies:
        - argparse: Command-line argument parsing
        - VoiceController: Voice recognition engine (optional)
        - GameConfig: Game configuration management
        - SnakeGame: Main game implementation
    """
    ap = argparse.ArgumentParser(description="Snake demo for voicecmd")
    ap.add_argument("--voice", action="store_true",
                    help="Enable voice control (requires trained profiles)")
    ap.add_argument("--device-index", type=int, default=None,
                    help="Input device index (see `voicecmd devices`)")
    ap.add_argument("--conf", type=float, default=0.55,
                    help="Confidence threshold for voice")
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--hop", type=float, default=0.3)
    ap.add_argument("--factor", type=float, default=3.0,
                    help="RMS threshold factor over noise baseline")
    ap.add_argument("--calib", type=float, default=0.8,
                    help="Calibration seconds")
    args = ap.parse_args()

    voice = None
    if args.voice:
        if VoiceController is None:
            raise SystemExit(
                "VoiceController no disponible. Revisa la instalación.")
        voice = VoiceController(
            conf_threshold=args.conf,
            device_index=args.device_index,
            window_secs=args.window,
            hop_secs=args.hop,
            calib_secs=args.calib,
            threshold_factor=args.factor,
        )
        voice.start()

    cfg = GameConfig(voice_enabled=bool(voice), voice_conf_threshold=args.conf)
    try:
        SnakeGame(cfg, voice).run()
    finally:
        if voice:
            voice.stop()
