from dataclasses import dataclass

from torch import Tensor


@dataclass
class MazeTask:
    """Maze task class."""

    embedding: Tensor
    blocked: bool
    task_type: int  # unique number for each type (high vs low, heavy vs light, etc)
    word: str
    sentence: str
    object_type: str
    negation: bool
    direction: str

    # Directions of the map hallways
    short_direction: int
    short_hook_direction: int
    long_direction: int
    long_hook_direction: int
