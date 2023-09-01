from dataclasses import dataclass
from dataclasses import field

import numpy as np


@dataclass
class Object:
    """
    Defines an object with some of its properties.
    An object can be an obstacle, free space or food etc.
    It can also have properties like impassable, positions.
    """
    name: str
    value: int
    rgb: tuple
    impassable: bool
    positions: np.ndarray = field(default_factory=lambda: np.ndarray(0))
