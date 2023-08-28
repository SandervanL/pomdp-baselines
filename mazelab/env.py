""" Base class for all maze environments. """
from abc import ABC
from abc import abstractmethod
from typing import Optional, Any, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.utils import seeding
from PIL import Image


class BaseEnv(gym.Env, ABC):
    """ Base class for all maze environments. """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 3
    }
    reward_range = (-float('inf'), float('inf'))

    def __init__(self):
        self.viewer = None

    @abstractmethod
    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """ Perform one step in the environment."""

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> \
            tuple[np.ndarray, dict]:
        """ Resets the environment."""
        self.np_random, _ = seeding.np_random(seed)
        return np.ndarray([]), {}

    @abstractmethod
    def get_image(self):
        """ Returns an image of the environment."""

    def render(self, mode: str = 'human', max_width: int = 500):
        """
        Renders the environment.
        Args:
            mode: the mode to render. (human, rgb_array)
            max_width: the maximum width of the image.

        Returns:
            the rendered image.s
        """
        img = self.get_image()
        img = np.asarray(img).astype(np.uint8)
        img_height, img_width = img.shape[:2]
        ratio = max_width / img_width
        img = Image.fromarray(img).resize((int(ratio * img_width), int(ratio * img_height)))
        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gymnasium.envs.classic_control.rendering import SimpleImageViewer
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)

            return self.viewer.isopen

    def close(self) -> None:
        """ Closes the environment."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
