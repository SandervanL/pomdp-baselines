from typing import Optional, Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType


class ExampleEnv(gym.Env):
    def __init__(self):
        super(ExampleEnv, self).__init__()

    def get_task(self):
        """
        Return a task description, such as goal position or target velocity.
        """
        pass

    def set_goal(self, goal):
        """
        Sets goal manually. Mainly used for reward relabelling.
        """
        pass

    def reset_task(self, task=None):
        """
        Reset the task, either at random (if task=None) or the given task.
        """
        pass

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.
        Should return: state, reward, done, info
        where info has to include a field 'task'.
        """
        pass

    def reward(self, state, action):
        """
        Computes reward function of task.
        Returns the reward
        """
        pass

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> \
            tuple[np.ndarray, dict]:
        """
        Reset the environment. This should *NOT* reset the task!
        Resetting the task is handled in the varibad wrapper (see wrappers.py).
        """
        pass
