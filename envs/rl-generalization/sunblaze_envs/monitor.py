import json
import os
from typing import Optional, Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType


class MonitorParameters(gym.Wrapper):
    """Environment wrapper which records all environment parameters."""

    current_parameters = None

    def __init__(self, env, output_filename):
        """
        Construct parameter monitor wrapper.

        :param env: Wrapped environment
        :param output_filename: Output log filename
        """
        self._output_filename = output_filename
        with open(output_filename, "w"):
            # Truncate output file.
            pass

        super(MonitorParameters, self).__init__(env)

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        result = self.env.step(action)
        self.record_parameters()
        return result

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> \
            tuple[np.ndarray, dict]:
        result = self.env.reset(seed=seed, options=options)
        self.record_parameters()
        return result

    def record_parameters(self):
        """Record current environment parameters."""
        if not hasattr(self.env.unwrapped, "parameters"):
            return
        if self.env.unwrapped.parameters == self.current_parameters:
            return

        # Record parameter set in output file.
        self.current_parameters = self.env.unwrapped.parameters
        with open(self._output_filename, "a") as output_file:
            output_file.write(json.dumps(self.current_parameters))
            output_file.write("\n")
