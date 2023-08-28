""" Class that rescales the state of a wrapped environment to [-1, 1]. """
from typing import Optional, Tuple, Any

import numpy as np
from gymnasium import Env
from gymnasium.core import ObsType


class RescaledEnv(Env):
    """ Environment that rescales the state of a wrapped environment to [-1, 1]. """

    def __init__(self, env: Env, max_episode_length: Optional[int] = None):
        """
        Initializes the rescaled environment.
        Args:
            env: the environment to rescale.
            max_episode_length: the maximum number of steps of an episode.
        """
        self.env = env
        obs_space = env.observation_space
        self.bounds = list(zip(obs_space.low, obs_space.high))
        if max_episode_length is not None:
            self.env._max_episode_steps = max_episode_length

    def _rescale(self, state: np.ndarray) -> np.ndarray:
        """
        Private function that rescales a state to [-1, 1].
        Args:
            state: the state to rescale.

        Returns:
            the rescaled state.
        """
        return np.array([2 * (x - l) / (h - l) - 1 for x, (l, h) in zip(state, self.bounds)])

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Performs a step in the environment.
        Rescales the next_state
        Args:
            action: the action to perform.

        Returns:
            the rescaled next_state, reward, time, done and info. For more info, see env.step.
        """
        next_state, reward, time, done, info = self.env.step(action)
        return self._rescale(next_state), reward, time, done, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> \
            tuple[np.ndarray, dict]:
        """
        Resets the environment.
        Returns:
            the rescaled state and descriptor. For more info, see gym->env->reset.
        """
        state, descriptor = self.env.reset(seed=seed, options=options)
        return self._rescale(state), descriptor

    def render(self, mode: str = "human") -> None:
        """
        Renders the environment.
        Args:
            mode: the mode to render in. See gym->env->render for more info.
        """
        self.env.render(mode)

    def close(self) -> None:
        """ Closes the environment. """
        self.env.close()
