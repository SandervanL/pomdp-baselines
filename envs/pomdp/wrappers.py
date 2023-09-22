from typing import Optional, Any, SupportsFloat

import gymnasium as gym
from gymnasium import spaces, Env
import numpy as np
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Box


class POMDPWrapper(gym.Wrapper):
    def __init__(self, env: Env, partially_obs_dims: list[int]):
        super().__init__(env)
        self.partially_obs_dims: list[int] = partially_obs_dims
        # can equal to the fully-observed env
        assert 0 < len(self.partially_obs_dims) <= self.observation_space.shape[0]

        self.observation_space = spaces.Box(
            low=self.observation_space.low[self.partially_obs_dims],
            high=self.observation_space.high[self.partially_obs_dims],
            dtype=np.float32,
        )

        if self.env.action_space.__class__.__name__ == "Box":
            self.act_continuous = True
            # if continuous actions, make sure in [-1, 1]
            # NOTE: policy won't use action_space.low/high, just set [-1,1]
            # this is a bad practice...
        else:
            self.act_continuous = False

    def get_obs(self, state: np.ndarray) -> ObsType:
        """Get the partially observed state."""
        return state[self.partially_obs_dims].copy()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict]:
        state, info = self.env.reset(
            seed=seed, options=options
        )  # TODO said 'no kwargs'. Why?
        return self.get_obs(state), info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.act_continuous:
            # recover the action
            action = np.clip(action, -1, 1)  # first clip into [-1, 1]
            lb = self.env.action_space.low
            ub = self.env.action_space.high
            action = lb + (action + 1.0) * 0.5 * (ub - lb)
            action = np.clip(action, lb, ub)

        state, reward, terminated, truncated, info = self.env.step(action)

        return self.get_obs(state), reward, terminated, truncated, info


class POMDPMazeWrapper(gym.Wrapper):
    """Partially observable maze environment class."""

    metadata = {"render_modes": ["human", "rgb_array"], "video.frames_per_second": 3}

    def __init__(self, env: Env, window_size: int):
        super().__init__(env)
        self.window_size: int = window_size

        self.observation_space = Box(
            low=0,
            high=len(self.unwrapped.maze.objects),
            shape=[(2 * window_size + 1) ** 2],
            dtype=np.int32,
        )

    def _get_observation(self, state: np.ndarray) -> np.ndarray:
        """
        Gets the observation of the environment.
        Returns:
            the observation of the environment (n x n window around the agent).
        """
        agent_position = self.unwrapped.maze.objects.agent.positions[0]
        return state[
            agent_position[0]
            - self.window_size : agent_position[0]
            + self.window_size
            + 1,
            agent_position[1]
            - self.window_size : agent_position[1]
            + self.window_size
            + 1,
        ].flatten()

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Performs a step in the maze environment.
        Args:
            action: the action to perform. (0: up, 1: right, 2: down, 3: left)

        Returns:
            the observation, the reward, whether the episode is done, and additional information.
        """
        _, reward, done, truncated, info = self.env.step(action)
        return (
            self._get_observation(info["original_state"]),
            reward,
            done,
            truncated,
            info,
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict]:
        """
        Resets the environment.
        Args:
            seed: the seed to use.
            options: additional options to use.

        Returns:
            the observation after resetting the environment.
        """
        _, info = self.env.reset(seed=seed, options=options)
        return self._get_observation(info["original_state"]), info


def main():
    import envs

    env = gym.make("HopperBLT-F-v0")
    done = False
    step = 0
    while not done:
        next_obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
        step += 1
        print(step, terminated, truncated, info)


if __name__ == "__main__":
    main()
