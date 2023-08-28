from typing import Optional, Any, SupportsFloat

import gymnasium as gym
from gymnasium import spaces, Env
import numpy as np
from gymnasium.core import ObsType, ActType


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
        """ Get the partially observed state. """
        return state[self.partially_obs_dims].copy()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> \
            tuple[np.ndarray, dict]:
        state, info = self.env.reset(seed=seed, options=options)  # TODO said 'no kwargs'. Why?
        return self.get_obs(state), info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.act_continuous:
            # recover the action
            action = np.clip(action, -1, 1)  # first clip into [-1, 1]
            lb = self.env.action_space.low
            ub = self.env.action_space.high
            action = lb + (action + 1.0) * 0.5 * (ub - lb)
            action = np.clip(action, lb, ub)

        state, reward, terminated, truncated, info = self.env.step(action)

        return self.get_obs(state), reward, terminated, truncated, info


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
