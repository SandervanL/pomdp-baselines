import threading
from typing import Optional, Any, SupportsFloat

import gymnasium as gym
import gymnasium.envs.atari
import gymnasium.wrappers
import numpy as np
from gymnasium.core import ActType, ObsType


class Atari(gym.Env):
    """
    all follow DreamerV2
    NOTE: don't clip rewards here, as we need raw scores for comparison.
    """

    LOCK = threading.Lock()

    def __init__(
            self,
            name,
            action_repeat=4,
            size=(64, 64),
            grayscale=True,
            noops=30,
            life_done=False,
            sticky_actions=True,
            all_actions=True,
            flatten_img=True,
    ):
        assert size[0] == size[1]
        channels = 1 if grayscale else 3
        image_sizes = (channels, size[0], size[1])

        with self.LOCK:
            # https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/src/gym/envs/atari/environment.py
            env = gym.envs.atari.AtariEnv(
                game=name,
                obs_type="grayscale" if grayscale else "rgb",
                frameskip=1,
                repeat_action_probability=0.25 if sticky_actions else 0.0,
                full_action_space=all_actions,
            )

        # Avoid unnecessary rendering in inner env.
        env.get_obs = lambda: None  # type: ignore
        # Tell wrapper that the inner env has no action repeat.
        env.spec = gym.envs.registration.EnvSpec("NoFrameskip-v0")  # type: ignore
        env = gym.wrappers.AtariPreprocessing(
            env, noops, action_repeat, size[0], life_done, grayscale
        )
        self.env = env

        self.action_space = self.env.action_space
        self.image_space = gym.spaces.Box(
            low=0, high=255, shape=image_sizes, dtype=np.uint8
        )  # (1, )

        self.flatten_img = flatten_img
        if flatten_img:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(np.prod(image_sizes),), dtype=np.uint8
            )
        else:
            self.observation_space = self.image_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> \
            tuple[np.ndarray, dict]:

        with self.LOCK:
            image, info = self.env.reset(seed=seed, options=options)  # type: ignore
        return self.observe(image), {}

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        image, reward, terminated, truncated, info = self.env.step(action)
        return self.observe(image), reward, terminated, truncated, info

    def observe(self, image):
        if self.flatten_img:
            return image.flatten()
        else:
            return np.expand_dims(image, axis=0)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
