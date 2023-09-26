from copy import deepcopy
from typing import Optional, Any, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium.core import ActType, ObsType

from envs.meta.maze.MultitaskMaze import MultitaskMaze, load_tasks_file, MazeTask


class MultitaskMazeEnv(gym.Wrapper):
    """Multitask maze environment class."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 3}

    def __init__(self, env_id: str, task_file: str, **kwargs):
        print("breakpoint")
        env = gym.make(env_id, **kwargs)
        super().__init__(env)

        self.tasks: list[MazeTask] = load_tasks_file(task_file)
        self.task_dim = self.tasks[0].embedding.dim()

        assert hasattr(
            self.env.unwrapped, "maze"
        ), "The environment must have a maze attribute."
        self.raw_maze = deepcopy(self.env.unwrapped.maze.maze)
        self.task = None
        self.n_tasks = len(self.tasks)
        self.n_steps = 0

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
        if self.n_steps < 2:
            print("breakpoint")
        self.n_steps = 0
        obs, info = self.env.reset(seed=seed, options=options)
        info["embedding"] = self.task.embedding
        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.n_steps += 1
        data = self.env.step(action)
        data[-1]["embedding"] = self.task.embedding  # Last index is info
        return data

    def reset_task(
        self,
        task: Optional[int] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        """
        Resets to a new task.
        Args:
            task: the index of the task to reset to. If None, a random task is chosen.
            seed: the seed to use.
            options: additional options to use.

        Returns:
            the observation and info after resetting the environment.
        """
        if self.n_steps < 2:
            print("breakpoint")
        if task is None:
            self.task = self.get_random_task(seed=seed)
        else:
            self.task = self.tasks[task]
        self.env.unwrapped.maze = MultitaskMaze(self.task, self.raw_maze)

    def get_current_task(self):
        return self.task

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def get_all_task_types(self):
        for index in self.get_all_task_idx():
            return index, self.tasks[index].task_type

    def get_random_task(self, seed: Optional[int] = None):
        rng = np.random.RandomState(seed)
        return rng.choice(self.tasks)

    def get_image(self):
        """
        Gets the image of the maze.
        Returns:
            the image of the maze.
        """
        image = self.maze.to_rgb()
        if not self.objects.item.blocked:
            print("breakpoint")
        return image
