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
        self.task_dim = self.tasks[0].embedding.shape[0]

        assert hasattr(
            self.env.unwrapped, "maze"
        ), "The environment must have a maze attribute."
        self.raw_maze = deepcopy(self.env.unwrapped.maze.maze)
        self.start_position = self.env.unwrapped.start_position
        self.task = None
        self.blocked = False
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
        info["blocked"] = self.blocked
        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.n_steps += 1
        data = self.env.step(action)
        info = data[-1]  # Last index is info
        info["embedding"] = self.task.embedding
        info["blocked"] = self.blocked
        if "success" in info and "agent_position" in info:
            agent_x = info["agent_position"][0, 1]
            start_x = self.start_position[0, 1]
            info["success"] = self.blocked ^ (agent_x < start_x)
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
        self.blocked = self.env.unwrapped.maze.blocked

    def get_current_task(self):
        return self.task

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def get_all_tasks(self):
        return self.tasks

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

    def is_goal_state(self) -> bool:
        """
        Checks whether a position is a goal position.
        Args:
            position: the position to check.

        Returns:
            boolean indicating whether the position is a goal position.
        """
        if not self.env.unwrapped.is_goal_state():
            return False

        agent_x = self.env.unwrapped.maze.agent_positions[0][0, 1]
        start_x = self.start_position[0, 1]
        return self.env.unwrapped.maze.blocked ^ (agent_x < start_x)
