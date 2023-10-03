from typing import Optional, Any, SupportsFloat

import numpy as np
from gymnasium.core import ActType, ObsType

from envs.meta.maze.MultitaskMaze import MultitaskMaze, load_tasks_file, MazeTask
from envs.pomdp.MazeEnv import MazeEnv


class MultitaskMazeEnv(MazeEnv):
    """Multitask maze environment class."""

    def __init__(self, task_file: str, **kwargs):
        self.tasks: list[MazeTask] = load_tasks_file(task_file)
        self.task_dim = self.tasks[0].embedding.shape[0]
        maze = MultitaskMaze(self.tasks[0])
        super().__init__(maze.maze, **kwargs)
        self.maze = maze
        goal_location_distances = {
            self.maze.agent_distance(goal): goal for goal in self.goal_positions
        }
        self.short_goal_location = goal_location_distances[
            min(goal_location_distances.keys())
        ]
        self.long_goal_location = goal_location_distances[
            max(goal_location_distances.keys())
        ]

        self.task = None
        self.blocked = False
        self.n_tasks = len(self.tasks)
        self.left_counter = 0

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
        obs, info = super().reset(seed=seed, options=options)
        info["embedding"] = self.task.embedding
        info["blocked"] = self.blocked
        info["left_counter"] = self.left_counter = 0
        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        data = super().step(action)
        info = data[-1]  # Last index is info
        info["embedding"] = self.task.embedding
        info["blocked"] = self.blocked

        if (
            self.maze.agent_distance(self.start_position) <= 1.0001
            and self.maze.agent_distance(self.short_goal_location) < 3.0001
        ):
            self.left_counter += 1

        info["left_counter"] = self.left_counter
        if "success" in info:
            print(
                f"Impassable: {self.maze.objects.item.impassable}. Agent pos: {self.maze.objects.agent.positions[0]}"
            )
            info["success"] = self.is_right_goal()
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

        Returns:
            the observation and info after resetting the environment.
        """
        if task is None:
            self.task = self.get_random_task(seed=seed)
        else:
            self.task = self.tasks[task]
        self.maze = MultitaskMaze(self.task)
        self.blocked = self.maze.blocked
        print(f"Reset task. Blocked: {self.blocked}. Sentence: {self.task.sentence}")

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
        return image

    def is_right_goal(self) -> bool:
        """
        Checks whether a position is a goal position.

        Returns:
            boolean indicating whether the position is a goal position.
        """
        return super().is_goal_state() and (
            self.blocked
            ^ np.all((self.maze.objects.agent.positions[0] == self.short_goal_location))
        )
