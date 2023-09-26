import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import dill
import numpy as np
from torch import Tensor

from envs.pomdp.Maze import Maze
from mazelab import Object
from mazelab import DeepMindColor as Color
import torchkit.pytorch_utils as ptu


@dataclass
class MazeTask:
    """Maze task class."""

    embedding: Tensor
    right_direction: bool
    blocked: bool
    task_type: int  # unique number for each type (high vs low, heavy vs light, etc)
    word: str


class MultitaskMaze(Maze):
    def __init__(self, task: MazeTask, maze: np.ndarray, **kwargs):
        item_maze = deepcopy(maze)
        self.item_location = self.get_item_locations(item_maze)[0]
        self.blocked = task.blocked
        super().__init__(item_maze, **kwargs)

    @staticmethod
    def get_item_locations(maze: np.ndarray) -> list[tuple[int, int]]:
        """
        Get the locations of the items in the maze.
        Args:
            maze: The maze to get the locations from.

        Returns:
            The locations of the maze from left to right.
        """
        result = []

        # Traverse the maze from left to right
        for j in range(maze.shape[1]):
            for i in range(maze.shape[0]):
                if maze[i, j] == 4:
                    result.append((i, j))

        return result

    def make_objects(self):
        """
        Make the objects in the maze.
        Returns:
            free: Object representing free space.
            obstacle: Object representing obstacles.
            agent: Object representing the agent.
            goal: Object representing the goal.
        """
        free = Object("free", 0, Color.free, False, self._get_places([0, 2, 3]))
        obstacle = Object("obstacle", 1, Color.obstacle, True, self._get_places([1, 4]))
        agent = Object("agent", 2, Color.agent, False, self._get_places(2))
        goal = Object("goal", 3, Color.goal, False, self._get_places(3))
        color = Color.lava if self.blocked else Color.water
        item = Object("item", 4, color, self.blocked, self._get_places(4))
        return free, obstacle, agent, goal, item


def load_tasks_file(filename: str) -> list[MazeTask]:
    main_path = Path(__file__).resolve().parent.parent.parent.parent
    file_path = os.path.join(main_path, filename)
    with open(file_path, "rb") as file:
        tasks = dill.load(file)

    for task in tasks:
        task.embedding = ptu.to_device(task.embedding)

    return tasks
