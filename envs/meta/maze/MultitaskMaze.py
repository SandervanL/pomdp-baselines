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


class MultitaskMaze(Maze):
    def __init__(self, task: MazeTask, maze: np.ndarray, **kwargs):
        item_maze = deepcopy(maze)
        item_locations = self.get_item_locations(maze)
        self.blocked = task.blocked

        # If the blockage is to the right, set left free
        free_loc = item_locations[0] if task.right_direction else item_locations[1]
        item_maze[free_loc[0], free_loc[1]] = 0

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
        item = Object("item", 4, Color.box, self.blocked, self._get_places(4))
        return free, obstacle, agent, goal, item


def load_tasks_file(filename: str) -> list[MazeTask]:
    main_path = Path(__file__).resolve().parent.parent.parent.parent
    file_path = os.path.join(main_path, filename)
    with open(file_path, "rb") as file:
        tasks = dill.load(file)

    for task in tasks:
        task.embedding = ptu.to_device(task.embedding)

    return tasks


def make_single_tasks_file():
    no_blockage = MazeTask(
        embedding=Tensor([0.0]), blocked=False, right_direction=False, task_type=0
    )
    blockage = MazeTask(
        embedding=Tensor([1.0]),
        blocked=True,
        right_direction=False,
        task_type=1,
    )
    tasks = []
    for i in range(18):
        tasks.append(no_blockage)
        tasks.append(blockage)

    with open("configs/meta/maze/v/single_embeddings.pkl", "wb") as file:
        dill.dump(tasks, file)


def make_double_tasks_file():
    no_blockage_1 = MazeTask(
        embedding=Tensor([0.0, 0.0]),
        blocked=False,
        right_direction=True,
        task_type=1,
    )
    no_blockage_2 = MazeTask(
        embedding=Tensor([1.0, 1.0]),
        blocked=False,
        right_direction=True,
        task_type=2,
    )
    right_blockage_obj1 = MazeTask(
        embedding=Tensor([1.0, 0.0]), blocked=True, right_direction=True, task_type=0
    )
    left_blockage_obj1 = MazeTask(
        embedding=Tensor([-1.0, 0.0]), blocked=True, right_direction=False, task_type=4
    )
    right_blockage_obj2 = MazeTask(
        embedding=Tensor([0.0, 1.0]), blocked=False, right_direction=True, task_type=5
    )
    left_blockage_obj2 = MazeTask(
        embedding=Tensor([0.0, -1.0]),
        blocked=False,
        right_direction=False,
        task_type=6,
    )

    tasks = [
        no_blockage_1,
        no_blockage_2,
        right_blockage_obj1,
        left_blockage_obj1,
        right_blockage_obj2,
        left_blockage_obj2,
    ]

    # Write tasks to a pickle file
    with open("configs/meta/maze/v/simple_embeddings.pkl", "wb") as file:
        dill.dump(tasks, file)

    tasks = [
        right_blockage_obj1,
        left_blockage_obj1,
    ]
    with open("configs/meta/maze/v/double_embeddings.pkl", "wb") as file:
        dill.dump(tasks, file)


if __name__ == "__main__":
    make_single_tasks_file()
