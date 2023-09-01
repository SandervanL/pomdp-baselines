import pickle
from copy import deepcopy
from dataclasses import dataclass

import dill
import numpy as np
from torch import Tensor

from envs.pomdp.Maze import Maze
from mazelab import Object
from mazelab import DeepMindColor as Color
import torchkit.pytorch_utils as ptu


@dataclass
class MazeTask:
    """ Maze task class. """
    embedding: Tensor
    right_direction: bool
    blocked: bool


class MultitaskMaze(Maze):
    def __init__(self, task: MazeTask, maze: np.ndarray, **kwargs):
        item_maze = deepcopy(maze)
        item_locations = self.get_item_locations(maze)
        self.blocked = task.blocked

        # If the blockage is to the right, set left free
        if task.right_direction:
            left_loc = item_locations[0]
            item_maze[left_loc[0], left_loc[1]] = 0

        # If the blockage is to the left, or there is no blockage, set right free
        if (not task.blocked) or not task.right_direction:
            right_loc = item_locations[1]
            item_maze[right_loc[0], right_loc[1]] = 0

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
        free_numbers = [0, 2, 3] if self.blocked else [0, 2, 3, 4]
        obstacle_numbers = [1, 4] if self.blocked else [1]
        free = Object('free', 0, Color.free, False, self._get_places(free_numbers))
        obstacle = Object('obstacle', 1, Color.obstacle, True, self._get_places(obstacle_numbers))
        agent = Object('agent', 2, Color.agent, False, self._get_places(2))
        goal = Object('goal', 3, Color.goal, False, self._get_places(3))
        return free, obstacle, agent, goal


def load_tasks_file(filename: str) -> list[MazeTask]:
    with open(filename, 'rb') as file:
        tasks = dill.load(file)

    for task in tasks:
        task.embedding = ptu.to_device(task.embedding)

    return tasks


def make_tasks_file():
    no_blockage_1 = MazeTask(
        embedding=Tensor([0.0, 0.0]),
        blocked=False,
        right_direction=True,
    )
    no_blockage_2 = MazeTask(
        embedding=Tensor([1.0, 1.0]),
        blocked=False,
        right_direction=True,
    )
    right_blockage = MazeTask(
        embedding=Tensor([0.0, 1.0]),
        blocked=True,
        right_direction=True,
    )
    left_blockage = MazeTask(
        embedding=Tensor([1.0, 0.0]),
        blocked=True,
        right_direction=False,
    )
    tasks = [
        no_blockage_1, no_blockage_2, right_blockage, left_blockage,
        no_blockage_1, no_blockage_2, right_blockage, left_blockage,
        no_blockage_1, no_blockage_2, right_blockage, left_blockage,
        no_blockage_1, no_blockage_2, right_blockage, left_blockage,
        no_blockage_1, no_blockage_2, right_blockage, left_blockage,
        no_blockage_1, no_blockage_2, right_blockage, left_blockage
    ]

    # Write tasks to a pickle file
    with open('configs/meta/maze/v/simple_embeddings.pkl', 'wb') as file:
        dill.dump(tasks, file)


if __name__ == '__main__':
    make_tasks_file()
