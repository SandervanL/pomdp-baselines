import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import dill
import numpy as np
import torch
from torch import Tensor

from envs.meta.maze.MazeTask import MazeTask
from envs.pomdp.Maze import Maze
from mazelab import Object, VonNeumannMotion
from mazelab import DeepMindColor as Color
import torchkit.pytorch_utils as ptu


class MultitaskMaze(Maze):
    def __init__(self, task: MazeTask, **kwargs):
        item_maze = build_maze(task)
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
        # color = Color.lava if self.blocked else Color.water
        item = Object("item", 4, Color.amber, self.blocked, self._get_places(4))
        return free, obstacle, goal, item, agent


def load_tasks_file(filename: str) -> list[MazeTask]:
    main_path = Path(__file__).resolve().parent.parent.parent.parent
    file_path = os.path.join(main_path, filename)

    if os.path.isfile(file_path) and file_path.endswith(".dill"):
        return load_dill_tasks_file(file_path)

    # Check if it is a dir
    if os.path.isdir(file_path):
        return load_tasks_dir(file_path)

    raise ValueError(f"Could not find task file '{file_path}'")


def load_dill_tasks_file(file_path: str) -> list[MazeTask]:
    with open(file_path, "rb") as file:
        tasks = dill.load(file)

    for task in tasks:
        task.embedding = task.embedding.to(ptu.device)

    return tasks


def load_tasks_dir(dir_path: str) -> list[MazeTask]:
    json_path = os.path.join(dir_path, "tasks.json")
    with open(json_path, "r") as file:
        tasks: list[dict] = json.load(file)

    embeddings_path = os.path.join(dir_path, "embeddings.pt")
    embeddings = torch.load(embeddings_path).to(ptu.device)

    result: list[Optional[MazeTask]] = [None] * len(tasks)
    for index, (task, embedding) in enumerate(zip(tasks, embeddings)):
        result[index] = MazeTask(embedding=embedding, **task)

    return result


def build_maze(task: MazeTask) -> np.ndarray:
    assert (
        task.short_direction != task.long_direction
        and task.short_hook_direction != task.long_direction
        and anti_direction(task.short_direction) != task.short_hook_direction
        and anti_direction(task.long_direction) != task.long_hook_direction
    )
    maze = np.ones((23, 23), dtype=np.uint8)
    start_pos = np.array([11, 11])
    maze[start_pos[0], start_pos[0]] = 2
    motions = VonNeumannMotion()

    # Part before short hook
    current_pos = np.copy(start_pos)
    direction = np.array(motions[task.short_direction])
    current_pos += direction
    maze[current_pos[0], current_pos[1]] = 0
    current_pos += direction
    make_cross(maze, current_pos)
    maze[current_pos[0], current_pos[1]] = 4

    # Part after short hook
    direction = np.array(motions[task.short_hook_direction])
    current_pos += 2 * direction
    maze[current_pos[0], current_pos[1]] = 3

    # Part before long hook
    current_pos = np.copy(start_pos)
    direction = np.array(motions[task.long_direction])
    for i in range(3):
        current_pos += direction
        maze[current_pos[0], current_pos[1]] = 0
        current_pos += direction
        make_cross(maze, current_pos)

    # Part after long hook
    direction = np.array(motions[task.long_hook_direction])
    current_pos += 2 * direction
    maze[current_pos[0], current_pos[1]] = 0
    current_pos += direction
    make_cross(maze, current_pos)
    current_pos += np.array(motions[to_right(task.long_hook_direction)])
    maze[current_pos[0], current_pos[1]] = 3

    return maze


def make_cross(maze: np.ndarray, pos: np.ndarray):
    motions = VonNeumannMotion()
    maze[pos[0], pos[1]] = 0
    for i in range(4):
        new_pos = pos + motions[i]
        maze[new_pos[0], new_pos[1]] = 0


def anti_direction(direction: int) -> int:
    return ((direction + 1) % 2) + 2 * (direction // 2)


def to_right(direction: int) -> int:
    if direction == 0:
        return 3
    if direction == 1:
        return 2
    if direction == 2:
        return 0
    if direction == 3:
        return 1
