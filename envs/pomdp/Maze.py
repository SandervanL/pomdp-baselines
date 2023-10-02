from typing import Union

import numpy as np
from mazelab import BaseMaze, Object
from mazelab import DeepMindColor as Color


class Maze(BaseMaze):
    """Maze class to be used in the environment."""

    def __init__(self, maze: np.ndarray | list[list[int]], **kwargs):
        """Initialize the maze."""
        if isinstance(maze, np.ndarray):
            self.maze = maze
        else:
            self.maze = np.array(maze)
        super().__init__(**kwargs)

    @property
    def size(self):
        """Return the size of the maze."""
        return self.maze.shape[0], self.maze.shape[1]

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
        obstacle = Object("obstacle", 1, Color.obstacle, True, self._get_places(1))
        agent = Object("agent", 2, Color.agent, False, self._get_places(2))
        goal = Object("goal", 3, Color.goal, False, self._get_places(3))
        return free, obstacle, agent, goal

    def _get_places(self, nums: int | list[int]) -> np.ndarray:
        """
        Get the places where the object is located.
        Args:
            nums: Numbers of the object.

        Returns:
            locations: Locations of the object.
        """
        if isinstance(nums, int):
            nums = [nums]

        locations = None
        for num in nums:
            new_locations = np.stack(np.where(self.maze == num), axis=1)
            if locations is None:
                locations = new_locations
            else:
                locations = np.concatenate((locations, new_locations), axis=0)
        return locations

    def agent_distance(self, goal: np.ndarray) -> float:
        return np.linalg.norm(self.objects.agent.positions[0] - goal)
