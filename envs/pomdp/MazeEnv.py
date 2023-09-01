""" Maze environment class. """
from typing import Optional, Union, SupportsFloat, Any

import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Box, Discrete
from numpy.random import Generator

from envs.pomdp.Maze import Maze
from mazelab import BaseEnv, VonNeumannMotion
from torch import Tensor

Position = tuple[int, int]


class MazeEnv(BaseEnv):
    """ Maze environment class. """

    def __init__(self, maze: list[list[int]], seed: Union[int, Generator, None] = None):
        """
        Initializes the maze environment.
        Args:
            maze: the maze to use.
        """
        super().__init__()

        self.maze = Maze(maze)
        self.start_position = np.stack(np.where(np.array(maze) == 2), axis=1)
        self.goal_positions = np.stack(np.where(np.array(maze) == 3), axis=1)
        self.motions = VonNeumannMotion()

        maze_size = self.maze.size
        self.observation_space = Box(
            low=0,
            high=len(self.maze.objects),
            shape=[maze_size[0] * maze_size[1]],
            dtype=np.int32
        )
        self.action_space = Discrete(len(self.motions), seed=seed)

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Performs a step in the maze environment.
        Args:
            action: the action to perform. (0: up, 1: right, 2: down, 3: left)

        Returns:
            the observation, the reward, whether the episode is done, and additional information.
            See gym.env.step for more information.
        """

        if isinstance(action, (np.ndarray, Tensor)):
            action = action.item()

        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position: Position = current_position[0] + motion[0], current_position[1] + motion[1]
        valid = self._is_valid(new_position)

        if valid:
            self.maze.objects.agent.positions = [new_position]
        next_state = self.maze.to_value()
        info = {'valid_actions': self._get_valid_actions()}

        if self.is_goal_state():
            reward = +100
            info['success'] = done = True
        elif not valid:
            reward = -1  # -1
            done = False
        else:
            reward = -0.01
            done = False

        info['original_state'] = next_state
        return next_state.flatten(), reward, done, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> \
            tuple[np.ndarray, dict]:
        """
        Resets the environment.
        Args:
            seed: the seed to use.
            options: additional options to use.
        Returns:
            the observation after resetting the environment.
        """
        super().reset(seed=seed, options=options)
        self.maze.objects.agent.positions = self.start_position
        self.maze.objects.goal.positions = self.goal_positions
        next_state = self.maze.to_value()
        return next_state.flatten(), {
            'valid_actions': self._get_valid_actions(),
            'original_state': next_state
        }

    def _is_valid(self, position: Position) -> bool:
        """
        Checks whether a position is valid.
        Args:
            position: the position to check.

        Returns:
            boolean indicating whether the position is valid.
        """
        non_negative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return non_negative and within_edge and passable

    def _get_valid_actions(self) -> np.ndarray:
        """
        Gets the valid actions for the current state.
        Returns:
            list of numbers for if an action is valid (0 for invalid, 1 for valid).
        """
        valid_actions = np.int8([0] * len(self.motions))
        for action, motion in enumerate(self.motions):
            current_pos = self.maze.objects.agent.positions[0]
            new_position: Position = current_pos[0] + motion[0], current_pos[1] + motion[1]
            if self._is_valid(new_position):
                valid_actions[action] = 1

        return valid_actions

    def is_goal_state(self) -> bool:
        """
        Checks whether a position is a goal position.
        Args:
            position: the position to check.

        Returns:
            boolean indicating whether the position is a goal position.
        """
        position = self.maze.objects.agent.positions[0]
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                return True

        return False

    def get_image(self):
        """
        Gets the image of the maze.
        Returns:
            the image of the maze.
        """
        return self.maze.to_rgb()
