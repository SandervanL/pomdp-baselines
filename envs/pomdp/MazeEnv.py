""" Maze environment class. """
from typing import Optional, SupportsFloat, Any

import numpy as np
import pygame
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Box, Discrete
from numpy.random import Generator
from torch import Tensor

from envs.pomdp.Maze import Maze
from mazelab import BaseEnv, VonNeumannMotion
from torchkit import pytorch_utils as ptu

Position = tuple[int, int]


class MazeEnv(BaseEnv):
    """Maze environment class."""

    def __init__(
        self,
        maze: list[list[int]],
        seed: Optional[int | Generator] = None,
        render_mode: Optional[str] = None,
    ):
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
            dtype=np.int32,
        )
        self.action_space = Discrete(len(self.motions), seed=seed)
        self.left_counter = 0
        self.render_mode = render_mode
        self.window = self.clock = None
        self.window_size = 100
        self.train_mode = False
        self.image = None
        self.prev_type_image = None

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
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
        new_position: Position = (
            current_position[0] + motion[0],
            current_position[1] + motion[1],
        )
        valid = self._are_valid([new_position])[0]

        if valid:
            self.maze.objects.agent.positions = [new_position]
        next_state = self.maze.to_value()

        if self.maze.objects.agent.positions[0][1] == 4:
            self.left_counter += 1

        info = {
            "valid_actions": self._get_valid_actions(),
            "original_state": next_state,
            "agent_position": ptu.from_numpy(
                np.asarray(self.maze.objects.agent.positions[0])
            ).unsqueeze(0),
            "left_counter": self.left_counter,
        }

        if self.is_goal_state():
            reward = 100
            info["success"] = done = True
        elif not valid:
            reward = -1  # -1  # -1
            done = False
        else:
            reward = 0  # 0 -0.01
            done = False

        if self.render_mode == "human":
            self._render_frame()

        return next_state.flatten(), reward, done, False, info

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
        self.train_mode = (
            False
            if options is None
            else ("train_mode" in options and options["train_mode"] == True)
        )
        self.prev_type_image = None
        _, info = super().reset(seed=seed, options=options)
        self.maze.objects.agent.positions = self.start_position
        self.maze.objects.goal.positions = self.goal_positions
        next_state = self.maze.to_value()
        info["valid_actions"] = self._get_valid_actions()
        info["original_state"] = next_state
        info["agent_position"] = ptu.from_numpy(
            self.maze.objects.agent.positions[0]
        ).unsqueeze(0)
        info["left_counter"] = self.left_counter = 0

        if self.render_mode == "human":
            self._render_frame()
        return next_state.flatten(), info

    def _are_valid(self, positions: list[Position]) -> list[bool]:
        """
        Checks whether a position is valid.
        Args:
            positions: the positions to check.

        Returns:
            boolean indicating whether the position is valid.
        """
        result = [False] * len(positions)
        is_impassable_maze = self.maze.to_impassable()

        # Pycharm type annotator is not convinced that 'position' is of type Position, but it is.
        for index, position in enumerate(positions):
            non_negative = position[0] >= 0 and position[1] >= 0
            within_edge = (
                position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
            )
            passable = not is_impassable_maze[position[0]][position[1]]
            result[index] = non_negative and within_edge and passable

        return result

    def _get_valid_actions(self) -> np.ndarray:
        """
        Gets the valid actions for the current state.
        Returns:
            list of numbers for if an action is valid (0 for invalid, 1 for valid).
        """
        new_positions: list[Position] = [(0, 0)] * len(self.motions)
        current_pos = self.maze.objects.agent.positions[0]
        for action, motion in enumerate(self.motions):
            new_positions[action] = (
                current_pos[0] + motion[0],
                current_pos[1] + motion[1],
            )

        valid_actions = self._are_valid(new_positions)
        return np.array(valid_actions, dtype=np.uint8)

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

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        self.image = super().render(mode="rgb_array")

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.image.shape[1], 2 * self.image.shape[0])
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        pygame.event.pump()

        if self.prev_type_image is None:
            self.prev_type_image = self.image

        if self.train_mode:
            new_image = np.concatenate([self.image, self.prev_type_image], axis=0)
        else:
            new_image = np.concatenate([self.prev_type_image, self.image], axis=0)

        surf = pygame.surfarray.make_surface(np.rot90(new_image, axes=[1, 0]))
        self.window.blit(surf, (0, 0))
        pygame.display.update()

        # canvas = pygame.Surface((self.window_size, self.window_size))
        # canvas.fill((255, 255, 255))
        # pix_square_size = (
        #     self.window_size / self.size
        # )  # The size of a single grid square in pixels
        #
        # # First we draw the target
        # target_loc = self.maze.objects.goal.positions[0]
        # pygame.draw.rect(
        #     canvas,
        #     (255, 0, 0),
        #     pygame.Rect(
        #         pix_square_size * target_loc,
        #         (pix_square_size, pix_square_size),
        #     ),
        # )
        # agent_loc = self.maze.objects.agent.positions[0]
        # # Now we draw the agent
        # pygame.draw.circle(
        #     canvas,
        #     (0, 0, 255),
        #     (agent_loc + 0.5) * pix_square_size,
        #     pix_square_size / 3,
        # )
        #
        # # Finally, add some gridlines
        # for x in range(self.size + 1):
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (0, pix_square_size * x),
        #         (self.window_size, pix_square_size * x),
        #         width=3,
        #     )
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (pix_square_size * x, 0),
        #         (pix_square_size * x, self.window_size),
        #         width=3,
        #     )
        #
        # if self.render_mode == "human":
        #     # The following line copies our drawings from `canvas` to the visible window
        #     self.window.blit(canvas, canvas.get_rect())
        #     pygame.event.pump()
        #     pygame.display.update()
        #
        #     # We need to ensure that human-rendering occurs at the predefined framerate.
        #     # The following line will automatically add a delay to keep the framerate stable.
        #     self.clock.tick(self.metadata["render_fps"])
        # else:  # rgb_array
        #     return np.transpose(
        #         np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        #     )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
