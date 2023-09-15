from gymnasium.envs import register
import gymnasium as gym

# TODO rewrite as an import * as mazes

import envs.pomdp.Mazes as mazes

# Notation:
# F: full observed (original env)
# P: position/angle observed
# V: velocity observed

register(
    "Pendulum-F-v0",
    entry_point="envs.pomdp.wrappers:POMDPWrapper",
    kwargs=dict(
        env=gym.make("Pendulum-v1"), partially_obs_dims=[0, 1, 2]
    ),  # angle & velocity
    max_episode_steps=200,
)

register(
    "Pendulum-P-v0",
    entry_point="envs.pomdp.wrappers:POMDPWrapper",
    kwargs=dict(env=gym.make("Pendulum-v1"), partially_obs_dims=[0, 1]),  # angle
    max_episode_steps=200,
)

register(
    "Pendulum-V-v0",
    entry_point="envs.pomdp.wrappers:POMDPWrapper",
    kwargs=dict(env=gym.make("Pendulum-v1"), partially_obs_dims=[2]),  # velocity
    max_episode_steps=200,
)

register(
    "CartPole-F-v1",
    entry_point="envs.pomdp.wrappers:POMDPWrapper",
    kwargs=dict(
        env=gym.make("CartPole-v1"), partially_obs_dims=[0, 1, 2, 3]
    ),  # angle & velocity
    max_episode_steps=200,  # reward threshold for solving the task: 195
)

register(
    "CartPole-P-v1",
    entry_point="envs.pomdp.wrappers:POMDPWrapper",
    kwargs=dict(env=gym.make("CartPole-v1"), partially_obs_dims=[0, 2]),
    max_episode_steps=200,
)

register(
    "CartPole-V-v1",
    entry_point="envs.pomdp.wrappers:POMDPWrapper",
    kwargs=dict(env=gym.make("CartPole-v1"), partially_obs_dims=[1, 3]),
    max_episode_steps=200,
)


# register(
#     "LunarLander-F-v0",
#     entry_point="envs.pomdp.wrappers:POMDPWrapper",
#     kwargs=dict(
#         env=gym.make("LunarLander-v2"), partially_obs_dims=list(range(8))
#     ),  # angle & velocity
#     max_episode_steps=1000,  # reward threshold for solving the task: 200
# )
#
# register(
#     "LunarLander-P-v0",
#     entry_point="envs.pomdp.wrappers:POMDPWrapper",
#     kwargs=dict(env=gym.make("LunarLander-v2"), partially_obs_dims=[0, 1, 4, 6, 7]),
#     max_episode_steps=1000,
# )
#
# register(
#     "LunarLander-V-v0",
#     entry_point="envs.pomdp.wrappers:POMDPWrapper",
#     kwargs=dict(env=gym.make("LunarLander-v2"), partially_obs_dims=[2, 3, 5, 6, 7]),
#     max_episode_steps=1000,
# )


def _register_maze(name: str, maze: list[list[int]]):
    register(
        name,
        entry_point="envs.pomdp.MazeEnv:MazeEnv",
        kwargs=dict(maze=maze),
        max_episode_steps=MAX_MAZE_STEPS,
    )


MAX_MAZE_STEPS = 1000
_register_maze("corridor-maze-v0", mazes.CORRIDOR_MAZE)
_register_maze("u-maze-v0", mazes.U_MAZE)
_register_maze("t-maze-v0", mazes.T_MAZE)
_register_maze("block-maze-v0", mazes.BLOCK_MAZE)
_register_maze("block0-maze-v0", mazes.BLOCK_MAZE_0)
_register_maze("block1-maze-v0", mazes.BLOCK_MAZE_1)
_register_maze("block2-maze-v0", mazes.BLOCK_MAZE_2)
_register_maze("block3-maze-v0", mazes.BLOCK_MAZE_3)
_register_maze("block4-maze-v0", mazes.BLOCK_MAZE_4)
_register_maze("block6-maze-v0", mazes.BLOCK_MAZE_6)
_register_maze("block8-maze-v0", mazes.BLOCK_MAZE_8)
_register_maze("block12-maze-v0", mazes.BLOCK_MAZE_12)
_register_maze("test-t-fully-v0", mazes.TEST_T_MAP)
_register_maze("test-bend-fully-v0", mazes.TEST_BEND_MAP)
_register_maze("test-cross-fully-v0", mazes.TEST_CROSS_MAP)
_register_maze("full-maze-without-traps-v0", mazes.FULL_MAZE_WITHOUT_TRAPS)
_register_maze("full-maze-v0", mazes.FULL_MAZE)
_register_maze("double-maze-v0", mazes.DOUBLE_MAP)
_register_maze("double-blocked-1-maze-v0", mazes.DOUBLE_MAP_BLOCKED_1)
_register_maze("double-blocked-2-maze-v0", mazes.DOUBLE_MAP_BLOCKED_2)
_register_maze("double-blocked-3-maze-v0", mazes.DOUBLE_MAP_BLOCKED_3)
_register_maze("double-blocked-4-maze-v0", mazes.DOUBLE_MAP_BLOCKED_4)
_register_maze("double-blocked-5-maze-v0", mazes.DOUBLE_MAP_BLOCKED_5)
_register_maze("double-blocked-6-maze-v0", mazes.DOUBLE_MAP_BLOCKED_6)
_register_maze("double-blocked-7-maze-v0", mazes.DOUBLE_MAP_BLOCKED_7)


def _register_partial_maze(name: str, original_maze: str):
    register(
        name,
        entry_point="envs.pomdp.wrappers:POMDPMazeWrapper",
        kwargs=dict(env=gym.make(original_maze), window_size=1),
        max_episode_steps=MAX_MAZE_STEPS,
    )


_register_partial_maze("test-t-partial-v0", "test-t-fully-v0")
_register_partial_maze("test-bend-partial-v0", "test-bend-fully-v0")
_register_partial_maze("test-cross-partial-v0", "test-cross-fully-v0")
_register_partial_maze(
    "full-maze-without-traps-partial-v0", "full-maze-without-traps-v0"
)
_register_partial_maze("full-maze-partial-v0", "full-maze-v0")
_register_partial_maze("double-maze-partial-v0", "double-maze-v0")
_register_partial_maze("double-blocked-1-maze-partial-v0", "double-blocked-1-maze-v0")
_register_partial_maze("double-blocked-2-maze-partial-v0", "double-blocked-2-maze-v0")
_register_partial_maze("double-blocked-3-maze-partial-v0", "double-blocked-3-maze-v0")
_register_partial_maze("double-blocked-4-maze-partial-v0", "double-blocked-4-maze-v0")
_register_partial_maze("double-blocked-5-maze-partial-v0", "double-blocked-5-maze-v0")
_register_partial_maze("double-blocked-6-maze-partial-v0", "double-blocked-6-maze-v0")
_register_partial_maze("double-blocked-7-maze-partial-v0", "double-blocked-7-maze-v0")

### Below are pybullect (roboschool) environments, using BLT for Bullet
# import pybullet_envs

"""
The observation space can be divided into several parts:
np.concatenate(
[
    z - self.initial_z, # pos
    np.sin(angle_to_target), # pos
    np.cos(angle_to_target), # pos
    0.3 * vx, # vel
    0.3 * vy, # vel
    0.3 * vz, # vel
    r, # pos
    p # pos
], # above are 8 dims
[j], # even elements [0::2] position, scaled to -1..+1 between limits
    # odd elements  [1::2] angular speed, scaled to show -1..+1
[self.feet_contact], # depends on foot_list, belongs to pos
])
"""
# register(
#     "HopperBLT-F-v0",
#     entry_point="envs.pomdp.wrappers:POMDPWrapper",
#     kwargs=dict(
#         env=gym.make("HopperBulletEnv-v0"),
#         partially_obs_dims=list(range(15)),
#     ),  # full obs
#     max_episode_steps=1000,
# )
#
# register(
#     "HopperBLT-P-v0",
#     entry_point="envs.pomdp.wrappers:POMDPWrapper",
#     kwargs=dict(
#         env=gym.make("HopperBulletEnv-v0"),
#         partially_obs_dims=[0, 1, 2, 6, 7, 8, 10, 12, 14],  # one foot
#     ),  # pos
#     max_episode_steps=1000,
# )
#
# register(
#     "HopperBLT-V-v0",
#     entry_point="envs.pomdp.wrappers:POMDPWrapper",
#     kwargs=dict(
#         env=gym.make("HopperBulletEnv-v0"),
#         partially_obs_dims=[3, 4, 5, 9, 11, 13],
#     ),  # vel
#     max_episode_steps=1000,
# )
#
# register(
#     "WalkerBLT-F-v0",
#     entry_point="envs.pomdp.wrappers:POMDPWrapper",
#     kwargs=dict(
#         env=gym.make("Walker2DBulletEnv-v0"),
#         partially_obs_dims=list(range(22)),
#     ),  # full obs
#     max_episode_steps=1000,
# )
#
# register(
#     "WalkerBLT-P-v0",
#     entry_point="envs.pomdp.wrappers:POMDPWrapper",
#     kwargs=dict(
#         env=gym.make("Walker2DBulletEnv-v0"),
#         partially_obs_dims=[0, 1, 2, 6, 7, 8, 10, 12, 14, 16, 18, 20, 21],  # 2 feet
#     ),  # pos
#     max_episode_steps=1000,
# )
#
# register(
#     "WalkerBLT-V-v0",
#     entry_point="envs.pomdp.wrappers:POMDPWrapper",
#     kwargs=dict(
#         env=gym.make("Walker2DBulletEnv-v0"),
#         partially_obs_dims=[3, 4, 5, 9, 11, 13, 15, 17, 19],
#     ),  # vel
#     max_episode_steps=1000,
# )
#
# register(
#     "AntBLT-F-v0",
#     entry_point="envs.pomdp.wrappers:POMDPWrapper",
#     kwargs=dict(
#         env=gym.make("AntBulletEnv-v0"),
#         partially_obs_dims=list(range(28)),
#     ),  # full obs
#     max_episode_steps=1000,
# )
#
# register(
#     "AntBLT-P-v0",
#     entry_point="envs.pomdp.wrappers:POMDPWrapper",
#     kwargs=dict(
#         env=gym.make("AntBulletEnv-v0"),
#         partially_obs_dims=[
#             0,
#             1,
#             2,
#             6,
#             7,
#             8,
#             10,
#             12,
#             14,
#             16,
#             18,
#             20,
#             22,
#             24,
#             25,
#             26,
#             27,
#         ],  # 4 feet
#     ),  # pos
#     max_episode_steps=1000,
# )
#
# register(
#     "AntBLT-V-v0",
#     entry_point="envs.pomdp.wrappers:POMDPWrapper",
#     kwargs=dict(
#         env=gym.make("AntBulletEnv-v0"),
#         partially_obs_dims=[3, 4, 5, 9, 11, 13, 15, 17, 19, 21, 23],
#     ),  # vel
#     max_episode_steps=1000,
# )
#
# register(
#     "HalfCheetahBLT-F-v0",
#     entry_point="envs.pomdp.wrappers:POMDPWrapper",
#     kwargs=dict(
#         env=gym.make("HalfCheetahBulletEnv-v0"),
#         partially_obs_dims=list(range(26)),
#     ),  # full obs
#     max_episode_steps=1000,
# )
#
# register(
#     "HalfCheetahBLT-P-v0",
#     entry_point="envs.pomdp.wrappers:POMDPWrapper",
#     kwargs=dict(
#         env=gym.make("HalfCheetahBulletEnv-v0"),
#         partially_obs_dims=[
#             0,
#             1,
#             2,
#             6,
#             7,
#             8,
#             10,
#             12,
#             14,
#             16,
#             18,
#             20,
#             21,
#             22,
#             23,
#             24,
#             25,
#         ],  # 6 feet
#     ),  # pos
#     max_episode_steps=1000,
# )
#
# register(
#     "HalfCheetahBLT-V-v0",
#     entry_point="envs.pomdp.wrappers:POMDPWrapper",
#     kwargs=dict(
#         env=gym.make("HalfCheetahBulletEnv-v0"),
#         partially_obs_dims=[3, 4, 5, 9, 11, 13, 15, 17, 19],
#     ),  # vel
#     max_episode_steps=1000,
# )
