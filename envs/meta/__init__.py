from gymnasium.envs.registration import register
import gymnasium as gym

from envs.meta.maze.Mazes import ITEM_META_MAP, BIG_ITEM_META_MAP
from envs.pomdp import MAX_MAZE_STEPS

## off-policy variBAD benchmark

register(
    "PointRobot-v0",
    entry_point="envs.meta.toy_navigation.point_robot:PointEnv",
    kwargs={"max_episode_steps": 60, "n_tasks": 2},
)

register(
    "PointRobotSparse-v0",
    entry_point="envs.meta.toy_navigation.point_robot:SparsePointEnv",
    kwargs={"max_episode_steps": 60, "n_tasks": 2, "goal_radius": 0.2},
)

register(
    "Wind-v0",
    entry_point="envs.meta.toy_navigation.wind:WindEnv",
)

register(
    "HalfCheetahVel-v0",
    entry_point="envs.meta.wrappers:mujoco_wrapper",
    kwargs={
        "entry_point": "envs.meta.mujoco.half_cheetah_vel:HalfCheetahVelEnv",
        "max_episode_steps": 200,
    },
    max_episode_steps=200,
)

## on-policy variBAD benchmark

register(
    "AntDir-v0",
    entry_point="envs.meta.wrappers:mujoco_wrapper",
    kwargs={
        "entry_point": "envs.meta.mujoco.ant_dir:AntDirEnv",
        "max_episode_steps": 200,
        "forward_backward": True,
        "n_tasks": None,
    },
    max_episode_steps=200,
)

register(
    "CheetahDir-v0",
    entry_point="envs.meta.wrappers:mujoco_wrapper",
    kwargs={
        "entry_point": "envs.meta.mujoco.half_cheetah_dir:HalfCheetahDirEnv",
        "max_episode_steps": 200,
        "n_tasks": None,
    },
    max_episode_steps=200,
)

register(
    "HumanoidDir-v0",
    entry_point="envs.meta.wrappers:mujoco_wrapper",
    kwargs={
        "entry_point": "envs.meta.mujoco.humanoid_dir:HumanoidDirEnv",
        "max_episode_steps": 200,
        "n_tasks": None,
    },
    max_episode_steps=200,
)

# MDP Item maps
register(
    "item-maze-fully-v0",
    entry_point="envs.pomdp.MazeEnv:MazeEnv",
    kwargs=dict(maze=ITEM_META_MAP),
    max_episode_steps=MAX_MAZE_STEPS
)
register(
    "big-item-maze-fully-v0",
    entry_point="envs.pomdp.MazeEnv:MazeEnv",
    kwargs=dict(maze=BIG_ITEM_META_MAP),
    max_episode_steps=MAX_MAZE_STEPS
)

# Partial obs maps
register(
    "item-maze-partial-v0",
    entry_point="envs.pomdp.wrappers:POMDPMazeWrapper",
    kwargs=dict(env=gym.make("item-maze-fully-v0"), window_size=1),
    max_episode_steps=MAX_MAZE_STEPS,
)
register(
    "big-item-maze-partial-v0",
    entry_point="envs.pomdp.wrappers:POMDPMazeWrapper",
    kwargs=dict(env=gym.make("big-item-maze-fully-v0"), window_size=1),
    max_episode_steps=MAX_MAZE_STEPS,
)

# Meta maps
register(
    'item-maze-meta-partial-v0',
    entry_point="envs.meta.maze.MultitaskMazeEnv:MultitaskMazeEnv",
    kwargs=dict(env=gym.make("item-maze-partial-v0")),
    max_episode_steps=MAX_MAZE_STEPS
)
register(
    'big-item-maze-meta-partial-v0',
    entry_point="envs.meta.maze.MultitaskMazeEnv:MultitaskMazeEnv",
    kwargs=dict(env=gym.make("big-item-maze-partial-v0")),
    max_episode_steps=MAX_MAZE_STEPS
)
