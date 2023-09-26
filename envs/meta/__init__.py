from gymnasium.envs.registration import register

from envs.meta.maze.Mazes import ITEM_META_MAP, BIG_ITEM_META_MAP, DOUBLE_MAP_10
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
def register_fully(name: str, map: list[list[int]]):
    register(
        name,
        entry_point="envs.pomdp.MazeEnv:MazeEnv",
        kwargs=dict(maze=map),
        max_episode_steps=MAX_MAZE_STEPS,
    )


register_fully("item-maze-fully-v0", ITEM_META_MAP)
register_fully("big-item-maze-fully-v0", BIG_ITEM_META_MAP)
register_fully("double-map-10-v0", DOUBLE_MAP_10)


# Partial obs maps
def register_partial(name: str, fully_name: str):
    register(
        name,
        entry_point="envs.pomdp.wrappers:POMDPMazeWrapper",
        kwargs=dict(env=fully_name, window_size=1),
        max_episode_steps=MAX_MAZE_STEPS,
    )


register_partial("item-maze-partial-v0", "item-maze-fully-v0")
register_partial("big-item-maze-partial-v0", "big-item-maze-fully-v0")
register_partial("double-map-10-partial-v0", "double-map-10-v0")

# Meta maps


def register_meta(name: str, partial_name: str):
    register(
        name,
        entry_point="envs.meta.maze.MultitaskMazeEnv:MultitaskMazeEnv",
        kwargs=dict(env_id=partial_name),
        max_episode_steps=MAX_MAZE_STEPS,
    )


register_meta("item-maze-meta-partial-v0", "item-maze-partial-v0")
register_meta("big-item-maze-meta-partial-v0", "big-item-maze-partial-v0")
register_meta("double-map-10-meta-partial-v0", "double-map-10-partial-v0")
