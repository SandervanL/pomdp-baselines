from typing import SupportsFloat, Any

import numpy as np
import random

from gymnasium.core import ActType, ObsType

from .half_cheetah import HalfCheetahEnv


class HalfCheetahDirEnv(HalfCheetahEnv):
    """Half-cheetah environment with target direction, as described in [1]. The
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand_direc.py

    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each
    time step a reward composed of a control cost and a reward equal to its
    velocity in the target direction. The tasks are generated by sampling the
    target directions from a Bernoulli distribution on {-1, 1} with parameter
    0.5 (-1: backward, +1: forward).

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for
        model-based control", 2012
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """

    def __init__(self, n_tasks=None, max_episode_steps=200):
        self.n_tasks = n_tasks
        assert n_tasks == None
        self._goal = self._sample_raw_task()["goal"]
        self.spec.max_episode_steps = max_episode_steps
        super(HalfCheetahDirEnv, self).__init__()

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self._goal * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward, reward_ctrl=-ctrl_cost)
        return observation, reward, done, done, infos  # TODO this might not be right

    def get_current_task(self):
        # for multi-task MDP
        return np.array([self._goal])

    def _sample_raw_task(self):
        # for fwd/bwd env, goal direc is backwards if - 1.0, forwards if + 1.0
        direction = np.random.choice([-1.0, 1.0])  # 180 degree
        task = {"goal": direction}
        return task

    def reset_task(self, task_info):
        assert task_info is None
        self._goal = self._sample_raw_task()[
            "goal"
        ]  # assume parameterization of task by single vector
        self.reset()


class HalfCheetahRandDirOracleEnv(HalfCheetahDirEnv):
    def _get_obs(self):
        return (
            np.concatenate(
                [
                    self.sim.data.qpos.flat[1:],
                    self.sim.data.qvel.flat,
                    self.get_body_com("torso").flat,
                    [self._goal],
                ]
            )
            .astype(np.float32)
            .flatten()
        )
