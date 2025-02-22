from typing import Optional, Any, SupportsFloat

from gymnasium.core import ActType, ObsType
import gymnasium as gym
import numpy as np
from gymnasium import spaces


def mujoco_wrapper(entry_point, **kwargs):
    # Load the environment from its entry point
    from gymnasium.envs.registration import load

    env_cls = load(entry_point)
    env = env_cls(**kwargs)
    return env


class VariBadWrapper(gym.Wrapper):
    def __init__(
        self,
        env: str,
        episodes_per_task: int,
        oracle: bool = False,  # default no
        **kwargs,
    ):
        """
        Wrapper, creates a multi-episode (BA)MDP around a one-episode MDP. Automatically deals with
        - horizons H in the MDP vs horizons H+ in the BAMDP,
        - resetting the tasks
        - normalized actions in case of continuous action space
        - adding the timestep / done info to the state (might be needed to make states markov)
        """
        env_cls = gym.make(env, **kwargs)
        super().__init__(env_cls)

        # if continuous actions, make sure in [-1, 1]
        # NOTE: policy won't use action_space.low/high, just set [-1,1]
        # this is a bad practice...
        if isinstance(self.env.action_space, gym.spaces.Box):
            self._normalize_actions = True
        else:
            self._normalize_actions = False

        self.oracle = oracle
        if self.oracle == True:
            print("WARNING: YOU ARE RUNNING MDP, NOT POMDP!\n")
            tmp_task = self.env.get_current_task()
            self.observation_space = spaces.Box(
                low=np.array(
                    [*self.observation_space.low, *([0] * len(tmp_task))]
                ),  # shape will be deduced from this
                high=np.array([*self.observation_space.high, *([1] * len(tmp_task))]),
                dtype=np.float32,
            )

        self.add_done_info = episodes_per_task > 1
        if self.add_done_info:
            self.observation_space = spaces.Box(
                low=np.array(
                    [*self.observation_space.low, 0]
                ),  # shape will be deduced from this
                high=np.array([*self.observation_space.high, 1]),
                dtype=np.float32,
            )

        # calculate horizon length H^+
        self.episodes_per_task = episodes_per_task
        # counts the number of episodes
        self.episode_count = 0

        # count timesteps in BAMDP
        self.step_count_bamdp = 0.0
        # the horizon in the BAMDP is the one in the MDP times the number of episodes per task,
        # and if we train a policy that maximises the return over all episodes
        # we add transitions to the reset start in-between episodes
        try:
            self.horizon_bamdp = (
                self.episodes_per_task * self.env.spec.max_episode_steps
            )
        except AttributeError:
            self.horizon_bamdp = (
                self.episodes_per_task * self.env.unwrapped.spec.max_episode_steps
            )

        # this tells us if we have reached the horizon in the underlying MDP
        self.done_mdp = True

    def _get_obs(self, state):
        if self.oracle:
            tmp_task = self.env.get_current_task().copy()
            state = np.concatenate([state, tmp_task])
        if self.add_done_info:
            state = np.concatenate((state, [float(self.done_mdp)]))
        return state

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict]:
        assert "task" in options
        task = options["task"]

        # reset task -- this sets goal and state -- sets self.env._goal and self.env._state
        task_reset = self.env.get_wrapper_attr("reset_task")
        task_reset(task)

        self.episode_count = 0
        self.step_count_bamdp = 0

        # normal reset
        try:
            state, info = self.env.reset(seed=seed, options=options)
        except AttributeError:
            state, info = self.env.unwrapped.reset(seed=seed, options=options)

        self.done_mdp = False

        return self._get_obs(state), info

    def wrap_state_with_done(self, state: np.ndarray) -> np.ndarray:
        # for some custom evaluation like semicircle
        if self.add_done_info:
            state = np.concatenate((state, [float(self.done_mdp)]))
        return state

    def reset_mdp(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        state, info = self.env.reset(seed=seed, options=options)
        self.done_mdp = False

        return self._get_obs(state), info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self._normalize_actions:  # from [-1, 1] to [lb, ub]
            action = np.clip(action, -1, 1)  # first clip into [-1, 1]
            lb = self.env.action_space.low
            ub = self.env.action_space.high
            action = lb + (action + 1.0) * 0.5 * (ub - lb)
            action = np.clip(action, lb, ub)

        # do normal environment step in MDP
        # TODO could be that truncated/terminated is not done correctly here
        state, reward, self.done_mdp, truncated, info = self.env.step(action)

        info["done_mdp"] = self.done_mdp
        state = self._get_obs(state)

        self.step_count_bamdp += 1
        # if we want to maximise performance over multiple episodes,
        # only say "done" when we collected enough episodes in this task
        done_bamdp = False
        if self.done_mdp:
            self.episode_count += 1
            if self.episode_count == self.episodes_per_task:
                done_bamdp = True

        if self.done_mdp and not done_bamdp:
            info["start_state"] = self.reset_mdp()

        return state, reward, done_bamdp, truncated, info


class TimeLimitMask(gym.Wrapper):
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, rew, terminated, truncated, info = self.env.step(action)
        if truncated and self.env.spec.max_episode_steps == self.env._elapsed_steps:
            info["bad_transition"] = True

        return obs, rew, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
