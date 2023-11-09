import os, sys
import time

import math
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import numpy as np
import torch
import wandb

# import wandb
from torch import Tensor
from torch.nn import functional as F
import gymnasium as gym

from buffers import SeqReplayBufferGPU
from envs.meta.maze.MultitaskMaze import MazeTask
from policies.models import AGENT_CLASSES, AGENT_ARCHS
from torchkit.networks import ImageEncoder

# Markov policy
from buffers.simple_replay_buffer import SimpleReplayBuffer

# RNN policy on vector-based task
from buffers.seq_replay_buffer_vanilla import SeqReplayBuffer

# RNN policy on image/vector-based task
from buffers.seq_replay_buffer_efficient import RAMEfficient_SeqReplayBuffer

from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from utils import logger


@dataclass
class EvaluationResults:
    returns_per_episode: np.ndarray | dict = field(
        default_factory=lambda: np.ndarray(0)
    )
    success_rate: np.ndarray = field(default_factory=lambda: np.ndarray(0))
    observations: np.ndarray = field(default_factory=lambda: np.ndarray(0))
    total_steps: np.ndarray = field(default_factory=lambda: np.ndarray(0))
    left_steps: np.ndarray = field(default_factory=lambda: np.ndarray(0))
    blocked_indices: np.ndarray = field(default_factory=lambda: np.ndarray(0))
    unblocked_indices: np.ndarray = field(default_factory=lambda: np.ndarray(0))


class Learner:
    def __init__(self, env_args, train_args, eval_args, policy_args, seed, **kwargs):
        self.seed = seed

        self._init_env(**env_args)

        self.init_agent(**policy_args, tasks=self.train_tasks)

        self.init_train(**train_args)

        self.init_eval(**eval_args)

    def init_env(
        self,
        env_type: str,
        env_name: str,
        num_eval_tasks: Optional[int] = None,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        if self.env_type in [
            "pomdp",
            "credit",
        ]:  # pomdp/mdp task, using pomdp wrapper
            import envs.pomdp

            # import envs.credit_assign

            assert num_eval_tasks > 0
            self.train_env = gym.make(env_name, render_mode=render_mode)
            self.train_env.reset(seed=self.seed)
            # self.train_env.action_space.np_random.seed(self.seed)  # TODO 'crucial' but not possible

            self.eval_env = self.train_env
            self.eval_env.reset(seed=self.seed + 1)

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env.spec.max_episode_steps

        else:
            raise ValueError

    def _init_env(self, env_type: str, valid_actions: Any, **kwargs):
        # initialize environment
        assert env_type in [
            "meta",
            "pomdp",
            "credit",
            "rmdp",
            "generalize",
            "atari",
        ]
        self.env_type = env_type
        self.env_valid_actions = valid_actions == True
        self.init_env(env_type=env_type, **kwargs)

        # get action / observation dimensions
        if self.train_env.action_space.__class__.__name__ == "Box":
            # continuous action space
            self.act_dim = self.train_env.action_space.shape[0]
            self.act_continuous = True
        else:
            assert self.train_env.action_space.__class__.__name__ == "Discrete"
            self.act_dim = self.train_env.action_space.n
            self.act_continuous = False
        self.obs_dim: int = self.train_env.observation_space.shape[
            0
        ]  # include 1-dim done
        try:
            self.task_dim: Optional[int] = self.train_env.get_wrapper_attr("task_dim")
        except AttributeError:
            self.task_dim: Optional[int] = None
        logger.log(
            "obs_dim", self.obs_dim, "act_dim", self.act_dim, "task_dim", self.task_dim
        )

    def init_agent(
        self,
        seq_model: str,
        separate: bool = True,
        image_encoder=None,
        reward_clip: bool = False,
        **kwargs,
    ):
        """Initialize the agent that is going to be taking actions in the environment."""
        # initialize agent
        rnn_encoder_type: Optional[str] = None
        if seq_model == "mlp":
            agent_class = AGENT_CLASSES["Policy_MLP"]
            assert separate
        elif "-mlp" in seq_model:
            agent_class = AGENT_CLASSES["Policy_RNN_MLP"]
            rnn_encoder_type = seq_model.split("-")[0]
            assert separate
        else:
            rnn_encoder_type = seq_model
            if separate:
                agent_class = AGENT_CLASSES["Policy_Separate_RNN"]
            else:
                agent_class = AGENT_CLASSES["Policy_Shared_RNN"]

        self.agent_arch = agent_class.ARCH
        logger.log(agent_class, self.agent_arch)

        if image_encoder is not None:  # catch, keytodoor
            image_encoder_fn = lambda: ImageEncoder(
                image_shape=self.train_env.image_space.shape, **image_encoder
            )
        else:
            image_encoder_fn = lambda: None

        self.agent = agent_class(
            encoder=rnn_encoder_type,
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            task_dim=self.task_dim,
            image_encoder_fn=image_encoder_fn,
            **kwargs,
        ).to(ptu.device)
        logger.log(self.agent)

        self.reward_clip = reward_clip  # for atari

    def init_train(
        self,
        buffer_size,
        batch_size,
        num_iters,
        num_init_rollouts_pool,
        num_rollouts_per_iter,
        num_updates_per_iter=None,
        sampled_seq_len=None,
        sample_weight_baseline=None,
        buffer_type=None,
        **kwargs,
    ):
        """Initializes the buffer."""

        if num_updates_per_iter is None:
            num_updates_per_iter = 1.0
        assert isinstance(num_updates_per_iter, int) or isinstance(
            num_updates_per_iter, float
        )
        # if int, it means absolute value; if float, it means the multiplier of collected env steps
        self.num_updates_per_iter = num_updates_per_iter

        if self.agent_arch == AGENT_ARCHS.Markov:
            self.policy_storage = SimpleReplayBuffer(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                max_trajectory_len=self.max_trajectory_len,
                add_timeout=False,  # no timeout storage
            )

        else:  # memory, memory-markov
            if sampled_seq_len == -1:
                sampled_seq_len = self.max_trajectory_len

            if buffer_type is None or buffer_type == SeqReplayBuffer.buffer_type:
                buffer_class = SeqReplayBuffer
            elif buffer_type == RAMEfficient_SeqReplayBuffer.buffer_type:
                buffer_class = RAMEfficient_SeqReplayBuffer
            logger.log(buffer_class)

            self.policy_storage = buffer_class(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                sampled_seq_len=sampled_seq_len,
                sample_weight_baseline=sample_weight_baseline,
                observation_type=self.train_env.observation_space.dtype,
                task_dim=self.task_dim,
                state_dim=2,  # Only if directly using agent positions
            )

        self.batch_size = batch_size
        self.num_iters = num_iters
        self.num_init_rollouts_pool = num_init_rollouts_pool
        self.num_rollouts_per_iter = num_rollouts_per_iter

        total_rollouts = num_init_rollouts_pool + num_iters * num_rollouts_per_iter
        self.n_env_steps_total = self.max_trajectory_len * total_rollouts
        logger.log(
            "*** total rollouts",
            total_rollouts,
            "total env steps",
            self.n_env_steps_total,
        )

    def init_eval(
        self,
        log_interval,
        save_interval,
        log_tensorboard,
        eval_stochastic=False,
        num_episodes_per_task=1,
        **kwargs,
    ):
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.log_tensorboard = log_tensorboard
        self.eval_stochastic = eval_stochastic
        self.eval_num_episodes_per_task = num_episodes_per_task
        self.total_eval_success = 0

    def _reset_train_variables(self):
        self._n_env_steps_total = 0
        self._n_env_steps_total_last = 0
        self._n_rl_update_steps_total = 0
        self._n_rollouts_total = 0
        self._successes_in_buffer = 0

        self._start_time = time.time()
        self._start_time_last = time.time()

    def train(self):
        """
        training loop
        """

        self._reset_train_variables()

        if self.num_init_rollouts_pool > 0:
            logger.log("Collecting initial pool of data..")
            while (
                self._n_env_steps_total
                < self.num_init_rollouts_pool * self.max_trajectory_len
            ):
                self.collect_rollouts(
                    num_rollouts=1, random_actions=True, observe_uncertainty=False
                )
            logger.log(
                "Done! env steps",
                self._n_env_steps_total,
                "rollouts",
                self._n_rollouts_total,
            )

            if isinstance(self.num_updates_per_iter, float):
                # update: pomdp task updates more for the first iter_
                train_stats = self.update(
                    int(self._n_env_steps_total * self.num_updates_per_iter)
                )
                self.log_train_stats(train_stats)

        last_eval_num_iters = current_num_iters = 0
        last_eval_env_steps = 0
        while self._n_env_steps_total < self.n_env_steps_total:
            # collect data from num_rollouts_per_iter train tasks:
            env_steps = self.collect_rollouts(num_rollouts=self.num_rollouts_per_iter)
            logger.log("env steps", self._n_env_steps_total)

            train_stats = self.update(
                self.num_updates_per_iter
                if isinstance(self.num_updates_per_iter, int)
                else int(math.ceil(self.num_updates_per_iter * env_steps))
            )  # NOTE: ceil to make sure at least 1 step
            self.log_train_stats(train_stats)

            # evaluate and log
            current_num_iters = self._n_env_steps_total // (
                self.num_rollouts_per_iter * self.max_trajectory_len
            )
            if last_eval_env_steps + self.log_interval <= self._n_env_steps_total:
                # if (
                #     current_num_iters != last_eval_num_iters
                #     and current_num_iters % self.log_interval == 0
                # ):
                #     last_eval_num_iters = current_num_iters
                last_eval_env_steps = self._n_env_steps_total
                perf = self.log()
                if (
                    self.save_interval > 0
                    and self._n_env_steps_total > 0.75 * self.n_env_steps_total
                    and current_num_iters % self.save_interval == 0
                ):
                    # save models in later training stage
                    self.save_model(current_num_iters, perf)
        self.save_model(current_num_iters, perf)

    @torch.no_grad()
    def collect_rollouts(
        self, num_rollouts: int, random_actions=False, observe_uncertainty=True
    ) -> int:
        """collect num_rollouts of trajectories in task and save into policy buffer
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        """

        before_env_steps = self._n_env_steps_total
        for idx in range(num_rollouts):
            steps = 0

            if (
                self.env_type == "meta"
                and self.train_env.get_wrapper_attr("n_tasks") is not None
            ):
                task_index = self.train_tasks[np.random.randint(len(self.train_tasks))]
                obs_numpy, info = self.train_env.reset(
                    seed=self.seed, options={"task": task_index, "train_mode": True}
                )
            else:
                obs_numpy, info = self.train_env.reset(
                    seed=self.seed, options={"train_mode": True}
                )

            obs = ptu.from_numpy(obs_numpy)  # reset task
            obs = obs.reshape(1, obs.shape[-1])
            task_embedding: Optional[Tensor] = (
                info["embedding"].unsqueeze(0) if "embedding" in info else None
            )
            done_rollout = False

            if self.agent_arch == AGENT_ARCHS.Memory:
                # get hidden state at timestep=0, None for markov
                # NOTE: assume initial reward = 0.0 (no need to clip)
                action, reward, internal_state = self.agent.get_initial_info(
                    task=task_embedding
                )

            if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
                # temporary storage
                obs_list, act_list, rew_list, next_obs_list, term_list = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                orig_state_list = []

            while not done_rollout:
                valid_actions: Optional[np.ndarray] = (
                    info.get("valid_actions") if self.env_valid_actions else None
                )
                if random_actions:
                    action = self.select_random_action(valid_actions)
                else:
                    # policy takes hidden state as input for memory-based actor,
                    # while takes obs for markov actor
                    if self.agent_arch == AGENT_ARCHS.Memory:
                        intrinsic_reward = 0
                        if "agent_position" in info:
                            intrinsic_reward = self.agent.uncertainty(
                                info["agent_position"],
                            ).squeeze(0)
                        (action, _, _, _), internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            reward=reward + intrinsic_reward,
                            obs=obs,
                            deterministic=False,
                            task=task_embedding,
                            valid_actions=valid_actions,
                        )
                    else:
                        action, _, _, _ = self.agent.act(
                            obs, deterministic=False, task=task_embedding
                        )

                # observe reward and next obs (B=1, dim)
                next_obs, reward, terminated, truncated, info = utl.env_step(
                    self.train_env, action.squeeze(dim=0)
                )
                if self.reward_clip and self.env_type == "atari":
                    reward = torch.tanh(reward)

                done_rollout = (
                    ptu.get_numpy(terminated[0][0]) == 1.0
                    or ptu.get_numpy(truncated[0][0]) == 1.0
                )
                # update statistics
                steps += 1

                ## determine terminal flag per environment
                if self.env_type == "meta" and "success" in info:
                    # NOTE: following varibad practice: for meta env, even if reaching the goal (term=True),
                    # the episode still continues.
                    self._successes_in_buffer += int(info["success"])
                elif self.env_type == "credit":  # delayed rewards
                    term = done_rollout
                else:
                    # term ignore time-out scenarios, but record early stopping
                    term = (
                        "TimeLimit.truncated" not in info
                        and steps < self.max_trajectory_len
                        and done_rollout
                    )

                # add data to policy buffer
                if self.agent_arch == AGENT_ARCHS.Markov:
                    self.policy_storage.add_sample(
                        observation=ptu.get_numpy(obs.squeeze(dim=0)),
                        action=ptu.get_numpy(
                            action.squeeze(dim=0)
                            if self.act_continuous
                            else torch.argmax(
                                action.squeeze(dim=0), dim=-1, keepdims=True
                            )  # (1,)
                        ),
                        reward=ptu.get_numpy(reward.squeeze(dim=0)),
                        terminal=np.array([term], dtype=float),
                        next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)),
                    )
                else:  # append tensors to temporary storage
                    obs_list.append(obs)  # (1, dim)
                    act_list.append(action)  # (1, dim)
                    rew_list.append(reward)  # (1, dim)
                    term_list.append(term)  # bool
                    next_obs_list.append(next_obs)  # (1, dim)

                    if "agent_position" in info:
                        orig_state_list.append(info["agent_position"])

                # set: obs <- next_obs
                obs = next_obs.clone()

            if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
                # add collected sequence to buffer
                act_buffer = torch.cat(act_list, dim=0)  # (L, dim)
                if not self.act_continuous:
                    act_buffer = torch.argmax(
                        act_buffer, dim=-1, keepdims=True
                    )  # (L, 1)

                task_embedding = (
                    None if task_embedding is None else ptu.get_numpy(task_embedding)
                )
                orig_states = None
                if len(orig_state_list) > 0:
                    orig_states_tensor = torch.cat(orig_state_list, dim=0)
                    orig_states = ptu.get_numpy(orig_states_tensor)

                    if observe_uncertainty:
                        self.agent.uncertainty.observe(orig_states_tensor)

                self.policy_storage.add_episode(
                    observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                    actions=ptu.get_numpy(act_buffer),  # (L, dim)
                    rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                    terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                    next_observations=ptu.get_numpy(
                        torch.cat(next_obs_list, dim=0)
                    ),  # (L, dim),
                    task=task_embedding,  # (L, dim),
                    orig_states=orig_states,
                )
                # print(
                #     f"steps: {steps} term: {term} ret: {torch.cat(rew_list, dim=0).sum().item():.2f}"
                # )
            self._n_env_steps_total += steps
            self._n_rollouts_total += 1
        return self._n_env_steps_total - before_env_steps

    def select_random_action(self, valid_actions: Optional[np.ndarray]) -> Tensor:
        action = ptu.FloatTensor(
            [self.train_env.action_space.sample()]
        )  # (1, A) for continuous action, (1) for discrete action
        if not self.act_continuous:
            if valid_actions is not None:
                valid_action_index = (action.long() % valid_actions.sum(axis=-1)).item()
                action = ptu.tensor(
                    [np.where(valid_actions == 1)[0][valid_action_index]]
                )
            action = F.one_hot(
                action.long(), num_classes=self.act_dim
            ).float()  # (1, A)

        return action

    def sample_rl_batch(self, batch_size) -> dict[str, Tensor]:
        """sample batch of episodes for vae training"""
        if self.agent_arch == AGENT_ARCHS.Markov:
            batch = self.policy_storage.random_batch(batch_size)
        else:  # rnn: all items are (sampled_seq_len, B, dim)
            batch = self.policy_storage.random_episodes(batch_size)
        return ptu.np_to_pytorch_batch(batch)

    def update(self, num_updates: int) -> dict[str, float]:
        rl_losses_agg = {}
        for update in range(num_updates):
            # sample random RL batch: in transitions
            batch = self.sample_rl_batch(self.batch_size)

            # RL update
            rl_losses = self.agent.update(batch)

            for k, v in rl_losses.items():
                if update == 0:  # first iterate - create list
                    rl_losses_agg[k] = [v]
                else:  # append values
                    rl_losses_agg[k].append(v)
        # statistics
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        self._n_rl_update_steps_total += num_updates

        return rl_losses_agg

    @torch.no_grad()
    def evaluate(
        self, tasks: list[MazeTask], deterministic: bool = True
    ) -> EvaluationResults:
        is_task_blocked = np.ones(len(tasks))
        num_episodes = self.max_rollouts_per_task  # k
        # max_trajectory_len = k*H
        results = EvaluationResults()
        results.returns_per_episode = np.zeros((len(tasks), num_episodes))
        results.success_rate = np.zeros(len(tasks))
        results.total_steps = np.zeros(len(tasks))
        results.left_steps = np.zeros(len(tasks))

        if self.env_type == "meta":
            num_steps_per_episode = self.eval_env.spec.max_episode_steps  # H
            obs_size = self.eval_env.observation_space.shape[0]  # original size
            results.observations = np.zeros(
                (len(tasks), self.max_trajectory_len + 1, obs_size)
            )
        else:  # pomdp, rmdp, generalize
            num_steps_per_episode = self.eval_env.spec.max_episode_steps
            results.observations = None

        for task_idx, task in enumerate(tasks):
            step = 0

            if (
                self.env_type == "meta"
                and self.eval_env.get_wrapper_attr("n_tasks") is not None
            ):
                obs_numpy, info = self.eval_env.reset(
                    seed=self.seed + 1, options={"task": task}
                )
                obs = ptu.from_numpy(obs_numpy)  # reset task
                results.observations[task_idx, step, :] = ptu.get_numpy(obs[:obs_size])
            else:
                obs_numpy, info = self.eval_env.reset(seed=self.seed + 1)
                obs = ptu.from_numpy(obs_numpy)  # reset

            if "blocked" in info and not info["blocked"]:
                is_task_blocked[task_idx] = 0

            obs = obs.reshape(1, obs.shape[-1])
            task_embedding: Optional[Tensor] = (
                info["embedding"].unsqueeze(0) if "embedding" in info else None
            )
            if self.agent_arch == AGENT_ARCHS.Memory:
                # assume initial reward = 0.0
                action, reward, internal_state = self.agent.get_initial_info(
                    task=task_embedding
                )

            for episode_idx in range(num_episodes):
                running_reward = 0.0
                for _ in range(num_steps_per_episode):
                    valid_actions: Optional[np.ndarray] = (
                        info.get("valid_actions") if self.env_valid_actions else None
                    )
                    if self.agent_arch == AGENT_ARCHS.Memory:
                        intrinsic_reward = 0
                        if "agent_position" in info:
                            intrinsic_reward = self.agent.uncertainty(
                                info["agent_position"],
                            ).squeeze(0)
                        (action, _, _, _), internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            reward=reward + intrinsic_reward,
                            obs=obs,
                            deterministic=deterministic,
                            task=task_embedding,
                            valid_actions=valid_actions,
                        )
                    else:
                        action, _, _, _ = self.agent.act(
                            obs,
                            deterministic=deterministic,
                            task=task_embedding,
                            valid_actions=valid_actions,
                        )

                    # observe reward and next obs
                    next_obs, reward, terminated, truncated, info = utl.env_step(
                        self.eval_env, action.squeeze(dim=0)
                    )

                    # add raw reward
                    running_reward += reward.item()
                    # clip reward if necessary for policy inputs
                    if self.reward_clip and self.env_type == "atari":
                        reward = torch.tanh(reward)

                    step += 1
                    done_rollout = (
                        ptu.get_numpy(terminated[0][0]) == 1.0
                        or ptu.get_numpy(truncated[0][0]) == 1.0
                    )

                    if self.env_type == "meta":
                        results.observations[task_idx, step, :] = ptu.get_numpy(
                            next_obs[0, :obs_size]
                        )

                    # set: obs <- next_obs
                    obs = next_obs.clone()

                    # if (
                    #     self.env_type == "generalize"
                    #     and self.eval_env.unwrapped.is_success()
                    # ):
                    #     results.success_rate[task_idx] = 1.0  # ever once reach
                    # elif (
                    #     self.env_type == "meta"
                    #     and "is_goal_state" in dir(self.eval_env.unwrapped)
                    #     and self.eval_env.unwrapped.is_goal_state()
                    # ):
                    #     results.success_rate[task_idx] = 1.0  # ever once reach
                    #     self.total_eval_success += 1
                    if done_rollout:
                        # for all env types, same
                        break
                    if self.env_type == "meta" and info["done_mdp"]:
                        # for early stopping meta episode like Ant-Dir
                        break

                if "success" in info and info["success"]:  # keytodoor
                    results.success_rate[task_idx] = 1.0
                    self.total_eval_success += 1
                if "left_counter" in info:
                    results.left_steps[task_idx] = info["left_counter"]

                results.returns_per_episode[task_idx, episode_idx] = running_reward
            results.total_steps[task_idx] = step

        results.blocked_indices = np.where(is_task_blocked)
        results.unblocked_indices = np.where(1 - is_task_blocked)
        return results

    def log_train_stats(self, train_stats: dict[str, float]) -> None:
        logger.record_step(self._n_env_steps_total)

        # Log losses
        for key, value in train_stats.items():
            logger.record_tabular("rl_loss/" + key, value)

        # Gradient norms
        if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
            results = self.agent.report_grad_norm()
            for key, value in results.items():
                logger.record_tabular("rl_loss/" + key, value)

        logger.dump_tabular()

    def log_evaluate(self) -> EvaluationResults:
        if self.env_type in ["pomdp", "credit", "atari"]:
            eval_results = self.evaluate(self.eval_tasks)
            logger.record_tabular(
                "metrics/total_steps_eval", np.mean(eval_results.total_steps)
            )
            logger.record_tabular(
                "metrics/return_eval_total",
                np.mean(np.sum(eval_results.returns_per_episode, axis=-1)),
            )
            logger.record_tabular(
                "metrics/success_rate_eval", np.mean(eval_results.success_rate)
            )
            logger.record_tabular("metrics/total_eval_success", self.total_eval_success)

            if self.eval_stochastic:
                eval_sto = self.evaluate(self.eval_tasks, deterministic=False)
                logger.record_tabular(
                    "metrics/total_steps_eval_sto", np.mean(eval_sto.total_steps)
                )
                logger.record_tabular(
                    "metrics/return_eval_total_sto",
                    np.mean(np.sum(eval_sto.returns_per_episode, axis=-1)),
                )
                logger.record_tabular(
                    "metrics/success_rate_eval_sto", np.mean(eval_results.success_rate)
                )
            return eval_results
        else:
            raise ValueError

    def log(self):
        # --- log training  ---
        # Set env steps for tensorboard: z is for lowest order
        logger.record_step(self._n_env_steps_total)
        logger.record_tabular("z/env_steps", self._n_env_steps_total)
        logger.record_tabular("z/rollouts", self._n_rollouts_total)
        logger.record_tabular("z/rl_steps", self._n_rl_update_steps_total)

        # --- evaluation ----
        eval_results = self.log_evaluate()

        logger.record_tabular("z/time_cost", int(time.time() - self._start_time))
        logger.record_tabular(
            "z/fps",
            (self._n_env_steps_total - self._n_env_steps_total_last)
            / (time.time() - self._start_time_last),
        )
        self._n_env_steps_total_last = self._n_env_steps_total
        self._start_time_last = time.time()

        # if wandb.run is not None:
        #     wandb.log(logger.getkvs())

        logger.dump_tabular()

        if self.env_type == "generalize":
            return sum([v.mean() for v in eval_results.success_rate.values()]) / len(
                eval_results.success_rate
            )
        else:
            return np.mean(np.sum(eval_results.returns_per_episode, axis=-1))

    def save_model(self, iter, perf):
        save_path = os.path.join(
            logger.get_dir(), "save", f"agent_{iter}_perf{perf:.3f}.pt"
        )
        torch.save(self.agent.state_dict(), save_path)

    def load_model(self, ckpt_path):
        self.agent.load_state_dict(torch.load(ckpt_path, map_location=ptu.device))
        print("load successfully from", ckpt_path)
