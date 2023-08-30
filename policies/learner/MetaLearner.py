from typing import Optional

import numpy as np

from .Learner import Learner
from utils import logger


class MetaLearner(Learner):
    def init_env(
            self,
            env_type: str,
            env_name: str,
            max_rollouts_per_task: Optional[int] = None,
            num_tasks: Optional[int] = None,
            num_train_tasks: Optional[int] = None,
            num_eval_tasks: Optional[int] = None,
            **kwargs
    ):

        # initialize environment
        assert env_type == "meta"
        from envs.meta.make_env import make_env

        self.train_env = make_env(
            env_name,
            max_rollouts_per_task,
            seed=self.seed,
            n_tasks=num_tasks,
            **kwargs,
        )  # oracle in kwargs
        self.eval_env = self.train_env
        self.eval_env.seed(self.seed + 1)

        if self.train_env.n_tasks is not None:
            # NOTE: This is off-policy varibad's setting, i.e. limited training tasks
            # split to train/eval tasks
            assert num_train_tasks >= num_eval_tasks > 0
            shuffled_tasks = np.random.permutation(
                self.train_env.unwrapped.get_all_task_idx()
            )
            self.train_tasks = shuffled_tasks[:num_train_tasks]
            self.eval_tasks = shuffled_tasks[-num_eval_tasks:]
        else:
            # NOTE: This is on-policy varibad's setting, i.e. unlimited training tasks
            assert num_tasks == num_train_tasks == None
            assert (
                    num_eval_tasks > 0
            )  # to specify how many tasks to be evaluated each time
            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

        # calculate what the maximum length of the trajectories is
        self.max_rollouts_per_task = max_rollouts_per_task
        self.max_trajectory_len = self.train_env.horizon_bamdp  # H^+ = k * H

    def log_evaluate(self) -> tuple[np.ndarray, np.ndarray]:
        if self.train_env.n_tasks is not None:
            (
                returns_train,
                success_rate_train,
                observations,
                total_steps_train,
            ) = self.evaluate(self.train_tasks[: len(self.eval_tasks)])
        (
            returns_eval,
            success_rate_eval,
            observations_eval,
            total_steps_eval,
        ) = self.evaluate(self.eval_tasks)
        if self.eval_stochastic:
            (
                returns_eval_sto,
                success_rate_eval_sto,
                observations_eval_sto,
                total_steps_eval_sto,
            ) = self.evaluate(self.eval_tasks, deterministic=False)

        if self.train_env.n_tasks is not None and "plot_behavior" in dir(
                self.eval_env.unwrapped
        ):
            # plot goal-reaching trajs
            for i, task in enumerate(
                    self.train_tasks[: min(5, len(self.eval_tasks))]
            ):
                self.eval_env.reset(task=task, seed=self.seed + 1)  # must have task argument
                logger.add_figure(
                    "trajectory/train_task_{}".format(i),
                    utl_eval.plot_rollouts(observations[i, :], self.eval_env),
                )

            for i, task in enumerate(
                    self.eval_tasks[: min(5, len(self.eval_tasks))]
            ):
                self.eval_env.reset(task=task, seed=self.seed + 1)
                logger.add_figure(
                    "trajectory/eval_task_{}".format(i),
                    utl_eval.plot_rollouts(observations_eval[i, :], self.eval_env),
                )
                if self.eval_stochastic:
                    logger.add_figure(
                        "trajectory/eval_task_{}_sto".format(i),
                        utl_eval.plot_rollouts(
                            observations_eval_sto[i, :], self.eval_env
                        ),
                    )

        if "is_goal_state" in dir(
                self.eval_env.unwrapped
        ):  # goal-reaching success rates
            # some metrics
            logger.record_tabular(
                "metrics/successes_in_buffer",
                self._successes_in_buffer / self._n_env_steps_total,
            )
            if self.train_env.n_tasks is not None:
                logger.record_tabular(
                    "metrics/success_rate_train", np.mean(success_rate_train)
                )
            logger.record_tabular(
                "metrics/success_rate_eval", np.mean(success_rate_eval)
            )
            if self.eval_stochastic:
                logger.record_tabular(
                    "metrics/success_rate_eval_sto", np.mean(success_rate_eval_sto)
                )

        for episode_idx in range(self.max_rollouts_per_task):
            if self.train_env.n_tasks is not None:
                logger.record_tabular(
                    "metrics/return_train_episode_{}".format(episode_idx + 1),
                    np.mean(returns_train[:, episode_idx]),
                )
            logger.record_tabular(
                "metrics/return_eval_episode_{}".format(episode_idx + 1),
                np.mean(returns_eval[:, episode_idx]),
            )
            if self.eval_stochastic:
                logger.record_tabular(
                    "metrics/return_eval_episode_{}_sto".format(episode_idx + 1),
                    np.mean(returns_eval_sto[:, episode_idx]),
                )

        if self.train_env.n_tasks is not None:
            logger.record_tabular(
                "metrics/total_steps_train", np.mean(total_steps_train)
            )
            logger.record_tabular(
                "metrics/return_train_total",
                np.mean(np.sum(returns_train, axis=-1)),
            )
        logger.record_tabular("metrics/total_steps_eval", np.mean(total_steps_eval))
        logger.record_tabular(
            "metrics/return_eval_total", np.mean(np.sum(returns_eval, axis=-1))
        )
        if self.eval_stochastic:
            logger.record_tabular(
                "metrics/total_steps_eval_sto", np.mean(total_steps_eval_sto)
            )
            logger.record_tabular(
                "metrics/return_eval_total_sto",
                np.mean(np.sum(returns_eval_sto, axis=-1)),
            )

        return returns_eval, success_rate_eval
