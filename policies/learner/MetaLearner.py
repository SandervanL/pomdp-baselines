from collections import defaultdict
from functools import reduce
from typing import Optional

import numpy as np

from .Learner import Learner, EvaluationResults
from utils import logger
from utils import evaluation as utl_eval


class MetaLearner(Learner):
    def init_env(
        self,
        env_type: str,
        env_name: str,
        task_selection: str = "random",
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
            n_tasks=num_tasks,
            **kwargs,
        )  # oracle in kwargs
        self.eval_env = self.train_env

        if self.train_env.get_wrapper_attr("n_tasks") is not None:
            # NOTE: This is off-policy varibad's setting, i.e. limited training tasks
            # split to train/eval tasks
            assert num_train_tasks >= num_eval_tasks > 0

            all_tasks_indices = self.train_env.get_wrapper_attr("get_all_task_idx")()

            if task_selection == "random":
                shuffled_tasks = np.random.permutation(all_tasks_indices)
                self.train_tasks = shuffled_tasks[:num_train_tasks]
                self.eval_tasks = shuffled_tasks[-num_eval_tasks:]
            elif task_selection == "even":
                self.train_tasks = all_tasks_indices
                self.eval_tasks = all_tasks_indices
            elif task_selection == "random-class":
                # TODO make this work with a percentage perhaps
                tasks = self.train_env.get_wrapper_attr("get_all_task_types")()
                tasks_by_class: list[list[int]] = list(
                    reduce(
                        lambda acc, x: acc[x[1]].append(x[0]),
                        tasks,
                        defaultdict(lambda: []),
                    ).values()
                )
                self.train_tasks = []
                self.eval_tasks = []
                for tasks in tasks_by_class:
                    assert num_eval_tasks <= num_train_tasks <= len(tasks)
                    shuffled_tasks = np.random.permutation(tasks)
                    self.train_tasks += shuffled_tasks[:num_train_tasks]
                    self.eval_tasks += shuffled_tasks[-num_eval_tasks:]
        else:
            # NOTE: This is on-policy varibad's setting, i.e. unlimited training tasks
            assert num_tasks is num_train_tasks is None
            assert (
                num_eval_tasks > 0
            )  # to specify how many tasks to be evaluated each time
            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

        # calculate what the maximum length of the trajectories is
        self.max_rollouts_per_task = max_rollouts_per_task
        self.max_trajectory_len = self.train_env.horizon_bamdp  # H^+ = k * H

    def log_evaluate(self) -> EvaluationResults:
        if self.train_env.get_wrapper_attr("n_tasks") is not None:
            train_results = self.evaluate(self.train_tasks[: len(self.eval_tasks)])

        eval_results = self.evaluate(self.eval_tasks)
        if self.eval_stochastic:
            eval_sto_results = self.evaluate(self.eval_tasks, deterministic=False)

        if self.train_env.get_wrapper_attr(
            "n_tasks"
        ) is not None and "plot_behavior" in dir(self.eval_env.unwrapped):
            # plot goal-reaching trajs
            for i, task in enumerate(self.train_tasks[: min(5, len(self.eval_tasks))]):
                self.eval_env.reset(
                    seed=self.seed + 1, options={"task": task}
                )  # must have task argument
                logger.add_figure(
                    "trajectory/train_task_{}".format(i),
                    utl_eval.plot_rollouts(
                        train_results.observations[i, :], self.eval_env
                    ),
                )

            for i, task in enumerate(self.eval_tasks[: min(5, len(self.eval_tasks))]):
                self.eval_env.reset(seed=self.seed + 1, options={"task": task})
                logger.add_figure(
                    "trajectory/eval_task_{}".format(i),
                    utl_eval.plot_rollouts(
                        eval_results.observations[i, :], self.eval_env
                    ),
                )
                if self.eval_stochastic:
                    logger.add_figure(
                        "trajectory/eval_task_{}_sto".format(i),
                        utl_eval.plot_rollouts(
                            eval_sto_results.observations[i, :], self.eval_env
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
            if self.train_env.get_wrapper_attr("n_tasks") is not None:
                logger.record_tabular(
                    "metrics/success_rate_train", np.mean(train_results.success_rate)
                )
            logger.record_tabular(
                "metrics/success_rate_eval", np.mean(eval_results.success_rate)
            )
            if self.eval_stochastic:
                logger.record_tabular(
                    "metrics/success_rate_eval_sto",
                    np.mean(eval_sto_results.success_rate),
                )

        for episode_idx in range(self.max_rollouts_per_task):
            if self.train_env.get_wrapper_attr("n_tasks") is not None:
                logger.record_tabular(
                    "metrics/return_train_episode_{}".format(episode_idx + 1),
                    np.mean(train_results.returns_per_episode[:, episode_idx]),
                )
            logger.record_tabular(
                "metrics/return_eval_episode_{}".format(episode_idx + 1),
                np.mean(eval_results.returns_per_episode[:, episode_idx]),
            )
            if self.eval_stochastic:
                logger.record_tabular(
                    "metrics/return_eval_episode_{}_sto".format(episode_idx + 1),
                    np.mean(eval_sto_results[:, episode_idx]),
                )

        if self.train_env.get_wrapper_attr("n_tasks") is not None:
            logger.record_tabular(
                "metrics/total_steps_train", np.mean(train_results.total_steps)
            )
            logger.record_tabular(
                "metrics/return_train_total",
                np.mean(np.sum(train_results.returns_per_episode, axis=-1)),
            )
        logger.record_tabular(
            "metrics/total_steps_eval", np.mean(eval_results.total_steps)
        )
        logger.record_tabular(
            "metrics/return_eval_total",
            np.mean(np.sum(eval_results.total_steps, axis=-1)),
        )
        if self.eval_stochastic:
            logger.record_tabular(
                "metrics/total_steps_eval_sto", np.mean(eval_sto_results.total_steps)
            )
            logger.record_tabular(
                "metrics/return_eval_total_sto",
                np.mean(np.sum(eval_sto_results.returns_per_episode, axis=-1)),
            )

        return eval_results
