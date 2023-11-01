from collections import defaultdict
from functools import reduce
from typing import Optional

import numpy as np
from gymnasium import Env

from envs.meta.maze.MultitaskMaze import MazeTask
from .Learner import Learner, EvaluationResults
from utils import logger
from utils import evaluation as utl_eval


class MetaLearner(Learner):
    def init_env(
        self,
        env_type: str,
        env_name: str,
        task_selection: str = "random",
        train_test_split: Optional[float] = None,
        max_rollouts_per_task: Optional[int] = None,
        num_train_tasks: Optional[int] = None,
        num_eval_tasks: Optional[int] = None,
        eval_on_train_tasks: bool = False,
        **kwargs,
    ):
        # initialize environment
        assert env_type == "meta"
        from envs.meta.make_env import make_env

        self.train_env = make_env(
            env_name,
            max_rollouts_per_task,
            **kwargs,
        )  # oracle in kwargs
        self.eval_env = self.train_env
        self.eval_on_train_tasks = eval_on_train_tasks

        if self.train_env.get_wrapper_attr("n_tasks") is not None:
            all_tasks_indices = self.train_env.get_wrapper_attr("get_all_task_idx")()
            # NOTE: This is off-policy varibad's setting, i.e. limited training tasks
            # split to train/eval tasks. If train_test_split is None, then num_train_tasks
            # and num_eval_tasks must be specified.
            train_test_split_present = (
                train_test_split is not None and 0 <= train_test_split <= 1
            )
            eval_tasks_present = (
                num_train_tasks is not None
                and num_eval_tasks is not None
                and num_train_tasks >= num_eval_tasks > 0
                and len(all_tasks_indices) >= num_train_tasks + num_eval_tasks
            )
            self.num_eval_tasks = None
            rng = np.random.default_rng(self.seed)

            shuffled_tasks = rng.permutation(all_tasks_indices)
            if task_selection == "random":
                assert train_test_split_present ^ eval_tasks_present
                if train_test_split_present:
                    num_train_tasks = int(train_test_split * len(all_tasks_indices))
                    num_eval_tasks = len(all_tasks_indices) - num_train_tasks

                self.train_tasks = shuffled_tasks[:num_train_tasks]
                self.eval_tasks = shuffled_tasks[-num_eval_tasks:]
            elif task_selection == "even":
                self.train_tasks = shuffled_tasks
                self.eval_tasks = shuffled_tasks
                if num_eval_tasks is not None:
                    assert (
                        num_eval_tasks <= len(shuffled_tasks)
                        and num_eval_tasks % 2 == 0
                    )
                    self.num_eval_tasks = num_eval_tasks
            elif task_selection == "random-word":
                assert train_test_split_present

                # Find which tasks belong to the same word
                tasks_by_class = np.array(get_tasks_by_type(self.train_env, "word"))
                shuffled_words = rng.permutation(list(range(len(tasks_by_class))))

                # Select which words to include in test and train
                num_train_words = int(train_test_split * tasks_by_class.shape[0])
                train_words = shuffled_words[:num_train_words]
                eval_words = shuffled_words[num_train_words:]

                self.train_tasks = tasks_by_class[train_words].reshape(-1)
                self.eval_tasks = tasks_by_class[eval_words].reshape(-1)
            elif task_selection == "random-within-word":
                # Add 80% of tasks that belong to the same word to train and 20% to eval
                assert train_test_split_present

                tasks_by_class = np.array(
                    get_tasks_by_type(self.train_env, "word")
                )  # (num_words, num_tasks_per_word)
                rng.shuffle(
                    tasks_by_class.transpose()
                )  # (num_tasks_per_word, num_words)

                # Select which words to include in test and train
                num_train_sentences = int(train_test_split * tasks_by_class.shape[1])

                # Select how many sentences to include in test and train
                self.train_tasks = tasks_by_class[:, num_train_sentences].reshape(-1)
                self.eval_tasks = tasks_by_class[:, num_train_sentences:].reshape(-1)
            else:
                raise ValueError(f"Unknown task selection '{task_selection}'")

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
        do_train_eval = (
            self.eval_on_train_tasks
            and self.train_env.get_wrapper_attr("n_tasks") is not None
        )
        if do_train_eval:
            train_results = self.evaluate(self.train_tasks[: len(self.eval_tasks)])

        eval_tasks = self.eval_tasks
        if self.num_eval_tasks is not None:
            task_list = get_tasks_by_type(self.eval_env, "task_type")
            eval_tasks = np.concatenate(
                [
                    np.random.choice(
                        task, size=self.num_eval_tasks // len(task_list), replace=False
                    )
                    for task in task_list
                ]
            )

        eval_results = self.evaluate(eval_tasks)
        if self.eval_stochastic:
            eval_sto_results = self.evaluate(self.eval_tasks, deterministic=False)

        if do_train_eval and "plot_behavior" in dir(self.eval_env.unwrapped):
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

        if (
            self.eval_env.get_wrapper_attr("is_goal_state") is not None
        ):  # goal-reaching success rates
            # some metrics
            logger.record_tabular(
                "metrics/successes_in_buffer",
                self._successes_in_buffer / self._n_env_steps_total,
            )
            if do_train_eval:
                logger.record_tabular(
                    "metrics/success_rate_train_blocked",
                    np.mean(train_results.success_rate[train_results.blocked_indices]),
                )
                logger.record_tabular(
                    "metrics/success_rate_train_unblocked",
                    np.mean(
                        train_results.success_rate[train_results.unblocked_indices]
                    ),
                )
            logger.record_tabular(
                "metrics/success_rate_eval_blocked",
                np.mean(eval_results.success_rate[eval_results.blocked_indices]),
            )
            logger.record_tabular(
                "metrics/success_rate_eval_unblocked",
                np.mean(eval_results.success_rate[eval_results.unblocked_indices]),
            )
            if self.eval_stochastic:
                logger.record_tabular(
                    "metrics/success_rate_eval_sto",
                    np.mean(eval_sto_results.success_rate),
                )

        # for episode_idx in range(self.max_rollouts_per_task):
        #     if do_train_eval:
        #         logger.record_tabular(
        #             "metrics/return_train_blocked_episode_{}".format(episode_idx + 1),
        #             np.mean(train_blocked_results.returns_per_episode[:, episode_idx]),
        #         )
        #         logger.record_tabular(
        #             "metrics/unblocked_return_train_unblocked_episode_{}".format(episode_idx + 1),
        #             np.mean(train_unblocked_results.returns_per_episode[:, episode_idx]),
        #         )
        #     logger.record_tabular(
        #         "metrics/return_eval_blocked_episode_{}".format(episode_idx + 1),
        #         np.mean(eval_blocked_results.returns_per_episode[:, episode_idx]),
        #     )
        #     logger.record_tabular(
        #         "metrics/unblocked_return_eval_unblocked_episode_{}".format(episode_idx + 1),
        #         np.mean(eval_unblocked_results.returns_per_episode[:, episode_idx]),
        #     )
        #     if self.eval_stochastic:
        #         logger.record_tabular(
        #             "metrics/return_eval_episode_{}_sto".format(episode_idx + 1),
        #             np.mean(eval_sto_results[:, episode_idx]),
        #         )

        if do_train_eval:
            logger.record_tabular(
                "metrics/total_steps_train_blocked",
                np.mean(train_results.total_steps[train_results.blocked_indices]),
            )
            logger.record_tabular(
                "metrics/total_steps_train_unblocked",
                np.mean(train_results.total_steps[train_results.unblocked_indices]),
            )

            logger.record_tabular(
                "metrics/return_train_total_blocked",
                np.mean(
                    np.sum(
                        train_results.returns_per_episode[
                            train_results.blocked_indices, :
                        ],
                        axis=-1,
                    )
                ),
            )
            logger.record_tabular(
                "metrics/return_train_total_unblocked",
                np.mean(
                    np.sum(
                        train_results.returns_per_episode[
                            train_results.unblocked_indices, :
                        ],
                        axis=-1,
                    )
                ),
            )
        logger.record_tabular(
            "metrics/total_steps_eval_blocked",
            np.mean(eval_results.total_steps[eval_results.blocked_indices]),
        )
        logger.record_tabular(
            "metrics/total_steps_eval_unblocked",
            np.mean(eval_results.total_steps[eval_results.unblocked_indices]),
        )
        logger.record_tabular(
            "metrics/return_eval_total_blocked",
            np.mean(
                np.sum(
                    eval_results.returns_per_episode[eval_results.blocked_indices, :],
                    axis=-1,
                )
            ),
        )
        logger.record_tabular(
            "metrics/return_eval_total_unblocked",
            np.mean(
                np.sum(
                    eval_results.returns_per_episode[eval_results.unblocked_indices, :],
                    axis=-1,
                )
            ),
        )
        logger.record_tabular(
            "metrics/left_steps_blocked",
            np.mean(eval_results.left_steps[eval_results.blocked_indices]),
        )
        logger.record_tabular(
            "metrics/left_steps_unblocked",
            np.mean(eval_results.left_steps[eval_results.unblocked_indices]),
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


def get_tasks_by_type(env: Env, key: str = "task_type") -> list[list[int]]:
    tasks: list[MazeTask] = env.get_wrapper_attr("get_all_tasks")()
    tasks_by_class_dict: dict[int, list[int]] = defaultdict(lambda: [])
    for index, task in enumerate(tasks):
        tasks_by_class_dict[getattr(task, key)].append(index)

    return list(tasks_by_class_dict.values())
