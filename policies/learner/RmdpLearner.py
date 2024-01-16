import sys
from typing import Optional

import numpy as np

from .Learner import Learner, EvaluationResults
from utils import logger


class RmdpLearner(Learner):

    def init_env(
            self,
            env_name: str,
            num_eval_tasks: Optional[int] = None,
            worst_percentile: Optional[int] = None,
            **kwargs
    ):
        sys.path.append("envs/rl-generalization")
        import sunblaze_envs

        assert num_eval_tasks > 0 and 0.0 < worst_percentile < 1.0
        self.train_env = sunblaze_envs.make(env_name, **kwargs)  # oracle
        self.train_env.seed(self.seed)
        assert np.all(self.train_env.action_space.low == -1)
        assert np.all(self.train_env.action_space.high == 1)

        self.eval_env = self.train_env
        self.eval_env.seed(self.seed + 1)

        self.worst_percentile = worst_percentile

        self.train_tasks = []
        self.eval_tasks = num_eval_tasks * [None]

        self.max_rollouts_per_task = 1
        self.max_trajectory_len = self.train_env.spec.max_episode_steps

    def log_evaluate(self) -> EvaluationResults:
        eval_results = self.evaluate(self.eval_tasks)
        eval_results.returns_per_episode = eval_results.returns_per_episode.squeeze(-1)
        # np.quantile is introduced in np v1.15, so we have to use np.percentile
        cutoff = np.percentile(eval_results.returns_per_episode, 100 * self.worst_percentile)
        worst_indices = np.where(
            eval_results.returns_per_episode <= cutoff
        )  # must be "<=" to avoid empty set
        returns_eval_worst, total_steps_eval_worst = (
            eval_results.returns_per_episode[worst_indices],
            eval_results.total_steps[worst_indices],
        )

        logger.record_tabular("metrics/return_eval_avg", eval_results.returns_per_episode.mean())
        logger.record_tabular(
            "metrics/return_eval_worst", returns_eval_worst.mean()
        )
        logger.record_tabular(
            "metrics/total_steps_eval_avg", eval_results.total_steps.mean()
        )
        logger.record_tabular(
            "metrics/total_steps_eval_worst", total_steps_eval_worst.mean()
        )

        return eval_results
