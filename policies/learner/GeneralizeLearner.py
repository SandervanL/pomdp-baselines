import sys
from typing import Optional

import numpy as np

from .Learner import Learner, EvaluationResults
from utils import logger


class GeneralizeLearner(Learner):
    def init_env(
            self,
            env_name: str,
            eval_envs: Optional[dict[str, int]] = None,
            **kwargs
    ):
        sys.path.append("envs/rl-generalization")
        import sunblaze_envs

        self.train_env = sunblaze_envs.make(env_name, **kwargs)  # oracle in kwargs
        self.train_env.seed(self.seed)
        assert np.all(self.train_env.action_space.low == -1)
        assert np.all(self.train_env.action_space.high == 1)

        def check_env_class(env_name):
            if "Normal" in env_name:
                return "R"
            if "Extreme" in env_name:
                return "E"
            return "D"

        self.train_env_name = check_env_class(env_name)

        self.eval_envs = {}
        for env_name, num_eval_task in eval_envs.items():
            eval_env = sunblaze_envs.make(env_name, **kwargs)  # oracle in kwargs
            eval_env.seed(self.seed + 1)
            self.eval_envs[eval_env] = (
                check_env_class(env_name),
                num_eval_task,
            )  # several types of evaluation envs

        logger.log(self.train_env_name, self.train_env)
        logger.log(self.eval_envs)

        self.train_tasks = []
        self.max_rollouts_per_task = 1
        self.max_trajectory_len = self.train_env.spec.max_episode_steps

    def log_evaluate(self) -> EvaluationResults:
        eval_results = EvaluationResults()
        eval_results.success_rate = {}
        eval_results.total_steps = {}
        eval_results.returns_per_episode = {}
        eval_results.observations = {}
        for env, (env_name, eval_num_episodes_per_task) in self.eval_envs.items():
            self.eval_env = env  # assign eval_env, not train_env
            for suffix, deterministic in zip(["", "_sto"], [True, False]):
                if (not deterministic) and (not self.eval_stochastic):
                    continue
                return_eval, success_eval, _, total_step_eval = self.evaluate(
                    eval_num_episodes_per_task * [None],
                    deterministic=deterministic,
                )
                key = self.train_env_name + env_name + suffix
                eval_results.returns_per_episode[key] = return_eval.squeeze(-1)
                eval_results.success_rate[key] = success_eval
                eval_results.total_steps[key] = total_step_eval

        for k, v in eval_results.returns_per_episode.items():
            logger.record_tabular(f"metrics/return_eval_{k}", np.mean(v))
        for k, v in eval_results.success_rate.items():
            logger.record_tabular(f"metrics/succ_eval_{k}", np.mean(v))
        for k, v in eval_results.total_steps.items():
            logger.record_tabular(f"metrics/total_steps_eval_{k}", np.mean(v))

        return eval_results
