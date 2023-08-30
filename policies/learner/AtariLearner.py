from typing import Optional

from .Learner import Learner


class AtariLearner(Learner):
    def init_env(
            self,
            env_name: str,
            num_eval_tasks: Optional[int] = None,
            **kwargs
    ):
        from envs.atari import create_env

        assert num_eval_tasks > 0
        self.train_env = create_env(env_name)
        self.train_env.seed(self.seed)
        self.train_env.action_space.np_random.seed(self.seed)  # crucial

        self.eval_env = self.train_env
        self.eval_env.seed(self.seed + 1)

        self.train_tasks = []
        self.eval_tasks = num_eval_tasks * [None]

        self.max_rollouts_per_task = 1
        self.max_trajectory_len = self.train_env._max_episode_steps
