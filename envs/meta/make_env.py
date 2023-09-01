import gymnasium as gym

from envs.meta.wrappers import VariBadWrapper


# In VariBAD, they use on-policy PPO by vectorized env.
# In BOReL, they use off-policy SAC by single env.


def make_env(env_id: str, episodes_per_task: int, oracle: bool = False, **kwargs):
    """
    kwargs: include n_tasks=num_tasks
    """
    return VariBadWrapper(
        env=gym.make(env_id, **kwargs),
        episodes_per_task=episodes_per_task,
        oracle=oracle,
    )
