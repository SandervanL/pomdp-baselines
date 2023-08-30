from typing import Any, Tuple

from gymnasium.core import ObsType
from torch import nn, Tensor

from policies.models.actor import MarkovPolicyBase


class RLAlgorithmBase:
    name = "rl"
    continuous_action = True
    use_target_actor = True

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def build_actor(input_size: int, action_dim: int, hidden_sizes: list[int]) -> MarkovPolicyBase:
        raise NotImplementedError

    @staticmethod
    def build_critic(input_size: int, hidden_sizes: list[int], **kwargs) -> Tuple[Any, Any]:
        """
        return two critics
        """
        raise NotImplementedError

    def select_action(
            self, actor: nn.Module, observ: ObsType, deterministic: bool, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor, Any]:
        """
        actor: defined by build_actor
        observ: (B, dim), could be history embedding
        return (action, mean*, log_std*, log_prob*) * if exists
        """
        raise NotImplementedError

    @staticmethod
    def forward_actor(actor, observ) -> Tuple[Any, Any]:
        """
        actor: defined by build_actor
        observ: (B, dim), could be history embedding
        return (action, log_prob*)
        """
        raise NotImplementedError

    def critic_loss(
            self,
            markov_actor: bool,
            markov_critic: bool,
            actor,
            actor_target,
            critic,
            critic_target,
            observations,
            actions,
            rewards,
            dones,
            gamma,
            next_observations,
    ) -> Tuple[Tuple[Any, Any], Any]:
        """
        return (q1_pred, q2_pred), q_target
        """
        raise NotImplementedError

    def actor_loss(
            self,
            markov_actor: bool,
            markov_critic: bool,
            actor,
            actor_target,
            critic,
            critic_target,
            observations,
            actions,
            rewards,
    ) -> Tuple[Any, Any]:
        """
        return policy_loss, log_probs*
        """
        raise NotImplementedError

    def update_others(self, **kwargs):
        pass
