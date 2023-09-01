from typing import Any, Tuple, Optional

from gymnasium.core import ObsType
from torch import nn, Tensor
from torch.nn import Module

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
            actor: Module,
            actor_target: Module,
            critic: Module,
            critic_target: Module,
            observations: Tensor,
            actions: Tensor,
            rewards: Tensor,
            dones: Tensor,
            gamma: float,
            next_observations: Optional[Tensor] = None,  # used in markov_critic
            task_embeddings: Optional[Tensor] = None,
    ) -> Tuple[Tuple[Any, Any], Any]:
        """
        return (q1_pred, q2_pred), q_target
        """
        raise NotImplementedError

    def actor_loss(
            self,
            markov_actor: bool,
            markov_critic: bool,
            actor: Module,
            actor_target: Module,
            critic: Module,
            critic_target: Module,
            observations: Tensor,
            actions: Tensor,
            rewards: Tensor,
            task_embeddings: Optional[Tensor] = None,
    ) -> Tuple[Any, Any]:
        """
        return policy_loss, log_probs*
        """
        raise NotImplementedError

    def update_others(self, **kwargs):
        pass
