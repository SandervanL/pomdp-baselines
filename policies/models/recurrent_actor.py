from typing import Optional, Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from policies.models.recurrent_base import BaseRnn
from policies.models.actor import MarkovPolicyBase
from utils import helpers as utl
import torchkit.pytorch_utils as ptu


class ActorRnn(BaseRnn):
    def __init__(self, policy_layers: list[int], observ_embedding_size: int, **kwargs):
        super().__init__(observ_embedding_size=observ_embedding_size, **kwargs)

        # 3. build another obs branch
        if self.image_encoder is None:
            self.current_observ_embedder = utl.FeatureExtractor(
                self.obs_dim, observ_embedding_size, F.relu
            )

        # 4. build policy
        self.policy: MarkovPolicyBase = self.algo.build_actor(
            input_size=self.rnn_hidden_size
            + observ_embedding_size
            + self.task_proxy_embedding_size,
            action_dim=self.action_dim,
            hidden_sizes=policy_layers,
        )

    def _get_shortcut_obs_embedding(self, observations: Tensor) -> Tensor:
        if self.image_encoder is None:  # vector obs
            return self.current_observ_embedder(observations)
        else:  # pixel obs
            return self.image_encoder(observations)

    def get_state_proxy(
        self,
        prev_internal_state: tuple[Tensor, Tensor],
        prev_action: Tensor,
        reward: Tensor,
        obs: Tensor,
        task: Optional[Tensor] = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Calculate the state proxy for the current history.
        This consists of the hidden state of the RNN, the embedding of the current observation,
        and the embedding of the current task.
        Args:
            prev_internal_state: the previous internal state of the RNN.
            prev_action: the action that was taken one step ago.
            reward: the reward that was received in the previous step.
            obs: The newest observation.
            task: the embedding of the current task.

        Returns:
            joint_embeds: the state proxy for the current history.
            current_internal_state: the current internal state of the RNN.
        """
        # for evaluation (not training), so no target actor, and T = 1
        # a function that generates action, works like a pytorch module
        # 1. get hidden state and current internal state
        # NOTE: in T=1 step rollout (and RNN layers = 1), for GRU they are the same,
        # for LSTM, current_internal_state also includes cell state, i.e.
        # hidden state: (1, B, dim)
        # current_internal_state: (layers, B, dim) or ((layers, B, dim), (layers, B, dim))
        hidden_state, current_internal_state = self.get_hidden_states(
            prev_actions=prev_action,
            rewards=reward,
            observations=obs,
            tasks=task,
            initial_internal_state=prev_internal_state,
        )
        # 2. another branch for current obs
        curr_embed = self._get_shortcut_obs_embedding(obs)  # (1, B, dim)

        # 3. another branch for current task
        task_embedding = (
            ptu.empty_tensor_like(reward)
            if task is None
            else self.task_proxy_embedder(task)
        )

        # 4. joint embed
        joint_embeds = torch.cat(
            (hidden_state, curr_embed, task_embedding), dim=-1
        )  # (1, B, dim)

        return joint_embeds, current_internal_state

    def forward(
        self,
        prev_actions: Tensor,
        rewards: Tensor,
        observations: Tensor,
        tasks: Optional[Tensor] = None,
        initial_state: Optional[tuple[Tensor, Tensor]] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        For prev_actions a, rewards r, observations o: (T+1, B, dim)
                a[t] -> r[t], o[t]

        :returncurrent actions a' (T+1, B, dim) based on previous history

        """
        assert prev_actions.dim() == rewards.dim() == observations.dim() == 3
        assert prev_actions.shape[0] == rewards.shape[0] == observations.shape[0]

        joint_embeds, _ = self.get_state_proxy(
            prev_internal_state=initial_state,
            prev_action=prev_actions,
            reward=rewards,
            obs=observations,
            task=tasks,
        )

        # 4. Actor
        return self.algo.forward_actor(actor=self.policy, observ=joint_embeds)

    @torch.no_grad()
    def act(
        self,
        prev_internal_state: tuple[Tensor, Tensor],
        prev_action: Tensor,
        reward: Tensor,
        obs: Tensor,
        deterministic: bool = False,
        return_log_prob: bool = False,
        task: Optional[Tensor] = None,
        valid_actions: Optional[np.ndarray] = None,
    ) -> tuple[tuple[Tensor, Tensor, Tensor, Any], tuple[Tensor, Tensor]]:
        """
        Select an action based on the previous action, internal state, and reward, and 
        current observation.
        Args:
            prev_internal_state: the previous internal state of the RNN.
            prev_action: the action that was taken one step ago.
            reward: the reward that was received in the previous step.
            obs: The newest observation.
            deterministic: whether to choose a discrete action or sample a continuous action.
            return_log_prob: whether to return the log probability of the action.
            task: the embedding of the current task.
            valid_actions: an array for each action whether it is valid to take (1) or not (0).

        Returns:
            action_tuple: a tuple of the action, the probability of the action, and the log
                probability of the action.
            current_internal_state: the current internal state of the RNN.
        """ ""
        joint_embeds, current_internal_state = self.get_state_proxy(
            prev_internal_state=prev_internal_state,
            prev_action=prev_action,
            reward=reward,
            obs=obs,
            task=task,
        )
        if joint_embeds.dim() == 3:
            joint_embeds = joint_embeds.squeeze(0)  # (B, dim)

        # 4. Actor head, generate action tuple
        action_tuple = self.algo.select_action(
            actor=self.policy,
            observ=joint_embeds,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
            valid_actions=valid_actions,
        )

        return action_tuple, current_internal_state
