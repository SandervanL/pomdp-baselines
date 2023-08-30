from typing import Optional, Tuple, Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from policies.models.recurrent_base import Base_RNN
from policies.models.actor import MarkovPolicyBase
from utils import helpers as utl
from torchkit.constant import *
import torchkit.pytorch_utils as ptu


class ActorRnn(Base_RNN):
    def __init__(
            self,
            policy_layers: list[int],
            observ_embedding_size: int,
            **kwargs
    ):
        super().__init__(observ_embedding_size=observ_embedding_size, **kwargs)

        ## 3. build another obs branch
        if self.image_encoder is None:
            self.current_observ_embedder = utl.FeatureExtractor(
                self.obs_dim, observ_embedding_size, F.relu
            )

        ## 4. build policy
        self.policy: MarkovPolicyBase = self.algo.build_actor(
            input_size=self.rnn_hidden_size + observ_embedding_size,
            action_dim=self.action_dim,
            hidden_sizes=policy_layers,
        )

    def _get_shortcut_obs_embedding(self, observations: Tensor) -> Tensor:
        if self.image_encoder is None:  # vector obs
            return self.current_observ_embedder(observations)
        else:  # pixel obs
            return self.image_encoder(observations)

    def forward(self, prev_actions: Tensor, rewards: Tensor, observations: Tensor) -> tuple[
        Tensor, Tensor]:
        """
        For prev_actions a, rewards r, observations o: (T+1, B, dim)
                a[t] -> r[t], o[t]

        :returncurrent actions a' (T+1, B, dim) based on previous history

        """
        assert prev_actions.dim() == rewards.dim() == observations.dim() == 3
        assert prev_actions.shape[0] == rewards.shape[0] == observations.shape[0]

        ### 1. get hidden/belief states of the whole/sub trajectories, aligned with states
        # return the hidden states (T+1, B, dim)
        hidden_states, _ = self.get_hidden_states(
            prev_actions=prev_actions, rewards=rewards, observations=observations
        )

        # 2. another branch for current obs
        curr_embed = self._get_shortcut_obs_embedding(observations)  # (T+1, B, dim)

        # 3. joint embed
        joint_embeds = torch.cat((hidden_states, curr_embed), dim=-1)  # (T+1, B, dim)

        # 4. Actor
        return self.algo.forward_actor(actor=self.policy, observ=joint_embeds)

    # TODO perhaps enable again? I need grad for the task embedder @torch.no_grad()
    def get_initial_info(self, env_embedding: Optional[Tensor] = None, embedding_init: int = 0) -> \
            tuple[Tensor, Tensor, tuple[Tensor]]:
        """
        Initialize the initial internal state of the RNN.
        :param env_embedding: (1, B, internal_state_dim)
        :param embedding_init: 0 for zero initialization,
                               1 for hidden state initialization,
                               2 for cell_state initialization,
                               3 for both
        """
        assert (env_embedding is None and embedding_init == 0) or env_embedding is not None
        # here we assume batch_size = 1

        ## here we set the ndim = 2 for action and reward for compatibility
        prev_action = ptu.zeros((1, self.action_dim)).float()
        reward = ptu.zeros((1, 1)).float()

        hidden_state = ptu.zeros((self.num_layers, 1, self.rnn_hidden_size)).float()
        if self.encoder == GRU_name:
            internal_state = hidden_state
        else:
            cell_state = ptu.zeros((self.num_layers, 1, self.rnn_hidden_size)).float()
            if env_embedding is not None:
                if embedding_init | 1 > 0:
                    hidden_state = self.task_embedder(env_embedding)
                if embedding_init | 2 > 0:
                    cell_state = self.task_embedder(env_embedding)
            internal_state = (hidden_state, cell_state)

        return prev_action, reward, internal_state

    @torch.no_grad()
    def act(
            self,
            prev_internal_state: Tensor,
            prev_action: Tensor,
            reward: Tensor,
            obs: Tensor,
            deterministic: bool = False,
            return_log_prob: bool = False,
            valid_actions: Optional[np.ndarray] = None
    ) -> tuple[tuple[Tensor, Tensor, Tensor, Any], Optional[Tensor]]:
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
            valid_actions: an array for each action whether it is valid to take (1) or not (0).

        Returns:
            action_tuple: a tuple of the action, the probability of the action, and the log
                probability of the action.
            current_internal_state: the current internal state of the RNN.
        """""
        # for evaluation (not training), so no target actor, and T = 1
        # a function that generates action, works like a pytorch module

        # 1. get hidden state and current internal state
        ## NOTE: in T=1 step rollout (and RNN layers = 1), for GRU they are the same,
        # for LSTM, current_internal_state also includes cell state, i.e.
        # hidden state: (1, B, dim)
        # current_internal_state: (layers, B, dim) or ((layers, B, dim), (layers, B, dim))
        hidden_state, current_internal_state = self.get_hidden_states(
            prev_actions=prev_action,
            rewards=reward,
            observations=obs,
            initial_internal_state=prev_internal_state,
        )
        # 2. another branch for current obs
        curr_embed = self._get_shortcut_obs_embedding(obs)  # (1, B, dim)

        # 3. joint embed
        joint_embeds = torch.cat((hidden_state, curr_embed), dim=-1)  # (1, B, dim)
        if joint_embeds.dim() == 3:
            joint_embeds = joint_embeds.squeeze(0)  # (B, dim)

        # 4. Actor head, generate action tuple
        action_tuple = self.algo.select_action(
            actor=self.policy,
            observ=joint_embeds,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
            valid_actions=valid_actions
        )

        return action_tuple, current_internal_state
