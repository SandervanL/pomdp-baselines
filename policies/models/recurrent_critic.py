from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from policies.models.recurrent_base import BaseRnn
from utils import helpers as utl


class CriticRnn(BaseRnn):
    def __init__(self, dqn_layers: list[int], **kwargs):
        super().__init__(**kwargs)

        # 3. build another obs+act branch
        shortcut_embedding_size = self.rnn_input_size
        if self.algo.continuous_action and self.image_encoder is None:
            # for vector-based continuous action problems
            self.current_shortcut_embedder = utl.FeatureExtractor(
                self.obs_dim + self.action_dim, shortcut_embedding_size, F.relu
            )
        elif self.algo.continuous_action and self.image_encoder is not None:
            # for image-based continuous action problems
            self.current_shortcut_embedder = utl.FeatureExtractor(
                self.action_dim, shortcut_embedding_size, F.relu
            )
            shortcut_embedding_size += self.image_encoder.embed_size
        elif not self.algo.continuous_action and self.image_encoder is None:
            # for vector-based discrete action problems
            self.current_shortcut_embedder = utl.FeatureExtractor(
                self.obs_dim, shortcut_embedding_size, F.relu
            )
        elif not self.algo.continuous_action and self.image_encoder is not None:
            # for image-based discrete action problems
            shortcut_embedding_size = self.image_encoder.embed_size
        else:
            raise NotImplementedError

        # 4. build q networks
        self.qf1, self.qf2 = self.algo.build_critic(
            input_size=self.rnn_hidden_size + shortcut_embedding_size,
            hidden_sizes=dqn_layers,
            action_dim=self.action_dim,
        )

    def _get_shortcut_obs_act_embedding(self, observations, current_actions):
        if self.algo.continuous_action and self.image_encoder is None:
            # for vector-based continuous action problems
            return self.current_shortcut_embedder(
                torch.cat([observations, current_actions], dim=-1)
            )
        elif self.algo.continuous_action and self.image_encoder is not None:
            # for image-based continuous action problems
            return torch.cat(
                [
                    self.image_encoder(observations),
                    self.current_shortcut_embedder(current_actions),
                ],
                dim=-1,
            )
        elif not self.algo.continuous_action and self.image_encoder is None:
            # for vector-based discrete action problems (not using actions)
            return self.current_shortcut_embedder(observations)
        elif not self.algo.continuous_action and self.image_encoder is not None:
            # for image-based discrete action problems (not using actions)
            return self.image_encoder(observations)

    def forward(
        self,
        prev_actions: Tensor,
        rewards: Tensor,
        observations: Tensor,
        current_actions: Tensor,
        tasks: Tensor,
        initial_state: Optional[tuple[Tensor, Tensor]] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        For prev_actions a, rewards r, observations o: (T+1, B, dim)
                a[t] -> r[t], o[t]
        current_actions (or action probs for discrete actions) a': (T or T+1, B, dim)
                o[t] -> a'[t]
        NOTE: there is one timestep misalignment in prev_actions and current_actions
        """
        assert (
            prev_actions.dim()
            == rewards.dim()
            == observations.dim()
            == current_actions.dim()
            == tasks.dim()
            == 3
        )
        assert prev_actions.shape[0] == rewards.shape[0] == observations.shape[0]

        # 1. get hidden/belief states of the whole/sub trajectories, aligned with observations
        # return the hidden states (T+1, B, dim)
        hidden_states, _ = self.get_hidden_states(
            prev_actions=prev_actions,
            rewards=rewards,
            observations=observations,
            initial_internal_state=initial_state,
            tasks=tasks,
        )

        # 2. another branch for state & **current** action
        # if same_shape, current_actions does NOT include last obs's action
        # else, current_actions include last obs's action, i.e. we have a'[T] in reaction to o[T]
        same_shape = current_actions.shape[0] == observations.shape[0]
        observations = observations if same_shape else observations[:-1]
        hidden_states = hidden_states if same_shape else hidden_states[:-1]

        curr_embed = self._get_shortcut_obs_act_embedding(
            observations, current_actions
        )  # (T(+1 if not same_shape), B, dim)
        # 3. joint embeds
        joint_embeds = torch.cat(
            (hidden_states, curr_embed), dim=-1
        )  # (T(+1 if not same_shape), B, dim)

        # 4. q value
        q1 = self.qf1(joint_embeds)
        q2 = self.qf2(joint_embeds)

        return q1, q2  # (T or T+1, B, 1 or A)
