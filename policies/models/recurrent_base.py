import abc
from typing import Optional

import torch
from torch import Module, Tensor

from policies.rl.base import RLAlgorithmBase
from utils import helpers as utl
from torchkit.constant import *


class Base_RNN(Module, metaclass=abc.ABCMeta):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 encoder,
                 algo: RLAlgorithmBase,
                 action_embedding_size: int,
                 observ_embedding_size: int,
                 reward_embedding_size: int,
                 task_embedding_size: Optional[int],
                 rnn_hidden_size: int,
                 rnn_num_layers: int,
                 image_encoder=None):
        super().__init__()

        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.algo: RLAlgorithmBase = algo

        # Build Model
        # 1. embed action, state, reward (Feed-forward layers first)

        self.image_encoder = image_encoder
        if self.image_encoder is None:
            self.observ_embedder = utl.FeatureExtractor(
                obs_dim, observ_embedding_size, F.relu
            )
        else:  # for pixel observation, use external encoder
            assert observ_embedding_size == 0
            observ_embedding_size = self.image_encoder.embed_size  # reset it

        self.action_embedder = utl.FeatureExtractor(action_dim, action_embedding_size, F.relu)
        self.reward_embedder = utl.FeatureExtractor(1, reward_embedding_size, F.relu)
        self.task_embedder = None if task_embedding_size is None else utl.FeatureExtractor(
            task_embedding_size, rnn_hidden_size, F.relu)

        # 2. Build RNN model
        self.rnn_input_size = (
                action_embedding_size + observ_embedding_size + reward_embedding_size
        )
        self.rnn_hidden_size = rnn_hidden_size

        assert encoder in RNNs
        self.encoder = encoder
        self.num_layers = rnn_num_layers

        self.rnn = RNNs[encoder](
            input_size=self.rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.num_layers,
            batch_first=False,
            bias=True,
        )

        # never add activation after GRU cell, because the last operation of GRU is tanh
        # default gru initialization is uniform, not recommended
        # https://smerity.com/articles/2016/orthogonal_init.html orthogonal has eigenvalue = 1
        # to prevent grad explosion or vanishing
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

    def _get_obs_embedding(self, observations: Tensor) -> Tensor:
        if self.image_encoder is None:  # vector obs
            return self.observ_embedder(observations)
        else:  # pixel obs
            return self.image_encoder(observations)

    def get_hidden_states(
            self, prev_actions: Tensor, rewards: Tensor, observations: Tensor,
            initial_internal_state=None
    ) -> tuple[Tensor, Optional[Tensor]]:
        # all the input have the shape of (1 or T+1, B, *)
        # get embedding of initial transition
        input_a = self.action_embedder(prev_actions)
        input_r = self.reward_embedder(rewards)
        input_s = self._get_obs_embedding(observations)
        inputs = torch.cat((input_a, input_r, input_s), dim=-1)

        # feed into RNN: output (T+1, B, hidden_size)
        if initial_internal_state is None:  # initial_internal_state is zeros
            output, _ = self.rnn(inputs)
            return output, None
        else:  # useful for one-step rollout
            output, current_internal_state = self.rnn(inputs, initial_internal_state)
            return output, current_internal_state
