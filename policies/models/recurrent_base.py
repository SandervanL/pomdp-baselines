import abc
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, functional as F

from policies.rl.base import RLAlgorithmBase
from utils import helpers as utl
from torchkit.constant import *
import torchkit.pytorch_utils as ptu


class BaseRnn(Module, metaclass=abc.ABCMeta):
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
                 image_encoder=None,
                 embedding_init=None,
                 **kwargs):
        """

        Args:
            obs_dim:
            action_dim:
            encoder:
            algo:
            action_embedding_size:
            observ_embedding_size:
            reward_embedding_size:
            task_embedding_size:
            rnn_hidden_size:
            rnn_num_layers:
            image_encoder:
            embedding_init: 0 for zero initialization,
                           1 for hidden state initialization,
                           2 for cell_state initialization,
                           3 for both
            **kwargs:
        """
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

        self.task_hidden_embedder = None if embedding_init & 1 == 0 else (
            utl.FeatureExtractor(task_embedding_size, rnn_hidden_size, F.relu))
        self.task_cell_embedder = None if embedding_init & 2 == 0 else (
            utl.FeatureExtractor(task_embedding_size, rnn_hidden_size, F.relu))

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
            initial_internal_state: Optional[tuple[Tensor, Tensor]] = None
    ) -> tuple[Tensor, Optional[Tensor]]:
        # all the input have the shape of (1 or T+1, B, *)
        # get embedding of initial transition
        input_a = self.action_embedder(prev_actions)
        input_r = self.reward_embedder(rewards)
        input_s = self._get_obs_embedding(observations)
        inputs = torch.cat((input_a, input_r, input_s), dim=-1)

        # feed into RNN: output (T+1, B, hidden_size)
        if initial_internal_state is None:  # initial_internal_state is zeros
            # TODO we should not end up here
            output, _ = self.rnn(inputs)
            return output, None
        else:  # useful for one-step rollout
            output, current_internal_state = self.rnn(inputs, initial_internal_state)
            return output, current_internal_state

    def get_initial_info(self, batch_size: int = 1, task_embedding: Optional[Tensor] = None) -> \
            tuple[Tensor, Tensor, tuple[Tensor]]:
        """
        Initialize the initial internal state of the RNN.
        :param batch_size: the batch size of the upcoming operation.
        :param task_embedding: (1, B, internal_state_dim)
        """
        # here we assume batch_size = 1
        if (task_embedding is not None and task_embedding.dim() == 3
                and task_embedding.shape[0] != 1):
            task_embedding = task_embedding[0, :, :].unsqueeze(dim=0)

        # here we set the ndim = 2 for action and reward for compatibility
        prev_action = ptu.zeros((1, self.action_dim)).float()
        reward = ptu.zeros((1, 1)).float()

        hidden_state = ptu.zeros((self.num_layers, batch_size, self.rnn_hidden_size)).float()
        if self.encoder == GRU_name:
            internal_state = hidden_state
        else:
            cell_state = ptu.zeros((self.num_layers, batch_size, self.rnn_hidden_size)).float()
            hidden_state = hidden_state if self.task_hidden_embedder is None else (
                self.task_hidden_embedder(task_embedding).repeat(self.num_layers, 1, 1))
            cell_state = cell_state if self.task_cell_embedder is None else (
                self.task_cell_embedder(task_embedding).repeat(self.num_layers, 1, 1))
            internal_state = (hidden_state, cell_state)

        return prev_action, reward, internal_state
