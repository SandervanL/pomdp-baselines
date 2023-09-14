import abc
from typing import Optional, Callable

import torch
from torch import Tensor
from torch.nn import Module, functional as F

from policies.rl.base import RLAlgorithmBase
from utils import helpers as utl
from torchkit.constant import *
import torchkit.pytorch_utils as ptu


class BaseRnn(Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        encoder,
        algo: RLAlgorithmBase,
        action_embedding_size: int,
        observ_embedding_size: int,
        reward_embedding_size: int,
        task_dim: Optional[int],
        rnn_hidden_size: int,
        rnn_num_layers: int,
        image_encoder=None,
        embedding_rnn_init: int = 0,
        embedding_obs_init: int = 0,
        embedding_grad: str = "no-grad",
        **kwargs
    ):
        """

        Args:
            obs_dim:
            action_dim:
            encoder:
            algo:
            action_embedding_size:
            observ_embedding_size:
            reward_embedding_size:
            task_dim:
            rnn_hidden_size:
            rnn_num_layers:
            image_encoder:
            embedding_rnn_init: 0 for zero initialization,
                           1 for hidden state initialization,
                           2 for cell_state initialization,
                           3 for both
            embedding_grad: "no" returns an empty tensor,
                            "directly" returns the same tensor,
                            "grad" returns a linear layer with relu that can be trained
                            "no-grad" returns a linear layer without relu that cannot be trained.
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

        self.action_embedder = utl.FeatureExtractor(
            action_dim, action_embedding_size, F.relu
        )
        self.reward_embedder = utl.FeatureExtractor(1, reward_embedding_size, F.relu)

        assert (
            embedding_rnn_init == 0
            or embedding_grad != "direct"
            or (
                embedding_rnn_init > 0
                and embedding_grad == "direct"
                and rnn_hidden_size == task_dim
            )
        )

        self.task_hidden_embedder, _ = get_feature_extractor(
            embedding_rnn_init & 1 > 0, embedding_grad, task_dim, rnn_hidden_size
        )
        self.task_cell_embedder, _ = get_feature_extractor(
            embedding_rnn_init & 2 > 0, embedding_grad, task_dim, rnn_hidden_size
        )

        self.task_obs_embedder, task_obs_embedding_size = get_feature_extractor(
            embedding_obs_init & 1 > 0, embedding_grad, task_dim, observ_embedding_size
        )
        (
            self.task_proxy_embedder,
            self.task_proxy_embedding_size,
        ) = get_feature_extractor(
            embedding_obs_init & 2 > 0, embedding_grad, task_dim, rnn_hidden_size
        )

        # 2. Build RNN model
        self.rnn_input_size = (
            action_embedding_size
            + observ_embedding_size
            + reward_embedding_size
            + task_obs_embedding_size
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
        self,
        prev_actions: Tensor,
        rewards: Tensor,
        observations: Tensor,
        initial_internal_state: tuple[Tensor, Tensor],
        tasks: Optional[Tensor] = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        assert initial_internal_state is not None
        # all the input have the shape of (1 or T+1, B, *)
        # get embedding of initial transition
        input_a = self.action_embedder(prev_actions)
        input_r = self.reward_embedder(rewards)
        input_s = self._get_obs_embedding(observations)
        input_t = (
            self.task_obs_embedder(tasks)
            if tasks is not None
            else ptu.empty((rewards.shape[0], rewards.shape[1], 0))
        )
        inputs = torch.cat((input_a, input_r, input_s, input_t), dim=-1)

        # feed into RNN: output (T+1, B, hidden_size)
        output, current_internal_state = self.rnn(inputs, initial_internal_state)
        return output, current_internal_state

    def get_initial_info(
        self, batch_size: int = 1, task: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor, tuple[Tensor]]:
        """
        Initialize the initial internal state of the RNN.
        :param batch_size: the batch size of the upcoming operation.
        :param task: (1, B, internal_state_dim)
        """
        # here we assume batch_size = 1
        if task is not None and task.dim() == 3 and task.shape[0] != 1:
            task = task[0, :, :].unsqueeze(dim=0)

        # here we set the ndim = 2 for action and reward for compatibility
        prev_action = ptu.zeros((1, self.action_dim), requires_grad=False).float()
        reward = ptu.zeros((1, 1), requires_grad=False).float()

        hidden_state = ptu.zeros(
            (self.num_layers, batch_size, self.rnn_hidden_size), requires_grad=False
        ).float()
        if self.encoder == GRU_name:
            internal_state = hidden_state
        else:
            cell_state = ptu.zeros(
                (self.num_layers, batch_size, self.rnn_hidden_size), requires_grad=False
            ).float()
            hidden_embedding = self.task_hidden_embedder(task).repeat(
                self.num_layers, 1, 1
            )
            if hidden_embedding.shape[2] > 0:
                hidden_state = hidden_embedding

            cell_embedding = self.task_cell_embedder(task).repeat(self.num_layers, 1, 1)
            if cell_embedding.shape[2] > 0:
                cell_state = cell_embedding
            internal_state = (hidden_state, cell_state)

        return prev_action, reward, internal_state


def get_feature_extractor(
    enable: bool, grad: str, task_embedding_size: int, hidden_size: int
) -> tuple[Callable[[Tensor], Tensor], int]:
    """
    Get a feature extractor to transform embeddings.
    If grad, a ReLU is added. With no grad, no activation function is used.
    Args:
        enable: if false, an empty tensor is returned by the feature extractor.
        grad: "no" returns an empty tensor,
            "directly" returns the same tensor,
            "grad" returns a linear layer with relu that can be trained
            "no-grad" returns a linear layer without relu that cannot be trained.
        task_embedding_size: the embedding size of the task context.
        hidden_size: the target size of the output of the layer.

    Returns:
        A feature extractor, which is essentially a linear layer.
        Together with the output size of the feature extractor.
    """
    if not enable or grad == "no":
        return ptu.empty_tensor_like, 0
    if grad == "directly":
        return ptu.identity, task_embedding_size
    use_grad = grad == "grad"
    activation = F.relu if use_grad else ptu.identity
    return (
        utl.FeatureExtractor(
            task_embedding_size, hidden_size, activation
        ).requires_grad_(use_grad),
        hidden_size,
    )
