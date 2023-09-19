import torch
from gymnasium import Env
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU

from uncertainty import Uncertainty
from torchkit import pytorch_utils as ptu


class RNDUncertainty(Uncertainty):
    """This class uses Random Network Distillation to estimate the novelty of states."""

    def __init__(
        self,
        input_size: int,
        scale: float,
        hidden_dim: int = 1024,
        embed_dim: int = 256,
    ):
        super().__init__(scale)
        self.criterion = torch.nn.MSELoss(reduction="none")
        self.target_net = Sequential(
            Linear(input_size, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, embed_dim),
        )
        self.predict_net = Sequential(
            Linear(input_size, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, embed_dim),
        )
        self.target_net.to(ptu.device)
        self.predict_net.to(ptu.device)
        self.optimizer = torch.optim.Adam(self.predict_net.parameters())

    def error(self, state: Tensor, mask: Tensor) -> Tensor:
        """
        Computes the error between the prediction and target network.
        Args:
            state: The state(s) to compute the error for.
            mask: The done flag(s) for the state(s).

        Returns:
            The error between the prediction and target network.
        """
        if len(state.shape) == 1:
            state.unsqueeze_(dim=0)

        prediction = self.predict_net(state)
        target = self.target_net(state)
        return self.criterion(prediction * done, target * done)

    def observe(self, state: Tensor, mask: Tensor) -> None:
        """
        Observes state(s) and 'remembers' them using Random Network Distillation.
        Args:
            state: The state(s) to observe.
            mask: The done flag(s) for the state(s).
        """
        self.optimizer.zero_grad()
        self.error(state, mask).mean().backward()
        self.optimizer.step()

    def __call__(self, state: Tensor, done: Tensor) -> Tensor:
        """
        Returns the estimated uncertainty for observing a (minibatch of) state(s) as Tensor.
        Args:
            state: The state(s) to compute the uncertainty for.

        Returns:
            The estimated uncertainty for observing a (minibatch of) state(s) as Tensor.
        """
        return self.scale * self.error(state, done).sum(dim=-1)
