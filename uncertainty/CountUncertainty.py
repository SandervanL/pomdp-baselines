import numpy as np
import torch
from torch import Tensor

from uncertainty import Uncertainty
import torchkit.pytorch_utils as ptu


class CountUncertainty(Uncertainty):
    """Defines an uncertainty estimate based on counts over the state/observation space.
    Uncertainty will be scaled by 'scale'. Define boundaries either by 'state_bounds'
    or automatically by passing the environment 'env'. The counts will use
    'resolution'^m different bins for m-dimensional state vectors"""

    def __init__(self, scale: int = 1, resolution: float = 50, **kwargs):
        super().__init__(scale, **kwargs)
        self.resolution = resolution
        self.count: Tensor = ptu.zeros((2**12), dtype=torch.int64)
        self.scale = scale
        self.eps = 1e-7

    def state_bin(self, state: Tensor) -> Tensor:
        """Find the correct bin in 'self.count' for one state."""
        while state.dim() < 3:
            state = state.unsqueeze(dim=0)
        state = state.int() + 1  # Account that self.count[0] must be zero always
        return (state[:, :, 0] << 6 | state[:, :, 1]).unsqueeze(dim=-1)
        # return tuple(
        #     [
        #         int((x - l) / (h - l + self.eps) * self.resolution)
        #         for x, (l, h) in zip(state, self.bounds)
        #     ]
        # )

    def observe(self, state: Tensor) -> None:
        """
        Add counts for observed 'state'.
        Args:
            state: The state(s) to compute the error for.

        Returns:
            The error between the prediction and target network.
        """
        bins = self.state_bin(state)
        self.count[bins] += 1

    def __call__(self, state: Tensor, mask: Tensor) -> Tensor:
        """Returns the estimated uncertainty for observing a (minibatch of) state(s) ans Tensor.
        'state' can be either a Tuple, List, 1d Tensor or 2d Tensor (1d Tensors stacked in dim=0).
        Does not change the counters."""
        bins = self.state_bin(state)
        counts = self.count[bins] * mask
        return self.scale / torch.sqrt(counts + self.eps)
