from typing import Optional

import numpy as np
import torch
from gymnasium import Env
from torch import Tensor

from uncertainty import Uncertainty


class CountUncertainty(Uncertainty):
    """Defines an uncertainty estimate based on counts over the state/observation space.
    Uncertainty will be scaled by 'scale'. Define boundaries either by 'state_bounds'
    or automatically by passing the environment 'env'. The counts will use
    'resolution'^m different bins for m-dimensional state vectors"""

    def __init__(
        self,
        scale: int = 1,
        env: Optional[Env] = None,
        state_bounds: Optional[tuple[int, int]] = None,
        resolution: float = 50,
    ):
        super().__init__(scale)
        if state_bounds is None:
            self.bounds = [
                (l, h)
                for l, h in zip(env.observation_space.low, env.observation_space.high)
            ]
        else:
            self.bounds = state_bounds
        self.resolution = resolution
        self.count = torch.zeros(*([resolution for _ in self.bounds]), dtype=torch.long)
        self.scale = scale
        self.eps = 1e-7

    def state_bin(self, state: Tensor):
        """Find the correct bin in 'self.count' for one state."""
        return torch.where(state[0] == 2)
        # return tuple(
        #     [
        #         int((x - l) / (h - l + self.eps) * self.resolution)
        #         for x, (l, h) in zip(state, self.bounds)
        #     ]
        # )

    def observe(self, state: Tensor, mask: Tensor):
        """
        Add counts for observed 'state's.
        Args:
            state: The state(s) to compute the error for.
            mask: The done flag(s) for the state(s).

        Returns:
            The error between the prediction and target network.
        """
        if len(state.shape) == 1:
            state.unsqueeze_(dim=0)
        if len(mask.shape) == 1:
            mask.unsqueeze_(dim=0)

        for s in state * mask:
            if s.sum() > 0:
                b = self.state_bin(s)
                self.count[b] += 1

    def __call__(self, state: Tensor, mask: Tensor):
        """Returns the estimated uncertainty for observing a (minibatch of) state(s) ans Tensor.
        'state' can be either a Tuple, List, 1d Tensor or 2d Tensor (1d Tensors stacked in dim=0).
        Does not change the counters."""
        if len(state.shape) == 1:
            state.unsqueeze_(dim=0)
        n = torch.zeros(len(state))
        for i, s in enumerate(state * mask):
            if s.sum() > 0:
                b = self.state_bin(s)
                n[i] = self.count[b]
        return self.scale / np.sqrt(n + self.eps)
