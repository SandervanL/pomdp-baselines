from torch import Tensor
from torchkit import pytorch_utils as ptu


class Uncertainty:
    """Defines an uncertainty estimate based on counts over the state/observation space.
    Uncertainty will be scaled by 'scale'. Define boundaries either by 'state_bounds'
    or automatically by passing the environment 'env'. The counts will use
    'resolution'^m different bins for m-dimensional state vectors"""

    def __init__(self, scale: float = 1, **kwargs):
        self.scale = scale

    def observe(self, state: Tensor) -> None:
        """Add counts for observed state.
        'state' can be either a Tuple, List, 1d Tensor or 2d Tensor (1d Tensors stacked in dim=0).
        """

    def __call__(self, state: Tensor):
        """Returns the estimated uncertainty for observing a (minibatch of) state(s) ans Tensor.
        'state' can be either a Tuple, List, 1d Tensor or 2d Tensor (1d Tensors stacked in dim=0).
        Does not change the counters."""
        while state.dim() < 3:
            state = state.unsqueeze(0)
        return ptu.zeros((state.shape[0], state.shape[1], 1))
