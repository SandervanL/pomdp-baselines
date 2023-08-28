from typing import Optional, Any

import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit as TimeLimitBase, time


class TimeLimit(TimeLimitBase):
    """Updated to support reset() with reset_params flag for Adaptive"""

    def reset(self, reset_params=True, *, seed: Optional[int] = None,
              options: Optional[dict[str, Any]] = None) -> \
            tuple[np.ndarray, dict]:
        self._episode_started_at = time.time()
        self._elapsed_steps = 0
        return self.env.reset(reset_params, seed=seed, options=options)
