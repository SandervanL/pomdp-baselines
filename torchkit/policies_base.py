import abc
from typing import Optional, Any

import numpy as np


class Policy(object, metaclass=abc.ABCMeta):
    """
    General policy interface.
    """

    @abc.abstractmethod
    def get_action(self, observation):
        """
        :param observation:
        :return: action, debug_dictionary
        """
        pass

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> \
            tuple[np.ndarray, dict]:
        pass


class ExplorationPolicy(Policy, metaclass=abc.ABCMeta):
    def set_num_steps_total(self, t):
        pass


class SerializablePolicy(Policy, metaclass=abc.ABCMeta):
    """
    Policy that can be serialized.
    """

    def get_param_values(self):
        return None

    def set_param_values(self, values):
        pass

    """
    Parameters should be passed as np arrays in the two functions below.
    """

    def get_param_values_np(self):
        return None

    def set_param_values_np(self, values):
        pass
