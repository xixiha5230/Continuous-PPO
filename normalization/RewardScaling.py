import numpy as np

from normalization.RunningMeanStd import RunningMeanStd
from utils.ConfigHelper import ConfigHelper


class RewardScaling:
    """reward scaling"""

    def __init__(self, shape, gamma: float, config: ConfigHelper):
        """
        Args:
            shape {} -- shape of reward
            gamma {float} -- Name of the to be instantiated environment
        """
        self.shape = shape
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape, dtype=np.float64)
        self.config = config

    def __call__(self, x):
        """scaling reward
        Args:
            x {} -- reward
        """
        if not self.config.use_reward_scaling:
            return x
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        # Only divided std
        x = x / (self.running_ms.std + 1e-8)
        return x

    def reset(self):
        """When an episode is done,we should reset self.R"""
        self.R = np.zeros(self.shape, dtype=np.float64)
