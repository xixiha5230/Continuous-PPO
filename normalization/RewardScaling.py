import numpy as np
from normalization.Normalization import RunningMeanStd


class RewardScaling:
    def __init__(self, shape, gamma):
        # reward shape=1
        self.shape = shape
        # discount factor
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        # Only divided std
        x = x / (self.running_ms.std + 1e-8)
        return x

    # When an episode is done,we should reset 'self.R'
    def reset(self):
        self.R = np.zeros(self.shape)
