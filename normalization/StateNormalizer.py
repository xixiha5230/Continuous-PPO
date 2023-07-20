import gym.spaces as gym_space
import gymnasium.spaces as gymnasium_space

from normalization.RunningMeanStd import RunningMeanStd
from utils.ConfigHelper import ConfigHelper


class StateNormalizer:
    def __init__(self, space, config: ConfigHelper):
        assert isinstance(space, (gym_space.Tuple, gymnasium_space.Tuple, tuple))
        self.config = config
        space = space[:-1] if config.multi_task else space
        self.running_ms = [RunningMeanStd(shape=s.shape) for s in space]

    def __call__(self, data, update=True):
        if not self.config.use_state_normailzation:
            return data
        _data = data[:-1] if self.config.multi_task else data

        # Whether to update the mean and std,during the evaluating,update=False
        res = []
        for r, d in zip(self.running_ms, _data):
            if update:
                r.update(d)
            res.append((d - r.mean) / (r.std + 1e-8))
        if self.config.multi_task:
            res.append(data[-1])

        return res
