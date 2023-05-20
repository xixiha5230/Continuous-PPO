import gym.spaces as gym_space
import gymnasium.spaces as gymnasium_space

from normalization.RunningMeanStd import RunningMeanStd


class StateNormalizer:
    def __init__(self, space):
        if isinstance(space, (gym_space.Tuple, gymnasium_space.Tuple, tuple)):
            self.running_ms = [RunningMeanStd(shape=s.shape) for s in space]
        else:
            self.running_ms = [RunningMeanStd(shape=space.shape)]

    def __call__(self, data, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        res = []
        if isinstance(self.running_ms, list):
            for r, d in zip(self.running_ms, data):
                if update:
                    r.update(d)
                res.append((d - r.mean) / (r.std + 1e-8))
        else:
            raise NotImplementedError
        return res
