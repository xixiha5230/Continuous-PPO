import numpy as np

from utils import ConfigHelper


class RNDRewardScaling:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # -> It's indeed batch normalization. :D
    def __init__(self, epsilon=1e-4, shape=(), config: ConfigHelper = None) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

        if config.multi_task:
            self.task_num = config.task_num
            self.num_workers = config.num_workers // self.task_num
        else:
            self.num_workers = config.num_workers
        self.worker_steps = config.worker_steps
        self.gamma = config.gamma

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        # mean
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        # var
        self.var = M2 / tot_count
        # count
        self.count = tot_count

    def normalize_rnd_rewards(self, rewards):
        # OpenAI's usage of Forward filter is definitely wrong;
        # Because: https://github.com/openai/random-network-distillation/issues/16#issuecomment-488387659
        intrinsic_returns = [[] for _ in range(self.num_workers)]
        for worker in range(self.num_workers):
            rewems = 0
            for step in reversed(range(self.worker_steps)):
                rewems = rewems * self.gamma + rewards[worker][step]
                intrinsic_returns[worker].insert(0, rewems)
        self.update(np.ravel(intrinsic_returns).reshape(-1, 1))
        return rewards / (self.var ** 0.5)
