from typing import Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box
from torch.distributions import Categorical, Normal

from utils.ConfigHelper import ConfigHelper


class Actor(nn.Module):
    """Actor base Module"""

    def __init__(self, config: ConfigHelper, action_space: Box) -> None:
        """
        Args:
            config {dict} -- config dictionary
            action_space {Box} -- action space
        """
        super(Actor, self).__init__()
        self.action_type = config.env_action_type

        if self.action_type == "continuous":
            self.action_dim, self.action_max = (
                action_space.shape[0],
                max(action_space.high),
            )
        elif self.action_type == "discrete":
            self.action_dim = action_space
        else:
            raise NotImplementedError(self.action_type)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x {torch.Tensor} --- hight level feature
        """
        raise NotImplementedError()


class GaussianActor(Actor):
    """Gaussian Actor Module"""

    def __init__(
        self, config: ConfigHelper, input_size: int, action_space: tuple
    ) -> None:
        """
        Args:
            config {dict} -- config dictionary
            input_size {int} -- input feature size
            action_space {tuple} -- action space
        """
        super(GaussianActor, self).__init__(config, action_space)

        if self.action_type == "continuous":
            self.mu = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ReLU(),
                nn.Linear(input_size, self.action_dim),
                nn.Tanh(),
            )
            nn.init.orthogonal_(self.mu[0].weight, np.sqrt(2))
            nn.init.orthogonal_(self.mu[2].weight, np.sqrt(0.01))
            self.sigma = nn.Parameter(torch.zeros(1, self.action_dim))
        elif self.action_type == "discrete":
            self.mu = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ReLU(),
                nn.Linear(input_size, self.action_dim),
            )
            nn.init.orthogonal_(self.mu[0].weight, np.sqrt(2))
            nn.init.orthogonal_(self.mu[2].weight, np.sqrt(0.01))
        else:
            raise NotImplementedError(self.action_type)

    def forward(self, feature: torch.Tensor):
        """
        Args:
            feature {torch.Tensor} --- hight level feature
        """
        if self.action_type == "continuous":
            action_mean = self.action_max * self.mu(feature)
            action_std = self.sigma.exp()
            dist = Normal(action_mean, action_std)
        elif self.action_type == "discrete":
            action_probs = self.mu(feature)
            dist = Categorical(probs=None, logits=action_probs)
        else:
            raise NotImplementedError(self.action_type)
        return dist


class SigmaParameterModule(nn.Module):
    def __init__(self, size):
        super(SigmaParameterModule, self).__init__()
        self.param = nn.Parameter(torch.zeros(1, size))

    def exp(self):
        return self.param.exp()


class MultiGaussianActor(Actor):
    """Gaussian Multiple Actor Module"""

    def __init__(
        self, config: ConfigHelper, input_size: int, action_space: tuple, task_num: int
    ) -> None:
        """
        Args:
            config {dict} -- config dictionary
            input_size {int} -- input feature size
            action_space {tuple} -- action space
            task_num {int} -- num of task
        """
        super(MultiGaussianActor, self).__init__(config, action_space)

        if self.action_type == "continuous":
            self.m_mu = nn.ModuleDict(
                [
                    [
                        str(i),
                        nn.Sequential(
                            nn.Linear(input_size, input_size),
                            nn.ReLU(),
                            nn.Linear(input_size, self.action_dim),
                            nn.Tanh(),
                        ),
                    ]
                    for i in range(task_num)
                ]
            )
            for _, m in self.m_mu.items():
                nn.init.orthogonal_(m[0].weight, np.sqrt(2))
                nn.init.orthogonal_(m[2].weight, np.sqrt(0.01))
            self.m_sigma = nn.ModuleDict(
                [
                    [str(i), SigmaParameterModule(self.action_dim)]
                    for i in range(task_num)
                ]
            )
        elif self.action_type == "discrete":
            self.m_mu = nn.ModuleDict(
                [
                    [
                        str(i),
                        nn.Sequential(
                            nn.Linear(input_size, input_size),
                            nn.ReLU(),
                            nn.Linear(input_size, self.action_dim),
                        ),
                    ]
                    for i in range(task_num)
                ]
            )
            for _, m in self.m_mu.items():
                nn.init.orthogonal_(m[0].weight, np.sqrt(2))
                nn.init.orthogonal_(m[2].weight, np.sqrt(0.01))
        else:
            raise NotImplementedError(self.action_type)

    def forward(self, feature: torch.Tensor, actor_index: Union[torch.Tensor, int]):
        """
        Args:
            feature {torch.Tensor} --- hight level feature
            actor_index {int} -- module {actor[actor_index]} will be select to forward
        """
        if self.action_type == "continuous":
            if isinstance(actor_index, torch.Tensor):
                max_value, _ = torch.max(actor_index, dim=-1)
                mask = torch.eq(actor_index, max_value.unsqueeze(-1))
                action_mean = torch.sum(
                    torch.stack(
                        [
                            self.action_max * m(feature) * mask[:, int(i)].unsqueeze(-1)
                            for i, m in self.m_mu.items()
                        ]
                    ),
                    dim=0,
                )
                action_std = torch.sum(
                    torch.stack(
                        [
                            m.exp() * mask[:, int(i)].unsqueeze(-1)
                            for i, m in self.m_sigma.items()
                        ]
                    ),
                    dim=0,
                )
            else:
                action_mean = self.action_max * self.m_mu[str(actor_index)](feature)
                action_std = self.m_sigma[str(actor_index)].exp()
            dist = Normal(action_mean, action_std)
        elif self.action_type == "discrete":
            action_probs = self.m_mu[str(actor_index)](feature)
            dist = Categorical(probs=None, logits=action_probs)
        else:
            raise NotImplementedError(self.action_type)
        return dist
