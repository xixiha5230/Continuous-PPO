import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box
from torch.distributions import Categorical, Normal


class Actor(nn.Module):
    ''' Actor base Module '''

    def __init__(self, action_space: Box, config: dict) -> None:
        '''
        Args:
            action_space {Box} -- action space
            config {dict} -- config dictionary
        '''
        super(Actor, self).__init__()
        self.config = config
        self.conf_train = config['train']
        self.action_type = self.conf_train['action_type']

        if self.action_type == 'continuous':
            self.action_dim = action_space.shape[0]
            self.action_max = max(action_space.high)
        elif self.action_type == 'discrete':
            self.action_dim = action_space
        else:
            raise NotImplementedError(self.action_type)

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x {torch.Tensor} --- hight level feature
        '''
        raise NotImplementedError()


class GaussianActor(Actor):
    ''' Gaussian Actor Module '''

    def __init__(self, config: dict, in_size: int, action_space: tuple) -> None:
        '''
        Args:
            config {dict} -- config dictionary
            in_size {int} -- input feature size
            action_space {tuple} -- action space
        '''
        super(GaussianActor, self).__init__(action_space, config)

        if self.action_type == 'continuous':
            self.mu = nn.Sequential(
                nn.Linear(in_size, in_size),
                nn.ReLU(),
                nn.Linear(in_size, self.action_dim),
                nn.Tanh()
            )
            nn.init.orthogonal_(self.mu[0].weight, np.sqrt(2))
            nn.init.orthogonal_(self.mu[2].weight, np.sqrt(0.01))
            self.sigma = nn.Sequential(
                nn.Linear(in_size, self.action_dim),
                nn.Softmax(dim=-1)
            )
            nn.init.orthogonal_(self.sigma[0].weight, np.sqrt(0.01))
        elif self.action_type == 'discrete':
            self.mu = nn.Sequential(
                nn.Linear(in_size, in_size),
                nn.ReLU(),
                nn.Linear(in_size, self.action_dim),
            )
            nn.init.orthogonal_(self.mu[0].weight, np.sqrt(2))
            nn.init.orthogonal_(self.mu[2].weight, np.sqrt(0.01))
        else:
            raise NotImplementedError(self.action_type)

    def forward(self, feature: torch.Tensor):
        '''
        Args:
            feature {torch.Tensor} --- hight level feature
        '''
        if self.action_type == 'continuous':
            action_mean = self.action_max * self.mu(feature)
            action_std = self.sigma(feature)
            dist = Normal(action_mean, action_std)
        elif self.action_type == 'discrete':
            action_probs = self.mu(feature)
            dist = Categorical(probs=None, logits=action_probs)
        else:
            raise NotImplementedError(self.action_type)
        return dist


class MultiGaussianActor(Actor):
    ''' Gaussian Multiple Actor Module '''

    def __init__(self, config: dict, in_size: int, action_space: tuple, task_num: int) -> None:
        '''
        Args:
            config {dict} -- config dictionary
            in_size {int} -- input feature size
            action_space {tuple} -- action space
            task_num {int} -- num of task
        '''
        super(MultiGaussianActor, self).__init__(action_space, config)

        if self.action_type == 'continuous':
            self.m_mu = nn.ModuleDict(
                [[str(i), nn.Sequential(
                    nn.Linear(in_size, in_size),
                    nn.ReLU(),
                    nn.Linear(in_size, self.action_dim),
                    nn.Tanh()
                )]
                    for i in range(task_num)]
            )
            for _, m in self.m_mu.items():
                nn.init.orthogonal_(m[0].weight, np.sqrt(2))
                nn.init.orthogonal_(m[2].weight, np.sqrt(0.01))
            self.m_sigma = nn.ModuleDict(
                [[str(i), nn.Sequential(
                    nn.Linear(in_size, self.action_dim),
                    nn.Softmax(dim=-1)
                )] for i in range(task_num)]
            )
            for _, m in self.m_sigma.items():
                nn.init.orthogonal_(m[0].weight, np.sqrt(0.01))
        elif self.action_type == 'discrete':
            self.m_mu = nn.ModuleDict(
                [[str(i), nn.Sequential(
                    nn.Linear(in_size, in_size),
                    nn.ReLU(),
                    nn.Linear(in_size, self.action_dim),
                )] for i in range(task_num)]
            )
            for _, m in self.m_mu.items():
                nn.init.orthogonal_(m[0].weight, np.sqrt(2))
                nn.init.orthogonal_(m[2].weight, np.sqrt(0.01))
        else:
            raise NotImplementedError(self.action_type)

    def forward(self, feature: torch.Tensor, actor_index: int):
        '''
        Args:
            feature {torch.Tensor} --- hight level feature
            actor_index {int} -- module {actor[actor_index]} will be select to forward
        '''
        if self.action_type == 'continuous':
            action_mean = self.action_max * self.m_mu[str(actor_index)](feature)
            action_std = self.m_sigma[str(actor_index)](feature)
            dist = Normal(action_mean, action_std)
        elif self.action_type == 'discrete':
            action_probs = self.m_mu[str(actor_index)](feature)
            dist = Categorical(probs=None, logits=action_probs)
        else:
            raise NotImplementedError(self.action_type)
        return dist
