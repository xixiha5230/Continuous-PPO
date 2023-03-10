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
        self.hidden_layer_size = self.conf_train['hidden_layer_size']

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

    def __init__(self, config, action_space) -> None:
        '''
        Args:
            config {dict} -- config dictionary
            action_space {tuple} -- action space
        '''
        super(GaussianActor, self).__init__(config, action_space)

        if self.action_type == 'continuous':
            ac_h = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
            nn.init.orthogonal_(ac_h.weight, np.sqrt(2))
            ac = nn.Linear(self.hidden_layer_size, self.action_dim)
            nn.init.orthogonal_(ac.weight, np.sqrt(0.01))
            self.mu = nn.Sequential()
            self.mu.append(ac_h).append(nn.ReLU())
            self.mu.append(ac).append(nn.Tanh())
            std = nn.Linear(self.hidden_layer_size, self.action_dim)
            nn.init.orthogonal_(std.weight, np.sqrt(0.01))
            self.sigma = nn.Sequential()
            self.sigma.append(std).append(nn.Softmax(dim=-1))
        elif self.action_type == 'discrete':
            ac_h = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
            nn.init.orthogonal_(ac_h.weight, np.sqrt(2))
            ac = nn.Linear(self.hidden_layer_size, self.action_dim)
            nn.init.orthogonal_(ac.weight, np.sqrt(0.01))
            self.mu = nn.Sequential()
            self.mu.append(ac_h)
            self.mu.append(nn.ReLU())
            self.mu.append(ac)
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
