import numpy as np
import torch
import torch.nn as nn


class Critic(nn.Module):
    ''' Critic Module '''

    def __init__(self, out_size: int = 1, config: dict = None) -> None:
        '''
        Args:
            out_size {int} -- num of value
            config {dict} -- config dictionary
        '''
        super(Critic, self).__init__()
        self.config = config
        self.conf_train = config['train']
        self.action_type = self.conf_train['action_type']
        self.hidden_layer_size = self.conf_train['hidden_layer_size']

        vlaue_h = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        nn.init.orthogonal_(vlaue_h.weight, np.sqrt(2))
        value = nn.Linear(self.hidden_layer_size, out_size)
        nn.init.orthogonal_(value.weight, 1)
        self.critic = nn.Sequential()
        self.critic.append(vlaue_h).append(nn.ReLU())
        self.critic.append(value)

    def forward(self, feature: torch.Tensor):
        '''
        Args:
            x {torch.Tensor} --- hight level feature
        '''
        value = self.critic(feature)
        value = value.squeeze(-1)
        return value
