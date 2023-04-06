import numpy as np
import torch
import torch.nn as nn


class Critic(nn.Module):
    ''' Critic Module '''

    def __init__(self, in_size: int, out_size: int = 1, config: dict = None) -> None:
        '''
        Args:
            in_size {int} -- input feature size
            out_size {int} -- num of value
            config {dict} -- config dictionary
        '''
        super(Critic, self).__init__()
        self.config = config
        self.critic = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.ReLU(),
            nn.Linear(in_size, out_size)
        )
        nn.init.orthogonal_(self.critic[0].weight, np.sqrt(2))
        nn.init.orthogonal_(self.critic[2].weight, 1)

    def forward(self, feature: torch.Tensor):
        '''
        Args:
            x {torch.Tensor} --- hight level feature
        '''
        value = self.critic(feature)
        value = value.squeeze(-1)
        return value


class MultiCritic(nn.Module):
    ''' Multiple Critic Module '''

    def __init__(self, in_size: int, out_size: int = 1, config: dict = None, task_num: int = 1) -> None:
        '''
        Args:
            in_size {int} -- input feature size
            out_size {int} -- num of value
            config {dict} -- config dictionary
            task_num {int} -- num of task
        '''
        super(MultiCritic, self).__init__()
        self.config = config
        self.m_critic = nn.ModuleDict(
            [[str(i), nn.Sequential(
                nn.Linear(in_size, in_size),
                nn.ReLU(),
                nn.Linear(in_size, out_size)
            )] for i in range(task_num)]
        )
        for _, m in self.m_critic.items():
            nn.init.orthogonal_(m[0].weight, np.sqrt(2))
            nn.init.orthogonal_(m[2].weight, 1)

    def forward(self, feature: torch.Tensor, critic_index: int):
        '''
        Args:
            feature {torch.Tensor} --- hight level feature
            critic_index {int} -- module {critic[critic_index]} will be select to forward
        '''
        value = self.m_critic[str(critic_index)](feature)
        value = value.squeeze(-1)
        return value
