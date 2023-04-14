import numpy as np
import torch
import torch.nn as nn


class Critic(nn.Module):
    ''' Critic Module '''

    def __init__(self, config: dict, input_size: int, output_size: int = 1) -> None:
        '''
        Args:
            config {dict} -- config dictionary
            input_size {int} -- input feature size
            output_size {int} -- num of value
        '''
        super(Critic, self).__init__()
        self.config = config
        conf_train = config['train']
        self.use_rnd = conf_train['use_rnd']

        self.critic_hidden = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
        )
        nn.init.orthogonal_(self.critic_hidden[0].weight, np.sqrt(2))
        self.critic = nn.Linear(input_size, output_size)
        nn.init.orthogonal_(self.critic.weight, 1)
        if self.use_rnd:
            self.rnd_critic = nn.Linear(input_size, output_size)
            nn.init.orthogonal_(self.rnd_critic.weight, 1)

    def forward(self, feature: torch.Tensor):
        '''
        Args:
            x {torch.Tensor} --- hight level feature
        '''
        x = self.critic_hidden(feature)
        value = self.critic(x)
        value = value.squeeze(-1)
        if self.use_rnd:
            rnd_value = self.rnd_critic(x)
            rnd_value = rnd_value.squeeze(-1)
        else:
            rnd_value = None
        return value, rnd_value


class MultiCritic(nn.Module):
    ''' Multiple Critic Module '''

    def __init__(self, config: dict, input_size: int, output_size: int = 1,  task_num: int = 1) -> None:
        '''
        Args:
            config {dict} -- config dictionary
            input_size {int} -- input feature size
            output_size {int} -- num of value
            task_num {int} -- num of task
        '''
        super(MultiCritic, self).__init__()
        self.config = config
        self.m_critic = nn.ModuleDict(
            [[str(i), nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ReLU(),
                nn.Linear(input_size, output_size)
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
